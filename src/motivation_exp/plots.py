"""Two-panel motivation figure from the run JSONL.

Panel A: final-answer accuracy vs. context tokens, with 95% Wilson intervals.
Panel B: decode-phase per-accepted-token latency (median) vs. context tokens, with an IQR
band. Shared x-axis on a log2 scale. CPU-only; uses a non-interactive matplotlib backend.

The per-token latency denominator convention (per ACCEPTED output token, numerator incl.
rollback + embedder time for the semantic method) is fixed upstream in the runner; this
module just aggregates the ``per_token_ms`` field.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # no display on Colab/headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["unguided", "symbolic", "semantic"]
METHOD_STYLE = {
    "unguided": {"color": "#7f7f7f", "marker": "o", "label": "unguided (reference)"},
    "symbolic": {"color": "#1f77b4", "marker": "s", "label": "symbolic (grammar)"},
    "semantic": {"color": "#d62728", "marker": "^", "label": "semantic (step-verify)"},
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval (default z) for k successes out of n trials.

    Returns (lo, hi) as proportions in [0, 1]. For n == 0 returns (0.0, 1.0).
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_results(jsonl_path: str) -> pd.DataFrame:
    """Load the run JSONL into a DataFrame, tolerating a trailing partial line.

    Flattens the ``overhead`` sub-object into ``overhead_*`` columns. Rows that fail to
    parse (e.g. a truncated final line from a mid-write disconnect) are skipped.
    """
    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # trailing partial line from a disconnect; ignore
            overhead = obj.pop("overhead", None)
            if isinstance(overhead, dict):
                for k, v in overhead.items():
                    obj[f"overhead_{k}"] = v
            rows.append(obj)
    return pd.DataFrame(rows)


def _accuracy_table(df: pd.DataFrame) -> dict[str, dict[int, tuple[float, float, float]]]:
    """{method: {bucket: (acc, lo, hi)}} using Wilson intervals."""
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for method, mdf in df.groupby("method"):
        per_bucket: dict[int, tuple[float, float, float]] = {}
        for bucket, bdf in mdf.groupby("bucket"):
            n = len(bdf)
            k = int(bdf["correct"].sum())
            acc = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            per_bucket[int(bucket)] = (acc, lo, hi)
        out[str(method)] = per_bucket
    return out


def _latency_table(df: pd.DataFrame) -> dict[str, dict[int, tuple[float, float, float]]]:
    """{method: {bucket: (median, q25, q75)}} of per_token_ms."""
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for method, mdf in df.groupby("method"):
        per_bucket: dict[int, tuple[float, float, float]] = {}
        for bucket, bdf in mdf.groupby("bucket"):
            vals = bdf["per_token_ms"].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            per_bucket[int(bucket)] = (
                float(np.median(vals)),
                float(np.percentile(vals, 25)),
                float(np.percentile(vals, 75)),
            )
        out[str(method)] = per_bucket
    return out


def make_two_panel_figure(
    df: pd.DataFrame,
    out_path: str,
    *,
    model_name: str = "",
    also_pdf: bool = True,
    dpi: int = 300,
) -> str:
    """Render the two-panel figure and save PNG (and PDF). Returns the PNG path."""
    acc = _accuracy_table(df)
    lat = _latency_table(df)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    methods = [m for m in METHOD_ORDER if m in acc or m in lat]

    # Panel A: accuracy + Wilson CIs -----------------------------------------------------
    for method in methods:
        table = acc.get(method, {})
        if not table:
            continue
        buckets = sorted(table)
        ys = [table[b][0] * 100 for b in buckets]
        los = [(table[b][0] - table[b][1]) * 100 for b in buckets]
        his = [(table[b][2] - table[b][0]) * 100 for b in buckets]
        style = METHOD_STYLE[method]
        axA.errorbar(
            buckets, ys, yerr=[los, his], marker=style["marker"], color=style["color"],
            label=style["label"], capsize=3, linewidth=1.8, markersize=6,
        )
    axA.set_xscale("log", base=2)
    axA.set_xlabel("context length (tokens)")
    axA.set_ylabel("accuracy (%)")
    axA.set_title("Panel A — Performance")
    axA.set_ylim(0, 100)
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8, loc="best")

    # Panel B: median per-token latency + IQR band ---------------------------------------
    for method in methods:
        table = lat.get(method, {})
        if not table:
            continue
        buckets = sorted(table)
        med = [table[b][0] for b in buckets]
        q25 = [table[b][1] for b in buckets]
        q75 = [table[b][2] for b in buckets]
        style = METHOD_STYLE[method]
        axB.plot(buckets, med, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=1.8, markersize=6)
        axB.fill_between(buckets, q25, q75, color=style["color"], alpha=0.15)
    axB.set_xscale("log", base=2)
    axB.set_xlabel("context length (tokens)")
    axB.set_ylabel("per-output-token decode latency (ms)")
    axB.set_title("Panel B — Latency")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=8, loc="best")

    caption = (
        f"Check granularity: step-boundary. Latency: decode-phase per ACCEPTED output "
        f"token (prefill excluded; semantic numerator includes rollback + embedder). "
        f"Hardware: single T4 (sm75), fp16; SDPA math backend (GQA head mismatch precludes the "
        f"mem-efficient/flash kernels on this GPU), chunked prefill."
    )
    if model_name:
        caption = f"Model: {model_name}. " + caption
    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=7, wrap=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
