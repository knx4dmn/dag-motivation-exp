"""CPU-only tests for plots: Wilson CI numerics, JSONL loading (incl. partial line), render."""
from __future__ import annotations

import os

import pytest

from motivation_exp import plots


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_results.jsonl")


# --------------------------------------------------------------------------------------
# wilson_ci
# --------------------------------------------------------------------------------------
def test_wilson_ci_known_value():
    # 50/100 -> center ~0.5, symmetric, roughly (0.404, 0.596)
    lo, hi = plots.wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=0.005)
    assert hi == pytest.approx(0.596, abs=0.005)


def test_wilson_ci_boundaries():
    assert plots.wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = plots.wilson_ci(10, 10)  # all successes
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < lo < 1.0


def test_wilson_ci_within_unit_interval():
    for k, n in [(1, 3), (7, 20), (99, 100)]:
        lo, hi = plots.wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------------------
# load_results
# --------------------------------------------------------------------------------------
def test_load_results_flattens_overhead():
    df = plots.load_results(FIXTURE)
    assert len(df) == 8
    assert "overhead_check_total_s" in df.columns
    assert "overhead_candidate_set_size" in df.columns
    assert set(df["method"].unique()) == {"unguided", "symbolic", "semantic"}


def test_load_results_tolerates_trailing_partial_line(tmp_path):
    p = tmp_path / "partial.jsonl"
    good = open(FIXTURE).read().strip().splitlines()[0]
    p.write_text(good + "\n" + '{"run_id": "bad", "method": "sema')  # truncated mid-write
    df = plots.load_results(str(p))
    assert len(df) == 1  # partial line skipped


# --------------------------------------------------------------------------------------
# aggregation tables
# --------------------------------------------------------------------------------------
def test_accuracy_and_latency_tables():
    df = plots.load_results(FIXTURE)
    acc = plots._accuracy_table(df)
    # unguided @512: 1 correct / 2 -> 0.5
    assert acc["unguided"][512][0] == pytest.approx(0.5)
    lat = plots._latency_table(df)
    # semantic latency grows 512 -> 4096
    assert lat["semantic"][4096][0] > lat["semantic"][512][0]


# --------------------------------------------------------------------------------------
# figure render
# --------------------------------------------------------------------------------------
def test_make_two_panel_figure_writes_files(tmp_path):
    df = plots.load_results(FIXTURE)
    out = tmp_path / "fig.png"
    plots.make_two_panel_figure(df, str(out), model_name="llama-3.2-3b-instruct")
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "fig.pdf").exists()
