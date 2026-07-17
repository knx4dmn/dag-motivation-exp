"""Make ``src/`` importable for tests without an editable install (CPU-only dev convenience)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
