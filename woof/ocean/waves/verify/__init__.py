"""Rust pairing and scoring, with analysis-only charts of Rust results."""
from pathlib import Path
from ..data import invoke

def run(request):
    out = invoke(request, operation="verify")
    from .charts import render
    render(out / "verification.json", out / "pictures")
    return out
