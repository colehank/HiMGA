"""Eval module: evaluation runner, LLM judge, and metric aggregation."""

from himga.eval.metrics import compute_metrics
from himga.eval.runner import EvalResult, run_eval

__all__ = ["EvalResult", "run_eval", "compute_metrics"]
