"""Eval module: evaluation runner, LLM judge, and metric aggregation."""

from himga.eval.metrics import batch_bert_f1, batch_sbert_similarity, compute_metrics
from himga.eval.runner import EvalResult, run_eval

__all__ = ["EvalResult", "run_eval", "compute_metrics", "batch_bert_f1", "batch_sbert_similarity"]
