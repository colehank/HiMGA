"""Metric computation: token F1, exact match, ROUGE, BLEU, METEOR, BERTScore, SBERT.

Optional heavy dependencies
----------------------------
``bert_f1`` and ``sbert_similarity`` require the ``eval`` extras::

    pip install "himga[eval]"

For GPU acceleration install CUDA-compatible PyTorch **before** the extras::

    # Example: CUDA 11.8
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install "himga[eval]"

When running on a machine with CUDA-enabled PyTorch the BERTScore and
Sentence-BERT functions automatically use the GPU; no code changes are needed.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _meteor_score
from rouge_score import rouge_scorer as _rouge_scorer_mod

from himga.eval.runner import EvalResult

# ---------------------------------------------------------------------------
# NLTK corpus bootstrap — called lazily before first BLEU / METEOR use
# ---------------------------------------------------------------------------

_nltk_ready = False
_nltk_lock = threading.Lock()


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present.

    Downloads ``punkt_tab`` (tokeniser) and ``wordnet`` (METEOR).
    Safe to call multiple times; actual network I/O happens at most once per
    process.  Raises ``LookupError`` with installation instructions if a
    corpus cannot be found or downloaded.
    """
    global _nltk_ready
    if _nltk_ready:
        return
    with _nltk_lock:
        if _nltk_ready:
            return
        _required = [
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("wordnet", "corpora/wordnet"),
        ]
        for corpus, finder_path in _required:
            try:
                nltk.data.find(finder_path)
            except LookupError:
                nltk.download(corpus, quiet=True)
                try:
                    nltk.data.find(finder_path)
                except LookupError as exc:
                    raise LookupError(
                        f"NLTK corpus '{corpus}' is required for BLEU/METEOR metrics "
                        f"but could not be found or downloaded automatically.\n"
                        f"Run manually:  python -m nltk.downloader {corpus}"
                    ) from exc
        _nltk_ready = True


# ---------------------------------------------------------------------------
# Optional heavy-dependency imports (bert-score, sentence-transformers)
# ---------------------------------------------------------------------------

try:
    from bert_score import score as _bert_score_fn

    _BERT_SCORE_AVAILABLE = True
except ImportError:
    _BERT_SCORE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim as _cos_sim

    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

_sbert_model: object = None
_sbert_model_lock = threading.Lock()

_EVAL_INSTALL_HINT = (
    'Install the eval extras:  pip install "himga[eval]"\n'
    "For GPU: install CUDA-compatible PyTorch first — "
    "see https://pytorch.org/get-started/locally/"
)


def _require_bert_score() -> None:
    if not _BERT_SCORE_AVAILABLE:
        raise ImportError("bert-score is required for bert_f1().\n" + _EVAL_INSTALL_HINT)


def _require_sbert() -> None:
    if not _SBERT_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for sbert_similarity().\n" + _EVAL_INSTALL_HINT
        )


def _get_sbert_model() -> "_SentenceTransformer":
    global _sbert_model
    if _sbert_model is None:
        with _sbert_model_lock:
            if _sbert_model is None:
                _sbert_model = _SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if *prediction* equals *ground_truth* (case-insensitive), else 0.0.

    Parameters
    ----------
    prediction : str
        Model output string.
    ground_truth : str
        Gold-standard answer string.

    Returns
    -------
    float
        ``1.0`` on exact match, ``0.0`` otherwise.
    """
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between *prediction* and *ground_truth*.

    Tokens are whitespace-split and lowercased; punctuation replaced with
    spaces.  Uses bag-of-words intersection (duplicates counted once).

    Parameters
    ----------
    prediction : str
        Model output string.
    ground_truth : str
        Gold-standard answer string.

    Returns
    -------
    float
        F1 score in ``[0.0, 1.0]``.  Returns ``0.0`` when either string is empty.
    """

    def _tokenize(text: str) -> list[str]:
        for ch in (".", ",", "!", "?"):
            text = text.replace(ch, " ")
        return text.lower().split()

    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counts: dict[str, int] = defaultdict(int)
    for t in pred_tokens:
        pred_counts[t] += 1

    gt_counts: dict[str, int] = defaultdict(int)
    for t in gt_tokens:
        gt_counts[t] += 1

    overlap = sum(min(pred_counts[t], gt_counts[t]) for t in pred_counts if t in gt_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_scores(prediction: str, ground_truth: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Requires ``rouge-score`` (always installed as a core dependency).

    Parameters
    ----------
    prediction : str
        Model output.
    ground_truth : str
        Reference answer.

    Returns
    -------
    dict
        Keys: ``"rouge1"``, ``"rouge2"``, ``"rougeL"``.  Values in ``[0.0, 1.0]``.
    """
    if not prediction.strip() or not ground_truth.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = _rouge_scorer_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def bleu_scores(prediction: str, ground_truth: str) -> dict[str, float]:
    """Compute BLEU-1, BLEU-2, and BLEU-4 with NLTK SmoothingFunction.method1.

    Requires ``nltk`` (core dependency) with ``punkt_tab`` corpus.  The corpus
    is downloaded automatically on first call.

    Parameters
    ----------
    prediction : str
        Model output.
    ground_truth : str
        Reference answer.

    Returns
    -------
    dict
        Keys: ``"bleu1"``, ``"bleu2"``, ``"bleu4"``.  Values in ``[0.0, 1.0]``.
    """
    if not prediction.strip() or not ground_truth.strip():
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}
    _ensure_nltk_data()
    smooth = SmoothingFunction().method1
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(ground_truth.lower())]
    result: dict[str, float] = {}
    for name, weights in [
        ("bleu1", (1, 0, 0, 0)),
        ("bleu2", (0.5, 0.5, 0, 0)),
        ("bleu4", (0.25, 0.25, 0.25, 0.25)),
    ]:
        result[name] = float(
            sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        )
    return result


def meteor(prediction: str, ground_truth: str) -> float:
    """Compute METEOR score.

    Requires ``nltk`` (core dependency) with ``wordnet`` corpus.  The corpus
    is downloaded automatically on first call.

    Parameters
    ----------
    prediction : str
        Model output.
    ground_truth : str
        Reference answer.

    Returns
    -------
    float
        METEOR score in ``[0.0, 1.0]``.
    """
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    _ensure_nltk_data()
    return float(_meteor_score([ground_truth.split()], prediction.split()))


def bert_f1(prediction: str, ground_truth: str) -> float:
    """Compute BERTScore F1 between *prediction* and *ground_truth*.

    Requires the ``eval`` extras (``bert-score`` package).  Uses GPU
    automatically when CUDA-compatible PyTorch is installed.

    .. code-block:: bash

        pip install "himga[eval]"
        # GPU: install CUDA torch first — see https://pytorch.org/get-started/locally/

    Parameters
    ----------
    prediction : str
        Model output.
    ground_truth : str
        Reference answer.

    Returns
    -------
    float
        BERTScore F1 in ``[0.0, 1.0]``.

    Raises
    ------
    ImportError
        If ``bert-score`` is not installed.
    """
    _require_bert_score()
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    _, _, F1 = _bert_score_fn([prediction], [ground_truth], lang="en", verbose=False)
    return float(F1.item())


def sbert_similarity(prediction: str, ground_truth: str) -> float:
    """Compute cosine similarity via Sentence-BERT (``all-MiniLM-L6-v2``).

    Requires the ``eval`` extras (``sentence-transformers`` package).  The
    model is loaded lazily on first call and cached in memory.  Uses GPU
    automatically when CUDA-compatible PyTorch is installed.

    .. code-block:: bash

        pip install "himga[eval]"
        # GPU: install CUDA torch first — see https://pytorch.org/get-started/locally/

    Parameters
    ----------
    prediction : str
        Model output.
    ground_truth : str
        Reference answer.

    Returns
    -------
    float
        Cosine similarity in ``[-1.0, 1.0]``, typically ``[0.0, 1.0]``.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    """
    _require_sbert()
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    model = _get_sbert_model()
    emb_pred = model.encode([prediction], convert_to_tensor=True)
    emb_ref = model.encode([ground_truth], convert_to_tensor=True)
    return float(_cos_sim(emb_pred, emb_ref).item())


# All metric names in the order they appear in output
ALL_METRICS: tuple[str, ...] = (
    "judge_score",
    "exact_match",
    "f1",
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu1",
    "bleu2",
    "bleu4",
    "meteor",
    "bert_f1",
    "sbert_similarity",
)

# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def _compute_selected_metrics(
    prediction: str,
    ground_truth: str,
    judge_score: float,
    include: frozenset[str],
) -> dict[str, float]:
    """Compute only the metrics listed in *include* for a single prediction."""
    result: dict[str, float] = {}
    if "judge_score" in include:
        result["judge_score"] = judge_score
    if "exact_match" in include:
        result["exact_match"] = exact_match(prediction, ground_truth)
    if "f1" in include:
        result["f1"] = token_f1(prediction, ground_truth)
    if include & {"rouge1", "rouge2", "rougeL"}:
        r = rouge_scores(prediction, ground_truth)
        for k in ("rouge1", "rouge2", "rougeL"):
            if k in include:
                result[k] = r[k]
    if include & {"bleu1", "bleu2", "bleu4"}:
        b = bleu_scores(prediction, ground_truth)
        for k in ("bleu1", "bleu2", "bleu4"):
            if k in include:
                result[k] = b[k]
    if "meteor" in include:
        result["meteor"] = meteor(prediction, ground_truth)
    if "bert_f1" in include:
        result["bert_f1"] = bert_f1(prediction, ground_truth)
    if "sbert_similarity" in include:
        result["sbert_similarity"] = sbert_similarity(prediction, ground_truth)
    return result


def compute_metrics(
    results: list[EvalResult],
    judge_scores: list[float],
    *,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Aggregate metrics overall and per :class:`~himga.data.schema.QuestionType`.

    By default all twelve metrics are computed.  Pass *metrics* to request a
    subset (useful to avoid loading heavy ML models during fast test runs).

    Available metrics (all in ``[0.0, 1.0]``):

    - **judge_score** – LLM judge score (see :mod:`himga.eval.judge`)
    - **exact_match** – case-insensitive exact string match
    - **f1** – token-level F1 (SQuAD-style with punctuation normalisation)
    - **rouge1**, **rouge2**, **rougeL** – ROUGE F1 (``rouge-score``, core dep)
    - **bleu1**, **bleu2**, **bleu4** – BLEU n-gram with smoothing (``nltk``, core dep)
    - **meteor** – METEOR (``nltk`` + wordnet corpus, core dep)
    - **bert_f1** – BERTScore F1 — requires ``pip install "himga[eval]"``
    - **sbert_similarity** – Sentence-BERT cosine similarity — requires ``pip install "himga[eval]"``

    Parameters
    ----------
    results : list[EvalResult]
        Evaluation predictions.
    judge_scores : list[float]
        Parallel judge scores produced by :func:`~himga.eval.judge.batch_judge`.
    metrics : tuple or list or None
        Subset of metric names to compute.  ``None`` (default) computes all.
        Example: ``metrics=("judge_score", "f1", "rouge1")``

    Returns
    -------
    dict
        Structure::

            {
              "overall": {<metric_name>: float, ...},
              "by_type": {
                  "<question_type_value>": {<metric_name>: float, ..., "count": int},
                  ...
              }
            }

    Raises
    ------
    ImportError
        If ``bert_f1`` or ``sbert_similarity`` are requested but the
        ``himga[eval]`` extras are not installed.
    """
    include: frozenset[str] = frozenset(ALL_METRICS) if metrics is None else frozenset(metrics)
    _zero_overall: dict[str, float] = {k: 0.0 for k in ALL_METRICS if k in include}
    if not results:
        return {"overall": _zero_overall, "by_type": {}}

    per_result = [
        _compute_selected_metrics(r.prediction, r.ground_truth, js, include)
        for r, js in zip(results, judge_scores)
    ]

    active_keys = list(per_result[0].keys())
    overall = {k: sum(m[k] for m in per_result) / len(per_result) for k in active_keys}

    by_type_raw: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r, m in zip(results, per_result):
        by_type_raw[r.question_type.value].append(m)

    by_type: dict[str, dict] = {}
    for type_key, ms in by_type_raw.items():
        entry: dict = {k: sum(m[k] for m in ms) / len(ms) for k in active_keys}
        entry["count"] = len(ms)
        by_type[type_key] = entry

    return {"overall": overall, "by_type": by_type}
