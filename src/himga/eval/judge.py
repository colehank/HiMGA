"""LLM-as-a-Judge scoring: continuous 0-1 (LoCoMo) and binary yes/no (LongMemEval) modes."""

from __future__ import annotations

import json
from pathlib import Path

from himga.data.schema import QuestionType
from himga.eval.runner import EvalResult
from himga.llm.client import BaseLLMClient

# ---------------------------------------------------------------------------
# Unanswerable detection (for ADVERSARIAL questions)
# ---------------------------------------------------------------------------

_UNANSWERABLE_EXACT: frozenset[str] = frozenset({"", "n/a", "na", "none", "null", "unanswerable"})

_UNANSWERABLE_PATTERNS: tuple[str, ...] = (
    "not mentioned",
    "not in the conversation",
    "cannot answer",
    "can't answer",
    "insufficient",
    "unknown",
    "no information",
    "not provided",
    "information not found",
    "not found",
    "not available",
    "no data",
)


def is_unanswerable(text: str) -> bool:
    """Return True if *text* represents an unanswerable / no-information response.

    Used for LoCoMo ``ADVERSARIAL`` questions: the ground-truth answer is not
    in the conversation, so a correct system should decline to answer.

    Parameters
    ----------
    text : str
        The model's predicted answer.

    Returns
    -------
    bool
        ``True`` when *text* signals that no answer was found.
    """
    if not text:
        return True
    text_lower = text.strip().lower()
    if text_lower in _UNANSWERABLE_EXACT:
        return True
    return any(p in text_lower for p in _UNANSWERABLE_PATTERNS)


# ---------------------------------------------------------------------------
# Judge prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_CONTINUOUS = (
    "You are an expert grader that scores answers on a continuous scale from 0.0 to 1.0."
)

_PROMPT_CONTINUOUS = """\
Score the answer on a scale from 0.0 to 1.0 based on semantic correctness.

Scoring Scale:
- 1.0: Perfect match - contains all key information from gold answer, semantically equivalent
- 0.8: Mostly correct - captures main point but may have minor differences in wording or detail
- 0.6: Partially correct - has some correct information but incomplete or missing key details
- 0.4: Somewhat related - touches on the topic but misses significant information
- 0.2: Barely related - answer is mostly incorrect but has some connection to the topic
- 0.0: Completely wrong - answer is unrelated or contradicts gold answer

The point of the question is to ask about something one user should know about the other user \
based on their prior conversations.

For time-related questions:
- Be generous with date formats (e.g., "May 7th" vs "7 May" should both score highly)
- Accept relative time references if they refer to the same period
- Penalize if the time period is significantly different

For factual questions:
- Focus on semantic equivalence, not exact wording
- Partial credit for partial answers
- Consider whether key entities and relationships are preserved

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON with:
- "score": A float between 0.0 and 1.0
- "reasoning": One sentence explaining the score\
"""

_SYSTEM_BINARY = (
    "You are an expert grader that determines if answers to questions match a gold standard "
    "answer. Answer with only 'yes' or 'no'."
)

_PROMPT_TEMPORAL_REASONING = """\
I will give you a question, a correct answer, and a response from a model. Please answer yes \
if the response contains the correct answer. Otherwise, answer no. If the response is \
equivalent to the correct answer or contains all the intermediate steps to get the correct \
answer, you should also answer yes. If the response only contains a subset of the information \
required by the answer, answer no. In addition, do not penalize off-by-one errors for the \
number of days. If the question asks for the number of days/weeks/months, etc., and the model \
makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's \
response is still correct.

<QUESTION>
{question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
{response}
</RESPONSE>

Answer with only 'yes' or 'no':\
"""

_PROMPT_KNOWLEDGE_UPDATE = """\
I will give you a question, a correct answer, and a response from a model. Please answer yes \
if the response contains the correct answer. Otherwise, answer no. If the response contains \
some previous information along with an updated answer, the response should be considered as \
correct as long as the updated answer is the required answer.

<QUESTION>
{question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
{response}
</RESPONSE>

Answer with only 'yes' or 'no':\
"""

_PROMPT_PREFERENCE = """\
I will give you a question, a rubric for desired personalized response, and a response from a \
model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. \
The model does not need to reflect all the points in the rubric. The response is correct as \
long as it recalls and utilizes the user's personal information correctly.

<QUESTION>
{question}
</QUESTION>
<RUBRIC>
{gold_answer}
</RUBRIC>
<RESPONSE>
{response}
</RESPONSE>

Answer with only 'yes' or 'no':\
"""

_PROMPT_DEFAULT_BINARY = """\
I will give you a question, a correct answer, and a response from a model. Please answer yes \
if the response contains the correct answer or the key/core information from the correct \
answer. Otherwise, answer no.

Important evaluation guidelines:
- If the response contains the main factual content, answer yes
- Minor differences in articles (a/the), capitalization, or additional context should not \
affect correctness
- If the response captures the essential answer to the question, answer yes
- Only answer no if the response is factually wrong or completely missing the key information

<QUESTION>
{question}
</QUESTION>
<CORRECT ANSWER>
{gold_answer}
</CORRECT ANSWER>
<RESPONSE>
{response}
</RESPONSE>

Answer with only 'yes' or 'no':\
"""

# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------

_QUESTION_TYPE_MODE: dict[QuestionType, str] = {
    # LoCoMo — continuous 0~1
    QuestionType.SINGLE_HOP: "continuous",
    QuestionType.MULTI_HOP: "continuous",
    QuestionType.TEMPORAL: "continuous",
    QuestionType.OPEN_DOMAIN: "continuous",
    # LoCoMo — rule-based, no LLM
    QuestionType.ADVERSARIAL: "adversarial",
    # LongMemEval — type-specific binary
    QuestionType.TEMPORAL_REASONING: "temporal_reasoning",
    QuestionType.KNOWLEDGE_UPDATE: "knowledge_update",
    QuestionType.SINGLE_SESSION_PREFERENCE: "preference",
    QuestionType.SINGLE_SESSION_USER: "default_binary",
    QuestionType.SINGLE_SESSION_ASSISTANT: "default_binary",
    QuestionType.MULTI_SESSION: "default_binary",
}


def _auto_mode(question_type: QuestionType) -> str:
    """Return the judge mode for *question_type*."""
    return _QUESTION_TYPE_MODE.get(question_type, "continuous")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    ground_truth: str,
    prediction: str,
    *,
    llm: BaseLLMClient | None = None,
    mode: str = "continuous",
) -> float:
    """Judge whether *prediction* is correct for *question* given *ground_truth*.

    Parameters
    ----------
    question : str
        The evaluation question.
    ground_truth : str
        Gold-standard answer.
    prediction : str
        Model-generated answer to evaluate.
    llm : BaseLLMClient or None
        Judge LLM client.  Not required when ``mode="adversarial"``.
    mode : str
        One of:

        ``"continuous"``
            MAGMA-style 0.0–1.0 rubric.  The LLM must return a JSON object
            with a ``"score"`` field.  Used for all LoCoMo question types.

        ``"adversarial"``
            Rule-based: checks whether the prediction is an "unanswerable"
            response.  No LLM call is made.

        ``"temporal_reasoning"`` | ``"knowledge_update"`` | ``"preference"`` |
        ``"default_binary"``
            LongMemEval question-type-specific binary prompts.  The LLM
            replies ``"yes"`` (1.0) or ``"no"`` (0.0).

    Returns
    -------
    float
        Score in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If *llm* is ``None`` and *mode* is not ``"adversarial"``.
    """
    if mode == "adversarial":
        return 1.0 if is_unanswerable(prediction) else 0.0

    if llm is None:
        raise ValueError("llm is required for all judge modes except 'adversarial'")

    if mode == "continuous":
        prompt = _PROMPT_CONTINUOUS.format(
            question=question,
            gold_answer=ground_truth,
            generated_answer=prediction,
        )
        raw = llm.chat(
            [
                {"role": "system", "content": _SYSTEM_CONTINUOUS},
                {"role": "user", "content": prompt},
            ]
        )
        try:
            result = json.loads(raw)
            score = float(result.get("score", 0.0))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            return 0.0

    # Binary LongMemEval modes
    if mode == "temporal_reasoning":
        prompt = _PROMPT_TEMPORAL_REASONING.format(
            question=question, gold_answer=ground_truth, response=prediction
        )
    elif mode == "knowledge_update":
        prompt = _PROMPT_KNOWLEDGE_UPDATE.format(
            question=question, gold_answer=ground_truth, response=prediction
        )
    elif mode == "preference":
        prompt = _PROMPT_PREFERENCE.format(
            question=question, gold_answer=ground_truth, response=prediction
        )
    else:  # "default_binary"
        prompt = _PROMPT_DEFAULT_BINARY.format(
            question=question, gold_answer=ground_truth, response=prediction
        )

    raw = llm.chat(
        [
            {"role": "system", "content": _SYSTEM_BINARY},
            {"role": "user", "content": prompt},
        ]
    )
    text = raw.strip().lower()
    return 1.0 if (text == "yes" or text.startswith("yes")) else 0.0


def batch_judge(
    results: list[EvalResult],
    *,
    llm: BaseLLMClient,
    mode: str = "auto",
    cache_path: Path | None = None,
) -> list[float]:
    """Judge all *results*, optionally persisting scores to *cache_path*.

    On the first call the scores are computed via *llm* and written to
    *cache_path* (if provided).  On subsequent calls with the same path the
    scores are loaded from disk, avoiding duplicate API charges.

    Parameters
    ----------
    results : list[EvalResult]
        Results to score.
    llm : BaseLLMClient
        Judge model client.
    mode : str
        ``"auto"`` selects the judge mode per result based on
        :attr:`~himga.eval.runner.EvalResult.question_type` (recommended).
        Pass an explicit mode string to override for every result.
    cache_path : Path or None
        Optional JSON file path for caching scores keyed by ``question_id``.

    Returns
    -------
    list[float]
        Per-result judge scores in the same order as *results*.
    """
    if not results:
        return []

    # Load from cache if all question_ids are already cached
    if cache_path is not None and cache_path.exists():
        cache: dict[str, float] = json.loads(cache_path.read_text())
        if all(r.question_id in cache for r in results):
            return [cache[r.question_id] for r in results]

    scores: list[float] = []
    for r in results:
        effective_mode = _auto_mode(r.question_type) if mode == "auto" else mode
        score = judge_answer(
            r.question,
            r.ground_truth,
            r.prediction,
            llm=llm,
            mode=effective_mode,
        )
        scores.append(score)

    if cache_path is not None:
        existing: dict[str, float] = {}
        if cache_path.exists():
            existing = json.loads(cache_path.read_text())
        existing.update({r.question_id: s for r, s in zip(results, scores)})
        cache_path.write_text(json.dumps(existing, indent=2))

    return scores
