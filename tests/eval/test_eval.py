"""Tests for himga.eval: runner, judge, and metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from himga.agent import BaseAgent
from himga.data.schema import Message, QAPair, QuestionType, Sample, Session
from himga.eval import EvalResult, compute_metrics, run_eval
from himga.eval.judge import (
    batch_judge,
    is_unanswerable,
    judge_answer,
)
from himga.eval.metrics import (
    ALL_METRICS,
    bert_f1,
    bleu_scores,
    exact_match,
    meteor,
    rouge_scores,
    sbert_similarity,
    token_f1,
)
from himga.memory import NullMemory

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Returns a configurable response or a per-call sequence of responses."""

    def __init__(self, responses: str | list[str] = "mock answer"):
        if isinstance(responses, str):
            self._seq = None
            self._fixed = responses
        else:
            self._seq = iter(responses)
            self._fixed = ""
        self.call_count = 0
        self.all_messages: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        self.all_messages.append(messages)
        if self._seq is not None:
            return next(self._seq)
        return self._fixed

    def batch_chat(self, requests: list[dict]) -> list[str]:
        return [
            self.chat(req["messages"], **{k: v for k, v in req.items() if k != "messages"})
            for req in requests
        ]


def _make_sample(
    sample_id: str = "s0",
    *,
    n_sessions: int = 1,
    messages_per_session: int = 2,
    qa_pairs: list[QAPair] | None = None,
) -> Sample:
    sessions = [
        Session(
            session_id=f"{sample_id}_sess{i}",
            messages=[
                Message(role="user", content=f"{sample_id} msg {j}")
                for j in range(messages_per_session)
            ],
        )
        for i in range(n_sessions)
    ]
    if qa_pairs is None:
        qa_pairs = [
            QAPair(
                question_id=f"{sample_id}_q0",
                question="What happened?",
                answer="Something happened.",
                question_type=QuestionType.SINGLE_HOP,
            )
        ]
    return Sample(
        sample_id=sample_id,
        dataset="locomo",
        sessions=sessions,
        qa_pairs=qa_pairs,
    )


def _make_eval_results(
    types_and_scores: list[tuple[QuestionType, float]],
) -> tuple[list[EvalResult], list[float]]:
    results = [
        EvalResult(
            sample_id="s",
            question_id=f"q{i}",
            question_type=qt,
            question="Q",
            ground_truth="A",
            prediction="A" if score == 1.0 else "wrong",
        )
        for i, (qt, score) in enumerate(types_and_scores)
    ]
    scores = [score for _, score in types_and_scores]
    return results, scores


def _json_response(score: float) -> str:
    """Build a JSON judge response as returned by a continuous-mode LLM."""
    return json.dumps({"score": score, "reasoning": "test"})


# ===========================================================================
# TestRunEval
# ===========================================================================


class TestRunEval:
    def test_returns_list(self):
        dataset = [_make_sample()]
        result = run_eval(dataset, agent=BaseAgent(NullMemory(), MockLLMClient()))
        assert isinstance(result, list)

    def test_result_count_equals_total_qa_pairs(self):
        qa_pairs = [
            QAPair(
                question_id=f"q{i}",
                question=f"Q{i}",
                answer=f"A{i}",
                question_type=QuestionType.SINGLE_HOP,
            )
            for i in range(3)
        ]
        dataset = [
            _make_sample("s0", qa_pairs=qa_pairs[:2]),
            _make_sample("s1", qa_pairs=qa_pairs[2:]),
        ]
        results = run_eval(dataset, agent=BaseAgent(NullMemory(), MockLLMClient()))
        assert len(results) == 3

    def test_each_item_is_eval_result(self):
        dataset = [_make_sample()]
        results = run_eval(dataset, agent=BaseAgent(NullMemory(), MockLLMClient()))
        assert all(isinstance(r, EvalResult) for r in results)

    def test_result_fields_populated(self):
        sample = _make_sample("s42")
        results = run_eval([sample], agent=BaseAgent(NullMemory(), MockLLMClient("pred")))
        r = results[0]
        assert r.sample_id == "s42"
        assert r.question_id == "s42_q0"
        assert r.question == "What happened?"
        assert r.ground_truth == "Something happened."
        assert r.prediction == "pred"
        assert r.question_type == QuestionType.SINGLE_HOP

    def test_memory_reset_called_between_samples(self):
        reset_count: list[int] = [0]

        class CountingMemory(NullMemory):
            def reset(self) -> None:
                reset_count[0] += 1

        dataset = [_make_sample("s0"), _make_sample("s1"), _make_sample("s2")]
        run_eval(dataset, agent=BaseAgent(CountingMemory(), MockLLMClient()))
        assert reset_count[0] == 3

    def test_progress_bar_disabled_by_default(self):
        dataset = [_make_sample()]
        run_eval(dataset, agent=BaseAgent(NullMemory(), MockLLMClient()), show_progress=False)

    def test_empty_dataset_returns_empty_list(self):
        results = run_eval([], agent=BaseAgent(NullMemory(), MockLLMClient()))
        assert results == []

    def test_multi_qa_sample_all_results_captured(self):
        qa = [
            QAPair(
                question_id=f"q{i}",
                question=f"Q{i}",
                answer=f"A{i}",
                question_type=QuestionType.TEMPORAL,
            )
            for i in range(4)
        ]
        dataset = [_make_sample("s0", qa_pairs=qa)]
        results = run_eval(dataset, agent=BaseAgent(NullMemory(), MockLLMClient()))
        assert len(results) == 4
        assert all(r.sample_id == "s0" for r in results)


# ===========================================================================
# TestEvalResult
# ===========================================================================


class TestEvalResult:
    def test_is_dataclass(self):
        import dataclasses

        assert dataclasses.is_dataclass(EvalResult)

    def test_fields_exist(self):
        r = EvalResult(
            sample_id="s",
            question_id="q",
            question_type=QuestionType.MULTI_HOP,
            question="Q?",
            ground_truth="GT",
            prediction="P",
        )
        assert r.sample_id == "s"
        assert r.question_id == "q"
        assert r.question_type == QuestionType.MULTI_HOP
        assert r.question == "Q?"
        assert r.ground_truth == "GT"
        assert r.prediction == "P"


# ===========================================================================
# TestIsUnanswerable
# ===========================================================================


class TestIsUnanswerable:
    def test_empty_string_is_unanswerable(self):
        assert is_unanswerable("") is True

    def test_none_like_values(self):
        for text in ("n/a", "NA", "None", "null", "UNANSWERABLE"):
            assert is_unanswerable(text) is True, f"Expected True for {text!r}"

    def test_pattern_not_mentioned(self):
        assert is_unanswerable("This was not mentioned in the conversation.") is True

    def test_pattern_not_found(self):
        assert is_unanswerable("Information not found") is True

    def test_pattern_unknown(self):
        assert is_unanswerable("The answer is unknown.") is True

    def test_pattern_no_information(self):
        assert is_unanswerable("No information available.") is True

    def test_normal_answer_is_not_unanswerable(self):
        assert is_unanswerable("Paris") is False

    def test_long_answer_not_unanswerable(self):
        assert is_unanswerable("The meeting was scheduled for May 7th, 2023.") is False

    def test_case_insensitive_matching(self):
        assert is_unanswerable("INFORMATION NOT FOUND") is True
        assert is_unanswerable("Cannot Answer this question") is True


# ===========================================================================
# TestJudgeAnswer — continuous mode (default)
# ===========================================================================


class TestJudgeAnswerContinuous:
    def test_perfect_score_returns_1(self):
        llm = MockLLMClient(_json_response(1.0))
        score = judge_answer("Q", "GT", "Pred", llm=llm, mode="continuous")
        assert score == 1.0

    def test_zero_score_returns_0(self):
        llm = MockLLMClient(_json_response(0.0))
        score = judge_answer("Q", "GT", "Pred", llm=llm, mode="continuous")
        assert score == 0.0

    def test_partial_score_preserved(self):
        llm = MockLLMClient(_json_response(0.6))
        score = judge_answer("Q", "GT", "Pred", llm=llm, mode="continuous")
        assert abs(score - 0.6) < 1e-9

    def test_score_clamped_above_1(self):
        llm = MockLLMClient(json.dumps({"score": 1.5}))
        score = judge_answer("Q", "GT", "Pred", llm=llm, mode="continuous")
        assert score == 1.0

    def test_score_clamped_below_0(self):
        llm = MockLLMClient(json.dumps({"score": -0.3}))
        score = judge_answer("Q", "GT", "Pred", llm=llm, mode="continuous")
        assert score == 0.0

    def test_invalid_json_returns_0(self):
        llm = MockLLMClient("not json")
        score = judge_answer("Q", "GT", "Pred", llm=llm, mode="continuous")
        assert score == 0.0

    def test_returns_float(self):
        llm = MockLLMClient(_json_response(0.8))
        result = judge_answer("Q", "GT", "P", llm=llm, mode="continuous")
        assert isinstance(result, float)

    def test_system_and_user_messages_sent(self):
        llm = MockLLMClient(_json_response(1.0))
        judge_answer("My question", "true answer", "predicted", llm=llm, mode="continuous")
        messages = llm.all_messages[0]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_prompt_contains_question(self):
        llm = MockLLMClient(_json_response(1.0))
        judge_answer("unique_question_xyz", "gt", "pred", llm=llm, mode="continuous")
        full = str(llm.all_messages)
        assert "unique_question_xyz" in full

    def test_prompt_contains_ground_truth(self):
        llm = MockLLMClient(_json_response(1.0))
        judge_answer("Q", "unique_ground_truth_abc", "pred", llm=llm, mode="continuous")
        full = str(llm.all_messages)
        assert "unique_ground_truth_abc" in full

    def test_prompt_contains_prediction(self):
        llm = MockLLMClient(_json_response(1.0))
        judge_answer("Q", "gt", "unique_prediction_def", llm=llm, mode="continuous")
        full = str(llm.all_messages)
        assert "unique_prediction_def" in full


# ===========================================================================
# TestJudgeAnswerAdversarial — no LLM needed
# ===========================================================================


class TestJudgeAnswerAdversarial:
    def test_unanswerable_prediction_returns_1(self):
        score = judge_answer("Q", "GT", "not found", mode="adversarial")
        assert score == 1.0

    def test_concrete_answer_returns_0(self):
        score = judge_answer("Q", "GT", "Paris", mode="adversarial")
        assert score == 0.0

    def test_empty_prediction_returns_1(self):
        score = judge_answer("Q", "GT", "", mode="adversarial")
        assert score == 1.0

    def test_llm_not_called_in_adversarial_mode(self):
        llm = MockLLMClient("yes")
        judge_answer("Q", "GT", "unknown", llm=llm, mode="adversarial")
        assert llm.call_count == 0

    def test_llm_none_allowed_in_adversarial_mode(self):
        score = judge_answer("Q", "GT", "information not found", llm=None, mode="adversarial")
        assert score == 1.0


# ===========================================================================
# TestJudgeAnswerBinaryModes — LongMemEval
# ===========================================================================


class TestJudgeAnswerBinaryModes:
    @pytest.mark.parametrize(
        "mode",
        ["temporal_reasoning", "knowledge_update", "preference", "default_binary"],
    )
    def test_yes_returns_1(self, mode: str):
        llm = MockLLMClient("yes")
        score = judge_answer("Q", "GT", "P", llm=llm, mode=mode)
        assert score == 1.0

    @pytest.mark.parametrize(
        "mode",
        ["temporal_reasoning", "knowledge_update", "preference", "default_binary"],
    )
    def test_no_returns_0(self, mode: str):
        llm = MockLLMClient("no")
        score = judge_answer("Q", "GT", "P", llm=llm, mode=mode)
        assert score == 0.0

    def test_yes_with_trailing_text_returns_1(self):
        llm = MockLLMClient("yes, because the answer is correct")
        score = judge_answer("Q", "GT", "P", llm=llm, mode="default_binary")
        assert score == 1.0

    def test_temporal_prompt_includes_question(self):
        llm = MockLLMClient("yes")
        judge_answer("my_temporal_question", "GT", "P", llm=llm, mode="temporal_reasoning")
        assert "my_temporal_question" in str(llm.all_messages)

    def test_knowledge_update_prompt_includes_ground_truth(self):
        llm = MockLLMClient("yes")
        judge_answer("Q", "unique_gold_answer", "P", llm=llm, mode="knowledge_update")
        assert "unique_gold_answer" in str(llm.all_messages)

    def test_preference_prompt_has_rubric_tag(self):
        llm = MockLLMClient("yes")
        judge_answer("Q", "GT", "P", llm=llm, mode="preference")
        assert "RUBRIC" in str(llm.all_messages)

    def test_llm_required_for_binary_modes(self):
        with pytest.raises(ValueError):
            judge_answer("Q", "GT", "P", llm=None, mode="temporal_reasoning")


# ===========================================================================
# TestBatchJudge
# ===========================================================================


class TestBatchJudge:
    def _make_results(
        self, n: int, question_type: QuestionType = QuestionType.SINGLE_HOP
    ) -> list[EvalResult]:
        return [
            EvalResult(
                sample_id="s",
                question_id=f"q{i}",
                question_type=question_type,
                question=f"Q{i}",
                ground_truth=f"GT{i}",
                prediction=f"P{i}",
            )
            for i in range(n)
        ]

    def test_returns_list_of_floats(self):
        results = self._make_results(3)
        llm = MockLLMClient([_json_response(1.0), _json_response(0.0), _json_response(0.8)])
        scores = batch_judge(results, llm=llm)
        assert isinstance(scores, list)
        assert all(isinstance(s, float) for s in scores)

    def test_length_matches_results(self):
        results = self._make_results(4)
        llm = MockLLMClient([_json_response(1.0)] * 4)
        scores = batch_judge(results, llm=llm)
        assert len(scores) == 4

    def test_scores_match_judge_responses(self):
        results = self._make_results(3)
        llm = MockLLMClient([_json_response(1.0), _json_response(0.0), _json_response(1.0)])
        scores = batch_judge(results, llm=llm)
        assert scores == [1.0, 0.0, 1.0]

    def test_empty_results_returns_empty(self):
        scores = batch_judge([], llm=MockLLMClient())
        assert scores == []

    def test_auto_mode_adversarial_skips_llm(self):
        results = self._make_results(2, question_type=QuestionType.ADVERSARIAL)
        results[0].prediction = "not found"  # unanswerable → 1.0
        results[1].prediction = "Paris"  # concrete → 0.0
        llm = MockLLMClient("yes")
        scores = batch_judge(results, llm=llm, mode="auto")
        assert scores == [1.0, 0.0]
        assert llm.call_count == 0

    def test_auto_mode_longmemeval_uses_binary_prompt(self):
        results = self._make_results(2, question_type=QuestionType.TEMPORAL_REASONING)
        llm = MockLLMClient(["yes", "no"])
        scores = batch_judge(results, llm=llm, mode="auto")
        assert scores == [1.0, 0.0]

    def test_explicit_mode_overrides_auto(self):
        results = self._make_results(2, question_type=QuestionType.ADVERSARIAL)
        # Forcing continuous mode even for ADVERSARIAL type
        llm = MockLLMClient([_json_response(0.9), _json_response(0.2)])
        scores = batch_judge(results, llm=llm, mode="continuous")
        assert abs(scores[0] - 0.9) < 1e-9
        assert abs(scores[1] - 0.2) < 1e-9

    def test_cache_prevents_second_llm_call(self):
        results = self._make_results(2)
        llm = MockLLMClient([_json_response(1.0), _json_response(0.0)])
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache.json"
            batch_judge(results, llm=llm, cache_path=cache)
            llm2 = MockLLMClient()
            batch_judge(results, llm=llm2, cache_path=cache)
            assert llm2.call_count == 0

    def test_cache_file_is_written(self):
        results = self._make_results(2)
        llm = MockLLMClient([_json_response(1.0), _json_response(0.0)])
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache.json"
            batch_judge(results, llm=llm, cache_path=cache)
            assert cache.exists()
            data = json.loads(cache.read_text())
            assert len(data) == 2

    def test_cache_scores_are_correct(self):
        results = self._make_results(2)
        llm = MockLLMClient([_json_response(1.0), _json_response(0.0)])
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache.json"
            batch_judge(results, llm=llm, cache_path=cache)
            llm2 = MockLLMClient()
            scores = batch_judge(results, llm=llm2, cache_path=cache)
            assert scores == [1.0, 0.0]


# ===========================================================================
# TestExactMatch
# ===========================================================================


class TestExactMatch:
    def test_identical_strings_return_1(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", "PARIS") == 1.0

    def test_different_strings_return_0(self):
        assert exact_match("Paris", "London") == 0.0

    def test_whitespace_stripped(self):
        assert exact_match("  Paris  ", "Paris") == 1.0

    def test_empty_prediction(self):
        assert exact_match("", "Paris") == 0.0

    def test_both_empty(self):
        assert exact_match("", "") == 1.0

    def test_returns_float(self):
        assert isinstance(exact_match("a", "b"), float)


# ===========================================================================
# TestTokenF1
# ===========================================================================


class TestTokenF1:
    def test_exact_match_is_1(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_no_overlap_is_0(self):
        assert token_f1("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self):
        score = token_f1("the cat sat on the mat", "the cat")
        assert 0.0 < score < 1.0

    def test_empty_prediction_is_0(self):
        assert token_f1("", "ground truth") == 0.0

    def test_empty_ground_truth_is_0(self):
        assert token_f1("prediction", "") == 0.0

    def test_both_empty_is_0(self):
        assert token_f1("", "") == 0.0

    def test_single_token_match(self):
        assert token_f1("cat", "cat") == 1.0

    def test_single_token_no_match(self):
        assert token_f1("cat", "dog") == 0.0

    def test_symmetric_partial_overlap(self):
        assert abs(token_f1("a b", "b c") - 0.5) < 1e-9

    def test_known_value(self):
        # pred={a,b,c}, gt={a,b}: overlap=2, prec=2/3, rec=1 → F1=4/5=0.8
        assert abs(token_f1("a b c", "a b") - 0.8) < 1e-9

    def test_case_normalisation(self):
        assert token_f1("The Cat", "the cat") == 1.0

    def test_punctuation_normalisation(self):
        assert token_f1("cat.", "cat") == 1.0


# ===========================================================================
# TestRougeScores
# ===========================================================================


class TestRougeScores:
    def test_returns_dict_with_required_keys(self):
        result = rouge_scores("The cat sat", "The cat sat on the mat")
        assert {"rouge1", "rouge2", "rougeL"} <= result.keys()

    def test_exact_match_gives_high_rouge1(self):
        result = rouge_scores("hello world", "hello world")
        assert result["rouge1"] == pytest.approx(1.0, abs=1e-3)

    def test_no_overlap_gives_zero(self):
        result = rouge_scores("apple banana", "cat dog fish")
        assert result["rouge1"] == pytest.approx(0.0, abs=1e-3)

    def test_empty_prediction_gives_zero(self):
        result = rouge_scores("", "reference text")
        assert all(v == 0.0 for v in result.values())

    def test_empty_ground_truth_gives_zero(self):
        result = rouge_scores("prediction text", "")
        assert all(v == 0.0 for v in result.values())

    def test_scores_in_range(self):
        result = rouge_scores("some text here", "some different text")
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_returns_floats(self):
        result = rouge_scores("text", "text")
        assert all(isinstance(v, float) for v in result.values())


# ===========================================================================
# TestBleuScores
# ===========================================================================


class TestBleuScores:
    def test_returns_dict_with_required_keys(self):
        result = bleu_scores("the cat sat", "the cat sat on the mat")
        assert {"bleu1", "bleu2", "bleu4"} <= result.keys()

    def test_empty_prediction_gives_zero(self):
        result = bleu_scores("", "reference text")
        assert all(v == 0.0 for v in result.values())

    def test_empty_ground_truth_gives_zero(self):
        result = bleu_scores("prediction", "")
        assert all(v == 0.0 for v in result.values())

    def test_scores_in_range(self):
        result = bleu_scores("hello world there", "hello world again")
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_returns_floats(self):
        result = bleu_scores("text", "text")
        assert all(isinstance(v, float) for v in result.values())

    def test_exact_match_bleu1_high(self):
        result = bleu_scores("the quick brown fox", "the quick brown fox")
        assert result["bleu1"] > 0.9


# ===========================================================================
# TestMeteor
# ===========================================================================


class TestMeteor:
    def test_returns_float(self):
        score = meteor("the cat sat", "the cat sat on the mat")
        assert isinstance(score, float)

    def test_score_in_range(self):
        score = meteor("hello world", "hello world again")
        assert 0.0 <= score <= 1.0

    def test_exact_match_high_score(self):
        score = meteor("the quick brown fox", "the quick brown fox")
        assert score > 0.9

    def test_empty_prediction_gives_zero(self):
        assert meteor("", "reference") == 0.0

    def test_empty_ground_truth_gives_zero(self):
        assert meteor("prediction", "") == 0.0


# ===========================================================================
# TestBertF1 (integration — downloads BERT model)
# ===========================================================================


class TestBertF1:
    """Tests for bert_f1().

    Tests that load the model (roberta-large, ~1.3 GB) are marked ``slow``
    and skipped in fast CI runs.  Empty-input tests short-circuit before any
    model load and run unconditionally.
    """

    def test_empty_prediction_gives_zero(self):
        assert bert_f1("", "reference") == 0.0

    def test_empty_ground_truth_gives_zero(self):
        assert bert_f1("prediction", "") == 0.0

    @pytest.mark.slow
    def test_returns_float(self):
        score = bert_f1("the cat sat", "the cat sat on the mat")
        assert isinstance(score, float)

    @pytest.mark.slow
    def test_score_in_range(self):
        score = bert_f1("hello world", "hello world again")
        assert 0.0 <= score <= 1.0

    @pytest.mark.slow
    def test_exact_match_high_bertscore(self):
        score = bert_f1("University of Melbourne", "University of Melbourne")
        assert score > 0.95


# ===========================================================================
# TestSbertSimilarity  (slow — downloads all-MiniLM-L6-v2, ~80 MB)
# ===========================================================================


class TestSbertSimilarity:
    """Tests for sbert_similarity().

    Tests that load the model (all-MiniLM-L6-v2, ~80 MB) are marked ``slow``
    and skipped in fast CI runs.  Empty-input tests short-circuit before any
    model load and run unconditionally.
    """

    def test_empty_prediction_gives_zero(self):
        assert sbert_similarity("", "reference") == 0.0

    def test_empty_ground_truth_gives_zero(self):
        assert sbert_similarity("prediction", "") == 0.0

    @pytest.mark.slow
    def test_returns_float(self):
        score = sbert_similarity("the cat sat", "the cat sat on the mat")
        assert isinstance(score, float)

    @pytest.mark.slow
    def test_score_in_range(self):
        score = sbert_similarity("hello world", "hello world again")
        assert -1.0 <= score <= 1.0

    @pytest.mark.slow
    def test_identical_strings_high_similarity(self):
        score = sbert_similarity("Paris is the capital of France", "Paris is the capital of France")
        assert score > 0.99


# ===========================================================================
# TestComputeMetrics
# ===========================================================================


# Lightweight subset for fast tests — no heavy models loaded
_FAST_METRICS = (
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
)


class TestComputeMetrics:
    """Tests for compute_metrics().

    Fast tests pass ``metrics=_FAST_METRICS`` to avoid loading BERTScore /
    SBERT models.  Slow tests (marked ``slow``) use the default (all metrics).
    """

    def test_returns_dict(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 1.0)])
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert isinstance(out, dict)

    def test_has_overall_key(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 1.0)])
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert "overall" in out

    def test_has_by_type_key(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 1.0)])
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert "by_type" in out

    def test_overall_has_all_base_metrics(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 1.0)])
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        for key in _FAST_METRICS:
            assert key in out["overall"], f"Missing key: {key}"

    def test_overall_judge_score_is_mean(self):
        results, scores = _make_eval_results(
            [(QuestionType.SINGLE_HOP, 1.0), (QuestionType.TEMPORAL, 0.0)]
        )
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert abs(out["overall"]["judge_score"] - 0.5) < 1e-9

    def test_overall_exact_match_correct(self):
        results = [
            EvalResult(
                sample_id="s",
                question_id="q0",
                question_type=QuestionType.SINGLE_HOP,
                question="Q",
                ground_truth="Paris",
                prediction="Paris",
            ),
            EvalResult(
                sample_id="s",
                question_id="q1",
                question_type=QuestionType.SINGLE_HOP,
                question="Q",
                ground_truth="Paris",
                prediction="London",
            ),
        ]
        out = compute_metrics(results, [1.0, 0.0], metrics=_FAST_METRICS)
        assert abs(out["overall"]["exact_match"] - 0.5) < 1e-9

    def test_by_type_contains_present_types(self):
        results, scores = _make_eval_results(
            [(QuestionType.SINGLE_HOP, 1.0), (QuestionType.TEMPORAL, 0.0)]
        )
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert "single_hop" in out["by_type"]
        assert "temporal" in out["by_type"]

    def test_by_type_entry_has_required_keys(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 1.0)])
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        entry = out["by_type"]["single_hop"]
        for key in ("judge_score", "exact_match", "f1", "count"):
            assert key in entry

    def test_by_type_count_is_correct(self):
        results, scores = _make_eval_results(
            [
                (QuestionType.SINGLE_HOP, 1.0),
                (QuestionType.SINGLE_HOP, 0.0),
                (QuestionType.TEMPORAL, 1.0),
            ]
        )
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert out["by_type"]["single_hop"]["count"] == 2
        assert out["by_type"]["temporal"]["count"] == 1

    def test_by_type_judge_score_per_type(self):
        results, scores = _make_eval_results(
            [
                (QuestionType.SINGLE_HOP, 1.0),
                (QuestionType.SINGLE_HOP, 0.0),
                (QuestionType.TEMPORAL, 1.0),
            ]
        )
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert abs(out["by_type"]["single_hop"]["judge_score"] - 0.5) < 1e-9
        assert abs(out["by_type"]["temporal"]["judge_score"] - 1.0) < 1e-9

    def test_exact_match_prediction_raises_f1_to_1(self):
        results = [
            EvalResult(
                sample_id="s",
                question_id="q0",
                question_type=QuestionType.SINGLE_HOP,
                question="Q",
                ground_truth="Paris",
                prediction="Paris",
            )
        ]
        out = compute_metrics(results, [1.0], metrics=_FAST_METRICS)
        assert abs(out["overall"]["f1"] - 1.0) < 1e-9

    def test_no_overlap_reduces_f1_to_0(self):
        results = [
            EvalResult(
                sample_id="s",
                question_id="q0",
                question_type=QuestionType.SINGLE_HOP,
                question="Q",
                ground_truth="Paris",
                prediction="London",
            )
        ]
        out = compute_metrics(results, [0.0], metrics=_FAST_METRICS)
        assert abs(out["overall"]["f1"] - 0.0) < 1e-9

    def test_empty_results_returns_zeros_with_fast_keys(self):
        out = compute_metrics([], [], metrics=_FAST_METRICS)
        assert out["overall"]["judge_score"] == 0.0
        assert out["overall"]["f1"] == 0.0
        assert out["overall"]["exact_match"] == 0.0
        assert out["by_type"] == {}

    def test_longmemeval_type_grouped_correctly(self):
        results, scores = _make_eval_results(
            [
                (QuestionType.TEMPORAL_REASONING, 1.0),
                (QuestionType.KNOWLEDGE_UPDATE, 0.0),
            ]
        )
        out = compute_metrics(results, scores, metrics=_FAST_METRICS)
        assert "temporal_reasoning" in out["by_type"]
        assert "knowledge_update" in out["by_type"]

    def test_metrics_subset_only_returns_requested_keys(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 0.8)])
        out = compute_metrics(results, scores, metrics=("judge_score", "f1"))
        assert set(out["overall"].keys()) == {"judge_score", "f1"}

    def test_all_metrics_constant_covers_all_names(self):
        assert "bert_f1" in ALL_METRICS
        assert "sbert_similarity" in ALL_METRICS
        assert "f1" in ALL_METRICS

    @pytest.mark.slow
    def test_full_metrics_includes_bert_and_sbert(self):
        results, scores = _make_eval_results([(QuestionType.SINGLE_HOP, 1.0)])
        out = compute_metrics(results, scores)  # default = all metrics
        assert "bert_f1" in out["overall"]
        assert "sbert_similarity" in out["overall"]
        assert -0.01 <= out["overall"]["bert_f1"] <= 1.01  # BERTScore can slightly exceed 1.0
        assert -1.01 <= out["overall"]["sbert_similarity"] <= 1.01
