"""Tests for the LongMemEval dataset loader.

Fixture layout (tests/data/fixtures/longmemeval/):
  sample.json       — 7 questions covering all 6 QuestionTypes, int answer,
                      and one unknown type to test fallback behaviour.
  sample_extra.json — 1 additional question to exercise multi-file loading.

Alphabetical sort: sample.json < sample_extra.json, so samples[0] is always
from sample.json (question_id="q_001").
"""

from __future__ import annotations

from datetime import datetime

import pytest

from himga.data.loaders.longmemeval import load_longmemeval
from himga.data.schema import QuestionType, Sample


@pytest.fixture
def samples(longmemeval_path) -> list[Sample]:
    return load_longmemeval(longmemeval_path)


@pytest.fixture
def sample(samples) -> Sample:
    return next(s for s in samples if s.sample_id == "q_001")


# ---------------------------------------------------------------------------
# Directory scanning — multi-file loading
# ---------------------------------------------------------------------------


class TestDirectoryScanning:
    def test_multi_file_dir_loads_all_samples(self, longmemeval_path, samples):
        """Both sample.json and sample_extra.json are loaded."""
        extra_file = longmemeval_path / "sample_extra.json"
        assert extra_file.exists(), "fixture must include sample_extra.json"
        ids = {s.sample_id for s in samples}
        assert "q_001" in ids
        assert "q_extra_001" in ids

    def test_total_sample_count(self, samples):
        """7 items in sample.json + 1 in sample_extra.json = 8 total."""
        assert len(samples) == 8

    def test_non_json_files_are_ignored(self, tmp_path):
        import shutil
        from pathlib import Path

        src = Path(__file__).parent / "fixtures" / "longmemeval"
        dest = tmp_path / "longmemeval"
        shutil.copytree(src, dest)
        (dest / "README.txt").write_text("not json")
        result = load_longmemeval(dest)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# limit and sample_ids parameters
# ---------------------------------------------------------------------------


class TestLoadingOptions:
    def test_limit_zero_returns_empty(self, longmemeval_path):
        assert load_longmemeval(longmemeval_path, limit=0) == []

    def test_limit_one_returns_one_sample(self, longmemeval_path):
        result = load_longmemeval(longmemeval_path, limit=1)
        assert len(result) == 1

    def test_limit_three_returns_three_samples(self, longmemeval_path):
        result = load_longmemeval(longmemeval_path, limit=3)
        assert len(result) == 3

    def test_limit_larger_than_total_returns_all(self, longmemeval_path):
        result = load_longmemeval(longmemeval_path, limit=9999)
        assert len(result) == 8

    def test_sample_ids_exact_match(self, longmemeval_path):
        result = load_longmemeval(
            longmemeval_path,
            sample_ids=frozenset(["q_001", "q_003"]),
        )
        assert len(result) == 2
        assert {s.sample_id for s in result} == {"q_001", "q_003"}

    def test_sample_ids_no_match_returns_empty(self, longmemeval_path):
        result = load_longmemeval(
            longmemeval_path,
            sample_ids=frozenset(["nonexistent"]),
        )
        assert result == []

    def test_sample_ids_across_files(self, longmemeval_path):
        """q_001 is in sample.json, q_extra_001 is in sample_extra.json."""
        result = load_longmemeval(
            longmemeval_path,
            sample_ids=frozenset(["q_001", "q_extra_001"]),
        )
        assert {s.sample_id for s in result} == {"q_001", "q_extra_001"}

    def test_limit_and_sample_ids_combined(self, longmemeval_path):
        result = load_longmemeval(
            longmemeval_path,
            limit=2,
            sample_ids=frozenset(["q_001", "q_002", "q_003"]),
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Top-level Sample fields
# ---------------------------------------------------------------------------


class TestSampleFields:
    def test_returns_list_of_samples(self, samples):
        assert isinstance(samples, list)
        assert all(isinstance(s, Sample) for s in samples)

    def test_dataset_tag_is_longmemeval(self, samples):
        assert all(s.dataset == "longmemeval" for s in samples)

    def test_one_qa_pair_per_sample(self, samples):
        assert all(len(s.qa_pairs) == 1 for s in samples)

    def test_sample_id_equals_question_id(self, samples):
        for s in samples:
            assert s.sample_id == s.qa_pairs[0].question_id

    def test_speaker_a_is_none(self, samples):
        assert all(s.speaker_a is None for s in samples)

    def test_speaker_b_is_none(self, samples):
        assert all(s.speaker_b is None for s in samples)


# ---------------------------------------------------------------------------
# Session (haystack) parsing
# ---------------------------------------------------------------------------


class TestSessionParsing:
    def test_haystack_sessions_nonempty(self, samples):
        assert all(len(s.sessions) > 0 for s in samples)

    def test_session_messages_nonempty(self, samples):
        for s in samples:
            assert all(len(sess.messages) > 0 for sess in s.sessions)

    def test_session_count_matches_haystack_list(self, sample):
        """q_001 fixture has 2 haystack sessions."""
        assert len(sample.sessions) == 2

    def test_session_id_preserved(self, sample):
        assert sample.sessions[0].session_id == "sess_1"
        assert sample.sessions[1].session_id == "sess_2"

    def test_session_date_str_preserved(self, sample):
        assert sample.sessions[0].date_str == "2024/03/01 (Fri) 10:00"
        assert sample.sessions[1].date_str == "2024/03/10 (Sun) 14:30"

    def test_session_date_parsed_to_datetime(self, sample):
        assert sample.sessions[0].date == datetime(2024, 3, 1, 10, 0)
        assert sample.sessions[1].date == datetime(2024, 3, 10, 14, 30)

    def test_message_role_is_user_or_assistant(self, samples):
        for s in samples:
            for sess in s.sessions:
                for msg in sess.messages:
                    assert msg.role in {"user", "assistant"}, f"unexpected role {msg.role!r}"

    def test_message_turn_id_is_none(self, samples):
        """LongMemEval has no dia_id; turn_id must always be None."""
        for s in samples:
            for sess in s.sessions:
                for msg in sess.messages:
                    assert msg.turn_id is None

    def test_message_content_preserved(self, sample):
        first_msg = sample.sessions[0].messages[0]
        assert first_msg.content == "I bought 3 shirts at the mall today."


# ---------------------------------------------------------------------------
# QA pairs
# ---------------------------------------------------------------------------


class TestQAParsing:
    def test_all_answers_are_str(self, samples):
        for s in samples:
            assert isinstance(s.qa_pairs[0].answer, str), (
                f"answer for {s.sample_id} is not str: {s.qa_pairs[0].answer!r}"
            )

    def test_str_answer_preserved(self, sample):
        assert sample.qa_pairs[0].answer == "3 shirts"

    def test_int_answer_converted_to_str(self, samples):
        """q_003 has answer=3 (int) in the fixture — must become '3'."""
        q003 = next(s for s in samples if s.sample_id == "q_003")
        assert q003.qa_pairs[0].answer == "3"

    def test_question_id_preserved(self, sample):
        assert sample.qa_pairs[0].question_id == "q_001"

    def test_question_text_preserved(self, sample):
        assert sample.qa_pairs[0].question == "What did the user buy?"

    def test_evidence_session_ids_preserved(self, sample):
        assert sample.qa_pairs[0].evidence.session_ids == ["sess_1"]

    def test_multiple_evidence_session_ids(self, samples):
        """q_003 references two answer sessions."""
        q003 = next(s for s in samples if s.sample_id == "q_003")
        assert set(q003.qa_pairs[0].evidence.session_ids) == {"sess_1", "sess_3"}

    def test_empty_evidence_session_ids(self, samples):
        """q_007 has no answer_session_ids."""
        q007 = next(s for s in samples if s.sample_id == "q_007")
        assert q007.qa_pairs[0].evidence.session_ids == []


# ---------------------------------------------------------------------------
# Question type mapping
# ---------------------------------------------------------------------------


class TestQuestionTypeMapping:
    def test_all_six_types_present_across_fixture(self, samples):
        types = {s.qa_pairs[0].question_type for s in samples}
        assert QuestionType.SINGLE_SESSION_USER in types
        assert QuestionType.TEMPORAL_REASONING in types
        assert QuestionType.MULTI_SESSION in types
        assert QuestionType.KNOWLEDGE_UPDATE in types
        assert QuestionType.SINGLE_SESSION_PREFERENCE in types
        assert QuestionType.SINGLE_SESSION_ASSISTANT in types

    def test_hyphenated_type_single_session_user(self, sample):
        assert sample.qa_pairs[0].question_type == QuestionType.SINGLE_SESSION_USER

    def test_hyphenated_type_temporal_reasoning(self, samples):
        q002 = next(s for s in samples if s.sample_id == "q_002")
        assert q002.qa_pairs[0].question_type == QuestionType.TEMPORAL_REASONING

    def test_hyphenated_type_multi_session(self, samples):
        q003 = next(s for s in samples if s.sample_id == "q_003")
        assert q003.qa_pairs[0].question_type == QuestionType.MULTI_SESSION

    def test_hyphenated_type_knowledge_update(self, samples):
        q004 = next(s for s in samples if s.sample_id == "q_004")
        assert q004.qa_pairs[0].question_type == QuestionType.KNOWLEDGE_UPDATE

    def test_hyphenated_type_preference(self, samples):
        q005 = next(s for s in samples if s.sample_id == "q_005")
        assert q005.qa_pairs[0].question_type == QuestionType.SINGLE_SESSION_PREFERENCE

    def test_hyphenated_type_assistant(self, samples):
        q006 = next(s for s in samples if s.sample_id == "q_006")
        assert q006.qa_pairs[0].question_type == QuestionType.SINGLE_SESSION_ASSISTANT

    def test_unknown_question_type_falls_back_to_single_session_user(self, samples):
        """q_007 has type 'unknown-future-type' — must fall back gracefully."""
        q007 = next(s for s in samples if s.sample_id == "q_007")
        assert q007.qa_pairs[0].question_type == QuestionType.SINGLE_SESSION_USER


# ---------------------------------------------------------------------------
# question_date
# ---------------------------------------------------------------------------


class TestQuestionDate:
    def test_question_date_is_datetime(self, samples):
        for s in samples:
            assert isinstance(s.question_date, datetime), (
                f"{s.sample_id} question_date is {type(s.question_date)}"
            )

    def test_question_date_value(self, sample):
        """q_001 question_date = '2024/03/15 (Fri) 09:00'."""
        assert sample.question_date == datetime(2024, 3, 15, 9, 0)

    def test_question_date_raw_preserved_as_original_string(self, sample):
        assert sample.raw["question_date"] == "2024/03/15 (Fri) 09:00"

    def test_extra_file_question_date_parsed(self, samples):
        extra = next(s for s in samples if s.sample_id == "q_extra_001")
        assert extra.question_date == datetime(2024, 4, 1, 8, 0)
