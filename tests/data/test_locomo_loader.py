"""Tests for the LoCoMo dataset loader.

Fixture layout (tests/data/fixtures/locomo/):
  sample.json          — 1 LoCoMo sample with all 5 QA categories,
                         2 image turns, and an orphan session_3_date_time.
  msc_personas_all.json — non-LoCoMo file (train/valid/test dict) that
                          mirrors the real dataset directory and must be
                          silently skipped by the loader.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from himga.data.loaders.locomo import load_locomo
from himga.data.schema import QuestionType, Sample


@pytest.fixture
def samples(locomo_path) -> list[Sample]:
    return load_locomo(locomo_path)


@pytest.fixture
def sample(samples) -> Sample:
    return samples[0]


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------


class TestDirectoryScanning:
    def test_non_locomo_json_in_dir_is_silently_skipped(self, locomo_path):
        """msc_personas_all.json (train/valid/test dict) must not raise an error."""
        noise_file = locomo_path / "msc_personas_all.json"
        assert noise_file.exists(), "fixture must include msc_personas_all.json"
        result = load_locomo(locomo_path)
        assert len(result) > 0

    def test_only_locomo_samples_counted(self, locomo_path):
        """Non-LoCoMo files contribute zero samples to the result."""
        samples = load_locomo(locomo_path)
        # Fixture: 1 LoCoMo sample in sample.json, 0 from msc_personas_all.json
        assert len(samples) == 1

    def test_non_json_files_are_ignored(self, tmp_path):
        src = Path(__file__).parent / "fixtures" / "locomo"
        dest = tmp_path / "locomo"
        shutil.copytree(src, dest)
        (dest / "README.txt").write_text("not json")
        result = load_locomo(dest)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# limit and sample_ids parameters
# ---------------------------------------------------------------------------


class TestLoadingOptions:
    def test_limit_zero_returns_empty(self, locomo_path):
        assert load_locomo(locomo_path, limit=0) == []

    def test_limit_one_returns_one_sample(self, locomo_path):
        result = load_locomo(locomo_path, limit=1)
        assert len(result) == 1

    def test_limit_larger_than_total_returns_all(self, locomo_path):
        result = load_locomo(locomo_path, limit=9999)
        assert len(result) == 1

    def test_sample_ids_exact_match(self, locomo_path):
        result = load_locomo(locomo_path, sample_ids=frozenset(["conv-test-1"]))
        assert len(result) == 1
        assert result[0].sample_id == "conv-test-1"

    def test_sample_ids_no_match_returns_empty(self, locomo_path):
        result = load_locomo(locomo_path, sample_ids=frozenset(["nonexistent-id"]))
        assert result == []

    def test_limit_and_sample_ids_combined(self, locomo_path):
        result = load_locomo(
            locomo_path,
            limit=1,
            sample_ids=frozenset(["conv-test-1"]),
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Top-level Sample fields
# ---------------------------------------------------------------------------


class TestSampleFields:
    def test_returns_list_of_samples(self, samples):
        assert isinstance(samples, list)
        assert all(isinstance(s, Sample) for s in samples)

    def test_dataset_tag_is_locomo(self, samples):
        assert all(s.dataset == "locomo" for s in samples)

    def test_sample_id_preserved(self, sample):
        assert sample.sample_id == "conv-test-1"

    def test_speaker_a_preserved(self, sample):
        assert sample.speaker_a == "Alice"

    def test_speaker_b_preserved(self, sample):
        assert sample.speaker_b == "Bob"

    def test_question_date_is_none(self, samples):
        """LoCoMo has no per-question date — question_date must always be None."""
        assert all(s.question_date is None for s in samples)


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------


class TestSessionParsing:
    def test_sessions_list_is_nonempty(self, sample):
        assert len(sample.sessions) > 0

    def test_session_count_excludes_orphan_date_time(self, sample):
        """session_3_date_time exists in fixture but session_3 list does not —
        only sessions with actual message lists are created."""
        assert len(sample.sessions) == 2

    def test_orphan_session_3_date_time_not_in_result(self, sample):
        session_ids = {sess.session_id for sess in sample.sessions}
        assert "3" not in session_ids

    def test_sessions_sorted_numerically_by_id(self, samples):
        for s in samples:
            ids = [int(sess.session_id) for sess in s.sessions]
            assert ids == sorted(ids)

    def test_session_id_values(self, sample):
        ids = [sess.session_id for sess in sample.sessions]
        assert ids == ["1", "2"]

    def test_session_date_str_preserved(self, sample):
        assert sample.sessions[0].date_str == "2:00 pm on 1 Jan, 2024"
        assert sample.sessions[1].date_str == "3:00 pm on 2 Jan, 2024"

    def test_session_date_parsed_to_datetime(self, sample):
        assert sample.sessions[0].date == datetime(2024, 1, 1, 14, 0)
        assert sample.sessions[1].date == datetime(2024, 1, 2, 15, 0)


# ---------------------------------------------------------------------------
# Message parsing (including image turns)
# ---------------------------------------------------------------------------


class TestMessageParsing:
    def test_messages_nonempty_per_session(self, sample):
        assert all(len(sess.messages) > 0 for sess in sample.sessions)

    def test_message_role_is_speaker_name(self, sample):
        first_msg = sample.sessions[0].messages[0]
        assert first_msg.role == "Alice"

    def test_turn_id_preserved(self, sample):
        first_msg = sample.sessions[0].messages[0]
        assert first_msg.turn_id == "D1:1"

    def test_all_turn_ids_follow_dia_id_format(self, samples):
        msgs = [m for s in samples for sess in s.sessions for m in sess.messages]
        ids = [m.turn_id for m in msgs if m.turn_id is not None]
        assert len(ids) > 0
        assert all(tid.startswith("D") and ":" in tid for tid in ids)

    def test_image_turn_empty_text_becomes_image_tag_only(self, sample):
        """D1:3: text='' + blip_caption → '[Image: caption]'."""
        msgs = sample.sessions[0].messages
        img_msg = next(m for m in msgs if m.turn_id == "D1:3")
        assert img_msg.content == "[Image: a dog playing in the park]"

    def test_image_turn_nonempty_text_prepends_image_tag(self, sample):
        """D1:4: text='Great photo!' + blip_caption → '[Image: caption] Great photo!'."""
        msgs = sample.sessions[0].messages
        img_msg = next(m for m in msgs if m.turn_id == "D1:4")
        assert img_msg.content == "[Image: a sunset over mountains] Great photo!"

    def test_normal_turn_content_unchanged(self, sample):
        msg = next(m for m in sample.sessions[0].messages if m.turn_id == "D1:1")
        assert msg.content == "Hi Bob!"


# ---------------------------------------------------------------------------
# QA pairs
# ---------------------------------------------------------------------------


class TestQAParsing:
    def test_qa_pairs_nonempty(self, samples):
        assert all(len(s.qa_pairs) > 0 for s in samples)

    def test_all_five_question_types_present(self, sample):
        types = {qa.question_type for qa in sample.qa_pairs}
        assert types == {
            QuestionType.SINGLE_HOP,
            QuestionType.TEMPORAL,
            QuestionType.MULTI_HOP,
            QuestionType.OPEN_DOMAIN,
            QuestionType.ADVERSARIAL,
        }

    def test_all_answers_are_str(self, samples):
        for s in samples:
            for qa in s.qa_pairs:
                assert isinstance(qa.answer, str), f"{qa.answer!r} is not str"

    def test_int_answer_converted_to_str(self, sample):
        temporal = next(qa for qa in sample.qa_pairs if qa.question_type == QuestionType.TEMPORAL)
        assert temporal.answer == "2020"

    def test_adversarial_uses_adversarial_answer_not_regular_answer(self, sample):
        """category=5 fixture has NO 'answer' field — must read adversarial_answer."""
        adv = next(qa for qa in sample.qa_pairs if qa.question_type == QuestionType.ADVERSARIAL)
        assert adv.answer == "Bob moved from New York, not Los Angeles."

    def test_adversarial_answer_is_nonempty(self, sample):
        adv = next(qa for qa in sample.qa_pairs if qa.question_type == QuestionType.ADVERSARIAL)
        assert adv.answer != ""

    def test_single_hop_evidence_turn_ids(self, sample):
        single_hop = next(
            qa for qa in sample.qa_pairs if qa.question_type == QuestionType.SINGLE_HOP
        )
        assert single_hop.evidence.turn_ids == ["D2:2"]

    def test_multi_hop_evidence_multiple_turn_ids(self, sample):
        multi_hop = next(qa for qa in sample.qa_pairs if qa.question_type == QuestionType.MULTI_HOP)
        assert set(multi_hop.evidence.turn_ids) == {"D1:1", "D1:2"}

    def test_question_id_is_sequential_string(self, sample):
        ids = [qa.question_id for qa in sample.qa_pairs]
        assert ids == [str(i) for i in range(len(ids))]


# ---------------------------------------------------------------------------
# Auxiliary raw fields (event_summary, observation, session_summary)
# ---------------------------------------------------------------------------


class TestRawAuxFields:
    def test_all_three_aux_keys_present(self, sample):
        assert "event_summary" in sample.raw
        assert "observation" in sample.raw
        assert "session_summary" in sample.raw

    def test_event_summary_keys_match_real_format(self, sample):
        """Real format: events_session_N as top-level keys."""
        es = sample.raw["event_summary"]
        assert "events_session_1" in es
        assert "events_session_2" in es

    def test_event_summary_contains_speaker_lists_and_date(self, sample):
        entry = sample.raw["event_summary"]["events_session_1"]
        assert "Alice" in entry
        assert "Bob" in entry
        assert "date" in entry
        assert isinstance(entry["Alice"], list)
        assert isinstance(entry["Bob"], list)

    def test_observation_keys_match_real_format(self, sample):
        """Real format: session_N_observation as top-level keys."""
        obs = sample.raw["observation"]
        assert "session_1_observation" in obs
        assert "session_2_observation" in obs

    def test_observation_entries_are_text_dia_id_pairs(self, sample):
        obs = sample.raw["observation"]["session_1_observation"]
        alice_obs = obs["Alice"]
        assert len(alice_obs) > 0
        assert isinstance(alice_obs[0], list)
        assert len(alice_obs[0]) == 2  # [text, dia_id]
        assert alice_obs[0][1].startswith("D")  # dia_id format

    def test_session_summary_keys_match_real_format(self, sample):
        """Real format: session_N_summary as top-level keys."""
        ss = sample.raw["session_summary"]
        assert "session_1_summary" in ss
        assert "session_2_summary" in ss

    def test_session_summary_values_are_strings(self, sample):
        ss = sample.raw["session_summary"]
        assert isinstance(ss["session_1_summary"], str)
        assert len(ss["session_1_summary"]) > 0
