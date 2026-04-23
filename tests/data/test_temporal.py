"""Tests for himga.data.temporal — TemporalParser and parse_date()."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from himga.data.temporal import TemporalParser, parse_date

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# TemporalParser — LongMemEval format
# ---------------------------------------------------------------------------


class TestLMEFormat:
    parser = TemporalParser()

    def test_basic_lme(self):
        dt = self.parser.parse("2023/05/20 (Sat) 02:21")
        assert dt == datetime(2023, 5, 20, 2, 21)

    def test_lme_end_of_day(self):
        dt = self.parser.parse("2023/05/30 (Tue) 23:40")
        assert dt == datetime(2023, 5, 30, 23, 40)

    def test_lme_midnight(self):
        dt = self.parser.parse("2024/01/01 (Mon) 00:00")
        assert dt == datetime(2024, 1, 1, 0, 0)

    def test_lme_different_weekday_abbreviations(self):
        # Weekday token is ignored — only the date/time matters
        dt = self.parser.parse("2023/02/15 (Wed) 23:50")
        assert dt == datetime(2023, 2, 15, 23, 50)


# ---------------------------------------------------------------------------
# TemporalParser — LoCoMo format
# ---------------------------------------------------------------------------


class TestLoCoMoFormat:
    parser = TemporalParser()

    def test_basic_locomo_pm(self):
        dt = self.parser.parse("1:56 pm on 8 May, 2023")
        assert dt == datetime(2023, 5, 8, 13, 56)

    def test_basic_locomo_am(self):
        dt = self.parser.parse("10:37 am on 27 June, 2023")
        assert dt == datetime(2023, 6, 27, 10, 37)

    def test_locomo_noon_pm(self):
        # 12 pm == 12:00 (noon)
        dt = self.parser.parse("12:00 pm on 5 March, 2023")
        assert dt == datetime(2023, 3, 5, 12, 0)

    def test_locomo_midnight_am(self):
        # 12 am == 00:00 (midnight)
        dt = self.parser.parse("12:00 am on 5 March, 2023")
        assert dt == datetime(2023, 3, 5, 0, 0)

    def test_locomo_1_am(self):
        dt = self.parser.parse("1:00 am on 10 January, 2024")
        assert dt == datetime(2024, 1, 10, 1, 0)

    def test_locomo_11_pm(self):
        dt = self.parser.parse("11:59 pm on 31 December, 2023")
        assert dt == datetime(2023, 12, 31, 23, 59)

    def test_locomo_abbreviated_month(self):
        dt = self.parser.parse("2:00 pm on 1 Jan, 2024")
        assert dt == datetime(2024, 1, 1, 14, 0)

    def test_locomo_case_insensitive_ampm(self):
        dt_lower = self.parser.parse("3:30 pm on 15 Aug, 2023")
        dt_upper = self.parser.parse("3:30 PM on 15 Aug, 2023")
        assert dt_lower == dt_upper == datetime(2023, 8, 15, 15, 30)

    def test_locomo_no_comma_after_day(self):
        dt = self.parser.parse("7:55 pm on 9 June 2023")
        assert dt == datetime(2023, 6, 9, 19, 55)


# ---------------------------------------------------------------------------
# Edge cases — None / empty / unrecognised
# ---------------------------------------------------------------------------


class TestEdgeCases:
    parser = TemporalParser()

    def test_none_returns_none(self):
        assert self.parser.parse(None) is None

    def test_empty_string_returns_none(self):
        assert self.parser.parse("") is None

    def test_whitespace_only_returns_none(self):
        assert self.parser.parse("   ") is None

    def test_unrecognised_format_returns_none(self):
        assert self.parser.parse("March 5, 2023") is None

    def test_iso_format_returns_none(self):
        # ISO-8601 date strings (used in old fixtures) should not parse
        assert self.parser.parse("2024-03-01") is None


# ---------------------------------------------------------------------------
# Module-level parse_date convenience function
# ---------------------------------------------------------------------------


class TestParseDateFunction:
    def test_delegates_to_parser_lme(self):
        assert parse_date("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21)

    def test_delegates_to_parser_locomo(self):
        assert parse_date("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56)

    def test_none_input(self):
        assert parse_date(None) is None


# ---------------------------------------------------------------------------
# Loader integration — LoCoMo
# ---------------------------------------------------------------------------


class TestLoCoMoLoaderIntegration:
    def test_session_date_parsed(self, locomo_path):
        from himga.data.loaders.locomo import load_locomo

        samples = load_locomo(locomo_path)
        sessions_with_date = [sess for s in samples for sess in s.sessions if sess.date is not None]
        assert len(sessions_with_date) > 0

    def test_session_date_is_datetime(self, locomo_path):
        from himga.data.loaders.locomo import load_locomo

        samples = load_locomo(locomo_path)
        for s in samples:
            for sess in s.sessions:
                if sess.date_str is not None:
                    assert isinstance(sess.date, datetime)

    def test_session_date_matches_date_str(self, locomo_path):
        from himga.data.loaders.locomo import load_locomo

        samples = load_locomo(locomo_path)
        sess = samples[0].sessions[0]
        # Fixture: "2:00 pm on 1 Jan, 2024"  → 2024-01-01 14:00
        assert sess.date == datetime(2024, 1, 1, 14, 0)

    def test_locomo_question_date_is_none(self, locomo_path):
        from himga.data.loaders.locomo import load_locomo

        samples = load_locomo(locomo_path)
        assert all(s.question_date is None for s in samples)


# ---------------------------------------------------------------------------
# Loader integration — LongMemEval
# ---------------------------------------------------------------------------


class TestLMELoaderIntegration:
    def test_session_date_parsed(self, longmemeval_path):
        from himga.data.loaders.longmemeval import load_longmemeval

        samples = load_longmemeval(longmemeval_path)
        sessions_with_date = [sess for s in samples for sess in s.sessions if sess.date is not None]
        assert len(sessions_with_date) > 0

    def test_session_date_is_datetime(self, longmemeval_path):
        from himga.data.loaders.longmemeval import load_longmemeval

        samples = load_longmemeval(longmemeval_path)
        for s in samples:
            for sess in s.sessions:
                if sess.date_str is not None:
                    assert isinstance(sess.date, datetime)

    def test_session_date_matches_date_str(self, longmemeval_path):
        from himga.data.loaders.longmemeval import load_longmemeval

        samples = load_longmemeval(longmemeval_path)
        sess = samples[0].sessions[0]
        # Fixture: "2024/03/01 (Fri) 10:00"  → 2024-03-01 10:00
        assert sess.date == datetime(2024, 3, 1, 10, 0)

    def test_question_date_parsed(self, longmemeval_path):
        from himga.data.loaders.longmemeval import load_longmemeval

        samples = load_longmemeval(longmemeval_path)
        assert all(s.question_date is not None for s in samples)

    def test_question_date_is_datetime(self, longmemeval_path):
        from himga.data.loaders.longmemeval import load_longmemeval

        samples = load_longmemeval(longmemeval_path)
        for s in samples:
            assert isinstance(s.question_date, datetime)

    def test_question_date_value(self, longmemeval_path):
        from himga.data.loaders.longmemeval import load_longmemeval

        samples = load_longmemeval(longmemeval_path)
        # Fixture question_date: "2024/03/15 (Fri) 09:00" → 2024-03-15 09:00
        assert samples[0].question_date == datetime(2024, 3, 15, 9, 0)
