"""Temporal parsing utilities for LoCoMo and LongMemEval date strings."""

from __future__ import annotations

import re
from datetime import datetime

# LoCoMo: "1:56 pm on 8 May, 2023"  /  "10:37 am on 27 June, 2023"
_LOCOMO_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s+(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})",
    re.IGNORECASE,
)

# LongMemEval: "2023/05/20 (Sat) 02:21"
_LME_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s+\(\w+\)\s+(\d{2}):(\d{2})")

_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


class TemporalParser:
    """Parse raw date strings from LoCoMo and LongMemEval into datetime objects.

    Supports two formats:

    * **LoCoMo** — ``"H:MM am/pm on D Month, YYYY"``
      e.g. ``"1:56 pm on 8 May, 2023"``

    * **LongMemEval** — ``"YYYY/MM/DD (weekday) HH:MM"``
      e.g. ``"2023/05/20 (Sat) 02:21"``

    Returns ``None`` for ``None`` input, empty strings, or unrecognised formats.
    """

    def parse(self, date_str: str | None) -> datetime | None:
        """Parse a raw date string into a :class:`datetime`.

        Parameters
        ----------
        date_str : str or None
            Raw timestamp string from either dataset.

        Returns
        -------
        datetime or None
            Parsed datetime (no timezone), or ``None`` if unparseable.
        """
        if not date_str:
            return None

        s = date_str.strip()

        dt = self._try_lme(s)
        if dt is not None:
            return dt

        return self._try_locomo(s)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_lme(self, s: str) -> datetime | None:
        m = _LME_RE.match(s)
        if not m:
            return None
        year, month, day, hour, minute = (int(x) for x in m.groups())
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    def _try_locomo(self, s: str) -> datetime | None:
        m = _LOCOMO_RE.search(s)
        if not m:
            return None
        hour_s, min_s, ampm, day_s, month_s, year_s = m.groups()
        month = _MONTH_MAP.get(month_s.lower())
        if month is None:
            return None
        hour = int(hour_s)
        # Standard 12-hour conversion
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
        try:
            return datetime(int(year_s), month, int(day_s), hour, int(min_s))
        except ValueError:
            return None


# Module-level singleton — loaders import this directly.
_PARSER = TemporalParser()


def parse_date(date_str: str | None) -> datetime | None:
    """Parse a raw date string from either dataset into a :class:`datetime`.

    Convenience wrapper around :class:`TemporalParser` for import without
    instantiation.

    Parameters
    ----------
    date_str : str or None
        Raw timestamp string, e.g. ``"1:56 pm on 8 May, 2023"`` or
        ``"2023/05/20 (Sat) 02:21"``.

    Returns
    -------
    datetime or None
    """
    return _PARSER.parse(date_str)
