"""LongMemEval dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import ijson

from ..schema import EvidenceRef, Message, QAPair, QuestionType, Sample, Session
from ..temporal import parse_date

# Accepts both hyphenated (real dataset) and underscored (alternate) variants.
_QTYPE_MAP: dict[str, QuestionType] = {
    "single-session-preference": QuestionType.SINGLE_SESSION_PREFERENCE,
    "single-session-assistant": QuestionType.SINGLE_SESSION_ASSISTANT,
    "temporal-reasoning": QuestionType.TEMPORAL_REASONING,
    "multi-session": QuestionType.MULTI_SESSION,
    "knowledge-update": QuestionType.KNOWLEDGE_UPDATE,
    "single-session-user": QuestionType.SINGLE_SESSION_USER,
    "single_session_preference": QuestionType.SINGLE_SESSION_PREFERENCE,
    "single_session_assistant": QuestionType.SINGLE_SESSION_ASSISTANT,
    "temporal_reasoning": QuestionType.TEMPORAL_REASONING,
    "multi_session": QuestionType.MULTI_SESSION,
    "knowledge_update": QuestionType.KNOWLEDGE_UPDATE,
    "single_session_user": QuestionType.SINGLE_SESSION_USER,
}


def _parse_session(messages_raw: list[dict], session_id: str, date_str: str | None) -> Session:
    messages = [
        Message(role=m.get("role", "user"), content=m.get("content", "")) for m in messages_raw
    ]
    return Session(
        session_id=session_id, messages=messages, date_str=date_str, date=parse_date(date_str)
    )


def _parse_question(raw: dict) -> Sample:
    sessions = [
        _parse_session(msgs, sid, date)
        for msgs, sid, date in zip(
            raw["haystack_sessions"],
            raw["haystack_session_ids"],
            raw["haystack_dates"],
        )
    ]

    qtype = _QTYPE_MAP.get(raw["question_type"], QuestionType.SINGLE_SESSION_USER)
    qa = QAPair(
        question_id=raw["question_id"],
        question=raw["question"],
        answer=str(raw["answer"]),
        question_type=qtype,
        evidence=EvidenceRef(session_ids=raw.get("answer_session_ids", [])),
        raw=raw,
    )

    return Sample(
        sample_id=raw["question_id"],
        dataset="longmemeval",
        sessions=sessions,
        qa_pairs=[qa],
        question_date=parse_date(raw.get("question_date")),
        raw={"question_date": raw.get("question_date")},
    )


def _iter_file(fp: Path) -> Iterator[dict]:
    """Stream top-level JSON array items from *fp* one at a time."""
    with fp.open("rb") as f:
        yield from ijson.items(f, "item")


def load_longmemeval(
    path: Path,
    *,
    limit: int | None = None,
    sample_ids: frozenset[str] | None = None,
) -> list[Sample]:
    """Load LongMemEval samples from a directory of JSON files.

    Uses streaming JSON parsing so even the 2.7 GB ``_m`` file is read
    incrementally — ``limit=10`` reads only the first ~few MB.

    Parameters
    ----------
    path : Path
        Directory returned by ``get_dataset("longmemeval")``.
    limit : int or None
        Stop after loading this many samples. ``None`` loads all.
    sample_ids : frozenset[str] or None
        If given, only load samples whose ``question_id`` is in this set.

    Returns
    -------
    list[Sample]
    """
    samples: list[Sample] = []
    for fp in sorted(path.iterdir()):
        if not fp.is_file() or fp.suffix != ".json":
            continue
        for raw in _iter_file(fp):
            if sample_ids is not None and raw["question_id"] not in sample_ids:
                continue
            if limit is not None and len(samples) >= limit:
                return samples
            samples.append(_parse_question(raw))
    return samples
