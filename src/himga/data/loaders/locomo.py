"""LoCoMo dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import EvidenceRef, Message, QAPair, QuestionType, Sample, Session
from ..temporal import parse_date

_CATEGORY_TO_QTYPE: dict[int, QuestionType] = {
    1: QuestionType.SINGLE_HOP,
    2: QuestionType.TEMPORAL,
    3: QuestionType.MULTI_HOP,
    4: QuestionType.OPEN_DOMAIN,
    5: QuestionType.ADVERSARIAL,
}


def _parse_session(session_num: int, turns: list[dict], date_str: str | None) -> Session:
    messages = []
    for turn in turns:
        text = turn.get("text", "")
        if "blip_caption" in turn:
            caption = f"[Image: {turn['blip_caption']}]"
            text = f"{caption} {text}".strip() if text else caption
        messages.append(
            Message(
                role=turn["speaker"],
                content=text,
                turn_id=turn.get("dia_id"),
            )
        )
    return Session(
        session_id=str(session_num),
        messages=messages,
        date_str=date_str,
        date=parse_date(date_str),
    )


def _parse_qa(raw: dict, idx: int) -> QAPair:
    category = raw.get("category")
    qtype = _CATEGORY_TO_QTYPE.get(category, QuestionType.SINGLE_HOP)

    if category == 5:
        answer = raw.get("adversarial_answer") or raw.get("answer") or ""
    else:
        answer = raw.get("answer")
        answer = str(answer) if answer is not None else ""

    return QAPair(
        question_id=str(idx),
        question=raw["question"],
        answer=answer,
        question_type=qtype,
        evidence=EvidenceRef(turn_ids=raw.get("evidence", [])),
        raw=raw,
    )


def _parse_sample(raw: dict, fallback_id: str) -> Sample:
    conv = raw["conversation"]

    sessions: list[Session] = []
    for key, value in conv.items():
        if (
            key.startswith("session_")
            and not key.endswith("_date_time")
            and isinstance(value, list)
        ):
            session_num = int(key.split("_")[1])
            date_str = conv.get(f"{key}_date_time")
            session = _parse_session(session_num, value, date_str)
            if session.messages:
                sessions.append(session)
    sessions.sort(key=lambda s: int(s.session_id))

    qa_pairs = [_parse_qa(qa, i) for i, qa in enumerate(raw.get("qa", []))]

    aux_keys = ("event_summary", "observation", "session_summary")
    return Sample(
        sample_id=raw.get("sample_id", fallback_id),
        dataset="locomo",
        sessions=sessions,
        qa_pairs=qa_pairs,
        speaker_a=conv.get("speaker_a"),
        speaker_b=conv.get("speaker_b"),
        raw={k: raw[k] for k in aux_keys if k in raw},
    )


def load_locomo(
    path: Path,
    *,
    limit: int | None = None,
    sample_ids: frozenset[str] | None = None,
) -> list[Sample]:
    """Load LoCoMo samples from a directory of JSON files.

    Parameters
    ----------
    path : Path
        Directory returned by ``get_dataset("locomo")``.
    limit : int or None
        Stop after loading this many samples. ``None`` loads all.
    sample_ids : frozenset[str] or None
        If given, only load samples whose ``sample_id`` is in this set.

    Returns
    -------
    list[Sample]
    """
    samples: list[Sample] = []
    global_idx = 0
    for fp in sorted(path.iterdir()):
        if not fp.is_file() or fp.suffix != ".json":
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        for raw in data:
            if not isinstance(raw, dict) or "conversation" not in raw:
                continue
            sid = raw.get("sample_id", str(global_idx))
            if sample_ids is not None and sid not in sample_ids:
                global_idx += 1
                continue
            if limit is not None and len(samples) >= limit:
                return samples
            samples.append(_parse_sample(raw, str(global_idx)))
    return samples
