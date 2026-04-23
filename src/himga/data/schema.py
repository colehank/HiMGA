"""Unified data schema for LoCoMo and LongMemEval datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QuestionType(str, Enum):
    """Question categories covering both LoCoMo and LongMemEval benchmarks.

    Used for per-category metric aggregation aligned with MAGMA Table 1 / Table 2.
    """

    # LoCoMo (category int → enum)
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    OPEN_DOMAIN = "open_domain"
    ADVERSARIAL = "adversarial"

    # LongMemEval (question_type string → enum)
    SINGLE_SESSION_PREFERENCE = "single_session_preference"
    SINGLE_SESSION_ASSISTANT = "single_session_assistant"
    TEMPORAL_REASONING = "temporal_reasoning"
    MULTI_SESSION = "multi_session"
    KNOWLEDGE_UPDATE = "knowledge_update"
    SINGLE_SESSION_USER = "single_session_user"


@dataclass
class Message:
    """One utterance in a conversation turn.

    Parameters
    ----------
    role : str
        Speaker identifier. LoCoMo uses real names (e.g. "Caroline");
        LongMemEval uses "user" / "assistant".
    content : str
        Text content of the message.
    turn_id : str or None
        LoCoMo dia_id, e.g. "D1:3". None for LongMemEval.
    date_str : str or None
        Raw timestamp string. Parsing is deferred to TemporalParser.
    """

    role: str
    content: str
    turn_id: str | None = None
    date_str: str | None = None


@dataclass
class Session:
    """A temporally bounded conversation segment.

    Parameters
    ----------
    session_id : str
        Dataset-provided identifier.
    messages : list[Message]
        Ordered utterances within this session.
    date_str : str or None
        Raw session timestamp preserved as-is.
        LoCoMo example:  ``"1:56 pm on 8 May, 2023"``
        LongMemEval example: ``"2023/05/20 (Sat) 02:21"``
    date : datetime or None
        Parsed datetime object populated by the loader via
        :func:`~himga.data.temporal.parse_date`. ``None`` if ``date_str``
        is absent or unrecognised.
    title : str or None
        Optional title (LongMemEval provides this; LoCoMo does not).
    """

    session_id: str
    messages: list[Message]
    date_str: str | None = None
    date: datetime | None = None
    title: str | None = None


@dataclass
class EvidenceRef:
    """Flexible evidence pointer covering both datasets.

    Parameters
    ----------
    turn_ids : list[str]
        LoCoMo-style turn-level references, e.g. ["D1:3", "D2:7"].
    session_ids : list[str]
        LongMemEval-style session-level references, e.g. ["session_2"].
    """

    turn_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)


@dataclass
class QAPair:
    """One question–answer evaluation pair.

    Parameters
    ----------
    question_id : str
        Unique identifier within the sample.
    question : str
        Natural language question.
    answer : str
        Ground-truth answer. Always str; LoCoMo adversarial resolution and
        int-to-str normalisation are handled in the loader.
    question_type : QuestionType
        Normalised category for per-type metric aggregation.
    evidence : EvidenceRef
        Pointers to source turns or sessions.
    raw : dict
        Original dataset fields preserved without modification.
    """

    question_id: str
    question: str
    answer: str
    question_type: QuestionType
    evidence: EvidenceRef = field(default_factory=EvidenceRef)
    raw: dict = field(default_factory=dict)


@dataclass
class Sample:
    """One evaluation unit: a conversation history plus associated QA pairs.

    LoCoMo  : one LoCoMoSample  → one Sample (multiple sessions, multiple QA pairs).
    LongMemEval: one LongMemQuestion → one Sample (haystack sessions, one QA pair).

    Parameters
    ----------
    sample_id : str
        Unique identifier.
    dataset : str
        Source dataset name: "locomo" or "longmemeval".
    sessions : list[Session]
        Ordered conversation history.
    qa_pairs : list[QAPair]
        Questions to answer using the conversation history.
    speaker_a : str or None
        LoCoMo primary speaker name. None for LongMemEval.
    speaker_b : str or None
        LoCoMo secondary speaker name. None for LongMemEval.
    question_date : datetime or None
        Parsed datetime of when the question was asked.
        Populated for LongMemEval (from ``question_date`` field);
        ``None`` for LoCoMo.
    raw : dict
        Dataset-specific auxiliary fields (event_summary, session_summary, etc.).
    """

    sample_id: str
    dataset: str
    sessions: list[Session]
    qa_pairs: list[QAPair]
    speaker_a: str | None = None
    speaker_b: str | None = None
    question_date: datetime | None = None
    raw: dict = field(default_factory=dict)
