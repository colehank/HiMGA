from himga.data.schema import (
    EvidenceRef,
    Message,
    QAPair,
    QuestionType,
    Sample,
    Session,
)


class TestMessage:
    def test_required_fields(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_optional_defaults_are_none(self):
        m = Message(role="user", content="hello")
        assert m.turn_id is None
        assert m.date_str is None

    def test_optional_fields_set(self):
        m = Message(role="Alice", content="hi", turn_id="D1:3", date_str="2024-01-01")
        assert m.turn_id == "D1:3"
        assert m.date_str == "2024-01-01"


class TestSession:
    def test_required_fields(self):
        s = Session(session_id="1", messages=[])
        assert s.session_id == "1"
        assert s.messages == []

    def test_optional_defaults_are_none(self):
        s = Session(session_id="1", messages=[])
        assert s.date_str is None
        assert s.date is None
        assert s.title is None


class TestEvidenceRef:
    def test_defaults_are_empty_lists(self):
        ref = EvidenceRef()
        assert ref.turn_ids == []
        assert ref.session_ids == []

    def test_independent_instances(self):
        a, b = EvidenceRef(), EvidenceRef()
        a.turn_ids.append("D1:1")
        assert b.turn_ids == []


class TestQAPair:
    def test_required_fields(self):
        qa = QAPair(
            question_id="q1",
            question="What?",
            answer="something",
            question_type=QuestionType.SINGLE_HOP,
        )
        assert qa.question_id == "q1"
        assert isinstance(qa.answer, str)

    def test_evidence_default_is_empty(self):
        qa = QAPair(
            question_id="q1",
            question="What?",
            answer="x",
            question_type=QuestionType.TEMPORAL,
        )
        assert qa.evidence.turn_ids == []
        assert qa.evidence.session_ids == []

    def test_raw_default_is_empty_dict(self):
        qa = QAPair(
            question_id="q1",
            question="Q",
            answer="A",
            question_type=QuestionType.MULTI_HOP,
        )
        assert qa.raw == {}


class TestSample:
    def test_required_fields(self):
        s = Sample(sample_id="s1", dataset="locomo", sessions=[], qa_pairs=[])
        assert s.dataset == "locomo"

    def test_optional_speaker_defaults_are_none(self):
        s = Sample(sample_id="s1", dataset="locomo", sessions=[], qa_pairs=[])
        assert s.speaker_a is None
        assert s.speaker_b is None

    def test_question_date_default_is_none(self):
        s = Sample(sample_id="s1", dataset="longmemeval", sessions=[], qa_pairs=[])
        assert s.question_date is None

    def test_raw_default_is_empty_dict(self):
        s = Sample(sample_id="s1", dataset="longmemeval", sessions=[], qa_pairs=[])
        assert s.raw == {}


class TestQuestionType:
    def test_is_string_enum(self):
        assert isinstance(QuestionType.SINGLE_HOP, str)
        assert QuestionType.SINGLE_HOP == "single_hop"

    def test_covers_locomo_categories(self):
        locomo_types = {
            QuestionType.SINGLE_HOP,
            QuestionType.MULTI_HOP,
            QuestionType.TEMPORAL,
            QuestionType.OPEN_DOMAIN,
            QuestionType.ADVERSARIAL,
        }
        assert len(locomo_types) == 5

    def test_covers_longmemeval_types(self):
        lme_types = {
            QuestionType.SINGLE_SESSION_PREFERENCE,
            QuestionType.SINGLE_SESSION_ASSISTANT,
            QuestionType.TEMPORAL_REASONING,
            QuestionType.MULTI_SESSION,
            QuestionType.KNOWLEDGE_UPDATE,
            QuestionType.SINGLE_SESSION_USER,
        }
        assert len(lme_types) == 6
