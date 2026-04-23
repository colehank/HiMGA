"""Tests for himga.agent.BaseAgent."""

from himga.agent import BaseAgent
from himga.data.schema import Message, QAPair, QuestionType, Sample, Session
from himga.memory import NullMemory

# ---------------------------------------------------------------------------
# Helpers / Shared mock
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Minimal LLM client that records calls and returns a fixed response."""

    def __init__(self, response: str = "mock answer"):
        self._response = response
        self.call_count = 0
        self.last_messages: list[dict] | None = None

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        self.last_messages = messages
        return self._response


class SpyMemory(NullMemory):
    """NullMemory that records ingest call arguments."""

    def __init__(self, retrieve_result: str = ""):
        self._ingest_calls: list[tuple[Message, Session]] = []
        self._reset_count = 0
        self._retrieve_result = retrieve_result

    def ingest(self, message: Message, session: Session) -> None:
        self._ingest_calls.append((message, session))

    def retrieve(self, query: str) -> str:
        return self._retrieve_result

    def reset(self) -> None:
        self._reset_count += 1


def _make_sample(
    *,
    n_sessions: int = 2,
    messages_per_session: int = 2,
    n_qa: int = 1,
) -> Sample:
    sessions = [
        Session(
            session_id=f"s{i}",
            messages=[
                Message(role="user", content=f"s{i} msg {j}") for j in range(messages_per_session)
            ],
        )
        for i in range(n_sessions)
    ]
    qa_pairs = [
        QAPair(
            question_id=f"q{k}",
            question=f"Question {k}",
            answer=f"Answer {k}",
            question_type=QuestionType.SINGLE_HOP,
        )
        for k in range(n_qa)
    ]
    return Sample(
        sample_id="sample_0",
        dataset="locomo",
        sessions=sessions,
        qa_pairs=qa_pairs,
    )


# ---------------------------------------------------------------------------
# TestBaseAgentConstruction
# ---------------------------------------------------------------------------


class TestBaseAgentConstruction:
    def test_instantiates_with_memory_and_llm(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        assert isinstance(agent, BaseAgent)

    def test_stores_memory_ref(self):
        mem = NullMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        assert agent.memory is mem

    def test_stores_llm_ref(self):
        llm = MockLLMClient()
        agent = BaseAgent(memory=NullMemory(), llm=llm)
        assert agent.llm is llm


# ---------------------------------------------------------------------------
# TestIngestSample
# ---------------------------------------------------------------------------


class TestIngestSample:
    def test_ingest_called_for_every_message(self):
        mem = SpyMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        sample = _make_sample(n_sessions=2, messages_per_session=3)
        agent.ingest_sample(sample)
        assert len(mem._ingest_calls) == 6  # 2 sessions × 3 messages

    def test_ingest_called_with_correct_message(self):
        mem = SpyMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        sample = _make_sample(n_sessions=1, messages_per_session=2)
        agent.ingest_sample(sample)
        ingested_contents = [m.content for m, _ in mem._ingest_calls]
        assert ingested_contents == ["s0 msg 0", "s0 msg 1"]

    def test_ingest_called_with_correct_session(self):
        mem = SpyMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        sample = _make_sample(n_sessions=2, messages_per_session=1)
        agent.ingest_sample(sample)
        session_ids = [s.session_id for _, s in mem._ingest_calls]
        assert session_ids == ["s0", "s1"]

    def test_empty_sample_does_not_raise(self):
        mem = SpyMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        sample = _make_sample(n_sessions=0, messages_per_session=0)
        agent.ingest_sample(sample)
        assert len(mem._ingest_calls) == 0

    def test_sessions_ingested_in_order(self):
        mem = SpyMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        sample = _make_sample(n_sessions=3, messages_per_session=1)
        agent.ingest_sample(sample)
        session_ids = [s.session_id for _, s in mem._ingest_calls]
        assert session_ids == ["s0", "s1", "s2"]


# ---------------------------------------------------------------------------
# TestAnswer
# ---------------------------------------------------------------------------


class TestAnswer:
    def test_returns_string(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient(response="42"))
        result = agent.answer("What is the answer?")
        assert isinstance(result, str)

    def test_returns_llm_response(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient(response="Paris"))
        assert agent.answer("Capital of France?") == "Paris"

    def test_calls_llm_once_per_question(self):
        llm = MockLLMClient()
        agent = BaseAgent(memory=NullMemory(), llm=llm)
        agent.answer("Q1")
        agent.answer("Q2")
        assert llm.call_count == 2

    def test_calls_retrieve_with_question(self):
        retrieved: list[str] = []

        class TrackingMemory(SpyMemory):
            def retrieve(self, query: str) -> str:
                retrieved.append(query)
                return self._retrieve_result

        agent = BaseAgent(memory=TrackingMemory(retrieve_result="ctx"), llm=MockLLMClient())
        agent.answer("My question")
        assert "My question" in retrieved


# ---------------------------------------------------------------------------
# TestBuildMessages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_no_context_messages_have_no_context_prefix(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("What time is it?", "")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert "Context:" not in user_content

    def test_no_context_user_message_is_just_question(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("What time is it?", "")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert user_content == "What time is it?"

    def test_with_context_user_message_contains_context(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "some context here")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert "some context here" in user_content

    def test_with_context_user_message_contains_question(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "ctx")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert "Q?" in user_content

    def test_with_context_prefix_is_present(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "ctx")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert "Context:" in user_content

    def test_system_message_is_first(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "")
        assert msgs[0]["role"] == "system"

    def test_user_message_is_last(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "")
        assert msgs[-1]["role"] == "user"

    def test_returns_list_of_dicts(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "")
        assert isinstance(msgs, list)
        assert all(isinstance(m, dict) for m in msgs)

    def test_messages_have_role_and_content_keys(self):
        agent = BaseAgent(memory=NullMemory(), llm=MockLLMClient())
        msgs = agent._build_messages("Q?", "ctx")
        for m in msgs:
            assert "role" in m
            assert "content" in m


# ---------------------------------------------------------------------------
# TestAgentEndToEnd
# ---------------------------------------------------------------------------


class TestAgentEndToEnd:
    """Integration-style test: ingest_sample → answer uses retrieved context."""

    def test_answer_after_ingest_does_not_raise(self):
        sample = _make_sample(n_sessions=2, messages_per_session=3, n_qa=2)
        llm = MockLLMClient(response="answer text")
        agent = BaseAgent(memory=NullMemory(), llm=llm)
        agent.ingest_sample(sample)
        for qa in sample.qa_pairs:
            result = agent.answer(qa.question)
            assert isinstance(result, str)

    def test_context_from_memory_appears_in_llm_call(self):
        mem = SpyMemory(retrieve_result="important context")
        llm = MockLLMClient()
        agent = BaseAgent(memory=mem, llm=llm)
        agent.answer("Some question")
        assert llm.last_messages is not None
        full_prompt = str(llm.last_messages)
        assert "important context" in full_prompt

    def test_memory_reset_clears_state_between_samples(self):
        mem = SpyMemory()
        agent = BaseAgent(memory=mem, llm=MockLLMClient())
        s1 = _make_sample(n_sessions=1, messages_per_session=2)
        s2 = _make_sample(n_sessions=1, messages_per_session=1)
        agent.memory.reset()
        agent.ingest_sample(s1)
        first_count = len(mem._ingest_calls)
        agent.memory.reset()
        agent.ingest_sample(s2)
        # After reset, only s2 messages are in the new ingest batch
        assert len(mem._ingest_calls) == first_count + 1
