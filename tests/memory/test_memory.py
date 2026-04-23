"""Tests for himga.memory: BaseMemory contract and NullMemory implementation."""

import pytest

from himga.data.schema import Message, Session
from himga.memory import BaseMemory, NullMemory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def message() -> Message:
    return Message(role="user", content="Hello, how are you?")


@pytest.fixture
def session() -> Session:
    return Session(session_id="s1", messages=[])


# ---------------------------------------------------------------------------
# TestBaseMemoryInterface
# ---------------------------------------------------------------------------


class TestBaseMemoryInterface:
    """BaseMemory cannot be instantiated directly; subclasses must implement all methods."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseMemory()  # type: ignore[abstract]

    def test_subclass_missing_ingest_raises(self):
        class Incomplete(BaseMemory):
            def retrieve(self, query):
                return ""

            def reset(self):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_missing_retrieve_raises(self):
        class Incomplete(BaseMemory):
            def ingest(self, message, session):
                pass

            def reset(self):
                pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_missing_reset_raises(self):
        class Incomplete(BaseMemory):
            def ingest(self, message, session):
                pass

            def retrieve(self, query):
                return ""

        with pytest.raises(TypeError):
            Incomplete()

    def test_full_subclass_instantiates_successfully(self):
        class Full(BaseMemory):
            def ingest(self, message, session):
                pass

            def retrieve(self, query):
                return ""

            def reset(self):
                pass

        assert isinstance(Full(), BaseMemory)


# ---------------------------------------------------------------------------
# TestNullMemoryBasic
# ---------------------------------------------------------------------------


class TestNullMemoryBasic:
    """NullMemory basic construction and interface compliance."""

    def test_is_instance_of_base_memory(self):
        assert isinstance(NullMemory(), BaseMemory)

    def test_retrieve_returns_empty_string(self, message, session):
        nm = NullMemory()
        assert nm.retrieve("anything") == ""

    def test_retrieve_empty_query_returns_empty_string(self):
        nm = NullMemory()
        assert nm.retrieve("") == ""

    def test_ingest_does_not_raise(self, message, session):
        nm = NullMemory()
        nm.ingest(message, session)  # should not raise

    def test_reset_does_not_raise(self):
        nm = NullMemory()
        nm.reset()  # should not raise


# ---------------------------------------------------------------------------
# TestNullMemoryStatelessness
# ---------------------------------------------------------------------------


class TestNullMemoryStatelessness:
    """NullMemory must remain stateless regardless of ingest calls."""

    def test_retrieve_after_ingest_still_empty(self, message, session):
        nm = NullMemory()
        nm.ingest(message, session)
        assert nm.retrieve("anything") == ""

    def test_retrieve_after_many_ingests_still_empty(self, session):
        nm = NullMemory()
        for i in range(10):
            nm.ingest(Message(role="user", content=f"msg {i}"), session)
        assert nm.retrieve("msg 5") == ""

    def test_retrieve_after_reset_still_empty(self, message, session):
        nm = NullMemory()
        nm.ingest(message, session)
        nm.reset()
        assert nm.retrieve("anything") == ""

    def test_reset_after_reset_does_not_raise(self):
        nm = NullMemory()
        nm.reset()
        nm.reset()

    def test_two_instances_are_independent(self, message, session):
        nm1 = NullMemory()
        nm2 = NullMemory()
        nm1.ingest(message, session)
        # nm2 should be unaffected
        assert nm2.retrieve("anything") == ""


# ---------------------------------------------------------------------------
# TestMemoryInterfaceContract
# ---------------------------------------------------------------------------


class TestMemoryInterfaceContract:
    """Verify the ingest → retrieve → reset contract via a minimal stub."""

    def _make_stub_memory(self) -> BaseMemory:
        """A minimal stateful memory that stores all ingested content."""

        class StubMemory(BaseMemory):
            def __init__(self):
                self._store: list[str] = []

            def ingest(self, message: Message, session: Session) -> None:
                self._store.append(message.content)

            def retrieve(self, query: str) -> str:
                return "\n".join(self._store)

            def reset(self) -> None:
                self._store.clear()

        return StubMemory()

    def test_ingest_then_retrieve_returns_content(self, message, session):
        mem = self._make_stub_memory()
        mem.ingest(message, session)
        result = mem.retrieve("query")
        assert "Hello, how are you?" in result

    def test_multiple_ingest_all_appear_in_retrieve(self, session):
        mem = self._make_stub_memory()
        mem.ingest(Message(role="user", content="first"), session)
        mem.ingest(Message(role="assistant", content="second"), session)
        result = mem.retrieve("anything")
        assert "first" in result
        assert "second" in result

    def test_reset_clears_all_ingested_content(self, message, session):
        mem = self._make_stub_memory()
        mem.ingest(message, session)
        mem.reset()
        assert mem.retrieve("query") == ""

    def test_ingest_after_reset_works_normally(self, session):
        mem = self._make_stub_memory()
        mem.ingest(Message(role="user", content="before reset"), session)
        mem.reset()
        mem.ingest(Message(role="user", content="after reset"), session)
        result = mem.retrieve("query")
        assert "after reset" in result
        assert "before reset" not in result
