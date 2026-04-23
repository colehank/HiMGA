"""Session-scoped fixtures and CLI options shared across all test modules."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (loads large ML models / downloads data).",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.integration (requires real API keys / external services).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_slow = pytest.mark.skip(reason="pass --run-slow to include slow tests")
    skip_integration = pytest.mark.skip(
        reason="pass --run-integration to include integration tests"
    )
    run_slow = config.getoption("--run-slow")
    run_integration = config.getoption("--run-integration")

    for item in items:
        if not run_slow and item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
        if not run_integration and item.get_closest_marker("integration"):
            item.add_marker(skip_integration)


@pytest.fixture(scope="session", autouse=True)
def ensure_nltk_data() -> None:
    """Download required NLTK corpora once per test session.

    Downloads ``punkt_tab`` and ``wordnet`` if not already present.
    Runs automatically for every session so individual tests need not
    call ``_ensure_nltk_data()`` explicitly.
    """
    import nltk

    for corpus, finder_path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(finder_path)
        except LookupError:
            nltk.download(corpus, quiet=True)
