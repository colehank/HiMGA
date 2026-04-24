"""Session-scoped fixtures and CLI options shared across all test modules."""

from __future__ import annotations

from unittest.mock import patch

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
def _mock_nltk_download() -> None:
    """Prevent nltk.download() from hitting the network during tests.

    NLTK data (punkt_tab, wordnet) is pre-installed in CI via the workflow
    setup step and locally via prior developer setup.  Calling nltk.download()
    inside tests only wastes time waiting on a proxy timeout.
    """
    with patch("himga.eval.metrics._nltk_ready", True):
        yield
