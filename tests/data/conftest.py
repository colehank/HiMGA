from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def locomo_path() -> Path:
    return FIXTURES / "locomo"


@pytest.fixture
def longmemeval_path() -> Path:
    return FIXTURES / "longmemeval"
