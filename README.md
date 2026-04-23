# HiMGA
[![Tests](https://github.com/colehank/HiMGA/actions/workflows/tests.yml/badge.svg)](https://github.com/colehank/HiMGA/actions) [![PyPI](https://img.shields.io/pypi/v/himga?color=blue)](https://pypi.org/project/himga/)

HiMGA is a Hierarchical Multi-Graph Architecture for conversational memory.

## Installation

### User

Core install — includes exact_match, F1, ROUGE, BLEU, METEOR:

```bash
pip install himga
```

Full eval metrics — adds BERTScore and Sentence-BERT:

```bash
pip install "himga[eval]"
```

GPU acceleration (BERTScore / SBERT auto-detect CUDA, no code changes needed):

```bash
# Install CUDA-compatible PyTorch first, then eval extras
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "himga[eval]"
```

### Contributor

Project uses [`uv`](https://docs.astral.sh/uv/#installation) for dependency management.

```bash
git clone https://github.com/colehank/HiMGA
cd HiMGA
uv sync --dev                  # installs all deps including eval extras
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

Copy `.env_example` to `.env` and fill in the required values.

**Running tests:**

```bash
uv run pytest tests/            # fast tests only (default)
uv run pytest tests/ --run-slow # include BERTScore / SBERT model tests
uv run pytest tests/ --run-integration  # include real API calls (requires .env)
```

**Releasing to PyPI:**

```bash
git tag vx.x.x
git push origin vx.x.x          # triggers build + publish to PyPI
```

## Benchmarks

HiMGA is evaluated on `locomo` and `longmemeval`.
Set `DATASETS_ROOT` in `.env` (defaults to `.cache/datasets`) to the directory where datasets
exist or should be downloaded.

```python
from himga.utils import get_dataset

locomo   = get_dataset("locomo")       # downloads if not found locally, returns local path
longmem  = get_dataset("longmemeval")
```