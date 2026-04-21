# HiMGA
[![Tests](https://github.com/colehank/HiMGA/actions/workflows/tests.yml/badge.svg)](https://github.com/colehank/HiMGA/actions) [![PyPI](https://img.shields.io/pypi/v/VneuroTK?color=blue)](https://pypi.org/project/VneuroTK/)  

HiMGA is a Hierarchical Multi-Graph Architecture for conversational memory.

## installation
### for user
**not support yet, himga still in dev.**
```bash
pip install higma
```

### for contributors
Project uses `uv`([to install](https://docs.astral.sh/uv/#installation)) for dependency management.

```bash
git clone https://github.com/colehank/HiMGA
cd HiMGA
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
uv pip install -e .
```
Releasing to Pypi:

```bash
git tag vx.x.x
git push origin vx.x.x   # triggers build + publish to PyPI
```
## Configuration
Rename `.env_example` to `.env` and fill in the values.

## resolve benchmarks
We are using `locomo` and `longmemeval` now.  
Set `.env` file with `DATASETS_ROOT` (default to `.cache/datasets`)pointing to the directory 
where your datasets exsists or where they need to be downloaded.

to resolve them, simply:
```python
from himga.utils import get_dataset
## get_dataset() -> download if dataset not found locally, and return local path
locomo = get_dataset("locomo") 
longmem = get_dataset("longmemeval")
```