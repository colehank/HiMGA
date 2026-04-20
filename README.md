# HiMGA

[![Tests](https://img.shields.io/github/actions/workflow/status/colehank/HiMGA/tests.yml?branch=main)](https://github.com/colehank/HiMGA/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/VneuroTK?color=blue)](https://pypi.org/project/VneuroTK/)  
Hierarchical Multi-Graph Architecture for conversational memory

## for contributors
run once after cloning:

```bash
uv sync --dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

**CI checks** run automatically on every push to `main`:

| Check | Command |
|---|---|
| Lint + format | `ruff check . && ruff format --check .` |
| Tests | `pytest tests/ -v` |

**Releasing a new version** — version is derived from the git tag, no files need editing:

```bash
git tag vx.x.x
git push origin vx.x.x   # triggers build + publish to PyPI
```