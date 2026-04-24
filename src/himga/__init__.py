from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

# Always load the project-level .env (repo root) first so env vars like
# OPENAI_BASE_URL, OPENAI_API_KEY, LLM_PROVIDER are available regardless
# of where the process was launched from (e.g. Jupyter started from ~).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)  # fallback: walk up from cwd

try:
    __version__ = version("himga")
except PackageNotFoundError:
    __version__ = "unknown"
