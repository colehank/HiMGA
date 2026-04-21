"""Dataset utility: resolve local path, downloading if absent."""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger

load_dotenv()

ROOT = Path(os.getenv("DATASETS_ROOT", ".cache/datasets"))
GH_HEADERS = {"Authorization": f"token {t}"} if (t := os.getenv("GITHUB_TOKEN")) else {}
PROXIES = {"https": p} if (p := os.getenv("HTTPS_PROXY")) else None


def _gh_files(owner: str, repo: str, path: str, branch: str = "main") -> list[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    resp = requests.get(url, headers=GH_HEADERS, proxies=PROXIES, timeout=30)
    resp.raise_for_status()
    files = []
    for item in resp.json():
        if item["type"] == "file":
            files.append(item)
        elif item["type"] == "dir":
            files.extend(_gh_files(owner, repo, item["path"], branch))
    return files


def _gh_download(item: dict, out_dir: Path, strip_prefix: str = "") -> None:
    rel = Path(item["path"])
    if strip_prefix:
        rel = rel.relative_to(strip_prefix)
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(item["download_url"], headers=GH_HEADERS, proxies=PROXIES, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.debug(f"  {rel}")


def _fetch_locomo(out_dir: Path) -> None:
    files = _gh_files("snap-research", "locomo", "data")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(partial(_gh_download, out_dir=out_dir, strip_prefix="data"), files))


def _fetch_longmemeval(out_dir: Path) -> None:
    snapshot_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        repo_type="dataset",
        local_dir=out_dir,
        proxies=PROXIES,
    )


DATASETS: dict[str, callable] = {
    "locomo": _fetch_locomo,
    "longmemeval": _fetch_longmemeval,
}


def get_dataset(name: str) -> Path:
    """Return local path to *name* dataset, downloading it first if needed."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASETS)}")

    path = ROOT / name
    if path.exists():
        logger.info(f"[{name}] found at {path}")
        return path

    logger.info(f"[{name}] not found — downloading to {path}")
    path.mkdir(parents=True, exist_ok=True)
    DATASETS[name](path)
    logger.success(f"[{name}] ready at {path}")
    return path
