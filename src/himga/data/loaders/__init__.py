from loguru import logger

from himga.logger import setup_logger
from himga.utils import get_dataset

from ..schema import Sample
from .locomo import load_locomo
from .longmemeval import load_longmemeval

setup_logger()


def load_dataset(
    name: str,
    *,
    limit: int | None = None,
    sample_ids: list[str] | frozenset[str] | None = None,
) -> list[Sample]:
    """Load a dataset by name, downloading if absent.

    Parameters
    ----------
    name : str
        Dataset name: ``"locomo"`` or ``"longmemeval"``.
    limit : int or None
        Stop after this many samples. Useful for quick dev runs.
        ``None`` (default) loads everything.
    sample_ids : list[str] or frozenset[str] or None
        If given, only load samples whose ID is in this collection.
        For LoCoMo the ID is ``sample_id``; for LongMemEval it is
        ``question_id``.

    Returns
    -------
    list[Sample]
        Parsed samples ready for evaluation.
    """
    path = get_dataset(name)
    ids = frozenset(sample_ids) if sample_ids is not None else None
    if name == "locomo":
        dataset = load_locomo(path, limit=limit, sample_ids=ids)
    elif name == "longmemeval":
        dataset = load_longmemeval(path, limit=limit, sample_ids=ids)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    logger.success(f"Dataset {name!r} loaded! ({len(dataset)} samples)")
    return dataset


__all__ = ["load_dataset", "load_locomo", "load_longmemeval"]
