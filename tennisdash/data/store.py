"""Frame persistence with a graceful format fallback.

Parquet is preferred (compact, typed, fast) but it needs pyarrow, which is not
always installable in restricted environments. Rather than hard-failing, fall
back to compressed pickle so the pipeline still runs everywhere.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401  (import is the availability probe)
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401  (availability probe)
            return True
        except ImportError:
            return False


def save_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Persist ``frame``, preferring parquet and falling back to pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _parquet_available():
        target = path.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
    else:
        target = path.with_suffix(".pkl.gz")
        frame.to_pickle(target, compression="gzip")
        log.info("pyarrow unavailable; wrote %s instead of parquet", target.name)
    # Remove a stale copy in the other format so loads are unambiguous.
    for other in (path.with_suffix(".parquet"), path.with_suffix(".pkl.gz")):
        if other != target and other.exists():
            other.unlink()
    return target


def load_frame(path: Path) -> pd.DataFrame:
    """Read a frame written by :func:`save_frame`, whichever format was used."""
    path = Path(path)
    parquet, pickled = path.with_suffix(".parquet"), path.with_suffix(".pkl.gz")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if pickled.exists():
        return pd.read_pickle(pickled, compression="gzip")
    raise FileNotFoundError(
        f"No stored frame at {parquet} or {pickled}. Run `make data` first."
    )


def frame_exists(path: Path) -> bool:
    path = Path(path)
    return path.with_suffix(".parquet").exists() or path.with_suffix(".pkl.gz").exists()
