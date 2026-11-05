"""Canonical paths for the course repository.

Every notebook lives in a ``session-N-*/`` folder. Rather than have each of
them guess at ``../data``, they all import from here.

Usage from a notebook::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))       # make `common` importable

    from common.paths import PROCESSED, CHECKPOINTS, PLOTS
"""

from __future__ import annotations

from pathlib import Path

# This file is <repo>/common/paths.py, so the repository root is two levels up.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA: Path = REPO_ROOT / "data"
RAW: Path = DATA / "raw"
PROCESSED: Path = DATA / "processed"

CHECKPOINTS: Path = REPO_ROOT / "checkpoints"
SCRIPTS: Path = REPO_ROOT / "scripts"

ASSETS: Path = REPO_ROOT / "assets"
PLOTS: Path = ASSETS / "plots"
SCHEMATICS: Path = ASSETS / "schematics"

SESSIONS: dict[int, Path] = {
    1: REPO_ROOT / "session-1-foundations-autoencoders",
    2: REPO_ROOT / "session-2-vae",
    3: REPO_ROOT / "session-3-contrastive",
    4: REPO_ROOT / "session-4-multimodal",
    5: REPO_ROOT / "session-5-embedding-anatomy",
    6: REPO_ROOT / "session-6-production",
}


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist yet."""
    for directory in (RAW, PROCESSED, CHECKPOINTS, PLOTS, SCHEMATICS):
        directory.mkdir(parents=True, exist_ok=True)


def require(path: Path, hint: str = "python data/prepare_data.py --all") -> Path:
    """Return ``path``, or fail with an actionable message.

    Missing data is the single most common way a lab session stalls, so the
    error says what to run rather than just which file is absent.
    """
    if not path.exists():
        relative = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        raise FileNotFoundError(
            f"Missing required file: {relative}\n"
            f"Build it with:  {hint}"
        )
    return path
