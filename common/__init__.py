"""Shared utilities for DAT09-06 — Representation Learning.

Imported by every notebook and every training script in the course. Nothing
session-specific lives here: this package is the part that stays constant while
the representations change around it.

    common.paths       where things are on disk
    common.seeding     one seed, set everywhere
    common.evaluation  how we judge a representation (the methodological spine)
    common.plotting    figure styling shared with the slide theme

From a notebook inside a ``session-N-*/`` folder::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))

    from common import set_seed, evaluate_representation, use_course_style
"""

from .evaluation import (
    RepresentationReport,
    clustering_metrics,
    compare_reports,
    evaluate_representation,
    knn_probe,
    linear_probe,
    silhouette,
)
from .paths import (
    CHECKPOINTS,
    PLOTS,
    PROCESSED,
    RAW,
    REPO_ROOT,
    SCHEMATICS,
    ensure_dirs,
    require,
)
from .plotting import (
    CLASS_COLORS,
    PALETTE,
    image_grid,
    save_figure,
    scatter_latent,
    use_course_style,
)
from .seeding import SEED, cpu_threads, set_seed, worker_init_fn

__all__ = [
    # paths
    "REPO_ROOT",
    "RAW",
    "PROCESSED",
    "CHECKPOINTS",
    "PLOTS",
    "SCHEMATICS",
    "ensure_dirs",
    "require",
    # seeding
    "SEED",
    "set_seed",
    "worker_init_fn",
    "cpu_threads",
    # evaluation
    "RepresentationReport",
    "evaluate_representation",
    "linear_probe",
    "knn_probe",
    "clustering_metrics",
    "silhouette",
    "compare_reports",
    # plotting
    "PALETTE",
    "CLASS_COLORS",
    "use_course_style",
    "save_figure",
    "image_grid",
    "scatter_latent",
]
