"""Reproducibility helpers.

Every number printed in the notebooks, and every sanity-check value asserted
in a TODO, must be reproducible. That means one seed, set everywhere, once.

Call :func:`set_seed` in the first code cell of every notebook and at the top of
every training script.
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED: int = 42
"""The course-wide seed. Do not change it: the assert-based sanity checks in the
student notebooks are calibrated against it."""


def set_seed(seed: int = SEED, deterministic_torch: bool = True) -> int:
    """Seed Python, numpy and (if installed) PyTorch.

    Parameters
    ----------
    seed
        The seed to use. Defaults to the course-wide :data:`SEED`.
    deterministic_torch
        Also pin cuDNN/oneDNN into deterministic mode and disable benchmark
        autotuning. On CPU this costs essentially nothing and removes a source
        of run-to-run drift.

    Returns
    -------
    int
        The seed that was applied, so it can be logged.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is optional for the pure-numpy utilities
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker.

    PyTorch reseeds each worker process from the base seed, but numpy-based
    augmentations inside a worker do not inherit that. Pass this as
    ``DataLoader(..., worker_init_fn=worker_init_fn)`` whenever augmentations
    use numpy.
    """
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def cpu_threads(n: int | None = None) -> int:
    """Cap PyTorch's intra-op thread count.

    Left alone, PyTorch grabs every core. In a classroom that makes timings
    wildly machine-dependent and, on laptops with efficiency cores, often
    *slower*. The training scripts pin it so that timings mean something.

    Returns the thread count actually set.
    """
    try:
        import torch
    except ImportError:
        return 0

    if n is None:
        n = min(4, os.cpu_count() or 1)

    torch.set_num_threads(n)
    return n
