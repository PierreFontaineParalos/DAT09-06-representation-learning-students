#!/usr/bin/env python
"""Deterministic build of the course datasets.

Run once, after installing the requirements and before session 2::

    python data/prepare_data.py --all

Everything in ``data/processed/`` is produced here, and only here. No notebook
ever reads from ``raw/``: the contract is ``processed/``. Same seed, same
subset, same files, on any machine.

Right now that means Fashion-MNIST, which sessions 1 to 3 use: 70,000 greyscale
28x28 photographs of clothing, about 30 MB, downloaded by torchvision with no
account and rebuilt in a few seconds. Later sessions use larger data that is
prepared once and added to this repository, so there will be nothing more for
you to build.

Useful flags::

    python data/prepare_data.py --check    # report what is present, build nothing
    python data/prepare_data.py --force    # rebuild even if the files exist

Why this script imports almost nothing at the top
-------------------------------------------------
It is the first thing you run after ``pip install``, so a broken or partial
environment has to produce a readable sentence rather than a traceback from
inside a library. Every heavy import therefore happens inside the step that
needs it, behind a message that says which package is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (deliberately spelled out rather than imported from common/, so that
# this script runs before the course package is known to work)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"

FMNIST_OUTPUTS = ("fmnist_train.npz", "fmnist_test.npz")
FMNIST_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


class BuildBlocked(RuntimeError):
    """A build cannot proceed, for a reason you can act on."""


def say(message: str = "") -> None:
    print(message, flush=True)


def need(package: str):
    """Import ``package`` or explain, in one sentence, how to get it."""
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise BuildBlocked(
            f"This step needs `{package}`, which is not installed.\n"
            f"  Fix with:  pip install -r requirements.txt"
        ) from exc


# --------------------------------------------------------------------------
# Fashion-MNIST
# --------------------------------------------------------------------------


def build_fmnist(force: bool = False) -> str:
    say()
    say("Fashion-MNIST  (sessions 1-3)")
    say("-----------------------------")

    if not force and all((PROCESSED / name).exists() for name in FMNIST_OUTPUTS):
        say("  already built, skipping  (--force to rebuild)")
        return "skipped"

    np = need("numpy")
    need("torch")  # torchvision imports it anyway; fail with the clearer name first
    torchvision_datasets = need("torchvision.datasets")

    say(f"  downloading into {RAW.relative_to(REPO_ROOT).as_posix()}/ (about 30 MB, once)")

    for split, is_train in (("train", True), ("test", False)):
        dataset = torchvision_datasets.FashionMNIST(
            root=str(RAW), train=is_train, download=True
        )

        # .data is a uint8 tensor (N, 28, 28); .targets is an int64 tensor (N,).
        # The on-disk order of the IDX files is fixed, so nothing is shuffled
        # here and every machine gets byte-identical output.
        images = dataset.data.numpy().astype(np.uint8)
        labels = dataset.targets.numpy().astype(np.int64)

        expected = 60_000 if is_train else 10_000
        if images.shape != (expected, 28, 28):
            raise BuildBlocked(
                f"Unexpected Fashion-MNIST {split} shape {images.shape}; "
                f"expected ({expected}, 28, 28). Delete data/raw/ and retry."
            )
        if labels.shape != (expected,):
            raise BuildBlocked(f"Unexpected label shape {labels.shape} for {split}.")

        np.savez_compressed(
            PROCESSED / f"fmnist_{split}.npz",
            images=images,
            labels=labels,
            classes=np.array(FMNIST_CLASSES),
        )

    for name in FMNIST_OUTPUTS:
        size = (PROCESSED / name).stat().st_size / 1e6
        say(f"  wrote  {name:<24s} {size:>8.1f} MB")

    return "built"


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------

INVENTORY = (
    ("fmnist_train.npz", "Fashion-MNIST train", "sessions 1-3"),
    ("fmnist_test.npz", "Fashion-MNIST test", "sessions 1-3"),
)


def check() -> int:
    say()
    say("What is in data/processed/")
    say("--------------------------")
    say(f"  {'file':<26s} {'size':>10s}   needed for")
    say("  " + "-" * 62)

    missing = 0
    for name, description, when in INVENTORY:
        path = PROCESSED / name
        if path.exists():
            size = f"{path.stat().st_size / 1e6:.1f} MB"
        else:
            size = "missing"
            missing += 1
        say(f"  {name:<26s} {size:>10s}   {when} - {description}")

    say()
    if missing:
        say(f"  {missing} missing. Build with:  python data/prepare_data.py --all")
        return 1

    say("  Everything is in place.")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python data/prepare_data.py",
        description="Build the course datasets, deterministically.",
        epilog="`--all` is the only command you need.",
    )
    parser.add_argument("--all", action="store_true", help="build everything that is missing")
    parser.add_argument("--fmnist", action="store_true", help="Fashion-MNIST (sessions 1-3)")
    parser.add_argument("--check", action="store_true", help="report what is present, build nothing")
    parser.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    args = parser.parse_args(argv)

    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if args.check:
        return check()

    if not (args.all or args.fmnist):
        say("Nothing selected; running --all. (--help for the options.)")

    try:
        build_fmnist(force=args.force)
    except BuildBlocked as exc:
        say()
        say(str(exc))
        return 1

    say()
    say("  Done. Next: jupyter lab, then run environment-check.ipynb.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nInterrupted. Re-run to resume; finished steps are skipped.")
        sys.exit(130)
