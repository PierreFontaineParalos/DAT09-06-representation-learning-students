"""How we judge a representation.

This module is the methodological spine of the course. It is introduced in
session 1 and imported unchanged by sessions 2, 3, 4, 5 and 6, so that every
comparison we ever make — autoencoder against PCA, VAE against autoencoder,
SimCLR against both, whitened against raw — is measured the same way.

The mental loop it encodes is:

    train a representation  ->  freeze it  ->  probe it  ->  compare

Four complementary questions, because no single number captures "good":

``probe_accuracy``
    Train a *linear* classifier on frozen features. If a linear model can read
    the class off the representation, the representation has made the class
    linearly accessible. This is the headline number.

``knn_accuracy``
    Are semantically similar items actually *close*? A linear probe can succeed
    on a space where neighbourhoods are meaningless; k-NN cannot. Retrieval
    systems care about this one.

``nmi`` / ``ari``
    Run k-means without labels, then compare the clusters to the true classes.
    Measures whether the class structure is there *for the taking*, with no
    supervision at all.

``silhouette``
    Geometry, not accuracy: how tight are the true classes relative to the gaps
    between them. Can be poor even when probe accuracy is high — that gap is
    itself informative, and session 5 is largely about it.

A note on labels. Every model in this course is trained **without labels**.
Labels appear only in this file. That separation is the whole point: the label
is the measuring instrument, never the training signal.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .seeding import SEED

__all__ = [
    "RepresentationReport",
    "linear_probe",
    "knn_probe",
    "clustering_metrics",
    "silhouette",
    "evaluate_representation",
    "compare_reports",
]


# ---------------------------------------------------------------------------
# Defaults
#
# These caps exist for one reason: this course runs on laptop CPUs. A logistic
# regression on 60,000 x 64 takes long enough to break the rhythm of a lab, and
# silhouette is O(n^2) in memory. Capping keeps every evaluation call under a
# couple of seconds. The caps are applied with a fixed seed, so comparisons
# between representations remain apples-to-apples.
# ---------------------------------------------------------------------------

MAX_PROBE_TRAIN = 10_000
MAX_SILHOUETTE = 5_000
KNN_K = 10


@dataclass
class RepresentationReport:
    """The four-number verdict on one representation."""

    name: str
    dim: int
    probe_accuracy: float
    knn_accuracy: float
    nmi: float
    ari: float
    silhouette: float
    n_train: int = 0
    n_test: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.name:<22s} d={self.dim:<4d} "
            f"probe={self.probe_accuracy:.3f}  "
            f"knn={self.knn_accuracy:.3f}  "
            f"NMI={self.nmi:.3f}  "
            f"ARI={self.ari:.3f}  "
            f"sil={self.silhouette:+.3f}"
        )


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def _subsample(
    z: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a fixed random subset of at most ``n`` points."""
    if len(z) <= n:
        return z, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), size=n, replace=False)
    return z[idx], y[idx]


def linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    max_train: int = MAX_PROBE_TRAIN,
    seed: int = SEED,
) -> float:
    """Test accuracy of a multinomial logistic regression on frozen features.

    Features are standardised first. This matters more than it looks: latent
    codes from different models live on wildly different scales, and an
    unstandardised probe would partly be measuring scale rather than content.
    """
    z_train, y_train = _subsample(z_train, y_train, max_train, seed)

    scaler = StandardScaler().fit(z_train)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(scaler.transform(z_train), y_train)

    return float(clf.score(scaler.transform(z_test), y_test))


def knn_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int = KNN_K,
    metric: str = "cosine",
    max_train: int = MAX_PROBE_TRAIN,
    seed: int = SEED,
) -> float:
    """Test accuracy of a k-nearest-neighbour classifier on frozen features.

    Cosine by default, because that is what a vector store will use.
    """
    z_train, y_train = _subsample(z_train, y_train, max_train, seed)

    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    clf.fit(z_train, y_train)

    return float(clf.score(z_test, y_test))


def clustering_metrics(
    z: np.ndarray,
    y: np.ndarray,
    *,
    n_clusters: int | None = None,
    seed: int = SEED,
) -> tuple[float, float]:
    """Run k-means, then score the clusters against the true labels.

    Returns ``(nmi, ari)``. Both are 0 for a random partition and 1 for a
    perfect match; NMI is information-theoretic, ARI is pair-counting and is the
    stricter of the two.

    Note that k-means is given the *number* of classes, not the classes. That
    is the only supervision involved.
    """
    if n_clusters is None:
        n_clusters = len(np.unique(y))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    assignments = kmeans.fit_predict(z)

    return (
        float(normalized_mutual_info_score(y, assignments)),
        float(adjusted_rand_score(y, assignments)),
    )


def silhouette(
    z: np.ndarray,
    y: np.ndarray,
    *,
    metric: str = "euclidean",
    max_samples: int = MAX_SILHOUETTE,
    seed: int = SEED,
) -> float:
    """Mean silhouette coefficient of the *true* classes in this space.

    In [-1, 1]. Positive means classes sit in tight, separated blobs; near zero
    means they overlap; negative means points are typically closer to another
    class than to their own.

    This is a statement about geometry, and it is deliberately not the same
    question as "can a classifier separate them".
    """
    z, y = _subsample(z, y, max_samples, seed)
    return float(silhouette_score(z, y, metric=metric, random_state=seed))


# ---------------------------------------------------------------------------
# The one function every notebook calls
# ---------------------------------------------------------------------------


def evaluate_representation(
    z: np.ndarray,
    y: np.ndarray,
    z_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    name: str = "representation",
    test_size: float = 0.3,
    seed: int = SEED,
    verbose: bool = True,
) -> RepresentationReport:
    """Score one representation on all four axes.

    Two calling conventions:

    >>> evaluate_representation(z, y)                    # internal split
    >>> evaluate_representation(z_tr, y_tr, z_te, y_te)  # explicit split

    The first is convenient; the second is what you want when you already have a
    canonical train/test split and are comparing several representations of the
    *same* items.

    Clustering and silhouette are always computed on the held-out portion, so
    every number in the report refers to the same set of points.

    Returns
    -------
    RepresentationReport
        Printable, and convertible to a dict for building comparison tables.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y)

    if z.ndim != 2:
        raise ValueError(f"expected z with shape (n, d), got {z.shape}")
    if len(z) != len(y):
        raise ValueError(f"z has {len(z)} rows but y has {len(y)} labels")

    if z_test is None:
        z_train, z_eval, y_train, y_eval = train_test_split(
            z, y, test_size=test_size, random_state=seed, stratify=y
        )
    else:
        if y_test is None:
            raise ValueError("z_test was given without y_test")
        z_train, y_train = z, y
        z_eval = np.asarray(z_test, dtype=np.float64)
        y_eval = np.asarray(y_test)

    probe_acc = linear_probe(z_train, y_train, z_eval, y_eval, seed=seed)
    knn_acc = knn_probe(z_train, y_train, z_eval, y_eval, seed=seed)
    nmi, ari = clustering_metrics(z_eval, y_eval, seed=seed)
    sil = silhouette(z_eval, y_eval, seed=seed)

    report = RepresentationReport(
        name=name,
        dim=z.shape[1],
        probe_accuracy=probe_acc,
        knn_accuracy=knn_acc,
        nmi=nmi,
        ari=ari,
        silhouette=sil,
        n_train=len(z_train),
        n_test=len(z_eval),
    )

    if verbose:
        print(report)

    return report


def compare_reports(reports, sort_by: str | None = "probe_accuracy"):
    """Turn a list of reports into a comparison table.

    >>> compare_reports([raw, pca64, ae64])

    Returns a pandas DataFrame, which is what the notebooks display and what
    session 6 writes into its A/B comparison.
    """
    import pandas as pd

    if isinstance(reports, dict):
        reports = [
            r if isinstance(r, RepresentationReport) else r for _, r in reports.items()
        ]

    rows = [r.to_dict() for r in reports]
    df = pd.DataFrame(rows).drop(columns=["extra"], errors="ignore")
    df = df.set_index("name")

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    return df.round(4)
