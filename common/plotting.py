"""Figure styling and helpers.

One palette for the whole course, so that every figure you produce looks like
the ones projected in the lecture. Figures are written to
``assets/plots/{fig-id}.png``.

Typical use in a notebook::

    from common.plotting import use_course_style, save_figure
    use_course_style()

    fig, ax = plt.subplots()
    ...
    save_figure(fig, "fig-1-3")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .paths import PLOTS

__all__ = [
    "PALETTE",
    "CLASS_COLORS",
    "use_course_style",
    "save_figure",
    "image_grid",
    "scatter_latent",
]


# Mirrors the custom properties in slides/theme.css.
PALETTE = {
    "ink": "#16202E",
    "ink_soft": "#33415A",
    "paper": "#FCFBF9",
    "paper_alt": "#F2EFE9",
    "accent": "#E2643B",
    "accent_dim": "#F6D9CE",
    "teal": "#2F7A80",
    "teal_dim": "#D6E6E6",
    "muted": "#7A8496",
    "rule": "#E0DCD4",
}

# A qualitative sequence for the ten Fashion-MNIST classes and for the product
# categories in sessions 4-6. Chosen to stay distinguishable in greyscale print
# and to remain legible when projected.
CLASS_COLORS = [
    "#E2643B",  # accent orange
    "#2F7A80",  # teal
    "#4C6BAA",  # indigo
    "#C9A227",  # ochre
    "#8E5BA6",  # violet
    "#4E8B57",  # green
    "#C0435E",  # crimson
    "#7A8496",  # slate
    "#B5732E",  # amber-brown
    "#3D9CB0",  # cyan
]


def use_course_style() -> None:
    """Apply the course matplotlib style. Call once per notebook."""
    mpl.rcParams.update(
        {
            # --- figure ---
            "figure.figsize": (7.2, 4.4),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "figure.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            # --- axes ---
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["rule"],
            "axes.labelcolor": PALETTE["ink_soft"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.prop_cycle": mpl.cycler(color=CLASS_COLORS),
            # --- grid ---
            "grid.color": PALETTE["rule"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            # --- ticks ---
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # --- text ---
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 11,
            "text.color": PALETTE["ink"],
            # --- legend ---
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            # --- lines ---
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "image.cmap": "gray",
        }
    )


def save_figure(fig, fig_id: str, *, directory: Path | None = None) -> Path:
    """Write a figure to ``assets/plots/{fig_id}.png``.

    ``fig_id`` must match the ``id:`` of the corresponding ```figure``` block in
    the session's ``lecture-notes.md``, because that is the path the slide deck
    embeds.
    """
    directory = directory or PLOTS
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{fig_id}.png"
    fig.savefig(path)
    print(f"saved {path.name}")
    return path


def image_grid(
    images: np.ndarray,
    *,
    n_cols: int = 10,
    titles: list[str] | None = None,
    title: str | None = None,
    cmap: str = "gray",
    scale: float = 1.0,
):
    """Lay out a batch of images in a grid.

    ``images`` may be ``(n, h, w)`` or ``(n, 1, h, w)``; values in [0, 1] or
    [0, 255]. Returns the figure so the caller can pass it to
    :func:`save_figure`.
    """
    images = np.asarray(images)
    if images.ndim == 4:
        images = images.squeeze(1)

    n = len(images)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 0.95 * scale, n_rows * 1.0 * scale),
        squeeze=False,
    )

    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if i < n:
            ax.imshow(images[i], cmap=cmap, interpolation="nearest")
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=8, color=PALETTE["ink_soft"], pad=3)
        else:
            ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="semibold", y=1.0)

    fig.tight_layout()
    return fig


def scatter_latent(
    z: np.ndarray,
    y: np.ndarray,
    *,
    class_names: list[str] | None = None,
    title: str | None = None,
    ax=None,
    s: float = 4.0,
    alpha: float = 0.6,
    max_points: int = 8000,
    seed: int = 42,
):
    """Scatter a 2-D representation, coloured by class.

    Only ever fed two dimensions: either a latent space that is genuinely 2-D,
    or the output of PCA/UMAP/t-SNE applied to a wider one.

    Subsamples above ``max_points``: beyond that the plot is a solid blob and
    the render is slow, and neither helps anyone see anything.
    """
    z = np.asarray(z)
    y = np.asarray(y)

    if z.shape[1] != 2:
        raise ValueError(f"scatter_latent needs 2 dimensions, got {z.shape[1]}")

    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z), size=max_points, replace=False)
        z, y = z[idx], y[idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.4))

    classes = np.unique(y)
    for i, c in enumerate(classes):
        mask = y == c
        label = class_names[int(c)] if class_names is not None else str(c)
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=s,
            alpha=alpha,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            label=label,
            linewidths=0,
        )

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    if title:
        ax.set_title(title)

    ax.legend(
        markerscale=3,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )
    return ax
