from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style


def plot_fit(
    star_tiles: list[np.ndarray],
    points: np.ndarray,
    errors: np.ndarray,
    slope: float,
    intercept: float,
    slope_err: float,
    det: str,
    filt1: str,
    filt2: str,
    filty: str,
):
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.cm.cool(np.linspace(0, 1, len(star_tiles)))

    for i, tile in enumerate(star_tiles):
        if tile.size:
            ax.scatter(tile[:, 0], tile[:, 1], marker="d", s=10, c=[cmap[i]], alpha=0.5)

    ax.errorbar(
        points[:, 0],
        points[:, 1],
        yerr=errors,
        color="k",
        fmt="h",
        markersize=6,
        capsize=4,
        zorder=3,
    )

    xv = np.linspace(points[:, 0].min(), points[:, 0].max(), 400)
    yv = slope * xv + intercept
    ax.plot(xv, yv, "k:", lw=1.2, zorder=2)

    ax.set_xlabel(f"[{det}] {filt1} - {filt2} (mag)", fontsize=15)
    ax.set_ylabel(f"{filty} (mag)", fontsize=15)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02,
        0.02,
        f"slope = {slope:.3f} +/- {slope_err:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )
    fig.tight_layout()
    return fig, ax
