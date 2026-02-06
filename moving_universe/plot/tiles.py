from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_style


def plot_tiles(star_tiles: Iterable[np.ndarray], det: str, filt1: str, filt2: str, filty: str):
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    star_tiles = list(star_tiles)
    cmap = plt.cm.cool(np.linspace(0, 1, len(star_tiles)))

    for idx, stars in enumerate(star_tiles):
        if stars.size == 0:
            ax.scatter([], [], c=[cmap[idx]], marker="d", label=f"tile {idx:<2} | 0 stars")
            continue
        x, y = stars.T
        ax.scatter(-x, y, c=[cmap[idx]], s=10, marker="d", label=f"tile {idx:<2} | {len(x)} stars")

    ax.set_xlabel(f"[{det}] {filt1} - {filt2} (mag)", fontsize=15)
    ax.set_ylabel(f"{filty} (mag)", fontsize=15)
    ax.invert_yaxis()
    ax.axis("equal")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return fig, ax
