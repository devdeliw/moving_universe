from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import apply_style


def plot_annuli(
    annuli: Iterable[Tuple[int, pd.DataFrame]],
    sag_a_reference: Tuple[float, float],
):
    apply_style()
    annuli = list(annuli)
    n = len(annuli)
    cmap = plt.get_cmap("viridis")(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for idx, (annulus_index, catalog) in enumerate(annuli):
        x = catalog["x"]
        y = catalog["y"]
        ax.scatter(x, y, s=10, color=cmap[idx], label=f"Annulus {annulus_index} | {len(x)} stars")

    ax.scatter(sag_a_reference[0], sag_a_reference[1], marker="*", s=200, c="yellow", edgecolor="k")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.set_xlabel("x [pixel pos]", fontsize=15)
    ax.set_ylabel("y [pixel pos]", fontsize=15)
    fig.tight_layout()
    return fig, ax
