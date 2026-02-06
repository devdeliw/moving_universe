from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .isochrones import Isochrones
from .io_utils.paths import tile_filename


@dataclass
class TileResult:
    star_tiles: np.ndarray
    center: Tuple[float, float]
    rot_matrix: np.ndarray


class Tiles:
    """Generate tiles orthogonal to the RC ridge using Fritz+11 slope."""

    def __init__(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        my: np.ndarray,
        filt1: str,
        filt2: str,
        filty: str,
        det: str,
        *,
        clr_range: Optional[Tuple[float, float]],
        n_tiles: int,
        iso_dir: Path,
    ):
        self.m1 = np.asarray(m1)
        self.m2 = np.asarray(m2)
        self.my = np.asarray(my)
        self.filt1 = filt1
        self.filt2 = filt2
        self.filty = filty
        self.det = det
        self.n_tiles = n_tiles
        self.clr_range = clr_range
        self.iso_dir = iso_dir

        self._center: Optional[Tuple[float, float]] = None

    def rc_data(self):
        x = self.m1 - self.m2
        y = self.my

        if self.clr_range:
            x_lo, x_hi = self.clr_range
        else:
            x_lo, x_hi = np.min(x), np.max(x)

        mask = (
            (x >= x_lo)
            & (x <= x_hi)
            & np.isfinite(self.m1)
            & np.isfinite(self.m2)
            & np.isfinite(self.my)
        )
        x_f, y_f = x[mask], y[mask]

        return {"x": x_f, "y": y_f, "finite_mask": mask}

    def rc_slope(self) -> float:
        return Isochrones(self.filt1, self.filt2, self.filty, self.iso_dir).reddening_slope()

    def rot_matrix(self) -> np.ndarray:
        theta = np.arctan(self.rc_slope())
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, s], [-s, c]])

    def rotate_rc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0, y0 = np.nanmedian(x), np.nanmedian(y)
        self._center = (float(x0), float(y0))
        xy = np.column_stack((x - x0, y - y0))
        return xy @ self.rot_matrix().T

    def unrotate_rc(self, xy: np.ndarray) -> np.ndarray:
        if self._center is None:
            raise RuntimeError("call rotate_rc first to define center")
        x0, y0 = self._center
        rinv = self.rot_matrix().T
        xy = (rinv @ xy.T).T
        xy[:, 0] += x0
        xy[:, 1] += y0
        return xy

    def tiles(self) -> TileResult:
        data = self.rc_data()
        x, y = data["x"], data["y"]
        rotated = self.rotate_rc(x, y)

        if rotated.size == 0:
            star_tiles = np.empty(self.n_tiles, dtype=object)
            for i in range(self.n_tiles):
                star_tiles[i] = np.empty((0, 2))
            return TileResult(star_tiles=star_tiles, center=self._center or (0.0, 0.0), rot_matrix=self.rot_matrix())

        x_rot = rotated[:, 0]
        lo, hi = np.nanmin(x_rot), np.nanmax(x_rot)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(lo), float(lo) + 1e-6

        edges = np.linspace(lo, hi, self.n_tiles + 1)
        idxs = np.digitize(x_rot, edges, right=False) - 1
        idxs = np.clip(idxs, 0, self.n_tiles - 1)

        star_tiles = np.empty(self.n_tiles, dtype=object)
        for i in range(self.n_tiles):
            star_tiles[i] = self.unrotate_rc(rotated[idxs == i])

        return TileResult(star_tiles=star_tiles, center=self._center or (0.0, 0.0), rot_matrix=self.rot_matrix())


def save_tiles(
    tiles: TileResult,
    out_dir: Path,
    det: str,
    filt1: str,
    filt2: str,
    filty: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / tile_filename(det, filt1, filt2, filty)
    with fname.open("wb") as f:
        pickle.dump(tiles.star_tiles, f)
    return fname
