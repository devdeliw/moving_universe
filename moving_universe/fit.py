from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .compound_fit import CompoundTileFit
from .isochrones import Isochrones
from .io_utils.paths import tile_filename


@dataclass
class FitSummary:
    slope: float
    intercept: float
    slope_err: float
    n_tiles: int
    points: np.ndarray
    errors: np.ndarray
    fracs: np.ndarray
    amps: np.ndarray
    sigmas: np.ndarray


class RunCurveFit:
    def __init__(
        self,
        *,
        filt1: str,
        filt2: str,
        filty: str,
        det: str,
        tiles_dir: Path,
        iso_dir: Path,
        max_y_err: float = 0.20,
        bins_per_tile: int = 50,
    ):
        self.det = det
        self.filt1 = filt1
        self.filt2 = filt2
        self.filty = filty
        self.max_y_err = max_y_err
        self.bins_per_tile = bins_per_tile
        self.tiles_dir = tiles_dir

        self.star_tiles = self._load_tiles()

        pred_slope = Isochrones(self.filt1, self.filt2, self.filty, iso_dir).reddening_slope()
        theta = np.arctan(pred_slope)
        c, s = np.cos(theta), np.sin(theta)
        self._R = np.array([[c, s], [-s, c]])
        self._Rt = self._R.T

        all_xy = np.vstack([t for t in self.star_tiles if t.size])
        self._center = (np.median(all_xy[:, 0]), np.median(all_xy[:, 1]))

    def _load_tiles(self):
        fname = tile_filename(self.det, self.filt1, self.filt2, self.filty)
        with (self.tiles_dir / fname).open("rb") as f:
            return pickle.load(f)

    def _rotate_orig_to_rc(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self._center) @ self._Rt

    def _rotate_rc_to_orig(self, xy_rot: np.ndarray) -> np.ndarray:
        return (xy_rot @ self._R) + self._center

    def _fit_tile(self, stars: np.ndarray):
        xy_rot = self._rotate_orig_to_rc(stars)
        y_rot = xy_rot[:, 1]

        if y_rot.size < 10:
            return None

        fitter = CompoundTileFit(
            y_rot,
            bins=self.bins_per_tile,
            peak_frac=0.60,
            guard_frac=0.10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = fitter.fit()

        wL, wR = res.window
        wW = max(wR - wL, 1e-6)
        if (not res.mu_in_window) or (abs(res.mu - res.mode) > 0.8 * wW):
            mu = res.mode
        else:
            mu = res.mu

        n_eff = max(int(res.frac_rc * res.n), 1)
        mu_err_est = res.mu_err if (np.isfinite(res.mu_err) and res.mu_err > 0) else 0.0
        mu_err = max(
            0.5 * res.sig / np.sqrt(n_eff),
            0.3 * res.sig / np.sqrt(n_eff),
            mu_err_est,
        )

        x_med = float(np.median(xy_rot[:, 0]))
        pt_rot = np.array([x_med, mu])
        pt_orig = self._rotate_rc_to_orig(pt_rot[None, :])[0]
        y_err_orig = float(abs((np.array([[0.0, mu_err]]) @ self._R)[0, 1]))

        return pt_orig, y_err_orig, res.frac_rc, res.amp, res.sig

    def run(self) -> FitSummary:
        points: List[np.ndarray] = []
        errors: List[float] = []
        fracs: List[float] = []
        amps: List[float] = []
        sigmas: List[float] = []

        for tile in self.star_tiles:
            if tile.size < 100:
                continue
            out = self._fit_tile(tile)
            if out is None:
                continue
            pt, y_err, f_rc, amp, sig = out
            if (np.isfinite(y_err) and (y_err <= self.max_y_err)) and (f_rc > 1e-2):
                points.append(pt)
                errors.append(y_err)
                fracs.append(f_rc)
                amps.append(amp)
                sigmas.append(sig)

        if not points:
            raise RuntimeError("no usable tiles after QC")

        pts = np.vstack(points)
        xs, ys = pts[:, 0], pts[:, 1]
        if self.filt2 == "F405N" and len(xs) < 3:
            raise RuntimeError("insufficient usable tiles after QC for F405N")

        area = np.sqrt(2.0 * np.pi) * np.asarray(sigmas) * np.asarray(amps)
        w = (area / np.maximum(area.max(), 1e-12)) * (np.asarray(fracs) / (np.asarray(errors) ** 2))

        (slope, intercept), cov = np.polyfit(xs, ys, 1, w=w, cov=True)
        slope = float(slope)
        intercept = float(intercept)
        slope_err = float(np.sqrt(cov[0, 0])) if cov is not None else float("nan")

        return FitSummary(
            slope=slope,
            intercept=intercept,
            slope_err=slope_err,
            n_tiles=len(xs),
            points=pts,
            errors=np.asarray(errors),
            fracs=np.asarray(fracs),
            amps=np.asarray(amps),
            sigmas=np.asarray(sigmas),
        )


def run_all_filters(
    det: str,
    tiles_dir: Path,
    iso_dir: Path,
    filter_pairs: List[Tuple[str, str]],
    max_y_err: float,
    bins_per_tile: int,
) -> Dict[Tuple[str, str], FitSummary]:
    out: Dict[Tuple[str, str], FitSummary] = {}
    for filt1, filt2 in filter_pairs:
        runner = RunCurveFit(
            filt1=filt1,
            filt2=filt2,
            filty=filt1,
            det=det,
            tiles_dir=tiles_dir,
            iso_dir=iso_dir,
            max_y_err=max_y_err,
            bins_per_tile=bins_per_tile,
        )
        out[(filt1, filt2)] = runner.run()
    return out
