from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import AnnulusConfig
from .io_utils.paths import OutputPaths, annulus_detector_name


@dataclass(frozen=True)
class AnnulusMetadata:
    index: int
    r_inner: float
    r_outer: float
    n_stars: int
    sag_a_reference: Tuple[float, float]


def _build_catalog_with_distances(
    rc_frames: Iterable[Tuple[str, pd.DataFrame]],
    cfg: AnnulusConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    frames = []
    for det_name, df in rc_frames:
        tmp = df.copy()
        tmp["detector"] = det_name
        frames.append(tmp)

    if not frames:
        raise ValueError("No RC frames provided for annulus construction.")

    catalog = pd.concat(frames, ignore_index=True)
    x0, y0 = cfg.sag_a_reference
    x = catalog["x"].to_numpy(dtype=float)
    y = catalog["y"].to_numpy(dtype=float)
    distances = np.hypot(x - x0, y - y0)
    r_max = float(np.max(distances))
    if r_max <= 0:
        raise ValueError("r_max <= 0 (all stars at reference point?)")
    return catalog, distances


def build_annulus_catalog(
    rc_frames: Iterable[Tuple[str, pd.DataFrame]],
    cfg: AnnulusConfig,
) -> Tuple[Dict[int, pd.DataFrame], List[Tuple[float, float]]]:
    catalog, distances = _build_catalog_with_distances(rc_frames, cfg)

    if cfg.n_annuli is None or cfg.n_annuli < 1:
        raise ValueError("n_annuli must be >= 1")

    r_max = float(np.max(distances))
    edges = np.sqrt(np.linspace(0.0, 1.0, cfg.n_annuli + 1)) * r_max
    pairs = list(zip(edges[:-1], edges[1:]))
    bin_idx = np.searchsorted(edges, distances, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, cfg.n_annuli - 1)
    frames_by_bin = [catalog[bin_idx == k].copy() for k in range(cfg.n_annuli)]

    i = 0
    while i < len(frames_by_bin):
        if len(frames_by_bin[i]) >= cfg.min_stars or len(frames_by_bin) == 1:
            i += 1
            continue

        if i == 0:
            frames_by_bin[1] = pd.concat([frames_by_bin[1], frames_by_bin[0]], ignore_index=False)
            pairs[1] = (pairs[0][0], pairs[1][1])
            del frames_by_bin[0]
            del pairs[0]
        else:
            frames_by_bin[i - 1] = pd.concat([frames_by_bin[i - 1], frames_by_bin[i]], ignore_index=False)
            pairs[i - 1] = (pairs[i - 1][0], pairs[i][1])
            del frames_by_bin[i]
            del pairs[i]
            i -= 1

    disc_catalog_fused = {k: df.sort_index() for k, df in enumerate(frames_by_bin)}
    return disc_catalog_fused, pairs


def build_min_stars_annulus_catalog(
    rc_frames: Iterable[Tuple[str, pd.DataFrame]],
    cfg: AnnulusConfig,
) -> Tuple[Dict[int, pd.DataFrame], List[Tuple[float, float]]]:
    if cfg.min_stars < 1:
        raise ValueError("min_stars must be >= 1")

    catalog, distances = _build_catalog_with_distances(rc_frames, cfg)
    order = np.argsort(distances)
    distances_sorted = distances[order]
    n_total = len(distances_sorted)

    frames_by_bin: List[pd.DataFrame] = []
    pairs: List[Tuple[float, float]] = []
    start = 0
    r_inner = 0.0

    while start < n_total:
        target = start + cfg.min_stars - 1
        if target >= n_total:
            break

        r_outer = float(distances_sorted[target])
        while target + 1 < n_total and distances_sorted[target + 1] == r_outer:
            target += 1

        idx = order[start : target + 1]
        frames_by_bin.append(catalog.iloc[idx].copy())
        pairs.append((r_inner, r_outer))

        start = target + 1
        r_inner = r_outer

    disc_catalog = {k: df.sort_index() for k, df in enumerate(frames_by_bin)}
    return disc_catalog, pairs


def _persist_annuli(
    annuli: Dict[int, pd.DataFrame],
    radii: List[Tuple[float, float]],
    cfg: AnnulusConfig,
    out: OutputPaths,
) -> List[AnnulusMetadata]:
    metadata: List[AnnulusMetadata] = []
    for idx, df in annuli.items():
        annulus_out = out.annulus(idx)
        annulus_out.root.mkdir(parents=True, exist_ok=True)
        annulus_out.tiles_dir.mkdir(parents=True, exist_ok=True)
        annulus_out.fits_dir.mkdir(parents=True, exist_ok=True)
        annulus_out.ratios_dir.mkdir(parents=True, exist_ok=True)
        annulus_out.plots_dir.mkdir(parents=True, exist_ok=True)

        df.to_pickle(annulus_out.rc_stars)

        r_in, r_out = radii[idx]
        meta = AnnulusMetadata(
            index=idx,
            r_inner=float(r_in),
            r_outer=float(r_out),
            n_stars=int(len(df)),
            sag_a_reference=cfg.sag_a_reference,
        )

        with annulus_out.annulus_meta.open("w") as f:
            json.dump(meta.__dict__, f, indent=2)

        metadata.append(meta)

    return metadata


def generate_annuli(
    nrcb_pickles: Dict[str, Path],
    cfg: AnnulusConfig,
    out: OutputPaths,
) -> List[AnnulusMetadata]:
    out.ensure()

    rc_frames = []
    for det_name, p in nrcb_pickles.items():
        with p.open("rb") as f:
            df = pd.read_pickle(f)
        rc_frames.append((det_name, df))

    if cfg.n_annuli is None:
        annuli, radii = build_min_stars_annulus_catalog(rc_frames, cfg)
    else:
        annuli, radii = build_annulus_catalog(rc_frames, cfg)
    return _persist_annuli(annuli, radii, cfg, out)
