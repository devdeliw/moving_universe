from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .annuli import AnnulusMetadata, generate_annuli
from .config import PipelineConfig
from .fit import FitSummary, run_all_filters
from .io_utils.paths import OutputPaths, annulus_detector_name, tile_plot_filename, tile_filename, fit_plot_filename
from .ratios import compute_ratios, write_ratios
from .tiles import Tiles, save_tiles
from .plot.annuli import plot_annuli
from .plot.tiles import plot_tiles
from .plot.fits import plot_fit


@dataclass
class SlopeRecord:
    det: str
    filt1: str
    filt2: str
    slope: float
    intercept: float
    slope_err: float


def _annulus_dirs(out: OutputPaths) -> List[Path]:
    if not out.annuli_dir.exists():
        return []
    ann_dirs = [p for p in out.annuli_dir.iterdir() if p.is_dir() and p.name.startswith("Annuli_")]
    return sorted(ann_dirs, key=lambda p: int(p.name.split("_", 1)[1]))


def run_annuli(cfg: PipelineConfig, *, plot: bool = False) -> List[AnnulusMetadata]:
    out = OutputPaths(cfg.paths.output_dir, cfg.run_id)
    nrcb_pickles = {
        "NRCB1": cfg.paths.assets_dir / "NRCB1_red_clump_stars.pickle",
        "NRCB2": cfg.paths.assets_dir / "NRCB2_red_clump_stars.pickle",
        "NRCB3": cfg.paths.assets_dir / "NRCB3_red_clump_stars.pickle",
        "NRCB4": cfg.paths.assets_dir / "NRCB4_red_clump_stars.pickle",
    }
    for name, path in nrcb_pickles.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing RC pickle for {name}: {path}")

    metadata = generate_annuli(nrcb_pickles, cfg.annuli, out)
    out.summary_dir.mkdir(parents=True, exist_ok=True)
    annuli_mode = "min_stars_radial" if cfg.annuli.n_annuli is None else "equal_area_fuse"
    run_meta = {
        "run_id": cfg.run_id,
        "requested_annuli": cfg.annuli.n_annuli,
        "effective_annuli": len(metadata),
        "min_stars": cfg.annuli.min_stars,
        "sag_a_reference": cfg.annuli.sag_a_reference,
        "annuli_mode": annuli_mode,
    }
    with (out.summary_dir / "run.json").open("w") as f:
        json.dump(run_meta, f, indent=2)
    print(
        f"run_id={cfg.run_id} annuli_mode={annuli_mode} requested_annuli={cfg.annuli.n_annuli} "
        f"effective_annuli={len(metadata)} min_stars={cfg.annuli.min_stars}"
    )

    if plot:
        annuli_frames = []
        for meta in metadata:
            ann_dir = out.annulus(meta.index)
            df = pd.read_pickle(ann_dir.rc_stars)
            annuli_frames.append((meta.index, df))
        fig, _ = plot_annuli(annuli_frames, cfg.annuli.sag_a_reference)
        plot_dir = out.summary_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / "annuli_scatter.png", dpi=300)

    return metadata


def run_tiles(cfg: PipelineConfig, *, plot: bool = False) -> None:
    out = OutputPaths(cfg.paths.output_dir, cfg.run_id)
    iso_dir = cfg.paths.output_dir / "isochrones_cache"
    for ann_dir in _annulus_dirs(out):
        ann_index = int(ann_dir.name.split("_")[1])
        annulus = out.annulus(ann_index)
        rc = pd.read_pickle(annulus.rc_stars)
        det = annulus_detector_name(ann_index)

        for idx, (filt1, filt2) in enumerate(cfg.filter_pairs()):
            m1 = np.asarray(rc[f"m{filt1}"])
            m2 = np.asarray(rc[f"m{filt2}"])
            my = m1

            tiles = Tiles(
                m1,
                m2,
                my,
                filt1,
                filt2,
                filt1,
                det,
                n_tiles=cfg.tiles.n_tiles,
                clr_range=cfg.rc_color_range[idx],
                iso_dir=iso_dir,
            ).tiles()

            tiles_path = save_tiles(
                tiles,
                annulus.tiles_dir,
                det,
                filt1,
                filt2,
                filt1,
            )

            if plot:
                fig, _ = plot_tiles(tiles.star_tiles, det, filt1, filt2, filt1)
                plot_dir = annulus.tiles_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(plot_dir / tile_plot_filename(det, filt1, filt2, filt1), dpi=300)
                plt.close(fig)

            _ = tiles_path


def run_fits(cfg: PipelineConfig, *, plot: bool = False) -> Dict[Tuple[str, str, str], Tuple[float, float, float]]:
    out = OutputPaths(cfg.paths.output_dir, cfg.run_id)
    iso_dir = cfg.paths.output_dir / "isochrones_cache"

    slopes: Dict[Tuple[str, str, str], Tuple[float, float, float]] = {}

    for ann_dir in _annulus_dirs(out):
        ann_index = int(ann_dir.name.split("_")[1])
        annulus = out.annulus(ann_index)
        det = annulus_detector_name(ann_index)

        fit_results = run_all_filters(
            det,
            tiles_dir=annulus.tiles_dir,
            iso_dir=iso_dir,
            filter_pairs=list(cfg.filter_pairs()),
            max_y_err=cfg.tiles.max_y_err,
            bins_per_tile=cfg.tiles.bins_per_tile,
        )

        slopes_json = []
        for (filt1, filt2), summary in fit_results.items():
            slopes[(det, filt1, filt2)] = (summary.slope, summary.intercept, summary.slope_err)
            slopes_json.append(
                SlopeRecord(
                    det=det,
                    filt1=filt1,
                    filt2=filt2,
                    slope=summary.slope,
                    intercept=summary.intercept,
                    slope_err=summary.slope_err,
                ).__dict__
            )

            if plot:
                tile_path = annulus.tiles_dir / tile_filename(det, filt1, filt2, filt1)
                with tile_path.open("rb") as f:
                    star_tiles = pickle.load(f)
                fig, _ = plot_fit(
                    star_tiles=star_tiles,
                    points=summary.points,
                    errors=summary.errors,
                    slope=summary.slope,
                    intercept=summary.intercept,
                    slope_err=summary.slope_err,
                    det=det,
                    filt1=filt1,
                    filt2=filt2,
                    filty=filt1,
                )
                plot_dir = annulus.fits_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(plot_dir / fit_plot_filename(det, filt1, filt2, filt1), dpi=300)
                plt.close(fig)

        annulus.fits_dir.mkdir(parents=True, exist_ok=True)
        with (annulus.fits_dir / "slopes.json").open("w") as f:
            json.dump(slopes_json, f, indent=2)

    out.summary_dir.mkdir(parents=True, exist_ok=True)
    with (out.summary_dir / "slopes.pickle").open("wb") as f:
        pickle.dump(slopes, f)

    return slopes


def run_ratios(cfg: PipelineConfig, slopes: Optional[Dict[Tuple[str, str, str], Tuple[float, float, float]]] = None) -> None:
    out = OutputPaths(cfg.paths.output_dir, cfg.run_id)

    if slopes is None:
        slopes_path = out.summary_dir / "slopes.pickle"
        if not slopes_path.exists():
            raise FileNotFoundError("slopes.pickle not found; run fits first")
        with slopes_path.open("rb") as f:
            slopes = pickle.load(f)

    for ann_dir in _annulus_dirs(out):
        ann_index = int(ann_dir.name.split("_")[1])
        annulus = out.annulus(ann_index)
        det = annulus_detector_name(ann_index)

        det_slopes = {
            (filt1, filt2): slopes[(det, filt1, filt2)]
            for (det_name, filt1, filt2) in slopes.keys()
            if det_name == det
        }

        ratios = compute_ratios(det, det_slopes)
        write_ratios(ratios, annulus.ratios_dir)

    ratios_f115w: Dict[Tuple[str, str], Tuple[float, float]] = {}
    ratios_f212n: Dict[Tuple[str, str], Tuple[float, float]] = {}

    for ann_dir in _annulus_dirs(out):
        ann_index = int(ann_dir.name.split("_", 1)[1])
        annulus = out.annulus(ann_index)
        with (annulus.ratios_dir / "A_lambda_over_A_F115W.pickle").open("rb") as f:
            ratios_f115w.update(pickle.load(f))
        with (annulus.ratios_dir / "A_lambda_over_A_F212N.pickle").open("rb") as f:
            ratios_f212n.update(pickle.load(f))

    with (out.summary_dir / "ratios_A_lambda_over_A_F115W.pickle").open("wb") as f:
        pickle.dump(ratios_f115w, f)

    with (out.summary_dir / "ratios_A_lambda_over_A_F212N.pickle").open("wb") as f:
        pickle.dump(ratios_f212n, f)


def run_all(cfg: PipelineConfig, *, plot: bool = False) -> None:
    run_annuli(cfg, plot=plot)
    run_tiles(cfg, plot=plot)
    try:
        slopes = run_fits(cfg, plot=plot)
    except RuntimeError as exc:
        print(f"run_id={cfg.run_id} failed during fits: {exc}")
        return
    run_ratios(cfg, slopes=slopes)
