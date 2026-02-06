from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class AnnulusPaths:
    root: Path

    @property
    def annulus_meta(self) -> Path:
        return self.root / "annulus.json"

    @property
    def rc_stars(self) -> Path:
        return self.root / "rc_stars.pickle"

    @property
    def tiles_dir(self) -> Path:
        return self.root / "tiles"

    @property
    def fits_dir(self) -> Path:
        return self.root / "fits"

    @property
    def ratios_dir(self) -> Path:
        return self.root / "ratios"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"


@dataclass(frozen=True)
class OutputPaths:
    output_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.output_root / "runs" / self.run_id

    @property
    def annuli_dir(self) -> Path:
        return self.run_root / "annuli"

    @property
    def summary_dir(self) -> Path:
        return self.run_root / "summary"

    def annulus(self, index: int) -> AnnulusPaths:
        return AnnulusPaths(self.annuli_dir / f"Annuli_{index}")

    def ensure(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.annuli_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)


def tile_filename(det_name: str, filt1: str, filt2: str, filty: str) -> str:
    return f"[{det_name}] {filt1}-{filt2} vs. {filty}.pickle"


def tile_plot_filename(det_name: str, filt1: str, filt2: str, filty: str) -> str:
    return f"[{det_name}] {filt1}-{filt2} vs. {filty}.png"


def fit_plot_filename(det_name: str, filt1: str, filt2: str, filty: str) -> str:
    return f"[{det_name}] {filt1}-{filt2} vs. {filty}.png"


def annulus_detector_name(index: int) -> str:
    return f"Annuli_{index}"


def annulus_index_from_name(name: str) -> int:
    if not name.startswith("Annuli_"):
        raise ValueError(f"Unexpected annulus name: {name}")
    return int(name.split("_", 1)[1])
