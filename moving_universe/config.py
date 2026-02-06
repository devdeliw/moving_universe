from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

FilterPair = Tuple[str, str]

FILTER_COMBINATIONS: Sequence[FilterPair] = (
    ("F115W", "F140M"),
    ("F115W", "F182M"),
    ("F115W", "F212N"),
    ("F115W", "F323N"),
    ("F115W", "F405N"),
)

DEFAULT_RC_COLOR_RANGE = (
    (2.0, 4.8),
    (4.0, 9.0),
    (5.0, 10.0),
    (5.8, 11.0),
    (6.0, 12.5),
)


@dataclass(frozen=True)
class AnnulusConfig:
    sag_a_reference: Tuple[float, float] = (0.0, 20.0)
    n_annuli: Optional[int] = 10
    min_stars: int = 1000


@dataclass(frozen=True)
class TileConfig:
    n_tiles: int = 10
    bins_per_tile: int = 50
    max_y_err: float = 0.20


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    assets_dir: Path
    output_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "ProjectPaths":
        root = repo_root
        return ProjectPaths(
            root=root,
            assets_dir=root / "assets",
            output_dir=root / "output",
        )


@dataclass(frozen=True)
class PipelineConfig:
    paths: ProjectPaths
    annuli: AnnulusConfig = AnnulusConfig()
    tiles: TileConfig = TileConfig()
    rc_color_range: Sequence[Tuple[float, float]] = DEFAULT_RC_COLOR_RANGE
    filters: Sequence[FilterPair] = FILTER_COMBINATIONS
    run_id: str = "default"

    def rc_color_range_by_index(self, index: int) -> Tuple[float, float]:
        return tuple(self.rc_color_range[index])  # type: ignore[return-value]

    def filter_pairs(self) -> Iterable[FilterPair]:
        return self.filters
