from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from spisea import atmospheres, evolution, reddening, synthetic

SPISEA_FILTER_MAP = {
    "F115W": "jwst,F115W",
    "F140M": "jwst,F140M",
    "F182M": "jwst,F182M",
    "F212N": "jwst,F212N",
    "F323N": "jwst,F323N",
    "F405N": "jwst,F405N",
}


class Isochrones:
    """Predicted RC slopes from Fritz+11 for a given CMD."""

    def __init__(
        self,
        filt1: str,
        filt2: str,
        filty: str,
        iso_dir: Path,
    ):
        self.filt1 = filt1
        self.filt2 = filt2
        self.filty = filty

        self.evo_model = evolution.MISTv1()
        self.atm_func = atmospheres.get_merged_atmosphere
        self.red_law = reddening.RedLawFritz11(scale_lambda=2.166)

        iso_dir.mkdir(exist_ok=True, parents=True)
        self.iso_dir = iso_dir
        self._cache_path = self.iso_dir / "reddening_slope_cache.json"

    def _cache_key(self) -> str:
        return f"{self.filt1}|{self.filt2}|{self.filty}"

    def _load_cache(self) -> dict:
        if not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text())
        except Exception:
            return {}

    def _save_cache(self, data: dict) -> None:
        try:
            self._cache_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def render_isochrone(
        self,
        AKs: float,
        logAge: float = np.log10(10**9),
        distance: float = 8000.0,
    ):
        filt_list = [SPISEA_FILTER_MAP.get(filt) for filt in (self.filt1, self.filt2)]

        isochrone = synthetic.IsochronePhot(
            logAge=logAge,
            AKs=AKs,
            distance=distance,
            filters=filt_list,
            red_law=self.red_law,
            atm_func=self.atm_func,
            evo_model=self.evo_model,
            iso_dir=str(self.iso_dir / f"{self.filt1}-{self.filt2}_{self.filty}"),
        )

        mass = isochrone.points["mass"]
        idx = np.flatnonzero(abs(mass) == min(abs(mass)))
        return isochrone, idx

    def reddening_slope(self) -> float:
        cache = self._load_cache()
        key = self._cache_key()
        if key in cache:
            return float(cache[key])

        iso_ext_1, idx1 = self.render_isochrone(AKs=0)
        iso_ext_2, idx2 = self.render_isochrone(AKs=1)

        def get_pt0(ext, key_idx, star_idx):
            key = list(ext.points.keys())[key_idx]
            return ext.points[key][star_idx][0]

        iso_idx = 8 if self.filt1 == self.filty else 9

        y2_y1 = get_pt0(iso_ext_1, iso_idx, idx1) - get_pt0(iso_ext_2, iso_idx, idx2)
        x2_x1 = (
            (get_pt0(iso_ext_1, 8, idx1) - get_pt0(iso_ext_1, 9, idx1))
            - (get_pt0(iso_ext_2, 8, idx2) - get_pt0(iso_ext_2, 9, idx2))
        )
        slope = float(y2_y1 / x2_x1)
        cache[key] = slope
        self._save_cache(cache)
        return slope
