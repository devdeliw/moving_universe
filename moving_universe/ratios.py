from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class RatioResult:
    ratios_f115w: Dict[Tuple[str, str], Tuple[float, float]]
    ratios_f212n: Dict[Tuple[str, str], Tuple[float, float]]


def a_over_f115w_from_RB(rb: float, srb: float) -> Tuple[float, float]:
    r = 1.0 - 1.0 / rb
    sr = abs(srb) / (rb * rb)
    return r, sr


def invert(x: float, sx: float) -> Tuple[float, float]:
    y = 1.0 / x
    sy = abs(sx) / (x * x)
    return y, sy


def product(a: float, sa: float, b: float, sb: float) -> Tuple[float, float]:
    q = a * b
    if a == 0.0 or b == 0.0:
        sq = math.hypot(b * sa, a * sb)
    else:
        rel2 = (sa / a) ** 2 + (sb / b) ** 2
        sq = abs(q) * math.sqrt(rel2)
    return q, sq


def compute_ratios(
    det: str,
    slopes: Dict[Tuple[str, str], Tuple[float, float, float]],
    f_ref: str = "F115W",
    f_anchor: str = "F212N",
) -> RatioResult:
    ratios_f115w: Dict[Tuple[str, str], Tuple[float, float]] = {}
    ratios_f212n: Dict[Tuple[str, str], Tuple[float, float]] = {}

    f115_over_f212: Dict[str, Tuple[float, float]] = {}

    for (filt1, filt2), (slope, _, slope_err) in slopes.items():
        if filt1 != f_ref:
            continue
        r, sr = a_over_f115w_from_RB(slope, slope_err)
        ratios_f115w[(det, filt2)] = (r, sr)
        if filt2 == f_anchor:
            r212, sr212 = r, sr
            inv, sinv = invert(r212, sr212)
            f115_over_f212[det] = (inv, sinv)
            ratios_f212n[(det, f_ref)] = (inv, sinv)

    for (det_name, filt2), (r, sr) in ratios_f115w.items():
        if filt2 == f_anchor:
            ratios_f212n[(det_name, filt2)] = (1.0, 0.0)
            continue
        if det_name not in f115_over_f212:
            continue
        f115_over_f212_val, s_f115_over_f212 = f115_over_f212[det_name]
        q, sq = product(r, sr, f115_over_f212_val, s_f115_over_f212)
        ratios_f212n[(det_name, filt2)] = (q, sq)

    return RatioResult(ratios_f115w=ratios_f115w, ratios_f212n=ratios_f212n)


def write_ratios(result: RatioResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "A_lambda_over_A_F115W.pickle").open("wb") as f:
        pickle.dump(result.ratios_f115w, f)
    with (out_dir / "A_lambda_over_A_F212N.pickle").open("wb") as f:
        pickle.dump(result.ratios_f212n, f)

    with (out_dir / "A_lambda_over_A_F115W.json").open("w") as f:
        json.dump({str(k): v for k, v in result.ratios_f115w.items()}, f, indent=2)
    with (out_dir / "A_lambda_over_A_F212N.json").open("w") as f:
        json.dump({str(k): v for k, v in result.ratios_f212n.items()}, f, indent=2)
