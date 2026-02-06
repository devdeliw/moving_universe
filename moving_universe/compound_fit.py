from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _compound_model(x, u, amp, mu, sig, m, b):
    f_rc = _sigmoid(u)
    g = amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    lin = m * x + b
    y = f_rc * g + (1.0 - f_rc) * lin
    return np.clip(y, 1e-12, None)


@dataclass
class FitResult:
    theta: np.ndarray
    cov: Optional[np.ndarray]
    mu: float
    mu_err: float
    frac_rc: float
    amp: float
    sig: float
    m: float
    b: float
    x_hist: np.ndarray
    y_hist: np.ndarray
    n: int
    y_max: float
    mode: float
    window: Tuple[float, float]
    mu_in_window: bool


class CompoundTileFit:
    def __init__(
        self,
        data: np.ndarray,
        *,
        bins: int = 50,
        data_range: Optional[Tuple[float, float]] = None,
        maxfev: int = 20000,
        peak_frac: float = 0.60,
        guard_frac: float = 0.10,
    ):
        self.y = np.asarray(data).ravel()

        if data_range is None:
            lo, hi = np.percentile(self.y, [0.5, 99.5])
            pad = 0.05 * (hi - lo + 1e-12)
            data_range = (lo - pad, hi + pad)

        self.range = data_range
        self.bins = bins
        self.maxfev = maxfev
        self.peak_frac = float(np.clip(peak_frac, 0.2, 0.9))
        self.guard_frac = float(np.clip(guard_frac, 0.0, 0.5))

    def _hist(self):
        cnts, edges = np.histogram(self.y, bins=self.bins, range=self.range)
        xc = 0.5 * (edges[:-1] + edges[1:])
        yc = cnts.astype(float)
        sig = np.sqrt(np.maximum(yc, 1.0))
        return xc, yc, sig, edges

    def _dense_window(self, yc: np.ndarray, edges: np.ndarray) -> Tuple[float, float, float]:
        idx_pk = int(np.argmax(yc)) if yc.size else 0
        ypk = float(yc[idx_pk]) if yc.size else 1.0
        thr = self.peak_frac * max(ypk, 1.0)

        L = idx_pk
        while L > 0 and yc[L - 1] >= thr:
            L -= 1
        R = idx_pk
        while R < len(yc) - 1 and yc[R + 1] >= thr:
            R += 1

        wL, wR = edges[L], edges[R + 1]
        mode = 0.5 * (edges[idx_pk] + edges[idx_pk + 1])
        return wL, wR, mode

    def _init_guess(self, y, mode, wL, wR):
        L_full, U_full = self.range
        span_full = max(U_full - L_full, 1e-3)
        width_w = max(wR - wL, 0.05 * span_full)

        y_max = float(np.max(y)) if np.size(y) else 1.0
        y_med = float(np.median(y)) if np.size(y) else 1.0

        mu0 = float(mode)
        sig0 = 0.5 * width_w / np.sqrt(2.0 * np.log(2.0))
        sig0 = np.clip(sig0, 0.02 * span_full, 0.35 * span_full)

        amp0 = max(y_max - y_med, 1.0)
        m0 = 0.0
        b0 = max(y_med, 1.0)
        u0 = 0.0

        p0 = np.array([u0, amp0, mu0, sig0, m0, b0], dtype=float)

        guard = self.guard_frac * span_full
        mu_lo = wL - guard
        mu_hi = wR + guard

        lower = np.array([
            -8.0,
            0.0,
            mu_lo,
            0.02 * span_full,
            -5.0 * y_max / span_full,
            0.0,
        ])
        upper = np.array([
            8.0,
            5.0 * y_max,
            mu_hi,
            0.50 * span_full,
            5.0 * y_max / span_full,
            5.0 * y_max,
        ])
        return p0, (lower, upper)

    def fit(self) -> FitResult:
        xh, yh, sig_h, edges = self._hist()
        wL, wR, mode = self._dense_window(yh, edges)
        p0, bounds = self._init_guess(yh, mode, wL, wR)

        try:
            popt, pcov = curve_fit(
                _compound_model,
                xh,
                yh,
                p0=p0,
                sigma=sig_h,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=self.maxfev,
            )
        except Exception:
            popt, pcov = p0, None

        u, amp, mu, sig, m, b = popt
        mu_err = float(np.sqrt(pcov[2, 2])) if (pcov is not None and np.isfinite(pcov[2, 2])) else np.nan
        frac_rc = float(_sigmoid(np.array([u]))[0])

        idx_pk = int(np.argmax(yh)) if yh.size else 0
        y_pk = float(yh[idx_pk]) if yh.size else 1.0
        mu_in_window = (mu >= wL) and (mu <= wR)

        return FitResult(
            theta=popt,
            cov=pcov,
            mu=float(mu),
            mu_err=mu_err,
            frac_rc=frac_rc,
            amp=float(amp),
            sig=float(sig),
            m=float(m),
            b=float(b),
            x_hist=xh,
            y_hist=yh,
            n=int(len(self.y)),
            y_max=float(y_pk),
            mode=float(mode),
            window=(float(wL), float(wR)),
            mu_in_window=bool(mu_in_window),
        )
