"""
Remaining useful life (RUL) from the bearing health indicator.

Onset comes from the health indicator (health.py). After onset, the degradation level
D(t) = smoothed RMS / healthy RMS is modelled as exponential growth,
log D(t) = a + b (t - t_onset), fitted by least squares on the data available at
evaluation time t_eval. Predicted failure is when the fitted curve reaches the failure
threshold D_fail. We use the RMS ratio rather than the max-z HI because its end-of-life
value is far more consistent across bearings (about 2.5-5x vs a max-z HI of 14-80).

Leave-one-bearing-out (LOBO): D_fail for bearing i is the median end-of-life level of the
*other* failed bearings, so a bearing never sees its own failure level.

We evaluate at 50/75/90 % of each bearing's onset -> failure interval. Onset comes late in
life (58-96 %), so fixed percentages of *total* life would often fall before any
degradation had been seen.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from compressor_guard.vibration.io import KNOWN_FAILURES


SIGNAL = "rms_ratio"


def end_of_life_hi(g: pd.DataFrame, tail: int = 6, signal: str = SIGNAL) -> float:
    return float(g[signal].tail(tail).median())


def fit_exponential(hours: np.ndarray, hi: np.ndarray, t0: float):
    y = np.log(np.clip(hi, 1e-3, None))
    x = hours - t0
    b, a = np.polyfit(x, y, 1)
    return a, b


def predict_rul(g: pd.DataFrame, onset: float, t_eval: float, hi_fail: float,
                min_points: int = 12, max_rul: float = 2000.0) -> Optional[float]:
    win = g[(g["hours"] >= onset) & (g["hours"] <= t_eval)]
    if len(win) < min_points:
        return None
    a, b = fit_exponential(win["hours"].to_numpy(), win[SIGNAL].to_numpy(), onset)
    if b <= 0:
        return max_rul   # no upward trend yet: model says "not soon"
    t_fail = onset + (np.log(hi_fail) - a) / b
    return float(np.clip(t_fail - t_eval, 0.0, max_rul))


def lobo_rul(hi_all: pd.DataFrame, onsets: Dict[tuple, Optional[float]],
             fractions: Sequence[float] = (0.5, 0.75, 0.9), min_points: int = 12,
             max_rul: float = 2000.0) -> pd.DataFrame:
    """RUL predictions for every failed bearing with an onset, leave-one-bearing-out."""
    groups = {k: g for k, g in hi_all.groupby(["test", "bearing"])}
    failed = [k for k in KNOWN_FAILURES if k in groups]
    eol_hi = {k: end_of_life_hi(groups[k]) for k in failed}
    rows = []
    for key in failed:
        onset = onsets.get(key)
        g = groups[key]
        eol = float(g["hours"].max())
        others = [eol_hi[k] for k in failed if k != key]
        hi_fail = float(np.median(others))
        for f in fractions:
            row = {"test": key[0], "bearing": key[1], "failure": KNOWN_FAILURES[key],
                   "eval_fraction": f, "hi_fail": hi_fail, "own_eol_hi": eol_hi[key]}
            if onset is None:
                rows.append({**row, "t_eval": None, "actual_rul": None, "pred_rul": None})
                continue
            t_eval = onset + f * (eol - onset)
            actual = eol - t_eval
            pred = predict_rul(g, onset, t_eval, hi_fail, min_points, max_rul)
            rows.append({**row, "t_eval": t_eval, "actual_rul": actual, "pred_rul": pred,
                         "error_h": None if pred is None else pred - actual,
                         "abs_pct_error": None if pred is None or actual <= 0 else 100 * abs(pred - actual) / actual})
    return pd.DataFrame(rows)
