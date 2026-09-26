"""
Alert generation: smooth the raw anomaly score, then raise an alert only once the
smoothed score has stayed above the threshold for a sustained period. Both steps are
trailing (causal), and `stream/state.py` reproduces them row by row.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# A gap longer than this between scored minutes breaks a persistence run.
MAX_GAP_MINUTES = 5.0


def smooth_anomaly_scores(
    scores: Union[np.ndarray, pd.Series],
    window_steps: int = 30,
    method: str = "median",
    timestamps: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """
    Trailing rolling median (or mean) of the raw scores.

    With `timestamps`, the window is time-based: `window_steps` minutes, `(t - W, t]`.
    Otherwise it counts samples.
    """
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    if timestamps is not None:
        s.index = pd.DatetimeIndex(timestamps)
        roll = s.rolling(f"{int(window_steps)}min", min_periods=1)
    else:
        roll = s.rolling(int(window_steps), min_periods=1)
    return (roll.median() if method == "median" else roll.mean()).to_numpy()


def apply_persistence_filter(
    score_above_threshold: Union[np.ndarray, pd.Series],
    persistence_steps: int = 60,
    timestamps: Optional[Union[pd.Series, np.ndarray]] = None,
    max_gap_minutes: float = MAX_GAP_MINUTES,
) -> np.ndarray:
    """
    True where the condition has held for at least `persistence_steps` samples (or, with
    `timestamps`, spanned at least that many minutes counting the current one) with no
    data gap longer than `max_gap_minutes` inside the run.
    """
    above = np.asarray(score_above_threshold, dtype=bool)
    if timestamps is None:
        s = pd.Series(above.astype(int))
        consecutive = s.groupby((s != s.shift()).cumsum()).cumsum() * s
        return (consecutive >= persistence_steps).to_numpy()

    t = pd.to_datetime(pd.Series(timestamps)).to_numpy().astype("datetime64[s]").astype(np.int64) / 60.0
    gap = np.empty_like(t)
    gap[0] = np.inf
    gap[1:] = np.diff(t)
    prev_above = np.concatenate([[False], above[:-1]])
    run_start = np.where(above & ~(prev_above & (gap <= max_gap_minutes)), t, np.nan)
    run_start = pd.Series(run_start).ffill().to_numpy()
    elapsed = t - run_start + 1.0   # minutes spanned, counting the current minute
    return above & (elapsed >= persistence_steps)


def extract_alert_events(
    df_out: pd.DataFrame,
    timestamp_col: str = "timestamp",
    shift_col: str = "shift",
    max_gap_minutes: float = MAX_GAP_MINUTES,
) -> pd.DataFrame:
    """Group consecutive active `alert_flag` minutes into discrete alert events."""
    cols = ["event_id", "start", "end", "duration_hours", "peak_score", "shift"]
    active = df_out["alert_flag"].to_numpy(dtype=bool)
    if not active.any():
        return pd.DataFrame(columns=cols)
    t = df_out[timestamp_col]
    gap_min = t.diff().dt.total_seconds().div(60).fillna(np.inf).to_numpy()
    prev = np.concatenate([[False], active[:-1]])
    new_event = active & ~(prev & (gap_min <= max_gap_minutes))
    event_id = np.cumsum(new_event)
    sub = df_out.loc[active].assign(_eid=event_id[active])
    agg = sub.groupby("_eid").agg(
        start=(timestamp_col, "first"),
        end=(timestamp_col, "last"),
        peak_score=("smoothed_score", "max"),
    )
    agg["shift"] = sub.groupby("_eid")[shift_col].first().astype(str) if shift_col in sub else "N/A"
    agg["duration_hours"] = (agg["end"] - agg["start"]).dt.total_seconds() / 3600.0
    agg = agg.reset_index().rename(columns={"_eid": "event_id"})
    return agg[cols]


def generate_alerts(
    df: pd.DataFrame,
    raw_scores: np.ndarray,
    threshold: float,
    smoothing_window: int = 30,
    persistence_steps: int = 60,
    timestamp_col: str = "timestamp",
    shift_col: str = "shift",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end alert pipeline (time-based windows, in minutes):
    1. trailing rolling median of the raw score over `smoothing_window` minutes
    2. flag smoothed score > threshold
    3. alert once the flag has persisted for `persistence_steps` minutes
    4. group consecutive alert minutes into events

    Returns (df with raw_score / smoothed_score / alert_flag, alert events table).
    """
    ts = df[timestamp_col]
    smoothed = smooth_anomaly_scores(raw_scores, smoothing_window, "median", timestamps=ts)
    active = apply_persistence_filter(smoothed > threshold, persistence_steps, timestamps=ts)

    df_out = df.copy()
    df_out["raw_score"] = np.asarray(raw_scores, dtype=np.float64)
    df_out["smoothed_score"] = smoothed
    df_out["alert_flag"] = active
    return df_out, extract_alert_events(df_out, timestamp_col, shift_col)
