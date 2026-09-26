"""
Stateful, incremental version of the Module A pipeline.

Rows arrive one at a time in timestamp order, as they would from a historian or an
MQTT/OPC-UA subscriber. The state:

1. keeps the previous row and the current load-run start (the raw-row signals),
2. accumulates the current calendar minute,
3. when a minute closes, drops it if frozen, otherwise appends it to a 24 h buffer
   and computes the trailing-window features from that buffer,
4. applies drift z-scores, scores the minute with the trained detectors, and runs
   the smoothing + persistence alert rule.

Every step mirrors `features.py` / `alerts.py`. `tests/test_stream_parity.py` checks
that the streamed features and scores match the batch pipeline on the same data.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compressor_guard.alerts import MAX_GAP_MINUTES
from compressor_guard.features import (
    ANALOG,
    DEFAULT_SHIFTS,
    DRIFT_TARGETS,
    FEATURE_COLUMNS,
    MAX_GAP_SECONDS,
    MIN_ROWS_FOR_FROZEN,
    shift_label,
)

MINUTE_NS = 60 * 10**9
WINDOWS_MIN = {"15": 15, "60": 60, "6h": 360, "24h": 1440}


@dataclass
class _Bucket:
    ts: int                      # minute start, ns since epoch
    n: int = 0
    sums: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in ANALOG})
    mins: Dict[str, float] = field(default_factory=lambda: {c: np.inf for c in ANALOG})
    maxs: Dict[str, float] = field(default_factory=lambda: {c: -np.inf for c in ANALOG})
    loaded_sum: float = 0.0
    load_starts: float = 0.0
    run_elapsed_max: float = 0.0
    idle_dp_sum: float = 0.0
    idle_dt_sum: float = 0.0
    current_loaded_sum: float = 0.0
    dp_loaded_sum: float = 0.0
    LPS_sum: float = 0.0


class _AlertRule:
    """Streaming twin of alerts.generate_alerts for one score stream."""

    def __init__(self, threshold: float, smoothing_minutes: int, persistence_minutes: int):
        self.threshold = threshold
        self.smooth_ns = smoothing_minutes * MINUTE_NS
        self.persistence = persistence_minutes
        self.window: Deque[Tuple[int, float]] = deque()
        self.prev_t: Optional[int] = None
        self.prev_above = False
        self.run_start: Optional[int] = None
        self.active = False

    def update(self, t: int, score: float) -> Tuple[float, bool, Optional[str]]:
        self.window.append((t, score))
        while self.window[0][0] <= t - self.smooth_ns:
            self.window.popleft()
        smoothed = float(np.median([s for _, s in self.window]))
        above = smoothed > self.threshold
        gap_min = np.inf if self.prev_t is None else (t - self.prev_t) / MINUTE_NS
        if above and not (self.prev_above and gap_min <= MAX_GAP_MINUTES):
            self.run_start = t
        elapsed = (t - self.run_start) / MINUTE_NS + 1.0 if above else 0.0
        flag = above and elapsed >= self.persistence
        event = "start" if flag and not self.active else ("end" if self.active and not flag else None)
        self.active, self.prev_above, self.prev_t = flag, above, t
        return smoothed, flag, event


class StreamState:
    def __init__(
        self,
        baseline: Dict[str, Tuple[float, float]],
        detectors: Optional[Dict[str, object]] = None,
        alert_params: Optional[Dict] = None,
        shifts: List[Dict] = DEFAULT_SHIFTS,
    ):
        self.baseline = baseline
        self.detectors = detectors or {}
        self.shifts = shifts
        a = alert_params or {"smoothing_window_minutes": 30, "persistence_minutes": 60}
        self.rules = {name: _AlertRule(float(d.threshold), a["smoothing_window_minutes"], a["persistence_minutes"])
                      for name, d in self.detectors.items()}
        # raw-row state
        self.prev_ts: Optional[float] = None
        self.prev_loaded: Optional[int] = None
        self.prev_res: Optional[float] = None
        self.run_start: Optional[float] = None
        # minute state
        self.bucket: Optional[_Bucket] = None
        self.buffer: Deque[Dict[str, float]] = deque()
        self.frozen_minutes = 0

    # ---------------------------------------------------------------- raw rows
    def update(self, row: Dict) -> Optional[Dict]:
        """Ingest one raw 10 s row. Returns the scored minute when a minute closes, else None."""
        t_ns = pd.Timestamp(row["timestamp"]).value
        ts = t_ns / 1e9
        minute = t_ns - (t_ns % MINUTE_NS)
        out = None
        if self.bucket is not None and minute != self.bucket.ts:
            out = self._close_bucket()
        if self.bucket is None:
            self.bucket = _Bucket(ts=minute)

        loaded = 1 - int(row["COMP"])
        res = float(np.float32(row["Reservoirs"]))
        dt = np.nan if self.prev_ts is None else ts - self.prev_ts
        contiguous = (not np.isnan(dt)) and dt <= MAX_GAP_SECONDS
        prev_loaded = -1 if self.prev_loaded is None else self.prev_loaded

        load_start = loaded == 1 and prev_loaded == 0 and contiguous
        if loaded == 1 and not (prev_loaded == 1 and contiguous):
            self.run_start = ts
        run_elapsed = ts - self.run_start if loaded == 1 else 0.0
        idle_ok = loaded == 0 and prev_loaded == 0 and contiguous
        idle_dp = (self.prev_res - res) if idle_ok else 0.0
        idle_dt = dt if idle_ok else 0.0

        b = self.bucket
        b.n += 1
        vals = {c: float(np.float32(row[c])) for c in ANALOG}   # match the float32 parquet
        for c, v in vals.items():
            b.sums[c] += v
            b.mins[c] = min(b.mins[c], v)
            b.maxs[c] = max(b.maxs[c], v)
        b.loaded_sum += loaded
        b.load_starts += float(load_start)
        b.run_elapsed_max = max(b.run_elapsed_max, run_elapsed)
        b.idle_dp_sum += idle_dp
        b.idle_dt_sum += idle_dt
        b.current_loaded_sum += vals["Motor_current"] * loaded
        b.dp_loaded_sum += (vals["TP2"] - vals["TP3"]) * loaded
        b.LPS_sum += float(row["LPS"])

        self.prev_ts, self.prev_loaded, self.prev_res = ts, loaded, res
        return out

    def flush(self) -> Optional[Dict]:
        """Close the minute in progress (end of replay)."""
        out = self._close_bucket() if self.bucket is not None else None
        self.bucket = None
        return out

    # ---------------------------------------------------------------- minutes
    def _close_bucket(self) -> Optional[Dict]:
        b = self.bucket
        self.bucket = None
        rng = max(b.maxs[c] - b.mins[c] for c in ANALOG)
        if b.n >= MIN_ROWS_FOR_FROZEN and rng == 0:
            self.frozen_minutes += 1
            return None
        rec = {"ts": b.ts, "n": float(b.n), **{f"{c}_sum": b.sums[c] for c in ANALOG},
               "Reservoirs_min": b.mins["Reservoirs"], "run_elapsed_max": b.run_elapsed_max,
               "loaded_sum": b.loaded_sum, "load_starts": b.load_starts, "idle_dp_sum": b.idle_dp_sum,
               "idle_dt_sum": b.idle_dt_sum, "current_loaded_sum": b.current_loaded_sum,
               "dp_loaded_sum": b.dp_loaded_sum, "LPS_sum": b.LPS_sum}
        self.buffer.append(rec)
        while self.buffer[0]["ts"] <= b.ts - WINDOWS_MIN["24h"] * MINUTE_NS:
            self.buffer.popleft()
        return self._score(self._features(b.ts))

    def _features(self, t: int) -> Dict[str, float]:
        buf = pd.DataFrame(self.buffer)
        ts = buf["ts"].to_numpy()

        def win(minutes: int) -> pd.DataFrame:
            return buf[ts > t - minutes * MINUTE_NS]

        def ratio(w: pd.DataFrame, num: str, den: str) -> float:
            d = w[den].sum()
            return float(w[num].sum() / d) if d > 0 else np.nan

        def std(x: np.ndarray) -> float:
            return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan

        cur = buf.iloc[-1]
        w15, w60, w6h, w24 = win(15), win(60), win(360), win(1440)
        f = {c: float(cur[f"{c}_sum"] / cur["n"]) for c in ANALOG}
        f["loaded_frac"] = float(cur["loaded_sum"] / cur["n"])
        f["duty_cycle_15m"] = ratio(w15, "loaded_sum", "n")
        f["duty_cycle_60m"] = ratio(w60, "loaded_sum", "n")
        f["duty_cycle_6h"] = ratio(w6h, "loaded_sum", "n")
        f["load_starts_per_hour"] = float(w60["load_starts"].sum())
        f["max_load_run_min_60m"] = float(w60["run_elapsed_max"].max()) / 60.0
        f["motor_current_mean_15m"] = ratio(w15, "Motor_current_sum", "n")
        f["motor_current_std_15m"] = std((w15["Motor_current_sum"] / w15["n"]).to_numpy())
        f["motor_current_mean_60m"] = ratio(w60, "Motor_current_sum", "n")
        f["motor_current_loaded_60m"] = ratio(w60, "current_loaded_sum", "loaded_sum")
        f["oil_temp_mean_15m"] = ratio(w15, "Oil_temperature_sum", "n")
        f["oil_temp_mean_60m"] = ratio(w60, "Oil_temperature_sum", "n")
        f["oil_temp_std_60m"] = std((w60["Oil_temperature_sum"] / w60["n"]).to_numpy())
        f["oil_temp_excess_24h"] = f["oil_temp_mean_60m"] - ratio(w24, "Oil_temperature_sum", "n")
        f["delta_p_loaded_60m"] = ratio(w60, "dp_loaded_sum", "loaded_sum")
        f["reservoir_mean_15m"] = ratio(w15, "Reservoirs_sum", "n")
        f["reservoir_min_60m"] = float(w60["Reservoirs_min"].min())
        idle = ratio(w60, "idle_dp_sum", "idle_dt_sum")
        f["idle_decay_rate_60m"] = idle * 60.0 if np.isfinite(idle) else np.nan
        f["lps_frac_60m"] = ratio(w60, "LPS_sum", "n")
        f = {k: f[k] for k in FEATURE_COLUMNS}
        f["n_samples"] = float(cur["n"])
        for col, z in DRIFT_TARGETS:
            mu, sd = self.baseline[col]
            f[z] = (f[col] - mu) / sd
        stamp = pd.Timestamp(t)
        f["timestamp"] = stamp
        f["shift"] = shift_label(stamp.hour, self.shifts)
        return f

    def _score(self, f: Dict) -> Dict:
        if not self.detectors:
            return f
        row = pd.DataFrame([f])
        t = pd.Timestamp(f["timestamp"]).value
        f["events"] = []
        for name, det in self.detectors.items():
            s = float(det.score_samples(row)[0])
            smoothed, flag, event = self.rules[name].update(t, s)
            f[f"score_{name}"] = s
            f[f"smoothed_{name}"] = smoothed
            f[f"alert_{name}"] = flag
            if event:
                f["events"].append((name, event))
        return f
