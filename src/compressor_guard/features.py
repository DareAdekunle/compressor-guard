"""
MetroPT-3 domain features, operating shifts and healthy-baseline drift.

Pipeline (identical in batch here and incrementally in `stream/state.py`):

1. **Raw-row signals** (10 s rows). Each needs only the previous row and the current
   load-run start, so it can be computed causally:
   - `loaded = 1 - COMP`. Per the dataset docs, COMP is *active when there is no air
     intake* (compressor off or offloaded), so the compressor produces air when COMP == 0.
   - load starts (0 -> 1 transitions of `loaded`)
   - elapsed time of the current loaded run (an air leak makes the compressor run
     longer, or never stop, to hold pressure)
   - reservoir pressure drop while unloaded (idle decay, a direct leak signature)
2. **Minute buckets.** Sums, counts, min and max per calendar minute. Minutes where every
   analog signal is perfectly flat are marked *frozen* (logger outage) and dropped.
3. **Trailing windows** over the minute buckets `(t - W, t]` give duty cycle, rolling
   means and standard deviations, and rates. Nothing looks ahead: no centred windows,
   no back-fill.
4. **Drift z-scores** against statistics from the healthy training window only.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from compressor_guard.config import load_params, resolve_path

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]

# Two consecutive rows further apart than this are not treated as continuous
# (no load-start counted, load run and idle decay broken).
MAX_GAP_SECONDS = 60.0
# A minute is frozen when it has at least this many rows and every analog signal is constant.
MIN_ROWS_FOR_FROZEN = 3

# Per-minute sum/extreme columns produced by `aggregate_minutes` and the streaming state.
BUCKET_COLUMNS = (
    ["n"]
    + [f"{c}_sum" for c in ANALOG]
    + [
        "Motor_current_max",
        "Reservoirs_min",
        "Reservoirs_max",
        "loaded_sum",
        "load_starts",
        "run_elapsed_max",
        "idle_dp_sum",
        "idle_dt_sum",
        "current_loaded_sum",
        "dp_loaded_sum",
        "LPS_sum",
        "analog_range_max",
    ]
)

# Engineered features, in a fixed order, that models may use.
FEATURE_COLUMNS = [
    # minute means of the raw analog signals (used by the LSTM autoencoder)
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
    "loaded_frac",
    # duty cycle and cycling
    "duty_cycle_15m", "duty_cycle_60m", "duty_cycle_6h",
    "load_starts_per_hour",
    "max_load_run_min_60m",
    # electrical
    "motor_current_mean_15m", "motor_current_std_15m", "motor_current_mean_60m",
    "motor_current_loaded_60m",
    # thermal
    "oil_temp_mean_15m", "oil_temp_mean_60m", "oil_temp_std_60m", "oil_temp_excess_24h",
    # pneumatic
    "delta_p_loaded_60m", "reservoir_mean_15m", "reservoir_min_60m",
    "idle_decay_rate_60m", "lps_frac_60m",
]

# (source feature, drift column) pairs standardised against the healthy baseline.
DRIFT_TARGETS = [
    ("duty_cycle_60m", "duty_cycle_drift_z"),
    ("motor_current_mean_60m", "motor_current_drift_z"),
    ("oil_temp_mean_60m", "oil_temp_drift_z"),
    ("idle_decay_rate_60m", "idle_decay_drift_z"),
]
DRIFT_COLUMNS = [z for _, z in DRIFT_TARGETS]

DEFAULT_SHIFTS = [
    {"label": "Shift 1 (06-14)", "start_hour": 6, "end_hour": 14},
    {"label": "Shift 2 (14-22)", "start_hour": 14, "end_hour": 22},
    {"label": "Shift 3 (22-06)", "start_hour": 22, "end_hour": 6},
]


# --------------------------------------------------------------------------------------
# Shifts
# --------------------------------------------------------------------------------------
def shifts_from_params(params: Optional[Dict] = None) -> List[Dict]:
    if not params:
        return DEFAULT_SHIFTS
    shifts = params.get("metropt", {}).get("shifts")
    return list(shifts.values()) if shifts else DEFAULT_SHIFTS


def shift_label(hour: int, shifts: List[Dict] = DEFAULT_SHIFTS) -> str:
    for s in shifts:
        a, b = s["start_hour"], s["end_hour"]
        if (a < b and a <= hour < b) or (a > b and (hour >= a or hour < b)):
            return s["label"]
    return "Unassigned"


def assign_operating_shifts(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    shifts: List[Dict] = DEFAULT_SHIFTS,
) -> pd.DataFrame:
    """
    Add `shift`, `hour`, `day_of_week` and `is_weekend`. The 3 x 8 h schedule is an
    **assumption** set in params.yaml. MetroPT-3 does not record shifts.
    """
    df = df.copy()
    hour = df[timestamp_col].dt.hour
    lookup = {h: shift_label(h, shifts) for h in range(24)}
    labels = [s["label"] for s in shifts]
    df["shift"] = pd.Categorical(hour.map(lookup), categories=labels, ordered=True)
    df["hour"] = hour.astype(np.int8)
    df["day_of_week"] = df[timestamp_col].dt.day_name().astype("category")
    df["is_weekend"] = df[timestamp_col].dt.weekday >= 5
    return df


# --------------------------------------------------------------------------------------
# Step 1: raw-row signals
# --------------------------------------------------------------------------------------
def derive_raw_signals(df: pd.DataFrame, max_gap_s: float = MAX_GAP_SECONDS) -> pd.DataFrame:
    """Add the causal per-row signals described in the module docstring."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    loaded = (1 - df["COMP"].to_numpy()).astype(np.int8)
    res = df["Reservoirs"].to_numpy(dtype=np.float64)

    dt = np.empty_like(ts)
    dt[0] = np.nan
    dt[1:] = np.diff(ts)
    prev_loaded = np.empty_like(loaded)
    prev_loaded[0] = -1
    prev_loaded[1:] = loaded[:-1]
    prev_res = np.empty_like(res)
    prev_res[0] = np.nan
    prev_res[1:] = res[:-1]

    contiguous = np.nan_to_num(dt, nan=np.inf) <= max_gap_s

    load_start = (loaded == 1) & (prev_loaded == 0) & contiguous
    new_run = (loaded == 1) & ~((prev_loaded == 1) & contiguous)
    run_start = pd.Series(np.where(new_run, ts, np.nan)).ffill().to_numpy()
    run_elapsed = np.where(loaded == 1, ts - run_start, 0.0)

    idle_ok = (loaded == 0) & (prev_loaded == 0) & contiguous
    idle_dp = np.where(idle_ok, prev_res - res, 0.0)   # positive = pressure lost
    idle_dt = np.where(idle_ok, dt, 0.0)

    out = df.copy()
    out["loaded"] = loaded
    out["load_start"] = load_start.astype(np.int8)
    out["run_elapsed"] = run_elapsed
    out["idle_dp"] = idle_dp
    out["idle_dt"] = idle_dt
    return out


# --------------------------------------------------------------------------------------
# Step 2: minute buckets
# --------------------------------------------------------------------------------------
def aggregate_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw rows (with `derive_raw_signals` columns) into one row per non-empty minute."""
    r = raw
    work = pd.DataFrame({"bucket": r["timestamp"].dt.floor("1min")})
    for c in ANALOG:
        work[c] = r[c].astype(np.float64)
    work["loaded"] = r["loaded"].astype(np.float64)
    work["load_start"] = r["load_start"].astype(np.float64)
    work["run_elapsed"] = r["run_elapsed"]
    work["idle_dp"] = r["idle_dp"]
    work["idle_dt"] = r["idle_dt"]
    work["current_loaded"] = work["Motor_current"] * work["loaded"]
    work["dp_loaded"] = (work["TP2"] - work["TP3"]) * work["loaded"]
    work["LPS"] = r["LPS"].astype(np.float64)

    g = work.groupby("bucket", sort=True)
    sums = g[ANALOG + ["loaded", "load_start", "idle_dp", "idle_dt",
                       "current_loaded", "dp_loaded", "LPS"]].sum()
    mins = g[ANALOG].min()
    maxs = g[ANALOG].max()

    b = pd.DataFrame(index=sums.index)
    b["n"] = g.size().astype(np.float64)
    for c in ANALOG:
        b[f"{c}_sum"] = sums[c]
    b["Motor_current_max"] = maxs["Motor_current"]
    b["Reservoirs_min"] = mins["Reservoirs"]
    b["Reservoirs_max"] = maxs["Reservoirs"]
    b["loaded_sum"] = sums["loaded"]
    b["load_starts"] = sums["load_start"]
    b["run_elapsed_max"] = g["run_elapsed"].max()
    b["idle_dp_sum"] = sums["idle_dp"]
    b["idle_dt_sum"] = sums["idle_dt"]
    b["current_loaded_sum"] = sums["current_loaded"]
    b["dp_loaded_sum"] = sums["dp_loaded"]
    b["LPS_sum"] = sums["LPS"]
    b["analog_range_max"] = (maxs - mins).max(axis=1)
    b.index.name = "timestamp"
    return b.reset_index()


def is_frozen_bucket(n: float, analog_range_max: float) -> bool:
    return n >= MIN_ROWS_FOR_FROZEN and analog_range_max == 0


# --------------------------------------------------------------------------------------
# Step 3: trailing-window features
# --------------------------------------------------------------------------------------
def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(np.float64)


def window_features(buckets: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FEATURE_COLUMNS from valid (non-frozen) minute buckets using trailing
    time-based windows `(t - W, t]`. `buckets` must be sorted by timestamp.
    """
    b = buckets.set_index("timestamp")
    out = pd.DataFrame(index=b.index)

    for c in ANALOG:
        out[c] = b[f"{c}_sum"] / b["n"]
    out["loaded_frac"] = b["loaded_sum"] / b["n"]

    r15, r60, r6h, r24h = (b.rolling(w) for w in ("15min", "60min", "360min", "1440min"))
    s15, s60, s6h, s24h = r15.sum(), r60.sum(), r6h.sum(), r24h.sum()

    out["duty_cycle_15m"] = _ratio(s15["loaded_sum"], s15["n"])
    out["duty_cycle_60m"] = _ratio(s60["loaded_sum"], s60["n"])
    out["duty_cycle_6h"] = _ratio(s6h["loaded_sum"], s6h["n"])
    out["load_starts_per_hour"] = s60["load_starts"]
    out["max_load_run_min_60m"] = b["run_elapsed_max"].rolling("60min").max() / 60.0

    out["motor_current_mean_15m"] = _ratio(s15["Motor_current_sum"], s15["n"])
    out["motor_current_std_15m"] = out["Motor_current"].rolling("15min").std()
    out["motor_current_mean_60m"] = _ratio(s60["Motor_current_sum"], s60["n"])
    out["motor_current_loaded_60m"] = _ratio(s60["current_loaded_sum"], s60["loaded_sum"])

    out["oil_temp_mean_15m"] = _ratio(s15["Oil_temperature_sum"], s15["n"])
    out["oil_temp_mean_60m"] = _ratio(s60["Oil_temperature_sum"], s60["n"])
    out["oil_temp_std_60m"] = out["Oil_temperature"].rolling("60min").std()
    # Oil temperature relative to its own trailing 24 h level. This removes the slow
    # seasonal/ambient trend (Feb vs Jul) that would otherwise look like drift.
    out["oil_temp_excess_24h"] = out["oil_temp_mean_60m"] - _ratio(s24h["Oil_temperature_sum"], s24h["n"])

    out["delta_p_loaded_60m"] = _ratio(s60["dp_loaded_sum"], s60["loaded_sum"])
    out["reservoir_mean_15m"] = _ratio(s15["Reservoirs_sum"], s15["n"])
    out["reservoir_min_60m"] = b["Reservoirs_min"].rolling("60min").min()
    out["idle_decay_rate_60m"] = _ratio(s60["idle_dp_sum"], s60["idle_dt_sum"]) * 60.0  # bar/min
    out["lps_frac_60m"] = _ratio(s60["LPS_sum"], s60["n"])
    return out[FEATURE_COLUMNS].reset_index()


# --------------------------------------------------------------------------------------
# Step 4: baseline + drift
# --------------------------------------------------------------------------------------
def fit_baseline(features: pd.DataFrame, train_end: Union[str, pd.Timestamp],
                 train_start: Optional[Union[str, pd.Timestamp]] = None) -> Dict[str, Tuple[float, float]]:
    """Mean/std of drift source columns on the healthy training window only."""
    mask = features["timestamp"] <= pd.Timestamp(train_end)
    if train_start is not None:
        mask &= features["timestamp"] >= pd.Timestamp(train_start)
    healthy = features.loc[mask]
    stats = {}
    for col, _ in DRIFT_TARGETS:
        mu = float(healthy[col].mean())
        sd = float(healthy[col].std())
        stats[col] = (mu, sd if sd > 1e-9 else 1.0)
    return stats


def apply_drift(features: pd.DataFrame, baseline: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = features.copy()
    for col, z in DRIFT_TARGETS:
        mu, sd = baseline[col]
        out[z] = (out[col] - mu) / sd
    return out


def save_baseline(baseline: Dict, path: Union[str, Path] = "models/feature_baseline.json") -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: list(v) for k, v in baseline.items()}, indent=2))
    return p


def load_baseline(path: Union[str, Path] = "models/feature_baseline.json") -> Dict[str, Tuple[float, float]]:
    return {k: tuple(v) for k, v in json.loads(resolve_path(path).read_text()).items()}


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------
def build_feature_table(
    df_raw: pd.DataFrame,
    params: Optional[Dict] = None,
    baseline: Optional[Dict[str, Tuple[float, float]]] = None,
    failures: Optional[List[Dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Raw 10 s telemetry -> 1-minute feature table (frozen minutes removed) with shifts,
    drift z-scores and failure labels. Returns (features, baseline_stats).

    If `baseline` is None it is fitted on params.metropt.train_split, which ends
    before the first failure.
    """
    params = params or load_params()
    split = params["metropt"]["train_split"]

    raw = derive_raw_signals(df_raw)
    buckets = aggregate_minutes(raw)
    frozen = (buckets["n"] >= MIN_ROWS_FOR_FROZEN) & (buckets["analog_range_max"] == 0)
    valid = buckets.loc[~frozen].reset_index(drop=True)

    feats = window_features(valid)
    feats["n_samples"] = valid["n"].values
    if baseline is None:
        baseline = fit_baseline(feats, split["end"], split.get("start"))
    feats = apply_drift(feats, baseline)
    feats = assign_operating_shifts(feats, shifts=shifts_from_params(params))

    from compressor_guard.data import add_failure_labels  # local import avoids a cycle
    feats = add_failure_labels(feats, failures=failures) if failures is not None else add_failure_labels(feats)
    feats.attrs["frozen_minutes"] = int(frozen.sum())
    return feats, baseline


def save_features_parquet(df: pd.DataFrame,
                          output_path: Union[str, Path] = "data/processed/metropt_features.parquet") -> Path:
    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    return out


def load_features_parquet(path: Union[str, Path] = "data/processed/metropt_features.parquet") -> pd.DataFrame:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Features parquet not found at: {p}. Run `python -m compressor_guard.features` first.")
    return pd.read_parquet(p, engine="pyarrow")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build MetroPT-3 minute-level features.")
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--refresh", action="store_true", help="rebuild the clean parquet from the raw CSV")
    args = ap.parse_args(argv)

    from compressor_guard.data import load_labelled_metropt

    params = load_params(args.config)
    df = load_labelled_metropt(params, refresh=args.refresh)
    feats, baseline = build_feature_table(df, params)
    out = save_features_parquet(feats, params["metropt"]["processed_parquet_path"])
    bpath = save_baseline(baseline, params["metropt"].get("baseline_path", "models/feature_baseline.json"))
    print(f"[features] {len(feats):,} minute rows, {feats.attrs['frozen_minutes']:,} frozen minutes dropped")
    print(f"[features] saved {out}\n[features] baseline -> {bpath}")


if __name__ == "__main__":
    main()
