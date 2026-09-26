"""
Data acquisition, parsing, validation, and ground-truth labeling for MetroPT-3.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
import numpy as np
import pandas as pd

from compressor_guard.config import resolve_path


ANALOG_COLS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

DIGITAL_COLS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


def load_failures(config_path: Union[str, Path] = "config/failures.yaml") -> List[Dict]:
    """
    Load documented failure intervals from configuration YAML.
    """
    config_file = resolve_path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Failures config not found at: {config_file}")

    with open(config_file, "r") as f:
        data = yaml.safe_load(f)

    failures = []
    for item in data.get("failures", []):
        failures.append({
            "id": item["id"],
            "start": pd.to_datetime(item["start"]),
            "end": pd.to_datetime(item["end"]),
            "type": item.get("type", "air_leak"),
            "description": item.get("description", ""),
            # Time the operator logged the repair (UCI failure report). Alerts between
            # failure end and this time are the same incident, not false alarms.
            "maintenance": pd.to_datetime(item["maintenance"]) if item.get("maintenance") else None,
        })
    return failures


def load_raw_metropt(
    csv_path: Union[str, Path] = "data/raw/metropt3/MetroPT3(AirCompressor).csv",
    nrows: Optional[int] = None,
    optimize_types: bool = True,
) -> pd.DataFrame:
    """
    Load and parse the raw MetroPT-3 CSV dataset.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the MetroPT3 CSV file.
    nrows : int, optional
        Limit number of rows read (useful for testing and previews).
    optimize_types : bool
        Downcast numeric columns to float32/int8 to conserve memory.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed or ordered by timestamp.
    """
    path = resolve_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"MetroPT-3 CSV not found at: {path}")

    # Inspect columns to ignore Unnamed: 0 if present
    df = pd.read_csv(path, nrows=nrows)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if optimize_types:
        for col in ANALOG_COLS:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        for col in DIGITAL_COLS:
            if col in df.columns:
                df[col] = df[col].round().astype(np.int8)

    return df


def audit_sampling_gaps(df: pd.DataFrame, threshold_seconds: float = 60.0) -> pd.DataFrame:
    """
    Identify periods where sensor telemetry sampling interval exceeds threshold_seconds.
    
    Returns DataFrame with gap_start, gap_end, gap_seconds.
    """
    dt = df["timestamp"].diff().dt.total_seconds()
    gap_indices = dt[dt > threshold_seconds].index

    gaps = []
    for idx in gap_indices:
        gaps.append({
            "gap_start": df.loc[idx - 1, "timestamp"],
            "gap_end": df.loc[idx, "timestamp"],
            "gap_seconds": dt.loc[idx],
            "gap_hours": dt.loc[idx] / 3600.0,
        })
    return pd.DataFrame(gaps)


def audit_sensor_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit physical sensor ranges, quantiles, missing rates, and basic statistics.
    """
    records = []
    cols = [c for c in ANALOG_COLS + DIGITAL_COLS if c in df.columns]
    for col in cols:
        s = df[col]
        records.append({
            "signal": col,
            "type": "analog" if col in ANALOG_COLS else "digital",
            "missing_count": s.isna().sum(),
            "missing_pct": (s.isna().sum() / len(s)) * 100.0,
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "std": float(s.std()),
        })
    return pd.DataFrame(records)


def audit_frozen_periods(
    df: pd.DataFrame,
    min_samples_per_minute: int = 3,
    min_duration_minutes: int = 10,
) -> pd.DataFrame:
    """
    Find periods where the logger kept writing rows but every analog signal was stuck
    at a constant value. A live compressor never holds pressure, temperature and current
    perfectly flat for a whole minute, so these are data-quality outages, not machine
    behaviour, and must be masked before training or scoring.

    Uses the same per-minute rule as the feature pipeline (see features.MINUTE_FROZEN_RULE).
    """
    cols = [c for c in ANALOG_COLS if c in df.columns]
    g = df.set_index("timestamp")[cols].resample("1min")
    rng = (g.max() - g.min()).max(axis=1)
    n = g.count().iloc[:, 0]
    frozen = (rng == 0) & (n >= min_samples_per_minute)
    run_id = (frozen != frozen.shift()).cumsum()
    periods = (
        pd.DataFrame({"t": frozen.index, "frozen": frozen.values, "run": run_id.values})
        .loc[lambda d: d["frozen"]]
        .groupby("run")["t"].agg(["min", "max", "size"])
        .rename(columns={"min": "start", "max": "end", "size": "minutes"})
    )
    periods = periods[periods["minutes"] >= min_duration_minutes]
    return periods.sort_values("start").reset_index(drop=True)


def add_failure_labels(
    df: pd.DataFrame,
    failures: Optional[List[Dict]] = None,
    config_path: Union[str, Path] = "config/failures.yaml",
) -> pd.DataFrame:
    """
    Add boolean 'is_failure' and string 'failure_id' columns indicating
    whether each timestamp falls inside documented failure windows.
    """
    if failures is None:
        failures = load_failures(config_path)

    df = df.copy()
    df["is_failure"] = False
    df["failure_id"] = "healthy"

    ts = df["timestamp"]
    for f in failures:
        mask = (ts >= f["start"]) & (ts <= f["end"])
        df.loc[mask, "is_failure"] = True
        df.loc[mask, "failure_id"] = f["id"]

    return df


def save_clean_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "data/processed/metropt_clean.parquet",
) -> Path:
    """
    Save cleaned and type-optimized telemetry to Parquet for fast I/O.
    """
    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    return out


def load_clean_parquet(
    path: Union[str, Path] = "data/processed/metropt_clean.parquet",
) -> pd.DataFrame:
    """
    Load cleaned telemetry from Parquet.
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Clean parquet not found at: {p}")
    return pd.read_parquet(p, engine="pyarrow")


def load_labelled_metropt(
    params: Optional[Dict] = None,
    failures_path: Union[str, Path] = "config/failures.yaml",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Return the labelled 10 s telemetry, building the clean parquet cache from the raw
    CSV on first use (or when `refresh=True`). This is what the CLI pipeline calls, so
    `python -m compressor_guard.features` works on a fresh clone without notebook 01.
    """
    params = params or {}
    m = params.get("metropt", {})
    clean_path = resolve_path(m.get("clean_parquet_path", "data/processed/metropt_clean.parquet"))
    if clean_path.exists() and not refresh:
        return load_clean_parquet(clean_path)
    df = load_raw_metropt(m.get("raw_csv_path", "data/raw/metropt3/MetroPT3(AirCompressor).csv"))
    df = add_failure_labels(df, config_path=failures_path)
    save_clean_parquet(df, clean_path)
    return df
