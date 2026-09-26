import numpy as np
import pandas as pd
import pytest

from compressor_guard.config import resolve_path


def make_raw(hours: float = 36.0, seed: int = 0, start: str = "2020-03-01 00:00:00",
             frozen: tuple = (10.0, 11.0), gap: tuple = (20.0, 20.5), leak_from: float = None) -> pd.DataFrame:
    """
    Synthetic 10 s compressor telemetry with realistic MetroPT-3 semantics:
    COMP = 1 means *not* loading. The compressor loads for ~2 of every ~12 min, pressure
    rises while loaded and decays while idle. Optional frozen-logger hour, data gap,
    and a leak (faster decay, longer loading) from `leak_from` hours.
    """
    rng = np.random.default_rng(seed)
    n = int(hours * 360)
    ts = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * 10, unit="s")
    res = np.empty(n)
    comp = np.empty(n, dtype=int)
    p, loading = 8.5, False
    for i in range(n):
        h = i / 360
        leak = leak_from is not None and h >= leak_from
        decay = 0.004 * (4 if leak else 1)
        if loading:
            p += 0.02
            if p >= 9.5:
                loading = False
        else:
            p -= decay * rng.uniform(0.5, 1.5)
            if p <= 8.2:
                loading = True
        res[i] = p
        comp[i] = 0 if loading else 1
    loaded = 1 - comp
    df = pd.DataFrame({
        "timestamp": ts,
        "TP2": np.where(loaded == 1, res + 0.1, 0.0) + rng.normal(0, 0.005, n),
        "TP3": res + rng.normal(0, 0.005, n),
        "H1": res - 0.02 + rng.normal(0, 0.005, n),
        "DV_pressure": np.where(loaded == 1, 0.0, 0.02) + rng.normal(0, 0.002, n),
        "Reservoirs": res,
        "Oil_temperature": 60 + 10 * loaded + rng.normal(0, 0.3, n),
        "Motor_current": np.where(loaded == 1, 5.8, 0.04) + rng.normal(0, 0.05, n),
        "COMP": comp, "DV_eletric": loaded, "Towers": 1, "MPG": comp, "LPS": 0,
        "Pressure_switch": 1, "Oil_level": 1, "Caudal_impulses": 1,
    })
    for c in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]:
        df[c] = df[c].astype(np.float32)
    h = np.arange(n) / 360
    if frozen:
        m = (h >= frozen[0]) & (h < frozen[1])
        first = np.argmax(m)
        for c in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]:
            df.loc[m, c] = df.loc[first, c]
    if gap:
        df = df[~((h >= gap[0]) & (h < gap[1]))]
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_synth():
    return make_raw()


@pytest.fixture(scope="session")
def params():
    from compressor_guard.config import load_params
    p = load_params()
    p["metropt"]["train_split"] = {"start": "2020-03-01 00:00:00", "end": "2020-03-01 23:59:00"}
    p["metropt"]["train_exclude"] = []
    return p


def real_data_available() -> bool:
    return (resolve_path("data/processed/metropt_clean.parquet").exists()
            or resolve_path("data/raw/metropt3/MetroPT3(AirCompressor).csv").exists())


def models_available() -> bool:
    return (resolve_path("models/isolation_forest.joblib").exists()
            and resolve_path("models/feature_baseline.json").exists())


requires_data = pytest.mark.skipif(not real_data_available(), reason="MetroPT-3 data not downloaded")
requires_models = pytest.mark.skipif(not (real_data_available() and models_available()),
                                     reason="trained models not found (run compressor_guard.models --train)")
