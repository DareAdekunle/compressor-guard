import numpy as np
import pandas as pd

from compressor_guard.alerts import apply_persistence_filter, generate_alerts, smooth_anomaly_scores


def test_persistence_counts_steps():
    x = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    assert apply_persistence_filter(x, 3).tolist() == [0, 0, 0, 1, 0, 0, 0, 1, 1]


def test_persistence_resets_on_time_gap():
    ts = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 03:00", "2020-01-01 03:01"])
    out = apply_persistence_filter(np.ones(4, bool), 3, timestamps=ts)
    assert out.tolist() == [False, False, False, False]


def test_smoothing_is_trailing():
    s = np.r_[np.zeros(10), 10.0, np.zeros(10)]
    sm = smooth_anomaly_scores(s, 5)
    assert sm[:10].max() == 0       # a future spike never leaks backwards


def test_generate_alerts_events():
    ts = pd.date_range("2020-01-01", periods=300, freq="1min")
    df = pd.DataFrame({"timestamp": ts, "shift": "Shift 1 (06-14)"})
    raw = np.zeros(300)
    raw[100:250] = 5.0
    out, ev = generate_alerts(df, raw, threshold=1.0, smoothing_window=10, persistence_steps=30)
    assert len(ev) == 1
    first = out.loc[out["alert_flag"], "timestamp"].iloc[0]
    # smoothing (median of 10) needs 5 high samples, then 30 min persistence
    assert first == ts[100 + 5 + 29 - 1] or first == ts[100 + 5 + 29]
