import numpy as np
import pandas as pd

from compressor_guard.features import (FEATURE_COLUMNS, aggregate_minutes, build_feature_table,
                                       derive_raw_signals, shift_label, window_features)
from conftest import make_raw


def test_loaded_is_inverse_of_comp(raw_synth):
    r = derive_raw_signals(raw_synth)
    assert ((r["loaded"] == 1) == (raw_synth["COMP"] == 0)).all()
    # loaded rows draw motor current, idle rows do not (MetroPT-3 semantics)
    assert r.loc[r["loaded"] == 1, "Motor_current"].mean() > 5
    assert r.loc[r["loaded"] == 0, "Motor_current"].mean() < 1


def test_run_elapsed_and_starts():
    ts = pd.date_range("2020-01-01", periods=8, freq="10s")
    df = pd.DataFrame({"timestamp": ts, "COMP": [1, 0, 0, 0, 1, 0, 0, 1], "Reservoirs": np.linspace(9, 8, 8)})
    r = derive_raw_signals(df)
    assert r["load_start"].tolist() == [0, 1, 0, 0, 0, 1, 0, 0]
    assert r["run_elapsed"].tolist() == [0, 0, 10, 20, 0, 0, 10, 0]


def test_gap_breaks_run_and_idle_decay():
    ts = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:10", "2020-01-01 01:00:00", "2020-01-01 01:00:10"])
    df = pd.DataFrame({"timestamp": ts, "COMP": [0, 0, 0, 1], "Reservoirs": [8.0, 8.1, 8.2, 8.3]})
    r = derive_raw_signals(df)
    assert r["run_elapsed"].tolist() == [0, 10, 0, 0]       # run restarts after the 1 h gap
    assert r["load_start"].sum() == 0                          # no start counted across a gap


def test_frozen_minutes_dropped(raw_synth, params):
    feats, _ = build_feature_table(raw_synth, params, failures=[])
    assert feats.attrs["frozen_minutes"] >= 55
    hours = (feats["timestamp"] - feats["timestamp"].min()).dt.total_seconds() / 3600
    assert not ((hours > 10.05) & (hours < 10.95)).any()


def test_features_are_causal(raw_synth, params):
    """Appending future data must not change past feature values."""
    full, base = build_feature_table(raw_synth, params, failures=[])
    cut = raw_synth[raw_synth["timestamp"] < raw_synth["timestamp"].min() + pd.Timedelta(hours=8)]
    part, _ = build_feature_table(cut, params, baseline=base, failures=[])
    m = part.merge(full, on="timestamp", suffixes=("_p", "_f"))
    for c in FEATURE_COLUMNS:
        np.testing.assert_allclose(m[f"{c}_p"], m[f"{c}_f"], rtol=1e-9, atol=1e-9, err_msg=c)


def test_leak_raises_duty_and_decay(params):
    raw = make_raw(hours=30, frozen=None, gap=None, leak_from=24)
    feats, _ = build_feature_table(raw, params, failures=[])
    h = (feats["timestamp"] - feats["timestamp"].min()).dt.total_seconds() / 3600
    before, after = feats[(h > 12) & (h < 23)], feats[h > 27]
    assert after["duty_cycle_60m"].mean() > 1.5 * before["duty_cycle_60m"].mean()
    assert after["idle_decay_rate_60m"].mean() > 2 * before["idle_decay_rate_60m"].mean()


def test_shift_labels_wrap_midnight():
    assert shift_label(6) == "Shift 1 (06-14)"
    assert shift_label(14) == "Shift 2 (14-22)"
    assert shift_label(23) == "Shift 3 (22-06)"
    assert shift_label(0) == "Shift 3 (22-06)"
