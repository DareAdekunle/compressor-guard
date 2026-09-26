"""
Training/serving-skew guard: the streaming pipeline (stream/state.py) must produce the
same features, scores and alerts as the batch pipeline (features.py + alerts.py).
"""
import numpy as np
import pandas as pd
import pytest

from compressor_guard.alerts import generate_alerts
from compressor_guard.features import DRIFT_COLUMNS, FEATURE_COLUMNS, build_feature_table, load_baseline
from compressor_guard.models import build_detectors, score_detectors
from compressor_guard.stream.replay import run_replay
from compressor_guard.stream.state import StreamState
from conftest import make_raw, requires_models

TOL = dict(rtol=1e-6, atol=1e-6, equal_nan=True)


def _assert_parity(batch: pd.DataFrame, streamed: pd.DataFrame, model_names, alert_params):
    assert len(batch) == len(streamed), "stream and batch scored different minutes"
    m = batch.merge(streamed, on="timestamp", suffixes=("_b", "_s"))
    assert len(m) == len(batch)
    for c in FEATURE_COLUMNS + DRIFT_COLUMNS:
        np.testing.assert_allclose(m[f"{c}_b"].astype(float), m[f"{c}_s"].astype(float), **TOL, err_msg=c)
    for n in model_names:
        np.testing.assert_allclose(m[f"score_{n}_b"], m[f"score_{n}_s"], **TOL, err_msg=n)
        th = alert_params["_thresholds"][n]
        out, _ = generate_alerts(batch, batch[f"score_{n}"].to_numpy(), th,
                                 alert_params["smoothing_window_minutes"], alert_params["persistence_minutes"])
        mm = out[["timestamp", "smoothed_score", "alert_flag"]].merge(streamed, on="timestamp")
        np.testing.assert_allclose(mm["smoothed_score"], mm[f"smoothed_{n}"], **TOL)
        assert (mm["alert_flag"].to_numpy() == mm[f"alert_{n}"].to_numpy()).all(), f"alert flags differ ({n})"


def test_stream_matches_batch_synthetic(params):
    raw = make_raw(hours=30, leak_from=22)
    feats, baseline = build_feature_table(raw, params, failures=[])
    dets = build_detectors(params, include_lstm=False)
    train = feats[feats["timestamp"] <= pd.Timestamp(params["metropt"]["train_split"]["end"])]
    for d in dets.values():
        d.fit(train)
    dets["isolation_forest"].threshold = float(np.quantile(dets["isolation_forest"].score_samples(train), 0.99))
    a = dict(params["metropt"]["alert_logic"], persistence_minutes=20)
    batch = feats.merge(score_detectors(dets, feats)[["timestamp"] + [f"score_{n}" for n in dets]], on="timestamp")

    state = StreamState(baseline, dets, a)
    streamed = run_replay(raw[["timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                               "Oil_temperature", "Motor_current", "COMP", "LPS"]], state, speed=0)
    assert state.frozen_minutes == feats.attrs["frozen_minutes"]
    a["_thresholds"] = {n: d.threshold for n, d in dets.items()}
    _assert_parity(batch, streamed, list(dets), a)
    assert streamed["alert_isolation_forest"].any(), "the synthetic leak should raise an alert"


@requires_models
def test_stream_matches_batch_real_data():
    """1.5 days before F4 on real MetroPT-3 data with the trained models."""
    from compressor_guard.config import load_params
    from compressor_guard.data import load_labelled_metropt
    from compressor_guard.stream.replay import STREAM_COLUMNS, build_state

    params = load_params()
    raw = load_labelled_metropt(params)
    raw = raw[(raw["timestamp"] >= "2020-07-14 00:00") & (raw["timestamp"] < "2020-07-15 12:00")]
    state = build_state(params)
    streamed = run_replay(raw[STREAM_COLUMNS].reset_index(drop=True), state, speed=0)

    feats, _ = build_feature_table(raw, params, baseline=load_baseline())
    batch = feats.merge(score_detectors(state.detectors, feats)[["timestamp"] + [f"score_{n}" for n in state.detectors]],
                        on="timestamp")
    a = dict(params["metropt"]["alert_logic"], _thresholds={n: d.threshold for n, d in state.detectors.items()})
    _assert_parity(batch, streamed, list(state.detectors), a)
