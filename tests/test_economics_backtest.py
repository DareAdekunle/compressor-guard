import numpy as np
import pandas as pd

from compressor_guard.backtest import evaluate_backtest
from compressor_guard.economics import CostModel


def test_value_of_warning():
    cm = CostModel(25000, 4000, 5000, 500, 48, 12)
    assert cm.value_of_warning(None) == 0
    assert cm.value_of_warning(0) == 0
    assert cm.value_of_warning(10) == 16000
    assert cm.value_of_warning(60) == 21000
    assert cm.net_value([60, 10, None], 4) == 21000 + 16000 - 2000
    assert cm.break_even_false_alerts([60]) == 42


def _events(starts, ends):
    return pd.DataFrame({"event_id": range(1, len(starts) + 1), "start": pd.to_datetime(starts),
                         "end": pd.to_datetime(ends), "duration_hours": 1.0, "peak_score": 1.0,
                         "shift": "Shift 1 (06-14)"})


def test_backtest_lead_and_false_alerts():
    failures = [{"id": "F1", "start": pd.Timestamp("2020-05-10 12:00"), "end": pd.Timestamp("2020-05-10 18:00"),
                 "maintenance": pd.Timestamp("2020-05-11 12:00")}]
    ev = _events(["2020-05-09 12:00", "2020-05-11 06:00", "2020-05-20 00:00"],
                 ["2020-05-09 14:00", "2020-05-11 07:00", "2020-05-20 02:00"])
    ts = pd.date_range("2020-05-01", "2020-05-31", freq="1min")
    df = pd.DataFrame({"timestamp": ts, "shift": "Shift 1 (06-14)"})
    cm = CostModel(25000, 4000, 5000, 500, 12, 12)
    r = evaluate_backtest(df, ev, failures=failures, cost_model=cm, anticipation_hours=72)
    d = r["failure_details"].iloc[0]
    assert d["flagged_early"] and d["lead_time_hours"] == 24
    assert r["total_false_alerts"] == 1        # the post-failure alert before maintenance is not false
    assert r["net_economic_value"] == 21000 - 500


def test_backtest_detected_during_failure_is_not_early():
    failures = [{"id": "F1", "start": pd.Timestamp("2020-05-10 12:00"), "end": pd.Timestamp("2020-05-10 18:00"),
                 "maintenance": None}]
    ev = _events(["2020-05-10 13:00"], ["2020-05-10 15:00"])
    df = pd.DataFrame({"timestamp": pd.date_range("2020-05-01", "2020-05-31", freq="1min")})
    r = evaluate_backtest(df, ev, failures=failures, cost_model=CostModel())
    d = r["failure_details"].iloc[0]
    assert d["detected"] and not d["flagged_early"]
    assert np.isclose(d["detection_delay_hours"], 1.0)
    assert r["net_economic_value"] == 0
