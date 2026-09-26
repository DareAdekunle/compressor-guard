"""
Module A back-test: detection, lead time and false-alarm rate against the four
documented MetroPT-3 air-leak failures.

Definitions
-----------
- **Incident window** for failure F: from `start - anticipation_hours` to the logged
  maintenance time (or `end + post_failure_grace_hours` when none is logged). An alert
  event that overlaps it belongs to F. Any other alert event is a **false alert**.
- **Flagged early**: an associated alert starts before F's start. Lead time is
  `F.start - alert start`, capped at the anticipation window.
- **Detected**: an associated alert starts before F's end (early or during the failure).
- **False alerts / month** is measured only on the out-of-sample period (after the
  training window), per 30.44 days of calendar time.
"""

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from compressor_guard.alerts import generate_alerts
from compressor_guard.config import load_params, resolve_path
from compressor_guard.data import load_failures
from compressor_guard.economics import CostModel

DAYS_PER_MONTH = 30.4375


def incident_windows(failures: List[Dict], anticipation_hours: float, grace_hours: float) -> List[Dict]:
    out = []
    for f in failures:
        end = f.get("maintenance") or (f["end"] + pd.Timedelta(hours=grace_hours))
        out.append({**f, "window_start": f["start"] - pd.Timedelta(hours=anticipation_hours),
                    "window_end": max(end, f["end"])})
    return out


def evaluate_backtest(
    df_with_alerts: pd.DataFrame,
    alert_events: pd.DataFrame,
    failures: Optional[List[Dict]] = None,
    cost_model: Optional[CostModel] = None,
    anticipation_hours: float = 72.0,
    grace_hours: float = 24.0,
    eval_start: Optional[pd.Timestamp] = None,
    timestamp_col: str = "timestamp",
) -> Dict:
    """Score one set of alert events against the failures. See module docstring."""
    failures = failures if failures is not None else load_failures()
    cm = cost_model or CostModel.from_yaml()
    windows = incident_windows(failures, anticipation_hours, grace_hours)

    ev = alert_events.copy()
    if eval_start is not None and not ev.empty:
        ev = ev[ev["end"] >= eval_start]
    ev["failure_id"] = None

    rows = []
    for w in windows:
        hit = ev[(ev["start"] <= w["window_end"]) & (ev["end"] >= w["window_start"])] if not ev.empty else ev
        ev.loc[hit.index, "failure_id"] = w["id"]
        before_end = hit[hit["start"] <= w["end"]]
        first = before_end["start"].min() if not before_end.empty else pd.NaT
        first_eff = max(first, w["window_start"]) if pd.notna(first) else pd.NaT
        lead = (w["start"] - first_eff).total_seconds() / 3600.0 if pd.notna(first_eff) else np.nan
        rows.append({
            "failure_id": w["id"],
            "failure_start": w["start"],
            "failure_end": w["end"],
            "first_alert": first,
            "detected": bool(pd.notna(first)),
            "flagged_early": bool(pd.notna(lead) and lead > 0),
            "lead_time_hours": lead if pd.notna(lead) and lead > 0 else (0.0 if pd.notna(lead) else np.nan),
            "detection_delay_hours": -lead if pd.notna(lead) and lead <= 0 else (0.0 if pd.notna(lead) else np.nan),
        })
    fdf = pd.DataFrame(rows)

    false_events = ev[ev["failure_id"].isna()] if not ev.empty else ev
    ts = df_with_alerts[timestamp_col]
    t0 = eval_start if eval_start is not None else ts.min()
    months = max((ts.max() - t0).total_seconds() / 86400.0 / DAYS_PER_MONTH, 1e-9)

    # Shift view: false alerts normalised by how much scored time each shift has.
    shift_rates = {}
    if "shift" in df_with_alerts.columns:
        scored = df_with_alerts[ts >= t0]
        exposure_days = scored.groupby("shift", observed=True).size() / 1440.0
        fa = false_events["shift"].value_counts() if not false_events.empty else pd.Series(dtype=float)
        for s, days in exposure_days.items():
            shift_rates[str(s)] = float(fa.get(str(s), 0)) / max(days / DAYS_PER_MONTH, 1e-9)

    early = fdf.loc[fdf["flagged_early"], "lead_time_hours"]
    return {
        "failures_flagged_count": int(fdf["flagged_early"].sum()),
        "failures_detected_count": int(fdf["detected"].sum()),
        "total_failures": len(fdf),
        "failures_flagged_str": f"{int(fdf['flagged_early'].sum())} / {len(fdf)}",
        "median_lead_time_hours": float(early.median()) if len(early) else float("nan"),
        "total_false_alerts": int(len(false_events)),
        "false_alerts_per_month": len(false_events) / months,
        "false_alerts_per_month_by_shift": shift_rates,
        "eval_months": months,
        "net_economic_value": cm.net_value(fdf["lead_time_hours"].where(fdf["flagged_early"]), len(false_events)),
        "break_even_false_alerts": cm.break_even_false_alerts(fdf["lead_time_hours"].where(fdf["flagged_early"])),
        "failure_details": fdf,
        "false_events": false_events,
        "alert_events": ev,
    }


def backtest_scores(
    scores: pd.DataFrame,
    score_col: str,
    threshold: float,
    params: Optional[Dict] = None,
    failures: Optional[List[Dict]] = None,
    cost_model: Optional[CostModel] = None,
) -> Dict:
    """Alert logic + back-test for one score column at one threshold."""
    params = params or load_params()
    a = params["metropt"]["alert_logic"]
    b = params["metropt"].get("backtest", {})
    df_alerts, events = generate_alerts(
        scores, scores[score_col].to_numpy(), threshold,
        smoothing_window=a["smoothing_window_minutes"], persistence_steps=a["persistence_minutes"])
    res = evaluate_backtest(
        df_alerts, events, failures=failures, cost_model=cost_model,
        anticipation_hours=b.get("anticipation_hours", 72.0),
        grace_hours=b.get("post_failure_grace_hours", 24.0),
        eval_start=pd.Timestamp(params["metropt"]["train_split"]["end"]))
    res["df_alerts"] = df_alerts
    res["threshold"] = threshold
    return res


def sweep_thresholds(
    scores: pd.DataFrame,
    score_col: str,
    thresholds,
    params: Optional[Dict] = None,
    cost_models: Optional[Dict[str, CostModel]] = None,
    failures: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Net value, detections and false alerts at each alert threshold. Alerts are computed
    once per threshold and valued under every cost model (e.g. parts lead-time scenarios).
    """
    params = params or load_params()
    cost_models = cost_models or {"base": CostModel.from_yaml()}
    rows = []
    for th in thresholds:
        res = backtest_scores(scores, score_col, float(th), params, failures=failures)
        leads = res["failure_details"]["lead_time_hours"].where(res["failure_details"]["flagged_early"])
        for name, cm in cost_models.items():
            rows.append({
                "threshold": float(th), "scenario": name,
                "flagged_early": res["failures_flagged_count"], "detected": res["failures_detected_count"],
                "median_lead_h": res["median_lead_time_hours"],
                "false_alerts": res["total_false_alerts"], "false_alerts_per_month": res["false_alerts_per_month"],
                "net_value": cm.net_value(leads, res["total_false_alerts"]),
                "break_even_false_alerts_per_month": cm.break_even_false_alerts(leads) / res["eval_months"],
            })
    return pd.DataFrame(rows)


def results_row(name: str, res: Dict) -> Dict:
    return {
        "model": name,
        "flagged_early": res["failures_flagged_str"],
        "detected_by_end": f"{res['failures_detected_count']} / {res['total_failures']}",
        "median_lead_h": round(res["median_lead_time_hours"], 1) if np.isfinite(res["median_lead_time_hours"]) else None,
        "false_alerts": res["total_false_alerts"],
        "false_alerts_per_month": round(res["false_alerts_per_month"], 2),
        "net_value_usd": round(res["net_economic_value"]),
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Back-test Module A alerts against documented failures.")
    ap.add_argument("--config", default="config/params.yaml")
    args = ap.parse_args(argv)

    params = load_params(args.config)
    m = params["metropt"]
    scores = pd.read_parquet(resolve_path(m.get("scores_parquet_path", "data/processed/metropt_scores.parquet")))
    thresholds = json.loads((resolve_path(m.get("model_dir", "models")) / "thresholds.json").read_text())

    rows, details = [], []
    for name, th in thresholds.items():
        col = f"score_{name}"
        if col not in scores:
            continue
        res = backtest_scores(scores, col, th, params)
        rows.append(results_row(name, res))
        details.append(res["failure_details"].assign(model=name))
    table = pd.DataFrame(rows)
    out_dir = resolve_path("reports")
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "module_a_results.csv", index=False)
    pd.concat(details).to_csv(out_dir / "module_a_failure_details.csv", index=False)
    print(table.to_string(index=False))
    print(f"[backtest] -> {out_dir / 'module_a_results.csv'}")


if __name__ == "__main__":
    main()
