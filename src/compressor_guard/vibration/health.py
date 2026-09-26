"""
Bearing health indicator (HI), degradation-onset detection and fault diagnosis.

HI construction, per bearing:
1. Take the per-snapshot features; for test 1, with 2 channels per bearing, keep the
   channel-wise max.
2. Log-transform (the features are ratio-like and heavy-tailed), then standardise
   against the bearing's own early healthy life (`baseline_hours` after `skip_hours`
   of run-in).
3. Optional common-mode rejection (`reference="cross_bearing"`): subtract the log of the
   median of the *other* bearings on the same shaft at the same snapshot. Rig-wide
   shifts, such as the 148 h stop and restart in test 1, load changes or sensor gain,
   cancel out; damage local to one bearing does not.
4. HI = max over the chosen features of the one-sided z-score (how many healthy
   standard deviations the worst feature has moved up), smoothed with a trailing median.

Alarm episodes: an alarm is raised once the smoothed HI has stayed above `onset_k` for
`onset_persistence` consecutive snapshots, and it clears when HI drops back below k.
HI is already in healthy-sigma units, so `onset_k` reads directly as "k sigma above the
bearing's own normal".

As in Module A (72 h anticipation), an alarm counts as a *detection* only if it is raised
within `max_lead_hours` of the failure. An alarm weeks earlier would have been inspected
and found nothing, so it counts as a false alarm.
"""

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from compressor_guard.config import load_params, resolve_path
from compressor_guard.vibration.io import EXPECTED_BAND, KNOWN_FAILURES

ENV_BANDS = ["env_bpfo", "env_bpfi", "env_bsf"]
BAND_TO_FAULT = {"env_bpfo": "outer_race", "env_bpfi": "inner_race", "env_bsf": "roller_element"}


def bearing_table(features: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """One row per (bearing, snapshot): channel-wise max of `cols`."""
    keep = ["test", "timestamp", "hours", "bearing"]
    return (features.groupby(keep, sort=True)[cols].max().reset_index()
            .sort_values(["bearing", "timestamp"]).reset_index(drop=True))


def cross_bearing_log_ratio(bt: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """log(x) - log(median of the other bearings' x) at the same snapshot."""
    out = bt.copy()
    logs = {c: np.log(np.clip(bt[c], 1e-9, None)) for c in cols}
    for c in cols:
        wide = logs[c].to_frame("v").assign(ts=bt["timestamp"], b=bt["bearing"]).pivot(index="ts", columns="b", values="v")
        ref = pd.DataFrame({b: wide.drop(columns=b).median(axis=1) for b in wide.columns})
        ref_long = ref.stack().rename("ref")
        idx = pd.MultiIndex.from_arrays([bt["timestamp"], bt["bearing"]])
        out[f"lr_{c}"] = logs[c].to_numpy() - ref_long.reindex(idx).to_numpy()
    return out


def health_indicator(
    features: pd.DataFrame,
    hi_features: List[str],
    baseline_hours: float = 24.0,
    skip_hours: float = 1.0,
    smoothing: int = 6,
    reference: str = "cross_bearing",
) -> pd.DataFrame:
    """Add per-feature z-scores, `hi_raw` and `hi` (smoothed) to the bearing table."""
    cols = list(dict.fromkeys(hi_features + ENV_BANDS))
    bt = bearing_table(features, list(dict.fromkeys(cols + ["rms", "kurtosis"])))
    if reference == "cross_bearing":
        bt = cross_bearing_log_ratio(bt, cols)
    else:
        for c in cols:
            bt[f"lr_{c}"] = np.log(np.clip(bt[c], 1e-9, None))
    out = []
    for _, g in bt.groupby("bearing", sort=True):
        g = g.copy()
        base = g[(g["hours"] >= skip_hours) & (g["hours"] <= skip_hours + baseline_hours)]
        for c in cols:
            lx = g[f"lr_{c}"]
            lb = base[f"lr_{c}"]
            sd = lb.std() if lb.std() > 1e-9 else 1.0
            g[f"z_{c}"] = (lx - lb.mean()) / sd
        z = g[[f"z_{c}" for c in hi_features]].clip(lower=0)
        g["hi_raw"] = z.max(axis=1)
        g["hi_driver"] = z.idxmax(axis=1).str[2:]
        g["hi"] = g["hi_raw"].rolling(smoothing, min_periods=1).median()
        # Degradation level for RUL: RMS relative to the bearing's own healthy RMS
        # ("x times normal"), without cross-bearing referencing. Practitioners set trip
        # levels this way, and its value at failure is far more consistent across
        # bearings than the max-z HI.
        g["rms_ratio"] = (g["rms"] / base["rms"].median()).rolling(smoothing, min_periods=1).median()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def alarm_episodes(hi: pd.Series, hours: pd.Series, k: float, persistence: int) -> List[float]:
    """Hours at which each alarm episode is raised (persistence met), with reset below k."""
    starts, run, active = [], 0, False
    for a, h in zip((hi > k).to_numpy(), hours.to_numpy()):
        if a:
            run += 1
            if not active and run >= persistence:
                starts.append(float(h))
                active = True
        else:
            run, active = 0, False
    return starts


def detect_onset(hi: pd.Series, hours: pd.Series, k: float, persistence: int) -> Optional[float]:
    """Hours at which HI first stays above k for `persistence` consecutive snapshots."""
    above = (hi > k).to_numpy()
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= persistence:
            return float(hours.iloc[i])   # alert is raised when persistence is satisfied
    return None


def diagnose(hi_df: pd.DataFrame, onset_h: Optional[float]) -> Optional[str]:
    """Envelope band with the largest median z-score after onset -> fault type."""
    if onset_h is None:
        return None
    post = hi_df[hi_df["hours"] >= onset_h]
    med = post[[f"z_{c}" for c in ENV_BANDS]].median()
    return BAND_TO_FAULT[med.idxmax()[2:]]


def summarise_test(hi_all: pd.DataFrame, k: float, persistence: int,
                   max_lead_hours: Optional[float] = 168.0) -> pd.DataFrame:
    """
    One row per bearing. For failed bearings, `onset_hours` is the first alarm raised within
    `max_lead_hours` of failure, and earlier alarms are counted in `early_false_alarms`. For
    surviving bearings, `onset_hours` is their first alarm (if any).
    """
    rows = []
    for (test, b), g in hi_all.groupby(["test", "bearing"]):
        eol = float(g["hours"].max())
        mode = KNOWN_FAILURES.get((int(test), int(b)))
        eps = alarm_episodes(g["hi"], g["hours"], k, persistence)
        if mode and max_lead_hours is not None:
            valid = [e for e in eps if eol - e <= max_lead_hours]
            early = [e for e in eps if eol - e > max_lead_hours]
            onset = valid[0] if valid else None
        else:
            early, onset = [], (eps[0] if eps else None)
        dx = diagnose(g, onset)
        rows.append({
            "test": int(test), "bearing": int(b), "known_failure": mode or "none",
            "life_hours": eol, "onset_hours": onset,
            "hours_before_failure": (eol - onset) if (onset is not None and mode) else None,
            "onset_pct_life": 100 * onset / eol if onset is not None else None,
            "alarm_episodes": len(eps), "early_false_alarms": len(early), "alarm_hours": eps,
            "diagnosed_band": dx,
            "diagnosis_correct": (dx == mode) if (mode and dx) else None,
        })
    return pd.DataFrame(rows)


def health_params(params: Dict) -> Dict:
    h = params["ims"]["health"]
    return dict(hi_features=h["hi_features"], baseline_hours=h["baseline_hours"],
                skip_hours=h.get("skip_hours", 1.0), smoothing=h["smoothing_snapshots"],
                reference=h.get("reference", "cross_bearing"))


def summarise_from_params(hi_all: pd.DataFrame, params: Dict, **overrides) -> pd.DataFrame:
    h = params["ims"]["health"]
    kw = dict(k=h["onset_k"], persistence=h["onset_persistence"], max_lead_hours=h.get("max_lead_hours", 168.0))
    kw.update(overrides)
    return summarise_test(hi_all, **kw)


def run_all(params: Optional[Dict] = None, tests=(1, 2, 3)) -> pd.DataFrame:
    from compressor_guard.vibration.features import load_test_features
    params = params or load_params()
    hp = health_params(params)
    return pd.concat([health_indicator(load_test_features(t, params), **hp) for t in tests], ignore_index=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="IMS health indicator + onset detection.")
    ap.add_argument("--test", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--config", default="config/params.yaml")
    args = ap.parse_args(argv)
    params = load_params(args.config)
    h = params["ims"]["health"]
    hi = run_all(params, args.test)
    out = resolve_path(params["ims"].get("processed_dir", "data/processed")) / "ims_health.parquet"
    hi.to_parquet(out, index=False)
    summary = summarise_from_params(hi, params)
    print(summary.drop(columns="alarm_hours").round(1).to_string(index=False))
    e = evaluate_onsets(summary)
    print(f"[health] detected {e['detected']}/{e['failed']} failed bearings, median lead "
          f"{e['median_lead_hours']:.0f} h, {e['false_alarms']} false alarm(s)")
    print(f"[health] -> {out}")



def onset_sweep(hi_all: pd.DataFrame, ks, persistence: int, max_lead_hours: Optional[float] = 168.0) -> pd.DataFrame:
    """Detections, lead times and false alarms for a grid of HI thresholds k."""
    rows = []
    for k in ks:
        e = evaluate_onsets(summarise_test(hi_all, k, persistence, max_lead_hours))
        rows.append({"k": float(k), "detected": e["detected"], "failed": e["failed"],
                     "median_lead_h": e["median_lead_hours"], "false_alarms": e["false_alarms"],
                     "lead_hours": e["lead_hours"]})
    return pd.DataFrame(rows)


def evaluate_onsets(summary: pd.DataFrame) -> Dict:
    """
    Rig-aware scoring of one onset configuration.

    - Failed bearing: detected if an alarm was raised within the lead window; alarms
      before that window are false alarms (inspected, nothing actionable found).
    - Surviving bearing: every alarm episode raised *before* the first detection on a
      failed bearing of the same rig is a false alarm. After that, the shaft is already
      scheduled for maintenance, and neighbour vibration spreading along the shaft is expected.
    """
    leads, false_alarms = [], 0
    for test, g in summary.groupby("test"):
        failed = g[g["known_failure"] != "none"]
        first_true = failed["onset_hours"].min()
        for _, r in g.iterrows():
            if r["known_failure"] != "none":
                leads.append(r["hours_before_failure"] if pd.notna(r["onset_hours"]) else None)
                false_alarms += int(r.get("early_false_alarms", 0) or 0)
            else:
                eps = r["alarm_hours"] if isinstance(r.get("alarm_hours"), list) else (
                    [r["onset_hours"]] if pd.notna(r["onset_hours"]) else [])
                false_alarms += sum(1 for e in eps if pd.isna(first_true) or e < first_true)
    detected = [x for x in leads if x is not None and pd.notna(x)]
    return {"detected": len(detected), "failed": len(leads), "lead_hours": leads,
            "median_lead_hours": float(np.median(detected)) if detected else float("nan"),
            "false_alarms": false_alarms}


if __name__ == "__main__":
    main()
