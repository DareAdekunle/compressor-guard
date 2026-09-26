"""
Accelerated, timestamp-ordered replay of MetroPT-3 telemetry through the streaming
pipeline. It mimics a historian feed, and this is **simulated** real time on
historical data, not a live plant connection.

    python -m compressor_guard.stream.replay --speed 1440          # 1 simulated day per minute
    python -m compressor_guard.stream.replay --start "2020-07-13" --end "2020-07-16" --speed 0   # as fast as possible

In production, `iter_rows` would be replaced by a historian / MQTT / OPC-UA subscriber;
`StreamState` and everything after it would stay the same.
"""

import argparse
import json
import time
from typing import Dict, Iterator, List, Optional

import pandas as pd

from compressor_guard.config import load_params, resolve_path
from compressor_guard.features import load_baseline, shifts_from_params
from compressor_guard.stream.state import StreamState

STREAM_COLUMNS = ["timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
                  "Motor_current", "COMP", "LPS"]


def load_replay_frame(start: Optional[str], end: Optional[str], params: Dict) -> pd.DataFrame:
    from compressor_guard.data import load_labelled_metropt
    df = load_labelled_metropt(params)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] < pd.Timestamp(end)]
    return df[STREAM_COLUMNS].reset_index(drop=True)


def iter_rows(df: pd.DataFrame, speed: float = 1440.0) -> Iterator[Dict]:
    """
    Yield rows in timestamp order. `speed` = simulated seconds per wall-clock second
    (1440 -> one simulated day per minute; 0 -> no sleeping). Long data gaps are not
    replayed in real time: a single sleep is capped at 2 s.
    """
    prev = None
    for row in df.itertuples(index=False):
        r = row._asdict()
        if speed and prev is not None:
            dt = (r["timestamp"] - prev).total_seconds() / speed
            if dt > 0:
                time.sleep(min(dt, 2.0))
        prev = r["timestamp"]
        yield r


def build_state(params: Dict, model_names=("isolation_forest", "control_limits")) -> StreamState:
    from compressor_guard.models import load_detectors
    m = params["metropt"]
    dets = load_detectors(m.get("model_dir", "models"), names=model_names)
    if not dets:
        raise FileNotFoundError("No trained detectors in models/. Run `python -m compressor_guard.models --train` first.")
    return StreamState(load_baseline(m.get("baseline_path", "models/feature_baseline.json")), dets,
                       m["alert_logic"], shifts_from_params(params))


def run_replay(df: pd.DataFrame, state: StreamState, speed: float = 0.0, on_minute=None) -> pd.DataFrame:
    """Drive the state with the rows and collect one output record per scored minute."""
    out: List[Dict] = []

    def emit(rec):
        if rec is None:
            return
        out.append(rec)
        if on_minute:
            on_minute(rec)

    for row in iter_rows(df, speed):
        emit(state.update(row))
    emit(state.flush())
    return pd.DataFrame(out)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Replay MetroPT-3 through the streaming pipeline.")
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--start", default="2020-07-12 00:00:00", help="default: 3 days before F4")
    ap.add_argument("--end", default="2020-07-16 00:00:00")
    ap.add_argument("--speed", type=float, default=1440.0, help="simulated s per wall s (0 = max speed)")
    ap.add_argument("--model", default="isolation_forest", choices=["isolation_forest", "control_limits"])
    ap.add_argument("--out", default="data/processed/stream_output.jsonl")
    args = ap.parse_args(argv)

    params = load_params(args.config)
    df = load_replay_frame(args.start, args.end, params)
    state = build_state(params, (args.model,))
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[replay] {len(df):,} rows {args.start} -> {args.end} at speed x{args.speed:g} ({args.model})")

    with open(out_path, "w") as fh:
        def on_minute(rec):
            fh.write(json.dumps({k: (str(v) if k == "timestamp" else v) for k, v in rec.items()
                                 if k in ("timestamp", "shift", "duty_cycle_60m", "idle_decay_rate_60m",
                                          f"score_{args.model}", f"smoothed_{args.model}", f"alert_{args.model}")},
                                default=float) + "\n")
            fh.flush()
            for name, ev in rec.get("events", []):
                verb = "ALERT RAISED " if ev == "start" else "alert cleared"
                print(f"  {rec['timestamp']}  {verb}  [{name}] smoothed score={rec[f'smoothed_{name}']:.4f}")

        res = run_replay(df, state, args.speed, on_minute)
    n_alert = int(res[f"alert_{args.model}"].sum()) if len(res) else 0
    print(f"[replay] scored {len(res):,} minutes, {state.frozen_minutes} frozen minutes skipped, "
          f"{n_alert} alert minutes -> {out_path}")


if __name__ == "__main__":
    main()
