"""
FastAPI scoring service for Module A.

    uvicorn compressor_guard.api:app --reload

Endpoints
- GET  /health       -> models loaded + alert thresholds
- POST /score        -> score pre-computed minute feature rows
- POST /score/raw    -> raw 10 s telemetry rows in; per-minute features, scores and alert
                        flags out (runs a fresh StreamState over the batch you send, so
                        send at least a few hours for the rolling windows to fill)
"""

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from compressor_guard.config import load_params
from compressor_guard.features import FEATURE_COLUMNS, load_baseline, shifts_from_params
from compressor_guard.models import load_detectors
from compressor_guard.stream.state import StreamState

app = FastAPI(title="CompressorGuard", version="0.1.0",
              description="Anomaly scoring for MetroPT-3-style compressor telemetry (Module A).")

STREAM_MODELS = ("isolation_forest", "control_limits")


@lru_cache(maxsize=1)
def _resources():
    params = load_params()
    m = params["metropt"]
    dets = load_detectors(m.get("model_dir", "models"), names=STREAM_MODELS)
    if not dets:
        raise RuntimeError("No trained models found. Run `python -m compressor_guard.models --train`.")
    return params, dets, load_baseline(m.get("baseline_path", "models/feature_baseline.json"))


class FeatureRows(BaseModel):
    features: List[Dict[str, Optional[float]]] = Field(..., description=f"rows with keys from {FEATURE_COLUMNS}")


class RawRow(BaseModel):
    timestamp: str
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float
    COMP: int
    LPS: int = 0


class RawRows(BaseModel):
    rows: List[RawRow]


def _clean(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


@app.get("/health")
def health():
    try:
        _, dets, _ = _resources()
    except RuntimeError as e:
        return {"status": "no-models", "detail": str(e)}
    return {"status": "ok", "models": {n: {"threshold": float(d.threshold)} for n, d in dets.items()}}


@app.post("/score")
def score(body: FeatureRows):
    _, dets, _ = _resources()
    df = pd.DataFrame(body.features)
    needed = sorted({c for d in dets.values() for c in d.feature_cols} - set(df.columns))
    if needed:
        raise HTTPException(422, f"missing feature columns: {needed}")
    out = []
    scores = {n: d.score_samples(df.astype(float)) for n, d in dets.items()}
    for i in range(len(df)):
        out.append({n: {"score": float(s[i]), "above_threshold": bool(s[i] > dets[n].threshold)}
                    for n, s in scores.items()})
    return {"results": out}


@app.post("/score/raw")
def score_raw(body: RawRows):
    params, dets, baseline = _resources()
    state = StreamState(baseline, dets, params["metropt"]["alert_logic"], shifts_from_params(params))
    recs = []
    for r in sorted((r.model_dump() for r in body.rows), key=lambda r: r["timestamp"]):
        rec = state.update(r)
        if rec:
            recs.append(rec)
    last = state.flush()
    if last:
        recs.append(last)
    keep = ["timestamp", "shift", "duty_cycle_60m", "idle_decay_rate_60m", "max_load_run_min_60m"]
    return {"frozen_minutes_skipped": state.frozen_minutes,
            "minutes": [{**{k: _clean(rec[k]) if k != "timestamp" else str(rec[k]) for k in keep},
                         **{f"{n}_{x}": _clean(rec[f"{x}_{n}"]) for n in dets for x in ("score", "smoothed", "alert")}}
                        for rec in recs]}
