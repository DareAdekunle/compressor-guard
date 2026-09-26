import pytest
from fastapi.testclient import TestClient

from compressor_guard.api import app
from conftest import make_raw, requires_models

client = TestClient(app)


def test_health_responds():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "no-models")


@requires_models
def test_score_raw_roundtrip():
    raw = make_raw(hours=3, frozen=None, gap=None)
    rows = [{**r, "timestamp": str(r["timestamp"])} for r in
            raw[["timestamp", "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
                 "Motor_current", "COMP", "LPS"]].to_dict("records")]
    r = client.post("/score/raw", json={"rows": rows})
    assert r.status_code == 200
    mins = r.json()["minutes"]
    assert len(mins) == 180
    assert "isolation_forest_score" in mins[-1]
