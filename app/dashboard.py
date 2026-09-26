"""
CompressorGuard dashboard.

    streamlit run app/dashboard.py

Tabs
- Module A: MetroPT-3 sensors, anomaly score and alerts for a chosen model and period
- Module B: IMS bearing health indicators with onset and failure marked
- Live replay: MetroPT-3 rows replayed through the streaming pipeline (simulated real time)

Needs the processed artefacts from the pipeline (see README §9).
"""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compressor_guard.alerts import generate_alerts  # noqa: E402
from compressor_guard.config import load_params, resolve_path  # noqa: E402
from compressor_guard.data import load_failures  # noqa: E402
from compressor_guard.vibration.health import summarise_from_params  # noqa: E402
from compressor_guard.vibration.io import KNOWN_FAILURES  # noqa: E402

st.set_page_config(page_title="CompressorGuard", layout="wide")
params = load_params()
M = params["metropt"]
LABELS = {"isolation_forest": "Isolation Forest", "lstm_autoencoder": "LSTM autoencoder",
          "control_limits": "±3σ control limits"}


@st.cache_data(show_spinner="Loading MetroPT-3 features and scores ...")
def load_module_a():
    feats = pd.read_parquet(resolve_path(M["processed_parquet_path"]))
    scores = pd.read_parquet(resolve_path(M["scores_parquet_path"]))
    thresholds = pd.read_json(resolve_path(M["model_dir"]) / "thresholds.json", typ="series").to_dict()
    return feats, scores, thresholds


@st.cache_data(show_spinner="Loading IMS health indicators ...")
def load_module_b():
    hi = pd.read_parquet(resolve_path(params["ims"]["processed_dir"]) / "ims_health.parquet")
    return hi, summarise_from_params(hi, params)


def missing(msg):
    st.warning(f"{msg}\n\nRun the pipeline first (README §9).")


st.title("CompressorGuard: predictive maintenance demo")
st.caption("Module A: MetroPT-3 air-compressor telemetry · Module B: IMS run-to-failure bearings · "
           "public data, illustrative costs")
tab_a, tab_b, tab_live = st.tabs(["Module A: telemetry alerts", "Module B: bearing health", "Live replay"])

# ------------------------------------------------------------------ Module A
with tab_a:
    try:
        feats, scores, thresholds = load_module_a()
    except FileNotFoundError:
        missing("MetroPT-3 features/scores not found.")
    else:
        failures = load_failures()
        c1, c2, c3 = st.columns([2, 2, 3])
        models = [m for m in LABELS if f"score_{m}" in scores]
        model = c1.selectbox("Model", models, format_func=LABELS.get)
        focus = c2.selectbox("Focus", ["Whole period"] + [f["id"] for f in failures])
        if focus == "Whole period":
            lo, hi_ = scores["timestamp"].min(), scores["timestamp"].max()
        else:
            f = next(x for x in failures if x["id"] == focus)
            lo, hi_ = f["start"] - pd.Timedelta(hours=96), f["end"] + pd.Timedelta(hours=24)
        th = c3.slider("Alert threshold (× default)", 0.5, 3.0, 1.0, 0.05) * thresholds[model]

        a = M["alert_logic"]
        df_alerts, events = generate_alerts(scores, scores[f"score_{model}"].to_numpy(), th,
                                            a["smoothing_window_minutes"], a["persistence_minutes"])
        view = df_alerts[(df_alerts["timestamp"] >= lo) & (df_alerts["timestamp"] <= hi_)]
        fv = feats[(feats["timestamp"] >= lo) & (feats["timestamp"] <= hi_)]
        step = max(1, len(view) // 20000)   # keep the browser responsive on the full period
        view, fv = view.iloc[::step], fv.iloc[::step]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=["Loaded duty cycle (60 min) and idle pressure decay",
                                            "Oil temperature (60 min mean)", f"{LABELS[model]} smoothed score"])
        fig.add_trace(go.Scattergl(x=fv["timestamp"], y=fv["duty_cycle_60m"], name="duty cycle", line=dict(width=1)), 1, 1)
        fig.add_trace(go.Scattergl(x=fv["timestamp"], y=fv["idle_decay_rate_60m"], name="idle decay (bar/min)",
                                   line=dict(width=1)), 1, 1)
        fig.add_trace(go.Scattergl(x=fv["timestamp"], y=fv["oil_temp_mean_60m"], name="oil °C",
                                   line=dict(width=1, color="#ff7f0e")), 2, 1)
        fig.add_trace(go.Scattergl(x=view["timestamp"], y=view["smoothed_score"], name="score",
                                   line=dict(width=1, color="#1f77b4")), 3, 1)
        fig.add_hline(y=th, line_dash="dash", line_color="black", row=3, col=1)
        for f in failures:
            if f["end"] >= lo and f["start"] <= hi_:
                fig.add_vrect(x0=f["start"], x1=f["end"], fillcolor="red", opacity=0.2, line_width=0)
        for e in events.itertuples():
            if e.end >= lo and e.start <= hi_:
                fig.add_vrect(x0=e.start, x1=e.end, fillcolor="orange", opacity=0.35, line_width=0, row=3, col=1)
        fig.update_layout(height=720, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        from compressor_guard.backtest import evaluate_backtest
        res = evaluate_backtest(df_alerts, events, failures=failures,
                                anticipation_hours=M["backtest"]["anticipation_hours"],
                                grace_hours=M["backtest"]["post_failure_grace_hours"],
                                eval_start=pd.Timestamp(M["train_split"]["end"]))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Failures flagged early", res["failures_flagged_str"])
        k2.metric("Detected by failure end", f"{res['failures_detected_count']} / {res['total_failures']}")
        k3.metric("False alerts / month", f"{res['false_alerts_per_month']:.1f}")
        k4.metric("Net value (illustrative)", f"${res['net_economic_value']:,.0f}")
        st.dataframe(res["failure_details"], width="stretch", hide_index=True)

# ------------------------------------------------------------------ Module B
with tab_b:
    try:
        hi, summary = load_module_b()
    except FileNotFoundError:
        missing("IMS health indicators not found.")
    else:
        c1, c2 = st.columns(2)
        test = c1.selectbox("Test", [1, 2, 3], index=1)
        bearings = c2.multiselect("Bearings", [1, 2, 3, 4], default=[1, 2, 3, 4])
        k = params["ims"]["health"]["onset_k"]
        fig = go.Figure()
        for b in bearings:
            g = hi[(hi["test"] == test) & (hi["bearing"] == b)]
            lab = f"B{b}" + (f" ({KNOWN_FAILURES[(test, b)].replace('_', ' ')})" if (test, b) in KNOWN_FAILURES else "")
            fig.add_trace(go.Scattergl(x=g["hours"], y=g["hi"], name=lab, mode="lines"))
            r = summary[(summary["test"] == test) & (summary["bearing"] == b)].iloc[0]
            if pd.notna(r["onset_hours"]):
                fig.add_vline(x=r["onset_hours"], line_dash="dot", annotation_text=f"B{b} alarm")
        fig.add_hline(y=k, line_dash="dash", annotation_text=f"k = {k:g}σ")
        fig.update_layout(height=520, yaxis_type="log", xaxis_title="hours since test start",
                          yaxis_title="health indicator (healthy σ, log)", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summary[summary["test"] == test].round(1), width="stretch", hide_index=True)

# ------------------------------------------------------------------ Live replay
with tab_live:
    st.markdown("Replays MetroPT-3 rows **in timestamp order** through `StreamState`: incremental "
                "features, online scoring and the persistence alert rule. This is *simulated* real time on "
                "historical data. In production, the row source would be a historian or MQTT/OPC-UA feed.")
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start", pd.Timestamp("2020-07-13"))
    days = c2.slider("Days to replay", 1, 5, 3)
    speed = c3.select_slider("Speed (simulated s per wall s)", [0, 1440, 5760, 23040], value=5760,
                             help="1440 = one simulated day per minute; 0 = as fast as possible")
    if st.button("Start replay", type="primary"):
        try:
            from compressor_guard.stream.replay import build_state, iter_rows, load_replay_frame
            state = build_state(params, ("isolation_forest",))
        except FileNotFoundError as e:
            missing(str(e))
        else:
            df = load_replay_frame(str(start), str(pd.Timestamp(start) + pd.Timedelta(days=days)), params)
            chart, status = st.empty(), st.empty()
            recs, last_draw = [], 0.0
            th = state.detectors["isolation_forest"].threshold
            for row in iter_rows(df, speed):
                rec = state.update(row)
                if rec is None:
                    continue
                recs.append({k: rec[k] for k in ("timestamp", "duty_cycle_60m", "smoothed_isolation_forest",
                                                  "alert_isolation_forest")})
                for _, ev in rec["events"]:
                    status.warning(f"{rec['timestamp']}: alert {'RAISED' if ev == 'start' else 'cleared'}")
                if time.time() - last_draw > 0.5:
                    d = pd.DataFrame(recs)
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
                    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["duty_cycle_60m"], name="duty cycle 60m"), 1, 1)
                    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["smoothed_isolation_forest"], name="IF score"), 2, 1)
                    fig.add_hline(y=th, line_dash="dash", row=2, col=1)
                    al = d[d["alert_isolation_forest"]]
                    fig.add_trace(go.Scatter(x=al["timestamp"], y=al["smoothed_isolation_forest"], mode="markers",
                                             marker=dict(color="red", size=4), name="alert"), 2, 1)
                    fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
                    chart.plotly_chart(fig, width="stretch", key=f"live_{len(recs)}")
                    last_draw = time.time()
            st.success(f"Replay finished: {len(recs):,} minutes scored, {state.frozen_minutes} frozen minutes skipped.")
