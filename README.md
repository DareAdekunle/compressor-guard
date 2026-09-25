# CompressorGuard: Predictive Maintenance for Industrial Air Compressors

> Unsupervised anomaly detection on real compressor sensor data (MetroPT-3), tested against documented failures, with a maintenance-economics model to set the alert threshold.

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-portfolio%20project-orange)

---

## 1. The business problem

Air compressors power pneumatic systems, instruments and conveying lines in cement, fertiliser, refining and manufacturing plants. When one fails without warning, the costs are unplanned downtime, emergency repairs and lost production.

Most plants either run compressors to failure or service them on fixed schedules. This project tests a third option:

> **Can we detect a compressor developing a fault early enough to plan maintenance, without flooding engineers with false alarms?**

## 2. Dataset

**[MetroPT-3 (UCI)](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)** contains real readings from the air production unit (APU) of a metro train compressor in Porto, Portugal, collected over several months in 2020.

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository ([paper](https://www.nature.com/articles/s41597-022-01877-3)) |
| Type | Real operational data (not simulated) |
| Analog signals | `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current` |
| Digital signals | `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` |
| Failure labels | Not in the CSV. Failure windows come from the maintenance reports documented in the dataset paper and are encoded in `config/failures.yaml` |

**Why unsupervised:** in real plants, labelled failures are rare. A detector that learns "normal" and flags deviations is closer to how this would be deployed on a new asset than a classifier trained on a handful of failures.

## 3. Approach

```
raw sensors ──► cleaning & resampling ──► operating-cycle features ──► anomaly models ──► alert logic ──► failure back-test ──► cost model
```

1. **Data audit.** Sampling gaps, sensor ranges, and daily and weekly operating patterns. Failure periods are marked on the timeline.
2. **Domain features.** These carry most of the signal:
   - Compressor **duty cycle** (share of time `COMP` is on) and **cycles per hour**
   - Time to reach target pressure after each start
   - Rolling mean and standard deviation of `Motor_current` and `Oil_temperature`, and their drift against a baseline
   - Pressure differentials (`TP2 − TP3`, reservoir recovery rate)
3. **Models**, all trained **only on data well before the first failure**:
   - **Baseline:** statistical control limits (±3σ) on key signals
   - **Isolation Forest** on engineered features
   - **LSTM autoencoder** on sensor windows, where high reconstruction error means an anomaly
4. **Alert logic.** Smooth the raw scores, then alert only when the score stays above the threshold for N minutes. One noisy reading shouldn't page an engineer.
5. **Back-test against documented failures:**
   - Was each failure flagged in advance? **How many hours before?**
   - How many false alerts in failure-free periods?
   - Time-ordered splits only, never shuffled

## 4. From alerts to decisions: maintenance economics

```
Net value = (failures caught early × (cost of unplanned failure − cost of planned repair))
          − (false alerts × cost of an unnecessary inspection)
```

`notebooks/05_maintenance_economics.ipynb` sweeps the alert threshold and persistence window, then shows:
- **Net value vs threshold**, marking the operating point that maximises value
- The **break-even false-alarm rate**: how many false alerts per month the programme can absorb and still pay off

All costs sit in `config/costs.yaml` as **illustrative assumptions**, meant to be replaced with a plant's actual downtime cost, repair cost and technician rates.

## 5. Results

> Filled in from the final back-test.

| Model | Failures flagged early | Median lead time (hrs) | False alerts / month | Net value (illustrative) |
|---|---|---|---|---|
| ±3σ control limits | — / — | — | — | — |
| Isolation Forest | — / — | — | — | — |
| LSTM autoencoder | — / — | — | — | — |

**Key findings:**
- _Which signals moved first before failure (feature attribution)_
- _Recommended threshold and persistence window, and why_

## 6. Limitations

- There are only a few documented failures, so lead-time estimates are indicative, not statistically robust.
- One asset type in one operating environment. A plant compressor would need recalibration on its own baseline.
- Cost figures are illustrative.
- Batch scoring only; a real deployment would run on streaming data from a plant historian.

## 7. How this would deploy in a plant

```
Sensors / PLC ──► Historian (e.g. OSIsoft PI) ──► feature job (every N min) ──► model API ──► alert + CMMS work order
                                                                                  │
                                                                         monitoring: drift, alert rate
```

The FastAPI service and Streamlit dashboard in this repo stand in for the "model API" and "alert" steps.

## 8. Repository structure

```
compressor-guard/
├── config/
│   ├── params.yaml
│   ├── failures.yaml        # failure windows from the dataset paper
│   └── costs.yaml           # illustrative cost assumptions
├── data/                    # not committed
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_lstm_autoencoder.ipynb
│   └── 05_maintenance_economics.ipynb
├── src/compressor_guard/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── alerts.py            # smoothing + persistence logic
│   ├── backtest.py
│   └── api.py               # FastAPI scoring endpoint
├── app/dashboard.py         # Streamlit: sensor traces, anomaly score, alerts
├── tests/
├── requirements.txt
└── README.md
```

## 9. Getting started

```bash
git clone https://github.com/<your-username>/compressor-guard.git
cd compressor-guard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download MetroPT-3 from UCI and place the CSV in data/raw/
# https://archive.ics.uci.edu/dataset/791/metropt+3+dataset

python -m compressor_guard.features --config config/params.yaml
python -m compressor_guard.models --train --config config/params.yaml
python -m compressor_guard.backtest --config config/params.yaml
mlflow ui

uvicorn compressor_guard.api:app --reload
streamlit run app/dashboard.py
```

## 10. Tech stack

Python · pandas · scikit-learn · PyTorch · MLflow · FastAPI · Streamlit · pytest

## 11. Acknowledgements

Dataset: Veloso, B., Ribeiro, R.P., Gama, J., Pereira, P.M., *The MetroPT dataset for predictive maintenance*, Scientific Data (2022).

---

**Author:** Oludare Adekunle · [LinkedIn](#) · [Email](mailto:oludareadekunle@gmail.com)
_Portfolio project on public data._
