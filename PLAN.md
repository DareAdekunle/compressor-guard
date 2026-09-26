# CompressorGuard: Weekend Build Plan

> **Status (2026-09-26): all core items and all stretch items are built.** Both modules run end to end from the CLI and the notebooks, the LSTM-AE is included, the streaming simulation has a passing parity test, and FastAPI and Streamlit work. The README results tables are filled from the notebook runs. Still open: repo screenshots/GIF of the live replay, and deploying to Streamlit Community Cloud (needs the processed artefacts hosted somewhere, since `data/` is not committed).
>
> Deviations from the plan, found in the data:
> - `COMP` turned out to mean "no air intake", so the duty cycle is `1 − COMP`.
> - ~7 days of frozen-logger data are masked.
> - Two unreported leak-like episodes are excluded from training.
> - The IMS sampling rate is 20,480 Hz, not 20 kHz.
> - The bearing HI uses cross-bearing referencing to survive the test 1 restart.
> - RUL is reported, but it does not beat a naive baseline with four bearings, so it is flagged as future work.

**Goal by Sunday night:** a public repo with both modules working end to end, real results in the README tables, and a Streamlit demo. The streaming simulation comes last, only once the core is done. Honest and finished beats ambitious and half-done.

**Rule:** if a block runs over, cut the item marked *(stretch)* and don't touch the core.

---

## Friday night / Saturday early: setup (≈1.5 h)

- [ ] Create the GitHub repo `compressor-guard` (MIT licence, Python `.gitignore`, add `data/`)
- [ ] Add `README.md`, `DATA.md`, `PLAN.md`; create the folder skeleton from README §8
- [ ] `requirements.txt`: pandas, numpy, scipy, scikit-learn, torch, pyarrow, mlflow, fastapi, uvicorn, streamlit, plotly, joblib, pyyaml, pytest, ucimlrepo, kaggle
- [ ] Set up your Kaggle API token (see DATA.md §2)
- [ ] **Start both downloads now** (`python scripts/download_data.py`). IMS is large, so let it run while you work on MetroPT-3.

## Saturday: Module A, MetroPT-3 (≈7 h)

| Time | Task | Output |
|---|---|---|
| 1 h | Load CSV, parse timestamps, check gaps and ranges. Enter failure windows in `failures.yaml` **after verifying the times** | `01_metropt_audit.ipynb`: timeline plot with failures shaded |
| 1.5 h | Features: duty cycle, cycles/hour, time-to-pressure, rolling stats, drift, `TP2−TP3`, shift labels | `features.py`, processed parquet |
| 1 h | Baseline ±3σ + Isolation Forest, trained on healthy data before the first failure. Log runs to MLflow | `03_metropt_models.ipynb` |
| 1 h | Alert logic (smoothing + persistence) + back-test: flagged? lead time? false alerts/month? Rates by shift | Module A results table filled |
| 1.5 h | LSTM autoencoder on sensor windows *(stretch: skip if behind; Isolation Forest is enough)* | extra row in results |
| 1 h | Commit, push, write key findings in README | ✅ Module A done |

## Sunday: Module B, IMS bearings + economics + demo (≈8 h)

| Time | Task | Output |
|---|---|---|
| 1.5 h | IMS loader + features for **Test 2 only**: RMS, kurtosis, crest factor, FFT band energy at BPFO/BPFI/BSF, envelope spectrum. One row per snapshot → parquet | `vibration/features.py`, `04_ims_features.ipynb` |
| 1 h | Health indicator + onset detection on Test 2. Plot HI over time with the onset marked | Hours-before-failure for Test 2 / B1 |
| 0.5 h | Fault diagnosis: does BPFO energy dominate after onset? (It should, since it's an outer-race failure) | Plot of fault-band energies |
| 1 h | Run the same pipeline on Tests 1 and 3 | Module B table filled (4 bearings) |
| 1 h | RUL: exponential fit on the HI after onset, leave-one-bearing-out evaluation *(stretch: if behind, report onset + diagnosis only and list RUL as future work)* | RUL column |
| 1.5 h | `economics.py` + `06_maintenance_economics.ipynb`: net value vs threshold, break-even false-alarm rate, parts lead-time sensitivity | Economics plots |
| 1 h | Streamlit dashboard: Module A sensor + anomaly score + alerts; Module B HI curves. FastAPI `/score` endpoint *(stretch)* | `app/dashboard.py` |
| 1.5 h | **Streaming simulation** *(stretch)*: `stream/replay.py` sends MetroPT-3 rows in timestamp order at accelerated speed; `stream/state.py` keeps a rolling buffer and computes features incrementally; each window is scored and the alert rule fires live; the Streamlit page updates as the replay runs. Add `tests/test_stream_parity.py` checking that streamed scores match batch scores on the same data | Live demo + parity test passing |
| 0.5 h | Final README pass, screenshots (or a GIF of the live replay), push. Deploy to Streamlit Community Cloud *(stretch)* | ✅ Done |

---

## Definition of done

- [x] Both modules run from a clean clone following README §9
- [x] Every number in the README results tables comes from your own runs
- [x] At least 3 plots in the README: MetroPT-3 timeline with alerts, IMS health indicator with onset, net value vs threshold
- [x] Limitations section is accurate
- [x] No raw data in the repo

## Pitfalls to avoid

- **Leakage:** never fit scalers or models on data that includes failure periods or anything after them. Use time-ordered splits only.
- **Unverified failure times:** lead-time results are only as good as `failures.yaml`.
- **IMS memory blow-up:** process one file at a time and save features; never load a whole test into memory.
- **Over-claiming:** 4 failures and 4 bearings aren't enough for statistical confidence. Say so.
- **Training-serving skew:** streamed features must be computed exactly like batch features. The parity test is what proves it. Describe the demo as *simulated* real-time on replayed data, not a live plant feed.

---

## Resume bullets, to use **only after** the matching item is built

- Processed months of real IIoT compressor telemetry (pressure, temperature, motor current, valve and oil signals) and **20 kHz run-to-failure bearing vibration data**, engineering duty-cycle, drift, spectral and envelope features across operating shifts.
- Built unsupervised anomaly detection (Isolation Forest, LSTM autoencoder) and a vibration health indicator to flag degradation early, back-tested against documented failures (**detected N of 4 failures a median of X hours ahead**; **bearing degradation onset Y hours before failure**).
- Built a maintenance-economics model that balances alert thresholds against break-even false-alarm rates and spare-parts lead time to optimise maintenance scheduling.

- *(Only if the streaming stretch is built)* Built a streaming simulation that replays sensor telemetry through stateful feature computation and online scoring, with a parity test confirming live scores match batch scores.

Replace N, X and Y with your results. Drop "LSTM autoencoder" if you skip the stretch item.

**Filled in from the final runs.** Note that "detected" and "flagged early" are different claims. Use the honest version:

- Built unsupervised anomaly detection (±3σ control limits, Isolation Forest, LSTM autoencoder) and a vibration health indicator, back-tested against documented failures. **All 4 air leaks were detected within 1–4 h of onset at ≤ 1.7 false alerts/month (LSTM-AE / Isolation Forest), with 2 of 4 flagged a median 52 h ahead (control limits)**. Bearing degradation onset was flagged **a median 65 h before failure (4/4 bearings, 4/4 fault types correctly diagnosed)**.
