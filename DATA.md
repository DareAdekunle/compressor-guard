# Data Acquisition Guide

Raw data is **not committed** to this repo (`data/*` is in `.gitignore`). Download it with the steps below or with `python scripts/download_data.py`.

Target layout:

```
data/
├── raw/
│   ├── metropt3/
│   │   └── MetroPT3(AirCompressor).csv
│   └── ims/
│       ├── 1st_test/1st_test/          # 8 channels (Kaggle nests each test one level deep)
│       ├── 2nd_test/2nd_test/          # 4 channels
│       └── 3rd_test/4th_test/txt/      # 4 channels (yes, test 3 unpacks as "4th_test/txt")
└── processed/             # parquet outputs from the pipelines (regenerated, never committed)
```

---

## 1. MetroPT-3 (Module A)

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset **791** |
| Page | https://archive.ics.uci.edu/dataset/791/metropt+3+dataset |
| Size | ~208 MB CSV |
| Licence | CC BY 4.0 (credit the authors) |

**Option A: browser.** Click **Download** on the UCI page, unzip, and move the CSV to `data/raw/metropt3/`.

**Option B: Python.**
```bash
pip install ucimlrepo
```
```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=791)
df = ds.data.original            # full table incl. timestamp
df.to_csv("data/raw/metropt3/MetroPT3(AirCompressor).csv", index=False)
```
If `ucimlrepo` fails (large datasets sometimes do), use Option A.

**Option C: Kaggle mirror.** https://www.kaggle.com/datasets/pattinson9999/uci-metropt-3-dataset

### Failure labels
The CSV has **no failure column**. The operator's failure report is published on the UCI page, and the paper is [Scientific Data, 2022](https://www.nature.com/articles/s41597-022-01877-3). It was checked on 2026-09-26 and encoded in `config/failures.yaml`:

| Nr | Start | End | Failure | Report |
|---|---|---|---|---|
| F1 | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak | — |
| F2 | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak | Maintenance on 30Apr at 12:00 |
| F3 | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak | Maintenance on 8Jun at 16:00 |
| F4 | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak | Maintenance on 16Jul at 00:00 |

> ✅ Start and end times match the source. The source table numbers row 2 as "#1", and its "30Apr" maintenance date comes before the failure; we read it as **30 May 12:00**. The maintenance times are stored as `maintenance:` and close each incident in the back-test.

### Quick sanity checks (all verified in notebook 01)
- 1,516,948 rows, 1 Feb – 1 Sep 2020, ~97 % of intervals 9–11 s, 0 nulls. The UCI text says "logged at 1 Hz", so the public CSV is a ~10× downsample.
- `timestamp` parses cleanly; digital columns are 0/1
- **`COMP` = 1 means the compressor is *not* taking in air** (off or offloaded). Loaded = `COMP == 0` (motor current ~5.6 A).
- **Frozen-logger periods:** ~10 k minutes where all analog sensors hold a constant value. The pipeline masks them automatically.
- Plot `Motor_current` and `Oil_temperature` around each failure window before modelling

---

## 2. NASA IMS Bearing Dataset (Module B)

| | |
|---|---|
| Source | IMS, University of Cincinnati, via the NASA Prognostics Data Repository |
| NASA page | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ ("Bearings") |
| Kaggle mirror (easiest) | https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset |
| Format | Thousands of ASCII files, one per snapshot, **named by timestamp** (e.g. `2004.02.12.10.32.39`) |
| Size | ~1 GB+ compressed, several GB extracted, so check disk space first |

**Option A: Kaggle CLI (recommended).**
```bash
pip install kaggle
# 1. kaggle.com → Settings → API → "Create New Token" → downloads kaggle.json
# 2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d vinayak123tyagi/bearing-dataset -p data/raw/ims --unzip
```
> Folder names inside the Kaggle archive can be nested or inconsistent (e.g. a test folder inside another folder). The loader in `src/compressor_guard/vibration/io.py` **searches for the three test folders** rather than hard-coding paths.

**Option B: NASA direct.** Download "Bearings" from the NASA page. The archive is a `.7z` containing `.rar` files per test, so extract with `7z x` then `unrar x` (or 7-Zip on Windows).

### Structure to expect

| Test | Channels | Files (approx.) | Failure |
|---|---|---|---|
| 1st_test | 8 (bearing 1: ch 1–2, bearing 2: ch 3–4, bearing 3: ch 5–6, bearing 4: ch 7–8) | ~2,150 | B3 inner race, B4 roller element |
| 2nd_test | 4 (one per bearing) | ~980 | B1 outer race |
| 3rd_test | 4 (one per bearing) | 6,324 on disk (readme: 4,448) | B3 outer race |

Each file: **20,480 rows × channels**, tab-separated, no header, one 1-second snapshot. The readme says "20 kHz", but the data are consistent with **20,480 Hz**: the outer-race envelope peak of test 2 B1 lands at 236.0 Hz vs the geometric BPFO of 236.4 Hz (at 20 kHz it would be 230.5 Hz). Test 3 on disk runs to 18 Apr 2004, past the readme's 4 Apr.

### Loading pattern
```python
from pathlib import Path
import numpy as np, pandas as pd

def load_snapshot(path):
    return np.loadtxt(path)                      # shape (20480, n_channels)

def iter_test(folder):
    files = sorted(Path(folder).glob("*"))       # names sort chronologically
    for f in files:
        ts = pd.to_datetime(f.name, format="%Y.%m.%d.%H.%M.%S")
        yield ts, load_snapshot(f)
```

`src/compressor_guard/vibration/io.py` implements this with automatic folder discovery.

**Performance:** features are computed file by file with `joblib.Parallel` and saved as **one row per snapshot and channel** to `data/processed/ims_test{N}_features.parquet`. A whole raw test is never loaded into memory. All three tests take ~2 min on 8 cores.

**Start with Test 2.** It's the cleanest (4 channels, ~980 files, clear outer-race degradation). Get the full pipeline working there, then run Tests 1 and 3.

### Bearing fault frequencies
Bearings are Rexnord ZA-2115 (16 rollers per row, pitch diameter 2.815 in, roller diameter 0.331 in, contact angle 15.17°) at 2,000 RPM. `config/params.yaml` stores the geometry, and the code derives **BPFO 236.4 Hz, BPFI 296.9 Hz, BSF 139.9 Hz, FTF 14.8 Hz** (verified in notebook 04 and `tests/test_vibration.py`).

---

## 3. Citation / licence checklist

- [x] MetroPT-3 credited per CC BY 4.0 (README §11)
- [x] IMS credited to IMS, University of Cincinnati & NASA PCoE (README §11)
- [x] `data/*` in `.gitignore`, so no raw data is pushed to GitHub
