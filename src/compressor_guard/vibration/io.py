"""
IMS bearing dataset loading.

The Kaggle/NASA archives nest folders inconsistently. For example, test 3 unpacks as
`3rd_test/4th_test/txt/`. So instead of hard-coding paths, we search `raw_dir` for the
deepest folder of each test that holds the timestamp-named snapshot files.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from compressor_guard.config import resolve_path

SNAPSHOT_NAME = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
TEST_FOLDER_NAMES = {1: ("1st_test",), 2: ("2nd_test",), 3: ("3rd_test", "4th_test")}

# channel indices (0-based) per bearing, from the IMS readme
CHANNELS: Dict[int, Dict[int, List[int]]] = {
    1: {1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7]},
    2: {1: [0], 2: [1], 3: [2], 4: [3]},
    3: {1: [0], 2: [1], 3: [2], 4: [3]},
}

# Documented end-of-test failures: (test, bearing) -> failure mode
KNOWN_FAILURES: Dict[Tuple[int, int], str] = {
    (1, 3): "inner_race",
    (1, 4): "roller_element",
    (2, 1): "outer_race",
    (3, 3): "outer_race",
}
# fault-frequency feature expected to dominate for each failure mode
EXPECTED_BAND = {"inner_race": "bpfi", "outer_race": "bpfo", "roller_element": "bsf"}


def _has_snapshots(folder: Path, min_files: int = 10) -> bool:
    n = 0
    for p in folder.iterdir():
        if p.is_file() and SNAPSHOT_NAME.match(p.name):
            n += 1
            if n >= min_files:
                return True
    return False


@lru_cache(maxsize=None)
def find_test_folder(test: int, raw_dir: Union[str, Path] = "data/raw/ims") -> Path:
    """Locate the folder that holds test `test`'s snapshot files."""
    root = resolve_path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"IMS raw directory not found: {root} (see DATA.md §2)")
    candidates = []
    for name in TEST_FOLDER_NAMES[test]:
        for d in [root / name, *root.rglob(name)]:
            if not d.is_dir():
                continue
            for sub in [d, *[p for p in d.rglob("*") if p.is_dir()]]:
                if _has_snapshots(sub):
                    candidates.append(sub)
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No snapshot folder found for IMS test {test} under {root}")
    return sorted(set(candidates), key=lambda p: len(p.parts))[0]


def list_snapshots(test: int, raw_dir: Union[str, Path] = "data/raw/ims") -> pd.DataFrame:
    """One row per snapshot file: timestamp, path, hours since test start. Sorted by time."""
    folder = find_test_folder(test, raw_dir)
    files = [p for p in folder.iterdir() if p.is_file() and SNAPSHOT_NAME.match(p.name)]
    df = pd.DataFrame({"path": files})
    df["timestamp"] = pd.to_datetime([p.name for p in files], format="%Y.%m.%d.%H.%M.%S")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hours"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
    return df


def load_snapshot(path: Union[str, Path]) -> np.ndarray:
    """Load one snapshot: shape (20480, n_channels), float32."""
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float32).to_numpy()


def bearing_channels(test: int) -> Dict[int, List[int]]:
    return CHANNELS[test]
