"""
Per-snapshot vibration features for the IMS bearings.

For each 1 s snapshot and each bearing channel:
- time domain: RMS, peak, peak-to-peak, crest factor, kurtosis, skewness
- spectrum: energy in +/-tol bands around the first `n_harmonics` of BPFO, BPFI, BSF
  (2x BSF for rollers, which strike both races) plus high-frequency energy
- envelope spectrum (band-pass -> Hilbert -> FFT): amplitude at BPFO, BPFI, 2xBSF and
  FTF, each divided by the median envelope level (a unitless SNR). This is the classic
  early bearing-defect detector.

Output: one row per (snapshot, bearing, channel) -> data/processed/ims_test{N}_features.parquet
"""

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

from compressor_guard.config import load_params, resolve_path
from compressor_guard.vibration.io import bearing_channels, list_snapshots, load_snapshot

FAULTS = ("bpfo", "bpfi", "bsf", "ftf")


def defect_frequencies(shaft_hz: float, n_rollers: int, pitch_d: float, roller_d: float,
                       contact_deg: float) -> Dict[str, float]:
    """Standard rolling-element bearing kinematics."""
    r = roller_d / pitch_d * np.cos(np.deg2rad(contact_deg))
    return {
        "bpfo": n_rollers / 2 * shaft_hz * (1 - r),
        "bpfi": n_rollers / 2 * shaft_hz * (1 + r),
        "bsf": pitch_d / (2 * roller_d) * shaft_hz * (1 - r ** 2),
        "ftf": shaft_hz / 2 * (1 - r),
    }


def fault_freqs_from_params(ims: Dict) -> Dict[str, float]:
    g = ims.get("bearing_geometry")
    if g:
        return defect_frequencies(ims["shaft_speed_rpm"] / 60.0, g["n_rollers"], g["pitch_diameter_in"],
                                  g["roller_diameter_in"], g["contact_angle_deg"])
    return dict(ims["fault_frequencies_hz"])


class FeatureExtractor:
    def __init__(self, fs: float, freqs: Dict[str, float], tol: float = 0.05, n_harmonics: int = 3,
                 env_band=(2000.0, 8000.0), n_points: int = 20480):
        self.fs, self.freqs, self.tol, self.nh = fs, freqs, tol, n_harmonics
        self.sos = butter(4, list(env_band), btype="band", fs=fs, output="sos")
        self.window = np.hanning(n_points)
        self.f = np.fft.rfftfreq(n_points, 1.0 / fs)
        self.n_points = n_points
        # characteristic line per fault: rollers hit both races, so use 2x BSF
        self.lines = {"bpfo": freqs["bpfo"], "bpfi": freqs["bpfi"], "bsf": 2 * freqs["bsf"], "ftf": freqs["ftf"]}
        self._masks = {k: self._band_mask(v) for k, v in self.lines.items()}
        self._hf = self.f >= 5000.0

    def _band_mask(self, f0: float) -> np.ndarray:
        m = np.zeros_like(self.f, dtype=bool)
        for h in range(1, self.nh + 1):
            m |= np.abs(self.f - h * f0) <= self.tol * h * f0
        return m

    def _first_harmonic(self, f0: float) -> np.ndarray:
        return np.abs(self.f - f0) <= self.tol * f0

    def channel_features(self, x: np.ndarray) -> Dict[str, float]:
        x = x.astype(np.float64)
        x = x - x.mean()
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        out = {
            "rms": rms,
            "peak": peak,
            "p2p": float(x.max() - x.min()),
            "crest": peak / rms if rms > 0 else np.nan,
            "kurtosis": float(kurtosis(x, fisher=False)),
            "skewness": float(skew(x)),
        }
        spec = np.abs(np.fft.rfft(x * self.window)) ** 2 / self.n_points
        total = spec.sum()
        for k, m in self._masks.items():
            out[f"band_{k}"] = float(spec[m].sum() / total)
        out["hf_energy"] = float(spec[self._hf].sum() / total)

        env = np.abs(hilbert(sosfiltfilt(self.sos, x)))
        env = env - env.mean()
        espec = np.abs(np.fft.rfft(env * self.window)) / self.n_points
        floor = np.median(espec[(self.f > 10) & (self.f < 1000)]) + 1e-12
        for k, f0 in self.lines.items():
            out[f"env_{k}"] = float(espec[self._first_harmonic(f0)].max() / floor)
        return out


def _process_file(path, timestamp, hours, channel_map, extractor: FeatureExtractor) -> List[Dict]:
    data = load_snapshot(path)
    rows = []
    for bearing, chans in channel_map.items():
        for ci, ch in enumerate(chans):
            r = extractor.channel_features(data[:, ch])
            r.update(timestamp=timestamp, hours=hours, bearing=bearing, channel=ci + 1)
            rows.append(r)
    return rows


def extract_test_features(test: int, params: Optional[Dict] = None, n_jobs: int = -1,
                          limit: Optional[int] = None, verbose: int = 0) -> pd.DataFrame:
    params = params or load_params()
    ims = params["ims"]
    snaps = list_snapshots(test, ims.get("raw_dir", "data/raw/ims"))
    if limit:
        snaps = snaps.iloc[:limit]
    ex = FeatureExtractor(fs=ims["sampling_frequency_hz"], freqs=fault_freqs_from_params(ims),
                          tol=ims.get("band_tolerance", 0.05), n_harmonics=ims.get("n_harmonics", 3),
                          env_band=tuple(ims.get("envelope_band_hz", (2000, 8000))),
                          n_points=ims.get("points_per_file", 20480))
    cmap = bearing_channels(test)
    rows = Parallel(n_jobs=n_jobs, batch_size=16, verbose=verbose)(
        delayed(_process_file)(p, t, h, cmap, ex)
        for p, t, h in zip(snaps["path"], snaps["timestamp"], snaps["hours"]))
    df = pd.DataFrame([r for rs in rows for r in rs])
    df.insert(0, "test", test)
    lead = ["test", "timestamp", "hours", "bearing", "channel"]
    return df[lead + [c for c in df.columns if c not in lead]].sort_values(["bearing", "channel", "timestamp"]).reset_index(drop=True)


def features_path(test: int, params: Optional[Dict] = None):
    params = params or load_params()
    return resolve_path(params["ims"].get("processed_dir", "data/processed")) / f"ims_test{test}_features.parquet"


def load_test_features(test: int, params: Optional[Dict] = None) -> pd.DataFrame:
    p = features_path(test, params)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run `python -m compressor_guard.vibration.features --test {test}`.")
    return pd.read_parquet(p)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Extract IMS vibration features (one row per snapshot/channel).")
    ap.add_argument("--test", type=int, nargs="+", default=[2], choices=[1, 2, 3])
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N snapshots (smoke test)")
    args = ap.parse_args(argv)
    params = load_params(args.config)
    for t in args.test:
        df = extract_test_features(t, params, n_jobs=args.n_jobs, limit=args.limit, verbose=5)
        out = features_path(t, params)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[ims] test {t}: {df['timestamp'].nunique():,} snapshots -> {out}")


if __name__ == "__main__":
    main()
