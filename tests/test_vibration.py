import numpy as np
import pandas as pd
import pytest

from compressor_guard.config import load_params
from compressor_guard.vibration.features import FeatureExtractor, defect_frequencies, fault_freqs_from_params
from compressor_guard.vibration.health import detect_onset, health_indicator
from compressor_guard.vibration.rul import fit_exponential

FS, N = 20480, 20480


def test_defect_frequencies_match_config():
    ims = load_params()["ims"]
    f = fault_freqs_from_params(ims)
    for k, v in ims["fault_frequencies_hz"].items():
        assert abs(f[k] - v) < 0.2, k


def _bearing_signal(defect_hz=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = rng.normal(0, 0.05, N)
    if defect_hz:
        # impacts at the defect rate exciting a 4 kHz resonance
        for t0 in np.arange(0, 1, 1 / defect_hz):
            i = int(t0 * FS)
            k = np.arange(0, min(200, N - i))
            x[i:i + len(k)] += 0.5 * np.exp(-k / 30) * np.sin(2 * np.pi * 4000 * k / FS)
    return x


def test_envelope_detects_outer_race_defect():
    freqs = fault_freqs_from_params(load_params()["ims"])
    ex = FeatureExtractor(FS, freqs)
    healthy = ex.channel_features(_bearing_signal())
    faulty = ex.channel_features(_bearing_signal(freqs["bpfo"]))
    assert faulty["env_bpfo"] > 5 * healthy["env_bpfo"]
    assert faulty["env_bpfo"] > faulty["env_bpfi"]
    assert faulty["kurtosis"] > healthy["kurtosis"]


def test_onset_needs_persistence():
    hi = pd.Series([0, 0, 20, 0, 20, 20, 20, 20])
    hrs = pd.Series(np.arange(8) * 1.0)
    assert detect_onset(hi, hrs, k=10, persistence=3) == 6.0
    assert detect_onset(hi, hrs, k=50, persistence=1) is None


def test_cross_bearing_reference_cancels_common_mode():
    """A rig-wide step change (all bearings x2) must not raise the HI."""
    rows = []
    for i in range(400):
        for b in (1, 2, 3, 4):
            level = 2.0 if i > 200 else 1.0
            if b == 1 and i > 350:
                level *= 5.0                       # genuine local damage on bearing 1
            rows.append({"test": 2, "timestamp": pd.Timestamp("2004-01-01") + pd.Timedelta(minutes=10 * i),
                         "hours": i / 6, "bearing": b, "channel": 1,
                         **{c: level * (1 + 0.01 * np.sin(i + b)) for c in
                            ["rms", "kurtosis", "env_bpfo", "env_bpfi", "env_bsf"]}})
    hi = health_indicator(pd.DataFrame(rows), ["rms", "kurtosis", "env_bpfo"], baseline_hours=24, skip_hours=0,
                          smoothing=3, reference="cross_bearing")
    b2 = hi[(hi["bearing"] == 2) & (hi["hours"].between(34, 58))]
    assert b2["hi"].max() < 3, "common-mode step leaked into the HI"
    b1 = hi[(hi["bearing"] == 1) & (hi["hours"] > 60)]
    assert b1["hi"].max() > 10


def test_exponential_fit_recovers_rate():
    h = np.linspace(0, 50, 60)
    a, b = fit_exponential(h + 100, 1.5 * np.exp(0.03 * h), 100)
    assert np.isclose(b, 0.03, atol=1e-6) and np.isclose(np.exp(a), 1.5, atol=1e-6)


def test_alarm_episodes_reset_and_max_lead():
    from compressor_guard.vibration.health import alarm_episodes, summarise_test, evaluate_onsets
    hi = pd.Series([20, 20, 0, 0, 20, 20, 20, 0, 20, 20])
    hrs = pd.Series(np.arange(10) * 100.0)
    assert alarm_episodes(hi, hrs, k=10, persistence=2) == [100.0, 500.0, 900.0]
    # failed bearing (test 2, B1): only the alarm within 168 h of end-of-life (900 h) counts
    df = pd.DataFrame({"test": 2, "bearing": 1, "hours": hrs, "hi": hi,
                       **{f"z_{c}": 1.0 for c in ["env_bpfo", "env_bpfi", "env_bsf"]}})
    s = summarise_test(df, k=10, persistence=2, max_lead_hours=168)
    assert s.loc[0, "onset_hours"] == 900.0 and s.loc[0, "early_false_alarms"] == 2
    assert evaluate_onsets(s)["false_alarms"] == 2
