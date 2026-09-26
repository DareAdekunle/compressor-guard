"""
Unsupervised Anomaly Detection Models for Compressor Condition Monitoring:
1. Baseline Statistical Process Control Limits (±3σ)
2. Multidimensional Isolation Forest
3. Deep Learning Sequence Reconstruction (PyTorch LSTM Autoencoder)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ControlLimitsDetector:
    """
    Statistical Process Control (SPC) Limits (±3σ baseline detector).
    Calculates upper and lower control limits on healthy training data.
    """

    def __init__(self, sigma_multiplier: float = 3.0, feature_cols: Optional[List[str]] = None):
        self.sigma_multiplier = sigma_multiplier
        self.feature_cols = feature_cols or []
        self.limits: Dict[str, Tuple[float, float, float, float]] = {}
        # Alert threshold on the score (max |z| across features) is the sigma multiplier.
        self.threshold: float = sigma_multiplier

    def fit(self, df_train: pd.DataFrame) -> "ControlLimitsDetector":
        """Compute mean and std on healthy training data."""
        self.limits = {}
        for col in self.feature_cols:
            vals = df_train[col].dropna()
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma < 1e-6:
                sigma = 1e-3
            ucl = mu + self.sigma_multiplier * sigma
            lcl = mu - self.sigma_multiplier * sigma
            self.limits[col] = (mu, sigma, lcl, ucl)
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute an aggregate anomaly score based on maximum normalized excursion
        beyond the mean across monitored features.
        """
        excursions = []
        for col in self.feature_cols:
            if col in self.limits:
                mu, sigma, _, _ = self.limits[col]
                z = np.abs((df[col].fillna(mu) - mu) / sigma)
                excursions.append(z)
        if not excursions:
            return np.zeros(len(df))
        stacked = np.column_stack(excursions)
        # Score is maximum z-score across monitored features
        return np.max(stacked, axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return boolean mask where score exceeds the sigma multiplier."""
        scores = self.score_samples(df)
        return scores > self.sigma_multiplier


class IsolationForestDetector:
    """
    Multidimensional Anomaly Detection using scikit-learn Isolation Forest.
    Fitted strictly on healthy operational data.
    """

    def __init__(
        self,
        feature_cols: List[str],
        contamination: float = 0.015,
        n_estimators: int = 150,
        max_samples: float = 0.8,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=1,  # deterministic scores (and parity between batch and stream)
        )
        self.is_fitted = False
        self.threshold: float = 0.0

    def fit(self, df_train: pd.DataFrame) -> "IsolationForestDetector":
        """Fit scaler and Isolation Forest on healthy training data."""
        # Missing window features (e.g. no idle period in the last hour) are imputed with
        # the healthy training median, which is fixed at fit time and so streaming-safe.
        self.fill_values = df_train[self.feature_cols].median()
        X_train = df_train[self.feature_cols].fillna(self.fill_values)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        self.is_fitted = True

        # In sklearn, decision_function returns negative values for anomalies.
        # We invert it so higher score = more anomalous.
        train_raw = -self.model.decision_function(X_scaled)
        # Threshold at chosen contamination percentile of healthy data
        self.threshold = float(np.percentile(train_raw, 100 * (1 - self.contamination)))
        return self

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute inverted decision function scores (higher = more anomalous).
        Normalized so 0 is typical healthy median and > 1.0 indicates strong anomaly.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        X = df[self.feature_cols].fillna(self.fill_values)
        X_scaled = self.scaler.transform(X)
        raw_scores = -self.model.decision_function(X_scaled)
        return raw_scores

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Return boolean indicator for samples exceeding threshold."""
        th = threshold if threshold is not None else self.threshold
        scores = self.score_samples(df)
        return scores > th


class _LSTMEncoderDecoder(nn.Module):
    """Internal PyTorch Module for Sequence Reconstruction."""

    def __init__(self, n_features: int, hidden_dim: int = 32, latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, n_features)
        batch_size, seq_len, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (num_layers, batch_size, hidden_dim) -> use last layer
        last_hidden = h_n[-1]
        latent = self.enc_to_latent(last_hidden)
        
        # Decode: repeat latent vector across sequence length
        dec_input = self.latent_to_dec(latent).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_input)
        reconstruction = self.output_layer(dec_out)
        return reconstruction


class LSTMAutoencoderDetector:
    """
    Multivariate sequence anomaly detection with a PyTorch LSTM autoencoder.
    Each minute is scored by the reconstruction error of the window that *ends* at that
    minute, so scoring is causal. Training windows that span a data gap are dropped.
    """

    def __init__(
        self,
        feature_cols: List[str],
        seq_len: int = 30,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 1,
        batch_size: int = 256,
        epochs: int = 8,
        learning_rate: float = 0.002,
        contamination: float = 0.015,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.device = torch.device("cpu")  # small model; CPU keeps runs reproducible
        self.model: Optional[_LSTMEncoderDecoder] = None
        self.threshold: float = 0.0
        self.train_loss: List[float] = []

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Sliding windows: shape (n_samples - seq_len + 1, seq_len, n_features)."""
        n_samples, n_features = data.shape
        if n_samples < self.seq_len:
            raise ValueError(f"Data length {n_samples} is less than sequence length {self.seq_len}")
        sequences = np.lib.stride_tricks.sliding_window_view(data, window_shape=(self.seq_len, n_features))
        return sequences.squeeze(axis=1)

    def _contiguous_windows(self, timestamps: pd.Series) -> np.ndarray:
        """True for windows whose seq_len minutes are consecutive (no gap inside)."""
        t = pd.to_datetime(timestamps).values.astype("datetime64[m]").astype(np.int64)
        span = t[self.seq_len - 1:] - t[: len(t) - self.seq_len + 1]
        return span == self.seq_len - 1

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].fillna(self.fill_values).to_numpy(dtype=np.float64)
        return self.scaler.transform(X).astype(np.float32)

    def fit(self, df_train: pd.DataFrame) -> "LSTMAutoencoderDetector":
        """Fit scaler and train the autoencoder on healthy, gap-free windows."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.fill_values = df_train[self.feature_cols].median()
        self.scaler.fit(df_train[self.feature_cols].fillna(self.fill_values).to_numpy(dtype=np.float64))
        X_scaled = self._prepare(df_train)
        sequences = self._create_sequences(X_scaled)
        if "timestamp" in df_train.columns:
            sequences = sequences[self._contiguous_windows(df_train["timestamp"])]

        g = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(sequences))),
                            batch_size=self.batch_size, shuffle=True, generator=g)

        self.model = _LSTMEncoderDecoder(
            n_features=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loss = []
        self.model.train()
        for _ in range(self.epochs):
            total = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_x)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch_x)
            self.train_loss.append(total / len(sequences))

        train_scores = self.score_samples(df_train)
        self.threshold = float(np.percentile(train_scores, 100 * (1 - self.contamination)))
        return self

    def score_sequences(self, scaled_data: np.ndarray) -> np.ndarray:
        """Reconstruction MSE per window."""
        self.model.eval()
        sequences = self._create_sequences(scaled_data)
        loader = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(sequences))),
                            batch_size=self.batch_size * 8, shuffle=False)
        scores = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                err = (self.model(batch_x) - batch_x) ** 2
                scores.append(err.mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(scores)

    def score_samples(self, df: pd.DataFrame) -> np.ndarray:
        """Score each row by the window ending at it. The first seq_len-1 rows reuse the first score."""
        seq_scores = self.score_sequences(self._prepare(df))
        pad_len = len(df) - len(seq_scores)
        return np.pad(seq_scores, (pad_len, 0), mode="edge") if pad_len > 0 else seq_scores

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return self.score_samples(df) > th


# --------------------------------------------------------------------------------------
# Training / scoring pipeline
# --------------------------------------------------------------------------------------
MODEL_NAMES = ("control_limits", "isolation_forest", "lstm_autoencoder")


def training_mask(features: pd.DataFrame, params: Dict) -> pd.Series:
    """
    Healthy training rows: inside metropt.train_split (ends 8 days before F1) and outside
    any metropt.train_exclude episode. Failure labels are never used to pick training data.
    """
    m = params["metropt"]
    ts = features["timestamp"]
    mask = (ts >= pd.Timestamp(m["train_split"]["start"])) & (ts <= pd.Timestamp(m["train_split"]["end"]))
    for ep in m.get("train_exclude", []) or []:
        mask &= ~((ts >= pd.Timestamp(ep["start"])) & (ts <= pd.Timestamp(ep["end"])))
    return mask


def build_detectors(params: Dict, include_lstm: bool = True) -> Dict[str, object]:
    cfg = params["metropt"]["models"]
    cl, iso, ls = cfg["control_limits"], cfg["isolation_forest"], cfg["lstm_autoencoder"]
    dets: Dict[str, object] = {
        "control_limits": ControlLimitsDetector(cl["sigma"], cl["features"]),
        "isolation_forest": IsolationForestDetector(
            feature_cols=iso["features"], contamination=iso["contamination"],
            n_estimators=iso["n_estimators"], max_samples=iso["max_samples"],
            random_state=iso["random_state"]),
    }
    if include_lstm:
        dets["lstm_autoencoder"] = LSTMAutoencoderDetector(
            feature_cols=ls["features"], seq_len=ls["sequence_length"], hidden_dim=ls["hidden_dim"],
            latent_dim=ls["latent_dim"], num_layers=ls["num_layers"], batch_size=ls["batch_size"],
            epochs=ls["epochs"], learning_rate=ls["learning_rate"],
            contamination=ls.get("contamination", iso["contamination"]), random_state=ls["random_state"])
    return dets


def calibrate_threshold(detector, train_scores: np.ndarray, params: Dict,
                        timestamps: Optional[pd.Series] = None) -> float:
    """
    Alert threshold for IF / LSTM: a high quantile of the *smoothed* training scores, the
    same quantity the alert rule compares against. Control limits keep their k-sigma rule.
    """
    from compressor_guard.alerts import smooth_anomaly_scores  # avoid import cycle
    if isinstance(detector, ControlLimitsDetector):
        return detector.sigma_multiplier
    a = params["metropt"]["alert_logic"]
    q = a.get("threshold_quantile", 0.995)
    sm = smooth_anomaly_scores(train_scores, window_steps=a["smoothing_window_minutes"], timestamps=timestamps)
    return float(np.quantile(sm, q))


def train_detectors(features: pd.DataFrame, params: Dict, include_lstm: bool = True,
                    verbose: bool = True) -> Dict[str, object]:
    train = features.loc[training_mask(features, params)].reset_index(drop=True)
    dets = build_detectors(params, include_lstm)
    for name, det in dets.items():
        if verbose:
            print(f"[models] fitting {name} on {len(train):,} healthy minutes ...", flush=True)
        det.fit(train)
        det.threshold = calibrate_threshold(det, det.score_samples(train), params, train["timestamp"])
    return dets


def score_detectors(dets: Dict[str, object], features: pd.DataFrame) -> pd.DataFrame:
    out = features[["timestamp", "shift", "is_failure", "failure_id"]].copy()
    for name, det in dets.items():
        out[f"score_{name}"] = det.score_samples(features)
    return out


def save_detectors(dets: Dict[str, object], model_dir: Union[str, Path] = "models") -> Path:
    from compressor_guard.config import resolve_path
    d = resolve_path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, det in dets.items():
        joblib.dump(det, d / f"{name}.joblib")
    (d / "thresholds.json").write_text(json.dumps({n: float(x.threshold) for n, x in dets.items()}, indent=2))
    return d


def load_detectors(model_dir: Union[str, Path] = "models", names=MODEL_NAMES) -> Dict[str, object]:
    from compressor_guard.config import resolve_path
    d = resolve_path(model_dir)
    return {n: joblib.load(d / f"{n}.joblib") for n in names if (d / f"{n}.joblib").exists()}


def _log_mlflow(dets: Dict[str, object], params: Dict, n_train: int) -> None:
    """Log one MLflow run per detector if MLflow is installed; silently skip otherwise."""
    try:
        import mlflow
    except ImportError:
        return
    from compressor_guard.config import resolve_path
    cfg = params.get("mlflow", {})
    uri = cfg.get("tracking_uri", "sqlite:///mlflow.db")
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{resolve_path(uri[len('sqlite:///'):])}"   # anchor at project root
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.get("experiment", "compressor-guard-module-a"))
        for name, det in dets.items():
            with mlflow.start_run(run_name=name):
                hp = {k: v for k, v in vars(det).items() if isinstance(v, (int, float, str)) and k != "threshold"}
                hp["feature_cols"] = ",".join(det.feature_cols)
                hp["n_train_minutes"] = n_train
                mlflow.log_params(hp)
                mlflow.log_metric("alert_threshold", float(det.threshold))
                for i, loss in enumerate(getattr(det, "train_loss", []) or []):
                    mlflow.log_metric("train_mse", loss, step=i)
        print(f"[models] logged runs to MLflow at {uri}")
    except Exception as e:  # tracking must never break training
        print(f"[models] MLflow logging skipped: {e}")


def main(argv: Optional[List[str]] = None) -> None:
    from compressor_guard.config import load_params, resolve_path
    from compressor_guard.features import load_features_parquet

    ap = argparse.ArgumentParser(description="Train and score Module A anomaly detectors.")
    ap.add_argument("--config", default="config/params.yaml")
    ap.add_argument("--train", action="store_true", help="fit detectors (otherwise load from models/)")
    ap.add_argument("--no-lstm", action="store_true", help="skip the LSTM autoencoder")
    args = ap.parse_args(argv)

    params = load_params(args.config)
    m = params["metropt"]
    feats = load_features_parquet(m["processed_parquet_path"])
    if args.train:
        dets = train_detectors(feats, params, include_lstm=not args.no_lstm)
        save_detectors(dets, m.get("model_dir", "models"))
    else:
        dets = load_detectors(m.get("model_dir", "models"))
    scores = score_detectors(dets, feats)
    out = resolve_path(m.get("scores_parquet_path", "data/processed/metropt_scores.parquet"))
    scores.to_parquet(out, index=False)
    if args.train:
        _log_mlflow(dets, params, int(training_mask(feats, params).sum()))
    for n, d in dets.items():
        print(f"[models] {n:18s} threshold={d.threshold:.4f}")
    print(f"[models] scores -> {out}")


if __name__ == "__main__":
    # Run via the imported module, so pickled detectors reference
    # `compressor_guard.models.<Class>` rather than `__main__.<Class>`.
    from compressor_guard.models import main as _main
    _main()
