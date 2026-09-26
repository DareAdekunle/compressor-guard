"""
Project paths and YAML config loading.

Every path in `config/*.yaml` is relative to the project root, so code resolves
it here. That way notebooks (run from `notebooks/`), scripts and the API all
find the same files whatever the working directory is.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def resolve_path(path: Union[str, Path]) -> Path:
    """Return `path` unchanged if absolute, else resolve it against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def load_params(path: Union[str, Path] = "config/params.yaml") -> Dict[str, Any]:
    return load_yaml(path)


def load_costs(path: Union[str, Path] = "config/costs.yaml") -> Dict[str, Any]:
    return load_yaml(path)
