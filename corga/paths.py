"""Filesystem locations of the bundled data and config (repo root in development, /app in Docker)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT_DIR / "conf"
DATA_DIR = ROOT_DIR / "data"
DATASET_PATH = DATA_DIR / "corfu_dataset.json"
RAIN_FIXTURE_PATH = DATA_DIR / "corfu_synthetic_rain.json"
