"""Shared project paths used by training and inference scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
CKPT_PATH = OUT_DIR / "ckpt.pt"
