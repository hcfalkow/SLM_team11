"""Shared project paths used by training and inference scripts."""

from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
CKPT_PATH = OUT_DIR / "ckpt.pt"
RUNS_DIR = OUT_DIR / "runs"
SUMMARY_DIR = OUT_DIR / "scenario_summaries"


def now_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def phase_runs_dir(phase: str) -> Path:
    return RUNS_DIR / phase


def scenario_runs_dir(phase: str, scenario_id: str) -> Path:
    return phase_runs_dir(phase) / scenario_id


def make_run_dir(phase: str, scenario_id: str, run_id: str | None = None) -> tuple[str, Path]:
    resolved_run_id = run_id or now_timestamp()
    run_dir = scenario_runs_dir(phase, scenario_id) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return resolved_run_id, run_dir
