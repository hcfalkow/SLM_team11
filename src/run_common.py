"""Shared runtime utilities for training and inference flows."""

import csv
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from codecarbon import EmissionsTracker, OfflineEmissionsTracker


def parse_bool(value: str) -> bool:
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def load_emissions_metrics(emissions_csv: Path) -> dict[str, float | None]:
    if not emissions_csv.exists():
        return {"energy_kwh": None, "emissions_kg": None}

    with emissions_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"energy_kwh": None, "emissions_kg": None}

    last_row = rows[-1]
    energy_kwh = float(last_row["energy_consumed"]) if last_row.get("energy_consumed") else None
    emissions_kg = float(last_row["emissions"]) if last_row.get("emissions") else None
    return {"energy_kwh": energy_kwh, "emissions_kg": emissions_kg}


def collect_model_stats(model: torch.nn.Module) -> dict[str, int | float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    state_dict_bytes = parameter_bytes + buffer_bytes
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "state_dict_bytes_estimate": int(state_dict_bytes),
        "state_dict_mb_estimate": round(state_dict_bytes / (1024**2), 4),
    }


def cuda_memory_stats(device: str) -> dict[str, float | int | str | None]:
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return {
            "device": str(device),
            "gpu_name": None,
            "peak_allocated_bytes": None,
            "peak_allocated_mb": None,
            "peak_reserved_bytes": None,
            "peak_reserved_mb": None,
        }

    cuda_device = torch.device(device)
    allocated_bytes = int(torch.cuda.max_memory_allocated(cuda_device))
    reserved_bytes = int(torch.cuda.max_memory_reserved(cuda_device))
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(cuda_device),
        "peak_allocated_bytes": allocated_bytes,
        "peak_allocated_mb": round(allocated_bytes / (1024**2), 4),
        "peak_reserved_bytes": reserved_bytes,
        "peak_reserved_mb": round(reserved_bytes / (1024**2), 4),
    }


def build_tracker(
    *,
    project_name: str,
    output_dir: Path,
    output_file: str,
    offline: bool,
    country_iso_code: str,
) -> EmissionsTracker | OfflineEmissionsTracker:
    tracker_kwargs = {
        "project_name": project_name,
        "output_dir": str(output_dir),
        "output_file": output_file,
        "save_to_file": True,
        "log_level": "error",
    }
    if offline:
        return OfflineEmissionsTracker(country_iso_code=country_iso_code, **tracker_kwargs)
    return EmissionsTracker(**tracker_kwargs)


def reset_cuda_peak_memory(device: str) -> None:
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def make_amp_ctx_factory(device: str, dtype: str):
    ptdtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if device_type == "cuda" and dtype in {"float16", "bfloat16"}:

        def amp_ctx_factory():
            return torch.autocast(device_type=device_type, dtype=ptdtype)

        return amp_ctx_factory

    def amp_ctx_factory():
        return nullcontext()

    return amp_ctx_factory
