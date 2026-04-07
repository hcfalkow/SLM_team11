"""Inference script with fixed-workload benchmark support and sustainability logging."""

import argparse
import csv
import datetime as dt
import json
import pickle
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from model import GPT, GPTConfig
from paths import CKPT_PATH, OUT_DIR, SUMMARY_DIR, make_run_dir, scenario_runs_dir

DEFAULT_PROMPT = "To be, or not to be"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_meta(data_dir: str | Path):
    meta_path = Path(data_dir) / "meta.pkl"
    with meta_path.open("rb") as f:
        return pickle.load(f)


def parse_bool(value: str) -> bool:
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible inference benchmark scenarios.")

    parser.add_argument("--scenario-id", type=str, default="inference_manual")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=str(CKPT_PATH))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32")

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-workload-file", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--codecarbon-offline", type=parse_bool, default=True)
    parser.add_argument("--country-iso-code", type=str, default="SWE")

    return parser.parse_args()


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_workload_file:
        prompt_path = Path(args.prompt_workload_file)
        with prompt_path.open("r", encoding="utf-8") as f:
            prompts = [line.strip("\n") for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"Prompt workload file is empty: {prompt_path}")
        return prompts
    return [args.prompt]


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


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = make_run_dir("inference", args.scenario_id, args.run_id)
    scenario_dir = scenario_runs_dir("inference", args.scenario_id)
    emissions_csv = run_dir / "emissions.csv"
    outputs_jsonl = run_dir / "generated_outputs.jsonl"

    ckpt = torch.load(args.checkpoint_path, map_location=args.device)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]  # char to index mapping
    itos = meta["itos"]  # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ptdtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    if device_type == "cuda" and args.dtype in {"float16", "bfloat16"}:

        def amp_ctx_factory():
            return torch.autocast(device_type=device_type, dtype=ptdtype)
    else:

        def amp_ctx_factory():
            return nullcontext()

    prompts = load_prompts(args)

    effective_config = {
        "phase": "inference",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "inference": {
            "checkpoint_path": str(args.checkpoint_path),
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "prompt_workload_file": args.prompt_workload_file,
            "prompt_count": len(prompts),
        },
        "checkpoint_model": model_cfg,
    }
    with (run_dir / "effective_config.json").open("w", encoding="utf-8") as f:
        json.dump(effective_config, f, indent=2)

    tracker_kwargs = {
        "project_name": f"inference_{args.scenario_id}",
        "output_dir": str(run_dir),
        "output_file": emissions_csv.name,
        "save_to_file": True,
        "log_level": "error",
    }
    if args.codecarbon_offline:
        tracker = OfflineEmissionsTracker(country_iso_code=args.country_iso_code, **tracker_kwargs)
    else:
        tracker = EmissionsTracker(**tracker_kwargs)

    total_prompt_tokens = 0
    total_generated_tokens = 0
    generated_rows: list[dict[str, object]] = []

    tracker.start()
    t0 = time.time()

    for prompt_idx, prompt_text in enumerate(prompts):
        idx = torch.tensor([encode(prompt_text)], dtype=torch.long, device=args.device)
        total_prompt_tokens += idx.size(1)

        with amp_ctx_factory():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        generated = out[0].tolist()
        generated_text = decode(generated)
        generated_tokens = len(generated) - idx.size(1)
        total_generated_tokens += generated_tokens

        generated_rows.append(
            {
                "prompt_index": prompt_idx,
                "prompt": prompt_text,
                "prompt_tokens": idx.size(1),
                "generated_tokens": generated_tokens,
                "output": generated_text,
            }
        )

    runtime_seconds = time.time() - t0
    tracker.stop()

    with outputs_jsonl.open("w", encoding="utf-8") as f:
        for row in generated_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    emissions_metrics = load_emissions_metrics(emissions_csv)

    run_metadata = {
        "phase": "inference",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "status": "completed",
        "runtime_seconds": runtime_seconds,
        "device": args.device,
        "dtype": args.dtype,
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_block_size": model_cfg.get("block_size"),
        "sampling": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
        "workload": {
            "prompt_workload_file": args.prompt_workload_file,
            "prompt_count": len(prompts),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
        },
        "codecarbon": {
            "offline": args.codecarbon_offline,
            "country_iso_code": args.country_iso_code,
            "energy_kwh": emissions_metrics["energy_kwh"],
            "emissions_kg": emissions_metrics["emissions_kg"],
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "effective_config_json": str(run_dir / "effective_config.json"),
            "emissions_csv": str(emissions_csv),
            "generated_outputs_jsonl": str(outputs_jsonl),
        },
    }

    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    with (scenario_dir / "latest_run.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "inference",
                "scenario_id": args.scenario_id,
                "run_id": run_id,
                "status": "completed",
                "run_metadata_path": str(run_dir / "run_metadata.json"),
            },
            f,
            indent=2,
        )

    append_row(
        SUMMARY_DIR / "inference_summary.csv",
        {
            "scenario_id": args.scenario_id,
            "run_id": run_id,
            "timestamp_utc": run_metadata["timestamp_utc"],
            "git_commit": run_metadata["git_commit"],
            "status": run_metadata["status"],
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "prompt_count": len(prompts),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "runtime_seconds": round(runtime_seconds, 4),
            "energy_kwh": emissions_metrics["energy_kwh"],
            "emissions_kg": emissions_metrics["emissions_kg"],
            "checkpoint_path": str(args.checkpoint_path),
            "checkpoint_block_size": model_cfg.get("block_size"),
        },
    )

    for row in generated_rows:
        print(f"[{row['prompt_index']}] {row['output']}")


if __name__ == "__main__":
    main()
