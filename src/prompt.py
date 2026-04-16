"""Inference script with fixed-workload benchmark support and sustainability logging."""

import argparse
import datetime as dt
import json
import pickle
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from paths import CKPT_PATH, OUT_DIR, SUMMARY_DIR, make_run_dir, scenario_runs_dir
from run_common import (
    append_row,
    build_tracker,
    collect_model_stats,
    cuda_memory_stats,
    get_git_commit,
    load_emissions_metrics,
    make_amp_ctx_factory,
    parse_bool,
    reset_cuda_peak_memory,
    write_json,
)

DEFAULT_PROMPT = "To be, or not to be"
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_meta(data_dir: str | Path):
    meta_path = Path(data_dir) / "meta.pkl"
    with meta_path.open("rb") as f:
        return pickle.load(f)


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
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--codecarbon-offline", type=parse_bool, default=True)
    parser.add_argument("--country-iso-code", type=str, default="SWE")

    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_workload_file:
        prompt_path = Path(args.prompt_workload_file)
        with prompt_path.open("r", encoding="utf-8") as f:
            prompts = [line.strip("\n") for line in f if line.strip()]
        if not prompts:
            raise ValueError(f"Prompt workload file is empty: {prompt_path}")
        return prompts
    return [args.prompt]


def resolve_checkpoint_path(checkpoint_path_arg: str) -> Path:
    checkpoint_path = Path(checkpoint_path_arg).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. Provide --checkpoint-path to an existing ckpt.pt file."
        )
    return checkpoint_path


def resolve_context_window(context_window: int, checkpoint_block_size: int) -> int:
    if context_window < 0:
        raise ValueError(f"context_window must be >= 0, got {context_window}")
    if checkpoint_block_size <= 0:
        raise ValueError("Checkpoint is missing a valid block_size in model config.")

    effective_context_window = context_window if context_window > 0 else checkpoint_block_size
    if effective_context_window > checkpoint_block_size:
        raise ValueError(
            "context_window cannot exceed checkpoint block_size "
            f"({effective_context_window} > {checkpoint_block_size})."
        )
    return effective_context_window


def build_token_codec(meta: dict):
    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(text: str) -> list[int]:
        return [stoi.get(ch, stoi[" "]) for ch in text]

    def decode(tokens: list[int]) -> str:
        return "".join([itos[t] for t in tokens])

    return encode, decode


def build_effective_config(
    *,
    args: argparse.Namespace,
    run_id: str,
    checkpoint_path: Path,
    effective_context_window: int,
    prompt_count: int,
    model_cfg: dict,
) -> dict:
    return {
        "phase": "inference",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "inference": {
            "checkpoint_path": str(checkpoint_path),
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "context_window": effective_context_window,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "prompt_workload_file": args.prompt_workload_file,
            "prompt_count": prompt_count,
        },
        "checkpoint_model": model_cfg,
    }


def build_run_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    runtime_seconds: float,
    checkpoint_path: Path,
    checkpoint_size_bytes: int | None,
    checkpoint_size_mb: float | None,
    model_cfg: dict,
    model_stats: dict[str, int | float],
    gpu_memory: dict[str, float | int | str | None],
    effective_context_window: int,
    prompts: list[str],
    total_prompt_tokens: int,
    total_generated_tokens: int,
    emissions_metrics: dict[str, float | None],
    run_dir: Path,
    emissions_csv: Path,
    outputs_jsonl: Path,
) -> dict:
    return {
        "phase": "inference",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "status": "completed",
        "runtime_seconds": runtime_seconds,
        "device": args.device,
        "dtype": args.dtype,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_block_size": model_cfg.get("block_size"),
        "model_stats": {
            **model_stats,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "checkpoint_size_mb": checkpoint_size_mb,
        },
        "gpu_memory": gpu_memory,
        "sampling": {
            "max_new_tokens": args.max_new_tokens,
            "context_window": effective_context_window,
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


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = make_run_dir("inference", args.scenario_id, args.run_id)
    scenario_dir = scenario_runs_dir("inference", args.scenario_id)
    emissions_csv = run_dir / "emissions.csv"
    outputs_jsonl = run_dir / "generated_outputs.jsonl"

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location=args.device)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]
    checkpoint_block_size = int(model_cfg.get("block_size", 0))
    effective_context_window = resolve_context_window(args.context_window, checkpoint_block_size)

    meta = load_meta(data_dir)
    encode, decode = build_token_codec(meta)

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model_stats = collect_model_stats(model)

    checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else None
    checkpoint_size_mb = round(checkpoint_size_bytes / (1024**2), 4) if checkpoint_size_bytes is not None else None
    amp_ctx_factory = make_amp_ctx_factory(args.device, args.dtype)

    prompts = load_prompts(args)
    write_json(
        run_dir / "effective_config.json",
        build_effective_config(
            args=args,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            effective_context_window=effective_context_window,
            prompt_count=len(prompts),
            model_cfg=model_cfg,
        ),
    )

    tracker = build_tracker(
        project_name=f"inference_{args.scenario_id}",
        output_dir=run_dir,
        output_file=emissions_csv.name,
        offline=args.codecarbon_offline,
        country_iso_code=args.country_iso_code,
    )

    total_prompt_tokens = 0
    total_generated_tokens = 0
    generated_rows: list[dict[str, object]] = []

    reset_cuda_peak_memory(args.device)

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
                context_window=effective_context_window,
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
    gpu_memory = cuda_memory_stats(args.device)

    with outputs_jsonl.open("w", encoding="utf-8") as f:
        for row in generated_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    emissions_metrics = load_emissions_metrics(emissions_csv)

    run_metadata = build_run_metadata(
        args=args,
        run_id=run_id,
        runtime_seconds=runtime_seconds,
        checkpoint_path=checkpoint_path,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_size_mb=checkpoint_size_mb,
        model_cfg=model_cfg,
        model_stats=model_stats,
        gpu_memory=gpu_memory,
        effective_context_window=effective_context_window,
        prompts=prompts,
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=total_generated_tokens,
        emissions_metrics=emissions_metrics,
        run_dir=run_dir,
        emissions_csv=emissions_csv,
        outputs_jsonl=outputs_jsonl,
    )

    write_json(run_dir / "run_metadata.json", run_metadata)

    write_json(
        scenario_dir / "latest_run.json",
        {
            "phase": "inference",
            "scenario_id": args.scenario_id,
            "run_id": run_id,
            "status": "completed",
            "run_metadata_path": str(run_dir / "run_metadata.json"),
            "checkpoint_path": str(checkpoint_path),
        },
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
            "context_window": effective_context_window,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "prompt_count": len(prompts),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "runtime_seconds": round(runtime_seconds, 4),
            "model_total_params": model_stats["total_params"],
            "model_trainable_params": model_stats["trainable_params"],
            "model_state_dict_mb_estimate": model_stats["state_dict_mb_estimate"],
            "checkpoint_size_mb": checkpoint_size_mb,
            "gpu_peak_allocated_mb": gpu_memory["peak_allocated_mb"],
            "gpu_peak_reserved_mb": gpu_memory["peak_reserved_mb"],
            "energy_kwh": emissions_metrics["energy_kwh"],
            "emissions_kg": emissions_metrics["emissions_kg"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_block_size": model_cfg.get("block_size"),
        },
    )

    for row in generated_rows:
        print(f"[{row['prompt_index']}] {row['output']}")


if __name__ == "__main__":
    main()
