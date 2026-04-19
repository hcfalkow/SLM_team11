"""Training script with scenario-aware sustainability measurement."""

import argparse
import datetime as dt
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from paths import DATA_DIR, OUT_DIR, SUMMARY_DIR, make_run_dir, scenario_runs_dir
from run_common import (
    append_row,
    collect_model_stats,
    cuda_memory_stats,
    get_git_commit,
    load_emissions_metrics_total,
    make_amp_ctx_factory,
    parse_bool,
    reset_cuda_peak_memory,
    run_stage_with_tracker,
    write_json,
)

# -----------------------------------------------------------------------------
# Experiment configuration

# I/O
DEFAULT_EVAL_INTERVAL = 200
DEFAULT_EVAL_ITERS = 50
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SAVE_CHECKPOINT = True

# Model (main tunables)
DEFAULT_N_LAYER = 4
DEFAULT_N_HEAD = 4
DEFAULT_N_EMBD = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_BIAS = True

# Training (main parameters you can also experiment with)
DEFAULT_SEED = 1
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float32"
DEFAULT_BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 256
DEFAULT_MAX_ITERS = 2000
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0

# Validation threshold stopping
DEFAULT_VAL_THRESHOLD_STOPPING = True
DEFAULT_VAL_LOSS_THRESHOLD = 2.3
DEFAULT_VAL_THRESHOLD_MIN_EVALS = 1

TRAIN_STAGE_DEFINITIONS = {
    "tr1_setup_init": "Setup and initialization",
    "tr2_core_training_compute": "Core training compute",
    "tr3_periodic_evaluation_control": "Periodic evaluation and control",
    "tr4_finalization_artifact_write": "Finalization and artifact write",
}

# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-style SLM with reproducible scenario logging.")

    parser.add_argument("--scenario-id", type=str, default="train_manual")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))

    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=DEFAULT_EVAL_ITERS)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--save-checkpoint", type=parse_bool, default=DEFAULT_SAVE_CHECKPOINT)

    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--bias", type=parse_bool, default=DEFAULT_BIAS)

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default=DEFAULT_DTYPE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)

    parser.add_argument("--val-threshold-stopping", type=parse_bool, default=DEFAULT_VAL_THRESHOLD_STOPPING)
    parser.add_argument("--val-loss-threshold", type=float, default=DEFAULT_VAL_LOSS_THRESHOLD)
    parser.add_argument("--val-threshold-min-evals", type=int, default=DEFAULT_VAL_THRESHOLD_MIN_EVALS)

    parser.add_argument("--codecarbon-offline", type=parse_bool, default=True)
    parser.add_argument("--country-iso-code", type=str, default="SWE")

    return parser.parse_args()


def load_meta(data_dir: str | Path):
    meta_path = Path(data_dir) / "meta.pkl"
    if not meta_path.exists():
        return None
    with meta_path.open("rb") as f:
        return pickle.load(f)


def get_batch(split: str, data_dir: str | Path, block_size: int, batch_size: int, device: str):
    # simple, robust memmap loader
    bin_path = Path(data_dir) / f"{split}.bin"
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, data_dir: str | Path, block_size: int, batch_size: int, device: str, eval_iters: int):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses[k] = loss
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def save_checkpoint(out_path: Path, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict):
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, out_path)


def build_checkpoint_config(args: argparse.Namespace, data_dir: Path, cfg: GPTConfig, run_id: str) -> dict:
    return {
        "data_dir": str(data_dir),
        "train": {
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "max_iters": args.max_iters,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "dtype": args.dtype,
            "device": args.device,
        },
        "scenario": {"id": args.scenario_id, "run_id": run_id},
        "model": asdict(cfg),
    }


def build_effective_config(args: argparse.Namespace, run_id: str, data_dir: Path, cfg: GPTConfig) -> dict:
    return {
        "phase": "train",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "data": {
            "data_dir": str(data_dir),
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
        },
        "train": {
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "max_iters": args.max_iters,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "save_checkpoint": args.save_checkpoint,
        },
        "validation_threshold_stopping": {
            "enabled": args.val_threshold_stopping,
            "val_loss_threshold": args.val_loss_threshold,
            "min_evals": args.val_threshold_min_evals,
        },
        "model": asdict(cfg),
    }


def run_training_loop(
    *,
    args: argparse.Namespace,
    run_id: str,
    data_dir: Path,
    cfg: GPTConfig,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    amp_ctx_factory,
    metrics_csv: Path,
    checkpoint_path: Path,
    run_stage_train_compute,
    run_stage_eval_control,
) -> dict[str, object]:
    training_tokens_processed = 0
    eval_tokens_processed = 0
    best_val_loss = float("inf")
    best_iter = 0
    last_eval_losses: dict[str, float] = {"train": float("nan"), "val": float("nan")}
    stop_reason = "max_iters_reached"
    eval_count = 0

    t0 = time.time()
    last_iter = 0
    it = 0

    while it <= args.max_iters:
        if it % args.eval_interval == 0:

            def eval_stage_step() -> tuple[dict[str, float], float, int, bool, str | None]:
                with amp_ctx_factory():
                    losses = estimate_loss(
                        model,
                        data_dir,
                        args.block_size,
                        args.batch_size,
                        args.device,
                        args.eval_iters,
                    )

                elapsed_s = time.time() - t0
                print(
                    f"iter {it:5d} | train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f} | elapsed {elapsed_s:.1f}s"
                )

                threshold_reached = losses["val"] <= args.val_loss_threshold
                next_best_val_loss = best_val_loss
                next_best_iter = best_iter
                if losses["val"] < next_best_val_loss:
                    next_best_val_loss = losses["val"]
                    next_best_iter = it

                append_row(
                    metrics_csv,
                    {
                        "iter": it,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "elapsed_seconds": round(elapsed_s, 4),
                        "best_val_loss": next_best_val_loss,
                        "best_iter": next_best_iter,
                        "val_loss_threshold": args.val_loss_threshold,
                        "threshold_reached": threshold_reached,
                    },
                )

                if args.save_checkpoint and it > 0:
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        it,
                        build_checkpoint_config(args=args, data_dir=data_dir, cfg=cfg, run_id=run_id),
                    )

                stage_stop_reason = None
                should_stop = (
                    args.val_threshold_stopping
                    and eval_count + 1 >= args.val_threshold_min_evals
                    and threshold_reached
                    and it > 0
                )
                if should_stop:
                    stage_stop_reason = (
                        f"validation_threshold_reached(val_loss={losses['val']:.6f}, "
                        f"threshold={args.val_loss_threshold}, min_evals={args.val_threshold_min_evals})"
                    )

                return losses, next_best_val_loss, next_best_iter, should_stop, stage_stop_reason

            losses, best_val_loss, best_iter, should_stop, stage_stop_reason = run_stage_eval_control(eval_stage_step)

            eval_tokens_processed += 2 * args.eval_iters * args.batch_size * args.block_size
            eval_count += 1
            last_eval_losses = losses

            if should_stop:
                stop_reason = str(stage_stop_reason)
                last_iter = it
                print(f"Stopping early at iter {it}: {stop_reason}")
                break

        next_eval_it = min(((it // args.eval_interval) + 1) * args.eval_interval, args.max_iters + 1)

        def train_stage_block() -> int:
            nonlocal training_tokens_processed
            for train_it in range(it, next_eval_it):
                x, y = get_batch("train", data_dir, args.block_size, args.batch_size, args.device)
                with amp_ctx_factory():
                    _, loss = model(x, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                training_tokens_processed += args.batch_size * args.block_size

                if train_it % args.log_interval == 0:
                    print(f"iter {train_it:5d} | loss {loss.item():.4f}")

            return next_eval_it - 1

        last_iter = run_stage_train_compute(train_stage_block)
        it = next_eval_it

    return {
        "runtime_seconds": time.time() - t0,
        "last_iter": last_iter,
        "best_val_loss": best_val_loss,
        "best_iter": best_iter,
        "last_eval_losses": last_eval_losses,
        "stop_reason": stop_reason,
        "eval_count": eval_count,
        "training_tokens_processed": training_tokens_processed,
        "eval_tokens_processed": eval_tokens_processed,
    }


def main():
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = make_run_dir("train", args.scenario_id, args.run_id)
    scenario_dir = scenario_runs_dir("train", args.scenario_id)
    metrics_csv = run_dir / "train_metrics.csv"
    checkpoint_path = run_dir / "ckpt.pt"

    stage_order = list(TRAIN_STAGE_DEFINITIONS.keys())
    stage_emissions_files = {stage_id: run_dir / f"emissions_{stage_id}.csv" for stage_id in stage_order}
    stage_runtime_seconds = dict.fromkeys(stage_order, 0.0)

    def run_stage(stage_id: str, fn):
        result, duration_s = run_stage_with_tracker(
            project_name=f"train_{args.scenario_id}_{stage_id}",
            output_dir=run_dir,
            output_file=stage_emissions_files[stage_id].name,
            offline=args.codecarbon_offline,
            country_iso_code=args.country_iso_code,
            fn=fn,
        )
        stage_runtime_seconds[stage_id] += duration_s
        return result

    def setup_stage():
        data_dir = Path(args.data_dir)
        set_seed(args.seed)

        meta = load_meta(data_dir)
        vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

        cfg = GPTConfig(
            block_size=args.block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bias=args.bias,
        )

        model = GPT(cfg).to(args.device)
        model_stats = collect_model_stats(model)
        amp_ctx_factory = make_amp_ctx_factory(args.device, args.dtype)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        write_json(run_dir / "effective_config.json", build_effective_config(args, run_id, data_dir, cfg))

        return {
            "data_dir": data_dir,
            "cfg": cfg,
            "model": model,
            "model_stats": model_stats,
            "amp_ctx_factory": amp_ctx_factory,
            "optimizer": optimizer,
        }

    setup_state = run_stage("tr1_setup_init", setup_stage)
    data_dir = setup_state["data_dir"]
    cfg = setup_state["cfg"]
    model = setup_state["model"]
    model_stats = setup_state["model_stats"]
    amp_ctx_factory = setup_state["amp_ctx_factory"]
    optimizer = setup_state["optimizer"]

    reset_cuda_peak_memory(args.device)

    training_result = run_training_loop(
        args=args,
        run_id=run_id,
        data_dir=data_dir,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        amp_ctx_factory=amp_ctx_factory,
        metrics_csv=metrics_csv,
        checkpoint_path=checkpoint_path,
        run_stage_train_compute=lambda fn: run_stage("tr2_core_training_compute", fn),
        run_stage_eval_control=lambda fn: run_stage("tr3_periodic_evaluation_control", fn),
    )

    def finalization_stage() -> None:
        if args.save_checkpoint:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                int(training_result["last_iter"]),
                build_checkpoint_config(args=args, data_dir=data_dir, cfg=cfg, run_id=run_id),
            )

    run_stage("tr4_finalization_artifact_write", finalization_stage)
    print("Training completed.")

    training_loop_runtime_seconds = float(training_result["runtime_seconds"])
    runtime_seconds = float(sum(stage_runtime_seconds.values()))
    last_iter = int(training_result["last_iter"])
    best_val_loss = float(training_result["best_val_loss"])
    best_iter = int(training_result["best_iter"])
    last_eval_losses = training_result["last_eval_losses"]
    stop_reason = str(training_result["stop_reason"])
    eval_count = int(training_result["eval_count"])
    training_tokens_processed = int(training_result["training_tokens_processed"])
    eval_tokens_processed = int(training_result["eval_tokens_processed"])

    checkpoint_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else None
    checkpoint_size_mb = round(checkpoint_size_bytes / (1024**2), 4) if checkpoint_size_bytes is not None else None
    gpu_memory = cuda_memory_stats(args.device)

    stage_metrics = {}
    for stage_id in stage_order:
        stage_totals = load_emissions_metrics_total(stage_emissions_files[stage_id])
        stage_metrics[stage_id] = {
            "label": TRAIN_STAGE_DEFINITIONS[stage_id],
            "runtime_seconds": round(stage_runtime_seconds[stage_id], 4),
            "energy_kwh": stage_totals["energy_kwh"],
            "emissions_kg": stage_totals["emissions_kg"],
            "emissions_csv": str(stage_emissions_files[stage_id]),
        }

    total_energy_values = [
        stage_metrics[s]["energy_kwh"] for s in stage_order if stage_metrics[s]["energy_kwh"] is not None
    ]
    total_emissions_values = [
        stage_metrics[s]["emissions_kg"] for s in stage_order if stage_metrics[s]["emissions_kg"] is not None
    ]
    emissions_metrics = {
        "energy_kwh": sum(total_energy_values) if total_energy_values else None,
        "emissions_kg": sum(total_emissions_values) if total_emissions_values else None,
    }

    run_metadata = {
        "phase": "train",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "status": "completed",
        "stop_reason": stop_reason,
        "runtime_seconds": runtime_seconds,
        "training_loop_runtime_seconds": training_loop_runtime_seconds,
        "device": args.device,
        "dtype": args.dtype,
        "model": asdict(cfg),
        "model_stats": {
            **model_stats,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "checkpoint_size_mb": checkpoint_size_mb,
        },
        "gpu_memory": gpu_memory,
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "betas": [0.9, 0.95],
        },
        "validation_threshold_stopping": {
            "enabled": args.val_threshold_stopping,
            "val_loss_threshold": args.val_loss_threshold,
            "min_evals": args.val_threshold_min_evals,
            "best_val_loss": best_val_loss,
            "best_iter": best_iter,
            "eval_count": eval_count,
        },
        "metrics": {
            "final_iter": last_iter,
            "final_train_loss": last_eval_losses["train"],
            "final_val_loss": last_eval_losses["val"],
            "best_val_loss": best_val_loss,
        },
        "workload": {
            "training_tokens_processed": training_tokens_processed,
            "evaluation_tokens_processed": eval_tokens_processed,
            "total_tokens_processed": training_tokens_processed + eval_tokens_processed,
        },
        "codecarbon": {
            "offline": args.codecarbon_offline,
            "country_iso_code": args.country_iso_code,
            "energy_kwh": emissions_metrics["energy_kwh"],
            "emissions_kg": emissions_metrics["emissions_kg"],
        },
        "lifecycle_stages": {
            "order": stage_order,
            "metrics": stage_metrics,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "metrics_csv": str(metrics_csv),
            "stage_emissions_csv": {stage_id: str(path) for stage_id, path in stage_emissions_files.items()},
            "effective_config_json": str(run_dir / "effective_config.json"),
        },
    }

    write_json(run_dir / "run_metadata.json", run_metadata)

    write_json(
        scenario_dir / "latest_run.json",
        {
            "phase": "train",
            "scenario_id": args.scenario_id,
            "run_id": run_id,
            "status": "completed",
            "run_metadata_path": str(run_dir / "run_metadata.json"),
            "checkpoint_path": str(checkpoint_path),
        },
    )

    append_row(
        SUMMARY_DIR / "train_summary.csv",
        {
            "scenario_id": args.scenario_id,
            "run_id": run_id,
            "timestamp_utc": run_metadata["timestamp_utc"],
            "git_commit": run_metadata["git_commit"],
            "status": run_metadata["status"],
            "stop_reason": stop_reason,
            "device": args.device,
            "dtype": args.dtype,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_threshold_stopping": args.val_threshold_stopping,
            "val_loss_threshold": args.val_loss_threshold,
            "val_threshold_min_evals": args.val_threshold_min_evals,
            "runtime_seconds": round(runtime_seconds, 4),
            "model_total_params": model_stats["total_params"],
            "model_trainable_params": model_stats["trainable_params"],
            "model_state_dict_mb_estimate": model_stats["state_dict_mb_estimate"],
            "checkpoint_size_mb": checkpoint_size_mb,
            "gpu_peak_allocated_mb": gpu_memory["peak_allocated_mb"],
            "gpu_peak_reserved_mb": gpu_memory["peak_reserved_mb"],
            "best_val_loss": best_val_loss,
            "final_val_loss": last_eval_losses["val"],
            "energy_kwh": emissions_metrics["energy_kwh"],
            "emissions_kg": emissions_metrics["emissions_kg"],
            "total_tokens_processed": training_tokens_processed + eval_tokens_processed,
            "checkpoint_path": str(checkpoint_path),
        },
    )

    append_row(
        SUMMARY_DIR / "train_stage_summary.csv",
        {
            "scenario_id": args.scenario_id,
            "run_id": run_id,
            "timestamp_utc": run_metadata["timestamp_utc"],
            "runtime_seconds_total": round(runtime_seconds, 4),
            "runtime_seconds_tr1": stage_metrics["tr1_setup_init"]["runtime_seconds"],
            "runtime_seconds_tr2": stage_metrics["tr2_core_training_compute"]["runtime_seconds"],
            "runtime_seconds_tr3": stage_metrics["tr3_periodic_evaluation_control"]["runtime_seconds"],
            "runtime_seconds_tr4": stage_metrics["tr4_finalization_artifact_write"]["runtime_seconds"],
            "energy_kwh_total": emissions_metrics["energy_kwh"],
            "energy_kwh_tr1": stage_metrics["tr1_setup_init"]["energy_kwh"],
            "energy_kwh_tr2": stage_metrics["tr2_core_training_compute"]["energy_kwh"],
            "energy_kwh_tr3": stage_metrics["tr3_periodic_evaluation_control"]["energy_kwh"],
            "energy_kwh_tr4": stage_metrics["tr4_finalization_artifact_write"]["energy_kwh"],
            "emissions_kg_total": emissions_metrics["emissions_kg"],
            "emissions_kg_tr1": stage_metrics["tr1_setup_init"]["emissions_kg"],
            "emissions_kg_tr2": stage_metrics["tr2_core_training_compute"]["emissions_kg"],
            "emissions_kg_tr3": stage_metrics["tr3_periodic_evaluation_control"]["emissions_kg"],
            "emissions_kg_tr4": stage_metrics["tr4_finalization_artifact_write"]["emissions_kg"],
        },
    )


if __name__ == "__main__":
    main()
