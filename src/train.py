"""Training script with scenario-aware sustainability measurement."""

import argparse
import csv
import datetime as dt
import json
import pickle
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from model import GPT, GPTConfig
from paths import DATA_DIR, OUT_DIR, SUMMARY_DIR, make_run_dir, scenario_runs_dir

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
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float32"
DEFAULT_BATCH_SIZE = 32
DEFAULT_BLOCK_SIZE = 256
DEFAULT_MAX_ITERS = 2000
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0

# Early stopping (validation-based)
DEFAULT_EARLY_STOPPING = True
DEFAULT_EARLY_STOPPING_PATIENCE = 4
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_EARLY_STOPPING_MIN_EVALS = 3

# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_bool(value: str) -> bool:
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


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

    parser.add_argument("--early-stopping", type=parse_bool, default=DEFAULT_EARLY_STOPPING)
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--early-stopping-min-delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA)
    parser.add_argument("--early-stopping-min-evals", type=int, default=DEFAULT_EARLY_STOPPING_MIN_EVALS)

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


def main():  # noqa: C901
    args = parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = make_run_dir("train", args.scenario_id, args.run_id)
    scenario_dir = scenario_runs_dir("train", args.scenario_id)
    metrics_csv = run_dir / "train_metrics.csv"
    checkpoint_path = run_dir / "ckpt.pt"
    emissions_csv = run_dir / "emissions.csv"

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

    # create the model and move it to the device
    model = GPT(cfg).to(args.device)

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

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    effective_config = {
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
        "early_stopping": {
            "enabled": args.early_stopping,
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "min_evals": args.early_stopping_min_evals,
        },
        "model": asdict(cfg),
    }
    with (run_dir / "effective_config.json").open("w", encoding="utf-8") as f:
        json.dump(effective_config, f, indent=2)

    tracker_kwargs = {
        "project_name": f"train_{args.scenario_id}",
        "output_dir": str(run_dir),
        "output_file": emissions_csv.name,
        "save_to_file": True,
        "log_level": "error",
    }
    if args.codecarbon_offline:
        tracker = OfflineEmissionsTracker(country_iso_code=args.country_iso_code, **tracker_kwargs)
    else:
        tracker = EmissionsTracker(**tracker_kwargs)

    training_tokens_processed = 0
    eval_tokens_processed = 0
    best_val_loss = float("inf")
    best_iter = 0
    last_eval_losses: dict[str, float] = {"train": float("nan"), "val": float("nan")}
    stop_reason = "max_iters_reached"
    eval_count = 0
    no_improve_count = 0

    tracker.start()
    t0 = time.time()
    last_iter = 0
    for it in range(args.max_iters + 1):
        last_iter = it
        # periodic evaluation
        if it % args.eval_interval == 0:
            with amp_ctx_factory():
                losses = estimate_loss(model, data_dir, args.block_size, args.batch_size, args.device, args.eval_iters)

            eval_tokens_processed += 2 * args.eval_iters * args.batch_size * args.block_size
            elapsed_s = time.time() - t0
            print(
                f"iter {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | elapsed {elapsed_s:.1f}s"
            )
            eval_count += 1
            last_eval_losses = losses

            improved = losses["val"] < (best_val_loss - args.early_stopping_min_delta)
            if improved:
                best_val_loss = losses["val"]
                best_iter = it
                no_improve_count = 0
            else:
                no_improve_count += 1

            append_row(
                metrics_csv,
                {
                    "iter": it,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "elapsed_seconds": round(elapsed_s, 4),
                    "best_val_loss": best_val_loss,
                    "best_iter": best_iter,
                    "no_improve_evals": no_improve_count,
                },
            )

            if args.save_checkpoint and it > 0:
                config_dump = {
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

                save_checkpoint(checkpoint_path, model, optimizer, it, config_dump)

            if (
                args.early_stopping
                and eval_count >= args.early_stopping_min_evals
                and no_improve_count >= args.early_stopping_patience
                and it > 0
            ):
                stop_reason = (
                    f"early_stopping(no_improve_evals={no_improve_count}, "
                    f"patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})"
                )
                print(f"Stopping early at iter {it}: {stop_reason}")
                break

        # training step
        x, y = get_batch("train", data_dir, args.block_size, args.batch_size, args.device)
        with amp_ctx_factory():
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        training_tokens_processed += args.batch_size * args.block_size

        if it % args.log_interval == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f}")

    runtime_seconds = time.time() - t0
    tracker.stop()
    print("Training completed.")

    # Save final checkpoint
    if args.save_checkpoint:
        config_dump = {
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
        save_checkpoint(checkpoint_path, model, optimizer, last_iter, config_dump)

    emissions_metrics = load_emissions_metrics(emissions_csv)

    run_metadata = {
        "phase": "train",
        "scenario_id": args.scenario_id,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "status": "completed",
        "stop_reason": stop_reason,
        "runtime_seconds": runtime_seconds,
        "device": args.device,
        "dtype": args.dtype,
        "model": asdict(cfg),
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "betas": [0.9, 0.95],
        },
        "early_stopping": {
            "enabled": args.early_stopping,
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "min_evals": args.early_stopping_min_evals,
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
        "artifacts": {
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "metrics_csv": str(metrics_csv),
            "emissions_csv": str(emissions_csv),
            "effective_config_json": str(run_dir / "effective_config.json"),
        },
    }

    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    with (scenario_dir / "latest_run.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "train",
                "scenario_id": args.scenario_id,
                "run_id": run_id,
                "status": "completed",
                "run_metadata_path": str(run_dir / "run_metadata.json"),
                "checkpoint_path": str(checkpoint_path),
            },
            f,
            indent=2,
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
            "early_stopping": args.early_stopping,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "runtime_seconds": round(runtime_seconds, 4),
            "best_val_loss": best_val_loss,
            "final_val_loss": last_eval_losses["val"],
            "energy_kwh": emissions_metrics["energy_kwh"],
            "emissions_kg": emissions_metrics["emissions_kg"],
            "total_tokens_processed": training_tokens_processed + eval_tokens_processed,
            "checkpoint_path": str(checkpoint_path),
        },
    )


if __name__ == "__main__":
    main()
