"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from paths import DATA_DIR, OUT_DIR

# -----------------------------------------------------------------------------
# Experiment configuration

# I/O
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LOG_INTERVAL = 50
SAVE_CHECKPOINT = True

# Model (main tunables)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
BIAS = True

# Training (main parameters you can also experiment with)
SEED = 1
DEVICE = "cpu"          # Try GPU too if available:
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float32"
BATCH_SIZE = 32         # Number of sequences processed in parallel.
BLOCK_SIZE = 256        # Maximum context length for predictions (e.g. 128 or 256).
# Larger block sizes need more memory/compute, but can improve performance.
MAX_ITERS = 2000        # Total number of training iterations.
# More iterations can improve quality, but cost more time and energy.
LEARNING_RATE = 3e-4    # the standard starting learning rate, often good enough for a first try
WEIGHT_DECAY = 0.1      # L2 Regularization
GRAD_CLIP = 1.0         # To prevent exploding gradients

# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

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

def save_checkpoint(out_dir: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, out_path / "ckpt.pt")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    meta = load_meta(DATA_DIR)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )

    # create the model and move it to the device
    model = GPT(cfg).to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # (optional) uncomment this for printing model size once
    # print(f"Device: {DEVICE}")
    # print(f"Model parameters: {model.get_num_params():,}")
    # print(f"Training for {MAX_ITERS} iterations | batch={BATCH_SIZE} | block={BLOCK_SIZE}")

    t0 = time.time()
    for it in range(MAX_ITERS + 1):
        # periodic evaluation
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS)
            dt = time.time() - t0
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s")

            if SAVE_CHECKPOINT and it > 0:
                config_dump = {
                    "data_dir": DATA_DIR,
                    "train": {
                        "batch_size": BATCH_SIZE,
                        "block_size": BLOCK_SIZE,
                        "max_iters": MAX_ITERS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "grad_clip": GRAD_CLIP,
                        "dtype": DTYPE,
                        "device": DEVICE,
                    },
                    "model": asdict(cfg),
                }
                save_checkpoint(OUT_DIR, model, optimizer, it, config_dump)

        # training step
        x, y = get_batch("train", DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        if it % LOG_INTERVAL == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f}")

    print("Training completed.")

    # Save final checkpoint
    if SAVE_CHECKPOINT:
        config_dump = {
            "data_dir": DATA_DIR,
            "train": {
                "batch_size": BATCH_SIZE,
                "block_size": BLOCK_SIZE,
                "max_iters": MAX_ITERS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "dtype": DTYPE,
                "device": DEVICE,
            },
            "model": asdict(cfg),
        }
        save_checkpoint(OUT_DIR, model, optimizer, MAX_ITERS, config_dump)

if __name__ == "__main__":
    main()
