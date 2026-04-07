"""
Prepare the Tiny Shakespeare dataset for character-level language modeling.

It will create (in the same folder):
  data/input.txt
  data/train.bin        ->
  data/val.bin          -> binary files containing the tokenized data for training and validation
  data/meta.pkl         -> contains char to index mapping for use in prompt.py and get vocab size for train.py
"""

import importlib
import pickle
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = importlib.import_module("paths").DATA_DIR

INPUT_FILE = DATA_DIR / "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

TRAIN_FRACTION = 0.9  # 90% train / 10% val


def download_if_missing():
    if INPUT_FILE.exists():
        return
    print("input.txt not found. Downloading Tiny Shakespeare...")
    with urlopen(DATA_URL) as r:
        text = r.read().decode("utf-8")
    with INPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"Downloaded to {INPUT_FILE}")


def main():
    download_if_missing()

    with INPUT_FILE.open(encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # unique characters
    chars = sorted(set(data))
    vocab_size = len(chars)
    print("vocab size:", vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = dict(enumerate(chars))

    def encode(s: str):
        return [stoi[c] for c in s]

    # split
    n = len(data)
    n_train = int(n * TRAIN_FRACTION)
    train_data = data[:n_train]
    val_data = data[n_train:]

    # encode
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has   {len(val_ids):,} tokens")

    # write binaries
    train_path = DATA_DIR / "train.bin"
    val_path = DATA_DIR / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    # write meta
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "train_fraction": TRAIN_FRACTION,
        "dataset": "tiny_shakespeare_char",
        "source_url": DATA_URL,
    }
    meta_path = DATA_DIR / "meta.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(meta, f)

    print("Done.")
    print(f"Wrote: {train_path}, {val_path}, {meta_path}")


if __name__ == "__main__":
    main()
