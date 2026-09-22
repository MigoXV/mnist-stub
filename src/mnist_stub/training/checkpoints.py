from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from mnist_stub.models.cnn import ARCHITECTURE


def save_checkpoint(path: Path, state: dict) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("architecture") != ARCHITECTURE:
        raise ValueError("不支持的训练 Checkpoint 格式或架构")
    required = {
        "model",
        "optimizer",
        "epoch",
        "global_step",
        "rng",
        "config",
        "data",
        "best_model",
        "best_accuracy",
        "run_id",
        "frozen",
    }
    if not required <= state.keys():
        raise ValueError(f"Checkpoint 缺少字段: {sorted(required - state.keys())}")
    return state
