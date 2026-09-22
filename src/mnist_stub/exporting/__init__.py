from __future__ import annotations

from pathlib import Path

from mnist_stub.assets import save_asset
from mnist_stub.checkpointing import load_checkpoint


def export_checkpoint(checkpoint: Path, destination: Path, best: bool = True) -> Path:
    state = load_checkpoint(checkpoint)
    metadata = state["mnist"]
    weights = (
        metadata["best_model"]
        if best
        else {key.removeprefix("model."): value for key, value in state["state_dict"].items()}
    )
    return save_asset(
        weights,
        destination,
        {
            "run_id": metadata["run_id"],
            "selection": "best" if best else "current",
            "monitor": metadata["config"]["checkpoint"]["monitor"],
            "best_score": metadata["best_score"],
            "data_fingerprint": metadata["data"]["fingerprint"],
        },
    )
