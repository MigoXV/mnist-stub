from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint


def load_checkpoint(path: Path) -> dict:
    if path.suffix != ".ckpt":
        raise ValueError("仅支持新的 Lightning .ckpt，不支持旧 .pt checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "pytorch-lightning_version" not in state or "mnist" not in state:
        raise ValueError("不是 MNIST Lightning checkpoint")
    return state


def random_state(device: torch.device) -> dict:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_random_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, gaussian, cached))
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


class TrainingState(Callback):
    def __init__(
        self, config: dict, run_id: str, monitor: str, trainer_config: dict | None = None
    ) -> None:
        self.config, self.run_id, self.monitor = config, run_id, monitor
        self.best_score: float | None = None
        self.best_model: dict = {}
        self.pending_rng: dict | None = None
        self.trainer_config = trainer_config

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        score = float(trainer.callback_metrics[self.monitor])
        improved = self.best_score is None or (
            score > self.best_score if self.monitor == "val_accuracy" else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.best_model = {
                k: v.detach().cpu().clone() for k, v in pl_module.model.state_dict().items()
            }

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        checkpoint["mnist"] = {
            "config": self.config,
            "run_id": self.run_id,
            "best_score": self.best_score,
            "best_model": self.best_model,
            "rng": random_state(pl_module.device),
            "data": trainer.datamodule.metadata,
            "trainer_config": self.trainer_config,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        state = checkpoint["mnist"]
        self.best_score, self.best_model = state["best_score"], state["best_model"]
        self.pending_rng = state["rng"]

    def on_train_start(self, trainer, pl_module) -> None:
        if self.pending_rng is not None:
            restore_random_state(self.pending_rng, pl_module.device)
            self.pending_rng = None


class RunCheckpoint(ModelCheckpoint):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        # Lightning 2.6 only refreshes last when top-k improved. Resume needs every epoch.
        if self._last_global_step_saved != trainer.global_step:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def load_state_dict(self, state_dict: dict) -> None:
        # A resumed run owns its files; never remove checkpoints in the parent run.
        state = copy.deepcopy(state_dict)
        state["dirpath"] = self.dirpath
        state["best_k_models"] = {}
        state["best_model_path"] = ""
        state["last_model_path"] = ""
        state["kth_best_model_path"] = ""
        super().load_state_dict(state)
