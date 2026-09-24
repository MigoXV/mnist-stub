from __future__ import annotations

import json
import logging
import signal
import sys
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.cli import LightningCLI

from mnist_stub.configs.training import (
    CheckpointConfig,
    DataConfig,
    EvaluationConfig,
    LoggerConfig,
    OptimizerConfig,
    TrainConfig,
)
from mnist_stub.tasks.classification import MnistTask
from mnist_stub.tasks.data import MnistDataModule
from mnist_stub.tasks.run import train


def create_data(
    path: str = "data-bin/MigoXV/mnist-4",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    subset: str = "full",
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    validation_size: int = 5000,
    synthetic: bool = False,
    cache_dir: str | None = None,
    seed: int = 42,
) -> MnistDataModule:
    values = locals().copy()
    values.pop("seed")
    return MnistDataModule(DataConfig.model_validate(values), seed)


class MnistCLI(LightningCLI):
    @staticmethod
    def subcommands():
        return {"fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"}}

    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(OptimizerConfig, "optimizer")
        parser.add_class_arguments(CheckpointConfig, "checkpoint")
        parser.add_class_arguments(LoggerConfig, "logging")
        parser.add_class_arguments(EvaluationConfig, "evaluation")
        parser.add_argument("--threads", type=int, default=2)
        parser.add_argument("--init_model", type=Path, default=None)
        parser.link_arguments("optimizer.lr", "model.optimizer.lr")
        parser.link_arguments("optimizer.weight_decay", "model.optimizer.weight_decay")
        parser.link_arguments("seed_everything", "data.seed")

    def before_instantiate_classes(self):
        import os

        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cfg = self.config.fit
        if cfg.threads < 1:
            raise ValueError("threads 必须 >= 1")
        torch.set_num_threads(cfg.threads)
        if cfg.trainer.accelerator not in ("cpu", "gpu", "cuda"):
            raise ValueError("trainer.accelerator 必须显式为 cpu 或 gpu")
        devices = cfg.trainer.devices
        if cfg.trainer.num_nodes != 1 or not (
            devices in (1, "1")
            or (isinstance(devices, list) and len(devices) == 1)
            or (cfg.trainer.accelerator != "cpu" and isinstance(devices, str) and devices.isdigit())
        ):
            raise ValueError("当前 MNIST 训练仅支持单设备")
        if cfg.trainer.precision not in ("32-true", 32):
            raise ValueError("当前 MNIST 训练仅支持 precision=32-true")

    def fit(self, model, datamodule, ckpt_path=None, **kwargs):
        cfg = self.config.fit
        native = cfg.trainer.as_dict()
        device = str(self.trainer.strategy.root_device)
        settings = TrainConfig.model_validate(
            {
                "model": {"freeze_epochs": cfg.model.freeze_epochs, "path": cfg.model.model_path},
                "data": datamodule.config.model_dump(),
                "optimizer": model.optimizer_config.model_dump(),
                "checkpoint": cfg.checkpoint.as_dict(),
                "logger": cfg.logging.as_dict(),
                "evaluation": cfg.evaluation.as_dict(),
                "trainer": {
                    "epochs": native["max_epochs"],
                    "device": device,
                    "seed": cfg.seed_everything,
                    "threads": cfg.threads,
                    "output_dir": native["default_root_dir"],
                },
            }
        )
        # Replace framework defaults with the project's checkpoint and stderr tqdm callbacks.
        self.trainer.callbacks = [
            callback
            for callback in self.trainer.callbacks
            if not isinstance(callback, (ModelCheckpoint, ProgressBar))
        ]
        run = train(
            settings,
            Path(ckpt_path) if ckpt_path else None,
            cfg.init_model,
            {
                "lightning_cli": self._parser("fit").dump(cfg),
                "argv": sys.argv[1:],
                "trainer_config": native,
            },
            trainer=self.trainer,
            task=model,
            data=datamodule,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run),
                    "model": str(run / "model"),
                    "checkpoint": str(run / "checkpoints/last.ckpt"),
                },
                ensure_ascii=False,
            )
        )


def main(args: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    def interrupt(signum, frame):
        raise KeyboardInterrupt("收到 SIGTERM，从最近完整 Epoch 的 Checkpoint 恢复")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        MnistCLI(
            MnistTask,
            create_data,
            args=args,
            auto_configure_optimizers=False,
            seed_everything_default=42,
            load_from_checkpoint_support=False,
            parser_kwargs={"default_env": True, "env_prefix": "MNIST"},
            trainer_defaults={
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "max_epochs": 3,
                "default_root_dir": "outputs",
                "deterministic": True,
                "benchmark": False,
                "num_sanity_val_steps": 0,
                "enable_model_summary": False,
                "logger": False,
                "log_every_n_steps": 1,
                "enable_progress_bar": False,
            },
            save_config_callback=None,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
