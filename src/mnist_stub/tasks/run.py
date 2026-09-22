from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from mnist_stub.assets import load_asset, save_asset
from mnist_stub.checkpointing import RunCheckpoint, TrainingState, load_checkpoint
from mnist_stub.common import write_json
from mnist_stub.configs.training import TrainConfig
from mnist_stub.logging import StderrProgressBar, training_logging
from mnist_stub.models.cnn import resolve_device
from mnist_stub.tasks.classification import MnistTask
from mnist_stub.tasks.data import MnistDataModule

logger = logging.getLogger(__name__)


def environment() -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "revision": revision,
        "packages": {
            name: version(name)
            for name in ("torch", "lightning", "datasets", "transformers", "tqdm")
        },
        "cuda": torch.version.cuda,
    }


def trainer_options(config: TrainConfig) -> dict:
    device = resolve_device(config.trainer.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.trainer.threads)
    seed_everything(config.trainer.seed, workers=True, verbose=False)
    return {
        "accelerator": "gpu" if device.type == "cuda" else "cpu",
        "devices": [device.index or 0] if device.type == "cuda" else 1,
        "precision": "32-true",
        "deterministic": True,
        "benchmark": False,
        "max_epochs": config.trainer.epochs,
        "num_sanity_val_steps": 0,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
    }


def validate_resume(config: TrainConfig, checkpoint: dict) -> None:
    old = TrainConfig.model_validate(checkpoint["mnist"]["config"]).model_dump()
    new = config.model_dump()
    for values in (old, new):
        for key in ("epochs", "output_dir"):
            values["trainer"].pop(key)
        for key in ("path", "cache_dir"):
            values["data"].pop(key)
        values.pop("logger")
        values.pop("evaluation")
    if old != new:
        raise ValueError("恢复配置不兼容；更改训练策略请使用 --init-model")
    if config.trainer.epochs <= checkpoint["epoch"] + 1:
        raise ValueError("trainer.epochs 必须大于已完成 Epoch 数")


def train(
    config: TrainConfig,
    resume: Path | None = None,
    init_model: Path | None = None,
    audit: dict | None = None,
    trainer: Trainer | None = None,
    task: MnistTask | None = None,
    data: MnistDataModule | None = None,
) -> Path:
    if resume and init_model:
        raise ValueError("--resume 与 --init-model 互斥")
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        validate_resume(config, checkpoint)
        previous_native = checkpoint["mnist"].get("trainer_config")
        current_native = (audit or {}).get("trainer_config")
        if previous_native is not None and current_native is not None:
            mutable = {
                "max_epochs",
                "default_root_dir",
                "logger",
                "enable_progress_bar",
                "enable_model_summary",
                "log_every_n_steps",
                "profiler",
            }
            previous_native = {k: v for k, v in previous_native.items() if k not in mutable}
            current_native = {k: v for k, v in current_native.items() if k not in mutable}
            if previous_native != current_native:
                raise ValueError("恢复配置不兼容: Lightning Trainer 参数已改变")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(config.trainer.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True)
    progress = config.logger.progress
    if progress is None:
        progress = sys.stderr.isatty()
    status = {
        "run_id": run_id,
        "status": "running",
        "parent_run": checkpoint["mnist"]["run_id"] if checkpoint else None,
        "initialized_from": str(init_model or config.model.path or "") or None,
    }
    write_json(run_dir / "status.json", status)
    with training_logging(run_dir / "train.log", progress):
        try:
            options = trainer_options(config)
            write_json(run_dir / "config.json", config.model_dump())
            if audit and "lightning_cli" in audit:
                (run_dir / "config.yaml").write_text(audit["lightning_cli"], encoding="utf-8")
            write_json(run_dir / "environment.json", environment())
            logger.info(
                "run=%s config=%s invocation=%s trainer=%s",
                run_id,
                config.model_dump(),
                audit or {},
                options,
            )
            data = data or MnistDataModule(config.data, config.trainer.seed)
            data.setup()
            write_json(run_dir / "data.json", data.metadata)
            task = task or MnistTask(config.optimizer, config.model.freeze_epochs)
            source = (
                None
                if resume
                else (init_model or (Path(config.model.path) if config.model.path else None))
            )
            if source:
                if source.is_dir():
                    task.model.load_state_dict(load_asset(source)[0].state_dict())
                else:
                    state = load_checkpoint(source)
                    task.load_state_dict(state["state_dict"], strict=True)
            state_callback = TrainingState(
                config.model_dump(),
                run_id,
                config.checkpoint.monitor,
                trainer_config=(audit or {}).get("trainer_config"),
            )
            saving = RunCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="epoch={epoch}-step={step}-"
                + config.checkpoint.monitor
                + "={"
                + config.checkpoint.monitor
                + ":.4f}",
                auto_insert_metric_name=False,
                monitor=config.checkpoint.monitor,
                mode="max" if config.checkpoint.monitor == "val_accuracy" else "min",
                save_top_k=config.checkpoint.save_top_k,
                save_last=True,
                save_on_train_epoch_end=True,
            )
            callbacks = [state_callback, saving]
            if progress:
                callbacks.append(StderrProgressBar(refresh_rate=config.logger.refresh_rate))
            experiment_logger = (
                CSVLogger(str(run_dir), name="metrics", version="")
                if (config.logger.enabled)
                else False
            )
            if trainer is None:
                trainer = Trainer(
                    **options,
                    callbacks=callbacks,
                    logger=experiment_logger,
                    enable_progress_bar=progress,
                    default_root_dir=run_dir,
                )
            else:
                # LightningCLI owns Trainer/model/data construction; attach run artifacts here.
                trainer.callbacks.extend(callbacks)
                trainer.logger = experiment_logger
            if experiment_logger:
                experiment_logger.log_hyperparams(config.model_dump())
            trainer.fit(task, datamodule=data, ckpt_path=str(resume) if resume else None)
            if trainer.interrupted:
                raise KeyboardInterrupt("训练中断；可从最近完整 Epoch 恢复")
            task.model.load_state_dict(state_callback.best_model)
            testing = None
            if config.evaluation.test_after_fit:
                metrics = trainer.test(task, datamodule=data, verbose=False)[0]
                testing = {
                    "loss": metrics["test_loss"],
                    "accuracy": metrics["test_accuracy"],
                    "samples": len(data.splits["test"]),
                }
                write_json(run_dir / "evaluation.json", testing)
            export_path = save_asset(
                state_callback.best_model,
                run_dir / "model",
                {
                    "run_id": run_id,
                    "test": testing,
                    "monitor": config.checkpoint.monitor,
                    "best_score": state_callback.best_score,
                    "data_fingerprint": data.metadata["fingerprint"],
                },
            )
            status.update(
                status="completed",
                epoch=config.trainer.epochs,
                global_step=trainer.global_step,
                model=str(export_path),
                test=testing,
            )
            logger.info("completed model=%s test=%s", export_path, testing)
            return run_dir
        except BaseException as exc:
            status.update(
                status="interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "failed",
                error=str(exc),
            )
            logger.exception("训练未完成")
            raise
        finally:
            write_json(run_dir / "status.json", status)


def evaluate(asset: Path, config: TrainConfig) -> dict:
    options = trainer_options(config)
    data = MnistDataModule(config.data, config.trainer.seed)
    task = MnistTask(config.optimizer)
    task.model.load_state_dict(load_asset(asset)[0].state_dict())
    progress = config.logger.progress
    if progress is None:
        progress = sys.stderr.isatty()
    with training_logging(os.devnull, progress):
        trainer = Trainer(
            **options,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=progress,
            callbacks=[StderrProgressBar()] if progress else [],
        )
        result = trainer.test(task, datamodule=data, verbose=False)[0]
    return {
        "loss": result["test_loss"],
        "accuracy": result["test_accuracy"],
        "samples": len(data.splits["test"]),
    }
