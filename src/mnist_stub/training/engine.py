from __future__ import annotations

import copy
import json
import logging
import os
import platform
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mnist_stub.assets import load_asset, save_asset, validate_weights
from mnist_stub.common import write_json
from mnist_stub.configs.training import TrainConfig
from mnist_stub.models.cnn import ARCHITECTURE, MnistCNN, resolve_device
from mnist_stub.training.checkpoints import load_checkpoint, save_checkpoint
from mnist_stub.training.data import datasets

logger = logging.getLogger(__name__)


def cpu_weights(model: MnistCNN) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def rng_state(generator: torch.Generator, device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "loader": generator.get_state(),
    }


def restore_rng(state: dict, generator: torch.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("CUDA Checkpoint 精确恢复需要原 CUDA 设备配置")
        torch.cuda.set_rng_state(state["cuda"], device)
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, gaussian, cached))
    generator.set_state(state["loader"])


def evaluate_model(model: MnistCNN, dataset, device: torch.device, batch_size: int) -> dict:
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in DataLoader(dataset, batch_size=batch_size):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


def seed_everything(config: TrainConfig) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(config.device)
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return device


def environment() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        revision = None
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "revision": revision,
    }


def train(config: TrainConfig, resume: Path | None = None, init_model: Path | None = None) -> Path:
    if resume and init_model:
        raise ValueError("--resume 与 --init-model 互斥")
    device = seed_everything(config)
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        previous = checkpoint["config"]
        for key, value in config.model_dump().items():
            if key not in {"output_dir", "epochs", "data_dir"} and previous[key] != value:
                raise ValueError(f"恢复配置不兼容: {key}；更改训练策略请使用 --init-model")
        if config.epochs <= checkpoint["epoch"]:
            raise ValueError("epochs 必须大于 Checkpoint 已完成 Epoch 数")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(config.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    write_json(run_dir / "config.json", config.model_dump())
    write_json(run_dir / "environment.json", environment())
    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    status = {
        "run_id": run_id,
        "parent_run": checkpoint["run_id"] if checkpoint else None,
        "initialized_from": str(init_model) if init_model else None,
        "status": "running",
    }
    write_json(run_dir / "status.json", status)
    try:
        data, split = datasets(config)
        write_json(run_dir / "data.json", split)
        if checkpoint and checkpoint["data"] != split:
            raise ValueError("恢复训练的数据内容或划分与 Checkpoint 不一致")
        model = MnistCNN().to(device)
        generator = torch.Generator().manual_seed(config.seed)
        initial_state = None
        if checkpoint:
            initial_state = checkpoint["model"]
        elif init_model:
            initial_state = (
                load_asset(init_model)[0].state_dict()
                if init_model.is_dir()
                else load_checkpoint(init_model)["model"]
            )
        if initial_state is not None:
            validate_weights(initial_state, model)
            model.load_state_dict(initial_state)
        start = checkpoint["epoch"] if checkpoint else 0
        global_step = checkpoint["global_step"] if checkpoint else 0
        best_accuracy = checkpoint["best_accuracy"] if checkpoint else -1.0
        best_state = checkpoint["best_model"] if checkpoint else cpu_weights(model)
        best_snapshot = checkpoint.get("best_snapshot") if checkpoint else None
        model.freeze_backbone(config.freeze_epochs > 0)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
        )
        if checkpoint:
            if config.freeze_epochs > 0 and not checkpoint["frozen"]:
                model.freeze_backbone(False)
                optimizer.add_param_group(
                    {"params": list(model.conv1.parameters()) + list(model.conv2.parameters())}
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            restore_rng(checkpoint["rng"], generator, device)
        loader = DataLoader(
            data["train"],
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
        )
        logger.info("run=%s config=%s", run_id, config.model_dump())
        with (run_dir / "metrics.jsonl").open("a", buffering=1) as metrics:
            for epoch in range(start, config.epochs):
                frozen = epoch < config.freeze_epochs
                if not frozen and not model.conv1.weight.requires_grad:
                    model.freeze_backbone(False)
                    # Preserve classifier Adam state while adding the newly trainable backbone.
                    optimizer.add_param_group(
                        {"params": list(model.conv1.parameters()) + list(model.conv2.parameters())}
                    )
                model.train()
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info("epoch=%s frozen=%s trainable=%s", epoch + 1, frozen, trainable)
                metrics.write(
                    json.dumps(
                        {
                            "event": "epoch_start",
                            "epoch": epoch + 1,
                            "frozen": frozen,
                            "trainable_parameters": trainable,
                        }
                    )
                    + "\n"
                )
                total_loss, correct, count = 0.0, 0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    total_loss += loss.item() * len(y)
                    correct += (logits.argmax(1) == y).sum().item()
                    count += len(y)
                    metrics.write(
                        json.dumps(
                            {
                                "event": "step",
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "loss": loss.item(),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            }
                        )
                        + "\n"
                    )
                validation = evaluate_model(model, data["validation"], device, config.batch_size)
                row = {
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": total_loss / count,
                    "train_accuracy": correct / count,
                    "val_loss": validation["loss"],
                    "val_accuracy": validation["accuracy"],
                }
                metrics.write(json.dumps(row) + "\n")
                logger.info("metrics=%s", row)
                improved = validation["accuracy"] > best_accuracy
                if improved:
                    best_accuracy = validation["accuracy"]
                    best_state = cpu_weights(model)
                state = {
                    "format_version": 1,
                    "architecture": ARCHITECTURE,
                    "run_id": run_id,
                    "config": config.model_dump(),
                    "data": split,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model": cpu_weights(model),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "best_model": best_state,
                    "best_accuracy": best_accuracy,
                    "frozen": frozen,
                    "rng": rng_state(generator, device),
                }
                if improved:
                    best_snapshot = {
                        key: copy.deepcopy(state[key])
                        for key in (
                            "run_id",
                            "epoch",
                            "global_step",
                            "model",
                            "optimizer",
                            "rng",
                            "frozen",
                        )
                    }
                state["best_snapshot"] = best_snapshot
                save_checkpoint(run_dir / "checkpoints" / "last.pt", state)
                if best_snapshot is not None:
                    save_checkpoint(run_dir / "checkpoints" / "best.pt", {**state, **best_snapshot})
        model.load_state_dict(best_state)
        testing = evaluate_model(model, data["test"], device, config.batch_size)
        write_json(run_dir / "evaluation.json", testing)
        export_path = save_asset(
            best_state,
            run_dir / "model",
            {
                "run_id": run_id,
                "test": testing,
                "validation_accuracy": best_accuracy,
                "environment": environment(),
                "data_fingerprint": split["fingerprint"],
            },
        )
        status.update(
            status="completed",
            epoch=config.epochs,
            global_step=global_step,
            model=str(export_path),
            test=testing,
        )
        logger.info("completed model=%s test=%s", export_path, testing)
        return run_dir
    except BaseException as exc:
        status.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
        )
        logger.exception("训练未完成")
        raise
    finally:
        write_json(run_dir / "status.json", status)
        logger.removeHandler(handler)
        handler.close()


def evaluate(asset: Path, config: TrainConfig) -> dict:
    device = seed_everything(config)
    model, _ = load_asset(asset)
    data, _ = datasets(config)
    return evaluate_model(model.to(device), data["test"], device, config.batch_size)


def export_checkpoint(checkpoint: Path, destination: Path, best: bool = True) -> Path:
    state = load_checkpoint(checkpoint)
    return save_asset(
        state["best_model"] if best else state["model"],
        destination,
        {
            "run_id": state["run_id"],
            "epoch": (state.get("best_snapshot") or state)["epoch"] if best else state["epoch"],
            "selection": "best" if best else "current",
            "validation_accuracy": state["best_accuracy"] if best else None,
        },
    )
