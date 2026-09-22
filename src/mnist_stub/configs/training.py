from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ModelConfig(Schema):
    path: str | None = None
    freeze_epochs: int = Field(default=0, ge=0)


class DataConfig(Schema):
    path: str = "data-bin/MigoXV/mnist-4"
    batch_size: int = Field(default=64, ge=1)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    subset: Literal["full", "a", "b"] = "full"
    train_limit: int | None = Field(default=None, ge=1)
    validation_limit: int | None = Field(default=None, ge=1)
    test_limit: int | None = Field(default=None, ge=1)
    validation_size: int = Field(default=5000, ge=10)
    synthetic: bool = False
    cache_dir: str | None = None


class OptimizerConfig(Schema):
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)


class EvaluationConfig(Schema):
    test_after_fit: bool = True


class CheckpointConfig(Schema):
    monitor: Literal["val_accuracy", "val_loss"] = "val_accuracy"
    save_top_k: int = Field(default=1, ge=1)


class LoggerConfig(Schema):
    enabled: bool = True
    progress: bool | None = None
    refresh_rate: int = Field(default=10, ge=1)


class TrainerConfig(Schema):
    epochs: int = Field(default=3, ge=1)
    device: str = "cpu"
    seed: int = Field(default=42, ge=0, lt=2**32)
    threads: int = Field(default=2, ge=1)
    output_dir: str = "outputs"


class TrainConfig(Schema):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)


def assign(values: dict, key: str, value: object) -> None:
    parts = key.split(".")
    target = values
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"配置路径冲突: {key}")
    target[parts[-1]] = value


def resolve_environment(value):
    if isinstance(value, dict):
        return {k: resolve_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_environment(v) for v in value]
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{oc\.env:([A-Za-z_][A-Za-z_0-9]*)(?:,(.*))?\}", value)
        if match:
            name, default = match.groups()
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise ValueError(f"缺少配置环境变量: {name}")
            return yaml.safe_load(resolved)
    return value


def read_config(
    path: Path | None, overrides: dict, dotlist: list[str] | None = None, base: dict | None = None
) -> TrainConfig:
    values = yaml.safe_load(path.read_text()) if path else (base or {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("配置必须为 YAML mapping")
    for item in dotlist or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"override 必须为 group.field=value: {item}")
        assign(values, key, yaml.safe_load(value))
    for key, value in overrides.items():
        if value is not None:
            assign(values, key, value)
    return TrainConfig.model_validate(resolve_environment(values))
