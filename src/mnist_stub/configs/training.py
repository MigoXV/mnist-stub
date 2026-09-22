from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    data_dir: str = "data-bin"
    output_dir: str = "outputs"
    device: str = "cpu"
    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    num_workers: int = Field(default=0, ge=0)
    threads: int = Field(default=2, ge=1)
    subset: Literal["full", "a", "b"] = "full"
    freeze_epochs: int = Field(default=0, ge=0)
    train_limit: int | None = Field(default=None, ge=1)
    validation_limit: int | None = Field(default=None, ge=1)
    test_limit: int | None = Field(default=None, ge=1)
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_finite(self) -> TrainConfig:
        if not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate 必须为有限数")
        return self


def read_config(path: Path | None, overrides: dict) -> TrainConfig:
    values = yaml.safe_load(path.read_text()) if path else {}
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("配置必须为 YAML mapping")
    return TrainConfig.model_validate(
        {**values, **{k: v for k, v in overrides.items() if v is not None}}
    )
