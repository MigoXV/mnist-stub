from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForImageClassification, AutoProcessor

from mnist_stub.common import sha256, write_json
from mnist_stub.models.cnn import ARCHITECTURE, FEATURES, PREPROCESSING, MnistCNN
from mnist_stub.models.processing import MnistProcessor


def load_asset(path: Path) -> tuple[MnistCNN, dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {
        "format_version": 2,
        "architecture": ARCHITECTURE,
        "labels": list(range(10)),
        "preprocessing": PREPROCESSING,
        "features": FEATURES,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"模型资产字段不兼容: {key}")
    if manifest.get("weights_sha256") != sha256(path / "model.safetensors"):
        raise ValueError("模型权重 SHA-256 校验失败")
    for filename in ("config.json", "preprocessor_config.json"):
        if manifest.get("files", {}).get(filename) != sha256(path / filename):
            raise ValueError(f"模型配置 SHA-256 校验失败: {filename}")
    validate_weights(load_file(path / "model.safetensors"), MnistCNN())
    model = AutoModelForImageClassification.from_pretrained(path, local_files_only=True)
    if not isinstance(model, MnistCNN):
        raise ValueError("模型架构必须为 MnistCNN")
    validate_weights(model.state_dict(), MnistCNN())
    AutoProcessor.from_pretrained(path, local_files_only=True)
    return model.eval(), manifest


def validate_weights(state: dict, model: MnistCNN) -> None:
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("模型权重 key 不匹配")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape:
            raise ValueError(f"模型权重 shape 不匹配: {key}")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"模型权重 dtype 或数值非法: {key}")


def save_asset(state: dict, destination: Path, metadata: dict[str, Any]) -> Path:
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖模型资产: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        model = MnistCNN().eval()
        cpu_state = {key: tensor.detach().cpu().clone() for key, tensor in state.items()}
        validate_weights(cpu_state, model)
        model.load_state_dict(cpu_state)
        model.save_pretrained(temporary)
        MnistProcessor().save_pretrained(temporary)
        digest = sha256(temporary / "model.safetensors")
        manifest = {
            "format_version": 2,
            "architecture": ARCHITECTURE,
            "model_id": digest,
            "weights_sha256": digest,
            "files": {
                name: sha256(temporary / name)
                for name in ("config.json", "preprocessor_config.json")
            },
            "labels": list(range(10)),
            "preprocessing": PREPROCESSING,
            "features": FEATURES,
            "metadata": metadata,
        }
        write_json(temporary / "manifest.json", manifest)
        loaded, _ = load_asset(temporary)
        sample = torch.linspace(-1, 1, 784).reshape(1, 1, 28, 28)
        with torch.inference_mode():
            torch.testing.assert_close(model(sample).logits, loaded(sample).logits, rtol=0, atol=0)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
