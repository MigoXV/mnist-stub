from __future__ import annotations

import time
from pathlib import Path

import torch

from mnist_stub.assets import load_asset
from mnist_stub.inference.images import Polarity, prepare_image
from mnist_stub.models.cnn import resolve_device
from mnist_stub.models.processing import MnistProcessor


class Runtime:
    def __init__(self, asset: Path, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        self.model, self.manifest = load_asset(asset)
        self.model.to(self.device).eval()
        with torch.inference_mode():
            self.model(torch.zeros(1, 1, 28, 28, device=self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, payload: bytes, features: bool = False, polarity: Polarity = "auto") -> dict:
        start = time.perf_counter()
        pixels = prepare_image(payload, polarity)
        tensor = MnistProcessor()(pixels)["pixel_values"].to(self.device)
        with torch.inference_mode():
            logits, activations = self.model.forward_features(tensor)
            probabilities = logits.softmax(-1)[0].cpu().tolist()
            result = {
                "prediction": int(logits.argmax(-1).item()),
                "probabilities": probabilities,
                "input": pixels.tolist(),
                "model_id": self.manifest["model_id"],
                "features": {},
            }
            if features:
                result["features"] = {
                    name: {"shape": list(value.shape[1:]), "values": value[0].cpu().tolist()}
                    for name, value in activations.items()
                }
        result["inference_ms"] = (time.perf_counter() - start) * 1000
        return result

    def metadata(self) -> dict:
        return {**self.manifest, "device": str(self.device), "dtype": "float32", "runner": "eager"}
