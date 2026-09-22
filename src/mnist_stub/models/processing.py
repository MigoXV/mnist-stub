from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoProcessor, BaseImageProcessor
from transformers.image_processing_base import BatchFeature

from mnist_stub.models.cnn import PREPROCESSING, MnistConfig


class MnistProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def preprocess(self, images, return_tensors="pt", **kwargs) -> BatchFeature:
        if return_tensors != "pt":
            raise ValueError("MNIST processor 仅支持 return_tensors=pt")
        if not isinstance(images, (list, tuple)):
            images = [images]
        tensors = []
        for image in images:
            array = np.asarray(image)
            if array.shape != (28, 28) or array.dtype != np.uint8:
                raise ValueError("image 必须为 28×28 uint8 灰度图")
            tensor = torch.from_numpy(array.copy()).float().unsqueeze(0).div(255)
            tensors.append((tensor - PREPROCESSING["mean"]) / PREPROCESSING["std"])
        return BatchFeature({"pixel_values": torch.stack(tensors)})


AutoImageProcessor.register(MnistConfig, slow_image_processor_class=MnistProcessor)
AutoProcessor.register(MnistConfig, MnistProcessor)
