from __future__ import annotations

import logging

import torch
from lightning import LightningModule
from transformers import AutoModelForImageClassification

from mnist_stub.configs.training import OptimizerConfig
from mnist_stub.criterions import classification_loss
from mnist_stub.models.cnn import MnistConfig

logger = logging.getLogger(__name__)


class MnistTask(LightningModule):
    def __init__(
        self, optimizer: OptimizerConfig, freeze_epochs: int = 0, model_path: str | None = None
    ) -> None:
        super().__init__()
        self.model = (
            AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
            if model_path
            else AutoModelForImageClassification.from_config(MnistConfig())
        )
        self.optimizer_config = optimizer
        self.freeze_epochs = freeze_epochs
        self.model.freeze_backbone(freeze_epochs > 0)

    def forward(self, pixel_values):
        return self.model(pixel_values).logits

    def step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits = self(batch["pixel_values"])
        loss = classification_loss(logits, batch["label"])
        accuracy = (logits.argmax(-1) == batch["label"]).float().mean()
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["label"]),
        )
        self.log(
            f"{prefix}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["label"]),
        )
        if prefix == "train":
            self.log(
                "learning_rate",
                self.optimizers().param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")

    def configure_optimizers(self):
        if not self.freeze_epochs:
            groups = [{"params": list(self.model.parameters())}]
        else:
            groups = [
                {"params": list(self.model.fc1.parameters()) + list(self.model.fc2.parameters())}
            ]
            if self.model.conv1.weight.requires_grad:
                groups.append({"params": self.backbone_parameters()})
        return torch.optim.AdamW(groups, **self.optimizer_config.model_dump())

    def backbone_parameters(self):
        return list(self.model.conv1.parameters()) + list(self.model.conv2.parameters())

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.model.freeze_backbone(checkpoint["epoch"] < self.freeze_epochs)

    def on_train_epoch_start(self) -> None:
        frozen = self.current_epoch < self.freeze_epochs
        was_frozen = not self.model.conv1.weight.requires_grad
        self.model.freeze_backbone(frozen)
        if was_frozen and not frozen:
            self.optimizers().add_param_group({"params": self.backbone_parameters()})
        logger.info(
            "epoch=%s frozen=%s trainable=%s",
            self.current_epoch + 1,
            frozen,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
