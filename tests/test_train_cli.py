from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from mnist_stub.checkpointing import load_checkpoint
from tests.test_training import assert_nested_equal

ENTRYPOINT = [sys.executable, "-m", "mnist_stub.commands.train"]


def execute(tmp_path, *args):
    result = subprocess.run(
        [
            *ENTRYPOINT,
            "fit",
            "--data.synthetic=true",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--logging.progress=false",
            *args,
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_lightning_cli_config_and_resume(tmp_path):
    config = tmp_path / "input.yaml"
    config.write_text("data:\n  batch_size: 16\ntrainer:\n  max_epochs: 1\n")
    full = execute(tmp_path, "--config", str(config), "--trainer.max_epochs=2")
    first = execute(tmp_path, "--config", str(config))
    resumed = execute(
        tmp_path,
        "--config",
        str(Path(first["run_dir"]) / "config.yaml"),
        "--ckpt_path",
        first["checkpoint"],
        "--trainer.max_epochs=2",
    )
    expected = load_checkpoint(Path(full["checkpoint"]))
    actual = load_checkpoint(Path(resumed["checkpoint"]))
    for key in ("state_dict", "optimizer_states", "global_step", "MnistDataModule"):
        assert_nested_equal(expected[key], actual[key])
    assert actual["mnist"]["config"]["data"]["batch_size"] == 16
    saved = yaml.safe_load((Path(resumed["run_dir"]) / "config.yaml").read_text())
    assert saved["trainer"]["max_epochs"] == 2
    assert saved["trainer"]["accelerator"] == "cpu"


def test_native_print_config_and_environment(tmp_path):
    env = {**os.environ, "MNIST_FIT__DATA__BATCH_SIZE": "17"}
    result = subprocess.run(
        [*ENTRYPOINT, "fit", "--print_config"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    parsed = yaml.safe_load(result.stdout)
    assert parsed["data"]["batch_size"] == 17
    assert "max_epochs" in parsed["trainer"]


def test_unknown_argument_and_typer_train_removed():
    result = subprocess.run(
        [*ENTRYPOINT, "fit", "--trainer.epochs=1"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    result = subprocess.run(
        [sys.executable, "-m", "mnist_stub.commands.app", "train"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "No such command" in result.stderr


def test_criterion_parameter_remains_trainable():
    from mnist_stub.configs.training import OptimizerConfig
    from mnist_stub.tasks.classification import MnistTask

    task = MnistTask(OptimizerConfig(), freeze_epochs=1)
    optimizer = task.configure_optimizers()
    assert all(p.requires_grad for group in optimizer.param_groups for p in group["params"])
    assert isinstance(optimizer, torch.optim.AdamW)
