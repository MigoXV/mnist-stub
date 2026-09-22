from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mnist_stub.checkpointing import load_checkpoint
from mnist_stub.configs.training import TrainConfig, read_config
from mnist_stub.tasks.data import MnistDataModule
from mnist_stub.tasks.run import train


def settings(tmp_path: Path, epochs=3, freeze_epochs=0, subset="full", device="cpu"):
    return TrainConfig.model_validate(
        {
            "trainer": {
                "output_dir": str(tmp_path),
                "epochs": epochs,
                "threads": 1,
                "device": device,
            },
            "data": {"synthetic": True, "batch_size": 32, "subset": subset},
            "model": {"freeze_epochs": freeze_epochs},
            "logger": {"progress": False},
        }
    )


def state(run):
    return load_checkpoint(run / "checkpoints/last.ckpt")


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("freeze_epochs,split_epoch,total", [(0, 1, 3), (1, 1, 3), (1, 2, 3)])
def test_epoch_resume_exact(tmp_path, freeze_epochs, split_epoch, total):
    full = train(settings(tmp_path, epochs=total, freeze_epochs=freeze_epochs))
    initial = train(settings(tmp_path, epochs=split_epoch, freeze_epochs=freeze_epochs))
    resumed = train(
        settings(tmp_path, epochs=total, freeze_epochs=freeze_epochs),
        resume=initial / "checkpoints/last.ckpt",
    )
    continuous, restored = state(full), state(resumed)
    for key in ("state_dict", "optimizer_states", "epoch", "global_step", "MnistDataModule"):
        assert_nested_equal(continuous[key], restored[key])
    for key in ("rng", "best_score", "best_model"):
        assert_nested_equal(continuous["mnist"][key], restored["mnist"][key])
    assert (
        json.loads((resumed / "status.json").read_text())["parent_run"]
        == (state(initial)["mnist"]["run_id"])
    )


def test_freeze_unfreeze_and_new_optimizer(tmp_path):
    initial = train(settings(tmp_path, epochs=1, subset="a"))
    frozen = train(
        settings(tmp_path, epochs=1, subset="b", freeze_epochs=1), init_model=initial / "model"
    )
    before, after = state(initial), state(frozen)
    assert_nested_equal(
        before["state_dict"]["model.conv1.weight"], after["state_dict"]["model.conv1.weight"]
    )
    assert not torch.equal(
        before["state_dict"]["model.fc2.weight"], after["state_dict"]["model.fc2.weight"]
    )
    assert after["global_step"] == 3
    for optimizer_state in after["optimizer_states"][0]["state"].values():
        assert optimizer_state["step"].item() == 3
    unfrozen = train(
        settings(tmp_path, epochs=2, subset="b", freeze_epochs=1),
        resume=frozen / "checkpoints/last.ckpt",
    )
    assert not torch.equal(
        after["state_dict"]["model.conv1.weight"],
        state(unfrozen)["state_dict"]["model.conv1.weight"],
    )


def test_disjoint_splits_and_validation_stable(tmp_path):
    metadata = []
    for subset in ("a", "b", "full"):
        cfg = settings(tmp_path, subset=subset)
        data = MnistDataModule(cfg.data, cfg.trainer.seed)
        data.setup()
        metadata.append(data.metadata)
        batch = next(iter(data.train_dataloader()))
        assert batch["pixel_values"].shape[1:] == (1, 28, 28)
        assert batch["label"].dtype == torch.int64
    a, b, full = metadata
    assert not set(a["indices"]["train"]) & set(b["indices"]["train"])
    assert set(full["indices"]["train"]) == set(a["indices"]["train"]) | set(b["indices"]["train"])
    assert not set(full["indices"]["train"]) & set(full["indices"]["validation"])
    assert a["indices"]["validation"] == b["indices"]["validation"]
    assert a["fingerprint"] == b["fingerprint"]


def test_resume_rejects_changed_data_or_policy(tmp_path):
    first = train(settings(tmp_path, epochs=1))
    changed = settings(tmp_path, epochs=2)
    changed.data.batch_size = 8
    with pytest.raises(ValueError, match="恢复配置不兼容"):
        train(changed, resume=first / "checkpoints/last.ckpt")
    with pytest.raises(ValueError, match="互斥"):
        train(
            settings(tmp_path), resume=first / "checkpoints/last.ckpt", init_model=first / "model"
        )
    with pytest.raises(ValueError, match="旧"):
        load_checkpoint(tmp_path / "old.pt")


def test_config_rejects_unknown_and_invalid(tmp_path, monkeypatch):
    for values in (
        {"wat": 1},
        {"trainer": {"epochs": 0}},
        {"optimizer": {"lr": float("nan")}},
        {"data": {"batch_size": "64"}},
        {"data": {"subset": "test"}},
    ):
        with pytest.raises(ValueError):
            TrainConfig.model_validate(values)
    path = tmp_path / "config.yaml"
    path.write_text("trainer:\n  epochs: 2\ndata:\n  batch_size: ${oc.env:BATCH,32}\n")
    monkeypatch.setenv("BATCH", "16")
    config = read_config(path, {"trainer.epochs": 4}, ["trainer.epochs=3"])
    assert config.trainer.epochs == 4
    assert config.data.batch_size == 16


def test_single_checkpoint_preserves_best_snapshot(tmp_path):
    first = train(settings(tmp_path, epochs=1))
    original = state(first)
    original["mnist"]["best_score"] = 1.0
    portable = tmp_path / "portable.ckpt"
    torch.save(original, portable)
    resumed = train(settings(tmp_path, epochs=2), resume=portable)
    best = state(resumed)["mnist"]
    assert best["best_score"] == 1.0
    assert_nested_equal(best["best_model"], original["mnist"]["best_model"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
def test_cuda_training_resume(tmp_path):
    full = train(settings(tmp_path, epochs=2, device="cuda:0"))
    first = train(settings(tmp_path, epochs=1, device="cuda:0"))
    resumed = train(
        settings(tmp_path, epochs=2, device="cuda:0"), resume=first / "checkpoints/last.ckpt"
    )
    for key in ("state_dict", "optimizer_states"):
        assert_nested_equal(state(full)[key], state(resumed)[key])
