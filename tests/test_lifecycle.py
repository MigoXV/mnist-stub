from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

from mnist_stub.checkpointing import load_checkpoint


def test_sigterm_keeps_last_complete_checkpoint(tmp_path):
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mnist_stub.commands.train",
                "fit",
                "--data.synthetic=true",
                "--trainer.max_epochs",
                "9999",
                "--trainer.default_root_dir",
                str(tmp_path),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 20
            checkpoints = []
            while time.monotonic() < deadline:
                checkpoints = list(tmp_path.glob("runs/*/checkpoints/last.ckpt"))
                if checkpoints:
                    break
                assert process.poll() is None, (tmp_path / "process.log").read_text()
                time.sleep(0.05)
            assert checkpoints, "首个 Epoch 未在截止时间内完成"
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)
            run_dir = checkpoints[0].parent.parent
            status = json.loads((run_dir / "status.json").read_text())
            assert status["status"] == "interrupted"
            checkpoint = load_checkpoint(checkpoints[0])
            assert checkpoint["epoch"] >= 0
            assert checkpoint["global_step"] > 0
            assert process.returncode != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
