from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import pytest
from tqdm.auto import tqdm

from mnist_stub.logging import training_logging


def test_cli_progress_uses_stderr(tmp_path):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mnist_stub.commands.train",
            "fit",
            "--data.synthetic=true",
            "--trainer.max_epochs",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
            "--logging.refresh_rate=1",
        ],
        stdout=subprocess.PIPE,
        stderr=slave,
        text=True,
    )
    os.close(slave)
    chunks = []
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                chunks.append(chunk)
            elif process.poll() is not None:
                break
        stdout, _ = process.communicate(timeout=5)
        terminal = b"".join(chunks).decode()
        assert process.returncode == 0, terminal
        result = json.loads(stdout)
        assert "\r" not in stdout
        assert "Epoch 0" in terminal and "100%" in terminal
        assert terminal.count("epoch=1 frozen=") == 1
        with open(os.path.join(result["run_dir"], "train.log")) as stream:
            file_log = stream.read()
        assert "epoch=1 frozen=" in file_log
        assert "\x1b" not in file_log and "\r" not in file_log
    finally:
        os.close(master)
        if process.poll() is None:
            process.kill()
            process.wait()


def test_logging_cleanup_after_exception(tmp_path, capsys):
    root = logging.getLogger()
    handlers = root.handlers[:]
    for index in range(2):
        with pytest.raises(RuntimeError):
            with training_logging(tmp_path / f"{index}.log", True):
                with tqdm(total=2, file=sys.stderr) as bar:
                    bar.update()
                    logging.getLogger("lightning.pytorch").warning("during-progress")
                    raise RuntimeError("expected")
        assert root.handlers == handlers
    assert capsys.readouterr().err.count("during-progress") == 2
    assert not tqdm._instances


def test_redirected_cli_has_no_progress(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "mnist_stub.commands.train",
            "fit",
            "--data.synthetic=true",
            "--trainer.max_epochs",
            "1",
            "--trainer.default_root_dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    json.loads(process.stdout)
    assert "\x1b" not in process.stderr
    assert "it/s]" not in process.stderr
    assert process.stderr.count("epoch=1 frozen=") == 1
