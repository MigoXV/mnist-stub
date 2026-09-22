from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from mnist_stub.common import write_json

CLI = [sys.executable, "-m", "mnist_stub.commands.app"]


def run_smoke(output_dir: Path, data_dir: Path, real: bool, device: str) -> dict:
    directory = output_dir.resolve() / uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    report = {
        "status": "running",
        "real": real,
        "device": device,
        "stages": [],
        "report": str(directory / "report.json"),
    }
    service = None

    def command(name: str, arguments: list[str]) -> dict:
        started = time.monotonic()
        process = subprocess.run(CLI + arguments, text=True, capture_output=True, timeout=1200)
        (directory / f"{name}.log").write_text(process.stderr + "\n" + process.stdout)
        stage = {
            "stage": name,
            "exit_code": process.returncode,
            "seconds": time.monotonic() - started,
        }
        report["stages"].append(stage)
        write_json(directory / "report.json", report)
        if process.returncode:
            raise RuntimeError(f"{name} 失败，详见 {directory / (name + '.log')}")
        result = json.loads(process.stdout.strip().splitlines()[-1])
        report.setdefault("artifacts", {})[name] = result
        return result

    try:
        common = [
            "--output-dir",
            str(directory),
            "--data-dir",
            str(data_dir.resolve()),
            "--device",
            device,
        ]
        if not real:
            common += ["--synthetic"]
        first = command("train", ["train", "--epochs", "1", *common])
        resumed = command(
            "resume",
            ["train", "--resume", first["checkpoint"], "--epochs", "3" if real else "2", *common],
        )
        pretrain = None
        for name, subset, freezes, lr in (
            ("pretrain", "a", 0, 0.001),
            ("finetune", "b", 1, 0.0001),
        ):
            config = directory / f"{name}.yaml"
            config.write_text(f"subset: {subset}\nfreeze_epochs: {freezes}\nlearning_rate: {lr}\n")
            args = ["train", "--config", str(config), "--epochs", "2", *common]
            if name == "finetune":
                assert pretrain is not None
                args += ["--init-model", pretrain["model"]]
            result = command(name, args)
            if name == "pretrain":
                pretrain = result
        asset = directory / "asset"
        command("export", ["export", resumed["checkpoint"], str(asset)])
        if real:
            evaluation = command(
                "evaluate",
                [
                    "evaluate",
                    "--model",
                    str(asset),
                    "--data-dir",
                    str(data_dir.resolve()),
                    "--device",
                    device,
                ],
            )
            report["accuracy"] = evaluation["accuracy"]
            if evaluation["accuracy"] < 0.97:
                raise RuntimeError("标准训练测试准确率低于 97%")
        # Use a fresh process with no accessible training imports, data or checkpoint paths.
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        env = os.environ.copy()
        env["MNIST_MODEL"] = str(asset)
        env["MNIST_DEVICE"] = device
        service_code = (
            "import sys, importlib.abc; "
            "exec('class Guard(importlib.abc.MetaPathFinder):\\n"
            " def find_spec(self, fullname, path=None, target=None):\\n"
            '  if fullname.startswith(("mnist_stub.training", "torchvision")):'
            ' raise ImportError("training import forbidden")\'); '
            "sys.meta_path.insert(0, Guard()); "
            "import torch; torch.set_num_threads(2); "
            "from mnist_stub.web.app import create_app; import uvicorn; "
            f"uvicorn.run(create_app(web_dir=None), host='127.0.0.1', port={port})"
        )
        with (directory / "service.log").open("w") as log:
            service = subprocess.Popen(
                [sys.executable, "-c", service_code], stdout=log, stderr=log, cwd=directory, env=env
            )
            base = f"http://127.0.0.1:{port}"
            for _ in range(200):
                if service.poll() is not None:
                    raise RuntimeError("独立服务启动失败，详见 service.log")
                try:
                    with urllib.request.urlopen(base + "/readyz", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("服务就绪超时")
            image = Image.new("L", (96, 96), 0)
            ImageDraw.Draw(image).line([(20, 20), (75, 20), (42, 80)], fill=255, width=9)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            request = urllib.request.Request(
                base + "/api/predict?features=true",
                data=buffer.getvalue(),
                headers={"Content-Type": "image/png"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                result = json.load(response)
            expected_id = json.loads((asset / "manifest.json").read_text())["model_id"]
            if result["model_id"] != expected_id:
                raise RuntimeError("服务加载的模型资产与本次导出不一致")
            if len(result["probabilities"]) != 10 or abs(sum(result["probabilities"]) - 1) > 1e-5:
                raise RuntimeError("概率契约不匹配")
            if set(result["features"]) != {"conv1", "pool1", "conv2", "pool2", "embedding"}:
                raise RuntimeError("特征契约不匹配")
            report["stages"].append({"stage": "independent_http_inference", "status": "passed"})
            service.terminate()
            service.wait(timeout=20)
            if "Application shutdown complete." not in (directory / "service.log").read_text():
                raise RuntimeError("服务未完成优雅关闭")
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError("服务退出后端口仍然开放")
            report["stages"].append({"stage": "shutdown", "exit_code": service.returncode})
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
    finally:
        if service and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=20)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait(timeout=5)
        write_json(directory / "report.json", report)
    return report
