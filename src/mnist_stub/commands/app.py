from __future__ import annotations

import json
import logging
import signal
from pathlib import Path

import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
app = typer.Typer(no_args_is_help=True, help="MNIST 训练与独立推理基准夹具")


def emit(value: dict) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False))


@app.command()
def prepare_data(
    data_dir: Path = typer.Option(Path("data-bin"), envvar="MNIST_DATA_DIR"),
) -> None:
    """显式下载并校验 MNIST 数据。"""
    from mnist_stub.training.data import prepare_data as prepare

    prepare(data_dir)
    emit({"data_dir": str(data_dir)})


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config"),
    resume: Path | None = typer.Option(None, "--resume"),
    init_model: Path | None = typer.Option(None, "--init-model"),
    epochs: int | None = typer.Option(None, "--epochs"),
    device: str | None = typer.Option(None, envvar="MNIST_DEVICE"),
    data_dir: str | None = typer.Option(None, envvar="MNIST_DATA_DIR"),
    output_dir: str | None = typer.Option(None, envvar="MNIST_OUTPUT_DIR"),
    synthetic: bool | None = typer.Option(None, "--synthetic/--real"),
    train_limit: int | None = typer.Option(None),
    validation_limit: int | None = typer.Option(None),
    test_limit: int | None = typer.Option(None),
) -> None:
    """训练；resume 的 epochs 表示目标总 Epoch 数。"""
    from mnist_stub.configs.training import read_config
    from mnist_stub.training.checkpoints import load_checkpoint
    from mnist_stub.training.engine import train as execute

    overrides = dict(
        epochs=epochs,
        device=device,
        data_dir=data_dir,
        output_dir=output_dir,
        synthetic=synthetic,
        train_limit=train_limit,
        validation_limit=validation_limit,
        test_limit=test_limit,
    )
    if resume and config is None:
        from mnist_stub.configs.training import TrainConfig

        values = load_checkpoint(resume)["config"]
        values.update({key: value for key, value in overrides.items() if value is not None})
        settings = TrainConfig.model_validate(values)
    else:
        settings = read_config(config, overrides)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("收到 SIGTERM，从最近完整 Epoch 的 Checkpoint 恢复")

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        run = execute(settings, resume, init_model)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    emit(
        {
            "run_dir": str(run),
            "model": str(run / "model"),
            "checkpoint": str(run / "checkpoints" / "last.pt"),
        }
    )


@app.command()
def export(
    checkpoint: Path = typer.Argument(...),
    destination: Path = typer.Argument(...),
    best: bool = typer.Option(True, "--best/--current"),
) -> None:
    """将训练 Checkpoint 转为可独立部署的模型资产。"""
    from mnist_stub.training.engine import export_checkpoint

    emit({"model": str(export_checkpoint(checkpoint, destination, best))})


@app.command()
def evaluate(
    model: Path = typer.Option(..., envvar="MNIST_MODEL"),
    config: Path | None = typer.Option(None),
    data_dir: str | None = typer.Option(None, envvar="MNIST_DATA_DIR"),
    device: str | None = typer.Option(None, envvar="MNIST_DEVICE"),
) -> None:
    from mnist_stub.configs.training import read_config
    from mnist_stub.training.engine import evaluate as execute

    emit(execute(model, read_config(config, {"data_dir": data_dir, "device": device})))


@app.command()
def predict(
    image: Path = typer.Argument(...),
    model: Path = typer.Option(..., envvar="MNIST_MODEL"),
    device: str = typer.Option("cpu", envvar="MNIST_DEVICE"),
    features: bool = typer.Option(False),
    polarity: str = typer.Option("auto"),
) -> None:
    from mnist_stub.inference.runtime import Runtime

    emit(Runtime(model, device).predict(image.read_bytes(), features, polarity))


@app.command()
def serve(
    model: Path = typer.Option(..., envvar="MNIST_MODEL"),
    device: str = typer.Option("cpu", envvar="MNIST_DEVICE"),
    host: str = typer.Option("127.0.0.1", envvar="MNIST_HOST"),
    port: int = typer.Option(8000, min=1, max=65535, envvar="MNIST_PORT"),
    api_only: bool = typer.Option(False, help="仅提供 API，不检查前端构建"),
    threads: int = typer.Option(2, min=1),
) -> None:
    import torch
    import uvicorn

    from mnist_stub.web.app import WEB_DIST, create_app

    torch.set_num_threads(threads)
    uvicorn.run(
        create_app(model, device, None if api_only else WEB_DIST),
        host=host,
        port=port,
        workers=1,
        timeout_graceful_shutdown=15,
    )


@app.command()
def smoke(
    output_dir: Path = typer.Option(Path("outputs/smoke"), envvar="MNIST_OUTPUT_DIR"),
    data_dir: Path = typer.Option(Path("data-bin"), envvar="MNIST_DATA_DIR"),
    real: bool = typer.Option(False, help="使用已准备的完整 MNIST，并要求准确率 >= 97%"),
    device: str = typer.Option("cpu", envvar="MNIST_DEVICE"),
) -> None:
    """通过独立子进程验收训练、恢复、微调、导出与 HTTP 推理。"""
    from mnist_stub.smoke import run_smoke

    report = run_smoke(output_dir, data_dir, real, device)
    emit(report)
    if report["status"] != "passed":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
