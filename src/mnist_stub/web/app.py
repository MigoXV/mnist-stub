from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from mnist_stub.inference.images import MAX_BYTES
from mnist_stub.inference.service import BusyError, InferenceService

logger = logging.getLogger(__name__)
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


class BodyLimit:
    def __init__(self, app, limit: int = MAX_BYTES):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/predict":
            return await self.app(scope, receive, send)
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                raise HTTPException(413, "图片超过 2 MiB")
            return message

        await self.app(scope, limited_receive, send)


def create_app(
    asset: Path | None = None,
    device: str | None = None,
    web_dir: Path | None = WEB_DIST,
    capacity: int = 8,
    timeout: float = 10,
) -> FastAPI:
    model_path = asset or Path(os.environ.get("MNIST_MODEL", "model-bin/mnist"))
    selected_device = device or os.environ.get("MNIST_DEVICE", "cpu")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = InferenceService(model_path, selected_device, capacity, timeout)
        application.state.service = service
        await service.start()
        logger.info(
            "ready model=%s device=%s pid=%s",
            service.runtime.manifest["model_id"],
            selected_device,
            os.getpid(),
        )
        try:
            yield
        finally:
            await service.close()

    application = FastAPI(title="MNIST Fixture", version="1.0", lifespan=lifespan)
    application.add_middleware(BodyLimit)

    @application.get("/healthz")
    async def health():
        return {"status": "alive"}

    @application.get("/readyz")
    async def ready(request: Request):
        service = request.app.state.service
        is_ready = service.ready and service.worker is not None and not service.worker.done()
        return JSONResponse({"ready": is_ready}, status_code=200 if is_ready else 503)

    @application.get("/api/model")
    async def metadata(request: Request):
        return request.app.state.service.runtime.metadata()

    @application.get("/api/metrics")
    async def metrics(request: Request):
        return request.app.state.service.metrics()

    @application.post("/api/predict")
    async def predict(
        request: Request,
        features: bool = Query(False),
        polarity: Literal["auto", "dark", "light"] = Query("auto"),
    ):
        content_type = request.headers.get("content-type", "").split(";")[0]
        if content_type not in {"image/png", "image/jpeg", "application/octet-stream"}:
            raise HTTPException(415, "请求体应为 PNG/JPEG 原始字节")
        payload = await request.body()
        try:
            return await request.app.state.service.predict(payload, features, polarity)
        except BusyError as exc:
            raise HTTPException(503, str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "推理请求超时") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("推理失败")
            raise HTTPException(500, "推理执行失败") from exc

    @application.api_route("/api", methods=["GET", "HEAD"], include_in_schema=False)
    @application.api_route("/api/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def missing_api(path: str = ""):
        raise HTTPException(404, "API 不存在")

    if web_dir is not None:
        if not (web_dir / "index.html").is_file():
            raise RuntimeError("前端构建缺失，请先执行 pnpm --dir src/web run build")
        application.frontend("/", directory=web_dir, fallback="index.html")
    return application


def app() -> FastAPI:
    """Uvicorn factory; constructing the app never loads a model."""
    return create_app()
