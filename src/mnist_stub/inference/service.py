from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from mnist_stub.inference.runtime import Runtime

logger = logging.getLogger(__name__)


class BusyError(Exception):
    pass


class InferenceService:
    def __init__(self, asset: Path, device: str, capacity: int = 8, timeout: float = 10) -> None:
        if capacity < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("队列容量及超时时间必须为正数")
        self.asset, self.device, self.timeout = asset, device, timeout
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mnist-model")
        self.runtime: Runtime | None = None
        self.worker: asyncio.Task | None = None
        self.ready = False
        self.active = 0
        self.high_water = 0
        self.counts: Counter = Counter()
        self.latencies: deque = deque(maxlen=1000)
        self.waits: deque = deque(maxlen=1000)

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self.runtime = await loop.run_in_executor(
                self.executor, Runtime, self.asset, self.device
            )
            self.worker = asyncio.create_task(self._work())
            self.ready = True
        except BaseException:
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise

    async def predict(self, payload: bytes, features: bool, polarity: str) -> dict:
        self.counts["requests"] += 1
        if not self.ready:
            self.counts["unavailable"] += 1
            raise BusyError("服务尚未就绪")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        start = time.monotonic()
        try:
            self.queue.put_nowait((future, start, payload, features, polarity))
        except asyncio.QueueFull as exc:
            self.counts["rejected"] += 1
            raise BusyError("推理队列已满") from exc
        self.high_water = max(self.high_water, self.queue.qsize())
        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
            self.counts["success"] += 1
            return result
        except asyncio.TimeoutError:
            self.counts["timeout"] += 1
            raise
        except asyncio.CancelledError:
            self.counts["cancelled"] += 1
            raise
        except ValueError:
            self.counts["invalid"] += 1
            raise
        except Exception:
            self.counts["errors"] += 1
            raise
        finally:
            self.latencies.append((time.monotonic() - start) * 1000)

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    self.queue.task_done()
                    return
                future, start, payload, features, polarity = item
                try:
                    if future.done():
                        continue
                    wait = time.monotonic() - start
                    self.waits.append(wait * 1000)
                    if wait >= self.timeout:
                        future.set_exception(asyncio.TimeoutError())
                        continue
                    self.active = 1
                    result = await loop.run_in_executor(
                        self.executor, partial(self.runtime.predict, payload, features, polarity)
                    )
                    if not future.done():
                        result["queue_ms"] = wait * 1000
                        future.set_result(result)
                    else:
                        self.counts["late_discarded"] += 1
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    self.active = 0
                    self.queue.task_done()
        finally:
            self.ready = False

    async def close(self) -> None:
        self.ready = False
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if item is not None and not item[0].done():
                item[0].set_exception(BusyError("服务正在关闭"))
            self.queue.task_done()
        if self.worker and not self.worker.done():
            self.queue.put_nowait(None)
            await self.worker
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.runtime = None

    def metrics(self) -> dict:
        ordered = sorted(self.latencies)
        return {
            "counts": dict(self.counts),
            "active": self.active,
            "queued": self.queue.qsize(),
            "capacity": self.queue.maxsize,
            "queue_high_water": self.high_water,
            "latency_window": len(ordered),
            "latency_ms": {
                f"p{p}": ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))]
                if ordered
                else 0
                for p in (50, 95, 99)
            },
            "mean_queue_ms": sum(self.waits) / len(self.waits) if self.waits else 0,
        }
