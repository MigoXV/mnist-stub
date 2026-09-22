from __future__ import annotations

import asyncio
import threading

import pytest

from mnist_stub.inference.service import BusyError, InferenceService


def test_bounded_queue_timeout_and_recovery(asset, image_bytes):
    async def run():
        service = InferenceService(asset, "cpu", capacity=1, timeout=0.15)
        await service.start()
        release = threading.Event()
        started = threading.Event()
        original = service.runtime.predict

        def slow(*args):
            started.set()
            release.wait(timeout=5)
            return original(*args)

        service.runtime.predict = slow
        try:
            first = asyncio.create_task(service.predict(image_bytes, False, "auto"))
            await asyncio.to_thread(started.wait, 2)
            second = asyncio.create_task(service.predict(image_bytes, False, "auto"))
            await asyncio.sleep(0)
            with pytest.raises(BusyError, match="队列已满"):
                await service.predict(image_bytes, False, "auto")
            # The event loop stays responsive even while the dedicated thread is busy.
            assert service.active == 1
            assert service.metrics()["queued"] == 1
            for task in (first, second):
                with pytest.raises(asyncio.TimeoutError):
                    await task
            release.set()
            await asyncio.wait_for(service.queue.join(), 5)
            assert service.metrics()["counts"]["late_discarded"] == 1
            service.runtime.predict = original
            service.timeout = 5
            result = await service.predict(image_bytes, True, "auto")
            assert len(result["probabilities"]) == 10
        finally:
            release.set()
            await service.close()
        assert not service.ready

    asyncio.run(run())


def test_concurrent_results_isolated(asset, image_bytes):
    async def run():
        service = InferenceService(asset, "cpu")
        await service.start()
        try:
            outputs = await asyncio.gather(
                *[service.predict(image_bytes, i % 2 == 0, "auto") for i in range(8)]
            )
            assert all(x["probabilities"] == outputs[0]["probabilities"] for x in outputs)
            for i, output in enumerate(outputs):
                assert bool(output["features"]) == (i % 2 == 0)
        finally:
            await service.close()

    asyncio.run(run())
