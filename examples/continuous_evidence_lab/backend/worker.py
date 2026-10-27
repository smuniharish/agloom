"""The procrastinate worker: `python -m backend.worker`."""

from __future__ import annotations

import logging
import os

from .config import Settings
from .runtime import open_worker_runtime, run_async


async def main() -> None:
    settings = Settings.from_env()
    async with open_worker_runtime(settings) as worker:
        try:
            await worker.run(
                name=os.getenv("WORKER_NAME", "evidence-worker"),
                shutdown_graceful_timeout=float(
                    os.getenv("WORKER_SHUTDOWN_TIMEOUT", "30")
                ),
            )
        finally:
            worker.telemetry.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_async(main())
