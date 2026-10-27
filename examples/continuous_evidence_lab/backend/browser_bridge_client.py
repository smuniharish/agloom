"""Authenticated stdio-to-TCP client for the isolated browser service."""

from __future__ import annotations

import asyncio
import os
import sys


async def relay() -> int:
    host = os.environ.get("BROWSER_BRIDGE_HOST", "browser")
    port = int(os.environ.get("BROWSER_BRIDGE_PORT", "8931"))
    token = os.environ.get("BROWSER_BRIDGE_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("BROWSER_BRIDGE_TOKEN must contain at least 32 characters")
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(token.encode() + b"\n")
    await writer.drain()

    async def send() -> None:
        while line := await asyncio.to_thread(sys.stdin.buffer.readline):
            writer.write(line)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()

    async def receive() -> None:
        while data := await reader.read(65536):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    await asyncio.gather(send(), receive())
    writer.close()
    await writer.wait_closed()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(relay()))
