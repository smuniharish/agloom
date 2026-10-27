"""Connection-time outbound policy for the Evidence Lab Chromium process."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import socket
from urllib.parse import urlsplit

MAX_HEADERS = 65536
UNSAFE_TOOL = "browser_run_code_unsafe"


def permitted_address(
    host: str, address: str, port: int, approved_targets: frozenset[str]
) -> bool:
    if host.lower().rstrip(".") == "target-site":
        ip = ipaddress.ip_address(address)
        return (
            port == 80
            and address in approved_targets
            and ip.is_private
            and not (ip.is_loopback or ip.is_link_local)
        )
    return ipaddress.ip_address(address).is_global


async def connect_target(
    host: str, port: int, approved_targets: frozenset[str] = frozenset()
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if not host or not 1 <= port <= 65535:
        raise ValueError("invalid destination")
    if host.lower().rstrip(".") == "target-site" and port != 80:
        raise ValueError("target-site is approved only on port 80")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError("destination cannot be resolved") from exc
    candidates = {item[4][0] for item in addresses}
    if not candidates or any(
        not permitted_address(host, address, port, approved_targets)
        for address in candidates
    ):
        raise ValueError("destination is not permitted")
    last_error: OSError | None = None
    for address in candidates:
        try:
            # Connect to the checked numeric peer, never resolve the hostname again.
            reader, writer = await asyncio.open_connection(address, port)
            peer = writer.get_extra_info("peername")
            if peer is None or peer[0] != address:
                writer.close()
                await writer.wait_closed()
                raise ValueError("connected peer differs from approved address")
            return reader, writer
        except OSError as exc:
            last_error = exc
    raise ConnectionError("approved destination could not be connected") from last_error


async def _relay(
    source: asyncio.StreamReader, destination: asyncio.StreamWriter
) -> None:
    try:
        while data := await source.read(65536):
            destination.write(data)
            await destination.drain()
    finally:
        if destination.can_write_eof():
            destination.write_eof()


async def handle_proxy(
    client: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    approved_targets: frozenset[str] = frozenset(),
) -> None:
    upstream: asyncio.StreamWriter | None = None
    try:
        headers = await client.readuntil(b"\r\n\r\n")
        if len(headers) > MAX_HEADERS:
            raise ValueError("request headers too large")
        lines = headers.split(b"\r\n")
        method, target, version = lines[0].decode("ascii").split(" ")
        if version != "HTTP/1.1" or method not in {
            "CONNECT",
            "GET",
            "HEAD",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        }:
            raise ValueError("unsupported proxy request")
        if method == "CONNECT":
            parsed = urlsplit("//" + target)
            if (
                parsed.username
                or parsed.password
                or not parsed.hostname
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("invalid CONNECT authority")
            port = parsed.port
            if port is None:
                raise ValueError("CONNECT port is required")
        else:
            parsed = urlsplit(target)
            if (
                parsed.scheme != "http"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError(
                    "only unauthenticated HTTP proxy requests are supported"
                )
            port = parsed.port or 80
        upstream_reader, upstream = await connect_target(
            parsed.hostname, port, approved_targets
        )
        if method == "CONNECT":
            client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            filtered = [
                line
                for line in lines[1:-2]
                if line and not line.lower().startswith((b"proxy-", b"connection:"))
            ]
            filtered.append(b"Connection: close")
            upstream.write(
                f"{method} {path} {version}\r\n".encode("ascii")
                + b"\r\n".join(filtered)
                + b"\r\n\r\n"
            )
            await upstream.drain()
        await client_writer.drain()
        await asyncio.gather(
            _relay(client, upstream), _relay(upstream_reader, client_writer)
        )
    except (
        ValueError,
        UnicodeError,
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
    ):
        client_writer.write(
            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        await client_writer.drain()
    finally:
        if upstream is not None:
            upstream.close()
            await upstream.wait_closed()
        client_writer.close()
        await client_writer.wait_closed()


def filter_mcp_message(
    data: bytes, pending_lists: set[str], *, from_browser: bool
) -> bytes:
    message = json.loads(data)
    if from_browser:
        if str(message.get("id")) in pending_lists:
            pending_lists.remove(str(message["id"]))
            result = message.get("result")
            if isinstance(result, dict) and isinstance(result.get("tools"), list):
                result["tools"] = [
                    item for item in result["tools"] if item.get("name") != UNSAFE_TOOL
                ]
    elif message.get("method") == "tools/list":
        pending_lists.add(str(message.get("id")))
    return json.dumps(message, separators=(",", ":")).encode() + b"\n"


def blocked_mcp_call(data: bytes) -> bytes | None:
    message = json.loads(data)
    if (
        message.get("method") != "tools/call"
        or message.get("params", {}).get("name") != UNSAFE_TOOL
    ):
        return None
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32602, "message": "Tool not permitted"},
            }
        ).encode()
        + b"\n"
    )


async def approved_target_addresses() -> frozenset[str]:
    try:
        resolved = await asyncio.get_running_loop().getaddrinfo(
            "target-site", 80, type=socket.SOCK_STREAM
        )
    except socket.gaierror:
        approved_targets: frozenset[str] = frozenset()
    else:
        approved_targets = frozenset(item[4][0] for item in resolved)
        if not approved_targets or any(
            not permitted_address("target-site", address, 80, approved_targets)
            for address in approved_targets
        ):
            raise RuntimeError("target-site resolved outside its approved network")
    return approved_targets


async def handle_mcp_bridge(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    approved_targets: frozenset[str],
    token: str,
) -> None:
    child: asyncio.subprocess.Process | None = None
    try:
        supplied = await asyncio.wait_for(reader.readline(), timeout=5)
        if len(supplied) > 4096 or not hmac.compare_digest(
            supplied.rstrip(b"\r\n"), token.encode()
        ):
            return
        server = await asyncio.start_server(
            lambda proxy_reader, proxy_writer: handle_proxy(
                proxy_reader, proxy_writer, approved_targets
            ),
            "127.0.0.1",
            0,
            limit=MAX_HEADERS,
        )
        port = server.sockets[0].getsockname()[1]
        env = os.environ.copy()
        env.pop("BROWSER_BRIDGE_TOKEN", None)
        env["EVIDENCE_BROWSER_PROXY"] = f"http://127.0.0.1:{port}"
        async with server:
            child = await asyncio.create_subprocess_exec(
                "node",
                "/opt/browser/node_modules/@playwright/mcp/cli.js",
                "--headless",
                "--isolated",
                "--block-service-workers",
                "--no-webmcp",
                "--executable-path",
                "/usr/local/bin/agloom-chromium",
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            assert child.stdin is not None
            assert child.stdout is not None
            pending_lists: set[str] = set()

            async def forward_requests() -> None:
                while line := await reader.readline():
                    if response := blocked_mcp_call(line):
                        writer.write(response)
                        await writer.drain()
                        continue
                    child.stdin.write(
                        filter_mcp_message(line, pending_lists, from_browser=False)
                    )
                    await child.stdin.drain()
                child.stdin.close()

            async def forward_responses() -> None:
                while line := await child.stdout.readline():
                    writer.write(
                        filter_mcp_message(line, pending_lists, from_browser=True)
                    )
                    await writer.drain()

            request_task = asyncio.create_task(forward_requests())
            response_task = asyncio.create_task(forward_responses())
            child_task = asyncio.create_task(child.wait())
            done, _ = await asyncio.wait(
                {request_task, child_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if request_task in done:
                request_task.result()
            if child_task not in done:
                child.terminate()
            await child_task
            await response_task
    except (TimeoutError, ConnectionError, json.JSONDecodeError):
        pass
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        writer.close()
        await writer.wait_closed()


async def serve_mcp() -> int:
    token = os.environ.get("BROWSER_BRIDGE_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("BROWSER_BRIDGE_TOKEN must contain at least 32 characters")
    approved_targets = await approved_target_addresses()
    server = await asyncio.start_server(
        lambda reader, writer: handle_mcp_bridge(
            reader, writer, approved_targets, token
        ),
        "0.0.0.0",
        8931,
        limit=MAX_HEADERS,
    )
    async with server:
        await server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(serve_mcp()))
