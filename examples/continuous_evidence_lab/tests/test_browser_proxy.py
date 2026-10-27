from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest
from backend import browser_proxy, mcp
from backend.mcp import BRIDGE_CLIENT, mcp_servers


def test_browser_bridge_receives_only_its_required_environment(monkeypatch):
    monkeypatch.setenv("BROWSER_BRIDGE_TOKEN", "t" * 32)
    monkeypatch.setenv("DATABASE_URL", "not-for-browser")
    monkeypatch.setenv("EXPLABS_API_KEY", "not-for-browser")
    config = mcp_servers()["evidence-browser"]
    assert config["command"] == sys.executable
    assert config["args"] == [str(BRIDGE_CLIENT)]
    assert set(config["env"]) == {
        "BROWSER_BRIDGE_HOST",
        "BROWSER_BRIDGE_PORT",
        "BROWSER_BRIDGE_TOKEN",
    }
    assert config["env"]["BROWSER_BRIDGE_TOKEN"] == "t" * 32


def test_browser_bridge_is_registered_only_when_configured(monkeypatch):
    monkeypatch.delenv("BROWSER_BRIDGE_TOKEN", raising=False)
    assert set(mcp_servers()) == {"evidence-filesystem", "evidence-everything"}


def test_stdio_servers_stop_when_the_runtime_closes_their_session(monkeypatch):
    monkeypatch.setenv("BROWSER_BRIDGE_TOKEN", "t" * 32)
    servers = mcp_servers()
    assert len(servers) == 3
    assert all(config["keep_alive"] is False for config in servers.values())


def test_installed_mcp_servers_do_not_contact_npm(monkeypatch):
    monkeypatch.setattr(mcp, "which", lambda binary: f"/usr/local/bin/{binary}")
    servers = mcp_servers()
    assert servers["evidence-filesystem"]["command"] == (
        "/usr/local/bin/mcp-server-filesystem"
    )
    assert servers["evidence-filesystem"]["args"] == [
        str(Path(__file__).parents[1] / "workspace")
    ]
    assert servers["evidence-everything"]["command"] == (
        "/usr/local/bin/mcp-server-everything"
    )
    assert servers["evidence-everything"]["args"] == ["stdio"]


def test_uninstalled_mcp_servers_use_pinned_packages(monkeypatch):
    monkeypatch.setattr(mcp, "which", lambda binary: None)
    servers = mcp_servers()
    assert servers["evidence-filesystem"]["command"] == "npx"
    assert servers["evidence-filesystem"]["args"][:3] == [
        "-y",
        "@modelcontextprotocol/server-filesystem@2026.8.31",
        str(Path(__file__).parents[1] / "workspace"),
    ]
    assert servers["evidence-everything"]["args"] == [
        "-y",
        "@modelcontextprotocol/server-everything@2026.8.31",
        "stdio",
    ]


async def request(port: int, target: str, method: str = "GET") -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"{method} {target} HTTP/1.1\r\nHost: example.com\r\n\r\n".encode())
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


@pytest.mark.asyncio
async def test_public_redirect_to_private_is_denied_before_connect(monkeypatch):
    attempts = []

    async def origin(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    origin_server = await asyncio.start_server(origin, "127.0.0.1", 0)
    origin_port = origin_server.sockets[0].getsockname()[1]
    real_connect = browser_proxy.connect_target

    async def connect(host, port, approved_targets=frozenset()):
        attempts.append(host)
        if host == "example.com":
            return await asyncio.open_connection("127.0.0.1", origin_port)
        return await real_connect(host, port, approved_targets)

    monkeypatch.setattr(browser_proxy, "connect_target", connect)
    async with origin_server:
        proxy = await asyncio.start_server(browser_proxy.handle_proxy, "127.0.0.1", 0)
        async with proxy:
            proxy_port = proxy.sockets[0].getsockname()[1]
            first = await request(proxy_port, "http://example.com/")
            second = await request(
                proxy_port, "http://169.254.169.254/latest/meta-data/"
            )
    assert b"302 Found" in first
    assert b"403 Forbidden" in second
    assert attempts == ["example.com", "169.254.169.254"]


@pytest.mark.asyncio
async def test_rebinding_is_rechecked_for_each_request_without_private_socket(
    monkeypatch,
):
    resolutions = 0
    connected = []

    async def resolve(host, port, **kwargs):
        nonlocal resolutions
        resolutions += 1
        address = "93.184.216.34" if resolutions == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    class ConnectedWriter:
        def get_extra_info(self, name):
            return ("93.184.216.34", 80)

    async def connect(address, port):
        connected.append((address, port))
        return asyncio.StreamReader(), ConnectedWriter()

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    monkeypatch.setattr(asyncio, "open_connection", connect)
    await browser_proxy.connect_target("example.com", 80)
    with pytest.raises(ValueError, match="not permitted"):
        await browser_proxy.connect_target("example.com", 80)
    assert connected == [("93.184.216.34", 80)]


@pytest.mark.asyncio
async def test_connected_peer_must_match_checked_ip(monkeypatch):
    async def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    class WrongPeer:
        def get_extra_info(self, name):
            return ("127.0.0.1", 80)

        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def connect(address, port):
        return asyncio.StreamReader(), WrongPeer()

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    monkeypatch.setattr(asyncio, "open_connection", connect)
    with pytest.raises(ValueError, match="connected peer"):
        await browser_proxy.connect_target("example.com", 80)


@pytest.mark.asyncio
async def test_target_site_rebinding_to_other_private_address_is_denied(monkeypatch):
    async def resolve(host, port, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("172.20.0.6", port),
            )
        ]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="not permitted"):
        await browser_proxy.connect_target("target-site", 80, frozenset({"172.20.0.5"}))


def test_target_site_only_allows_private_http_address():
    approved = frozenset({"172.20.0.5", "169.254.169.254", "127.0.0.1"})
    assert browser_proxy.permitted_address("target-site", "172.20.0.5", 80, approved)
    assert not browser_proxy.permitted_address(
        "target-site", "172.20.0.6", 80, approved
    )
    assert not browser_proxy.permitted_address(
        "target-site", "169.254.169.254", 80, approved
    )
    assert not browser_proxy.permitted_address("target-site", "127.0.0.1", 80, approved)
    assert not browser_proxy.permitted_address(
        "target-site", "172.20.0.5", 443, approved
    )


def test_mcp_bridge_hides_unsafe_browser_code_tool():
    pending = set()
    request_line = b'{"jsonrpc":"2.0","id":3,"method":"tools/list"}\n'
    browser_proxy.filter_mcp_message(request_line, pending, from_browser=False)
    assert pending == {"3"}
    response = {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "tools": [
                {"name": "browser_navigate"},
                {"name": "browser_run_code_unsafe"},
                {"name": "browser_snapshot"},
            ]
        },
    }
    filtered = browser_proxy.filter_mcp_message(
        json.dumps(response).encode(), pending, from_browser=True
    )
    assert [item["name"] for item in json.loads(filtered)["result"]["tools"]] == [
        "browser_navigate",
        "browser_snapshot",
    ]
    assert not pending
    unsafe_call = (
        b'{"jsonrpc":"2.0","id":4,"method":"tools/call",'
        b'"params":{"name":"browser_run_code_unsafe","arguments":{}}}\n'
    )
    assert (
        json.loads(browser_proxy.blocked_mcp_call(unsafe_call))["error"]["code"]
        == -32602
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "method"),
    [
        ("169.254.169.254:80", "CONNECT"),
        ("http://127.0.0.1/admin", "GET"),
        ("http://[::1]/private.js", "GET"),
        ("http://target-site:443/", "GET"),
    ],
)
async def test_tunnels_and_subresources_cannot_reach_internal_hosts(target, method):
    proxy = await asyncio.start_server(browser_proxy.handle_proxy, "127.0.0.1", 0)
    async with proxy:
        result = await request(proxy.sockets[0].getsockname()[1], target, method)
    assert result.startswith(b"HTTP/1.1 403 Forbidden")


@pytest.mark.asyncio
async def test_mcp_bridge_rejects_unauthenticated_clients_without_starting_child(
    monkeypatch,
):
    started = False

    async def start_child(*args, **kwargs):
        nonlocal started
        started = True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start_child)
    server = await asyncio.start_server(
        lambda reader, writer: browser_proxy.handle_mcp_bridge(
            reader, writer, frozenset(), "x" * 32
        ),
        "127.0.0.1",
        0,
    )
    async with server:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", server.sockets[0].getsockname()[1]
        )
        writer.write(b"wrong-token\n")
        await writer.drain()
        assert await reader.read() == b""
        writer.close()
        await writer.wait_closed()
    assert not started


def test_browser_runtime_is_isolated_and_pinned():
    root = Path(__file__).parents[1]
    compose = (root / "compose.yaml").read_text()
    containerfile = (root / "browser.Containerfile").read_text()
    backend_containerfile = (root / "backend.Containerfile").read_text()
    mcp_client = (root / "backend" / "mcp.py").read_text()

    browser_service = compose.split("  browser:\n", 1)[1].split("  backend:\n", 1)[0]
    assert "DATABASE_URL" not in browser_service
    assert "EXPLABS_API_KEY" not in browser_service
    assert "LANGFUSE_" not in browser_service
    assert "volumes:" not in browser_service
    assert "browser-control" in browser_service
    assert "browser-egress" in browser_service
    assert 'cap_drop: ["ALL"]' in browser_service
    assert 'cap_add: ["SYS_CHROOT"]' in browser_service
    assert "chromium-1247/chrome-linux64/chrome" in containerfile
    assert "Google Chrome for Testing 155.0.8059.12" in containerfile
    assert "--no-sandbox" not in containerfile
    assert "playwright" not in backend_containerfile.lower()
    assert "browser_bridge_client" in mcp_client
