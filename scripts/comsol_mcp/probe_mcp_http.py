"""Diagnose the community COMSOL MCP over an HTTP transport (path C).

Answers two questions, in order:

1. **Does FastMCP's DNS-rebinding guard reject the ``Origin`` a real client would send?**
   With ``host`` = 127.0.0.1 / localhost / ::1 the ``mcp`` package auto-enables
   ``transport_security`` with ``allowed_origins=["http://127.0.0.1:*", "http://localhost:*",
   "http://[::1]:*"]``. Qoder is an Electron app, so its ``Origin`` is unknown a priori -- if it
   does not match, every request is refused with 403 and path C is dead on arrival. A raw GET on
   the endpoint with several ``Origin`` values reveals the rule without needing Qoder at all.

2. **Does a real MCP handshake work end to end?** We initialize, list tools, and call
   ``pdf_list_modules`` -- deliberately a tool that only reads the filesystem, so the probe never
   triggers the lazy COMSOL start and therefore **never touches the single license**.

Readiness semantics (also relied on by ``mcp_server.interpret_probe``): a bare GET returns
200 + ``text/event-stream`` on ``sse``, and **400** on ``streamable-http`` (missing session id).
Both mean "uvicorn is up". A **404** means the URL path does not match the transport
(``/sse`` <-> sse, ``/mcp`` <-> streamable-http); they must be paired.

Run with the *community* venv's python (it owns the ``mcp`` / ``httpx`` deps), from anywhere::

    D:\\...\\COMSOL_Multiphysics_MCP\\.venv\\Scripts\\python.exe probe_mcp_http.py sse 8765

Exit codes: 0 = handshake succeeded, 2 = handshake failed (the Origin matrix is printed either way).
"""

from __future__ import annotations

import asyncio
import importlib
import sys

TRANSPORT = sys.argv[1] if len(sys.argv) > 1 else "sse"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
BASE = f"http://127.0.0.1:{PORT}"
PATH = "/sse" if TRANSPORT == "sse" else "/mcp"


def _community_import(name: str):
    """Import a module that only exists in the **community** venv.

    Resolved through ``importlib`` instead of a plain ``import``: this file lives in the project tree
    and is therefore type-checked against the **project** interpreter, where ``httpx`` / ``mcp`` are
    absent -- a top-level import would light up five phantom unresolved-reference errors in the IDE.
    Going through ``importlib`` keeps the file clean *and* turns the inevitable ModuleNotFoundError
    into an actionable message. Same approach as ``probe_kb.py``.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"cannot import {name!r}: {exc}\n"
            "This probe must run under the COMMUNITY venv's python (it owns mcp/httpx):\n"
            "  <REPO_DIR>\\.venv\\Scripts\\python.exe scripts\\comsol_mcp\\probe_mcp_http.py"
            f" {TRANSPORT} {PORT}"
        ) from exc


httpx = _community_import("httpx")

#: The Origin values worth testing. The first three are what the guard allows; the rest are what a
#: browser-ish or hostile client might send, and all of them are expected to be refused.
ORIGINS: list[str | None] = [
    None,
    f"http://127.0.0.1:{PORT}",
    f"http://localhost:{PORT}",
    "vscode-file://vscode-app",  # Electron scheme -- the value Qoder would send if it sends one
    "null",
    "http://evil.example.com",
]


def probe_origin(origin: str | None) -> tuple[str, str, str]:
    """Raw GET on the endpoint with a given ``Origin``; report status + content-type + first bytes.

    Reads at most one chunk: the SSE body is an endless event stream, so draining it would block
    until the timeout.
    """
    headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
    if origin is not None:
        headers["Origin"] = origin
    try:
        with httpx.Client(timeout=6.0) as client:
            with client.stream("GET", BASE + PATH, headers=headers) as resp:
                first = ""
                try:
                    for chunk in resp.iter_text():
                        first = chunk
                        break
                except httpx.ReadTimeout:
                    first = "<no body within 6s>"
                body = " ".join(first.split())[:100]
                return (
                    str(resp.status_code),
                    str(resp.headers.get("content-type", ""))[:40],
                    body,
                )
    except Exception as exc:  # noqa: BLE001 - a probe must report, never raise
        return "ERR", type(exc).__name__, " ".join(str(exc).split())[:100]


async def handshake() -> None:
    """Full MCP initialize + list_tools + one COMSOL-free tool call."""
    session_cls = _community_import("mcp").ClientSession

    # sse_client yields (read, write); streamablehttp_client yields (read, write, get_session_id).
    if TRANSPORT == "sse":
        connect = _community_import("mcp.client.sse").sse_client
        async with connect(BASE + PATH) as (read, write):
            async with session_cls(read, write) as session:
                await report(session)
    else:
        connect = _community_import("mcp.client.streamable_http").streamablehttp_client
        async with connect(BASE + PATH) as (read, write, _get_session_id):
            async with session_cls(read, write) as session:
                await report(session)


async def report(session) -> None:  # noqa: ANN001 - the session class is imported dynamically above
    info = await session.initialize()
    print(
        f"INIT_OK       : server={info.serverInfo.name!r} version={info.serverInfo.version!r}"
    )
    tools = await session.list_tools()
    names = [t.name for t in tools.tools]
    print(f"TOOLS         : {len(names)}")
    print(f"TOOLS_SAMPLE  : {', '.join(sorted(names)[:6])}")
    result = await session.call_tool("pdf_list_modules", {})
    text = result.content[0].text if result.content else ""
    print(f"CALL_OK       : pdf_list_modules -> {' '.join(text.split())[:160]}")


def main() -> int:
    print(f"TARGET        : {BASE}{PATH}  (transport={TRANSPORT})")
    print("--- Origin probe (DNS-rebinding guard) ---")
    for origin in ORIGINS:
        status, ctype, body = probe_origin(origin)
        print(f"  Origin={str(origin):28} -> {status:5} {ctype:24} {body}")
    print("--- Real MCP handshake ---")
    try:
        asyncio.run(handshake())
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        print(
            f"HANDSHAKE_ERR : {type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
