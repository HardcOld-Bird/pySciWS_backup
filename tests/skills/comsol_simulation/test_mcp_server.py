"""mcp_server（path C 服务端托管）的纯 Python 测试。

全部不启动 COMSOL、不派生任何进程、不占 license：只覆盖纯函数（transport↔端点映射、URL 拼装、
HTTP 状态码翻译、PowerShell 字面量转义、启动器脚本内容）与被 monkeypatch 隔离的状态机分支。
派生本身（WMI / CreateProcess）与真实 HTTP 探测属于硬件级验证，由 SKILL.md 的 Step 0 手工流程覆盖。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pysci.skills.comsol_simulation.tools import mcp_server as m
from pysci.skills.comsol_simulation.tools import simulation


@pytest.fixture
def fake_settings(tmp_path, monkeypatch):
    """把 mcp_server.settings 指到 tmp，并屏蔽一切真实的进程/网络探测。"""
    runs = tmp_path / "runs"
    runs.mkdir()
    st = SimpleNamespace(
        runs_dir=runs,
        mcp_repo_dir=tmp_path / "repo",
        mcp_exe=tmp_path / "repo" / ".venv" / "Scripts" / "comsol-mcp.exe",
        mcp_transport="sse",
        mcp_host="127.0.0.1",
        mcp_port=8765,
        #: 真实形状：.env 未设 COMSOL_MCP_VERSION，mph discovery 返回的是**完整**版本号。
        #: 这两个值曾经都写成 "6.4"，于是“完整版本号→短名”的规范化从未被测到，
        #: 而它一旦缺失就会让 path C 下每一次 comsol_start 都失败（见下方两个用例）。
        mcp_version=None,
        install=SimpleNamespace(version="6.4.0"),
    )
    monkeypatch.setattr(m, "settings", st)
    # 探测一律返回"端口没开、无人占 license"，让状态机走确定性分支。
    monkeypatch.setattr(m, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(m, "_pid_listening_on", lambda *a, **k: None)
    monkeypatch.setattr(m, "license_holders", lambda: [])
    monkeypatch.setattr(m, "pid_alive", lambda pid: False)
    monkeypatch.setattr(m, "pid_in_job", lambda pid: None)
    monkeypatch.setattr(m, "pid_holds_license", lambda pid: False)
    monkeypatch.setattr(m, "spawn_is_detached", lambda pid: None)
    return st


# ---------------------------------------------------------------------------
# transport ↔ 端点映射
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "transport,path", [("sse", "/sse"), ("streamable-http", "/mcp")]
)
def test_endpoint_path_mapping(transport, path):
    assert m.endpoint_path(transport) == path


@pytest.mark.parametrize("bad", ["http", "stdio", "", "SSE"])
def test_endpoint_path_rejects_non_managed_transport(bad):
    """'http' 是 Qoder mcp.json 的 type 取值，不是 FastMCP 的 transport；'stdio' 由 Qoder 自己派生。

    两者都必须报错，否则会拼出一个 404 的 URL，或去托管一个根本不该托管的 stdio 服务端。
    """
    with pytest.raises(m.McpServerError):
        m.endpoint_path(bad)


def test_mcp_url_and_registration_json(fake_settings):
    assert m.mcp_url() == "http://127.0.0.1:8765/sse"
    assert (
        m.mcp_url("streamable-http", "localhost", 8766) == "http://localhost:8766/mcp"
    )
    block = json.loads(m.mcp_registration_json())
    entry = block["mcpServers"]["comsol"]
    # type 恒为 'sse'：官方文档说 Qoder 会从 URL 自动识别 Streamable HTTP。
    assert entry == {"type": "sse", "url": "http://127.0.0.1:8765/sse"}


# ---------------------------------------------------------------------------
# HTTP 状态码翻译（就绪判据）
# ---------------------------------------------------------------------------
def test_interpret_probe_readiness_rules():
    # 无响应 = 没起来
    assert m.interpret_probe(None, "sse")[0] is False
    # sse 的 200 是就绪
    assert m.interpret_probe(200, "sse") == (True, "200 就绪")
    # streamable-http 的 400（缺 session id）【也是】就绪——这是最容易误判成失败的一条
    ok, why = m.interpret_probe(400, "streamable-http")
    assert ok is True and "400" in why
    # 但 sse 上的 400 不算就绪
    assert m.interpret_probe(400, "sse")[0] is False


def test_interpret_probe_404_means_path_transport_mismatch():
    ok, why = m.interpret_probe(404, "sse")
    assert ok is False
    assert "/sse" in why and "transport" in why


def test_interpret_probe_403_points_at_origin_guard():
    ok, why = m.interpret_probe(403, "sse")
    assert ok is False
    assert "Origin" in why and "vscode-file" in why


def test_interpret_probe_unexpected_code():
    assert m.interpret_probe(500, "sse")[0] is False


# ---------------------------------------------------------------------------
# PowerShell 字面量转义（WMI 派生靠它拼命令行）
# ---------------------------------------------------------------------------
def test_ps_literal_doubles_single_quotes():
    assert m._ps_literal("a b") == "'a b'"
    # PowerShell 单引号字面量里的 ' 必须翻倍，否则命令行会被截断
    assert m._ps_literal("it's") == "'it''s'"


# ---------------------------------------------------------------------------
# 状态文件
# ---------------------------------------------------------------------------
def test_read_state_missing_returns_empty(fake_settings):
    assert m.read_state() == {}


def test_read_state_corrupt_returns_empty(fake_settings):
    (fake_settings.runs_dir / m._STATE_NAME).write_text("{not json", encoding="utf-8")
    assert m.read_state() == {}


def test_read_state_non_dict_returns_empty(fake_settings):
    (fake_settings.runs_dir / m._STATE_NAME).write_text("[1, 2]", encoding="utf-8")
    assert m.read_state() == {}


def test_state_roundtrip(fake_settings):
    m._write_state({"pid": 1234, "spawn_method": "wmi"})
    assert m.read_state() == {"pid": 1234, "spawn_method": "wmi"}
    m._clear_state()
    assert m.read_state() == {}


# ---------------------------------------------------------------------------
# 状态汇总 / stop / ensure 的失败分支
# ---------------------------------------------------------------------------
def test_status_when_nothing_is_listening(fake_settings):
    info = m.mcp_server_status()
    assert info["running"] is False
    assert info["pid"] is None
    assert info["holds_license"] is False
    assert info["license_holders"] == []
    assert info["url"] == "http://127.0.0.1:8765/sse"
    assert info["endpoint_path"] == "/sse"
    assert "mcpServers" in info["registration_json"]


def test_stop_without_server_is_false(fake_settings):
    assert m.stop_mcp_server() is False


def test_ensure_raises_when_exe_missing(fake_settings):
    """找不到 comsol-mcp.exe 时必须给出可操作的修复指引，而不是抛 FileNotFoundError。"""
    with pytest.raises(m.McpServerError) as ei:
        m.ensure_mcp_server(timeout=0.1)
    assert "setup_comsol_mcp.ps1" in str(ei.value)


def test_ensure_rejects_bad_transport_before_spawning(fake_settings, monkeypatch):
    """transport 非法要在派生之前就失败，否则会起一个必然 404 的服务端。"""
    monkeypatch.setattr(fake_settings, "mcp_transport", "http")
    called = []
    monkeypatch.setattr(m, "_spawn_detached", lambda exe: called.append(exe))
    with pytest.raises(m.McpServerError):
        m.ensure_mcp_server(timeout=0.1)
    assert called == []


# ---------------------------------------------------------------------------
# 启动器脚本（WMI 不接受 env，环境变量只能靠它注入）
# ---------------------------------------------------------------------------
def test_write_launcher_injects_env_and_redirects_log(fake_settings):
    exe = fake_settings.mcp_exe
    launcher = m._write_launcher(exe)
    assert launcher.name == m._LAUNCHER_NAME
    text = launcher.read_text(encoding="ascii")
    assert text.startswith("@echo off")
    # 解析成 dict 后做**整值精确比较**。切勿用 substring 断言（"=6.4" in text）：
    # 它会被 "=6.4.0" 蒙混过关，而那正是 mph 拒收、导致 comsol_start 必然失败的格式。
    injected = dict(
        line[4:].split("=", 1) for line in text.splitlines() if line.startswith("set ")
    )
    assert injected["COMSOL_MCP_TRANSPORT"] == "sse"
    assert injected["COMSOL_MCP_HOST"] == "127.0.0.1"
    assert injected["COMSOL_MCP_PORT"] == "8765"
    # URL 型 mcp.json 没有 env 字段，COMSOL_MCP_VERSION 只能在这里给；且必须是 mph 短名
    assert injected["COMSOL_MCP_VERSION"] == "6.4"
    # 追加到日志（日志头由同一函数以 "w" 模式写好，故这里必须是 >>）
    assert f'"{exe}" >> ' in text and "2>&1" in text
    # .cmd 必须纯 ASCII，否则 cmd.exe 按 ANSI 解析会出错
    assert all(ord(ch) < 128 for ch in text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("6.4.0", "6.4"),  # mph discovery 的实际返回 → 上游要的短名（踩过的坑）
        ("6.4", "6.4"),  # 已是短名
        ("6.4.1", "6.4a"),  # patch>0 追加字母，与 mph.discovery.parse 同规则
        ("6.4.0.213", "6.4"),  # 带 build 号
        ("5.3a", "5.3a"),  # 带 patch 字母的短名不能被拆坏
        ("  6.4.0  ", "6.4"),  # 容忍空白
        ("", ""),
        (None, ""),
        ("garbage", "garbage"),  # 认不出来就原样传出，让上游如实报错
    ],
)
def test_mph_version_short_name(raw, expected):
    assert m.mph_version_short_name(raw) == expected


def test_server_env_normalizes_discovered_version(fake_settings):
    """未显式配置 COMSOL_MCP_VERSION 时，发现的完整版本号必须规范化后再注入。

    这是 path C 的致命点：服务端 HTTP 层一切正常（200 就绪、工具可列举、纯文档工具
    照常返回），但每一次 comsol_start 都报 ``Could not locate Comsol 6.4.0 installation.``。
    """
    assert fake_settings.mcp_version is None
    assert fake_settings.install.version == "6.4.0"
    assert m._server_env()["COMSOL_MCP_VERSION"] == "6.4"


def test_write_launcher_truncates_log_with_header(fake_settings):
    log = fake_settings.runs_dir / m._LOG_NAME
    log.write_text("stale output from a previous run\n", encoding="utf-8")
    m._write_launcher(fake_settings.mcp_exe)
    head = log.read_text(encoding="utf-8")
    assert "stale output" not in head
    assert "comsol-mcp sse on 127.0.0.1:8765" in head


# ---------------------------------------------------------------------------
# CLI 接线（parser + 只读子命令）
# ---------------------------------------------------------------------------
def test_parser_mcp_ensure_status_stop():
    p = simulation.build_parser()
    args = p.parse_args(["mcp", "ensure", "--timeout", "30", "--restart", "--json"])
    assert args.func is simulation.cmd_mcp_ensure
    assert args.timeout == 30.0 and args.restart and args.json
    assert p.parse_args(["mcp", "status"]).func is simulation.cmd_mcp_status
    assert p.parse_args(["mcp", "stop"]).func is simulation.cmd_mcp_stop


def test_parser_license():
    args = simulation.build_parser().parse_args(["license", "--json"])
    assert args.func is simulation.cmd_license and args.json


def test_cmd_license_reports_free(monkeypatch, capsys):
    monkeypatch.setattr(simulation._mcp, "license_holders", lambda: [])
    assert simulation.main(["license"]) == 0
    assert "空闲" in capsys.readouterr().out


def test_cmd_license_lists_holders(monkeypatch, capsys):
    holders = [
        {"pid": 24928, "exe": "python.exe", "jvm": r"D:\comsol\jre\bin\server\jvm.dll"}
    ]
    monkeypatch.setattr(simulation._mcp, "license_holders", lambda: holders)
    assert simulation.main(["license"]) == 0
    out = capsys.readouterr().out
    assert "24928" in out and "jvm.dll" in out


def test_cmd_mcp_status_is_readonly(fake_settings, monkeypatch, capsys):
    """status 绝不派生进程——它是只读诊断。"""
    monkeypatch.setattr(
        simulation._mcp,
        "_spawn_detached",
        lambda exe: pytest.fail("mcp status must never spawn"),
    )
    assert simulation.main(["mcp", "status"]) == 0
    out = capsys.readouterr().out
    assert "running       : False" in out
    assert "http://127.0.0.1:8765/sse" in out


def test_cmd_mcp_stop_reports_noop(fake_settings, capsys):
    assert simulation.main(["mcp", "stop"]) == 0
    assert "没有发现运行中的服务端" in capsys.readouterr().out
