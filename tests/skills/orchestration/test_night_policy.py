"""夜航模式 n1（task-20261014-psep2-n1）：单开关读取面 + 政策注入层（SessionStart/PostCompact）。

钉四件事：① 读取面 fail-open（文件/段/键缺席、非法值 → false 且出声；热切换现读现生效）；
② 政策全文含授权书四要素；③ 载荷形状＝既有钩子体系 ``hookSpecificOutput.additionalContext``
（ON 注出 / OFF 零注入）；④ 钩子 fail-open（异常 exit 0、零输出，不塌主链路）。
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import night, settings

HOOK = PROJECT_ROOT / ".qoder" / "hooks" / "night-policy.sh"


@pytest.fixture
def toml(tmp_path, monkeypatch):
    """读取面指向 tmp 的 pysci_settings.toml，返回写入助手。"""
    path = tmp_path / "pysci_settings.toml"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)

    def write(body: str) -> None:
        path.write_text(body, encoding="utf-8")

    return write


# ---------------------------------------------------------------------------
# ① 读取面 fail-open（缺席/非法一律 false 且出声）+ 热切换
# ---------------------------------------------------------------------------
def test_read_surface_missing_file_warns(toml):
    on, warns = settings.night_enabled()
    assert on is False and warns and "缺席" in warns[0]


def test_read_surface_missing_section_or_key_warns(toml):
    toml("[inspector]\nenabled = false\n")
    on, warns = settings.night_enabled()
    assert on is False and warns and "未配置" in warns[0]

    toml("[night]\n# 只有注释，没有键\n")
    on, warns = settings.night_enabled()
    assert on is False and warns and "未配置" in warns[0]


def test_read_surface_illegal_value_falls_back_and_speaks(toml):
    toml('[night]\nenabled = "maybe"\n')
    on, warns = settings.night_enabled()
    assert on is False and warns and "不是合法开关" in warns[0]
    assert "maybe" in warns[0], "非法值出声须含原值（否则写了没生效只能靠猜）"

    toml("[night]\nenabled = 7\n")
    on, warns = settings.night_enabled()
    assert on is False and warns and "不是合法开关" in warns[0]


def test_read_surface_valid_values_and_tolerant_spellings(toml):
    for body, expect in (
        ("[night]\nenabled = true\n", True),
        ("[night]\nenabled = false\n", False),
        ('[night]\nenabled = "on"\n', True),
        ('[night]\nenabled = "OFF"\n', False),
        ("[night]\nenabled = 1\n", True),
        ("[night]\nenabled = 0\n", False),
    ):
        toml(body)
        on, warns = settings.night_enabled()
        assert on is expect and warns == [], body


def test_read_surface_hot_switch_reads_fresh(toml):
    """热切换：改文件后**下一次读取**即生效（钩子每回合现读的语义基座）。"""
    toml("[night]\nenabled = false\n")
    assert settings.night_enabled()[0] is False
    toml("[night]\nenabled = true\n")
    assert settings.night_enabled()[0] is True
    toml("[night]\nenabled = false\n")
    assert settings.night_enabled()[0] is False


# ---------------------------------------------------------------------------
# ② 政策全文四要素（授权书关键词）
# ---------------------------------------------------------------------------
def test_policy_contains_authorization_four_elements():
    t = night.NIGHT_POLICY
    assert "夜航模式生效" in t
    # ① 持续推进不等用户
    assert "持续推进" in t and "不得因等待用户回复而停滞" in t
    # ② 一般决策自行执行 + 事后追溯责任在用户
    assert "自行执行" in t and "事后追溯责任在用户" in t
    # ③ twin 授权：决策权等同用户 + leader-only §3 限时预授权 + 不可达降级
    assert "咨询 twin 并依其裁决执行" in t and "决策权等同用户" in t
    assert "限时预授权" in t and "副组长咨询" in t
    # ④ 两终态 + 禁定时器 + 权限弹窗
    assert "全部完成" in t and "严重阻碍" in t
    assert "禁止把 LLM 当定时器" in t and "权限弹窗" in t
    assert "[night] enabled=true" in t, "授权书须锚在开关上（关闭即失效）"


# ---------------------------------------------------------------------------
# ③ 载荷形状：ON 注出 additionalContext；OFF 零注入
# ---------------------------------------------------------------------------
def test_hook_output_on_and_off():
    on_data = {"night": {"enabled": True}}
    off_data = {"night": {"enabled": False}}
    out = night.policy_hook_output("SessionStart", data=on_data)
    assert out is not None
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "SessionStart"
    assert hso["additionalContext"] == night.NIGHT_POLICY
    # 两触发点同一政策、事件名回填
    assert (
        night.policy_hook_output("PostCompact", data=on_data)["hookSpecificOutput"][
            "hookEventName"
        ]
        == "PostCompact"
    )
    assert night.policy_hook_output("SessionStart", data=off_data) is None


def test_cli_injects_on_and_silent_off(toml, monkeypatch, capsys):
    payload = json.dumps({"hook_event_name": "PostCompact"})
    # OFF：零输出
    toml("[night]\nenabled = false\n")
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    assert night.main(["--settings", str(settings.SETTINGS_PATH)]) == 0
    assert capsys.readouterr().out == ""

    # ON：JSON 注出（事件名回填 + 政策全文）
    toml("[night]\nenabled = true\n")
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    assert night.main(["--settings", str(settings.SETTINGS_PATH)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "PostCompact"
    assert out["hookSpecificOutput"]["additionalContext"] == night.NIGHT_POLICY


def test_cli_garbage_or_empty_stdin_still_exit0(tmp_path, monkeypatch, capsys):
    """载荷非 JSON / 空 → 按 SessionStart 名义；绝不因载荷坏而失败。"""
    toml_file = tmp_path / "pysci_settings.toml"
    toml_file.write_text("[night]\nenabled = true\n", encoding="utf-8")
    for bad in ("", "{broken", "[]", json.dumps({"hook_event_name": 42})):
        monkeypatch.setattr(sys, "stdin", io.StringIO(bad))
        assert night.main(["--settings", str(toml_file)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"


def test_cli_bad_settings_path_is_silent_and_exit0(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert night.main(["--settings", str(tmp_path / "absent.toml")]) == 0
    assert capsys.readouterr().out == "", "配置读不出 → 零注入（fail-open）"


# ---------------------------------------------------------------------------
# ④ 钩子 fail-open：bash 级（环境缺失路径零副作用）+ 脚本契约
# ---------------------------------------------------------------------------
def test_hook_script_fails_open_without_env(tmp_path):
    """QODER_PROJECT_DIR 指向无 .venv 的假根 → exit 0、零输出（不塌主链路）。"""
    if shutil.which("bash") is None:
        pytest.skip("需 bash（Git Bash）")
    proc = subprocess.run(
        ["bash", str(HOOK)],
        input=b'{"hook_event_name":"SessionStart"}',
        capture_output=True,
        env={"QODER_PROJECT_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == b""
