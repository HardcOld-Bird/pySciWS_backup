"""夜航**热生效**（adhoc-night-text，用户裁决 2026-10-15）：回合边界差分注入 + 文案修订。

钉五件事：
① 差分只在**跳变**那一轮产出（无变化零输出零写入；首次见会话只记基线不注入）；
② 关→开 = 授权书全文（同 SessionStart 文本）、**开→关 = 零注入**（用户裁决 2026-10-17）；
③ 上次读数**按 session_id 各存一份**（多会话并存不互相误报），且跨 watch.json 整体覆写；
④ 文案修订（④ 删「改回 false」、⑤ 去行话、§3.6 不再悬空、补 consult 调用面；关闭差分
   不得出现「立即停止自主推进」类字样）；
⑤ 通道契约：Stop 钩子（watch-arm.sh）按差分输出走 stderr + exit 2 的既有 rewake 路径。
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import night, settings, watch

HOOK = PROJECT_ROOT / ".qoder" / "hooks" / "watch-arm.sh"


@pytest.fixture
def toml(iso_state):
    """写隔离面的 pysci_settings.toml（iso_state 已把 SETTINGS_PATH/WATCH_PATH 归 tmp）。"""
    path = settings.SETTINGS_PATH

    def write(on: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"[night]\nenabled = {str(on).lower()}\n", encoding="utf-8")

    return write


# ---------------------------------------------------------------------------
# ① 只在跳变那一轮出声
# ---------------------------------------------------------------------------
def test_first_sight_records_baseline_without_injecting(toml):
    """首次见该会话：记基线、**不注入**（SessionStart 已按当时开关注过一次，再注纯烧 token）。"""
    toml(True)
    assert night.diff_message("s1") is None
    # 基线已记下 —— 此后真跳变才捕获
    assert watch.load_state()["night_seen"]["s1"] == "on"


def test_no_change_is_zero_output_and_zero_write(toml):
    toml(True)
    night.diff_message("s1")
    before = watch.WATCH_PATH.read_bytes()
    assert night.diff_message("s1") is None
    assert night.diff_message("s1") is None
    assert watch.WATCH_PATH.read_bytes() == before, "无变化不得改写运行态（零打扰）"


def test_off_to_on_injects_policy_full_text(toml):
    toml(False)
    night.diff_message("s1")
    toml(True)  # 关→开（用户手动拨）
    msg = night.diff_message("s1")
    assert msg == night.NIGHT_POLICY, "关→开须注入授权书全文（与 SessionStart 同一份）"
    assert night.diff_message("s1") is None, "注入后同值再比对不得重复出声"


def test_on_to_off_injects_nothing(toml):
    """**开→关 = 零注入**（用户裁决 2026-10-17）：关闭状态不注入任何提示词。"""
    toml(True)
    night.diff_message("s1")
    toml(False)  # 开→关
    assert night.diff_message("s1") is None
    # 读数仍记账（否则下一轮又会判一次跳变）
    assert night.diff_message("s1") is None


# ---------------------------------------------------------------------------
# ①′ 基线缺口（2026-10-15 真机缺陷）：SessionStart 注入即落基线
# ---------------------------------------------------------------------------
def _run_injection(toml_path, event: str, sid: str, monkeypatch, capsys) -> str:
    """按钩子的实际形态跑一次政策注入（stdin 载荷带 hook_event_name + session_id）。"""
    payload = json.dumps({"hook_event_name": event, "session_id": sid})
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    assert night.main(["--settings", str(toml_path)]) == 0
    return capsys.readouterr().out


def test_session_start_records_baseline(toml, monkeypatch, capsys):
    """SessionStart 注入时顺手落基线——否则基线只能等首个 Stop 补记（缺陷根因）。"""
    toml(True)
    out = _run_injection(
        settings.SETTINGS_PATH, "SessionStart", "s1", monkeypatch, capsys
    )
    assert (
        json.loads(out)["hookSpecificOutput"]["additionalContext"] == night.NIGHT_POLICY
    )
    assert watch.load_state()["night_seen"]["s1"] == "on", (
        "SessionStart 必须落基线（旧行为：等首个 Stop 才记，期间拨动会被当基线吞掉）"
    )


def test_toggle_after_session_start_is_not_swallowed(toml, monkeypatch, capsys):
    """本单核心回归：SessionStart(on) → 拨 off → 首个 Stop → **刹车差分照常产出**。"""
    toml(True)
    _run_injection(settings.SETTINGS_PATH, "SessionStart", "s1", monkeypatch, capsys)
    toml(False)  # 用户拨 off（发生在首个 Stop 之前）
    assert night.diff_message("s1") is None, "OFF 态零注入"


def test_session_start_with_switch_off_records_off(toml, monkeypatch, capsys):
    """反向同理：起始 off 落基线 off → 此后拨 on 才会产出授权书全文。"""
    toml(False)
    _run_injection(settings.SETTINGS_PATH, "SessionStart", "s1", monkeypatch, capsys)
    assert watch.load_state()["night_seen"]["s1"] == "off"
    toml(True)
    assert night.diff_message("s1") == night.NIGHT_POLICY


def test_postcompact_does_not_rewrite_baseline(toml, monkeypatch, capsys):
    """PostCompact 只补注入、**不动读数**：否则它会把 SessionStart 之后的跳变盖掉。"""
    toml(True)
    _run_injection(settings.SETTINGS_PATH, "SessionStart", "s1", monkeypatch, capsys)
    toml(False)  # 拨 off
    out = _run_injection(
        settings.SETTINGS_PATH, "PostCompact", "s1", monkeypatch, capsys
    )
    assert out == "", "off 时 PostCompact 零注入"
    assert watch.load_state()["night_seen"]["s1"] == "on", (
        "PostCompact 不得重写基线（它一写，跳变就被盖掉）"
    )
    assert night.diff_message("s1") is None


def test_session_start_without_session_id_is_silent_noop(toml, monkeypatch, capsys):
    """载荷缺 session_id（或垃圾载荷）：照常注入、但不落基线、不抛。"""
    toml(True)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"hook_event_name": "SessionStart"}))
    )
    assert night.main(["--settings", str(settings.SETTINGS_PATH)]) == 0
    capsys.readouterr()
    assert not watch.load_state().get("night_seen"), "无会话身份不落基线"
    monkeypatch.setattr(sys, "stdin", io.StringIO("{broken"))
    assert night.main(["--settings", str(settings.SETTINGS_PATH)]) == 0
    capsys.readouterr()


def test_cold_path_without_session_start_still_records_then_differs(toml):
    """冷路径不回归：没有 SessionStart 时，首个 ``--diff`` 仍只记基线不注入。"""
    toml(True)
    assert night.diff_message("cold") is None
    assert watch.load_state()["night_seen"]["cold"] == "on"
    toml(False)
    assert night.diff_message("cold") is None


def test_policy_hook_pipes_payload_so_baseline_can_be_written():
    """SessionStart 落基线的前提：``night-policy.sh`` 必须把 stdin 载荷转给 night.py。

    丢了载荷 → `session_id` 读不到 → 基线静默不落 → 本单缺陷原样复发（且注入看起来正常，
    最难发现）。故把这条接线钉死。
    """
    script = (PROJECT_ROOT / ".qoder" / "hooks" / "night-policy.sh").read_text(
        encoding="utf-8"
    )
    assert '<<<"$PAYLOAD"' in script, "必须把载荷喂给 night.py（否则读不到 session_id）"
    assert "pysci.skills.orchestration.tools.night" in script


# ---------------------------------------------------------------------------
# ② 多会话并存：上次读数按 session_id 各存一份（防会话间互相误报）
# ---------------------------------------------------------------------------
def test_last_seen_is_per_session(toml):
    toml(True)
    night.diff_message("s1")
    night.diff_message("s2")  # 两个会话各自记基线
    toml(False)
    assert night.diff_message("s1") is None
    # s2 尚未比对过任何跳变 → 它也是关，但它的基线是 on → 同样出声（各算各的账）
    assert night.diff_message("s2") is None
    seen = watch.load_state()["night_seen"]
    assert seen["s1"] == "off" and seen["s2"] == "off"

    # 关键：一个会话的读数不得被另一会话的比对消费掉
    toml(True)
    assert night.diff_message("s3") is None  # 新会话首次见：只记基线
    assert night.diff_message("s1") == night.NIGHT_POLICY, "s1 的账不该被 s3 吃掉"


def test_missing_session_or_broken_state_never_raises(toml):
    toml(True)
    assert night.diff_message("") is None  # 无会话身份 → 不注入（绝不跨会话误报）
    watch.WATCH_PATH.write_text("{ broken", encoding="utf-8")
    assert night.diff_message("s9") is None, "运行态损坏 → fail-open 零输出（当首次见）"


# ---------------------------------------------------------------------------
# ③ 上次读数须活过 watch.json 的整体覆写（arm 重建 dict；CARRY_KEYS 搬运）
# ---------------------------------------------------------------------------
def test_seen_marker_survives_watch_rearm(toml, monkeypatch):
    toml(True)
    assert night.diff_message("s1") is None  # 记基线 on
    # 看门狗/下一回合的实际形态：arm() 以**新 dict** 整体覆写 watch.json
    watch.arm(timeout_s=5, heartbeat=False)
    assert "night_seen" in watch.load_state(), "arm 覆写后读数必须仍在（CARRY_KEYS）"
    toml(False)
    assert night.diff_message("s1") is None, (
        "覆写丢了账 → 会被当「首次见」而静默漏掉这次关→开/开→关"
    )


# ---------------------------------------------------------------------------
# ④ CLI 差分口（钩子调的形态）
# ---------------------------------------------------------------------------
def test_cli_diff_prints_only_on_change(toml, capsys):
    toml(True)
    args = ["--diff", "--session", "s1", "--settings", str(settings.SETTINGS_PATH)]
    assert night.main(args) == 0
    assert capsys.readouterr().out == "", "首次见：零输出"
    toml(False)
    assert night.main(args) == 0
    assert capsys.readouterr().out == "", "开→关零注入（用户裁决 2026-10-17）"
    assert night.main(args) == 0
    assert capsys.readouterr().out == "", "无变化：零输出"


def test_cli_diff_bad_settings_is_silent_exit0(tmp_path, capsys):
    assert (
        night.main(
            ["--diff", "--session", "s1", "--settings", str(tmp_path / "absent.toml")]
        )
        == 0
    )
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# ⑤ 文案修订（B 组要求逐条）
# ---------------------------------------------------------------------------
def test_policy_terminal_state_no_switch_flipping():
    t = night.NIGHT_POLICY
    assert "全部完成" in t and "严重阻碍" in t, "终态两态保留"
    assert "即结束推进" in t
    # 删的是「叫 LLM 去拨」的指令，留的是禁令本身（「永不改写」）
    for banned in ("改回 false", "改回 `false`", "将 enabled 改回", "把 enabled 改回"):
        assert banned not in t, f"授权书不得出现「LLM 拨开关」类字样：{banned}"
    assert "永不改写" in t and "开关只归用户手动" in t
    assert "不做自拨回" in t, "禁令宜写实：明确不做自拨回/不设时限（用户裁决第 3 条）"


def test_policy_has_no_dangling_section_ref_and_lists_consult_commands():
    t = night.NIGHT_POLICY
    assert "见 §3.6" not in t, (
        "悬空引用须删/写全（§3.6 是设计文档 PSEP_2 的节号，读者没有）"
    )
    assert "references/night-plan-gate.md" in t, (
        "查实出处：权限弹窗与 Plan 两环节的细则见 references/night-plan-gate.md"
    )
    assert "pysci-orch twin consult <案卷.md>" in t
    assert 'pysci-orch deputy consult "<问题>"' in t
    # ⑤ 去行话：直白说明禁什么、等待走什么
    assert "空转打回" not in t
    assert "禁止把 LLM 当定时器" in t and "事件机制" in t


def test_gate_deny_text_unchanged_semantics():
    """既有判据面零回退：deny 文案仍指 twin consult 与降级路径。"""
    deny = night.gate_deny_message({"night": {"enabled": True}})
    assert deny and "twin", "consult" in deny and "副组长" in deny
    assert night.gate_deny_message({"night": {"enabled": False}}) is None


# ---------------------------------------------------------------------------
# ⑥ 通道契约：Stop 钩子按差分输出走 stderr + exit 2（既有 rewake 路径）
# ---------------------------------------------------------------------------
def _fake_root(tmp_path: Path, *, canned: str) -> Path:
    """假项目根：两个桩可执行 + 假配置。副作用全部落在 tmp（真会话的 watch 链零接触）。

    ``python.exe`` 桩照读 ``canned.txt`` 当「night.py --diff 的输出」；``pysci-orch.exe``
    桩让钩子走完归属判定与「一轮 watch」后干净退出（``watch`` 返回 rc=1 → 钩子 log 后 exit 0）。
    """
    root = tmp_path / "root"
    scripts = root / ".venv" / "Scripts"
    (root / ".qoder" / "hooks").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    scripts.mkdir(parents=True)
    (root / "data" / "pysci_settings.toml").write_text(
        "[night]\nenabled = true\n", encoding="utf-8"
    )
    (scripts / "canned.txt").write_text(canned, encoding="utf-8")
    py = scripts / "python.exe"
    py.write_text(
        '#!/bin/bash\ncat "$(dirname "$0")/canned.txt"\nexit 0\n', encoding="utf-8"
    )
    orch = scripts / "pysci-orch.exe"
    orch.write_text(
        '#!/bin/bash\ncase "$1" in\n  watch) echo TIMEOUT; exit 1 ;;\nesac\nexit 0\n',
        encoding="utf-8",
    )
    for p in (py, orch):
        p.chmod(0o755)
    return root


def _run_hook(root: Path) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "QODER_PROJECT_DIR": str(root)}
    return subprocess.run(
        ["bash", str(HOOK)],
        input=json.dumps({"session_id": "s-chan", "hook_event_name": "Stop"}).encode(),
        capture_output=True,
        env=env,
        timeout=120,
    )


def test_hook_end_to_end_injects_diff_via_exit2(tmp_path):
    """真跑 Stop 钩子：差分非空 → **stderr 注出 + exit 2**（harness 的 rewake 语义）。"""
    proc = _run_hook(_fake_root(tmp_path, canned="CANARY-DIFF"))
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-300:])
    assert b"CANARY-DIFF" in proc.stderr


def test_hook_end_to_end_silent_when_no_diff(tmp_path):
    """对照组：同一钩子、差分空 → **exit 0 且零 stderr**（无变化零打扰，watch 链照走）。"""
    proc = _run_hook(_fake_root(tmp_path, canned=""))
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-300:])
    assert proc.stderr.strip() == b"", "无变化不得向模型注入任何东西"


def test_hook_routes_diff_through_rewake_channel():
    text = HOOK.read_text(encoding="utf-8")
    body = text[text.index("PAYLOAD=$(cat)") :]  # 从脚本主体起（头部注释另有沿革记录）
    assert "--diff --session" in body, "Stop 钩子须按会话调 night --diff"
    assert "夜航差分注入" in body
    at_diff = body.index("ndiff=$(printf")
    # 位置：SessionEnd 之后（结束中的会话不注入）、归属判定/挂载之前（未命中时对 watch 语义零影响）
    assert body.index('if [ "$HEV" = "SessionEnd" ]') < at_diff
    assert at_diff < body.index("唤醒权自动归属闸"), (
        "差分块须在归属判定/挂载之前（未命中时对 watch 语义零影响）"
    )
    # fail-open：异常路径不得 exit 2
    seg = body[at_diff : body.index("唤醒权自动归属闸")]
    assert "fail-open 继续监听" in seg


def test_repo_night_switch_docs_aligned():
    """SKILL.md 与 patrol-watchdog §5 措辞对齐：不得再写「把 enabled 改回 false」。"""
    skill = (
        PROJECT_ROOT / "orchestration" / "skills" / "orchestration" / "SKILL.md"
    ).read_text(encoding="utf-8")
    ref = (
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "orchestration"
        / "references"
        / "patrol-watchdog.md"
    ).read_text(encoding="utf-8")
    for name, text in (("SKILL.md", skill), ("patrol-watchdog.md", ref)):
        flat = "".join(text.split())  # 折行无关
        assert "将enabled改回" not in flat and "把enabled改回" not in flat, (
            f"{name} 仍写着 LLM 拨开关"
        )
        assert "开关只归用户手动" in flat, name
        assert "下一轮" in flat, f"{name} 须写明『下一轮生效』"
