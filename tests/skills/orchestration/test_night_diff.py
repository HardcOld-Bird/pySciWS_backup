"""夜航**热生效**（adhoc-night-text，用户裁决 2026-10-15）：回合边界差分注入 + 文案修订。

钉五件事：
① 差分只在**跳变**那一轮产出（无变化零输出零写入；首次见会话只记基线不注入）；
② 关→开 = 授权书全文（同 SessionStart 文本）、开→关 = 刹车差分；
③ 上次读数**按 session_id 各存一份**（多会话并存不互相误报），且跨 watch.json 整体覆写；
④ 文案修订（④ 删「改回 false」、⑤ 去行话、§3.6 不再悬空、补 consult 调用面；关闭差分
   不得出现「立即停止自主推进」类字样）；
⑤ 通道契约：Stop 钩子（watch-arm.sh）按差分输出走 stderr + exit 2 的既有 rewake 路径。
"""

from __future__ import annotations

import json
import os
import subprocess
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


def test_on_to_off_injects_brake_diff_without_stop_autonomy_wording(toml):
    toml(True)
    night.diff_message("s1")
    toml(False)  # 开→关
    msg = night.diff_message("s1")
    assert msg == night.NIGHT_DIFF_OFF
    # 用户裁决措辞禁令：关闭**不得**写「立即停止自主推进」类字样
    assert "自主推进照常" in msg
    for banned in ("立即停止自主推进", "停止自主推进", "暂停自主推进", "不得自主推进"):
        assert banned not in msg, f"关闭差分出现禁令字样：{banned}"
    # 必含要素：授权书失效 / 拦截放行 / 恢复一问一答
    assert "授权书失效" in msg and "AskUserQuestion" in msg and "一问一答" in msg


# ---------------------------------------------------------------------------
# ② 多会话并存：上次读数按 session_id 各存一份（防会话间互相误报）
# ---------------------------------------------------------------------------
def test_last_seen_is_per_session(toml):
    toml(True)
    night.diff_message("s1")
    night.diff_message("s2")  # 两个会话各自记基线
    toml(False)
    assert night.diff_message("s1") == night.NIGHT_DIFF_OFF
    # s2 尚未比对过任何跳变 → 它也是关，但它的基线是 on → 同样出声（各算各的账）
    assert night.diff_message("s2") == night.NIGHT_DIFF_OFF
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
    assert night.diff_message("s1") == night.NIGHT_DIFF_OFF, (
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
    assert capsys.readouterr().out.strip() == night.NIGHT_DIFF_OFF
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
    assert "PSEP_2 §3.6" in t, (
        "查实出处：写全为 PSEP_2 §3.6（权限弹窗人工 checklist、不写钩子）"
    )
    assert "pysci-orch consult-twin <案卷.md>" in t
    assert 'pysci-orch consult "<问题>"' in t
    # ⑤ 去行话：直白说明禁什么、等待走什么
    assert "空转打回" not in t
    assert "禁止把 LLM 当定时器" in t and "事件机制" in t


def test_gate_deny_text_unchanged_semantics():
    """既有判据面零回退：deny 文案仍指 consult-twin 与降级路径。"""
    deny = night.gate_deny_message({"night": {"enabled": True}})
    assert deny and "consult-twin" in deny and "副组长" in deny
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
    proc = _run_hook(_fake_root(tmp_path, canned=night.NIGHT_DIFF_OFF))
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-300:])
    assert night.NIGHT_DIFF_OFF.encode("utf-8") in proc.stderr


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
