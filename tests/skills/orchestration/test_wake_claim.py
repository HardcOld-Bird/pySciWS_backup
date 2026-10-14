"""唤醒权认领制（wake-claim，用户设计 2026-10-14）：主会话定向唤醒。

事故现场：watch 钩子的 token 竞争是「最新在听者胜」（v2 修会话重启丢唤醒的规则）——
用户主会话工作时另开的**次要会话**一有回合就抢走唤醒权，drain/事件的 exit-2 被投到
次要会话（污染其上下文，正是用户要避免的）；凌晨「被取代却无接任」的谜之日志疑为同源。

本文件钉四类语义（任务书机械验收项）：
① owner 钩子持权（同身份 → 可监听）；② 非 owner 不抢（异身份 → 3，钩子不监听不唤醒）；
③ 认领迁移（新会话接管，旧 owner 即刻失效；release 退回 legacy）；
④ owner 缺席事件不丢（持久水位 + 迟到唤醒，a1 语义续接——不存在「转投其他会话」）。
"""

from __future__ import annotations

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import events, orch, watch

HOOK = PROJECT_ROOT / ".qoder" / "hooks" / "watch-arm.sh"


@pytest.fixture
def claim_face(tmp_path, monkeypatch):
    """唤醒权真源落到 tmp（真源是运行态，测试不许写仓库 state）。"""
    path = tmp_path / "wake-claim.json"
    monkeypatch.setattr(watch, "WAKE_CLAIM_PATH", path)
    return path


# ---------------------------------------------------------------------------
# ① owner 钩子持权 / ② 非 owner 不抢
# ---------------------------------------------------------------------------
def test_owner_hook_holds_the_claim(claim_face, monkeypatch):
    monkeypatch.setenv("WT_SESSION", "guid-main")
    code, lines = watch.wake_claim("claim", name="main")
    assert code == 0 and any("strict" in ln for ln in lines)

    assert watch.wake_claim_check() == 0, "同会话（身份一致）= owner，可监听可唤醒"
    assert claim_face.exists()
    claim = watch.load_claim()
    assert claim["owner"] == "main" and claim["strict"] is True
    assert claim["identity"] == {"WT_SESSION": "guid-main"}


def test_non_owner_hook_is_not_owner(claim_face, monkeypatch):
    monkeypatch.setenv("WT_SESSION", "guid-main")
    watch.wake_claim("claim", name="main")

    monkeypatch.setenv("WT_SESSION", "guid-secondary")  # 次要会话
    assert watch.wake_claim_check() == 3, "异身份会话必须被判为非 owner（钩子据此不抢）"


def test_hook_script_gates_non_owner_and_fails_open():
    """钩子面的机械判据：非 owner 分支 exit 0（不监听不抢 token，**绝不 exit 2**），
    检查本身失败走 fail-open，且留一次性会话身份取证。"""
    text = HOOK.read_text(encoding="utf-8")
    assert "wake-claim --check" in text, "钩子每回合必须自检唤醒权"
    assert "非 owner 会话 → 不监听不抢" in text
    assert 'case "$wcr" in' in text and "3)" in text
    assert "fail-open 按 legacy 监听" in text, "检查失败不得让唤醒链塌掉"
    assert "session-env.seen" in text, "一次性会话身份取证必须留痕"
    # 非 owner 分支的唯一出口是 exit 0（不存在 exit 2）
    branch = text.split('case "$wcr" in', 1)[1].split("esac", 1)[0]
    assert "3)" in branch and "exit 0" in branch
    assert "exit 2" not in branch


# ---------------------------------------------------------------------------
# ③ 认领迁移 / release 退回 legacy
# ---------------------------------------------------------------------------
def test_claim_migrates_to_new_session(claim_face, monkeypatch):
    monkeypatch.setenv("WT_SESSION", "guid-main")
    watch.wake_claim("claim", name="main")
    assert watch.wake_claim_check() == 0

    # 新会话接管（owner 永久迁移）
    monkeypatch.setenv("WT_SESSION", "guid-new")
    code, lines = watch.wake_claim("claim", name="new")
    assert code == 0
    assert watch.wake_claim_check() == 0, "新 owner 立即可监听"
    assert watch.load_claim()["owner"] == "new"

    # 旧会话即刻失效
    monkeypatch.setenv("WT_SESSION", "guid-main")
    assert watch.wake_claim_check() == 3, "被接管后旧 owner 降为非 owner（不许再唤醒）"


def test_release_returns_to_legacy(claim_face, monkeypatch):
    monkeypatch.setenv("WT_SESSION", "guid-main")
    watch.wake_claim("claim", name="main")
    assert watch.wake_claim_check() == 0

    code, _ = watch.wake_claim("release")
    assert code == 0 and not claim_face.exists()
    monkeypatch.setenv("WT_SESSION", "guid-secondary")
    assert watch.wake_claim_check() == 0, "退回 legacy 后任何钩子都可监听（旧行为）"


def test_claim_without_session_env_falls_back_to_legacy(claim_face, monkeypatch):
    """无会话区分变量 → legacy（documented limitation）：宁可退化，也不能静默禁掉 owner。"""
    monkeypatch.delenv("WT_SESSION", raising=False)
    for key in watch.SESSION_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    _, lines = watch.wake_claim("claim", name="main")
    assert any("legacy 模式" in ln for ln in lines)
    assert watch.load_claim()["strict"] is False
    assert watch.wake_claim_check() == 0
    monkeypatch.setenv("WT_SESSION", "whatever")
    assert watch.wake_claim_check() == 0, (
        "legacy 认领下不区分会话（局限已写进输出与文档）"
    )


def test_cli_smoke_status_release(claim_face, monkeypatch, capsys):
    monkeypatch.setenv("WT_SESSION", "guid-main")
    assert orch.main(["wake-claim", "--name", "main"]) == 0
    out = capsys.readouterr().out
    assert "唤醒权已认领" in out and "strict" in out

    assert orch.main(["wake-claim", "status"]) == 0
    out = capsys.readouterr().out
    assert "唤醒权 = main" in out and "strict" in out and "迁移" in out

    assert orch.main(["wake-claim", "--check"]) == 0
    assert orch.main(["wake-claim", "release"]) == 0
    assert orch.main(["wake-claim", "status"]) == 0
    assert "未认领（legacy" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ④ owner 缺席事件不丢：迟到唤醒在 owner 下次挂载时接续
# ---------------------------------------------------------------------------
def test_owner_absent_events_are_woken_on_owner_mount(
    claim_face, monkeypatch, iso_state
):
    monkeypatch.setenv("WT_SESSION", "guid-main")
    watch.wake_claim("claim", name="main")
    # 次要会话在场，但**非 owner 不许唤醒**（事件照常进流）
    monkeypatch.setenv("WT_SESSION", "guid-secondary")
    assert watch.wake_claim_check() == 3

    watch.arm()  # 无监听时的基线（首任）
    events.emit(
        events.QUEUE_EMPTY, "run-010000", "devops 队列消化完：处理 1 项", "drain"
    )

    # owner 下次挂载 → 继承持久水位 → 立即补唤醒（a1 语义，不因「转投」而丢）
    st = watch.arm()
    assert st["catchup"] is True
    evs, _verdict = watch.detect(st["base"], watch.snapshot())
    assert [e["event"] for e in evs] == [events.QUEUE_EMPTY], (
        "owner 挂载必须补上聋期事件"
    )
    watch._mark_late(evs, st)
    assert evs[0].get("late") is True and "迟到" in evs[0]["detail"]

    # 真唤醒（_emit 落盘 fired+消费账）并推进基线后重挂 → 水位单调，不重复消费
    echoed: list[str] = []
    assert watch._emit(st, evs, echoed.append) is True, "owner 侧（或补账）送达唤醒"
    assert any("迟到" in ln for ln in echoed)
    cur = watch.snapshot()
    st["base"] = watch.advance_base(st["base"], cur, set())
    watch.save_state(st)
    st2 = watch.arm()
    assert watch.detect(st2["base"], watch.snapshot())[0] == []
