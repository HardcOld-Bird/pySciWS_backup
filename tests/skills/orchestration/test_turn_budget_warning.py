"""会话余量告警（F1，tb-20261017-212141-3-3 L.l3）与禁回退清单模板钉子（F2）。

F1：生产单 60 轮耗尽即 ``fatal_error``、零交付——跳结束后才看 ``num_turns`` 已来不及。
实现走 CLI ``PostToolUse`` hook（每工具回合执行一次，exit 2 的 stderr 注入会话），
到达本跳上限的 80% 时提示「交付半成品＋接手清单」。测试覆盖：

- argv 注入（``--settings`` 内联 JSON，含本跳 budget 与 since）；
- 回合计数口径（``message.id`` 去重、``--since`` 排除 resume 历史）；
- **满轮模拟**：48/60 可见告警；47/60 静默；同跳只告警一次；
- 压测跳（eval_hop）不注入（不改变被测行为/不叠加 hook 耗时到基线）。

F2：模板含「禁回退清单组装指引」段与「对照源清单」产出位，并把细节分册登记在索引。
"""

from __future__ import annotations

import io
import json

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import dispatch, runner, turnbudget
from pysci.skills.orchestration.tools.registry import Member

from .conftest import envelope, fake_headless

OK = "<result>完成，commit aaa1111。</result>"
TEMPLATE = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "taskbook-template.md"
)
GUIDE = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "taskbook-template-02-rework-fallback-list.md"
)


def _member(tmp_path) -> Member:
    return Member(
        member_id="m",
        pod=tmp_path / "pod",
        max_turns=5,
        timeout_s=60,
        mcp_config=None,
    )


def _settings_of(cmd: list[str]) -> dict:
    assert "--settings" in cmd, "build_command 未注入余量告警 settings"
    return json.loads(cmd[cmd.index("--settings") + 1])


# ---------------------------------------------------------------------------
# F1-a：build_command 注入（budget/since/hook 形状）
# ---------------------------------------------------------------------------
def test_build_command_injects_turn_budget_hook(tmp_path):
    cmd = runner.build_command(
        _member(tmp_path),
        "p",
        session_id="s",
        resume=False,
        model="",
        exe=tmp_path / "exe",
    )
    settings = _settings_of(cmd)
    hooks = settings["hooks"]["PostToolUse"][0]["hooks"]
    assert hooks[0]["name"] == "turn-budget"
    command = hooks[0]["command"]
    assert "turnbudget.py" in command
    assert "--budget 5" in command  # 成员缺省轮次上限
    assert "--since " in command, "必须按跳起算，resume 历史不能算进本跳预算"
    assert "PostToolUse" in settings["hooks"]


def test_build_command_budget_follows_override(tmp_path):
    cmd = runner.build_command(
        _member(tmp_path),
        "p",
        session_id="s",
        resume=True,
        model="",
        max_turns=60,
        exe=tmp_path / "exe",
    )
    assert (
        "--budget 60"
        in _settings_of(cmd)["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
    )


def test_build_command_turn_warning_can_be_disabled(tmp_path):
    cmd = runner.build_command(
        _member(tmp_path),
        "p",
        session_id="s",
        resume=False,
        model="",
        exe=tmp_path / "exe",
        turn_warning=False,
    )
    assert "--settings" not in cmd


# ---------------------------------------------------------------------------
# F1-b：阈值与回合计数口径
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "budget,ratio,want",
    [(60, None, 48), (5, None, 4), (60, 0.5, 30), (1, 0.8, 1), (10, 1.0, 10)],
)
def test_warning_threshold(budget, ratio, want):
    assert turnbudget.warning_threshold(budget, ratio) == want


def test_invalid_ratio_falls_back(monkeypatch):
    monkeypatch.setenv("PYSCI_TURN_WARN_RATIO", "1.5")
    assert turnbudget.warn_ratio() == turnbudget.WARN_RATIO
    monkeypatch.setenv("PYSCI_TURN_WARN_RATIO", "0.5")
    assert turnbudget.warn_ratio() == 0.5


def _turn(idx: int, *, stamp: str, block: str = "tool_use") -> dict:
    """合成一条 assistant transcript 记录（同一回合的 thinking/tool_use 共用 message.id）。"""
    return {
        "type": "assistant",
        "timestamp": stamp,
        "message": {"id": f"msg-{idx}", "content": [{"type": block}]},
    }


def test_count_turns_dedupes_blocks_and_respects_since(tmp_path):
    path = tmp_path / "s.jsonl"
    records = []
    # 历史跳：3 个回合（since 之前，必须排除）
    for i in range(3):
        records.append(_turn(i, stamp="2026-10-18T01:00:00.000Z", block="thinking"))
        records.append(_turn(i, stamp="2026-10-18T01:00:00.000Z"))
    # 本跳：2 个回合，每回合 thinking+tool_use 两行（按 message.id 去重）
    for i in (10, 11):
        records.append(_turn(i, stamp="2026-10-18T03:00:00.000Z", block="thinking"))
        records.append(_turn(i, stamp="2026-10-18T03:00:00.000Z"))
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    since = turnbudget._parse_timestamp("2026-10-18T02:00:00.000Z")
    assert turnbudget.count_turns(path, since) == 2
    assert turnbudget.count_turns(path, None) == 5


def test_count_turns_ignores_subagent_sidechain(tmp_path):
    """子代理回合不属于主循环 num_turns：混进来会让长任务提前误报。"""
    path = tmp_path / "side.jsonl"
    main = _turn(1, stamp="2026-10-18T03:00:00.000Z")
    side = {**_turn(2, stamp="2026-10-18T03:00:00.000Z"), "isSidechain": True}
    path.write_text(
        json.dumps(main, ensure_ascii=False)
        + "\n"
        + json.dumps(side, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert turnbudget.count_turns(path) == 1


# ---------------------------------------------------------------------------
# F1-c：满轮模拟——告警注入可见、静默区、同跳只告警一次
# ---------------------------------------------------------------------------
def _run_hook(tmp_path, monkeypatch, capsys, *, turns: int, budget: int, sid: str):
    # 一次性标记落 tmp：系统临时目录跨 pytest 进程不清理，同一 sid 第二次跑就被吃掉
    markers = tmp_path / "markers"
    monkeypatch.setattr(
        turnbudget,
        "_marker_path",
        lambda session_id, since: markers / f"{session_id}-{(since or 0.0):.3f}.warned",
    )
    transcript = tmp_path / f"{sid}.jsonl"
    records = [
        _turn(i, stamp="2026-10-18T03:00:00.000Z", block="thinking")
        for i in range(turns)
    ]
    transcript.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"session_id": sid, "transcript_path": str(transcript)})
        ),
    )
    rc = turnbudget.main(["--budget", str(budget), "--since", "0"])
    return rc, capsys.readouterr().err


def test_full_turn_simulation_warns_at_80_percent(tmp_path, monkeypatch, capsys):
    sid = "warn-48-of-60"
    rc, err = _run_hook(tmp_path, monkeypatch, capsys, turns=48, budget=60, sid=sid)
    assert rc == 2, "达到 80% 必须以 exit 2 把告警注入会话"
    assert "余量告警" in err and "轮次即将耗尽" in err
    assert "48/60" in err and "阈值 48" in err
    assert "交付半成品＋接手清单" in err and "产物路径/已完成/待办" in err
    assert "勿硬扛" in err


def test_below_threshold_is_silent(tmp_path, monkeypatch, capsys):
    rc, err = _run_hook(tmp_path, monkeypatch, capsys, turns=47, budget=60, sid="ok-47")
    assert rc == 0 and err == ""


def test_warning_is_one_shot_per_hop(tmp_path, monkeypatch, capsys):
    sid = "once-48"
    rc1, err1 = _run_hook(tmp_path, monkeypatch, capsys, turns=48, budget=60, sid=sid)
    assert rc1 == 2 and "余量告警" in err1
    # 同一跳的下一工具回合：不再退回同一句（否则 80% 之后每回合都烧一遍提示）
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": sid,
                    "transcript_path": str(tmp_path / f"{sid}.jsonl"),
                }
            )
        ),
    )
    rc2 = turnbudget.main(["--budget", "60", "--since", "0"])
    assert rc2 == 0
    assert capsys.readouterr().err == ""


def test_missing_transcript_fails_open(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert turnbudget.main(["--budget", "60"]) == 0
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "ghost",
                    "transcript_path": str(tmp_path / "nope.jsonl"),
                }
            )
        ),
    )
    assert turnbudget.main(["--budget", "60"]) == 0


# ---------------------------------------------------------------------------
# F1-d：派发链集成——生产跳注入、压测跳不注入
# ---------------------------------------------------------------------------
def _settings_present(cmd: list[str]) -> bool:
    return "--settings" in cmd


def test_dispatch_injects_turn_warning_for_production(iso_dispatch, monkeypatch):
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    assert o.code == 0
    assert _settings_present(calls[0])
    assert "--budget 3" in calls[0][calls[0].index("--settings") + 1]


def test_dispatch_eval_hop_skips_turn_warning(iso_dispatch, monkeypatch):
    calls = fake_headless(
        monkeypatch, lambda n, cmd: envelope(OK, is_error=False, stop="end_turn", rc=0)
    )
    o = dispatch.do_dispatch(
        "quotamember", text="t", quiet=True, no_checks=True, eval_hop=True
    )
    assert o.code == 0
    assert not _settings_present(calls[0]), "压测跳不注入：被测行为与基线读数须可比"


# ---------------------------------------------------------------------------
# F2：模板含组装指引段 + 对照源清单产出位（lint 钉子）
# ---------------------------------------------------------------------------
def test_template_has_fallback_list_guide():
    md = TEMPLATE.read_text(encoding="utf-8")
    assert "## 禁回退清单（返工第 ≥2 轮必填" in md, "模板缺禁回退清单段"
    assert "对照源清单（强制产出位）" in md, "缺对照源清单产出位"
    assert "### 禁回退清单组装指引（返工面）" in md, "缺组装指引段"
    assert "历轮交付产物**既定做法**" in md and "old prompts/原始材料惯例" in md
    assert "标注/角标/脚注/配色" in md, "细节惯例类别须显式点名"
    assert "Exp.†" in md, "须点名本轮教训（角标丢失实案）"
    assert "taskbook-template-02-rework-fallback-list.md" in md, "细则分册未登记"


def test_fallback_list_detailed_guide_covers_sources():
    md = GUIDE.read_text(encoding="utf-8")
    for token in (
        "历轮交付产物",
        "old prompts / 原始材料惯例",
        "标注 / 角标 / 脚注 / 配色",
        "对照源清单（强制产出位）",
        "缺对照源清单＝组装不完整",
    ):
        assert token in md, f"组装细则缺要素：{token}"


def test_fallback_docs_stay_within_harness_budget():
    for path in (TEMPLATE, GUIDE):
        size = path.stat().st_size
        assert size <= 8192, f"{path.name} 超 harness 预算：{size}B"
