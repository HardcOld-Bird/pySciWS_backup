"""E2 harness 压测套件（minimum 档 + 案卷判据 + orch eval run + 基线对照）。

README §5.5/§3.5，backlog 20261011-233002-leader。这里测的是**测量仪本身**：
判据写错必须当场失败（不许静默少测一项）、二元判决不许平局、压测行不进成员台账、
压测跳不吃审批回复（与生产跳隔离）。真派发不外呼——``run_headless`` 假化。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import dispatch, presets, registry
from pysci.skills.orchestration.tools import eval as ev
from pysci.skills.orchestration.tools import ledger as ledgermod
from pysci.skills.orchestration.tools.delivery import Artifact
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.runner import Envelope

from .conftest import cmd_model, envelope, fake_headless, set_models

BODY = "## 任务书\n\n画一张单面板示意图，产物落在白名单目录内。\n"


@pytest.fixture
def iso_eval(tmp_path, monkeypatch, iso_dispatch):
    """案卷目录/压测台账/基线三面搬到 tmp；成员与 registry 沿用 iso_dispatch 的 quotamember。

    ``PROJECT_ROOT`` 也搬到 tmp：产物存在性与「断言于」文件面都以 tmp 为根，测试不落真树。
    """
    harness = tmp_path / "harness"
    harness.mkdir()
    monkeypatch.setattr(ev, "HARNESS_DIR", harness)
    monkeypatch.setattr(ev, "EVAL_LEDGER_PATH", tmp_path / "evals.jsonl")
    monkeypatch.setattr(ev, "BASELINE_PATH", tmp_path / "baseline.json")
    monkeypatch.setattr(ev, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ev, "run_checks", _stub_checks)
    # 脏面快照默认「跳前跳后都干净」：真 dirty_tracked_state 会起 git 看当前工作树，
    # 那会让每个 run_case 测试随仓库状态漂移。要测差集的用例再用 fake_dirt 覆盖。
    monkeypatch.setattr(ev, "dirty_tracked_state", lambda *, path="", timeout_s=10: [])
    return tmp_path


def _stub_checks(artifacts, reg, *, member_pod):
    """假验收：verdict 取自案卷外的 dict（键=产物路径），缺省 pass。

    真 run_checks 起 subprocess，把「判据读数如何变成判决」这层逻辑的测试变成
    环境测试——本模块只测裁决面，subprocess 语义由 checks 自己的测试覆盖。
    """
    table = getattr(reg, "stub_verdicts", {})
    return [
        {
            "path": a.path,
            "check": a.check,
            "verdict": table.get(a.path, "pass"),
            "output_tail": "",
        }
        for a in artifacts
    ]


class FakeReg:
    """最小注册表面：只给 eval 用到的三样（models / member().pod / stub 判据读数）。"""

    def __init__(self, pod: Path, verdicts: dict[str, str] | None = None):
        self.pod = pod
        self.models = {"minimum": "Qwen3.8-Flash", "flash": "Qwen3.8-Flash", "max": ""}
        self.stub_verdicts = verdicts or {}

    def member(self, member_id: str):
        return type("M", (), {"pod": self.pod})()


def stub_reg(pod: Path, verdicts: dict[str, str] | None = None):
    return FakeReg(pod, verdicts)


def seed_case(tmp_path: Path, text: str, case_id: str = "demo") -> Path:
    p = tmp_path / "harness" / f"{case_id}.md"
    p.write_text(text, encoding="utf-8")
    return p


def out_of(
    kind: str = "result", body: str = "已完成", *, tags=None, denials=0, turns: int = 4
) -> DispatchOutcome:
    env = Envelope.parse(
        json.dumps(
            {
                "result": body,
                "is_error": False,
                "stop_reason": "stop_sequence",
                "num_turns": turns,
                "duration_ms": 12000,
                "session_id": "aaaa1111",
                "total_credits": 0.0,
                "usage": {"context_usage_ratio": 0.3},
                "permission_denials": [{"tool": "Write"}] * denials,
            }
        )
    )
    return DispatchOutcome(
        code=0,
        kind=kind,
        member="quotamember",
        sid="aaaa1111",
        session_name="eval-demo",
        task_file="orchestration/pods/x/inbox/t.md",
        est_tokens=60,
        delta_chars=100,
        body=body,
        tags=tags or [],
        envelope=env,
    )


def pod(iso_eval) -> Path:
    """iso_dispatch 夹具里的成员 pod（quotamember）——案卷产物的第二解析基准。"""
    return Path(iso_eval) / "pods" / "quotamember"


# ---------------------------------------------------------------------------
# minimum 档（§1.1 护栏级：档位存在且指向零成本渠道）
# ---------------------------------------------------------------------------
def test_minimum_tier_is_free_channel():
    assert registry.DEFAULT_MODELS["minimum"] == "Qwen3.8-Flash"
    assert registry.DEFAULT_MODEL_CHAINS["minimum"] == (
        "Mimo-V2.6-Flash",
        "Qwen3.8-Flash",
    ), "压测最低成本链：Mimo-Flash 首席、内置兜底，**BYOK Qwen 不进链**（额度宝贵）"


def test_orch_cli_accepts_minimum_tier(iso_state, iso_dispatch, monkeypatch):
    """三处 --model-tier choices 都要认 minimum：argparse 是档位的唯一校验点
    （do_dispatch 自己不拒未知档），漏改一处就等于该面无法显式压测。

    两面都挂：iso_dispatch 给成员/registry，iso_state 挡 dispatch 入口内嵌的巡检拉起
    （只挂前者时本用例真起过 ``_inspect-worker``）。
    """
    from pysci.skills.orchestration.tools import orch, plans

    seen = {}

    def fake_dispatch(member_id, **kw):
        seen["member"] = member_id
        seen.update(kw)
        return out_of()

    monkeypatch.setattr(orch, "do_dispatch", fake_dispatch)
    assert (
        orch.main(["dispatch", "quotamember", "--text", "x", "--model-tier", "minimum"])
        == 0
    )
    assert seen["model_tier"] == "minimum"

    got = {}

    def fake_adhoc(args):
        got["tier"] = args.model_tier
        return 0

    monkeypatch.setattr(plans, "cmd_plan_adhoc", fake_adhoc)
    monkeypatch.setattr(orch._plans, "cmd_plan_adhoc", fake_adhoc)
    assert (
        orch.main(
            ["plan", "adhoc", "quotamember", "--text", "x", "--model-tier", "minimum"]
        )
        == 0
    )
    assert got["tier"] == "minimum"

    with pytest.raises(SystemExit) as exc:
        orch.main(["dispatch", "quotamember", "--text", "x", "--model-tier", "bogus"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# 案卷解析：判据写错必须当场失败
# ---------------------------------------------------------------------------
def test_parse_case_minimal(iso_eval):
    case = ev.parse_case(seed_case(iso_eval, "member: figure\n" + BODY))
    assert case.case_id == "demo" and case.member == "figure"
    assert case.tier == ev.DEFAULT_TIER
    assert case.max_turns == ev.DEFAULT_MAX_TURNS
    assert case.timeout_s == ev.DEFAULT_TIMEOUT_S
    assert "画一张" in case.brief and "## 任务书" not in case.brief
    assert case.artifacts == [] and case.expect_re == ""


def test_parse_case_all_keys(iso_eval):
    p = seed_case(
        iso_eval,
        "member: figure\ntier: flash\ndirs: a; b, c\nmax_turns: 9\ntimeout_s: 300\n"
        "产物: one check=figure-audit\n产物: two\n断言: 2310\\.16699\n断言于: out/r.md\n"
        "禁止标签: whitelist_conflict,quota\n权限拒绝上限: 2\n说明: 备注\n" + BODY,
    )
    case = ev.parse_case(p)
    assert case.tier == "flash"
    assert case.dirs == ["a", "b", "c"]
    assert case.max_turns == 9 and case.timeout_s == 300
    assert case.artifacts == [
        Artifact(path="one", check="figure-audit"),
        Artifact(path="two", check="none"),
    ]
    assert case.expect_re == r"2310\.16699" and case.expect_on == "out/r.md"
    assert case.forbid_tags == ["whitelist_conflict", "quota"]
    assert case.max_denials == 2 and case.note == "备注"


@pytest.mark.parametrize(
    "text,frag",
    [
        ("tier: minimum\n" + BODY, "缺 member"),
        ("member: figure\nbogus: 1\n" + BODY, "未知头部键"),
        ("member: figure\nmember: lit\n" + BODY, "重复"),
        ("member: figure\nmax_turns: many\n" + BODY, "不是整数"),
        ("member: figure\ntimeout_s: -1\n" + BODY, "不得为负"),
        ("member: figure\n产物: x check=a extra\n" + BODY, "不可解析"),
        ("member: figure\n断言: (unclosed\n" + BODY, "合法正则"),
        ("member: figure\n\n" + "## 任务书\n", "正文为空"),
        ("member: figure\n纯散文没有标记\n", "## 任务书"),
    ],
)
def test_parse_case_rejects(iso_eval, text, frag):
    p = seed_case(iso_eval, text)
    with pytest.raises(ValueError) as exc:
        ev.parse_case(p)
    assert frag in str(exc.value)


def test_load_cases_sorted_and_filtered(iso_eval):
    seed_case(iso_eval, "member: figure\n" + BODY, "b-second")
    seed_case(iso_eval, "member: figure\n" + BODY, "a-first")
    assert [c.case_id for c in ev.load_cases()] == ["a-first", "b-second"]
    assert [c.case_id for c in ev.load_cases("a-first")] == ["a-first"]
    with pytest.raises(ValueError):
        ev.load_cases("nope")


def test_repo_shipped_cases_are_loadable():
    """随仓三案必须真能解析——案卷坏了=套件静默失能（同 taskbook-lint 的 CI 钉法）。"""
    src = Path(ev.__file__).resolve().parents[4]  # tools→orchestration→skills→pysci→src
    harness = src.parent / "orchestration" / "state" / "evals" / "harness"
    assert harness.is_dir(), f"三案目录缺席：{harness}"
    got = sorted(ev.parse_case(f).case_id for f in harness.glob("*.md"))
    assert got == ["blocked-recheck", "fig-basic", "lit-arxiv"]
    figs = ev.parse_case(harness / "fig-basic.md")
    assert figs.member == "figure" and figs.tier == "minimum"
    assert figs.artifacts[0].check == "figure-audit"
    lit = ev.parse_case(harness / "lit-arxiv.md")
    assert lit.member == "lit" and lit.expect_on and lit.expect_re
    assert "whitelist_conflict" in lit.forbid_tags
    blocked = ev.parse_case(harness / "blocked-recheck.md")
    assert blocked.forbid_tags == ["whitelist_conflict"]
    assert blocked.max_denials == 0


# ---------------------------------------------------------------------------
# judge：判据读数 → 未满足项
# ---------------------------------------------------------------------------
def judge(iso_eval, text, o, verdicts=None):
    case = ev.parse_case(seed_case(iso_eval, text))
    return ev.judge(o, case, stub_reg(pod(iso_eval), verdicts), pod(iso_eval))


def test_judge_pass(iso_eval):
    (iso_eval / "out" / "fig").mkdir(parents=True)
    reasons, checks = judge(
        iso_eval, "member: figure\n产物: out/fig check=figure-audit\n" + BODY, out_of()
    )
    assert reasons == [] and checks[0]["verdict"] == "pass"


def test_judge_blocked_is_failure(iso_eval):
    reasons, _ = judge(iso_eval, "member: figure\n" + BODY, out_of(kind="blocked"))
    assert any("blocked" in r for r in reasons)


def test_judge_missing_artifact(iso_eval):
    reasons, _ = judge(
        iso_eval, "member: figure\n产物: out/fig check=figure-audit\n" + BODY, out_of()
    )
    assert any("产物缺失" in r for r in reasons)


def test_judge_check_fail_and_unrouted(iso_eval):
    (iso_eval / "out" / "fig").mkdir(parents=True)
    txt = "member: figure\n产物: out/fig check=figure-audit\n" + BODY
    reasons, checks = judge(iso_eval, txt, out_of(), {"out/fig": "fail"})
    assert checks[0]["verdict"] == "fail"
    assert any("机械判据 FAIL" in r for r in reasons)

    reasons, checks = judge(iso_eval, txt, out_of(), {"out/fig": "unrouted"})
    assert any("判据未注册" in r for r in reasons)


def test_judge_skipped_check_still_requires_file(iso_eval):
    """check=none → skipped：仍要求产物存在（「无机械检查」不等于「可以没有产物」）。"""
    reasons, checks = judge(
        iso_eval, "member: figure\n产物: out/n.md\n" + BODY, out_of()
    )
    assert checks[0]["verdict"] == "pass"  # stub 默认；真 run_checks 这里是 skipped
    assert any("产物缺失" in r for r in reasons)


def test_judge_forbidden_tag_and_denial_cap(iso_eval):
    txt = "member: figure\n禁止标签: whitelist_conflict\n权限拒绝上限: 1\n" + BODY
    reasons, _ = judge(
        iso_eval,
        txt,
        out_of(tags=["whitelist_conflict"], denials=3),
    )
    assert any("禁止标签" in r and "whitelist_conflict" in r for r in reasons)
    assert any("权限拒绝 3 次 > 上限 1" in r for r in reasons)


def test_judge_expect_on_body_by_default(iso_eval):
    txt = "member: figure\n断言: commit [0-9a-f]{7}\n" + BODY
    assert judge(iso_eval, txt, out_of(body="已合并 commit abc1234"))[0] == []
    reasons, _ = judge(iso_eval, txt, out_of(body="已合并。"))
    assert any("交付正文不含断言" in r for r in reasons)


def test_judge_expect_on_file_not_words(iso_eval):
    """断言面=文件：嘴上说了不算，必须真写进产物文件（防「正文声称、文件没有」）。"""
    txt = "member: figure\n断言: arxiv_id.*2310\\.16699\n断言于: out/r.md\n" + BODY
    o = out_of(body="命中 2310.16699 了")
    reasons, _ = judge(iso_eval, txt, o)
    assert any("断言面文件不存在" in r for r in reasons)

    (iso_eval / "out").mkdir(parents=True)
    (iso_eval / "out" / "r.md").write_text(
        '标题\n- arxiv_id: "2310.16699"\n', encoding="utf-8"
    )
    assert judge(iso_eval, txt, o)[0] == []

    (iso_eval / "out" / "r.md").write_text("没有 ID\n", encoding="utf-8")
    reasons, _ = judge(iso_eval, txt, o)
    assert any("out/r.md不含断言" in r for r in reasons)


# ---------------------------------------------------------------------------
# run_case：派发参数面 + 台账行
# ---------------------------------------------------------------------------
def capture_dispatch(monkeypatch, ret=None):
    seen = {}

    def fake(member_id, **kw):
        seen["member"] = member_id
        seen.update(kw)
        return ret or out_of()

    monkeypatch.setattr(ev, "do_dispatch", fake)
    return seen


def patch_reg(monkeypatch, iso_eval, verdicts=None):
    monkeypatch.setattr(
        ev,
        "Registry",
        type(
            "RG", (), {"load": staticmethod(lambda: stub_reg(pod(iso_eval), verdicts))}
        ),
    )


def test_run_case_isolates_from_production(iso_eval, monkeypatch):
    seen = capture_dispatch(monkeypatch)
    patch_reg(monkeypatch, iso_eval)
    case = ev.parse_case(
        seed_case(iso_eval, "member: figure\n产物: bench/x.md\n" + BODY)
    )
    (iso_eval / "bench").mkdir()
    (iso_eval / "bench" / "x.md").write_text("x", encoding="utf-8")
    row = ev.run_case(case, label="before")
    assert seen["session"] == "new", (
        "压测必须全新会话（旧会话带上下文=测的不是模型档位）"
    )
    assert seen["attach_replies"] is False, "压测跳不得吃掉组长的审批回复"
    assert seen["no_checks"] is True, "机械判据由案卷拥有，不由派发侧代跑"
    assert seen["model_tier"] == "minimum" and seen["slug"] == "eval-demo"
    assert seen["quiet"] is True
    assert row.case == "demo" and row.label == "before" and row.verdict == "pass"


def test_run_case_cli_tier_override(iso_eval, monkeypatch):
    seen = capture_dispatch(monkeypatch)
    patch_reg(monkeypatch, iso_eval)
    case = ev.parse_case(seed_case(iso_eval, "member: figure\n" + BODY))
    row = ev.run_case(case, tier="flash")
    assert seen["model_tier"] == "flash"
    assert row.tier == "flash" and row.model == "Qwen3.8-Flash"


def test_run_case_ledger_row_fields(iso_eval, monkeypatch):
    capture_dispatch(monkeypatch, ret=out_of(turns=7))
    patch_reg(monkeypatch, iso_eval)
    case = ev.parse_case(seed_case(iso_eval, "member: figure\n" + BODY))
    ev.run_case(case)
    rows = ev.read_rows()
    assert len(rows) == 1
    r = rows[0]
    assert {
        "case",
        "verdict",
        "model",
        "tier",
        "num_turns",
        "duration_ms",
        "est_tokens",
        "ctx_ratio",
        "sid",
        "label",
        "max_turns",
    } <= set(r)
    assert r["num_turns"] == 7 and r["est_tokens"] == 60
    assert r["ctx_ratio"] == pytest.approx(0.3) and r["model"] == "Qwen3.8-Flash"
    assert r["max_turns"] == ev.DEFAULT_MAX_TURNS, (
        "案内预算必须随行入台账：贴线判决只有台账可证，否则事后无从复核"
    )


def test_run_case_records_case_budget(iso_eval, monkeypatch):
    """``max_turns:`` 案内值入台账行（不是运行态猜的）。"""
    capture_dispatch(monkeypatch)
    patch_reg(monkeypatch, iso_eval)
    case = ev.parse_case(seed_case(iso_eval, "member: figure\nmax_turns: 8\n" + BODY))
    assert ev.run_case(case).max_turns == 8


# ---------------------------------------------------------------------------
# 压测跳隔离面：组员自维护层（跟踪的 AGENTS.md）不得被测量跳留下脏面
# ---------------------------------------------------------------------------
class FakeOk:
    returncode = 0


def fake_dirt(monkeypatch, *snapshots):
    """按调用顺序返回脏面快照（第 1 次=跳前，第 2 次=跳后），并记录查询范围。

    真 ``dirty_tracked_state`` 起 git subprocess 看**当前工作树**——单测里工作树内容
    随仓库状态变，断言会变成随机通过/失败。故快照由测试给定，只验差集逻辑与还原作用面。
    """
    seen: dict = {"scopes": [], "checkout": []}
    calls = list(snapshots)

    def fake(*, path: str = "", timeout_s: int = 10):
        seen["scopes"].append(path)
        return list(calls.pop(0)) if calls else []

    monkeypatch.setattr(ev, "dirty_tracked_state", fake)
    return seen


def test_run_case_reverts_pod_dirt_created_by_hop(iso_eval, monkeypatch):
    """本跳新弄脏的组员跟踪文件必须在判据前还原，并记进台账行。"""
    seen = capture_dispatch(monkeypatch)
    patch_reg(monkeypatch, iso_eval)
    dirt = fake_dirt(
        monkeypatch,
        [],
        ["orchestration/pods/figure/AGENTS.md"],
    )
    checkout: dict = {}

    class FakeProc:
        returncode = 0

    def fake_run(argv, **kw):
        checkout["argv"] = argv
        checkout["cwd"] = kw.get("cwd")
        return FakeProc()

    monkeypatch.setattr(ev, "procs", SimpleNamespace(run_hidden=fake_run))
    case = ev.parse_case(seed_case(iso_eval, "member: figure\n" + BODY))
    row = ev.run_case(case)
    assert row.reverted == ["orchestration/pods/figure/AGENTS.md"]
    assert checkout["argv"] == [
        "git",
        "checkout",
        "--",
        "orchestration/pods/figure/AGENTS.md",
    ]
    assert ev.read_rows()[0]["reverted"] == row.reverted, (
        "还原事实必须可事后复核（台账是唯一留痕面）"
    )
    assert dirt["scopes"] == ["orchestration/pods/figure"] * 2, (
        "快照范围=该组员的 pod，不是全树（全树会把别人的脏面也算进差集）"
    )


def test_revert_keeps_dirt_that_predated_hop(iso_eval, monkeypatch):
    """跳前就脏的文件是组长在飞的工作——还原它=覆盖他人未提交改动，禁止。"""
    monkeypatch.setattr(
        ev,
        "dirty_tracked_state",
        lambda *, path="", timeout_s=10: ["orchestration/pods/figure/AGENTS.md"],
    )
    ran = []
    monkeypatch.setattr(
        ev,
        "procs",
        SimpleNamespace(run_hidden=lambda argv, **kw: ran.append(argv) or FakeOk()),
    )
    before = {"orchestration/pods/figure/AGENTS.md"}
    assert ev.revert_new_dirt("figure", before) == []
    assert ran == [], "差集为空却调了 git checkout=无条件回滚，会把组长的工作抹掉"


def test_revert_refuses_paths_outside_pod(iso_eval, monkeypatch):
    """还原作用面钉在该 pod 前缀下（git checkout 是破坏性的，范围错就是事故）。"""
    monkeypatch.setattr(
        ev,
        "dirty_tracked_state",
        lambda *, path="", timeout_s=10: ["src/pysci/skills/orchestration/tools/x.py"],
    )
    ran = []
    monkeypatch.setattr(
        ev,
        "procs",
        SimpleNamespace(run_hidden=lambda argv, **kw: ran.append(argv) or FakeOk()),
    )
    assert ev.revert_new_dirt("figure", set()) == []
    assert ran == []


def test_revert_fail_open_on_git_error(iso_eval, monkeypatch):
    """git 非零退出 → 返回 []（还原失灵不该掀翻一次压测，也不该谎报「已还原」）。"""
    monkeypatch.setattr(
        ev,
        "dirty_tracked_state",
        lambda *, path="", timeout_s=10: ["orchestration/pods/figure/AGENTS.md"],
    )

    class Bad:
        returncode = 128

    monkeypatch.setattr(
        ev, "procs", SimpleNamespace(run_hidden=lambda argv, **kw: Bad())
    )
    assert ev.revert_new_dirt("figure", set()) == []


def test_run_case_brief_carries_isolation_note(iso_eval, monkeypatch):
    """任务书后必须附压测纪律：光靠事后还原，组员每轮都白写一次记忆。"""
    seen = capture_dispatch(monkeypatch)
    patch_reg(monkeypatch, iso_eval)
    fake_dirt(monkeypatch, [], [])
    ev.run_case(ev.parse_case(seed_case(iso_eval, "member: figure\n" + BODY)))
    text = seen["text"]
    assert "本跳是 harness 压测" in text and "AGENTS.md" in text, (
        "隔离纪律要进提示词，否则还原只是善后"
    )
    assert text.index("画一张单面板示意图") < text.index("本跳是 harness 压测"), (
        "案内任务书在前、纪律附在后——否则判据正文被 note 顶掉"
    )


def test_print_row_warns_when_budget_exhausted(capsys):
    """贴线 pass 必须被点名：turns 用满案内预算时判决由预算截断，不是干净通过。"""
    ev._print_row(
        ev.Row(
            ts="t",
            case="a",
            member="m",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=10,
            max_turns=10,
        )
    )
    out = capsys.readouterr().out
    assert "预算用满" in out and "贴线" in out and "turns=10/10" in out


def test_print_row_silent_within_budget(capsys):
    ev._print_row(
        ev.Row(
            ts="t",
            case="a",
            member="m",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=9,
            max_turns=10,
        )
    )
    assert "预算用满" not in capsys.readouterr().out


def test_run_case_does_not_write_member_ledger(iso_eval, monkeypatch):
    """压测行只进 evals.jsonl：混进 deliveries.jsonl 会把测量跳算进成员排名。"""
    seen = capture_dispatch(monkeypatch)
    patch_reg(monkeypatch, iso_eval)
    ev.run_case(ev.parse_case(seed_case(iso_eval, "member: figure\n" + BODY)))
    assert ev.EVAL_LEDGER_PATH.exists()
    assert seen["eval_hop"] is True, (
        "压测跳必须声明 eval_hop，否则生产台账/建议队列被污染"
    )
    assert ledgermod.LEDGER_PATH == Path(iso_eval) / "state" / "ledger.jsonl"
    assert not ledgermod.LEDGER_PATH.exists(), "生产台账不得被压测跳写入"


def _eval_dispatch(iso_dispatch, monkeypatch, **kw):
    set_models(iso_dispatch, minimum="Qwen3.8-Flash")
    fake_headless(
        monkeypatch,
        lambda i, cmd: envelope(
            '<result>done。</result>\n<infra_suggestion kind="usability">'
            "案卷太难读</infra_suggestion>",
            is_error=False,
            rc=0,
        ),
    )
    return dispatch.do_dispatch(
        "quotamember",
        text="压测正文",
        slug="eval-y",
        session="new",
        model_tier="minimum",
        max_turns=3,
        timeout=5,
        no_checks=True,
        quiet=True,
        **kw,
    )


def test_eval_hop_suppresses_production_side_effects(iso_dispatch, monkeypatch):
    """真实链路（假 exe）下 eval_hop=True：生产台账零行、建议队列零文件。"""
    out = _eval_dispatch(iso_dispatch, monkeypatch, eval_hop=True)
    assert out.kind == "result"
    assert ledgermod.read_all() == [], "压测跳不得写成员生产台账"
    assert out.suggestion_id == "", "压测跳不得往建议队列塞合成条目"


def test_production_hop_records_ledger_and_suggestion(iso_dispatch, monkeypatch):
    """对照：生产跳照旧记账并落建议——开关不得把正常链路一起关掉。"""
    out = _eval_dispatch(iso_dispatch, monkeypatch)
    assert len(ledgermod.read_all()) == 1
    assert out.suggestion_id, "生产跳的 <infra_suggestion> 必须入队"


# ---------------------------------------------------------------------------
# attach_replies 开关（dispatch 侧的真实链路，含假 headless）
# ---------------------------------------------------------------------------
def seed_reply(iso_dispatch, member="quotamember"):
    d = iso_dispatch / "replies" / member
    d.mkdir(parents=True, exist_ok=True)
    f = d / "20261011-233002-leader.md"
    f.write_text("# 组长审批：采纳\n\n同意。\n", encoding="utf-8")
    return f


def pod_prompt(calls) -> str:
    """从假 headless 记录的 argv 里取派发提示词（含任务书正文 + 附送的审批回复）。"""
    c = calls[0]
    return max((a for a in c if isinstance(a, str)), key=len)


def test_attach_replies_false_keeps_reply_pending(iso_dispatch, monkeypatch):
    set_models(
        iso_dispatch, minimum="byok-min"
    )  # BYOK 样本：内置席被硬过滤（2026-10-15）
    f = seed_reply(iso_dispatch)
    calls = fake_headless(
        monkeypatch,
        lambda i, cmd: envelope("<result>done。</result>", is_error=False, rc=0),
    )
    out = dispatch.do_dispatch(
        "quotamember",
        text="压测正文",
        slug="eval-x",
        session="new",
        model_tier="minimum",
        max_turns=3,
        timeout=5,
        no_checks=True,
        attach_replies=False,
        quiet=True,
    )
    assert out.kind == "result"
    assert f.exists(), "压测跳不得吃掉组长的审批回复（生产跳还要看它）"
    assert "组长审批" not in pod_prompt(calls)
    assert cmd_model(calls[0]) == "byok-min"


def test_attach_replies_default_appends_and_archives(iso_dispatch, monkeypatch):
    set_models(iso_dispatch, minimum="Qwen3.8-Flash")
    seed_reply(iso_dispatch)
    calls = fake_headless(
        monkeypatch,
        lambda i, cmd: envelope("<result>done。</result>", is_error=False, rc=0),
    )
    dispatch.do_dispatch(
        "quotamember",
        text="生产正文",
        slug="prod-x",
        session="new",
        model_tier="minimum",
        max_turns=3,
        timeout=5,
        no_checks=True,
        quiet=True,
    )
    d = iso_dispatch / "replies" / "quotamember"
    assert not (d / "20261011-233002-leader.md").exists()
    assert (d / "sent" / "20261011-233002-leader.md").exists()
    assert "组长审批：采纳" in pod_prompt(calls)


def test_production_dispatch_still_flash(iso_dispatch, monkeypatch):
    """E2 判据③：新增 minimum 档不得让常规派发降档——无覆盖时角色解析回 **main**
    （预设主力席），不是 minimum，也不是名单外成员够不着的 pro（档位退役 2026-10-12 22:4x）。"""
    calls = fake_headless(
        monkeypatch,
        lambda i, cmd: envelope("<result>done。</result>", is_error=False, rc=0),
    )
    out = dispatch.do_dispatch(
        "quotamember",
        text="正文",
        slug="prod-z",
        max_turns=3,
        timeout=5,
        no_checks=True,
        quiet=True,
    )
    assert out.kind == "result"
    reg = registry.Registry.load()
    assert presets.resolve_role("quotamember") == "main"
    assert cmd_model(calls[0]) == reg.model_chain("main")[0], (
        "无覆盖的生产派发取 main 席链首席"
    )


# ---------------------------------------------------------------------------
# 基线与对照：二元判决、无平局（§0.5）
# ---------------------------------------------------------------------------
def r(case, verdict, turns):
    return {
        "case": case,
        "verdict": verdict,
        "num_turns": turns,
        "est_tokens": turns * 100,
    }


def test_compare_improved_when_pass_up_turns_flat():
    base = {
        "cases": {
            "a": {"verdict": "pass", "num_turns": 10},
            "b": {"verdict": "fail", "num_turns": 12},
        }
    }
    got = ev.compare({"a": r("a", "pass", 10), "b": r("b", "pass", 12)}, base)
    assert got["verdict"] == "改善"
    assert (got["pass_base"], got["pass_now"]) == (1, 2)


def test_compare_improved_when_pass_flat_turns_down():
    base = {"cases": {"a": {"verdict": "pass", "num_turns": 12}}}
    assert ev.compare({"a": r("a", "pass", 8)}, base)["verdict"] == "改善"


def test_compare_tie_is_not_improvement():
    """§0.5 硬要求：完全持平也必须是二元之一——判「未改善」，不奖励无变化的改动。"""
    base = {"cases": {"a": {"verdict": "pass", "num_turns": 10}}}
    assert ev.compare({"a": r("a", "pass", 10)}, base)["verdict"] == "未改善"


def test_compare_regressions_are_not_improvement():
    base = {
        "cases": {
            "a": {"verdict": "pass", "num_turns": 10},
            "b": {"verdict": "pass", "num_turns": 10},
        }
    }
    assert (
        ev.compare({"a": r("a", "pass", 40), "b": r("b", "pass", 4)}, base)["verdict"]
        == "未改善"
    )
    assert (
        ev.compare({"a": r("a", "fail", 3), "b": r("b", "pass", 10)}, base)["verdict"]
        == "未改善"
    )


def test_compare_no_baseline_reports_only_now():
    got = ev.compare({"a": r("a", "pass", 5)}, {})
    assert (
        got["verdict"] == "无基线" and got["only_now"] == ["a"] and got["diffs"] == []
    )


def test_compare_new_case_without_baseline_entry(iso_eval):
    base = {"cases": {"a": {"verdict": "pass", "num_turns": 10}}}
    got = ev.compare({"a": r("a", "pass", 10), "n": r("n", "pass", 3)}, base)
    assert got["compared"] == ["a"] and got["only_now"] == ["n"]


def test_mean_turns_ignores_zeroes():
    """全 0 轮次（崩溃跳）不得伪装成「均值 0 = 大幅改善」。"""
    assert ev._mean_turns([r("a", "fail", 0), r("b", "fail", 0)]) == 0.0
    base = {"cases": {"a": {"verdict": "fail", "num_turns": 9}}}
    assert ev.compare({"a": r("a", "fail", 0)}, base)["mean_turns_now"] == 0.0


def test_baseline_roundtrip_and_label(iso_eval):
    ev.append_row(
        ev.Row(
            ts="t1",
            case="a",
            member="figure",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=9,
            label="before",
        )
    )
    ev.append_row(
        ev.Row(
            ts="t2",
            case="a",
            member="figure",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=4,
            label="after",
        )
    )
    assert ev.write_baseline("before")["cases"]["a"]["num_turns"] == 9
    assert ev.write_baseline("after")["cases"]["a"]["num_turns"] == 4
    doc = json.loads((iso_eval / "baseline.json").read_text(encoding="utf-8"))
    assert doc["version"] == 1 and doc["label"] == "after"
    assert ev.read_baseline()["cases"]["a"]["verdict"] == "pass"


def test_write_baseline_without_rows_raises(iso_eval):
    with pytest.raises(ValueError) as exc:
        ev.write_baseline()
    assert "eval run" in str(exc.value)


def test_read_rows_skips_corrupt_line(iso_eval):
    ev.append_row(
        ev.Row(ts="t", case="a", member="m", tier="minimum", model="x", verdict="pass")
    )
    with ev.EVAL_LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write("{截断的半行\n\n")
    assert len(ev.read_rows()) == 1


def test_read_baseline_corrupt_returns_empty(iso_eval):
    (iso_eval / "baseline.json").write_text("{坏 json", encoding="utf-8")
    assert ev.read_baseline() == {}


# ---------------------------------------------------------------------------
# CLI 面：rc=0 全通过 / rc=1 有失败 / rc=2 用法错
# ---------------------------------------------------------------------------
def ns(**kw):
    return argparse.Namespace(**kw)


ARGS = dict(case="", all=False, model_tier="", label="")


def test_cli_run_all_pass_rc0(iso_eval, monkeypatch, capsys):
    monkeypatch.setattr(ev, "do_dispatch", lambda m, **kw: out_of(turns=3))
    patch_reg(monkeypatch, iso_eval)
    seed_case(iso_eval, "member: figure\n" + BODY, "a")
    rc = ev.cmd_eval_run(ns(**ARGS))
    assert rc == 0
    o = capsys.readouterr().out
    assert "[√] a" in o and "无基线" in o and "set-baseline" in o


def test_cli_run_fail_rc1_with_reasons(iso_eval, monkeypatch, capsys):
    monkeypatch.setattr(ev, "do_dispatch", lambda m, **kw: out_of(kind="blocked"))
    patch_reg(monkeypatch, iso_eval)
    seed_case(iso_eval, "member: figure\n" + BODY, "a")
    assert ev.cmd_eval_run(ns(**ARGS)) == 1
    assert "[✗] a" in capsys.readouterr().out


def test_cli_run_rc_from_this_round_only(iso_eval, monkeypatch):
    """本轮全通过 → rc=0，即便台账里躺着上一轮的失败行。"""
    ev.append_row(
        ev.Row(
            ts="t",
            case="a",
            member="m",
            tier="minimum",
            model="x",
            verdict="fail",
            num_turns=1,
        )
    )
    monkeypatch.setattr(ev, "do_dispatch", lambda m, **kw: out_of())
    patch_reg(monkeypatch, iso_eval)
    seed_case(iso_eval, "member: figure\n" + BODY, "a")
    assert ev.cmd_eval_run(ns(**ARGS)) == 0


def test_cli_run_compares_with_baseline(iso_eval, monkeypatch, capsys):
    monkeypatch.setattr(ev, "do_dispatch", lambda m, **kw: out_of(turns=5))
    patch_reg(monkeypatch, iso_eval)
    seed_case(iso_eval, "member: figure\n" + BODY, "a")
    ev.append_row(
        ev.Row(
            ts="t",
            case="a",
            member="m",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=11,
        )
    )
    ev.write_baseline()
    assert ev.cmd_eval_run(ns(**ARGS)) == 0
    out = capsys.readouterr().out
    assert "改善" in out and "11 → 5" in out


def test_cli_run_marks_tight_case_in_diff_table(iso_eval, monkeypatch, capsys):
    """对照表也要点名贴线案：均值轮次被预算截断的案污染判决，读数不可信。"""
    monkeypatch.setattr(ev, "do_dispatch", lambda m, **kw: out_of(turns=8))
    patch_reg(monkeypatch, iso_eval)
    seed_case(iso_eval, "member: figure\nmax_turns: 8\n" + BODY, "a")
    ev.append_row(
        ev.Row(
            ts="t",
            case="a",
            member="m",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=4,
        )
    )
    ev.write_baseline()
    assert ev.cmd_eval_run(ns(**ARGS)) == 0
    out = capsys.readouterr().out
    assert "预算用满" in out and "4 → 8" in out


def test_cli_run_bad_case_rc2(iso_eval, capsys):
    assert ev.cmd_eval_run(ns(**ARGS)) == 2
    assert "案卷不可用" in capsys.readouterr().out


def test_cli_run_missing_harness_dir_rc2(iso_eval):
    assert ev.cmd_eval_run(ns(case="", all=True, model_tier="", label="")) == 2


def test_cli_set_baseline_rc2_then_rc0(iso_eval, capsys):
    assert ev.cmd_eval_set_baseline(ns(label="")) == 2
    ev.append_row(
        ev.Row(
            ts="t",
            case="a",
            member="m",
            tier="minimum",
            model="x",
            verdict="pass",
            num_turns=6,
        )
    )
    assert ev.cmd_eval_set_baseline(ns(label="")) == 0
    assert "基线已回填" in capsys.readouterr().out


def test_cli_eval_parser_shape(capsys):
    """``orch eval`` 的两子命令与必填互斥组：漏写 required 组会让 --all 变成哑参数。"""
    from pysci.skills.orchestration.tools import orch

    for argv in (["eval"], ["eval", "run"], ["eval", "run", "--case", "a", "--all"]):
        with pytest.raises(SystemExit) as exc:
            orch.main(argv)
        assert exc.value.code == 2, argv
    capsys.readouterr()

    seen = {}

    def fake_run(args):
        seen["args"] = args
        return 0

    mp = pytest.MonkeyPatch()
    mp.setattr(ev, "cmd_eval_run", fake_run)
    rc = orch.main(
        ["eval", "run", "--all", "--model-tier", "minimum", "--label", "before"]
    )
    assert rc == 0
    a = seen["args"]
    assert a.all and a.model_tier == "minimum" and a.label == "before"

    mp.setattr(ev, "cmd_eval_set_baseline", lambda args: seen.get("b") or 0)
    seen["b"] = 1
    assert orch.main(["eval", "set-baseline", "--label", "after"]) == 1
    mp.undo()
