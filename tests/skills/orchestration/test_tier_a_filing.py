"""§2.4 阈值自动立案（E1，backlog 20261011-233001-leader）：立案、去重、进 approve 面。

README §5.4 的设计纪律是「计数与立案全代码完成，人只在强制选择点出现」——故本文件锁死：
① 达阈值必须在**命令入口**（orch status / orch approve）自动出现，不靠人主动去查；
② 同指纹不重复立案（24h 内、以及旧案仍未处置时）；
③ 立出来的案子必须**能被现行 approve 面吃下**（kind/success_criteria 齐备），
   且入账条目带 ``source=tier-a``、审批回复归 devops（自动立案没有真提请成员）。

workflow/dispatch 改动须阳性覆盖（20261010-193601 教训）：真跑 cmd_*，断言落盘结果。
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta

from pysci.skills.orchestration.tools import orch, workflow
from pysci.skills.orchestration.tools.delivery import parse_suggestion_fields
from pysci.skills.orchestration.tools.ledger import LedgerEntry

from .conftest import read_backlog

TIER = {"window_days": 14, "window_hops": 20, "threshold": 3}


def _ts(days_ago: float = 0) -> str:
    ts = datetime.now().astimezone() - timedelta(days=days_ago)
    return ts.isoformat(timespec="seconds")


def _e(member: str, tags: list[str], *, ts: str = "") -> LedgerEntry:
    return LedgerEntry(
        ts=ts or _ts(),
        member=member,
        session_id=f"sid-{member}",
        kind="blocked",
        tags=tags,
        task_file=f"orchestration/pods/{member}/inbox/task-x.md",
        num_turns=4,
        duration_ms=2000,
    )


def _rows(tag: str, member: str, n: int = 3) -> list[LedgerEntry]:
    return [_e(member, [tag]) for _ in range(n)]


def sweep(state, entries, tier=TIER):
    """在隔离 state 里跑一次扫描，返回新立案 id 列表。"""
    return workflow.tier_a_sweep(
        entries=entries, tier=tier, filed_path=state / "tier-a-filed.json"
    )


# ---------------------------------------------------------------------------
# ① 达阈值 → 立案；未达 → 不立
# ---------------------------------------------------------------------------


def test_below_threshold_files_nothing(iso_state):
    assert sweep(iso_state, _rows("cli_missing", "drawing", 2)) == []
    assert workflow.pending_suggestions() == []


def test_threshold_files_one_proposal(iso_state):
    filed = sweep(iso_state, _rows("cli_missing", "drawing"))
    assert len(filed) == 1
    path = iso_state / "suggestions" / f"{filed[0]}.md"
    assert path.exists()
    body = path.read_text(encoding="utf-8")
    # 文件名尾段是 tier-a（自动立案的来源标记，approve 的 _member_of 据此识别）
    assert filed[0].endswith("-tier-a")
    assert "cli_missing" in body and "drawing" in body
    assert "累计 3 次" in body  # summary 含计数（规格要求）


def test_proposal_is_approvable(iso_state):
    """立出来的案子必须过 approve 的两道硬校验（kind + 可证伪判据），否则机制空转。"""
    filed = sweep(iso_state, _rows("quota", "figure"))
    body = (iso_state / "suggestions" / f"{filed[0]}.md").read_text(encoding="utf-8")
    kind, criteria = parse_suggestion_fields(body)
    assert kind == "defect"
    assert criteria.strip() and "阈值" in criteria


def test_evidence_rows_are_referenced(iso_state):
    """证据=相关台账行引用（时间戳+任务书），不是模型转述。"""
    filed = sweep(iso_state, _rows("tool_bug", "sim"))
    body = (iso_state / "suggestions" / f"{filed[0]}.md").read_text(encoding="utf-8")
    assert "deliveries.jsonl" in body
    assert body.count("- 2026") >= 3  # 每行证据都带 ts
    assert "task-x.md" in body


def test_two_members_same_tag_file_two_cases(iso_state):
    """立案单元是 (标签, 成员)：同一标签两个成员 → 两个案子（处置方向不同）。"""
    entries = _rows("cli_missing", "drawing") + _rows("cli_missing", "figure")
    assert len(sweep(iso_state, entries)) == 2


def test_window_params_come_from_tier(iso_state):
    """阈值读参数不读硬编码：threshold=5 时 3 次不立案。"""
    assert sweep(iso_state, _rows("quota", "lit"), {**TIER, "threshold": 5}) == []


# ---------------------------------------------------------------------------
# ② 去重
# ---------------------------------------------------------------------------


def test_repeat_sweep_does_not_refile(iso_state):
    """success_criteria：重复运行不产生第二条。"""
    entries = _rows("cli_missing", "drawing")
    assert len(sweep(iso_state, entries)) == 1
    assert sweep(iso_state, entries) == []
    assert len(workflow.pending_suggestions()) == 1


def test_fingerprint_persisted(iso_state):
    sweep(iso_state, _rows("env_broken", "devops"))
    data = (iso_state / "tier-a-filed.json").read_text(encoding="utf-8")
    assert "env_broken|devops|" in data


def test_new_bucket_while_pending_still_suppressed(iso_state):
    """窗口桶前进（新证据落到新的一天）也不重立：旧案未处置就是同一个案子。

    这条锁的是 ``open_keys`` 那一道（与 24h 指纹窗独立的机制）——桶每天前进，若只靠
    指纹去重，一个没修好的问题会被每天立一次案。
    """
    old = _rows("doc_stale", "theory")
    assert len(sweep(iso_state, old)) == 1
    aged = old + [_e("theory", ["doc_stale"], ts=_ts(days_ago=-2))]  # 未来时间戳=新桶
    assert sweep(iso_state, aged) == []
    assert len(workflow.pending_suggestions()) == 1


def test_pending_case_blocks_daily_refile(iso_state):
    """未处置的旧案不得每天重立：指纹换桶了也要被 pending 面挡住。"""
    old = _rows("design_smell", "writing")
    filed = sweep(iso_state, old)
    # 手工把指纹台账清空（模拟跨天换桶），旧案仍在 suggestions/ 顶层
    (iso_state / "tier-a-filed.json").unlink()
    assert sweep(iso_state, old) == []
    assert len(workflow.pending_suggestions()) == len(filed)


def test_resolved_case_can_file_again_next_bucket(iso_state):
    """案子被 reject（离开 pending 面）且换桶后，同一标签再次达阈值应能重新立案。"""
    old = _rows("model_capability", "deputy")
    filed = sweep(iso_state, old)
    (iso_state / "suggestions" / f"{filed[0]}.md").unlink()  # 已处置
    aged = old + [_e("deputy", ["model_capability"], ts=_ts(days_ago=-2))]
    assert len(sweep(iso_state, aged)) == 1


# ---------------------------------------------------------------------------
# ③ 命令入口：status / approve 都触发扫描
# ---------------------------------------------------------------------------


def test_status_triggers_sweep_and_shows_case(iso_state, monkeypatch, capsys):
    """orch status 入口机械重算并把案子呈现出来（不新增人工检查点）。"""
    entries = _rows("cli_missing", "drawing")
    monkeypatch.setattr(workflow, "read_all", lambda: entries)
    monkeypatch.setattr(orch, "read_all", lambda: entries)
    monkeypatch.setattr(orch._drain, "lock_status", lambda: {"state": "idle"})
    monkeypatch.setattr(orch._drain, "cancel_status", lambda: None)
    monkeypatch.setattr(orch._drain, "latest_done", lambda: None)

    orch.cmd_status(Namespace())

    out = capsys.readouterr().out
    assert "Tier A 阈值立案" in out
    assert "[建议]" in out and "tier-a" in out
    assert len(workflow.pending_suggestions()) == 1


def test_status_repeat_does_not_duplicate(iso_state, monkeypatch, capsys):
    entries = _rows("cli_missing", "drawing")
    monkeypatch.setattr(workflow, "read_all", lambda: entries)
    for mod in (orch, workflow):
        monkeypatch.setattr(mod, "read_all", lambda: entries)
    monkeypatch.setattr(orch._drain, "lock_status", lambda: {"state": "idle"})
    monkeypatch.setattr(orch._drain, "cancel_status", lambda: None)
    monkeypatch.setattr(orch._drain, "latest_done", lambda: None)
    orch.cmd_status(Namespace())
    capsys.readouterr()
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "Tier A 阈值立案" not in out  # 第二次不再立
    assert len(workflow.pending_suggestions()) == 1


def test_approve_entry_triggers_sweep(iso_state, monkeypatch):
    """approve 入口也重算：批一条案的同时把新越限的案立上。"""
    sid = None
    from .conftest import seed_suggestion

    sid = seed_suggestion(iso_state)
    entries = _rows("harness_gap", "reviewer")
    monkeypatch.setattr(workflow, "read_all", lambda: entries)
    rc = workflow.cmd_approve(
        Namespace(suggestion_id=sid, note="采纳", no_wake=True, override="测试自裁")
    )
    assert rc == 0
    stems = [p.stem for p in workflow.pending_suggestions()]
    assert any(s.endswith("-tier-a") for s in stems)


# ---------------------------------------------------------------------------
# ④ 立案条目经 approve 后的形态
# ---------------------------------------------------------------------------


def test_approve_of_auto_case_marks_source(iso_state, monkeypatch):
    filed = sweep(iso_state, _rows("whitelist_conflict", "lit"))
    monkeypatch.setattr(workflow, "read_all", lambda: [])
    rc = workflow.cmd_approve(
        Namespace(
            suggestion_id=filed[0], note="批准", no_wake=True, override="测试自裁"
        )
    )
    assert rc == 0
    item = read_backlog(iso_state)[0]
    assert item["source"] == "tier-a"
    assert item["kind"] == "defect"
    assert item["success_criteria"].strip()
    assert "累计 3 次" in item["summary"]


def test_auto_case_reply_goes_to_devops(iso_state, monkeypatch):
    """自动立案没有真提请成员，审批回复必须归 devops（否则回执永不到达）。"""
    filed = sweep(iso_state, _rows("tool_bug", "figure"))
    monkeypatch.setattr(workflow, "read_all", lambda: [])
    workflow.cmd_approve(
        Namespace(
            suggestion_id=filed[0], note="批准", no_wake=True, override="测试自裁"
        )
    )
    assert (iso_state / "replies" / "devops" / f"{filed[0]}.md").exists()
    assert not (iso_state / "replies" / "tier-a").exists()


def test_human_proposal_reply_still_goes_to_proposer(iso_state):
    """回归护栏：非自动案（组员建议）的回复面不变，仍归提请成员。"""
    from .conftest import seed_suggestion

    sid = seed_suggestion(iso_state, member="figure")
    workflow.cmd_approve(
        Namespace(suggestion_id=sid, note="采纳", no_wake=True, override="测试自裁")
    )
    assert (iso_state / "replies" / "figure" / f"{sid}.md").exists()
