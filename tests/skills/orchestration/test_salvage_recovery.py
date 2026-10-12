"""崩溃/腰斩后的**进度抢救 + 自动续作**（backlog 20261012-234000-leader）。

原本孤儿回收的做法是「删了再说」：``worktree remove --force`` + ``branch -D`` 把上一跳
没提交的半成品和没合并的提交一起销毁，续作还得组长手动盘点、手动立单。本组测试钉住
新契约的三段：

1. **回收先判脏再动手**：确认干净的面照旧销毁（不回归），没能确认干净的一律先打
   WIP-salvage 提交、分支与工作树都不删；旧会话没确认离场（隔离中）或目录只是摘不掉的
   残骸时，**绝不往那棵树里写**；
2. **续作自动接上**：条目还在队列里 → 就地注记抢救事实（下一跳回到同一棵树）；条目已落
   needs_leader → 另立 ``source=salvage`` 的 pending 续作单插到队首。两条通路都不必组长
   动手，且都带去重/链深闸，防队列自我复制；
3. **观察面看得见**：``orch status`` 的抢救段与 watch 的 SALVAGE 事件——知会不是待办。

git 读数走 :class:`conftest.GitFaces`（按面台账自洽应答）：判脏读的是「目录在不在、
登记不登记、脏不脏、领先几个提交」这组互相约束的事实，全绿/全红桩都只能测出一个方向。
"""

from __future__ import annotations

from argparse import Namespace

from pysci.skills.orchestration.tools import dispatch, drain, orch, watch, workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import GitFaces, read_backlog, seed_backlog

OID = "20261012-234000-leader"

#: 抢救记录的必备读数（drain 构造、任务书 / status / watch 三面读取）。
RECORD = {
    "worktree": OID,
    "branch": f"wt-{OID}",
    "commit": "c0ffee12",
    "path": f".qoder/worktrees/{OID}",
    "reason": "crash",
    "registered": True,
    "present": True,
    "uncommitted": False,
}


def _faces(iso_state, monkeypatch, tmp_path) -> GitFaces:
    """把 git 桩挂到本用例的隔离面根上（与 :func:`conftest.iso_state` 同一 tmp 锚）。"""
    return GitFaces(tmp_path / "worktrees").patch(monkeypatch)


# ---------------------------------------------------------------------------
# 判据读数：什么算「还有东西没保住」
# ---------------------------------------------------------------------------
def test_state_confirms_clean_face(iso_state, monkeypatch, tmp_path):
    _faces(iso_state, monkeypatch, tmp_path).face(OID)
    st = drain.worktree_salvage_state(OID)
    assert st["registered"] is True and st["dirty"] is False
    assert st["unmerged"] == 0 and st["unknown"] is False
    assert st["salvageable"] is False, "确认干净的面没有东西要救"


def test_state_reads_dirty_face_as_salvageable(iso_state, monkeypatch, tmp_path):
    _faces(iso_state, monkeypatch, tmp_path).face(OID, dirty=True)
    st = drain.worktree_salvage_state(OID)
    assert st["dirty"] is True and st["salvageable"] is True


def test_state_reads_unmerged_only_as_salvageable(iso_state, monkeypatch, tmp_path):
    """工作树干净但分支领先 main：进度在提交里，删分支=销毁。"""
    _faces(iso_state, monkeypatch, tmp_path).face(OID, unmerged=3)
    st = drain.worktree_salvage_state(OID)
    assert st["dirty"] is False and st["unmerged"] == 3
    assert st["salvageable"] is True


def test_state_unreadable_evidence_is_not_clean(iso_state, monkeypatch, tmp_path):
    """``rev-list`` 读不出数（rc≠0）→ 判 unknown → 往「有东西要救」的方向保守。"""
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face(OID)
    orig = git.run

    def flaky(*args):
        if args[:2] == ("rev-list", "--count"):
            return 1, "fatal: bad revision"
        return orig(*args)

    monkeypatch.setattr(drain, "_run_git", flaky)
    st = drain.worktree_salvage_state(OID)
    assert st["unknown"] is True and st["salvageable"] is True


def test_state_residue_dir_counts_as_dirty(iso_state, monkeypatch, tmp_path):
    """目录在、但未登记为 worktree（remove --force 摘不干净的残骸）→ 绝不判它干净。"""
    _faces(iso_state, monkeypatch, tmp_path).face(OID, registered=False, lingering=True)
    st = drain.worktree_salvage_state(OID)
    assert st["present"] is True and st["registered"] is False
    assert st["salvageable"] is True


def test_state_ghost_has_nothing_to_save(iso_state, monkeypatch, tmp_path):
    _faces(iso_state, monkeypatch, tmp_path)
    st = drain.worktree_salvage_state("nope")
    assert st["salvageable"] is False, "既无目录也无分支：无物可救，也别虚报抢救"


# ---------------------------------------------------------------------------
# 抢救动作：打成提交、保住面与分支（该不动的树绝不进）
# ---------------------------------------------------------------------------
def test_salvage_writes_wip_commit_and_keeps_branch(iso_state, monkeypatch, tmp_path):
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face(OID, dirty=True, unmerged=1, head="c0ffee12")
    record = drain.salvage_worktree(OID, entry_id=OID, reason="crash")
    assert record is not None
    assert git.commits and git.commits[0][0] == OID
    msg = git.commits[0][1]
    assert msg.startswith(drain.SALVAGE_COMMIT_TAG), "抢救提交要能被认出来"
    assert OID in msg and "勿据此销账" in msg
    assert ("branch", "-D", f"wt-{OID}") not in git.calls, "抢救不是清理"
    assert not [c for c in git.calls if c[:3] == ("worktree", "remove", "--force")]
    assert (tmp_path / "worktrees" / OID).is_dir(), "续作面留在原地才是设计意图"
    assert record["commit"] == "wipsalv1", "提交后的 head 才是续作要指的 sha"
    assert record["uncommitted"] is False and record["branch"] == f"wt-{OID}"


def test_salvage_never_writes_into_unregistered_dir(iso_state, monkeypatch, tmp_path):
    """残骸目录里的 git 读数不可信（最坏回落到父仓工作区）→ 只记账、不进去写。"""
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face(OID, registered=False, lingering=True)
    record = drain.salvage_worktree(OID, entry_id=OID, reason="crash")
    assert record is not None and record["uncommitted"] is True
    assert record["registered"] is False
    assert git.commits == [], "往残骸里 add/commit 就是把别人的人生卷进我们的提交"
    assert not [c for c in git.calls if c[:1] == ("-C",)]


def test_salvage_quarantined_face_is_read_only(iso_state, monkeypatch, tmp_path):
    """第四面没能确认旧写者离场 → 隔离：读证据、保面，但不碰工作树。"""
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face(OID, dirty=True)
    cleaned = drain.cleanup_orphan_worktrees([OID], None, quarantined=[OID])
    assert cleaned == []
    assert git.commits == [] and git.removed(OID) is False
    assert ("branch", "-D", f"wt-{OID}") not in git.calls


def test_salvage_noop_when_nothing_to_save(iso_state, monkeypatch, tmp_path):
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face(OID)
    assert drain.salvage_worktree(OID) is None
    assert git.commits == []


def test_cleanup_keeps_dirty_and_removes_clean(iso_state, monkeypatch, tmp_path):
    """同一批里两面各自走向：干净的照旧销毁，脏的保住并交回读数。"""
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face("clean", unmerged=0)
    git.face("dirty", dirty=True)
    sink: dict = {}
    cleaned = drain.cleanup_orphan_worktrees(
        ["clean", "dirty"], None, salvage_sink=sink
    )
    assert cleaned == ["clean"]
    assert set(sink) == {"dirty"}
    assert sink["dirty"]["commit"], "抢救读数要给出续作能指的提交"
    assert ("branch", "-D", "wt-clean") in git.calls
    assert ("branch", "-D", "wt-dirty") not in git.calls
    assert ("worktree", "prune") in git.calls


def test_salvage_failure_leaves_face_and_branch(iso_state, monkeypatch, tmp_path):
    """commit 打不成（读数说脏、落盘说没脏）→ 面与分支仍原样保留，记录标 uncommitted。"""
    git = _faces(iso_state, monkeypatch, tmp_path)
    git.face(OID, dirty=True)
    orig = git.run

    def no_commit(*args):
        if args[2:4] == ("commit", "--no-verify"):
            return 1, "nothing to commit, working tree clean"
        return orig(*args)

    monkeypatch.setattr(drain, "_run_git", no_commit)
    log = drain.new_run_log()
    record = drain.salvage_worktree(OID, log, entry_id=OID, reason="crash")
    assert record is not None and record["uncommitted"] is True
    assert "commit 未发生" in log.read_text(encoding="utf-8")
    assert (tmp_path / "worktrees" / OID).is_dir()
    assert ("branch", "-D", f"wt-{OID}") not in git.calls


# ---------------------------------------------------------------------------
# 队列接线：还在队里的就地注记，已落 needs_leader 的另立续作单
# ---------------------------------------------------------------------------
def test_attach_marks_entry_and_notes_fact(iso_state):
    seed_backlog(iso_state, [{"id": OID, "status": "pending", "member": "devops"}])
    hit = workflow.backlog_attach_salvage(OID, RECORD)
    assert hit is not None and hit["salvage"]["from"] == OID
    assert hit["salvage"]["gen"] == 1
    assert "抢救续作" in hit["note"] and RECORD["commit"] in hit["note"]


def test_attach_is_idempotent_for_same_commit(iso_state):
    seed_backlog(iso_state, [{"id": OID, "status": "pending", "member": "devops"}])
    assert workflow.backlog_attach_salvage(OID, RECORD) is not None
    assert workflow.backlog_attach_salvage(OID, RECORD) is None, "重跑回收不重复注记"
    on_disk = read_backlog(iso_state)[0]
    assert on_disk["salvage"]["gen"] == 1


def test_attach_skips_needs_leader_entry(iso_state):
    """needs_leader 是组长已做出的裁决，代码不撤状态——那条通路走另立续作单。"""
    seed_backlog(iso_state, [{"id": OID, "status": "needs_leader", "member": "devops"}])
    assert workflow.backlog_attach_salvage(OID, RECORD) is None
    assert read_backlog(iso_state)[0]["status"] == "needs_leader"


def test_append_registers_continuation_before_pending(iso_state):
    seed_backlog(
        iso_state,
        [
            {
                "id": OID,
                "status": "needs_leader",
                "member": "devops",
                "summary": "抢救单",
                "kind": "defect",
                "success_criteria": "进度进 main",
            },
            {"id": "later", "status": "pending", "member": "figure"},
        ],
    )
    cont = workflow.backlog_append_salvage(read_backlog(iso_state)[0], RECORD)
    assert cont is not None
    assert cont["source"] == workflow.SALVAGE_SOURCE
    assert cont["status"] == "pending" and cont["kind"] == "defect"
    assert cont["success_criteria"] == "进度进 main", "原判据随续作走，别另立一套"
    assert cont["summary"].startswith(f"（续作 {OID}）")
    assert cont["salvage"]["commit"] == RECORD["commit"]
    items = read_backlog(iso_state)
    assert [i["id"] for i in items if i["status"] == "pending"][0] == cont["id"], (
        "续作不是新提案，插到 pending 队首而不是排队尾"
    )
    orig = next(i for i in items if i["id"] == OID)
    assert orig["status"] == "needs_leader" and "已自动立续作单" in orig["note"]


def test_append_dedupes_commit_already_queued(iso_state):
    seed_backlog(
        iso_state,
        [
            {"id": OID, "status": "needs_leader", "member": "devops"},
            {"id": "que", "status": "pending", "salvage": {"commit": RECORD["commit"]}},
        ],
    )
    assert workflow.backlog_append_salvage(read_backlog(iso_state)[0], RECORD) is None
    assert len(read_backlog(iso_state)) == 2, "同一份 WIP 已有单在队，不再复制"


def test_append_chain_cap_hands_back_to_leader(iso_state):
    seed_backlog(
        iso_state,
        [
            {
                "id": OID,
                "status": "needs_leader",
                "member": "devops",
                "salvage": {"gen": workflow.SALVAGE_MAX_CHAIN, "commit": "old"},
            }
        ],
    )
    assert workflow.backlog_append_salvage(read_backlog(iso_state)[0], RECORD) is None
    note = read_backlog(iso_state)[0]["note"]
    assert "上限" in note and "须组长处置" in note, "链到顶要可见，不能静默吞掉"


def test_salvage_entries_are_never_grouped(iso_state):
    """续作单各带自己的抢救面：一跳一树，组包会把两份半成品塞进同一棵树。"""
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "devops", "salvage": RECORD},
            {"id": "s2", "status": "pending", "member": "devops"},
            {"id": "s3", "status": "pending", "member": "devops"},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "s1" and group == []

    seed_backlog(
        iso_state,
        [
            {"id": "n1", "status": "pending", "member": "devops"},
            {"id": "n2", "status": "pending", "member": "devops", "salvage": RECORD},
        ],
    )
    seed, group = workflow.backlog_take_batch(4)
    assert seed["id"] == "n1"
    assert [g["id"] for g in group] == [], "带抢救面的项不进别人的组包"


# ---------------------------------------------------------------------------
# 任务书：把「上次到哪了」写成可执行指令
# ---------------------------------------------------------------------------
def test_task_text_injects_continuation_instruction(iso_state):
    item = {
        "id": OID,
        "summary": "s",
        "evidence": "e",
        "salvage": RECORD,
    }
    text = workflow.backlog_task_text(item)
    assert "续作指令（自动抢救）" in text
    assert "还在原地且登记在册" in text
    assert f"wt-{OID}" in text and RECORD["commit"] in text
    assert "不要** reset" in text


def test_task_text_rebuild_face_pulls_branch(iso_state):
    """升代后的续作面换了名 → 指令改成「新面 + cherry-pick 取回」，不复用旧名。"""
    item = {
        "id": OID,
        "summary": "s",
        "worktree_gen": 2,
        "salvage": {**RECORD, "registered": False},
    }
    text = workflow.backlog_task_text(item)
    assert f"{OID}-r2" in text
    assert "cherry-pick --no-commit" in text and RECORD["commit"] in text


def test_task_text_branch_only_without_dir(iso_state):
    """工作树已被摘走、分支还在：pysci-dev add 会从 main 建同名分支而失败，指令须给对命令。"""
    item = {
        "id": OID,
        "summary": "s",
        "salvage": {**RECORD, "registered": False, "present": False},
    }
    text = workflow.backlog_task_text(item)
    assert "git worktree add" in text and f"wt-{OID}" in text


def test_task_text_warns_uncommitted_not_recovered(iso_state):
    item = {"id": OID, "summary": "s", "salvage": {**RECORD, "uncommitted": True}}
    text = workflow.backlog_task_text(item)
    assert "status --short" in text and "手工移植" in text


def test_task_text_without_salvage_is_unchanged():
    text = workflow.backlog_task_text({"id": "plain", "summary": "s"})
    assert "续作指令" not in text


# ---------------------------------------------------------------------------
# drain 集成：崩溃 → 抢救 → 同一轮自动续完（零组长手动）
# ---------------------------------------------------------------------------
def _drain_with_dispatch(iso_state, monkeypatch, tmp_path, items, outcome_of, *, face):
    """摆孤儿条目 + 一个按调用序应答的假派发，跑一次真 drain。"""
    seed_backlog(iso_state, items)
    git = GitFaces(tmp_path / "worktrees").patch(monkeypatch)
    git.face(face[0], **face[1])
    texts: list[str] = []

    def fake(member, *, text="", **kw):
        texts.append(text)
        return outcome_of(len(texts))

    monkeypatch.setattr(dispatch, "do_dispatch", fake)
    summary = drain.drain_backlog(drain.new_run_log())
    return summary, texts, git, read_backlog(iso_state)


def test_drain_crash_salvages_then_finishes_same_run(iso_state, monkeypatch, tmp_path):
    orphan = {
        "id": OID,
        "status": "in_progress",
        "started_at": "x",
        "member": "devops",
        "summary": "被打断的单",
        "kind": "defect",
        "success_criteria": "跑完",
    }
    summary, texts, git, items = _drain_with_dispatch(
        iso_state,
        monkeypatch,
        tmp_path,
        [orphan],
        lambda n: DispatchOutcome(code=0, kind="result", body="done 全绿"),
        face=(OID, {"dirty": True, "unmerged": 1, "head": "c0ffee12"}),
    )
    assert summary["reclaimed"] == [OID]
    assert summary["worktrees_salvaged"] == [OID]
    assert "worktrees_cleaned" not in summary and "worktrees_regenerated" not in summary
    assert git.commits and git.commits[0][1].startswith(drain.SALVAGE_COMMIT_TAG)
    assert ("branch", "-D", f"wt-{OID}") not in git.calls
    assert "续作指令" in texts[0] and "还在原地且登记在册" in texts[0]
    assert items[0]["status"] == "done", "下一跳自动回到同一棵树把单子做完"
    assert items[0]["salvage"]["commit"] == "wipsalv1"


def test_drain_run_failed_registers_continuation(iso_state, monkeypatch, tmp_path):
    """失败收场（needs_leader 不再自动重派）→ 自动立 source=salvage 的续作单并接着消化。"""
    seed = {
        "id": "faily",
        "status": "pending",
        "member": "devops",
        "summary": "干不完的单",
        "kind": "defect",
        "success_criteria": "跑完",
    }
    outcomes = [
        DispatchOutcome(code=2, kind="run_failed", error="进程半路死了"),
        DispatchOutcome(code=0, kind="result", body="续完 全绿"),
    ]
    summary, texts, git, items = _drain_with_dispatch(
        iso_state,
        monkeypatch,
        tmp_path,
        [seed],
        lambda n: outcomes[min(n, len(outcomes)) - 1],
        face=("faily", {"dirty": True, "unmerged": 2, "head": "c0ffee12"}),
    )
    orig = next(i for i in items if i["id"] == "faily")
    assert orig["status"] == "needs_leader", "失败这一跳仍是组长要看到的事实"
    cont = [i for i in items if i.get("source") == workflow.SALVAGE_SOURCE]
    assert len(cont) == 1 and cont[0]["salvage"]["from"] == "faily"
    assert cont[0]["status"] == "done", "续作单自动入队、自动被消化（零手动）"
    assert git.commits and git.commits[0][1].startswith(drain.SALVAGE_COMMIT_TAG)
    assert "cherry-pick --no-commit" in texts[1], "续作面换了名 → 指令给出取回方式"
    assert summary["items"][0]["continuation"] == cont[0]["id"]


def test_drain_quota_stop_salvages_partial_work(iso_state, monkeypatch, tmp_path):
    """渠道半路断（额度/渠道故障）：复位回队的那一条也要把进度保住并指回同一棵树。"""
    seed = {
        "id": "half",
        "status": "pending",
        "member": "devops",
        "summary": "干到一半",
        "kind": "defect",
        "success_criteria": "跑完",
    }
    summary, texts, git, items = _drain_with_dispatch(
        iso_state,
        monkeypatch,
        tmp_path,
        [seed],
        lambda n: DispatchOutcome(code=2, kind="quota_exhausted", body="credit limit"),
        face=("half", {"dirty": True, "unmerged": 1, "head": "c0ffee12"}),
    )
    assert summary["stopped"] == "quota_exhausted"
    assert git.commits and git.commits[0][1].startswith(drain.SALVAGE_COMMIT_TAG)
    assert items[0]["status"] == "pending" and items[0]["salvage"]["from"] == "half"
    assert summary["items"][0]["salvage"] == "wipsalv1"
    assert "续作指令" not in texts[0], "全局停止后不再派下一跳"


def test_salvage_progress_noop_on_clean_face(iso_state, monkeypatch, tmp_path):
    """面上确实没东西（干净且已合并）→ 不抢救、不立单，failure 通路保持原样。"""
    seed = {"id": "tidy", "status": "pending", "member": "devops", "summary": "s"}
    summary, texts, git, items = _drain_with_dispatch(
        iso_state,
        monkeypatch,
        tmp_path,
        [seed],
        lambda n: DispatchOutcome(code=2, kind="blocked", body="缺授权"),
        face=("tidy", {"unmerged": 0}),
    )
    assert git.commits == []
    assert [i for i in items if i.get("source") == workflow.SALVAGE_SOURCE] == []
    assert next(i for i in items if i["id"] == "tidy")["status"] == "needs_leader"
    assert summary["items"][0].get("salvage", "") == ""


# ---------------------------------------------------------------------------
# 观察面：知会（不要求动手）
# ---------------------------------------------------------------------------
def test_watch_reports_salvage_once_by_identity(iso_state):
    """SALVAGE 以「条目 id 集合」为身份：新出现的续作面报一次，之后进基线不再打扰。"""
    base = {
        "lock_state": "idle",
        "lock_pid": 0,
        "lock_run": "",
        "lock_done": "",
        "crash_run": "",
        "done": "",
        "done_stopped": "",
        "done_completed": "",
        "done_count": 0,
        "done_elapsed_s": "?",
        "needs_leader": [],
        "salvage": [],
        "pending": 0,
        "quota_run": "",
        "channel_run": "",
    }
    cur = {**base, "salvage": [OID], "pending": 1}
    events = watch.detect_events(base, cur)
    assert [e["event"] for e in events if e["event"] == "SALVAGE"] == ["SALVAGE"]
    assert events[0]["subject"] == OID
    again = watch.detect_events({**base, "salvage": [OID]}, cur)
    assert [e for e in again if e["event"] == "SALVAGE"] == []


def test_salvage_sits_at_lowest_priority():
    """抢救是「不必动手」的知会，不该盖住 CRASH/额度这类真要处理的信号。"""
    assert watch.PRIORITY[-1] == "SALVAGE"
    assert "SALVAGE" in watch.PRIORITY


def test_snapshot_collects_salvage_ids(iso_state):
    seed_backlog(iso_state, [{"id": OID, "status": "pending", "salvage": RECORD}])
    snap = watch.snapshot()
    assert snap["salvage"] == [OID]


def test_orch_status_shows_salvage_section(iso_state, capsys):
    seed_backlog(
        iso_state,
        [{"id": OID, "status": "pending", "member": "devops", "salvage": RECORD}],
    )
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "抢救续作面" in out and OID in out and RECORD["commit"] in out
