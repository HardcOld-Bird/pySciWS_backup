"""孤儿**会话进程**回收（第四面）+ 异名隔离重建（backlog 20261012-184000-leader）。

事故本体：worker 被杀后它派生的 headless 会话并不随之退出，仍在原 worktree 写入；
下一次 drain 回收条目、删同名 worktree、重派同单 → 新旧两个会话并发双写同一个隔离面
（实证见该单的证据：新会话读到另一条时间线，整跳白跑）。

原本回收只动三面（锁、backlog 条目、worktree 名），本组测试钉住补上的两条通路：

1. **终止通路**：派发瞬间把会话进程 pid 记进运行态（pid 键的存活期=该跳在途期）；
   回收者按孤儿条目找到 pid，判活 + 核对身份后终止进程树；
2. **异名重建通路**：只要没能确认「旧写者已离场」——杀不掉、pid 已易主、或 remove 后
   目录仍趴在原地——条目就升一代（``<id>-r<gen>``），新一代派发绝不复用旧名。

身份闸（pid 会被系统复用）与「判不出来就不开火」的保守方向是本单的安全核心，故
每个通路都各钉正反两例，而不是只看日志文案。

与抢救面的分工（backlog 20261012-234000-leader）：本模块只管「同名会不会双写」；
升代之前「面上还有没有没保住的进度」由 ``test_salvage_recovery.py`` 钉——过去这两件事
被混成了一个 ``worktree remove --force``，升代顺带把进度也销毁了。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    drain,
    procs,
    registry,
    runner,
    sessions,
    workflow,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import FAKE_HOP_PID, GitFaces, envelope, read_backlog, seed_backlog

OID = "20261012-184000-leader"
SID = "11111111-2222-3333-4444-555555555555"
TASKBOOK = f"task-20261012-185839-{OID}.md"

#: 孤儿条目（回收前状态：in_progress + 已死的 started_at）。
ORPHAN_ITEM = {
    "id": OID,
    "status": "in_progress",
    "started_at": "x",
    "member": "devops",
    "summary": "孤儿单",
    "effort": "",
}


# ---------------------------------------------------------------------------
# 夹具：backlog（iso_state）+ 运行态会话池（iso_registry）+ 隔离 worktrees 根
# ---------------------------------------------------------------------------
@pytest.fixture
def takeover(iso_state, iso_registry, tmp_path, monkeypatch):
    """返回 ``SimpleNamespace(state, runtime, wt_root)``，三面都在 tmp 里。

    ``iso_registry`` 只 patch ``REGISTRY_PATH`` 而**不** no-op save，所以
    :func:`registry.note_hop_pid` 真的落盘、也真能读回来——这正是第四面判据的形态。
    """
    reg_file, runtime_file, _backup = iso_registry
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "models": {"max": "", "flash": ""},
                "members": {
                    "devops": {
                        "pod": str(tmp_path / "pods" / "devops"),
                        "model_tier": "flash",
                        "max_turns": 3,
                        "timeout_s": 5,
                    }
                },
                "checks": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    wt_root = tmp_path / "worktrees"
    wt_root.mkdir(
        exist_ok=True
    )  # iso_state 已按同一锚建过（本夹具只是把 patch 再钉一次）
    # drain 读的是 workflow.WORKTREES_ROOT（动态属性），一处 patch 即两面生效
    monkeypatch.setattr(workflow, "WORKTREES_ROOT", wt_root)
    return SimpleNamespace(
        state=iso_state, runtime=runtime_file, wt_root=wt_root, tmp=tmp_path
    )


def _seed_hop(fx, *, pid=FAKE_HOP_PID, task=TASKBOOK, sid=SID, member="devops") -> None:
    """在运行态会话池里摆一条「在途跳」（pid 非零 = 该跳没走完）。"""
    fx.runtime.write_text(
        json.dumps(
            {
                "version": 1,
                "members": {
                    member: {
                        "sessions": [
                            {
                                "sid": sid,
                                "name": member,
                                "hops": 1,
                                "pid": pid,
                                "pid_task": task,
                            }
                        ]
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _hop(fx, member="devops", sid=SID) -> dict:
    data = json.loads(fx.runtime.read_text(encoding="utf-8"))
    for s in data["members"][member]["sessions"]:
        if s["sid"] == sid:
            return s
    raise AssertionError(f"会话 {sid} 不在运行态面里")


# ---------------------------------------------------------------------------
# 归属面：note_hop_pid 落运行态、pid=0 收工
# ---------------------------------------------------------------------------
def test_note_hop_pid_lands_on_runtime_face(takeover):
    _seed_hop(takeover, pid=0, task="")

    registry.note_hop_pid("devops", SID, pid=4321, task=TASKBOOK)
    assert _hop(takeover)["pid"] == 4321
    assert _hop(takeover)["pid_task"] == TASKBOOK

    registry.note_hop_pid("devops", SID, pid=0, task="")
    assert _hop(takeover)["pid"] == 0


def test_note_hop_pid_is_fail_open(takeover, monkeypatch, capsys):
    """记账面坏掉（registry 读不动）不得把派发拖下水——只是本跳失去第四面保护。"""

    def boom(*a, **k):
        raise ZeroDivisionError("registry 坏了")

    monkeypatch.setattr(registry.Registry, "load", staticmethod(boom))
    registry.note_hop_pid("devops", SID, pid=1, task=TASKBOOK)  # 不抛
    assert "Traceback" not in capsys.readouterr().err


def test_hop_pid_lives_exactly_as_long_as_the_hop(takeover):
    """**pid 标记的存活期 = 该跳的在途期**——所以回收不需要额外的收尾协议。

    复刻 dispatch 的真实时序：派发进程手里的 Member 是派生**之前**的快照，派生瞬间
    :func:`registry.note_hop_pid` 单独盖戳，跳结束时那份旧快照的 ``write_sessions``
    按 sid 整条覆盖（``_merge_sessions``「本进程条目为准」）→ 标记随跳自然消失。
    于是「看得见非零 pid」就等于「这跳没走完」，正是 worker 中途被杀时唯一的活口线索。
    """
    _seed_hop(takeover, pid=0, task="")
    stale = registry.Registry.load()  # 派发进程启动时的快照（不含在途 pid）
    registry.note_hop_pid("devops", SID, pid=FAKE_HOP_PID, task=TASKBOOK)
    assert _hop(takeover)["pid"] == FAKE_HOP_PID

    member = stale.member("devops")
    sessions.touch_session(member, SID, chars_offset=1234)
    stale.write_sessions(member)
    stale.save()
    assert _hop(takeover)["pid"] == 0, "收尾写要把在途标记一并带走"


# ---------------------------------------------------------------------------
# runner：派生瞬间交 pid + 超时会把子进程带走
# ---------------------------------------------------------------------------
def test_run_headless_reports_spawn_pid_and_timeout_kills(tmp_path):
    """钉住 subprocess.run → Popen 的重写：pid 在进程落地时就给出，超时仍要杀干净。"""
    seen: list[int] = []
    argv = [
        sys.executable,
        "-c",
        "import sys,time; sys.stdout.write('hi'); sys.stdout.flush(); time.sleep(60)",
    ]
    rc, out, err = runner.run_headless(
        argv,
        cwd=tmp_path,
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout_s=2,
        on_spawn=seen.append,
    )
    assert rc == -1 and err.startswith("TIMEOUT after 2s")
    assert out == "hi"
    assert len(seen) == 1 and seen[0] != os.getpid()
    assert not procs.pid_alive(seen[0]), "超时路径没把子进程带走"


def test_run_headless_without_on_spawn_still_works(tmp_path):
    """没传钩子时行为不变（其余调用方：orch 归档蒸馏、eval、probe）。"""
    rc, out, err = runner.run_headless(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONUTF8": "1"},
        timeout_s=60,
    )
    assert rc == 0 and out.strip() == "ok" and err == ""


# ---------------------------------------------------------------------------
# 身份闸：真进程、真命令行（mock 永远测不到 powershell/``/proc`` 那条读取路径）
# ---------------------------------------------------------------------------
def test_pid_is_session_reads_the_real_command_line():
    """安全边界的实测面：命令行里认得出那个 sid 才准动手，换一个 sid 必须认不出。

    回收侧的开火判定靠它，而 mock 里的 ``pid_is_session`` 只是被测函数自己写的假设。
    """
    argv = [sys.executable, "-c", "import time; time.sleep(30)", f"--session-id={SID}"]
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert procs.pid_alive(proc.pid)
        assert procs.pid_is_session(proc.pid, SID), "真命令行读不出 sid=身份闸是假闸"
        assert not procs.pid_is_session(
            proc.pid, "99999999-8888-7777-6666-555544443333"
        ), "换一个 sid 也算数就会误杀别人的会话"
        assert procs.terminate_tree(proc.pid)[0]
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)


# ---------------------------------------------------------------------------
# dispatch：一跳的 pid 在派生瞬间进运行态
# ---------------------------------------------------------------------------
def _fake_spawn(monkeypatch) -> list:
    """成功收尾的假 headless：把 on_spawn 用起来（派生瞬间回报 FAKE_HOP_PID）。"""
    calls: list[list[str]] = []

    def run(cmd, cwd=None, env=None, timeout_s=None, on_spawn=None):
        calls.append(cmd)
        if on_spawn is not None:
            on_spawn(FAKE_HOP_PID)
        return envelope("<result>done</result>", is_error=False, rc=0)

    monkeypatch.setattr(dispatch, "run_headless", run)
    return calls


def test_dispatch_notes_pid_at_spawn(iso_dispatch, monkeypatch):
    notes: list[tuple] = []
    monkeypatch.setattr(
        dispatch,
        "note_hop_pid",
        lambda member_id, sid, *, pid, task: notes.append((member_id, sid, pid, task)),
    )
    calls = _fake_spawn(monkeypatch)
    outcome = dispatch.do_dispatch("quotamember", text="做某事", quiet=True)

    assert outcome.code == 0, outcome.error
    assert calls, "假 headless 没被调用"
    assert notes and notes[0][0] == "quotamember"
    assert notes[0][2] == FAKE_HOP_PID, "派生瞬间没把 pid 交给记账面"
    assert notes[0][3].startswith("task-") and notes[0][3].endswith(".md")


# ---------------------------------------------------------------------------
# 第四面回收：开火与不开火
# ---------------------------------------------------------------------------
def test_reclaim_terminates_live_writer_and_clears_ledger(takeover, monkeypatch):
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    killed: list[int] = []
    state = {"alive": True}

    def fake_terminate(pid):
        killed.append(pid)
        state["alive"] = False
        return True, "taskkill ok"

    monkeypatch.setattr(drain, "_pid_alive", lambda pid: state["alive"])
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(procs, "terminate_tree", fake_terminate)

    hot = drain.reclaim_orphan_sessions([OID], None, sleep_s=0)

    assert killed == [FAKE_HOP_PID], "该开的火没开"
    assert hot == [], "已确认离场却仍被列进异名重建"
    assert _hop(takeover)["pid"] == 0, "处置完还留陈记账（下次对着复用的 pid 误动手）"


def test_reclaim_logs_the_fourth_face(takeover, monkeypatch):
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(procs, "terminate_tree", lambda pid: (True, "ok"))
    log = drain.new_run_log()

    drain.reclaim_orphan_sessions([OID], log, sleep_s=0)

    body = log.read_text(encoding="utf-8")
    assert "第四面回收：终止孤儿会话进程" in body
    assert f"pid={FAKE_HOP_PID}" in body and OID in body


def test_reclaim_skips_when_pid_no_longer_the_session(takeover, monkeypatch):
    """pid 被系统复用 → **绝不开火**（误杀别人的作业比双写更糟），条目转异名重建。"""
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    killed: list[int] = []
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: False)
    monkeypatch.setattr(
        procs, "terminate_tree", lambda pid: killed.append(pid) or (True, "")
    )

    hot = drain.reclaim_orphan_sessions([OID], None, sleep_s=0)

    assert killed == [], "身份核对失败还开了火"
    assert hot == [OID]
    assert _hop(takeover)["pid"] == 0


def test_reclaim_hot_when_terminate_leaves_it_alive(takeover, monkeypatch):
    """发了终止但进程还挂着（孙进程/内核拖尾）→ 判「没干净」，条目进异名重建。"""
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    killed: list[int] = []
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)  # 杀不动
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(
        procs, "terminate_tree", lambda pid: killed.append(pid) or (True, "ok")
    )

    hot = drain.reclaim_orphan_sessions([OID], None, sleep_s=0)

    assert killed == [FAKE_HOP_PID]
    assert hot == [OID]


def test_reclaim_dead_pid_only_erases_stale_ledger(takeover, monkeypatch):
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    killed: list[int] = []
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(
        procs, "terminate_tree", lambda pid: killed.append(pid) or (True, "")
    )

    hot = drain.reclaim_orphan_sessions([OID], None, sleep_s=0)

    assert killed == [] and hot == []
    assert _hop(takeover)["pid"] == 0


def test_reclaim_ignores_sessions_of_other_entries(takeover, monkeypatch):
    """归属只认任务书里嵌的那个条目 id——别的在途跳（别的单）一律不碰。"""
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover, task="task-20261012-185839-20991231-other-leader.md")
    fired: list[int] = []
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(
        procs, "terminate_tree", lambda pid: fired.append(pid) or (True, "")
    )

    assert drain.reclaim_orphan_sessions([OID], None, sleep_s=0) == []
    assert fired == []
    assert _hop(takeover)["pid"] == FAKE_HOP_PID, "误清了别人一跳的记账"


# ---------------------------------------------------------------------------
# 异名重建：世代名进任务书
# ---------------------------------------------------------------------------
def test_worktree_name_defaults_to_bare_id():
    """第一代仍用裸条目 id（既有分支名 ``wt-<id>`` 纪律不变），只有升过的代才带后缀。"""
    assert workflow.worktree_name({"id": OID}) == OID
    assert workflow.worktree_name({"id": OID, "worktree_gen": 1}) == OID
    assert workflow.worktree_name({"id": OID, "worktree_gen": 4}) == f"{OID}-r4"


def test_generation_name_flows_into_task_text(takeover):
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    orphans = workflow.backlog_reclaim_orphans()
    assert orphans == [OID]

    assert workflow.bump_worktree_generation(orphans) == {OID: f"{OID}-r2"}

    seed, group = workflow.backlog_take_batch()
    assert seed["worktree"] == f"{OID}-r2", "新一代名没进条目，重派还会撞旧面"
    text = workflow.backlog_task_text(seed)
    assert f"pysci-dev worktree add {OID}-r2" in text
    assert f"wt-{OID}-r2" in text
    assert "第 2 代隔离面" in text
    assert not group


def test_generation_advances_from_disk_not_counter(takeover):
    """世代号读磁盘残骸推进：``-r2`` 还趴在树上时，下一批必须跳到 ``-r3``。"""
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    (takeover.wt_root / f"{OID}-r2").mkdir()

    assert workflow.bump_worktree_generation([OID]) == {OID: f"{OID}-r3"}
    on_disk = read_backlog(takeover.state)[0]
    assert on_disk["worktree_gen"] == 3
    assert "禁止复用旧名" in on_disk["note"]


def test_generation_skips_unlisted_items(takeover):
    seed_backlog(
        takeover.state,
        [ORPHAN_ITEM, {"id": "other", "status": "pending", "member": "devops"}],
    )
    assert workflow.bump_worktree_generation([OID]) == {OID: f"{OID}-r2"}
    assert read_backlog(takeover.state)[1].get("worktree_gen") is None


# ---------------------------------------------------------------------------
# drain 接线：四面顺序 + dry 不开火
# ---------------------------------------------------------------------------
class _FakeDispatch:
    """记下派发的任务书文本（断言隔离面名进了任务书），交付=本批全部完成。"""

    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, member_id, **kwargs):
        self.texts.append(kwargs.get("text", ""))
        return DispatchOutcome(
            code=0,
            kind="result",
            body=f"<result>完成 backlog id={kwargs.get('slug')}</result>",
        )


def test_drain_takeover_kills_then_reuses_name(takeover, monkeypatch):
    """能确认干净 → 原代名复用（不无故改单号），并留下第四面的日志痕迹。"""
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    killed: list[int] = []
    state = {"alive": True}

    def _kill(pid: int) -> tuple[bool, str]:
        killed.append(pid)
        state["alive"] = False
        return True, "成功终止 1 个进程"

    monkeypatch.setattr(drain, "_pid_alive", lambda pid: state["alive"])
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(procs, "terminate_tree", _kill)
    monkeypatch.setattr(drain, "KILL_GRACE_S", 0)  # 免真等复查间隔
    git = GitFaces(takeover.wt_root).patch(monkeypatch)
    git.face(OID)  # 干净面：无未提交、不领先 main
    fake = _FakeDispatch()
    monkeypatch.setattr(dispatch, "do_dispatch", fake)

    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)

    assert summary["reclaimed"] == [OID]
    assert killed == [FAKE_HOP_PID]
    assert "worktrees_regenerated" not in summary, "已确认干净却改了单号"
    assert summary["worktrees_cleaned"] == [OID] and git.commits == []
    assert _hop(takeover)["pid"] == 0, "处置完不留陈在途标记"
    text = run_log.read_text(encoding="utf-8")
    assert "终止孤儿会话进程" in text and "异名重建" not in text
    assert f"worktree add {OID}`" in fake.texts[0]


def test_drain_takeover_rebuilds_when_writer_wont_die(takeover, monkeypatch):
    """杀不掉 → 先升代再派：新一代隔离面进任务书，绝不与旧写者同名并发。"""
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    killed: list[int] = []
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)  # 一直活着
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(
        procs, "terminate_tree", lambda pid: killed.append(pid) or (True, "ok")
    )
    git = GitFaces(takeover.wt_root).patch(monkeypatch)
    git.face(OID)
    monkeypatch.setattr(drain, "KILL_GRACE_S", 0)
    fake = _FakeDispatch()
    monkeypatch.setattr(dispatch, "do_dispatch", fake)

    summary = drain.drain_backlog(drain.new_run_log())

    assert summary["worktrees_regenerated"] == {OID: f"{OID}-r2"}
    assert git.commits == [], "干净面隔离中也不该被打成抢救提交"
    assert f"worktree add {OID}-r2`" in fake.texts[0]
    assert read_backlog(takeover.state)[0]["worktree"] == f"{OID}-r2"
    assert _hop(takeover)["pid"] == 0


def test_drain_takeover_rebuilds_on_lingering_dir(takeover, monkeypatch):
    """remove 报 0 但目录还在（旧句柄不放）→ 判「没干净」开新一代名，**但进度先保住**。

    旧契约在这里销毁整个面（backlog 20261012-234000-leader 要消灭的那一半）：新契约里
    残骸目录不进去写（git 读数不可信），分支与提交被记进条目的抢救面，续作任务书直接
    给出 ``cherry-pick`` 取回方式——升代与保进度是两件事，过去被混成了「删」。
    """
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    (takeover.wt_root / OID).mkdir()  # 残骸目录
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    git = GitFaces(takeover.wt_root).patch(monkeypatch)
    git.face(OID, registered=False, lingering=True, unmerged=2, head="c0ffee12")
    fake = _FakeDispatch()
    monkeypatch.setattr(dispatch, "do_dispatch", fake)

    summary = drain.drain_backlog(drain.new_run_log())

    assert summary["worktrees_regenerated"] == {OID: f"{OID}-r2"}
    assert summary["worktrees_salvaged"] == [OID], "升代不等于销毁：读数得交回队列"
    assert not [c for c in git.calls if c[:1] == ("-C",)], "残骸目录不进去写"
    assert ("branch", "-D", f"wt-{OID}") not in git.calls, "未合并的分支就是进度本体"
    assert "cherry-pick --no-commit c0ffee12" in fake.texts[0]
    assert read_backlog(takeover.state)[0]["salvage"]["commit"] == "c0ffee12"


def test_drain_dry_does_not_fire_fourth_face(takeover, monkeypatch):
    seed_backlog(takeover.state, [ORPHAN_ITEM])
    _seed_hop(takeover)
    fired: list[int] = []
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(procs, "pid_is_session", lambda pid, sid: True)
    monkeypatch.setattr(
        procs, "terminate_tree", lambda pid: fired.append(pid) or (True, "")
    )
    git_calls: list[tuple] = []
    monkeypatch.setattr(drain, "_run_git", lambda *a: git_calls.append(a) or (0, ""))

    summary = drain.drain_backlog(drain.new_run_log(), dry=True)

    assert summary["reclaimed"] == [OID]
    assert fired == [], "dry 演练不该杀进程"
    assert git_calls == [], "dry 演练不该动 git"
    assert "worktrees_regenerated" not in summary
    assert _hop(takeover)["pid"] == FAKE_HOP_PID, "演练不该改运行态"


def test_worktree_dir_present_empty_name(takeover):
    assert drain.worktree_dir_present("") is False
    assert drain.worktree_dir_present("nope") is False
    assert isinstance(drain.KILL_GRACE_S, float)
    # drain 的判据确实读同一棵树（patch 生效=它没有绕回真实项目根）
    assert workflow.WORKTREES_ROOT == takeover.wt_root
    (takeover.wt_root / "seen").mkdir()
    assert drain.worktree_dir_present("seen") is True
