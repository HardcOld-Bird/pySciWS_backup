"""锁释放失败不留黑洞（README §6 锁面；backlog 20261011-225933-devops + 20261011-231237-devops）。

旧实现 `release_lock` 把 ``os.unlink`` 的 OSError 吞成 ``pass``：Windows 上他方仍持该
文件句柄时 DeleteFile 报共享冲突，锁原地留着自称「pid 活着」，``_lock_live`` 遂把残骸
判为活锁 → 下一任 drain ``skipped=lock_busy``，最长阻塞到 ``LOCK_STALE_S``（6 h）。
本文件锁死新语义：**失败不静默**（run 日志 ``[!]``）+ **失败不挡路**（released 墓碑被
``lock_status`` 判非 running、被 ``acquire_lock`` 就地接管）+ **墓碑也写不进时仍有第二条
收尾证据**（该 run 的 ``.done`` 存在即判非活，backlog 20261011-231237-devops）。
"""

from __future__ import annotations

import json
import os

from pysci.skills.orchestration.tools import drain, orch

from .conftest import seed_backlog

#: 真实共享冲突的形状（Windows：文件被他方读句柄占用，DeleteFile 报 EACCES/32）。
SHARING = OSError(
    13, "The process cannot access the file because it is being used by another process"
)


def _unlink_fails(monkeypatch):
    """让一切 unlink 失败——正是要复现「删不掉的锁」。"""

    def boom(path, *a, **kw):
        raise SHARING

    monkeypatch.setattr(os, "unlink", boom)


def _writes_blockable(monkeypatch) -> dict:
    """让 ``_overwrite_lock`` 可按开关拒绝写入（复现「连墓碑都写不进」）。"""
    gate = {"blocked": False}
    real = drain._overwrite_lock

    def gated(payload: str) -> bool:
        return False if gate["blocked"] else real(payload)

    monkeypatch.setattr(drain, "_overwrite_lock", gated)
    return gate


def _seed_done_for(run_log, *, declared=None, **body):
    """落该 run 的 ``.done`` 收尾摘要；``declared`` 可伪造摘要自证的 run_log 身份。"""
    done = run_log.with_suffix(".done")
    done.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started": "2026-10-11T23:00:00+08:00",
        "completed": "2026-10-11T23:30:00+08:00",
        "count": 1,
        "elapsed_s": 60.0,
        "items": [],
        "run_log": str(run_log) if declared is None else declared,
        **body,
    }
    done.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return done


def _seed_tombstone(state, *, pid, run_log="orchestration/state/devops-runs/r.log"):
    state.mkdir(parents=True, exist_ok=True)
    (state / "devops.lock").write_text(
        json.dumps(
            {
                "released": True,
                "pid": pid,
                "started": drain._now(),
                "run_log": run_log,
                "released_at": drain._now(),
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 阳性三断言（任务书修复面 3）
# ---------------------------------------------------------------------------
def test_unlink_failure_marks_tombstone_and_leaves_trace(iso_state, monkeypatch):
    rec = drain.acquire_lock("orchestration/state/devops-runs/r1.log")
    assert rec is not None
    run_log = iso_state / "devops-runs" / "r1.log"
    _unlink_fails(monkeypatch)

    assert drain.release_lock(run_log) is False

    # (a) run 日志必须有一行 [!]，写明失败原因与已写标记
    text = run_log.read_text(encoding="utf-8")
    assert "[!] 锁释放失败（共享句柄），已写 released 标记" in text
    assert "another process" in text

    # 锁记录被改写为墓碑，且保留溯源字段
    data = drain.read_lock()
    assert data["released"] is True
    assert data["pid"] == rec["pid"] == os.getpid()
    assert data["run_log"] == rec["run_log"]
    assert data["released_at"]


def test_residue_is_not_reported_as_running(iso_state, monkeypatch):
    """(b) 锁面不再骗人：``lock_status`` 不报 running，也不报崩溃。"""
    rec = drain.acquire_lock("orchestration/state/devops-runs/r2.log")
    assert rec is not None
    _unlink_fails(monkeypatch)
    drain.release_lock(iso_state / "devops-runs" / "r2.log")

    st = drain.lock_status()
    assert st["state"] != "running"
    assert st["state"] == "stale"
    assert st["released"] is True
    # 崩溃者走不到 release_lock，故墓碑绝非 crashed（防 watch 误报 CRASH）
    assert st["crashed"] is False


def test_next_drain_takes_over_undeletable_residue(iso_state, monkeypatch):
    """(c) 残骸挡不住下一任：就地改写接管，且接管后不再是墓碑。"""
    _seed_tombstone(iso_state, pid=4242)
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)  # 墓碑 pid 恰好「活着」
    assert drain._lock_live(drain.read_lock()) is False

    rec = drain.acquire_lock("orchestration/state/devops-runs/r3.log")
    assert rec is not None and rec["pid"] == os.getpid()
    assert not drain.read_lock().get("released")
    assert drain.lock_status()["state"] == "running"


def test_full_drain_twice_not_blocked_by_residue(iso_state, monkeypatch):
    """端到端：一次 drain 收尾删不掉锁，下一次仍照常消化（不 ``skipped=lock_busy``）。"""
    _unlink_fails(monkeypatch)  # 全程删不动任何文件，锁只能靠墓碑 + 就地接管自愈
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s1"}])
    first = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert first["count"] == 1

    # 首个 run 结束后锁成了删不掉的残骸，但不再自称在跑
    assert drain.LOCK_PATH.exists()
    assert drain.read_lock()["released"] is True
    assert drain.lock_status()["state"] != "running"

    seed_backlog(iso_state, [{"id": "b2", "status": "pending", "summary": "s2"}])
    second = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert second.get("skipped") != "lock_busy"
    assert second["count"] == 1


# ---------------------------------------------------------------------------
# 旧路径：unlink 成功时一切照旧（无标记、无噪音）
# ---------------------------------------------------------------------------
def test_unlink_success_still_removes_lock_quietly(iso_state):
    drain.acquire_lock("orchestration/state/devops-runs/ok.log")
    run_log = iso_state / "devops-runs" / "ok.log"

    assert drain.release_lock(run_log) is True
    assert not drain.LOCK_PATH.exists()
    assert drain.lock_status()["state"] == "idle"
    text = run_log.read_text(encoding="utf-8") if run_log.exists() else ""
    assert "[!]" not in text  # 成功路径一行噪音也不留


def test_non_owner_release_never_touchs_foreign_residue(iso_state, monkeypatch):
    """不是自己的锁就一律不改（含别人的墓碑）——清除归下一任 acquire。"""
    _seed_tombstone(iso_state, pid=999999)
    _unlink_fails(monkeypatch)
    before = drain.read_lock()

    assert drain.release_lock(iso_state / "devops-runs" / "x.log") is False
    assert drain.read_lock() == before


def test_no_run_log_still_marks_but_stays_silent(iso_state, monkeypatch):
    """无 run 日志可写时也必须打标——留痕是尽力，标记是硬要求。"""
    drain.acquire_lock("orchestration/state/devops-runs/s.log")
    _unlink_fails(monkeypatch)

    assert drain.release_lock() is False
    assert drain.read_lock()["released"] is True
    assert drain.lock_status()["state"] != "running"


def test_marker_write_failure_is_also_logged(iso_state, monkeypatch):
    """连墓碑都写不进的极端情形仍不留黑洞：日志说明「标记亦写入失败」。"""
    drain.acquire_lock("orchestration/state/devops-runs/t.log")
    run_log = iso_state / "devops-runs" / "t.log"
    _unlink_fails(monkeypatch)
    monkeypatch.setattr(drain, "_overwrite_lock", lambda payload: False)

    assert drain.release_lock(run_log) is False
    assert "released 标记亦写入失败" in run_log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 观察面：orch status 呈现「残留已标记」
# ---------------------------------------------------------------------------
def test_orch_status_shows_marked_residue(iso_state, iso_registry, capsys):
    _seed_tombstone(iso_state, pid=4242)
    assert drain.lock_status()["released"] is True

    orch.cmd_status(None)
    out = capsys.readouterr().out
    assert "残留已标记" in out
    assert "▶ running" not in out
    assert "⊘ 崩溃" not in out


# ---------------------------------------------------------------------------
# 第二道收尾证据：墓碑也写不进时，该 run 的 .done 判非活
# （backlog 20261011-231237-devops——E3 未竟格）
# ---------------------------------------------------------------------------
def test_undeletable_unwritable_residue_is_vetoed_by_done(iso_state, monkeypatch):
    """阳性：unlink 与墓碑写入双双失败 + 该 run 已有 ``.done`` → 不报 running，且可接管。"""
    gate = _writes_blockable(monkeypatch)
    run_log = drain.new_run_log()
    assert drain.acquire_lock(str(run_log)) is not None
    _seed_done_for(run_log)
    _unlink_fails(monkeypatch)
    gate["blocked"] = True  # 他方连写都不共享：墓碑写不进去

    assert drain.release_lock(run_log) is False
    data = drain.read_lock()
    assert not data.get("released")  # 确实没标上记号——只剩 .done 这条证据
    st = drain.lock_status()
    assert st["state"] != "running"
    assert st["residue"] == "done-exists" and st["crashed"] is False
    # 判活直接否决：pid 就是本进程（活着）也不认
    assert drain._pid_alive(os.getpid()) and drain._lock_live(data) is False
    # 组长侧据此判断的行为面：不再以为有 worker 在跑
    assert drain.request_stop()[0] is False

    gate["blocked"] = False  # 句柄散了：写能做了
    next_log = (
        drain.DEVOPS_RUNS_DIR / "next-run.log"
    )  # 另起名（同秒撞名会被 .done 否决）
    rec = drain.acquire_lock(str(next_log))
    assert rec is not None and rec["pid"] == os.getpid()
    assert drain.lock_status()["state"] == "running"


def test_drain_after_unwritable_residue_still_digests(iso_state, monkeypatch):
    """端到端（.done 否决版）：上一任留下「删不掉 + 标不上」的残骸，下一任照常消化。"""
    gate = _writes_blockable(monkeypatch)
    _unlink_fails(monkeypatch)
    seed_backlog(iso_state, [{"id": "b1", "status": "pending", "summary": "s1"}])
    gate["blocked"] = True
    first = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert first["count"] == 1
    assert drain.LOCK_PATH.exists() and not drain.read_lock().get("released")
    assert drain.lock_status()["residue"] == "done-exists"

    seed_backlog(iso_state, [{"id": "b2", "status": "pending", "summary": "s2"}])
    gate["blocked"] = False
    second = drain.drain_backlog(drain.new_run_log(), dry=True)
    assert second.get("skipped") != "lock_busy"
    assert second["count"] == 1


def test_live_run_without_done_is_not_stolen(iso_state, monkeypatch):
    """反面对照：还在跑的 run（有 .log 无该 run 的 .done）判活为真——别把收尾否决做成万能抢锁。"""
    gate = _writes_blockable(monkeypatch)
    run_log = drain.new_run_log()
    assert drain.acquire_lock(str(run_log)) is not None
    _seed_done_for(drain.DEVOPS_RUNS_DIR / "some-earlier-run.log")  # 别的 run 的摘要
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)

    data = drain.read_lock()
    assert drain._lock_live(data) is True
    st = drain.lock_status()
    assert st["state"] == "running" and "residue" not in st
    assert drain.acquire_lock(str(drain.new_run_log())) is None  # 互斥照旧
    gate["blocked"] = True
    _unlink_fails(monkeypatch)
    assert (
        drain.release_lock(run_log) is False
    )  # 收尾了却没删没标——由 done 证据接管下一任


def test_done_of_other_run_is_not_proof_of_conclusion(iso_state, monkeypatch):
    """摘要自证身份不符（同名撞车：别的 run 的摘要落到了这个文件名）→ 不作收尾证据。"""
    run_log = drain.new_run_log()
    assert drain.acquire_lock(str(run_log)) is not None
    _seed_done_for(run_log, declared="orchestration/state/devops-runs/other-run.log")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)

    assert drain._run_concluded(drain.read_lock()) is False
    assert drain.lock_status()["state"] == "running"
    assert drain.acquire_lock(str(drain.new_run_log())) is None


def test_half_written_done_still_counts_as_conclusion(iso_state, monkeypatch):
    """.done 半写（读不出 JSON）：文件已在即已收尾——它在 release_lock 之后才可能出现。"""
    run_log = drain.new_run_log()
    assert drain.acquire_lock(str(run_log)) is not None
    half = run_log.with_suffix(".done")
    half.parent.mkdir(parents=True, exist_ok=True)
    half.write_text('{"completed": "2026', encoding="utf-8")
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)

    assert drain._run_concluded(drain.read_lock()) is True
    assert drain.lock_status()["state"] != "running"


def test_orch_status_shows_concluded_residue(
    iso_state, iso_registry, monkeypatch, capsys
):
    """观察面：这种残骸报「⊘ 残骸已收尾」，不得读成 running，也不得误读成 pid 已死/崩溃。"""
    gate = _writes_blockable(monkeypatch)
    run_log = drain.new_run_log()
    assert drain.acquire_lock(str(run_log)) is not None
    _seed_done_for(run_log)
    _unlink_fails(monkeypatch)
    gate["blocked"] = True
    drain.release_lock(run_log)

    orch.cmd_status(None)
    out = capsys.readouterr().out
    assert "残骸已收尾" in out
    assert "▶ running" not in out and "⊘ 崩溃" not in out and "已死/超龄" not in out
