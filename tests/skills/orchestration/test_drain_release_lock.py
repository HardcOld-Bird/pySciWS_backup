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
from datetime import datetime, timedelta

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

    def gated(payload: str, paths=None) -> bool:
        return False if gate["blocked"] else real(payload, paths)

    monkeypatch.setattr(drain, "_overwrite_lock", gated)
    return gate


def _seed_done_for(run_log, *, declared=None, **body):
    """落该 run 的 ``.done`` 收尾摘要；``declared`` 可伪造摘要自证的 run_log 身份。

    时刻默认取「现在」而非历史字面量：判活的时序闸（:func:`drain._done_predates_lock`）
    要求摘要**不早于**本锁诞生才算它自己的收尾，而夹具里的锁 ``started`` 就是现在——
    写死旧时刻会让用例因为「摘要属于上一任」而被放行，测不到想测的那条否决。
    """
    done = run_log.with_suffix(".done")
    done.parent.mkdir(parents=True, exist_ok=True)
    now = drain._now()
    payload = {
        "started": now,
        "completed": now,
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
    monkeypatch.setattr(drain, "_overwrite_lock", lambda payload, paths=None: False)

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


# ---------------------------------------------------------------------------
# run 身份命名查重 + 收尾证据的时序闸（backlog 20261011-233248-devops）
#
# E4 把「该 run 已有 ``.done``」请进判活后，``.done`` 的**文件名**成了否决依据，而旧
# ``new_run_log()`` 只精确到秒且不查重：空队列 run 在同一秒内写出 ``.done``，紧随其后
# 的 drain 拿到**同名** run_log —— 它的锁新鲜、pid 存活、正在干活，却被上一任的摘要判成
# 已收尾残骸：``lock_status`` 谎报 ``stale/done-exists``，``acquire_lock`` 对它放行 →
# 两个 worker 同时自认持锁，README §6 单 worker 互斥击穿。
# ---------------------------------------------------------------------------
def _past(seconds: int) -> str:
    """N 秒前的 ISO 时刻（带本地时区，与 :func:`drain._now` 同形）。"""
    return (datetime.now().astimezone() - timedelta(seconds=seconds)).isoformat(
        timespec="seconds"
    )


#: 冻结用的「现在」：今天 09:00:00（带本机时区，与 :func:`drain._now` 同形）。同秒撞名
#: 场景必须消除时钟漂移，否则用例会在跨秒那一瞬假失败。
FROZEN = datetime.now().astimezone().replace(hour=9, minute=0, second=0, microsecond=0)
FROZEN_STAMP = FROZEN.strftime("%Y%m%d-%H%M%S")


def _freeze_clock(monkeypatch) -> None:
    """把 :mod:`drain` 看到的 ``datetime`` 换成 ``now()`` 恒为 :data:`FROZEN` 的子类。

    只换 ``now``，``fromisoformat`` 等原样继承——时序闸的比较逻辑仍走真实实现。
    """

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return FROZEN if tz is None else FROZEN.astimezone(tz)

    monkeypatch.setattr(drain, "datetime", _Fixed)


def _sep(n: int) -> str:
    """第 n 个顺延名的后缀串（与 :func:`drain.new_run_log` 的构造保持一致）。"""
    return f"{drain.RUN_LOG_SEP}{n:0{drain.RUN_LOG_PAD}}"


def test_same_second_new_run_log_never_reuses_name(iso_state, monkeypatch):
    """修复面 3-①：同一秒内连调 ``new_run_log()`` 五次必得五个异名。"""
    _freeze_clock(monkeypatch)
    names = [drain.new_run_log() for _ in range(5)]
    assert [p.stem for p in names] == [FROZEN_STAMP] + [
        f"{FROZEN_STAMP}{_sep(n)}" for n in range(2, 6)
    ], f"同秒顺延形状不符：{[p.name for p in names]}"


def test_dedup_counts_occupied_stems_including_done(iso_state, monkeypatch):
    """顺延只看「名字被占」：``<ts>.log`` 或 ``<ts>.done`` 任一存在即让位。

    ``.done`` 存在而 ``.log`` 从未写过正是真实形状（上一任只写了摘要，日志被清理或
    从未落盘），而**摘要侧才是否决依据**，故查重必须同时查这两个后缀。
    """
    _freeze_clock(monkeypatch)
    runs = iso_state / "devops-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{FROZEN_STAMP}.done").write_text("{}", encoding="utf-8")

    got = drain.new_run_log()

    assert got == runs / f"{FROZEN_STAMP}{_sep(2)}.log"
    got.touch()
    assert drain.new_run_log() == runs / f"{FROZEN_STAMP}{_sep(3)}.log"


def test_dedup_names_sort_after_the_occupied_done(iso_state, monkeypatch):
    """排序面：顺延名必须**晚于**被占的 ``<ts>.done``，否则 ``latest_done`` 把旧摘要当最新。

    这正是分隔符不能取 ``-`` 的原因（``-`` 码位 0x2D 低于 ``.`` 的 0x2E）。
    """
    _freeze_clock(monkeypatch)
    runs = iso_state / "devops-runs"
    runs.mkdir(parents=True, exist_ok=True)
    old = runs / f"{FROZEN_STAMP}.done"
    old.write_text(json.dumps({"completed": FROZEN.isoformat(), "count": 1}), "utf-8")
    fresh = drain.new_run_log()
    assert "-" not in fresh.stem[len(FROZEN_STAMP) :], fresh.name
    fresh.with_suffix(".done").write_text(
        json.dumps({"completed": FROZEN.isoformat(), "count": 2}), "utf-8"
    )

    assert drain.latest_done()["name"] == fresh.stem
    assert drain.latest_done()["count"] == 2


def test_generated_name_never_collides_with_written_pair(iso_state):
    """生成的名字当场占位：拿它 ``acquire_lock`` 后不会因前任摘要被否决（互斥不破）。"""
    first = drain.new_run_log()
    assert first.exists(), "new_run_log 必须当场占位，否则同秒连调仍会同名"
    assert drain.acquire_lock(str(first)) is not None
    # 模拟第一个 run 干净收尾：写出 <ts>.done 且释放锁
    _seed_done_for(first)
    assert drain.release_lock(first) is True

    second = drain.new_run_log()
    assert second != first
    assert drain.acquire_lock(str(second)) is not None
    st = drain.lock_status()
    assert st["state"] == "running", f"第二个 run 的活锁被误判残骸：{st}"
    assert st["pid"] == os.getpid()
    # 第三个 worker 必须被挡在门外（README §6）
    assert drain.acquire_lock(str(drain.new_run_log())) is None


def test_live_lock_with_same_named_old_done_still_running(iso_state, monkeypatch):
    """repro 钉成回归用例：先落 ``<ts>.done``（上一任的尾），再以**同名**生成锁。

    这是「绕过命名护栏」的极端形状（手工/异常路径拿到同名 run_log）。时序闸必须单独
    顶住：摘要早于本锁诞生 → 不是本锁的收尾 → 活锁照判 ``running``、``acquire_lock``
    照拒抢。``_pid_alive`` 桩成真值，复现 pid 回收/保守判活的谎话面。
    """
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    run_log = drain.new_run_log()
    _seed_done_for(run_log, completed=_past(3600))  # 同名，但收的是 1 h 前那一任的尾

    assert drain.acquire_lock(str(run_log)) is not None
    assert drain._run_concluded(drain.read_lock()) is False
    st = drain.lock_status()
    assert st["state"] == "running" and "residue" not in st, st
    assert drain.acquire_lock(str(drain.new_run_log())) is None


def test_same_second_self_conclusion_still_vetoed(iso_state, monkeypatch):
    """对照：同秒空跑的自收尾合法——闸取「严格早于」，不得把 E4 那条救援挡回来。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: True)
    now = drain._now()
    run_log = drain.new_run_log()
    _seed_done_for(run_log, started=now, completed=now)

    assert drain.acquire_lock(str(run_log)) is not None

    assert drain._run_concluded(drain.read_lock()) is True
    st = drain.lock_status()
    assert st["state"] == "stale" and st["residue"] == "done-exists"


def test_done_predates_lock_time_gate_unit():
    """:func:`drain._done_predates_lock` 的边界：严格早于否决；同刻/更晚/读不出都放行。"""
    started = "2026-10-12T09:00:00+08:00"
    lock = {"started": started}
    assert drain._done_predates_lock({"completed": "2026-10-12T08:59:59+08:00"}, lock)
    assert not drain._done_predates_lock({"completed": started}, lock)
    assert not drain._done_predates_lock(
        {"completed": "2026-10-12T09:00:01+08:00"}, lock
    )
    # 不确定就不否决：缺字段、非 ISO、锁侧读不出、时区一侧缺失（TypeError 路径）
    for body in ({}, {"completed": ""}, {"completed": "T1"}, "not-a-dict"):
        assert drain._done_predates_lock(body, lock) is False, body
    naive = {"completed": "2026-10-12T08:00:00"}  # 无时区 → 不可比 → 放行
    assert drain._done_predates_lock(naive, lock) is False
    no_started = {"completed": "2026-10-12T08:00:00+08:00"}
    assert (
        drain._done_predates_lock(no_started, {}) is False
    )  # 锁侧无 started → 无从比较
