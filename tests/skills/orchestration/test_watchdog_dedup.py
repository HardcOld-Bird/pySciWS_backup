"""看门狗**双生/多生**防护（2026-10-15 双生单）。

现场（组长 2026-10-15 18:01–18:04 实测）：`orch watchdog start` 波形里出现两个同 argv 的
``_watchdog-loop`` 进程、共存 ≥4 分钟。逐条钉死本单的四项要求：

1. 拉起原子化——并发 start×5（同秒）→ 至多派生 1 个（无标记时 5 个都通过幂等判定各派生）；
2. run_log 唯一化——同秒多次生成 → 文件名互不相同（命名面已有 O_EXCL 占位，此处钉死）；
3. 双实例自清——同参/不同参第二个 ``run_watchdog`` 秒退 ``lock_busy`` **且留痕**；
4. stop 广播纪元对不持锁孤儿的清退（现有语义保留）——见 ``test_watchdog_stop.py``。
"""

from __future__ import annotations

import json
import os
import threading

from pysci.skills.orchestration.tools import drain, watchdog


def _paths(iso_state, monkeypatch) -> drain.WorkerPaths:
    """watchdog 生命周期面落 iso_state（与 test_watchdog_stop 同构）。"""
    monkeypatch.setattr(
        watchdog,
        "WATCHDOG_PATHS",
        drain.WorkerPaths(
            lock=iso_state / "watchdog.lock",
            runs=iso_state / "watchdog-runs",
            cancel=iso_state / "watchdog.cancel",
        ),
    )
    return watchdog.WATCHDOG_PATHS


def _fake_spawn(paths: drain.WorkerPaths, spawned: list[list[str]]):
    """假派生：记录 argv，并像真看门狗那样**上锁**（子进程起来后第一件事）。"""

    def _spawn(argv, **_kw):
        spawned.append(list(argv))
        paths.lock.parent.mkdir(parents=True, exist_ok=True)
        paths.lock.write_text(
            json.dumps(
                {"pid": os.getpid(), "started": drain._now(), "run_log": "wd.log"}
            ),
            encoding="utf-8",
        )

    return _spawn


def test_concurrent_start_spawns_at_most_one(iso_state, monkeypatch):
    """并发 start×5（同秒）→ 恰 1 次派生，其余如实回报「已在跑」。

    这正是现场双生的入口：旧实现「查在跑 → 派生」无互斥，5 个调用可在子进程上锁前
    全部看到「不在跑」而各派生一个。
    """
    paths = _paths(iso_state, monkeypatch)
    spawned: list[list[str]] = []
    monkeypatch.setattr(watchdog, "_spawn_detached", _fake_spawn(paths, spawned))

    n = 5
    barrier = threading.Barrier(n)
    results: list[tuple[bool, str]] = []
    lock_guard = threading.Lock()

    def _start(_i: int) -> None:
        barrier.wait()
        out = watchdog.start_watchdog()
        with lock_guard:
            results.append(out)

    ts = [threading.Thread(target=_start, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)

    assert len(results) == n, "有 start 调用未返回（标记不可永久挡住后来者）"
    assert len(spawned) == 1, f"并发 start 派生次数应为 1，实测 {len(spawned)}"
    assert sum(1 for ok, _ in results if ok) == 1, "恰一个调用应报告「已派生」"
    # 派生用过的 run_log 至多一份（派生一次 ⇒ 一份日志），且标记已归还
    assert len(list(paths.runs.glob("*.log"))) == 1
    assert not watchdog._spawn_marker().exists(), "派生标记须在收尾归还"


def test_start_is_noop_when_running(iso_state, monkeypatch):
    """已在跑（锁活）→ 不派生、标记归还（幂等路径不吃标记）。"""
    paths = _paths(iso_state, monkeypatch)
    paths.runs.mkdir(parents=True, exist_ok=True)
    paths.lock.write_text(
        json.dumps({"pid": os.getpid(), "started": drain._now(), "run_log": "wd.log"}),
        encoding="utf-8",
    )
    spawned: list[list[str]] = []
    monkeypatch.setattr(watchdog, "_spawn_detached", _fake_spawn(paths, spawned))

    ok, msg = watchdog.start_watchdog()
    assert ok is False and "已在跑" in msg
    assert spawned == []
    assert not watchdog._spawn_marker().exists()


def test_stale_spawn_marker_is_reclaimed(iso_state, monkeypatch):
    """卡死持有者留下的标记超龄即回收——派生面不得被一个死标记永久挡死。"""
    paths = _paths(iso_state, monkeypatch)
    paths.runs.mkdir(parents=True, exist_ok=True)
    mark = watchdog._spawn_marker()
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text("999999 dead\n", encoding="utf-8")
    old = mark.stat().st_mtime - (watchdog.SPAWN_MARKER_TTL_S + 60)
    os.utime(mark, (old, old))
    spawned: list[list[str]] = []
    monkeypatch.setattr(watchdog, "_spawn_detached", _fake_spawn(paths, spawned))

    ok, _msg = watchdog.start_watchdog()
    assert ok is True and len(spawned) == 1, "超龄标记应被回收并照常派生"


def test_twin_run_watchdog_second_exits_lock_busy_and_logs(iso_state, monkeypatch):
    """并发两个 ``run_watchdog``（同参一份 + 不同参一份）→ 恰一个存活、另一个 lock_busy。

    锁是**按锁文件**互斥的，与 run_log 是否同名无关——故「不同参」同样必须被挡退
    （现场「键控 run_log 同名」的猜测由此证伪：互斥面从来不是 run_log）。
    """
    paths = _paths(iso_state, monkeypatch)
    paths.runs.mkdir(parents=True, exist_ok=True)
    a_log = paths.runs / "wd-a.log"
    b_log = paths.runs / "wd-b.log"

    entered = threading.Event()
    stop_a = threading.Event()
    summary_a: dict = {}

    class _Terminate(Exception):
        """哨兵：让 A 的循环以异常路径收尾（看门狗循环无自然退出，除非有停止纪元）。"""

    def _sleeps(_s: float) -> None:
        if stop_a.is_set():
            raise _Terminate("terminate-loop")
        entered.set()
        stop_a.wait(timeout=15)  # 阻塞在循环里 = 实例 A 存活且持锁

    def _run_a() -> None:
        summary_a.update(watchdog.run_watchdog(a_log, sleeps=_sleeps))

    t = threading.Thread(target=_run_a, daemon=True)
    t.start()
    assert entered.wait(10), "实例 A 未进入监督循环"

    # (a) 同参（同 run_log）第二实例
    s_same = watchdog.run_watchdog(a_log, sleeps=lambda _s: None)
    assert s_same.get("skipped") == "lock_busy"
    # (b) 不同参（不同 run_log）第二实例——锁面互斥，同样秒退
    s_other = watchdog.run_watchdog(b_log, sleeps=lambda _s: None)
    assert s_other.get("skipped") == "lock_busy"
    assert s_other.get("run_log"), "lock_busy 摘要须带 run_log 供取证定位"
    assert "lock_busy" in b_log.read_text(encoding="utf-8"), (
        "抢不到锁必须留痕（否则该 run_log 是空文件，取证看不出「为什么没跑」）"
    )

    stop_a.set()
    t.join(10)
    assert summary_a.get("completed"), "实例 A 应干净收尾（finally 里释放锁）"
    assert not paths.lock.exists(), "退出即释放锁，接棒者不被残骸挡住"


def test_run_log_unique_within_same_second(iso_state, monkeypatch):
    """同秒多次生成 → 文件名互不相同（撞名放大是双生的放大因子，命名面需被钉死）。"""
    paths = _paths(iso_state, monkeypatch)
    names = [watchdog.new_run_log(paths).name for _ in range(5)]
    assert len(set(names)) == 5, f"同秒生成撞名：{names}"
    assert all(n.endswith(".log") for n in names)
