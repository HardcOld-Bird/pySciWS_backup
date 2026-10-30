"""worker 的可注入测试缝必须经**模块属性**调用（backlog 20261014-015326-devops）。

事故形状（M6 实测）：worker 若用 ``from .drain import _sleep`` **值绑定**，测试对
``drain._sleep`` 的 monkeypatch **打不到**调用点——conftest 的 autouse「睡成 no-op」形同
虚设；成员 worker 的「睡到最近到期」循环在**真睡** + 未推进的时钟下会**挂死**。本测试把
「经模块属性」钉住：patch ``drain._sleep`` 后，进入该分支的 worker 必须**不真睡**、立即返回。

若日后有人把 ``_drain._sleep`` 改回 ``from .drain import _sleep``，本测试的 ``slept`` 会为空
（patch 落空）→ 断言失败；等待值取小（2s）以便回归时**快失败**而非挂几分钟。
"""

from __future__ import annotations

import time

import pytest

from pysci.skills.orchestration.tools import benchd, drain, queues, queueworker

WAIT_S = 2.0


@pytest.fixture
def state(tmp_path, monkeypatch):
    root = tmp_path / "state"
    monkeypatch.setattr(queues, "QUEUES_DIR", root / "queues")
    monkeypatch.setattr(queueworker, "ORCH_STATE_ROOT", root)
    return root


def _patch_sleep_and_wait(monkeypatch, wait_owner, wait_attr: str):
    """patch 唯一缝 ``drain._sleep``（记录调用）+ 对侧 ``next_defer_s``（先 WAIT_S 后 0）。"""
    slept: list[float] = []
    monkeypatch.setattr(drain, "_sleep", lambda s: slept.append(float(s)))
    seq = {"n": 0}

    def fake_wait(*a, **k):
        seq["n"] += 1
        return WAIT_S if seq["n"] == 1 else 0.0

    monkeypatch.setattr(wait_owner, wait_attr, fake_wait)
    return slept


def test_member_worker_sleeps_via_module_attribute_seam(state, monkeypatch):
    """成员 worker「睡到到期」经 ``drain._sleep``（模块属性）——patch 它即生效。"""
    item = queues.enqueue("figure", text="x", root=state)
    queues.defer(
        "figure",
        item["id"],
        until=time.time() + 300,
        count=1,
        elapsed_s=120,
        root=state,
    )
    slept = _patch_sleep_and_wait(monkeypatch, queues, "next_defer_s")

    summary = queueworker.run_worker("figure", state / "r.log", root=state)

    assert slept == [WAIT_S], f"未经 drain._sleep 缝（patch 落空）：{slept}"
    assert summary["count"] == 0  # 延期条目未被消化
    assert queues.items("figure", root=state)[0]["status"] == "pending"


def test_bencher_worker_sleeps_via_module_attribute_seam(iso_bench, monkeypatch):
    """bencher worker 同一缝隙（同族代码，别只修一处）。"""
    benchd.queue_save(
        [
            {
                "id": "b1",
                "kind": "run",
                "status": "pending",
                "defer_until": time.time() + 300,
                "defer_count": 1,
                "defer_elapsed_s": 120,
            }
        ]
    )
    slept = _patch_sleep_and_wait(monkeypatch, benchd, "next_defer_s")

    benchd.bench_drain(benchd.new_run_log(benchd.BENCH_PATHS))

    assert slept == [WAIT_S], f"未经 drain._sleep 缝（patch 落空）：{slept}"
    assert benchd.queue_load()[0]["status"] == "pending"
