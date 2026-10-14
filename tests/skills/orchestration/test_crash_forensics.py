"""崩溃现场取证两面（backlog 20261013-003000-leader）。

① **席位形状闸**：run 211429 尾实测 ``TypeError: expected str, bytes or os.PathLike
object, not list``——21:14 派生的 drain 进程内存里是旧读法（``Registry.models`` 裸合并
跟踪面），00:22 并入 main 的新跟踪面是**链形态**，整条链当一个席位取走，list 进 argv
后由 ``subprocess`` 抛错，而这条消息不指旗标、不指文件（长生命周期 worker 撞上配置迁移
是**一类**竞态，不是一次性事故）。
② **异常栈入 run 日志**：``except Exception`` 只留 ``{type}: {exc}`` 时，定位只能靠
git blame 猜落点——三面派生路径（drain / bencher / inspector）统一留全栈。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysci.skills.orchestration.tools import (
    benchd,
    dispatch,
    drain,
    inspectd,
    registry,
    runner,
)

from .conftest import (
    RUNTIME_FILE,
    cmd_model,
    drain_once,
    envelope,
    fake_headless,
    read_backlog,
    seed_backlog,
    set_presets,
)

A = "seat-a"
B = "seat-b"
C = "seat-c"
CRASH = "expected str, bytes or os.PathLike object, not list"
OK_BODY = "<result>完工，commit abc1234 已合并。backlog id=s1,s2</result>"


# ---------------------------------------------------------------------------
# ① 席位形状：读侧摊平 + argv 闸
# ---------------------------------------------------------------------------
def test_nested_chain_face_flattens_into_str_seats(iso_registry):
    """迁移期形状（链中链 / 混着 null 与 dict）不得变成 ``"['a', 'b']"`` 垃圾席位。"""
    reg_file, _runtime_file, _backup_file = iso_registry
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "members": {},
                "models": {"minimum": [[A, B], None, {"tier": A}, C]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reg = registry.Registry.load()
    assert reg.model_chain("minimum") == [A, B, C], "嵌套摊平、None 与 dict 丢弃"
    assert all(isinstance(s, str) for s in reg.model_chain("minimum"))
    assert isinstance(reg.models["minimum"], str)


def test_build_command_gate_names_the_flag_it_rejected():
    """非字符串席位在**派生之前**被拦，消息里带旗标名与实况值（旧 TypeError 两样都没有）。"""
    m = registry.Member(member_id="devops", pod=Path("pods/devops"))
    with pytest.raises(runner.SeatShapeError, match=r"-m.*list") as ei:
        runner.build_command(
            m, "任务书", session_id="sid", resume=False, model=[A, B], exe=Path("exe")
        )
    assert A in str(ei.value), "实况值进消息（不靠猜哪个档写坏了）"
    assert isinstance(ei.value, ValueError), "既有 ValueError 通路仍接得住"


def test_headless_argv_of_chain_shaped_batch_is_all_str_and_takes_orders(
    iso_state, iso_dispatch, tmp_path, monkeypatch
):
    """阳性单测（success_criteria 第 3 条同形参数）：种子+组包 ``(+1)``、``effort=high``、
    跟踪面是链形态——批次循环正常取单，argv 全字符串且 ``-m`` 取到链首席。"""
    pod = tmp_path / "pods" / "devops"
    (pod / "inbox").mkdir(parents=True)
    _register_devops(iso_dispatch, pod)
    # 事故当时的跟踪面形状：一个档 = 一条有序链
    set_presets(iso_dispatch, mimo={"main": [A, B], "pro": [A, B]})
    seed_backlog(
        iso_state,
        [
            {"id": "s1", "status": "pending", "member": "leader", "effort": "high"},
            {"id": "s2", "status": "pending", "member": "leader", "effort": "high"},
        ],
    )
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    run_log = drain.new_run_log()
    summary = drain_once(log=run_log)

    assert len(calls) == 1, "同提请者一批一跳"
    argv = calls[0]
    assert all(isinstance(part, str) for part in argv), f"argv 混入非字符串：{argv}"
    assert cmd_model(argv) == A
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    text = run_log.read_text(encoding="utf-8")
    assert "计划批次 s1(+1)" in text and "effort=high" in text
    assert summary["count"] == 2
    assert all(i["status"] == "done" for i in read_backlog(iso_state))


# ---------------------------------------------------------------------------
# ② 崩溃留栈：三面派生路径同口径
# ---------------------------------------------------------------------------
def test_drain_crash_leaves_full_stack_in_run_log(iso_state, monkeypatch):
    seed_backlog(iso_state, [{"id": "s1", "status": "pending", "member": "leader"}])

    def boom(member, **kw):
        raise TypeError(CRASH)

    monkeypatch.setattr(dispatch, "do_dispatch", boom)
    run_log = drain.new_run_log()
    summary = drain_once(log=run_log)

    text = run_log.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text
    assert "_execute" in text, (
        "栈里要有落点函数名（backlog-fifo-20261014 后落点是 queueworker._execute"
        "——旧 '_drain_batch' 已退役；有帧名才能 git blame 定位，只有一句裸消息不算）"
    )
    assert CRASH in text
    assert summary["error"] == f"TypeError: {CRASH}", "执行账仍收短消息（不被栈淹掉）"
    wdone = sorted((iso_state / "queues-runs" / "devops").glob("*.done"))[-1]
    assert (
        json.loads(wdone.read_text(encoding="utf-8"))["item_error"] == summary["error"]
    )


def test_benchd_item_crash_leaves_stack_in_run_log(iso_bench, monkeypatch, tmp_path):
    """项级失败也不再只留一行：bencher worker 的 ``run_item`` 同口径。"""
    item = benchd.enqueue("run", cases=["fig-basic"])
    monkeypatch.setitem(benchd._RUNNERS, "run", _boom_runner)
    log = tmp_path / "bench.log"
    result = benchd.run_item(benchd.queue_load()[0], log)

    assert result["status"] == "failed" and CRASH in result["error"]
    text = log.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text and "_boom_runner" in text
    assert item["id"] in text, "哪一项坏的仍可一眼认出（旧格式 {id} 失败：… 不回归）"


def test_inspectd_item_crash_leaves_stack_in_run_log(iso_state, monkeypatch, tmp_path):
    item = inspectd.enqueue("assignment", brief="核查：崩溃是否留栈")
    monkeypatch.setitem(inspectd._RUNNERS, "assignment", _boom_runner)
    log = tmp_path / "inspect.log"
    result = inspectd.run_item(item, log)

    assert result["status"] == "failed" and CRASH in result["error"]
    text = log.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text and "_boom_runner" in text


def test_log_exception_without_run_log_still_returns_brief():
    """``run_log=None`` = 静默，但短消息照旧返回（记账面不得因日志面缺失而空转）。"""
    try:
        _boom_runner({}, None)
    except TypeError as exc:
        assert drain._log_exception(None, exc, "x") == f"TypeError: {CRASH}"


# ---------------------------------------------------------------------------
# 夹具辅助
# ---------------------------------------------------------------------------
def _boom_runner(item, run_log, *, dry=False) -> dict:
    """把当年那条裸 TypeError 原样抛出来（复现现场，不换成自定义异常）。"""
    raise TypeError(CRASH)


def _register_devops(state, pod) -> None:
    """把 devops 成员补进隔离 registry（drain 硬编码派发给它，夹具默认只有 quotamember）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["members"]["devops"] = {"pod": str(pod), "max_turns": 3, "timeout_s": 5}
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    runtime_file = state / RUNTIME_FILE
    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    rt.setdefault("members", {})["devops"] = {"sessions": []}
    runtime_file.write_text(
        json.dumps(rt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
