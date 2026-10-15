"""交付正文全文留档（用户裁决 2026-10-15）：长交付入库 / body_file 引用 / 截断标记 / 失败亦留档。

落点 ``orchestration/state/deliveries/<job_id>.md``（**入 git**）——长交付全文的唯一入库落点
（队列条目 body 截 2000 字、queues-runs 日志被 ignore）。夹具走 ``iso_state``（已重定向
``queueworker.DELIVERIES_DIR`` → tmp，不写真实面、不豁免哨兵）。
"""

from __future__ import annotations

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import dispatch, queueworker
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

LONG = "甲" * 3000  # 远超 2000 字截断线
SHORT = "乙" * 120


def _item(**over) -> dict:
    base = {
        "id": "sim-20261015-000000-demo-s1",
        "member": "sim",
        "plan": "p1",
        "step": "s1",
        "task_file": "",
        "task": "任务书全文",
    }
    base.update(over)
    return base


def _out(
    body: str, *, code: int = 0, kind: str = "result", error: str = ""
) -> DispatchOutcome:
    return DispatchOutcome(
        code=code, kind=kind, member="sim", sid="sid-1", body=body, error=error
    )


# ---------------------------------------------------------------------------
# ① 全文落盘（含头部元数据）
# ---------------------------------------------------------------------------
def test_archive_writes_full_body_and_metadata(iso_state):
    item = _item()
    result = {
        "kind": "result",
        "member": "sim",
        "code": 0,
        "task_file": "",
        "sid": "sid-1",
        "duration_ms": 42,
    }
    rel = queueworker.archive_delivery(item, result, body=LONG)
    assert rel.endswith(f"deliveries/{item['id']}.md") or rel.endswith(
        "deliveries\\" + f"{item['id']}.md"
    )
    text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    # 头部元数据齐全
    for frag in (
        f"交付全文留档：{item['id']}",
        "- member: sim",
        "- kind: result",
        "- plan: p1",
        "- step: s1",
        "- duration_ms: 42",
    ):
        assert frag in text, frag
    # **全文**（非截断）在盘
    assert LONG in text and "## 交付正文" in text


def test_archive_failed_delivery_includes_error(iso_state):
    item = _item(id="sim-x-blocked")
    result = {
        "kind": "blocked",
        "member": "sim",
        "code": 2,
        "task_file": "",
        "sid": "",
        "duration_ms": 3,
    }
    rel = queueworker.archive_delivery(
        item, result, body="受阻正文", error="boom: 环境坏"
    )
    text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    assert "受阻正文" in text and "## 错误正文" in text and "boom: 环境坏" in text


def test_archive_empty_job_id_fails_soft(iso_state):
    assert queueworker.archive_delivery({"id": ""}, {}, body="x") == ""


# ---------------------------------------------------------------------------
# ② _execute：长交付 → body 截断 + 标记 + body_file；短交付 → 无标记
# ---------------------------------------------------------------------------
def test_execute_long_delivery_archives_and_marks(iso_state, monkeypatch):
    monkeypatch.setattr(dispatch, "do_dispatch", lambda member, **kw: _out(LONG))
    status, result, _outcome = queueworker._execute(_item(id="sim-long"), dry=False)
    assert result["body_file"], "长交付须有 body_file"
    assert result["body"].startswith(LONG[:2000])
    assert "截断" in result["body"] and result["body_file"] in result["body"]
    text = (PROJECT_ROOT / result["body_file"]).read_text(encoding="utf-8")
    assert LONG in text, "落盘须是**全文**（非 2000 字截断）"
    assert status  # done


def test_execute_short_delivery_no_truncation_mark(iso_state, monkeypatch):
    monkeypatch.setattr(dispatch, "do_dispatch", lambda member, **kw: _out(SHORT))
    _status, result, _o = queueworker._execute(_item(id="sim-short"), dry=False)
    assert result["body"] == SHORT, "短交付 body 原样（无截断标记）"
    assert "截断" not in result["body"]
    assert result["body_file"], "短交付同样留档（引用在，无标记）"
    assert SHORT in (PROJECT_ROOT / result["body_file"]).read_text(encoding="utf-8")


def test_execute_blocked_delivery_archived(iso_state, monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda member, **kw: _out(LONG, code=2, kind="blocked", error="拦住了"),
    )
    _status, result, _o = queueworker._execute(_item(id="sim-blk"), dry=False)
    assert result["kind"] == "blocked" and result["body_file"]
    text = (PROJECT_ROOT / result["body_file"]).read_text(encoding="utf-8")
    assert LONG in text and "拦住了" in text


# ---------------------------------------------------------------------------
# ③ 读取面透出：plans._outcome_from_job 带出 body_file
# ---------------------------------------------------------------------------
def test_outcome_from_job_carries_body_file(iso_state):
    from pysci.skills.orchestration.tools import plans

    item = {
        "id": "j1",
        "member": "sim",
        "status": "done",
        "result": {
            "code": 0,
            "kind": "result",
            "body": "…（截断，全文见 x）",
            "body_file": "orchestration/state/deliveries/j1.md",
            "member": "sim",
        },
    }
    oc = plans._outcome_from_job("sim", item)
    assert oc.body_file == "orchestration/state/deliveries/j1.md"


def test_batch_item_result_carries_body_file(iso_state, monkeypatch):
    """backlog-batch 条目终态存的是**账务行**（非交付正文）——`body_file` 必须一并补挂，
    否则留档虽在盘上、读取面却拿不到全文路径（2026-10-15 核验实测：10/10 批次条目缺该字段）。
    """
    from pysci.skills.orchestration.tools import drain, queues, workq

    from .conftest import seed_backlog

    seed_backlog(
        iso_state,
        [{"id": "s1", "status": "pending", "summary": "种子", "member": "devops"}],
    )
    drain.drain_backlog(drain.new_run_log())
    assert queues.items("devops")[0]["kind"] == drain.BACKLOG_BATCH_KIND

    monkeypatch.setattr(dispatch, "do_dispatch", lambda member, **kw: _out(LONG))
    wpaths = queueworker.member_paths("devops")
    wpaths.runs.mkdir(parents=True, exist_ok=True)
    queueworker.run_worker("devops", queueworker.new_run_log(wpaths))

    it = queues.items("devops")[0]
    assert it["status"] == workq.STATUS_DONE
    assert it["result"]["body_file"], "批次条目终态须带 body_file"
    text = (PROJECT_ROOT / it["result"]["body_file"]).read_text(encoding="utf-8")
    assert LONG in text, "补挂的引用须指向全文留档（非截断）"
