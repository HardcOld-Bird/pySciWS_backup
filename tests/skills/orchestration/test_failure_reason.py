"""``<blocked reason="…">`` 成因档位全链（backlog 20261013-191213-devops）：词表→解析→gate→
drain .done 分列→orch status 分列。

病灶：``<blocked>`` **语义过载**——「已侦察待裁决」（重派无效，等判决）与「执行端余量不足
待排期」（补足余量后重派即可落地）这两类**最优处置相反**，却都长成同一个 ``<blocked>``：
drain 的抢救链只看 ``<blocked>`` 与 ``tags``，``orch status`` 只报「谁被挡了」。``reason``
补上这一维，让两类在读数面分列。纪律与 ``tags`` 一致：受控词表（``state/failure-tags.json``
的 ``reasons`` 段）、gate 校验、缺省不标注合法。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from pysci.paths import ORCH_STATE_ROOT, ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import dispatch, drain, ledger, orch
from pysci.skills.orchestration.tools.delivery import parse_delivery
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.failure_tags import (
    DEFAULT_REASONS,
    load_reasons,
    parse_reason,
)
from pysci.skills.orchestration.tools.ledger import LedgerEntry

from .conftest import drain_once, seed_backlog

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"
VOCAB_FILE = ORCH_STATE_ROOT / "failure-tags.json"


# ---------------------------------------------------------------------------
# ① 词表只有一份：state 文件 == Python 兜底 == guard 源码兜底
# ---------------------------------------------------------------------------
def test_file_reasons_match_python_default():
    assert set(load_reasons(VOCAB_FILE)) == set(DEFAULT_REASONS)


def test_shipped_reasons_definitions_non_empty():
    """纪律条款要求「增补须附定义」：随附文件每档都得有定义。"""
    empty = [r for r, d in load_reasons(VOCAB_FILE).items() if not d.strip()]
    assert not empty, f"以下成因档位缺定义：{empty}"


def test_guard_default_matches_file_reasons():
    """guard 里的兜底清单必须与文件键集同——两面各执法一张不同的名单就会漂移。"""
    src = GUARD.read_text(encoding="utf-8")
    m = re.search(r"REASONS_DEFAULT\s*=\s*\[([^\]]*)\]", src, re.DOTALL)
    assert m, "guard 里找不到 REASONS_DEFAULT（改名须同步本哨兵）"
    listed = re.findall(r"""['"]([^'"]+)['"]""", m.group(1))
    assert listed and set(listed) == set(load_reasons(VOCAB_FILE))


def test_load_reasons_falls_back_when_section_absent(tmp_path):
    p = tmp_path / "failure-tags.json"
    p.write_text(json.dumps({"tags": {"tool_bug": "x"}}), encoding="utf-8")
    assert set(load_reasons(p)) == set(DEFAULT_REASONS)


def test_load_reasons_file_overrides_default(tmp_path):
    p = tmp_path / "failure-tags.json"
    p.write_text(json.dumps({"reasons": {"perf": "性能类"}}), encoding="utf-8")
    assert set(load_reasons(p)) == {"perf"}


# ---------------------------------------------------------------------------
# ② 属性解析：单值、规范化、不切分
# ---------------------------------------------------------------------------
def test_parse_reason_normalizes_case_and_quotes():
    assert parse_reason("  Capacity ") == "capacity"
    assert parse_reason('"needs_decision"') == "needs_decision"


def test_parse_reason_empty_is_legal():
    assert parse_reason(None) == "" and parse_reason("  ") == ""


def test_parse_reason_does_not_split():
    """**不切分**：``reason`` 是单值档位——切分会让 "capacity, foo" 静默丢掉 foo 又放行
    capacity，把「标错」伪装成「标对」。整串落词表外，交给 gate 退回。"""
    assert parse_reason("capacity, foo") == "capacity, foo"


# ---------------------------------------------------------------------------
# ③ 交付解析：reason 只归属 blocked
# ---------------------------------------------------------------------------
def test_blocked_with_reason_parsed():
    d = parse_delivery('<blocked reason="capacity">余量不足，半做</blocked>')
    assert d.kind == "blocked" and d.reason == "capacity"


def test_blocked_without_reason_is_empty():
    assert parse_delivery("<blocked>缺信息</blocked>").reason == ""


def test_result_never_carries_reason():
    d = parse_delivery('<result reason="capacity">做完了</result>')
    assert d.kind == "result" and d.reason == ""


def test_parse_error_never_carries_reason():
    d = parse_delivery(
        "两块并存 <result>a</result><blocked reason='capacity'>b</blocked>"
    )
    assert d.kind == "parse_error" and d.reason == ""


def test_reason_and_tags_coexist():
    d = parse_delivery(
        '<blocked tags="model_capability" reason="capacity">轮次耗尽</blocked>'
    )
    assert d.tags == ["model_capability"] and d.reason == "capacity"


# ---------------------------------------------------------------------------
# ④ 台账持久化与向后兼容
# ---------------------------------------------------------------------------
def test_ledger_roundtrip_keeps_reason(iso_state):
    ledger.append(
        LedgerEntry(
            ts="",
            member="devops",
            session_id="s",
            kind="blocked",
            reason="capacity",
        )
    )
    assert ledger.read_all()[0].reason == "capacity"


def test_old_line_without_reason_still_reads(iso_state):
    """追加式文件的历史行没有 reason 字段：取默认空串，不得抛。"""
    iso_state.joinpath("ledger").mkdir()
    (iso_state / "ledger" / "deliveries.jsonl").write_text(
        '{"ts": "2026-10-01T00:00:00+08:00", "member": "figure", "session_id": "s",'
        ' "kind": "blocked", "num_turns": 2, "duration_ms": 1}\n',
        encoding="utf-8",
    )
    assert ledger.read_all()[0].reason == ""


# ---------------------------------------------------------------------------
# ⑤ gate 面：词表来自文件（真跑 node，不走模拟实现）
# ---------------------------------------------------------------------------
@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    (r / "orchestration" / "state").mkdir(parents=True)
    (r / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tags": {"tool_bug": "工具缺陷"},
                "reasons": {"capacity": "待排期", "needs_decision": "待裁决"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (r / "pod").mkdir()
    return r


def run_gate(root: Path, msg: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYSCI_PROJECT_ROOT"] = str(root)
    env["PYSCI_POD"] = str(root / "pod")
    env.pop("PYSCI_DEPLOYED_SKILLS", None)
    proc = subprocess.run(
        ["node", str(GUARD)],
        input=json.dumps({"last_assistant_message": msg, "stop_hook_active": False}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return proc.returncode, proc.stderr


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="需要 node 运行 guard"
)


def test_gate_reason_in_vocab_passes(root):
    assert run_gate(root, '<blocked reason="capacity">余量不足</blocked>') == (0, "")


def test_gate_unlabeled_reason_passes(root):
    """不标注是协议里的合法选择，不得因「没标档位」拒收。"""
    assert run_gate(root, "<blocked>没查明</blocked>") == (0, "")


def test_gate_reason_out_of_vocab_rejected(root):
    rc, err = run_gate(root, '<blocked reason="bogus">乱写</blocked>')
    assert rc == 2
    assert "bogus" in err
    # 整改指令要给出可用清单，否则组员只能瞎猜
    assert "capacity" in err and "needs_decision" in err


def test_gate_reason_case_insensitive(root):
    rc, err = run_gate(root, '<blocked reason="CAPACITY">同上</blocked>')
    assert (rc, err) == (0, "")


def test_gate_result_reason_not_checked(root):
    """reason 只属于失败分类面；``<result>`` 带 reason 不校验。"""
    assert run_gate(root, '<result reason="bogus">做完了</result>') == (0, "")


def test_gate_reason_vocab_comes_from_the_file(root):
    """文件换了词表，guard 就换执法面（词表改动只改文件、不改代码）。"""
    vocab = root / "orchestration" / "state" / "failure-tags.json"
    vocab.write_text(
        json.dumps({"reasons": {"perf": "性能类"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert run_gate(root, '<blocked reason="capacity">x</blocked>')[0] == 2
    assert run_gate(root, '<blocked reason="perf">实测延迟超标</blocked>')[0] == 0


def test_gate_reason_falls_back_to_default_when_section_missing(root):
    """文件没 reasons 段 ≠ 不执法：回落兜底清单，非法档位仍被拦。"""
    (root / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps({"tags": {"tool_bug": "工具缺陷"}}), encoding="utf-8"
    )
    assert run_gate(root, '<blocked reason="bogus">x</blocked>')[0] == 2


# ---------------------------------------------------------------------------
# ⑥ drain 侧：.done 逐条落 reason + 分列计数（端到端，两条不同档位）
# ---------------------------------------------------------------------------
def _done(state, name: str, items: list) -> None:
    """落一份 ``devops-runs/<name>.done`` run 摘要（含 drain 会算的 ``by_reason``）。"""
    by_reason: dict[str, int] = {}
    for it in items:
        r = str(it.get("reason") or "")
        if r:
            by_reason[r] = by_reason.get(r, 0) + 1
    payload: dict = {"count": len(items), "elapsed_s": 1.0, "items": items}
    if by_reason:
        payload["by_reason"] = by_reason
    d = state / "devops-runs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.done").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_drain_done_splits_blocked_by_reason_and_status_shows_it(
    iso_state, monkeypatch, capsys
):
    """success_criteria：构造两条不同档位的 blocked → .done 与 orch status 两档各计 1。"""
    seed_backlog(
        iso_state,
        [
            {"id": "dec", "status": "pending", "summary": "待裁决"},
            {"id": "cap", "status": "pending", "summary": "待排期"},
        ],
    )

    def fake_dispatch(member, **kw):
        if "id=dec" in kw["text"]:
            return DispatchOutcome(
                code=2, kind="blocked", body="缺判决记录", reason="needs_decision"
            )
        return DispatchOutcome(
            code=2, kind="blocked", body="会话余量耗尽", reason="capacity"
        )

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    summary = drain_once()
    by_id = {i["id"]: i for i in summary["items"]}
    assert by_id["dec"]["reason"] == "needs_decision"
    assert by_id["cap"]["reason"] == "capacity"
    assert summary["by_reason"] == {"needs_decision": 1, "capacity": 1}

    # .done 落盘也要带上分列计数（组长/巡检读 `.done` 文件的那条路）——backlog-fifo 后
    # 消费账在执行侧（queues-runs/devops/<ts>.done），计划账（devops-runs）只记 queued。
    done_files = sorted((iso_state / "queues-runs" / "devops").glob("*.done"))
    persisted = json.loads(done_files[-1].read_text(encoding="utf-8"))
    assert persisted["by_reason"] == {"needs_decision": 1, "capacity": 1}

    # orch status 分列呈现（两档各计 1，用短标签）
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "待裁决=1" in out and "待排期=1" in out


def test_drain_item_without_reason_has_no_by_reason(iso_state, monkeypatch):
    """未标注 reason 的 blocked：逐条不落字段、也不进分列计数（缺省合法）。"""
    seed_backlog(iso_state, [{"id": "x", "status": "pending", "summary": "s"}])
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(code=2, kind="blocked", body="没查明"),
    )
    summary = drain.drain_backlog(drain.new_run_log())
    assert "reason" not in summary["items"][0]
    assert "by_reason" not in summary


def test_status_omits_reason_line_when_absent(iso_state, monkeypatch, capsys):
    """没有标注档位的 .done：status 不打「失败档位分列」行（不出噪声）。"""
    _done(iso_state, "20261013-010000.done", [{"id": "a", "status": "needs_leader"}])
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    assert "失败档位分列" not in capsys.readouterr().out


def test_status_unknown_gear_falls_back_to_raw_key(iso_state, monkeypatch, capsys):
    """词表外/新档位回落显示原键——不在显示表里也要可见，不得被静默吞掉。"""
    _done(
        iso_state,
        "20261013-020000.done",
        [{"id": "a", "status": "needs_leader", "reason": "brand_new_gear"}],
    )
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    orch.cmd_status(Namespace())
    assert "brand_new_gear=1" in capsys.readouterr().out
