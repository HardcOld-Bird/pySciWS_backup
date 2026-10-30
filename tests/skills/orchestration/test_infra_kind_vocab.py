"""提案分类词表的三面收口（backlog 20261012-120001-leader）。

病灶实测：组员 110101 那跳的最终回复里连发两个 ``<infra_suggestion>`` 块，占位块在前
（``kind="defusability-placeholder"``、正文只剩 ``placeholder``）。gate 用 ``.match`` 只
查第一块、解析器也只取第一块，于是非法 kind 被落盘进**待审批队列**；gate 退回后组员重
交付，另立新文件，空壳却永久留在队列里。

因此执法面有三处，本文件逐面钉住：① gate 逐块校验且词表来自
``state/failure-tags.json`` 的 ``infra_kinds`` 段；② dispatch 落盘前复核（非法不入库）；
③ ``orch suggestions`` 枚举时标出空壳/非法分类（提案来源不止交付一条路：组长代补、
自动立案、手写文件都直接落进队列）。外加「词表只有一份」的一致性哨兵——三面各读一份
就会漂移，漂移的后果正是本单。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCH_STATE_ROOT, ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import (
    dispatch,
    failure_tags,
    ledger,
    orch,
    workflow,
)
from pysci.skills.orchestration.tools.failure_tags import (
    DEFAULT_INFRA_KINDS,
    load_infra_kinds,
)

from .conftest import envelope, fake_headless, seed_raw_suggestion

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"
VOCAB_FILE = ORCH_STATE_ROOT / "failure-tags.json"

RESULT = "已完成。<result>交付正文</result>"
BLOCK = '<infra_suggestion kind="defect">修复面：把词表做成单一源。</infra_suggestion>'


# ---------------------------------------------------------------------------
# ① 词表只有一份：state 文件 == Python 兜底 == guard 源码兜底
# ---------------------------------------------------------------------------
def test_file_vocab_matches_python_default():
    assert set(load_infra_kinds(VOCAB_FILE)) == set(DEFAULT_INFRA_KINDS)


def test_guard_default_matches_file_vocab():
    """guard 里的兜底清单必须与文件键集同——两面各执法一张不同的名单就是本单的病灶。"""
    src = GUARD.read_text(encoding="utf-8")
    m = re.search(r"INFRA_KINDS_DEFAULT\s*=\s*\[([^\]]*)\]", src)
    assert m, "guard 里找不到 INFRA_KINDS_DEFAULT（改名须同步本哨兵）"
    listed = re.findall(r"""['"]([^'"]+)['"]""", m.group(1))
    assert listed and set(listed) == set(load_infra_kinds(VOCAB_FILE))


def test_load_infra_kinds_falls_back_when_section_absent(tmp_path):
    """文件在但没有该段 → 回落兜底清单（不给一张空白名单，也不炸）。"""
    p = tmp_path / "failure-tags.json"
    p.write_text(
        json.dumps({"version": 1, "tags": {"tool_bug": "x"}}), encoding="utf-8"
    )
    assert set(load_infra_kinds(p)) == set(DEFAULT_INFRA_KINDS)


def test_load_infra_kinds_file_overrides_default(tmp_path):
    p = tmp_path / "failure-tags.json"
    p.write_text(json.dumps({"infra_kinds": {"perf": "性能类"}}), encoding="utf-8")
    assert set(load_infra_kinds(p)) == {"perf"}


# ---------------------------------------------------------------------------
# ② gate 面：逐块校验 + 词表来自文件（真跑 node，不走模拟实现）
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """自带词表（含 infra_kinds 段）的项目根 + 一个干净 pod。"""
    r = tmp_path / "proj"
    (r / "orchestration" / "state").mkdir(parents=True)
    (r / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tags": {"tool_bug": "工具缺陷"},
                "infra_kinds": {"defect": "缺陷", "usability": "难用"},
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


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_gate_rejects_kind_outside_vocab(root):
    rc, err = run_gate(
        root,
        RESULT
        + '<infra_suggestion kind="foo">修复面：把词表做成单一源，三面共读。</infra_suggestion>',
    )
    assert rc == 2
    assert "defect 或 usability" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_gate_valid_kind_passes(root):
    assert run_gate(root, RESULT + BLOCK) == (0, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_gate_checks_every_block_not_just_the_first(root):
    """110101 的实形：合法块在前、非法块在后——只查第一块就会放过去。"""
    rc, err = run_gate(
        root,
        RESULT
        + BLOCK
        + '<infra_suggestion kind="defusability-placeholder">placeholder</infra_suggestion>',
    )
    assert rc == 2
    assert "defusability-placeholder" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_gate_flags_multi_block_delivery(root):
    """两块都合法也要拦：解析器只取第一块，其余内容会被静默丢弃。"""
    rc, err = run_gate(
        root,
        RESULT
        + BLOCK
        + '<infra_suggestion kind="usability">另一条建议。</infra_suggestion>',
    )
    assert rc == 2
    assert "只取第一块" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_gate_vocab_comes_from_the_file(root):
    """文件换了词表，guard 就换执法面（词表改动只改文件、不改代码的纪律）。"""
    vocab = root / "orchestration" / "state" / "failure-tags.json"
    vocab.write_text(
        json.dumps({"infra_kinds": {"perf": "性能类"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert run_gate(root, RESULT + BLOCK)[0] == 2  # defect 已不在词表内
    ok = run_gate(
        root,
        RESULT
        + '<infra_suggestion kind="perf">修复面：实测延迟超标。</infra_suggestion>',
    )
    assert ok[0] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_gate_falls_back_to_default_when_section_missing(root):
    """文件没这段 ≠ 不执法：回落兜底清单，非法 kind 仍被拦。"""
    (root / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps({"tags": {"tool_bug": "工具缺陷"}}), encoding="utf-8"
    )
    rc, _ = run_gate(
        root,
        RESULT
        + '<infra_suggestion kind="foo">修复面：把词表做成单一源，三面共读。</infra_suggestion>',
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# ③ 落盘面：dispatch 是入队的唯一门，非法分类不得进队列
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_suggestions(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "suggestions"
    monkeypatch.setattr(dispatch, "SUGGESTIONS_DIR", d)
    monkeypatch.setattr(failure_tags, "TAGS_PATH", tmp_path / "absent.json")
    return d


def test_illegal_kind_is_not_persisted(iso_suggestions, capsys):
    sid = dispatch._persist_suggestion(
        "devops", "placeholder", "defusability-placeholder"
    )
    assert sid == ""
    assert not list(iso_suggestions.glob("*.md")), "非法 kind 不得落进审批队列"
    out = capsys.readouterr().out
    assert "改进建议未入库" in out and "defusability-placeholder" in out


def test_missing_kind_is_not_persisted(iso_suggestions):
    assert dispatch._persist_suggestion("devops", "正文若干。", "") == ""
    assert not list(iso_suggestions.glob("*.md"))


def test_valid_kind_is_persisted_with_head_marker(iso_suggestions):
    sid = dispatch._persist_suggestion("devops", "正文若干。", "defect")
    assert sid
    text = (iso_suggestions / f"{sid}.md").read_text(encoding="utf-8")
    assert text.splitlines()[2] == "kind: defect"


# ---------------------------------------------------------------------------
# ④ 审批面：手写/代补的文件绕过 gate 直达队列，approve 必须复核词表
# ---------------------------------------------------------------------------
def test_approve_rejects_kind_outside_vocab(iso_state):
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defusability-placeholder\n"
        "证据与修复面若干，够长够具体以满足判据门槛。\n\n"
        "success_criteria：\n- 构造非法 kind 的交付 → gate exit 2\n",
    )

    rc = workflow.cmd_approve(
        argparse.Namespace(suggestion_id=sid, note="", no_wake=True)
    )

    assert rc == 2
    assert not (iso_state / "backlog.json").exists()


# ---------------------------------------------------------------------------
# ⑤ 呈现面：枚举时机械标注，组长不必逐条打开才发现是空壳
# ---------------------------------------------------------------------------
def test_suggestions_flags_shell_and_bad_kind(iso_state, capsys):
    seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defusability-placeholder\n\nplaceholder\n",
        sid="20261012-110101-devops",
    )
    seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\n"
        "实测两面读数分裂：gate 只查第一个块，解析器也只取第一个块，"
        "于是非法分类被落盘、合法块反而没入库。\n\n"
        "success_criteria：\n- 三面共读同一份词表\n",
        sid="20261012-114239-devops",
    )

    workflow.cmd_suggestions(argparse.Namespace())
    out = capsys.readouterr().out

    lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.startswith("  ")}
    assert "空壳建议" in lines["20261012-110101-devops"]
    assert "kind 非法" in lines["20261012-110101-devops"]
    assert "[!]" not in lines["20261012-114239-devops"], "合规提案不该被标注"


def test_suggestions_flags_shell_even_with_valid_kind(iso_state):
    sid = seed_raw_suggestion(
        iso_state,
        "# 改进建议（devops，待审批）\n\nkind: defect\n\nTODO\n",
    )
    assert "[!] 空壳建议" in workflow.suggestion_marks(
        (iso_state / "suggestions" / f"{sid}.md").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# ⑥ 读数面：台账「建议✓」⟺ 审批队列里有这一条（backlog 20261012-132138-devops）
# ---------------------------------------------------------------------------
BAD_BLOCK = (
    '<infra_suggestion kind="defusability-placeholder">placeholder</infra_suggestion>'
)
GOOD_BLOCK = (
    '<infra_suggestion kind="defect">修复面：台账建议字段改判「已入库」，'
    "与审批队列同真同假，组长照读数找提案不会落空。</infra_suggestion>"
)
NO_BLOCK = "<result>交付正文，本跳没提提案。</result>"


@pytest.fixture
def hop_reads(iso_dispatch, monkeypatch, capsys):
    """真跑一跳，返回三个各自独立的读数：台账字段 / 队列文件 / 两个 CLI 的打印。

    断言走**呈现面**而不是私有函数——口径分裂正是在呈现面上伤人的：组长读
    ``orch ledger`` 见「建议✓」再去 ``orch suggestions`` 找，找不到就以为设施丢了件。
    """
    queue = iso_dispatch / "suggestions"
    monkeypatch.setattr(workflow, "SUGGESTIONS_DIR", queue)

    def read(body: str) -> dict:
        fake_headless(
            monkeypatch,
            lambda n, cmd: envelope(body, is_error=False, stop="end_turn", rc=0),
        )
        dispatch.do_dispatch("quotamember", text="提案", quiet=True, no_checks=True)
        capsys.readouterr()
        orch.cmd_ledger(argparse.Namespace(member=None, days=0, stats=False, tail=5))
        ledger_out = capsys.readouterr().out
        workflow.cmd_suggestions(argparse.Namespace())
        sugg_out = capsys.readouterr().out
        return {
            "row": ledger.read_all()[-1],
            "queue": sorted(p.stem for p in queue.glob("*.md")),
            "ledger_out": ledger_out,
            "sugg_out": sugg_out,
        }

    return read


def test_illegal_kind_hop_shows_no_suggestion_on_either_face(hop_reads):
    r = hop_reads(NO_BLOCK + BAD_BLOCK)
    assert r["row"].infra_suggestion is False, "未入库的块不得点亮台账建议字段"
    assert r["row"].infra_kind == "defusability-placeholder", (
        "分类原值仍要留档（可追溯）"
    )
    assert r["queue"] == []
    assert "建议✓" not in r["ledger_out"]
    assert "建议✗未入库" in r["ledger_out"], "分裂要可见，不是静默抹掉那一跳提过案"
    assert "无待审批建议" in r["sugg_out"]


def test_filed_kind_hop_shows_suggestion_on_both_faces(hop_reads):
    r = hop_reads(NO_BLOCK + GOOD_BLOCK)
    assert r["row"].infra_suggestion is True
    assert len(r["queue"]) == 1
    assert "建议✓" in r["ledger_out"] and "✗" not in r["ledger_out"]
    assert r["queue"][0] in r["sugg_out"], "两个读数面要指向同一条提案"


@pytest.mark.parametrize("block", [BAD_BLOCK, GOOD_BLOCK], ids=["未入库", "已入库"])
def test_ledger_field_and_queue_entry_are_same_truth_value(hop_reads, block):
    """§12 判据要求的不变量本身：台账建议字段 ⟺ 队列存在条目，同真同假。"""
    r = hop_reads(NO_BLOCK + block)
    assert bool(r["row"].infra_suggestion) == bool(r["queue"])


def test_hop_without_suggestion_is_not_marked(hop_reads):
    """没提提案的跳：既不显示 ✓ 也不显示 ✗——「未入库」只属于真的提过却被闸拦下的跳。"""
    r = hop_reads(NO_BLOCK)
    assert r["row"].infra_suggestion is False and r["row"].infra_kind == ""
    assert "建议" not in r["ledger_out"]
