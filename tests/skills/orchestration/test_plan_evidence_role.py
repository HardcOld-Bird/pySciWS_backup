"""evidence lint 按 role 内容级校验（backlog 20261015-135000-f2-ledger）。

实案：``b_last``（role=final_report，全流程报告组装）拿 **c2 的方案文件**当证据竟然通过——
lint 只查「至少一个标题段」，而 ``b_last``/``b1`` 在返工模板里**没有** ``evidence_sections``
（只有 a1/a2/a3 显式写了），于是退回弱判据；对照之下 a2/a3 的标题段要求又很严。

本文件钉：**按 role 判内容特征段**，误收同款场景**拒收并点名缺哪一段**。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.orchestration.tools import plans

# 复用 test_plan_guards 的计划助手（``qface`` 夹具已上提到 conftest）
from .test_plan_guards import _args, _new

#: 误收样本形状（真文件 tb-20261015-091655-1.c2-监察改进方案.md）：只有问题定式/方案/结论。
C2_LIKE = "# 监察改进方案：x\n\n## 问题定式\n\n…\n\n## 改进方案\n\n…\n\n## 结论\n\n…\n"
#: 真·全流程报告形状（`taskbooks.report_sections` 的通用两段，任何任务书都有）。
REPORT_LIKE = "# 全流程报告：x\n\n## 任务概述\n\n…\n\n## 环节完成清单\n\n…\n"
#: 真·重拆解形状（tb-20261015-091655-1.b1-重拆解.md）。
REWORK_LIKE = "# 重拆解返工：x\n\n## 返工目标\n\n…\n\n## 拆解（B 系列叶子填充）\n\n…\n\n## 结论\n\n…\n"


def _ev(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "ev.md"
    p.write_text(text, encoding="utf-8")
    return p


def test_final_report_rejects_a_foreign_plan_doc(tmp_path):
    """误收同款：final_report 收到方案文件 → 拒收，并**点名**缺的段。"""
    ok, msg = plans._check_evidence(
        str(_ev(tmp_path, C2_LIKE)), {"role": "final_report"}
    )
    assert not ok
    assert "任务概述" in msg and "环节完成清单" in msg, msg
    assert "role=final_report" in msg, msg


def test_final_report_accepts_the_report_shaped_doc(tmp_path):
    ok, msg = plans._check_evidence(
        str(_ev(tmp_path, REPORT_LIKE)), {"role": "final_report"}
    )
    assert ok, msg


def test_rework_plan_requires_a_decomposition_section(tmp_path):
    """重拆解必须有拆解段——光有结论（或别的环节的成品）不算。"""
    assert not plans._check_evidence(
        str(_ev(tmp_path, REPORT_LIKE)), {"role": "rework_plan"}
    )[0]
    ok, msg = plans._check_evidence(
        str(_ev(tmp_path, REWORK_LIKE)), {"role": "rework_plan"}
    )
    assert ok, msg


def test_explicit_evidence_sections_win_over_role_default(tmp_path):
    """YAML 显式声明是每份计划的可定制面，优先于 role 缺省表。"""
    ok, _ = plans._check_evidence(
        str(_ev(tmp_path, REPORT_LIKE)),
        {"role": "final_report", "evidence_sections": ["任务概述"]},
    )
    assert ok
    bad, msg = plans._check_evidence(
        str(_ev(tmp_path, REPORT_LIKE)),
        {"role": "final_report", "evidence_sections": ["送审结论"]},
    )
    assert not bad and "标题段" in msg, msg


def test_cause_analysis_rule_is_unchanged(tmp_path):
    """现状保持：cause（a1）仍要「成因分析」＋「结论」。"""
    ok, msg = plans._check_evidence(
        str(_ev(tmp_path, "# 成因分析报告：x\n\n## 成因分析\n\n…\n\n## 结论\n\n…\n")),
        {"role": "cause"},
    )
    assert ok, msg
    assert not plans._check_evidence(
        str(_ev(tmp_path, REPORT_LIKE)), {"role": "cause"}
    )[0]


def test_unknown_role_falls_back_to_heading_presence(tmp_path):
    """无 role、也不在表里 → 仍退回「至少一个标题段」（fail-open：不误杀未登记的角色）。"""
    ok, msg = plans._check_evidence(
        str(_ev(tmp_path, "# 随手写的报告\n\n正文\n")), {"role": ""}
    )
    assert ok, msg


def test_advance_evidence_end_to_end_rejects_and_names_the_missing_section(
    qface, tmp_path, capsys
):
    """端到端（b_last 误收同款）：`advance --evidence <方案文件>` → rc=2 + 点名缺段。"""
    pid = "p-ev"
    assert (
        _new(
            qface,
            pid,
            [
                {
                    "id": "b_last",
                    "kind": "self",
                    "brief": "全流程报告",
                    "role": "final_report",
                }
            ],
        )
        == 0
    )
    plans.cmd_plan_advance(_args(pid))  # 引导 self 环节（self_awaiting）
    capsys.readouterr()

    bad = _ev(tmp_path, C2_LIKE)
    rc = plans.cmd_plan_advance(_args(pid, evidence=str(bad)))
    out = capsys.readouterr().out

    assert rc == 2
    assert "任务概述" in out and "环节完成清单" in out, out
    assert plans._load_state(pid)["steps"][0]["status"] == "self_awaiting", (
        "拒收后环节应留在待收尾态（可换文件重试）"
    )
