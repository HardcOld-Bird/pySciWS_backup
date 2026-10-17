"""reviewer rubrics 的在位性与预算（G1：PPT 审查标准 presentation.md）。

用户硬要求「不要让组员在不经视觉检查的情况下提交 ppt 产物」的把关端标准：reviewer 的
rubrics 里此前只有 generic + scientific-plotting，pptx/幻灯片类产物**无标准可依**。本文件
把「presentation.md 在位、通用、预算内、被 agents.md 索引」钉成机械判据——这些错了不会让
任何逻辑测试变红，只会让终审在下一次 PPT 任务里**静默缺标准**。
"""

from __future__ import annotations

from pysci.paths import PODS_ROOT
from pysci.skills.devops.tools import budget as budget_mod

REVIEWER = PODS_ROOT / "reviewer"
RUBRICS = REVIEWER / "rubrics"
AGENTS = REVIEWER / "agents.md"

#: 演示文稿 rubric 里**不该出现**的本任务专名（reviewer 标准须通用，不绑单一课题）。
TASK_SPECIFIC_TOKENS = ("gain_ep", "1_gain_ep", "20260918", "1016PPT")


def test_reviewer_rubrics_present_and_within_budget():
    """三张 rubric 都在位，且各 ≤8192 **CRLF 等价字节**（harness 全量注入档硬上限）。"""
    for name in ("generic.md", "scientific-plotting.md", "presentation.md"):
        f = RUBRICS / name
        assert f.is_file(), f"缺 rubric：{f}"
        n = budget_mod.disk_size(f)
        assert n <= budget_mod.LIMIT, f"{name} 超 8KB 预算：{n} B"


def test_presentation_rubric_is_generic():
    """presentation.md 通用（不含本任务专名），且带四步流程与 H/D 硬项/评分骨架。"""
    text = (RUBRICS / "presentation.md").read_text(encoding="utf-8")
    for tok in TASK_SPECIFIC_TOKENS:
        assert tok not in text, f"presentation.md 混入本任务专名：{tok}"
    for marker in ("硬项", "评分维度", "审查流程", "VERDICT"):
        assert marker in text, f"presentation.md 缺结构段：{marker}"
    assert "逐页" in text and "零遮挡" in text, "缺逐页视觉核验/零遮挡硬项"


def test_reviewer_agents_indexes_the_presentation_rubric():
    """reviewer/agents.md 的 rubric 索引里点到 presentation.md（终审才找得到新标准）。"""
    text = AGENTS.read_text(encoding="utf-8")
    assert "rubrics/presentation.md" in text, "agents.md 未索引演示文稿 rubric"
