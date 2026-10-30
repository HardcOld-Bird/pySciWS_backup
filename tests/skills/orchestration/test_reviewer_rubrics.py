"""reviewer rubrics 的在位性与预算（G1：PPT 审查标准 presentation.md）。

用户硬要求「不要让组员在不经视觉检查的情况下提交 ppt 产物」的把关端标准：reviewer 的
rubrics 里此前只有 generic + scientific-plotting，pptx/幻灯片类产物**无标准可依**。本文件
把「presentation.md 在位、通用、预算内、发现性锚在只读 charter」钉成机械判据——这些错了
不会让任何逻辑测试变红，只会让终审在下一次 PPT 任务里**静默缺标准**。
"""

from __future__ import annotations

from pysci.paths import PODS_ROOT
from pysci.skills.devops.tools import budget as budget_mod

REVIEWER = PODS_ROOT / "reviewer"
RUBRICS = REVIEWER / "rubrics"

#: 演示文稿 rubric 里**不该出现**的本任务专名（reviewer 标准须通用，不绑单一课题）。
TASK_SPECIFIC_TOKENS = ("example", "1_example", "20260918", "1016PPT")


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


def test_reviewer_discoverability_anchored_on_charter_not_memory():
    """发现性锚点改钉**只读层**（charter 的 rubrics 话语），不钉 agents.md 记忆。

    ADR（backlog 20261029-152100-devops，与 test_a5 同款）：reviewer 的 ``agents.md``
    是自维护记忆，预算整改/自主改写即断字符串（原钉「rubrics/presentation.md」索引行）。
    发现性硬保证＝rubric 在位+通用（上两测）＋ charter 的判定入口话语；记忆索引属
    成员自维护，不再作测试钉子。
    """
    charter = REVIEWER / ".qoder" / "rules" / "charter.md"
    text = charter.read_text(encoding="utf-8")
    assert "rubrics" in text, "charter 缺 rubrics 判定入口话语（发现性锚点）"


def test_cross_render_colormap_clause_pinned():
    """跨渲染/跨色标比对口径（G11）在位：generic 全条款 + scientific-plotting 流程步。

    backlog 20261026-054555-reviewer（2026-10-26 A/B 色标语义不同把同一数据误判「结构
    不一致」）：条款若被后续精简丢掉，同类假阳性 FAIL 会静默复发——钉成机械判据。
    """
    g = (RUBRICS / "generic.md").read_text(encoding="utf-8")
    for marker in ("G11", "跨渲染", "呈现差异", "数值窗口", "三方对照"):
        assert marker in g, f"generic.md 缺跨渲染比对口径要素：{marker}"
    s = (RUBRICS / "scientific-plotting.md").read_text(encoding="utf-8")
    for marker in ("跨色标比对", "generic G11", "不得单独支撑「内容不一致」"):
        assert marker in s, f"scientific-plotting.md 缺跨色标比对口径：{marker}"
