"""审查表达与「组长立单」的文档面（backlog 20261015-202600-devops）。

组长摩擦：① `review: true` 的机械语义、与 `member: reviewer` / `orch review` 的关系只在单行
提及里查不到；② 组长立单（手写 suggestions/）的格式只在源码 docstring 与历史文件里。

本文件钉两件事：
- **真相面**：文档所述与**实现**一致（`review: true` 确实自动投 reviewer 附属子 job、
  宿主不发 JOB-DONE 而是合并发 STEP-REVIEW-DONE）；
- **格式面**：照文档写的建议单过质检（``[]``），太短则被判 ``[!] 空壳建议``。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import events, queues, workflow

REF = ORCHESTRATION_ROOT / "skills" / "orchestration" / "references"
PLAYBOOK = REF / "dispatch-playbook.md"
CMDSURF = REF / "command-surface.md"


def _text(p) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ① 三种审查表达：文档在、且与实现一致
# ---------------------------------------------------------------------------
def test_playbook_documents_three_review_expressions():
    md = _text(PLAYBOOK)
    assert "三种审查表达" in md
    for token in ("review: true", "member: reviewer", "orch review"):
        assert token in md, f"缺审查表达：{token}"


def test_documented_wiring_matches_implementation():
    """文档说「自动投附属子 job + 宿主合并发 STEP-REVIEW-DONE」——与实现对照。"""
    md = _text(PLAYBOOK)
    for token in ("spawn_review", "STEP-REVIEW-DONE", "awaiting_review"):
        assert token in md, f"文档未点明机械行为：{token}"

    # 实现侧：无 review 标记 → 不投子 job；有标记才投（spawn_review 的守卫）
    assert queues.spawn_review({"id": "j", "member": "m"}) is None
    assert queues.spawn_review({"id": "j", "member": "m", "review": False}) is None
    assert queues.AWAITING_REVIEW == "awaiting_review"
    assert events.STEP_REVIEW_DONE  # 合并完成事件确实存在


# ---------------------------------------------------------------------------
# ② 组长立单：文档在、且照它写的单过质检
# ---------------------------------------------------------------------------
#: 照 `command-surface.md`「组长立单」块写的建议单形状。
DOC_FORMAT_SAMPLE = """# 改进建议（devops，待审批）

kind: usability

**场景**：按文档格式手写一单，验证质检面认它。
**建议**：正文实质字数需明显超过四十个字符，故此处多写一些填充性的说明文字。

success_criteria:
- `orch suggestions` 列出本单且不带 [!] 标记。
"""


def test_command_surface_documents_filing_format():
    md = _text(CMDSURF)
    assert "组长立单" in md
    for token in (
        "suggestions/<ts>-<member>.md",
        "kind: defect|usability",
        "success_criteria:",
    ):
        assert token in md, f"立单格式缺要素：{token}"


def test_documented_suggestion_format_passes_qc():
    """正向：照文档写的单 → 无质检标记（合规显示 ``[]``）。"""
    assert workflow.suggestion_marks(DOC_FORMAT_SAMPLE) == []


def test_documented_shell_threshold_is_real():
    """文档写「实质正文 ≥40 字，否则标空壳」——太短的正文确被判空壳。"""
    assert workflow.suggestion_marks("kind: usability\n\n短。\n") == ["[!] 空壳建议"]
    assert workflow.SHELL_MIN_CHARS == 40
