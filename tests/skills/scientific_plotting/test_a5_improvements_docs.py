"""A.a5 防复发 5 项的文档面钉子（tb-20261017-212141-3-3）。

每项至少一处可 grep 的成文条款，并守住 harness 预算；工具面（ANNOTATIONS.yaml 机械核验）
另有端到端测试（``test_annotation_geometry.py``）。
"""

from __future__ import annotations

import inspect

from pysci.paths import ORCHESTRATION_ROOT

R = ORCHESTRATION_ROOT
TEMPLATE = R / "skills" / "orchestration" / "references" / "taskbook-template.md"
RUBRIC = R / "pods" / "reviewer" / "rubrics" / "scientific-plotting.md"
SKILL = R / "skills" / "scientific-plotting" / "SKILL.md"
REF4 = (
    R
    / "skills"
    / "scientific-plotting"
    / "references"
    / "plotting-04-annotation-semantics.md"
)
SIM = R / "pods" / "sim" / "AGENTS.md"

_LAYOUT_RULES = (
    "标题与上方外缘间距 > 与下方子图间距",
    "图例置于数据图下方、不压主体",
    "子图组与脚注段留清晰间距",
    "整体 bbox（含图例/colorbar/标注）左上角",
)


def _t(path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. 处置分支不可替换 + 用户显式指令=禁回退
# ---------------------------------------------------------------------------
def test_template_has_disposal_branch_and_no_fallback_clause():
    md = _t(TEMPLATE)
    assert "处置分支声明" in md, "模板缺「处置分支声明」位"
    assert "处置必须走任务书指定分支" in md
    assert "不得静默替换" in md
    assert "用户显式指令＝硬性禁回退项" in md
    assert "成员技术结论可**呈疑**" in md, "缺「可呈疑、不得否决」的措辞"


def test_rubric_has_h6_disposal_fail_item():
    md = _t(RUBRIC)
    assert "| H6 |" in md, "量表缺 H6 硬项"
    h6 = " ".join(line for line in md.splitlines() if "| H6 |" in line)
    assert "处置分支" in h6 and "用户显式指令" in h6
    assert "上报留痕" in h6 and "FAIL" in h6


# ---------------------------------------------------------------------------
# 2. 参考图逐要素对照表
# ---------------------------------------------------------------------------
def test_template_has_reference_image_column():
    md = _t(TEMPLATE)
    assert "参考图对照" in md, "解构表缺「参考图对照」列"
    assert "照搬/适配/偏离" in md
    assert "无对照表＝缺理解痕迹" in md


def test_skill_and_rubric_hook_reference_table():
    sk = _t(SKILL)
    assert "参考图对照表" in sk and "照搬/适配/偏离" in sk
    md = _t(RUBRIC)
    g2 = md.split("### G2", 1)[1].split("## 硬项", 1)[0]
    assert "参考图逐要素对照表" in g2 and "缺理解痕迹" in g2


# ---------------------------------------------------------------------------
# 3. 标注几何语义机械检查（声明入口 + 分册 schema）
# ---------------------------------------------------------------------------
def test_template_and_skill_declare_annotations_sidecar():
    assert "ANNOTATIONS.yaml" in _t(TEMPLATE)
    sk = _t(SKILL)
    assert "ANNOTATIONS.yaml" in sk
    assert 'gid="annot:' in sk
    assert "plotting-04-annotation-semantics.md" in sk


def test_annotation_semantics_reference_schema():
    assert REF4.is_file(), "缺 plotting-04 分册"
    md = _t(REF4)
    for token in ("counts", "contain", "sizes", "connectors", "margin_ratio"):
        assert token in md, f"分册缺 schema 项：{token}"
    assert "annot-count" in md and "annot-connect" in md, "缺失败码清单"


def test_audit_face_exposes_annotations_input():
    from pysci.skills.scientific_plotting.tools import audit, geometry

    sig = inspect.signature(audit.audit_figure_dir)
    assert "annotations" in sig.parameters
    assert hasattr(geometry, "audit_annotations")
    assert hasattr(geometry, "collect_primitives")


# ---------------------------------------------------------------------------
# 4. sim 经验清单：背景场隔离
# ---------------------------------------------------------------------------
def test_sim_experience_background_field_isolation():
    md = _t(SIM)
    assert "背景" in md and "管槽" in md, "sim 经验缺「背景/管槽」条目"
    assert "不得覆盖散射体内部子域" in md
    assert "均一化内部声压" in md and "超表面失效" in md
    # 与既有两条经验并列
    assert "完美匹配边界" in md
    assert "查手册" in md and "留痕" in md


# ---------------------------------------------------------------------------
# 5. 布局节奏与编号锚定（SKILL + reviewer 各挂钩一次）
# ---------------------------------------------------------------------------
def test_skill_has_layout_rhythm_four_rules():
    sk = _t(SKILL)
    assert "布局节奏/编号锚定" in sk
    for token in _LAYOUT_RULES:
        assert token in sk, f"SKILL 缺布局准则：{token}"


def test_rubric_has_layout_rhythm_four_rules():
    md = _t(RUBRIC)
    d5 = " ".join(line for line in md.splitlines() if "| D5 " in line)
    assert "布局节奏/编号锚定四条" in d5
    for token in _LAYOUT_RULES:
        assert token in d5, f"量表 D5 缺布局准则：{token}"


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------
def test_touched_docs_within_budget():
    for p in (TEMPLATE, RUBRIC, SKILL, REF4, SIM):
        size = p.stat().st_size
        assert size <= 8192, f"{p.name} 超预算：{size}B"
