"""A.a5 五项通用机制（W1–W5）的文档面钉子（tb-20261017-044604-3-2，jury 裁决 A）。

五个盲区面各钉一处「成文且可查」：关系级解构列 / 近而不交词表（reviewer+SKILL 同口径）/
几何可行性触发 / sim 两表义务 / reviewer G2 四项判法；并守住各文件 ≤8192B 预算。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT

R = ORCHESTRATION_ROOT
TEMPLATE = R / "skills" / "orchestration" / "references" / "taskbook-template.md"
RUBRIC = R / "pods" / "reviewer" / "rubrics" / "scientific-plotting.md"
SKILL = R / "skills" / "scientific-plotting" / "SKILL.md"
SHEET = (
    R
    / "skills"
    / "scientific-plotting"
    / "references"
    / "plotting-03-typesetting-and-style.md"
)
SIM = R / "pods" / "sim" / ".qoder" / "rules" / "charter.md"
FIG = R / "pods" / "figure" / ".qoder" / "rules" / "charter.md"
WRIT = R / "pods" / "writing" / ".qoder" / "rules" / "charter.md"


def _t(p) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# W1 解构表升关系级
# ---------------------------------------------------------------------------
def test_w1_template_has_relation_column():
    md = _t(TEMPLATE)
    assert "关系（元素间）" in md, "解构表缺「关系」列头"
    for token in ("包含拓扑", "相对大小", "标注注意力归属"):
        assert token in md, f"关系列缺口径：{token}"


def test_w1_figure_writing_execution_notes_reference_it():
    for p, who in ((FIG, "figure"), (WRIT, "writing")):
        md = _t(p)
        assert "关系" in md and "taskbook-template.md" in md, (
            f"{who} 侧执行说明未引用解构表"
        )


# ---------------------------------------------------------------------------
# W2 近而不交入词表（reviewer H4 与 SKILL 同口径）
# ---------------------------------------------------------------------------
def test_w2_rubric_h4_adds_near_miss():
    h4 = _t(RUBRIC)
    assert "近而不交" in h4, "H4 未增「近而不交」"
    assert "贴边/临界间距" in h4 and "缺陷" in h4, "未把贴边/临界间距定为缺陷"


def test_w2_skill_matches_rubric_wording():
    sk = _t(SKILL)
    assert "近而不交" in sk, "SKILL 排版指导未同步「近而不交」"
    sheet = _t(SHEET)
    assert "近而不交" in sheet and "同级" in sheet, (
        "分册 3 未同步「近而不交（与重叠同级）」"
    )


# ---------------------------------------------------------------------------
# W3 几何可行性核对义务
# ---------------------------------------------------------------------------
def test_w3_template_has_geometry_feasibility_trigger():
    md = _t(TEMPLATE)
    assert "几何可行性核对" in md, "解构表缺几何可行性触发"
    for token in ("无法", "完整容纳", "扩大仿真区域", "禁裁剪"):
        assert token in md, f"几何可行性义务缺要素：{token}"


# ---------------------------------------------------------------------------
# W4 sim 两表义务
# ---------------------------------------------------------------------------
def test_w4_sim_charter_requires_two_tables():
    md = _t(SIM)
    assert "设置依据表" in md and "理论预期预登记" in md, "sim charter 缺两表义务"
    assert "手册推荐" in md, "设置依据表缺「手册推荐设置」列要求"
    assert "外场计算" in md and "必列" in md, "未写明外场计算/边界类必列"


# ---------------------------------------------------------------------------
# W5 reviewer 关系级核对
# ---------------------------------------------------------------------------
def test_w5_g2_expands_four_relation_checks():
    md = _t(RUBRIC)
    g2 = md.split("### G2", 1)[1].split("## 硬项", 1)[0]
    for token in ("包含拓扑", "相对大小", "标注归属", "近而不交"):
        assert token in g2, f"G2 缺关系级判据：{token}"


# ---------------------------------------------------------------------------
# 预算：五处真本/量表 ≤8192B
# ---------------------------------------------------------------------------
def test_all_touched_files_within_budget():
    for p in (TEMPLATE, RUBRIC, SKILL, SHEET, SIM, FIG, WRIT):
        size = p.stat().st_size
        assert size <= 8192, f"{p.name} 超预算：{size}B"
