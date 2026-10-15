"""「绘图色系克制」通用审美标准入 harnessing（backlog 20261015-212300，**用户裁决**）。

用户裁决：整图**设计用色**的色系数量克制，一般 **1~2 个色系**（同图红黄蓝绿并用显杂乱）。
补充（用户指令，适用边界以此为准）：**数据 colormap 豁免色系计数，但不豁免相协调义务**——
Rainbow 类 colormap 为必选项时，设计用色应取自其**同族或协调中性色**（灰阶/暖冷同谱点缀），
**尽量避免另起色系**；「1~2 色系」按「数据 colormap 同族 + 至多 1 个协调点缀色系」执行。

三处落点（各自 harness 面）：reviewer 量表（**可执行判定**）/ ai-drawing SKILL / scientific-plotting SKILL。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT

SKILLS = ORCHESTRATION_ROOT / "skills"
SURFACES = {
    "reviewer-rubric": ORCHESTRATION_ROOT
    / "pods/reviewer/rubrics/scientific-plotting.md",
    "ai-drawing-skill": SKILLS / "ai-drawing" / "SKILL.md",
    "sciplot-skill": SKILLS / "scientific-plotting" / "SKILL.md",
}


def _text(rel: str) -> str:
    return SURFACES[rel].read_text(encoding="utf-8")


@pytest.mark.parametrize("name", list(SURFACES))
def test_surface_carries_the_rule(name):
    """三处各有一处「1~2 色系克制」表述（criteria：grep「色系」命中 ≥3）。"""
    md = _text(name)
    assert "色系" in md, f"{name} 未含该标准"
    assert "1~2" in md, f"{name} 未写明 1~2 色系"


@pytest.mark.parametrize("name", list(SURFACES))
def test_surface_states_the_colormap_boundary(name):
    """三处都写明适用边界：数据 colormap **不计入**色数，但设计用色须与其**协调**。"""
    md = _text(name)
    assert "colormap" in md
    assert "不计入" in md
    assert "同族" in md and "中性" in md
    assert "勿另起色系" in md


def test_rubric_rule_is_executable():
    """reviewer 量表可执行：给出**计数口径**（数设计用色、不数数据 colormap）与协调义务。"""
    md = _text("reviewer-rubric")
    assert "色系计数口径" in md
    assert "不数数据 colormap" in md
    assert "豁免计数但不免协调义务" in md
    assert "至多 1 个协调点缀色系" in md


def test_rule_sits_in_d5_row():
    """该条落在 D5（评分维度，不阻断 PASS）——与「记录用、低分不阻断」的量表纪律一致。"""
    md = _text("reviewer-rubric")
    row = next(ln for ln in md.splitlines() if ln.startswith("| D5 "))
    assert "色系" in row and "设计用色" in row


def test_all_three_surfaces_within_budget():
    """三处都要过 8KB 真本/量表层预算（落盘字节口径）。"""
    for name, p in SURFACES.items():
        size = p.stat().st_size
        assert size <= 8192, f"{name} 超预算：{size}B"


def test_grep_hits_at_least_three_files():
    """criteria 的字面口径：三处 harness 合计 grep「色系」命中 ≥3。"""
    hits = sum(1 for p in SURFACES.values() if "色系" in p.read_text(encoding="utf-8"))
    assert hits >= 3, f"仅 {hits} 处含该标准"


def test_surfaces_are_the_declared_paths():
    """落点自检：三处路径都真实存在（防「写进不存在的文件」）。"""
    for name, p in SURFACES.items():
        assert Path(p).is_file(), f"{name} 路径不存在：{p}"
