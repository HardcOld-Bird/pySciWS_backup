"""`compose slides geometry` 的几何体检回归（缺陷 1：遮挡/越界/图片互叠的机械原料）。

渲染闸只防「没渲染 / 渲染成空白」，命中不了「渲染出来了但版面是坏的」——本文件钉住几何端
的三类硬判据：外框边被更高 z 序图片**覆盖**、图元**越界**（画布/外框）、**图片互叠**面积。
用临时构造的迷你 pptx 制造可控缺陷，断言三条各有命中，且该命令**只读不写盘**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pptx = pytest.importorskip("pptx", reason="需要 python-pptx")
Image = pytest.importorskip("PIL.Image", reason="需要 Pillow")

from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.util import Inches  # noqa: E402

from pysci.skills.document_writing.tools import compose, slides_visual  # noqa: E402


def _png(path: Path, color: tuple[int, int, int] = (30, 90, 200)) -> Path:
    Image.new("RGB", (60, 40), color).save(path)
    return path


def _defective_deck(tmp_path: Path) -> Path:
    """构造三类缺陷各一的迷你 deck：外框、压外框上边的图、互叠的两图、越出画布的矩形。"""
    prs = Presentation()  # 默认 10x7.5 in 画布
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    frame = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(8), Inches(5)
    )
    frame.name = "外框测试"
    img = _png(tmp_path / "shot.png")
    # 压住外框上边（x 6–8 段）
    slide.shapes.add_picture(str(img), Inches(6), Inches(0.5), Inches(2), Inches(2))
    # 两张互叠（交叠 1.5x1.5 = 2.25 in²）
    slide.shapes.add_picture(str(img), Inches(2), Inches(2), Inches(3), Inches(2))
    slide.shapes.add_picture(str(img), Inches(3.5), Inches(2.5), Inches(3), Inches(2))
    # 越出画布右侧
    slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(9), Inches(6), Inches(3), Inches(3)
    )
    deck = tmp_path / "defective.pptx"
    prs.save(str(deck))
    return deck


def _clean_deck(tmp_path: Path) -> Path:
    """无缺陷对照：只有一个居中矩形 + 一张位于其内的图。"""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(8), Inches(5)
    )
    slide.shapes.add_picture(
        str(_png(tmp_path / "ok.png")), Inches(2), Inches(2), Inches(2), Inches(2)
    )
    deck = tmp_path / "clean.pptx"
    prs.save(str(deck))
    return deck


def test_geometry_reports_edge_coverage_overlap_and_overflow(tmp_path):
    rep = slides_visual.geometry_report(_defective_deck(tmp_path))
    sg = rep.slides[0]
    assert sg.frame is not None and sg.frame_source == "auto"
    assert sg.frame.name == "外框测试"
    covered = [e for e in sg.edges if e.frac > 0]
    assert any(e.edge == "上边" and e.frac > 0.2 for e in covered), sg.edges
    assert sg.overlaps, "互叠的两张图未被报出"
    assert sg.off_canvas, "越出画布的矩形未被报出"
    assert rep.violations > 0 and not rep.ok


def test_clean_deck_has_no_violations(tmp_path):
    rep = slides_visual.geometry_report(_clean_deck(tmp_path))
    assert rep.ok and rep.violations == 0, rep.summary()


def test_explicit_frame_name_selects_the_frame(tmp_path):
    deck = _defective_deck(tmp_path)
    rep = slides_visual.geometry_report(deck, frame="外框测试")
    assert rep.slides[0].frame_source == "explicit"
    # 名字不存在时不臆造外框（只报越界/互叠）
    rep2 = slides_visual.geometry_report(deck, frame="不存在的外框")
    assert rep2.slides[0].frame is None and rep2.slides[0].edges == []


def test_off_frame_only_judges_pictures(tmp_path):
    """外框之外的标题文本框不算「越出外框」（原型期实测的假阳性面）。"""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1), Inches(1), Inches(8), Inches(5)
    )
    tb = slide.shapes.add_textbox(Inches(1), Inches(0.2), Inches(4), Inches(0.5))
    tb.text_frame.text = "标题（在外框上方，属设计）"
    deck = tmp_path / "titled.pptx"
    prs.save(str(deck))
    rep = slides_visual.geometry_report(deck)
    assert rep.slides[0].off_frame == [], rep.slides[0].off_frame


def test_cli_geometry_reports_only_by_default_and_strict_fails(tmp_path, capsys):
    deck = _defective_deck(tmp_path)
    assert compose.main(["slides", "geometry", str(deck)]) == 0  # 默认只报告
    out = capsys.readouterr().out
    assert "外框" in out and "违规" in out
    assert compose.main(["slides", "geometry", str(deck), "--strict"]) == 1
    assert (
        compose.main(["slides", "geometry", str(_clean_deck(tmp_path)), "--strict"])
        == 0
    )


def test_cli_geometry_json_is_machine_readable(tmp_path, capsys):
    import json

    deck = _defective_deck(tmp_path)
    assert compose.main(["slides", "geometry", str(deck), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["pptx"].endswith("defective.pptx")
    assert data["violations"] > 0 and data["slides"][0]["frame"]["name"] == "外框测试"


def test_geometry_is_read_only(tmp_path):
    """只读纪律：几何体检**不写盘**（受审目录只读时也照跑）。"""
    deck = _defective_deck(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    slides_visual.geometry_report(deck)
    compose.main(["slides", "geometry", str(deck), "--json"])
    assert {p.name for p in tmp_path.iterdir()} == before


def test_render_prints_ready_to_paste_verify_command(tmp_path, capsys):
    """缺陷 2 的用法面：渲染后打印可直接粘贴的 verify 命令（不必记 --render-dir 旗标名）。"""
    deck = _clean_deck(tmp_path)
    rc = compose.main(["slides", "render", str(deck), "--out", str(tmp_path / "r")])
    if rc == 3:  # 无 LibreOffice：渲染链不可用，跳过该断言
        pytest.skip("未检测到 LibreOffice")
    out = capsys.readouterr().out
    assert "--render-dir" in out and "slides verify" in out
