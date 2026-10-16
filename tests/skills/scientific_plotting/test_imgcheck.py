"""`figures inspect`：不依赖管线的图像体检（backlog 20261015-204912 §4.4）。

供**非 scaffold 产物**用——那类图目录 `audit` 恒 N/A（返回 applicable=False），审查四步的
「步骤 2 工具核验」此前只剩 doctor。本命令补上：尺寸 / 色域占用 / 饱和像素占比 /
**边缘水印·logo 检测**（I.i3 P2 的 COMSOL logo 应在样例上报出）。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pysci.paths import PROJECT_ROOT
from pysci.skills.scientific_plotting.tools import figures, imgcheck

#: I.i3 P2 的带 logo 真实样例；缺失则相关用例 skip。
REAL_BRANDED = (
    PROJECT_ROOT
    / "data/research/1_gain_ep/article/figures/beam_scatter/raw/far_full_L.png"
)

BRAND_BLUE = (100, 144, 200)
BRAND_YELLOW = (240, 240, 96)


def _logo_canvas(w=400, h=300, bg=(255, 255, 255)) -> np.ndarray:
    """贴右上角的「蓝+黄相邻」方块（仿真真实水印：小、贴边、两色连通成一块）。"""
    a = np.full((h, w, 3), bg, dtype=np.uint8)
    a[4:17, w - 18 : w - 2] = BRAND_BLUE
    a[4:17, w - 32 : w - 18] = BRAND_YELLOW
    return a


def _save(arr, p):
    Image.fromarray(arr).save(p)
    return p


# ---------------------------------------------------------------------------
# 体检读数（纯函数）
# ---------------------------------------------------------------------------
def test_clean_image_has_no_edge_marks(tmp_path):
    p = _save(np.full((300, 400, 3), 255, dtype=np.uint8), tmp_path / "clean.png")
    health, rc = imgcheck.inspect_image(p)
    assert rc == 0
    assert health["size_px"] == [400, 300]
    assert health["edge_marks"] == []
    assert health["colored_ratio"] == 0.0


def test_detects_brand_logo_at_corner(tmp_path):
    p = _save(_logo_canvas(), tmp_path / "logo.png")
    health, rc = imgcheck.inspect_image(p)

    assert rc == 1
    assert len(health["edge_marks"]) == 1
    m = health["edge_marks"][0]
    assert m["brand_like"] is True
    assert m["area"] >= imgcheck.MIN_MARK_AREA
    assert m["bbox"][2] > 300  # 落在右上角


def test_content_in_middle_is_not_an_edge_mark(tmp_path):
    """图内数据不落在边缘带 → 不报（防把正常图误报成带水印）。"""
    a = np.full((400, 600, 3), 255, dtype=np.uint8)
    a[150:260, 200:400] = (200, 40, 40)  # 中部大色块
    p = _save(a, tmp_path / "mid.png")
    health, rc = imgcheck.inspect_image(p)
    assert rc == 0 and health["edge_marks"] == []
    assert health["colored_ratio"] > 0.0  # 但色域占用读数如实反映


def test_detects_mark_on_dark_background(tmp_path):
    """底色自适应：黑底上的彩块同样命中（底色取自边缘带低色度像素）。"""
    p = _save(_logo_canvas(bg=(16, 16, 16)), tmp_path / "dark.png")
    health, rc = imgcheck.inspect_image(p)
    assert rc == 1
    assert health["edge_marks"][0]["bg"] == [16, 16, 16]


def test_small_speck_is_ignored(tmp_path):
    """单像素/毛刺 < MIN_MARK_AREA → 不报（抗锯齿噪声不是水印）。"""
    a = np.full((300, 400, 3), 255, dtype=np.uint8)
    a[3, 397] = (255, 0, 0)
    p = _save(a, tmp_path / "speck.png")
    assert imgcheck.inspect_image(p)[1] == 0


def test_missing_file_returns_rc2(tmp_path):
    assert imgcheck.inspect_image(tmp_path / "nope.png") == ({}, 2)


def test_report_lines_cover_required_items(tmp_path):
    p = _save(_logo_canvas(), tmp_path / "logo.png")
    health, _ = imgcheck.inspect_image(p)
    text = "\n".join(imgcheck.report_lines(health))
    for token in ("尺寸", "色域占用", "饱和像素占比", "边缘标记"):
        assert token in text, f"体检报告缺项：{token}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_inspect_reports_logo_and_rc1(tmp_path, capsys):
    p = _save(_logo_canvas(), tmp_path / "logo.png")
    assert figures.main(["inspect", str(p)]) == 1
    out = capsys.readouterr().out
    assert "蓝黄成对" in out and "有边缘标记" in out


def test_cli_inspect_clean_is_rc0(tmp_path, capsys):
    p = _save(np.full((200, 200, 3), 255, dtype=np.uint8), tmp_path / "c.png")
    assert figures.main(["inspect", str(p)]) == 0
    assert "边缘带无标记" in capsys.readouterr().out


def test_cli_inspect_bad_path_rc2(tmp_path, capsys):
    assert figures.main(["inspect", str(tmp_path / "nope.png")]) == 2
    assert "图像读不到" in capsys.readouterr().out


@pytest.mark.skipif(
    not REAL_BRANDED.exists(), reason="真实带 logo 素材缺失（worktree 无 data/）"
)
def test_real_i3_sample_reports_logo_hit():
    """I.i3 P2 的样例必须被报出（criteria：对带 logo 样例能报出 logo 命中）。"""
    health, rc = imgcheck.inspect_image(REAL_BRANDED)
    assert rc == 1
    assert any(m["brand_like"] for m in health["edge_marks"])


# ---------------------------------------------------------------------------
# 等效物理尺寸 + tight 裁切提示（backlog 20261015-221750）
#   审查员核验「生产者自述 mm 尺寸 vs 成品实测」时，不该被迫反解版面常量、核对样式源码。
# ---------------------------------------------------------------------------
def test_read_dpi_reads_png_metadata(tmp_path):
    p = tmp_path / "600.png"
    Image.fromarray(np.full((10, 10, 3), 255, dtype=np.uint8)).save(p, dpi=(600, 600))
    # PNG pHYs 存整数「像素/米」→ 回读 dpi 天然近似（600 dpi = 23622 ppm）
    assert imgcheck.read_dpi(p) == pytest.approx(600, abs=0.01)


def test_read_dpi_absent_or_broken_is_none(tmp_path):
    p = _save(np.full((10, 10, 3), 255, dtype=np.uint8), tmp_path / "nodpi.png")
    assert imgcheck.read_dpi(p) is None, "无 dpi 元数据须回 None（不猜）"
    assert imgcheck.read_dpi(tmp_path / "nope.png") is None


def test_size_mm_math_matches_the_reported_case():
    """实测案例：4178×2461 px @600 dpi → 176.9×104.2 mm（= 建议单里那份产物的读数）。"""
    h = imgcheck.image_health(np.zeros((2461, 4178, 3), dtype=np.uint8), dpi=600)
    assert h["dpi"] == 600
    assert h["size_mm"] == [176.9, 104.2]


def test_size_mm_absent_without_dpi():
    h = imgcheck.image_health(np.zeros((10, 10, 3), dtype=np.uint8))
    assert h["dpi"] is None and h["size_mm"] is None


def test_report_says_unknown_when_dpi_missing(tmp_path):
    p = _save(np.full((10, 10, 3), 255, dtype=np.uint8), tmp_path / "nodpi.png")
    health, _ = imgcheck.inspect_image(p)
    text = "\n".join(imgcheck.report_lines(health))
    assert "等效物理尺寸" in text and "未知" in text and "dpi 元数据" in text


def test_tight_bbox_hint_only_when_declared(tmp_path):
    p = tmp_path / "300.png"
    Image.fromarray(np.full((300, 400, 3), 255, dtype=np.uint8)).save(p, dpi=(300, 300))
    health, _ = imgcheck.inspect_image(p)

    with_hint = "\n".join(imgcheck.report_lines(health, tight_bbox=True))
    without = "\n".join(imgcheck.report_lines(health))

    assert "等效物理尺寸    : 33.9×25.4 mm" in with_hint
    assert "savefig.bbox=tight 已生效" in with_hint
    assert "裁切后包围盒" in with_hint and "figsize 名义值" in with_hint
    assert "savefig.bbox" not in without, "未声明 tight 时不许出裁切提示（防噪声）"


def test_style_is_the_single_source_of_the_hint():
    """提示真源＝绘图样式契约（`style.savefig_bbox()`），不在体检面写死常量。"""
    from pysci.skills.scientific_plotting.tools import style

    assert style.savefig_bbox() == "tight"


def test_cli_inspect_reports_mm_and_tight_hint(tmp_path, capsys):
    p = tmp_path / "300.png"
    Image.fromarray(np.full((300, 400, 3), 255, dtype=np.uint8)).save(p, dpi=(300, 300))
    assert figures.main(["inspect", str(p)]) == 0
    out = capsys.readouterr().out
    assert "尺寸            : 400×300 px" in out
    assert "等效物理尺寸    : 33.9×25.4 mm" in out
    assert "savefig.bbox=tight 已生效" in out, "本仓样式声明 tight → 命令行须出裁切提示"
