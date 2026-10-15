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
