"""公式 ink 裁剪机械断言：纯函数 ink_bbox（固定 alpha 样例）+ PIL 落盘裁剪。

ink_bbox 是裁剪核心（纯函数，恒跑）；crop_png 的 PNG 编解码走 Pillow
（python-pptx 传递依赖，由并行实施者装入），缺失时跳过落盘用例。
"""

from __future__ import annotations

import pytest

from pysci.skills.document_writing import formula_png

# 固定样例：5x4 全透明缓冲，(1,1)-(2,2) 有 ink
W, H = 5, 4
ALPHA = bytearray(W * H)
for _y in (1, 2):
    for _x in (1, 2):
        ALPHA[_y * W + _x] = 255


def test_ink_bbox_exact_box():
    assert formula_png.ink_bbox(bytes(ALPHA), W, H) == (1, 1, 3, 3)


def test_ink_bbox_padding_clamped():
    assert formula_png.ink_bbox(bytes(ALPHA), W, H, padding=2) == (0, 0, 5, 4)
    # padding=1：右/下不越界，左/上留 1 像素
    assert formula_png.ink_bbox(bytes(ALPHA), W, H, padding=1) == (0, 0, 4, 4)


def test_ink_bbox_threshold():
    soft = bytearray(ALPHA)
    soft[0] = 5  # 左上角微弱 alpha
    assert formula_png.ink_bbox(bytes(soft), W, H, threshold=0) == (0, 0, 3, 3)
    assert formula_png.ink_bbox(bytes(soft), W, H, threshold=10) == (1, 1, 3, 3)


def test_ink_bbox_all_transparent_is_none():
    assert formula_png.ink_bbox(bytes(W * H), W, H) is None


def test_ink_bbox_rejects_bad_shapes():
    with pytest.raises(ValueError, match="长度"):
        formula_png.ink_bbox(b"\x00" * 3, W, H)
    with pytest.raises(ValueError, match="尺寸"):
        formula_png.ink_bbox(b"", 0, H)


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _pil_available(), reason="Pillow 未安装（python-pptx 传递依赖）")
def test_crop_png_round_trip(tmp_path):
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("RGBA", (W, H), (255, 255, 255, 0)).save(src)
    img = Image.open(src)
    pixels = img.load()
    for y in (1, 2):
        for x in (1, 2):
            pixels[x, y] = (0, 0, 0, 255)
    img.save(src)
    out = formula_png.crop_png(src, tmp_path / "out.png", padding=1)
    with Image.open(out) as cropped:
        assert cropped.size == (4, 4)  # (0,0)-(3,3) 开区间
        assert cropped.convert("RGBA").getpixel((0, 0))[3] == 0  # 留白透明


@pytest.mark.skipif(not _pil_available(), reason="Pillow 未安装（python-pptx 传递依赖）")
def test_crop_png_all_transparent_raises(tmp_path):
    from PIL import Image

    src = tmp_path / "blank.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(src)
    with pytest.raises(formula_png.FormulaError, match="空输出"):
        formula_png.crop_png(src, tmp_path / "out.png")


def test_render_formula_calls_build_with_run_lint(monkeypatch, tmp_path):
    # 回归（2026-11-17 冒烟）：render_formula 曾误传 build(lint=False)（参数名是
    # run_lint），TypeError 直接炸穿 CLI。这里桩掉 build，签名不匹配即失败。
    from pysci.skills.document_writing import latex_workflow

    seen: dict[str, object] = {}

    def fake_build(tex_path, **kwargs):
        seen.update(kwargs)
        return latex_workflow.BuildResult(
            ok=False, pdf=None, issues=(), log_text="", returncode=1
        )

    monkeypatch.setattr(latex_workflow, "build", fake_build)
    with pytest.raises(formula_png.FormulaError, match="编译失败"):
        formula_png.render_formula("E=mc^2", tmp_path / "f.png", inline=True)
    assert seen == {"run_lint": False}
