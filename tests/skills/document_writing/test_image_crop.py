"""`compose crop-img`：置入前裁剪（去刊头/图注/冗余边条）正反例。"""

from __future__ import annotations

from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image", reason="需要 Pillow")

from pysci.skills.document_writing.tools import compose  # noqa: E402


def _sample(tmp_path: Path, name: str = "scan.png") -> Path:
    """100x80：顶部 12 行红（刊头）、中部蓝（正文）、底部 15 行绿（图注）。"""
    img = Image.new("RGB", (100, 80), (0, 0, 255))
    for y in range(12):
        for x in range(100):
            img.putpixel((x, y), (255, 0, 0))
    for y in range(65, 80):
        for x in range(100):
            img.putpixel((x, y), (0, 128, 0))
    path = tmp_path / name
    img.save(path)
    return path


def _px(path: Path, x: int, y: int) -> tuple[int, int, int]:
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((x, y))


def test_strip_top_and_caption_removes_banner_and_footer(tmp_path):
    src = _sample(tmp_path)
    out = tmp_path / "clean.png"
    assert (
        compose.main(
            [
                "crop-img",
                str(src),
                "--strip-top",
                "12",
                "--strip-caption",
                "15",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    with Image.open(out) as im:
        assert im.size == (100, 53)
    assert _px(out, 50, 0) == (0, 0, 255), "顶部红条（刊头）应被裁掉"
    assert _px(out, 50, 52) == (0, 0, 255), "底部绿条（图注）应被裁掉"


def test_ratio_crop_rounds_by_dimension(tmp_path):
    src = _sample(tmp_path)
    out = tmp_path / "ratio.png"
    assert (
        compose.main(
            [
                "crop-img",
                str(src),
                "--top-ratio",
                "0.1",
                "--bottom-ratio",
                "0.2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    with Image.open(out) as im:
        assert im.size == (100, 56), "top=8、bottom=16 → 高 56"


def test_left_right_crop(tmp_path):
    src = _sample(tmp_path)
    out = tmp_path / "lr.png"
    assert (
        compose.main(
            [
                "crop-img",
                str(src),
                "--left",
                "10",
                "--right-ratio",
                "0.1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    with Image.open(out) as im:
        assert im.size == (80, 80)


def test_default_output_does_not_touch_source(tmp_path):
    src = _sample(tmp_path, "fig.png")
    before = src.read_bytes()
    assert compose.main(["crop-img", str(src), "--strip-top", "5"]) == 0
    assert (tmp_path / "fig_cropped.png").exists()
    assert src.read_bytes() == before, "默认输出替换件，不覆盖源文件"


def test_over_crop_is_rejected_without_output(tmp_path, capsys):
    src = _sample(tmp_path)
    out = tmp_path / "bad.png"
    assert (
        compose.main(
            ["crop-img", str(src), "--top", "50", "--bottom", "40", "--out", str(out)]
        )
        == 2
    )
    assert "裁后无剩余" in capsys.readouterr().err
    assert not out.exists()


def test_conflicting_pixel_and_shortcut_is_rejected(tmp_path, capsys):
    src = _sample(tmp_path)
    assert (
        compose.main(
            [
                "crop-img",
                str(src),
                "--top",
                "5",
                "--strip-top",
                "6",
                "--out",
                str(tmp_path / "x.png"),
            ]
        )
        == 2
    )
    assert "只能给一个" in capsys.readouterr().err


def test_pixel_and_ratio_on_same_side_is_rejected(tmp_path, capsys):
    src = _sample(tmp_path)
    assert (
        compose.main(
            [
                "crop-img",
                str(src),
                "--top",
                "5",
                "--top-ratio",
                "0.1",
                "--out",
                str(tmp_path / "x.png"),
            ]
        )
        == 2
    )
    assert "像素与比例只能给一个" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path, capsys):
    src = _sample(tmp_path, "dry.png")
    assert compose.main(["crop-img", str(src), "--strip-top", "12", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "100x80 → 100x68" in out and "dry-run" in out
    assert not (tmp_path / "dry_cropped.png").exists()
