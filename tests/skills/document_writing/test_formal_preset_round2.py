"""学院预设返工第 2 轮入库：§4D–4H 落册、速查表同步、预算（tb-5 返工 2 轮 B.d2）。"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT

REF = ORCHESTRATION_ROOT / "skills" / "document-writing" / "references"


def _read(name: str) -> str:
    return (REF / name).read_text(encoding="utf-8")


def test_focus_paths_volume_covers_4d_to_4h():
    md = _read("ppt-preset-formal-05-focus-paths.md")
    for token in (
        "路径 1（默认",
        "路径 2（回退",
        "只有最外侧一个 6 pt 粗描边",
        "内部对象一律零描边",
        "3 pt（38100 EMU）",
        "默认 16 pt",
        "必须是渲染图",
        "≥300 dpi",
        "crop-img",
        "--strip-top",
        "--strip-caption",
    ):
        assert token in md, token


def test_main_quick_table_and_index_point_to_volume():
    md = _read("ppt-preset-formal.md")
    for row in ("| 焦点块 |", "| 块间距 |", "| 公式 |", "| 裁剪 |"):
        assert row in md, row
    assert "ppt-preset-formal-05-focus-paths.md" in md
    assert "6pt 外框、内部零描边" in md and "≥300dpi" in md


def test_layout_algebra_marks_path2_vocabulary():
    md = _read("ppt-preset-formal-04-layout-algebra.md")
    assert "3 pt" in md and "1.25 pt" not in md, "细边框须为 3 pt"
    assert "路径 2（回退方案）" in md and "路径 1" in md


def test_all_volumes_fit_budget():
    for name in (
        "ppt-preset-formal.md",
        "ppt-preset-formal-02-animation.md",
        "ppt-preset-formal-03-citations-and-acceptance.md",
        "ppt-preset-formal-04-layout-algebra.md",
        "ppt-preset-formal-05-focus-paths.md",
    ):
        assert len((REF / name).read_bytes()) <= 8192, name
