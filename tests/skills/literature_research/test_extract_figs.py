"""图区提取补缺：上限可配 / 原文图号映射 / 批注 bbox 落参 / 整页提示。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="需要 PyMuPDF")
Image = pytest.importorskip("PIL.Image", reason="需要 Pillow")

from pysci.paths import PROJECT_ROOT  # noqa: E402

_SCRIPT = PROJECT_ROOT / "orchestration" / "pods" / "lit" / "bench" / "extract_figs.py"


def _module():
    spec = importlib.util.spec_from_file_location("extract_figs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _pdf(tmp_path: Path, pages: int, *, full_page_last: bool = False) -> Path:
    png = tmp_path / "img.png"
    Image.new("RGB", (300, 300), (30, 80, 140)).save(png)
    doc = fitz.open()
    for n in range(1, pages + 1):
        page = doc.new_page(width=612, height=792)
        if full_page_last and n == pages:
            rect = fitz.Rect(10, 20, 600, 760)
        else:
            rect = fitz.Rect(100, 100, 400, 400)
        page.insert_image(rect, filename=str(png))
        page.insert_text((105, rect.y1 + 18), f"FIG. {n}. caption text", fontsize=11)
    out = tmp_path / "figs.pdf"
    doc.save(str(out))
    doc.close()
    return out


def test_default_cap_opens_beyond_six_and_maps_figure_numbers(tmp_path):
    mod = _module()
    man = mod.extract(_pdf(tmp_path, 8), tmp_path / "out")
    assert len(man) == 8, "默认上限应放开 6"
    assert [m["figure_no"] for m in man] == [f"Fig. {i}" for i in range(1, 9)]
    assert all(m["file"] == f"fig{i}.png" for i, m in enumerate(man, 1))
    assert (tmp_path / "out" / "fig8.png").exists()


def test_max_figs_zero_means_unlimited(tmp_path):
    mod = _module()
    assert len(mod.extract(_pdf(tmp_path, 8), tmp_path / "out", 0)) == 8
    assert len(mod.extract(_pdf(tmp_path, 8), tmp_path / "out3", 3)) == 3


def test_bbox_override_replaces_slot_and_marks_override(tmp_path):
    mod = _module()
    out = tmp_path / "out"
    man = mod.extract(
        _pdf(tmp_path, 3), out, bboxes={1: (2, 120.0, 120.0, 380.0, 380.0)}
    )
    assert man[0]["page"] == 2 and man[0]["bbox"] == [120.0, 120.0, 380.0, 380.0]
    assert man[0]["override"] is True and "bbox" in man[0]["hint"]
    assert man[0]["px"] != man[1]["px"] or man[0]["page"] == 2


def test_full_page_candidate_gets_hint(tmp_path):
    mod = _module()
    man = mod.extract(_pdf(tmp_path, 2, full_page_last=True), tmp_path / "out")
    full = [m for m in man if m.get("hint", "").startswith("整页图")]
    assert full, man
