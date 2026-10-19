"""图区提取补缺：上限可配 / 原文图号映射 / 批注 bbox 落参 / 整页提示。"""

from __future__ import annotations

import importlib.util
import json
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


# ---------------------------------------------------------------------------
# R4 提取器硬化（backlog 20261019-115501/120811/121447）
# ---------------------------------------------------------------------------
def _img(tmp_path, name="figimg.png", size=(300, 300), color=(30, 80, 140)):
    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return p


def _save(doc, tmp_path, name="synthetic.pdf"):
    out = tmp_path / name
    doc.save(str(out))
    doc.close()
    return out


def test_header_band_and_dense_text_are_not_figures(tmp_path):
    """首页刊头/作者块（题注缺失）不得成为 figs。"""
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    p1 = doc.new_page(width=612, height=792)
    p1.draw_rect(fitz.Rect(20, 20, 592, 120), color=(1, 0, 0), fill=(1, 0, 0))
    for i in range(6):
        p1.insert_text(
            (40, 45 + i * 14),
            "Author affiliation and abstract header text " * 2,
            fontsize=9,
        )
    p2 = doc.new_page(width=612, height=792)
    p2.insert_image(fitz.Rect(100, 100, 400, 400), filename=str(img))
    p2.insert_text((105, 420), "FIG. 1. A real figure.", fontsize=11)
    man = mod.extract(_save(doc, tmp_path), tmp_path / "out")
    assert [m["figure_no"] for m in man] == ["Fig. 1"], man
    assert all(m["page"] == 2 for m in man)


def test_full_page_text_scan_is_rejected(tmp_path):
    """整页正文扫描（含正文/批注）不得当图；同 PDF 的真图照常提取。"""
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    p2 = doc.new_page(width=612, height=792)
    p2.draw_rect(fitz.Rect(10, 20, 600, 760), color=(0, 0, 0), width=1)
    for i in range(40):
        p2.insert_text(
            (25, 40 + i * 16),
            "Body text body text body text body text " * 2,
            fontsize=9,
        )
    p3 = doc.new_page(width=612, height=792)
    p3.insert_image(fitz.Rect(100, 100, 400, 400), filename=str(img))
    p3.insert_text((105, 420), "FIG. 1. A real figure.", fontsize=11)
    out = tmp_path / "out"
    man = mod.extract(_save(doc, tmp_path), out)
    assert [m["figure_no"] for m in man] == ["Fig. 1"], man
    audit = json.loads((out / "figures-audit.json").read_text(encoding="utf-8"))
    reasons = {r["reason"] for r in audit["rejected"]}
    assert "text-heavy" in reasons or "full-page-text" in reasons, audit["rejected"]


def test_panels_with_24pt_gap_merge_into_one_figure(tmp_path):
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(100, 100, 200, 200), filename=str(img))
    page.insert_image(fitz.Rect(224, 100, 324, 200), filename=str(img))
    page.insert_text((100, 230), "FIG. 1. Two panels.", fontsize=11)
    man = mod.extract(_save(doc, tmp_path), tmp_path / "out")
    assert len(man) == 1, man
    x0, _, x1, _ = man[0]["bbox"]
    assert x0 <= 100 and x1 >= 324, man[0]["bbox"]


def test_small_right_column_figure_is_included(tmp_path):
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(470, 100, 570, 190), filename=str(img))
    page.insert_text((470, 210), "FIG. 1. Small right-column figure.", fontsize=11)
    man = mod.extract(_save(doc, tmp_path), tmp_path / "out")
    assert len(man) == 1 and man[0]["figure_no"] == "Fig. 1", man


def test_axis_labels_near_figure_are_included(tmp_path):
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(120, 120, 300, 300), filename=str(img))
    page.insert_text((95, 210), "0.5", fontsize=9)  # y 轴刻度（图区左侧）
    page.insert_text((150, 315), "N", fontsize=9)  # x 轴刻度（图区下方）
    page.insert_text((120, 340), "FIG. 1. Axis labels.", fontsize=11)
    man = mod.extract(_save(doc, tmp_path), tmp_path / "out")
    assert len(man) == 1, man
    x0, _, _, y1 = man[0]["bbox"]
    assert x0 <= 95 and y1 >= 315, man[0]["bbox"]


def test_audit_flags_missing_caption_figure(tmp_path):
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(100, 100, 400, 400), filename=str(img))
    page.insert_text((105, 420), "FIG. 1. Present.", fontsize=11)
    page.insert_text((105, 450), "FIG. 2. Missing image.", fontsize=11)
    pdf = _save(doc, tmp_path)
    man = mod.extract(pdf, tmp_path / "out")
    audit = mod.audit_figures(pdf, man)
    assert not audit["ok"]
    assert any(
        i["code"] == "missing-figures" and "Fig. 2" in i["detail"]
        for i in audit["issues"]
    )


def test_audit_flags_doi_mismatch(tmp_path):
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    doc.set_metadata({"subject": "doi:10.1234/wrong"})
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(100, 100, 400, 400), filename=str(img))
    page.insert_text((105, 420), "FIG. 1. Present.", fontsize=11)
    pdf = _save(doc, tmp_path)
    man = mod.extract(pdf, tmp_path / "out")
    audit = mod.audit_figures(pdf, man, expected_doi="10.1234/right")
    assert not audit["ok"]
    assert any(i["code"] == "doi-mismatch" for i in audit["issues"]), audit["issues"]


def test_audit_flags_label_topic_mismatch(tmp_path):
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 60), "A news feature about symmetry in nature.", fontsize=11)
    page.insert_image(fitz.Rect(100, 100, 400, 400), filename=str(img))
    page.insert_text((105, 420), "FIG. 1. Present.", fontsize=11)
    pdf = _save(doc, tmp_path)
    man = mod.extract(pdf, tmp_path / "out")
    audit = mod.audit_figures(pdf, man, label="PT对称性")
    assert not audit["ok"]
    assert any(i["code"] == "label-mismatch" for i in audit["issues"]), audit["issues"]


def test_audit_warnings_do_not_block(tmp_path):
    """无题注的新闻图只给 warning（不阻断）；label 不声称主题时 audit 仍 ok。"""
    mod = _module()
    img = _img(tmp_path)
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_image(fitz.Rect(100, 100, 400, 400), filename=str(img))
    pdf = _save(doc, tmp_path)
    man = mod.extract(pdf, tmp_path / "out")
    audit = mod.audit_figures(pdf, man)
    assert audit["ok"], audit["issues"]
    assert any(i["code"] == "no-captions" for i in audit["issues"])
