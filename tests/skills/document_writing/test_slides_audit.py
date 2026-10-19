"""`compose slides audit` 与字体全槽 API：正反例（装饰/ea/图槽/公式/引用框/动画）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pptx = pytest.importorskip("pptx", reason="需要 python-pptx")
Image = pytest.importorskip("PIL.Image", reason="需要 Pillow")
yaml = pytest.importorskip("yaml", reason="需要 PyYAML")

from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from pysci.skills.document_writing.tools import (  # noqa: E402
    compose,
    slides_audit,
    slides_fonts,
)

_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
_TIMING = (
    f'<p:timing xmlns:p="{_P_NS}"><p:tnLst><p:par><p:cTn id="1"><p:childTnLst>'
    '<p:seq nextAc="seek"><p:cTn id="2"><p:childTnLst><p:par><p:cTn id="3">'
    '<p:stCondLst><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>'
    "</p:stCondLst></p:cTn></p:par></p:childTnLst></p:cTn></p:seq>"
    "</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>"
)


def _png(path: Path, color=(10, 20, 30)) -> Path:
    Image.new("RGB", (64, 48), color).save(path)
    return path


def _set_dist(shape) -> None:
    for p in shape.text_frame.paragraphs:
        p.alignment = PP_ALIGN.DISTRIBUTE


def _deck(tmp_path: Path, *, break_decoration=False, break_ea=False, wrong_pic=False):
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    card = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.1), Inches(0.1), Inches(13.0), Inches(7.3)
    )
    card.name = "白卡"
    title = s.shapes.add_textbox(Inches(0.2), Inches(0.2), Inches(1.91), Inches(0.572))
    title.text_frame.text = "研究背景"
    _set_dist(title)
    cap = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.2), Inches(1.2), Inches(2.0175), Inches(0.438)
    )
    cap.name = "胶囊"
    cap.text_frame.text = "PT对称"
    _set_dist(cap)
    body = s.shapes.add_textbox(Inches(0.2), Inches(2.0), Inches(5.0), Inches(1.0))
    body.text_frame.text = "正文一行"
    ref = s.shapes.add_textbox(Inches(0.1), Inches(6.5), Inches(5.0), Inches(0.6))
    ref.text_frame.text = "[1] 文献甲\n[2] 文献乙"
    _png(tmp_path / "slot.png")
    if wrong_pic:
        _png(tmp_path / "other.png", (200, 0, 0))
    pic = s.shapes.add_picture(
        str(tmp_path / ("other.png" if wrong_pic else "slot.png")),
        Inches(6.0),
        Inches(2.0),
        Inches(3.0),
        Inches(2.0),
    )
    for sh in (title, cap, body, ref):
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                slides_fonts.set_font_slots(
                    r,
                    latin="汉仪大宋简",
                    ea="汉仪大宋简",
                    cs="汉仪大宋简",
                    panose="02010600030101010101",
                )
    if break_ea:
        r = title.text_frame.paragraphs[0].runs[0]
        ea = r._r.get_or_add_rPr().find(qn("a:ea"))
        if ea is not None:
            r._r.get_or_add_rPr().remove(ea)
    if break_decoration:
        card._element.getparent().remove(card._element)
    s.element.append(slides_audit_yaml_element())
    deck = tmp_path / f"deck-{break_decoration}-{break_ea}-{wrong_pic}.pptx"
    prs.save(str(deck))
    return deck, {
        "card": card.shape_id if not break_decoration else None,
        "title": title.shape_id,
        "cap": cap.shape_id,
        "body": body.shape_id,
        "ref": ref.shape_id,
        "pic": pic.shape_id,
    }


def slides_audit_yaml_element():
    from lxml import etree

    return etree.fromstring(_TIMING)


def _spec(tmp_path: Path, ids: dict) -> Path:
    md5 = hashlib.md5((tmp_path / "slot.png").read_bytes()).hexdigest()
    doc = {
        "deck": "test",
        "pages": [
            {
                "page": 1,
                "decoration": [
                    {"name": "白卡", "box_in": [0.1, 0.1, 13.1, 7.4]},
                ],
                "fonts": [
                    {"role": "标题", "shapes": [ids["title"]]},
                    {"role": "胶囊", "shapes": [ids["cap"]]},
                    {"role": "正文", "shapes": [ids["body"], ids["ref"]]},
                ],
                "slots": [
                    {
                        "shape": ids["pic"],
                        "figure_no": "Fig. 7",
                        "anchor": "RPP 86 Fig.7 场分布",
                        "md5": md5,
                        "px": [64, 48],
                        "source": "slot.png",
                    }
                ],
                "titles": [{"shape": ids["title"]}],
                "capsules": [{"shape": ids["cap"], "cite_chars": 3}],
                "refs": [{"ref": ids["ref"], "anchor": ids["card"] or 2}],
                "animation": True,
                "manual": ["装饰质感（目检）"],
            }
        ],
    }
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return p


def test_font_api_writes_all_slots_and_panose():
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1))
    tb.text_frame.text = "模板"
    src = tb.text_frame.paragraphs[0].runs[0]
    slides_fonts.set_font_slots(src, latin="L1", ea="E1", cs="C1", panose="P1")
    tb2 = s.shapes.add_textbox(Inches(1), Inches(2), Inches(2), Inches(1))
    tb2.text_frame.text = "目标"
    dst = tb2.text_frame.paragraphs[0].runs[0]
    dst.font.name = "only-latin"
    slides_fonts.copy_font_slots(src, dst)
    slots = slides_fonts.font_slots(dst)
    assert slots["latin"]["typeface"] == "L1"
    assert slots["ea"]["typeface"] == "E1" and slots["cs"]["typeface"] == "C1"
    assert slots["ea"]["panose"] == "P1"
    assert slides_fonts.slots_complete(slots)


def test_audit_positive_deck_passes(tmp_path):
    deck, ids = _deck(tmp_path)
    rep = slides_audit.audit_deck(deck, _spec(tmp_path, ids))
    assert rep.ok, rep.summary()
    assert rep.manual and "目检" in rep.manual[0].detail
    assert any(i.name.startswith("图槽") and i.status == "pass" for i in rep.items)


def test_audit_negative_decoration_missing(tmp_path):
    deck, ids = _deck(tmp_path, break_decoration=True)
    rep = slides_audit.audit_deck(deck, _spec(tmp_path, ids))
    assert not rep.ok and any("未找到匹配形状" in i.detail for i in rep.failures)


def test_audit_negative_missing_ea(tmp_path):
    deck, ids = _deck(tmp_path, break_ea=True)
    rep = slides_audit.audit_deck(deck, _spec(tmp_path, ids))
    assert any("缺槽" in i.detail and "ea" in i.detail for i in rep.failures), (
        rep.summary()
    )


def test_audit_negative_slot_md5_mismatch(tmp_path):
    deck, ids = _deck(tmp_path, wrong_pic=True)
    rep = slides_audit.audit_deck(deck, _spec(tmp_path, ids))
    assert any("md5" in i.detail and "≠" in i.detail for i in rep.failures), (
        rep.summary()
    )


def test_audit_cli_strict_and_json(tmp_path, capsys):
    deck, ids = _deck(tmp_path)
    spec = _spec(tmp_path, ids)
    assert (
        compose.main(["slides", "audit", str(deck), "--spec", str(spec), "--strict"])
        == 0
    )
    assert "PASS" in capsys.readouterr().out
    bad, bad_ids = _deck(tmp_path, break_ea=True)
    assert (
        compose.main(["slides", "audit", str(bad), "--spec", str(spec), "--json"]) == 0
    )
    import json

    data = json.loads(capsys.readouterr().out)
    assert data["failures"] >= 1 and data["ok"] is False


# ---------------------------------------------------------------------------
# A/B/C：绑定一致性 / 位置与文本判据 / 注记新鲜度（L.l3 三工单）
# ---------------------------------------------------------------------------
def _box_in(shape) -> list[float]:
    return [
        round(shape.left / 914400, 3),
        round(shape.top / 914400, 3),
        round((shape.left + shape.width) / 914400, 3),
        round((shape.top + shape.height) / 914400, 3),
    ]


def _mutated_spec(tmp_path, ids, mutate) -> Path:
    doc = yaml.safe_load(_spec(tmp_path, ids).read_text(encoding="utf-8"))
    mutate(doc["pages"][0], ids)
    p = tmp_path / "spec-mut.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return p


def test_duplicate_placeholder_binding_fails(tmp_path):
    deck, ids = _deck(tmp_path)
    page = {
        "page": 1,
        "placeholders": [{"md5": "x"}],
        "slots": [
            {"shape": ids["title"], "placeholder": {"md5": "x"}},
            {"shape": ids["cap"], "placeholder": {"md5": "x"}},
        ],
    }
    p = tmp_path / "spec-dup.yaml"
    p.write_text(yaml.safe_dump({"pages": [page]}), encoding="utf-8")
    rep = slides_audit.audit_deck(deck, p)
    assert any(
        i.name.startswith("绑定 p1") and i.status == "fail" for i in rep.items
    ), rep.summary()


def test_semantic_source_anchor_mismatch_is_manual(tmp_path):
    deck, ids = _deck(tmp_path)

    def mutate(page, ids):
        page["slots"][0]["anchor"] = "PRL 123 Fig.2 三面板"
        page["slots"][0]["source"] = (
            "文献图片/第5页-异常点-NatMater18-783-2019/fig7.png"
        )

    # 该来源不存在于 tmp_path 相对面 → 会同时报来源缺失；语义提示先于内容检查
    rep = slides_audit.audit_deck(deck, _mutated_spec(tmp_path, ids, mutate))
    assert any(
        "绑定存疑" in i.name and "NatMater" in i.detail and "PRL" in i.detail
        for i in rep.items
    )


def test_cyclic_rotation_is_suspected_and_ack_silences(tmp_path):
    deck, ids = _deck(tmp_path)

    def page_doc(ack: bool) -> dict:
        slots = [
            {"shape": ids["title"], "placeholder": {"md5": "b"}},
            {"shape": ids["cap"], "placeholder": {"md5": "c"}},
            {"shape": ids["body"], "placeholder": {"md5": "a"}},
        ]
        p = {
            "page": 1,
            "placeholders": [{"md5": "a"}, {"md5": "b"}, {"md5": "c"}],
            "slots": slots,
        }
        if ack:
            p["binding_ack"] = True
        return p

    for ack, expect in ((False, True), (True, False)):
        p = tmp_path / f"spec-rot-{ack}.yaml"
        p.write_text(yaml.safe_dump({"pages": [page_doc(ack)]}), encoding="utf-8")
        rep = slides_audit.audit_deck(deck, p)
        got = any("绑定错位疑似" in i.name for i in rep.items)
        assert got is expect, rep.summary()


def test_expect_box_pass_and_fail(tmp_path):
    deck, ids = _deck(tmp_path)
    prs = Presentation(str(deck))
    pic = next(sh for sh in prs.slides[0].shapes if sh.shape_id == ids["pic"])
    good = _box_in(pic)

    def mutate(page, ids):
        page["slots"][0]["expect_box"] = good
        page["slots"][0]["box_tol_in"] = 0.05

    rep = slides_audit.audit_deck(deck, _mutated_spec(tmp_path, ids, mutate))
    assert any(i.name.startswith("位置") and i.status == "pass" for i in rep.items)

    def mutate_bad(page, ids):
        page["slots"][0]["expect_box"] = [0.0, 0.0, 1.0, 1.0]

    rep_bad = slides_audit.audit_deck(deck, _mutated_spec(tmp_path, ids, mutate_bad))
    assert any(i.name.startswith("位置") and i.status == "fail" for i in rep_bad.items)


def test_expect_text_pass_and_fail(tmp_path):
    deck, ids = _deck(tmp_path)

    def mutate_ok(page, ids):
        page["slots"].append({"shape": ids["title"], "expect_text": "研究"})

    rep = slides_audit.audit_deck(deck, _mutated_spec(tmp_path, ids, mutate_ok))
    assert any(i.name.startswith("文本") and i.status == "pass" for i in rep.items)

    def mutate_bad(page, ids):
        page["slots"].append({"shape": ids["title"], "expect_text": "不存在的文字"})

    rep_bad = slides_audit.audit_deck(deck, _mutated_spec(tmp_path, ids, mutate_bad))
    assert any(i.name.startswith("文本") and i.status == "fail" for i in rep_bad.items)


def test_stale_note_hint_when_md5_matches(tmp_path):
    deck, ids = _deck(tmp_path)

    def mutate(page, ids):
        page["slots"][0]["note"] = "批语：待换"

    rep = slides_audit.audit_deck(deck, _mutated_spec(tmp_path, ids, mutate))
    assert any("注记过期" in i.name and i.status == "manual" for i in rep.items), (
        rep.summary()
    )
    assert rep.ok, "注记过期只是提示，不计 FAIL"


# ---------------------------------------------------------------------------
# 美学判据（R4 A.a5）：徽标 / 外框几何+上缘 / 标题边距
# ---------------------------------------------------------------------------
def _aes_deck(
    tmp_path: Path,
    *,
    stretch_logo: bool = False,
    frame_shape=MSO_SHAPE.ROUNDED_RECTANGLE,
    shift_title: bool = False,
    frame_top: float = 0.94,
    logo_px: tuple[int, int] = (64, 48),
    crop: tuple[int, int] | None = None,
    logo_box: tuple[float, float, float, float] = (0.76, 0.937, 3.18, 5.836),
):
    """两页达标样板：粗圆角外框 top 0.94/adj 3000、徽标 [0.76,0.937,3.18,5.836]、标题右缘公式。"""
    prs = Presentation()
    logo = tmp_path / "logo.png"
    Image.new("RGB", logo_px, (10, 20, 30)).save(logo)
    ids: list[dict] = []
    for _ in range(2):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        frame = s.shapes.add_shape(
            frame_shape, Inches(0.3), Inches(frame_top), Inches(4.0), Inches(5.0)
        )
        frame.line.width = Pt(6)
        if frame_shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            frame.adjustments[0] = 0.03
        title = s.shapes.add_textbox(
            Inches(0.55 if shift_title else 0.30),
            Inches(0.2),
            Inches(2.809),
            Inches(0.572),
        )
        title.text_frame.text = "声学超构表面"
        pic = s.shapes.add_picture(
            str(logo),
            Inches(logo_box[0]),
            Inches(logo_box[1]),
            Inches(logo_box[2] * (1.12 if stretch_logo else 1.0)),
            Inches(logo_box[3]),
        )
        if crop is not None:
            pic.crop_left = crop[0] / 100000.0
            pic.crop_bottom = crop[1] / 100000.0
        ids.append(
            {"frame": frame.shape_id, "title": title.shape_id, "pic": pic.shape_id}
        )
    deck = tmp_path / "aes-deck.pptx"
    prs.save(str(deck))
    return deck, ids


def _aes_spec(
    tmp_path: Path,
    ids: list[dict],
    *,
    src_rect: dict | str | None = None,
    logo_ref: list[float] | None = None,
) -> Path:
    aesthetics = {
        "logo_bbox_ref": logo_ref or [0.76, 0.937, 3.18, 5.836],
        "logo_aspect_tol": 0.01,
        "logo_bbox_tol_in": 0.02,
        "frame_geom": {"prst": "roundRect", "adj": 3000, "adj_tol": 500},
        "frame_top": 0.94,
        "frame_top_tol_in": 0.02,
        "title_left": 0.30,
        "title_right_formula": {"slope": 0.34, "intercept": 1.069, "tol_in": 0.1},
    }
    if src_rect is not None:
        aesthetics["expect_srcRect"] = src_rect
    doc = {
        "invariants": {"aesthetics": aesthetics},
        "pages": [
            {"page": i + 1, "titles": [{"shape": ids[i]["title"]}]}
            for i in range(len(ids))
        ],
    }
    p = tmp_path / "aes-spec.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return p


def test_aesthetics_compliant_deck_passes(tmp_path):
    deck, ids = _aes_deck(tmp_path)
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids))
    assert rep.ok, rep.summary()
    assert all(i.status == "pass" for i in rep.items), rep.summary()
    names = {i.name.split(" shape=")[0] for i in rep.items}
    for label in ("徽标 p1", "徽标 p2", "外框样式 p1", "上缘 p1", "标题边距 p1"):
        assert label in names, rep.summary()
    assert not any(i.name.startswith("标题公式") for i in rep.items)


def test_aesthetics_logo_horizontal_stretch_fails(tmp_path):
    """徽标横向拉伸 12% → 宽高比判 FAIL。"""
    deck, ids = _aes_deck(tmp_path, stretch_logo=True)
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids))
    assert any(
        i.name == "徽标 p1" and i.status == "fail" and "宽高比" in i.detail
        for i in rep.items
    ), rep.summary()


def test_aesthetics_frame_prst_rect_fails(tmp_path):
    """外框 prstGeom=rect → 外框样式判 FAIL。"""
    deck, ids = _aes_deck(tmp_path, frame_shape=MSO_SHAPE.RECTANGLE)
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids))
    assert any(
        i.name.startswith("外框样式 p1")
        and i.status == "fail"
        and "prst=rect" in i.detail
        for i in rep.items
    ), rep.summary()


def test_aesthetics_frame_top_offset_fails(tmp_path):
    """粗外框上缘偏离 frame_top → 上缘判 FAIL。"""
    deck, ids = _aes_deck(tmp_path, frame_top=1.0)
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids))
    assert any(
        i.name.startswith("上缘 p1") and i.status == "fail" for i in rep.items
    ), rep.summary()


def test_aesthetics_title_left_offset_fails(tmp_path):
    """标题 left 偏移 → 标题边距判 FAIL。"""
    deck, ids = _aes_deck(tmp_path, shift_title=True)
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids))
    assert any(
        i.name.startswith("标题边距 p1") and i.status == "fail" and "left" in i.detail
        for i in rep.items
    ), rep.summary()


def test_aesthetics_absent_adds_no_rows(tmp_path):
    """空 aesthetics 节不新增任何行（无节 spec 零回归）。"""
    deck, ids = _deck(tmp_path)
    base = slides_audit.audit_deck(deck, _spec(tmp_path, ids))
    doc = yaml.safe_load(_spec(tmp_path, ids).read_text(encoding="utf-8"))
    doc["invariants"] = {"aesthetics": {}}
    p = tmp_path / "spec-empty-aes.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    got = slides_audit.audit_deck(deck, p)
    assert [(i.name, i.status, i.detail) for i in base.items] == [
        (i.name, i.status, i.detail) for i in got.items
    ]


# ---------------------------------------------------------------------------
# 徽标裁切（R4 L.l3）：expect_srcRect 逐值 + 裁后源区素比 ≈ 框比
# ---------------------------------------------------------------------------
def test_aesthetics_src_rect_pass(tmp_path):
    """2000×2000 源图裁 l=49515/b=7343 → 源区比 0.5449 ≈ 框比，PASS。"""
    deck, ids = _aes_deck(tmp_path, logo_px=(200, 200), crop=(49515, 7343))
    rep = slides_audit.audit_deck(
        deck, _aes_spec(tmp_path, ids, src_rect={"l": 49515, "b": 7343, "tol": 100})
    )
    assert rep.ok, rep.summary()
    row = next(i for i in rep.items if i.name == "徽标裁切 p1")
    assert row.status == "pass" and "素比" in row.detail and "box_in[" in row.detail
    logo = next(i for i in rep.items if i.name == "徽标 p1")
    assert "box_in[" in logo.detail, "expect_srcRect 开启时徽标行须自描述坐标口径"


def test_aesthetics_src_rect_missing_fails(tmp_path):
    deck, ids = _aes_deck(tmp_path, logo_px=(200, 200))
    rep = slides_audit.audit_deck(
        deck, _aes_spec(tmp_path, ids, src_rect={"l": 49515, "b": 7343})
    )
    row = next(i for i in rep.items if i.name == "徽标裁切 p1")
    assert row.status == "fail" and "缺 srcRect" in row.detail


def test_aesthetics_src_rect_ratio_mismatch_fails(tmp_path):
    """l/b 值对但裁后源区比 ≠ 框比 → FAIL。"""
    deck, ids = _aes_deck(tmp_path, logo_px=(200, 200), crop=(30000, 7343))
    rep = slides_audit.audit_deck(
        deck, _aes_spec(tmp_path, ids, src_rect={"l": 30000, "b": 7343})
    )
    row = next(i for i in rep.items if i.name == "徽标裁切 p1")
    assert row.status == "fail" and "素比" in row.detail and "≠" in row.detail


def test_aesthetics_src_rect_absent_adds_no_rows(tmp_path):
    """无 expect_srcRect → 不增裁切行，也不给旧徽标行加 box_in（零回归）。"""
    deck, ids = _aes_deck(tmp_path)
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids))
    assert not any("徽标裁切" in i.name for i in rep.items)
    logo = next(i for i in rep.items if i.name == "徽标 p1")
    assert "box_in[" not in logo.detail


def test_aesthetics_src_rect_none_pass(tmp_path):
    """expect_srcRect: none + 1:1 完整图（无 srcRect）→ PASS（素比 1.0≈1.0）。"""
    deck, ids = _aes_deck(
        tmp_path, logo_px=(200, 200), logo_box=(0.09, 0.937, 5.836, 5.836)
    )
    rep = slides_audit.audit_deck(
        deck,
        _aes_spec(
            tmp_path,
            ids,
            src_rect="none",
            logo_ref=[0.09, 0.937, 5.836, 5.836],
        ),
    )
    assert rep.ok, rep.summary()
    row = next(i for i in rep.items if i.name == "徽标裁切 p1")
    assert row.status == "pass" and "无 srcRect" in row.detail
    assert "1.0000" in row.detail


def test_aesthetics_src_rect_none_rejects_crop(tmp_path):
    """expect_srcRect: none 但徽标带 srcRect → FAIL。"""
    deck, ids = _aes_deck(tmp_path, logo_px=(200, 200), crop=(49515, 7343))
    rep = slides_audit.audit_deck(deck, _aes_spec(tmp_path, ids, src_rect="none"))
    row = next(i for i in rep.items if i.name == "徽标裁切 p1")
    assert row.status == "fail" and "存在 srcRect" in row.detail


def test_expect_src_rect_annotates_decoration_box(tmp_path):
    """expect_srcRect 开启时装饰行尾标注 box_in[x0,y0,x1,y1]。"""
    deck, ids = _deck(tmp_path)
    doc = yaml.safe_load(_spec(tmp_path, ids).read_text(encoding="utf-8"))
    doc["invariants"] = {"aesthetics": {"expect_srcRect": {"l": 49515, "b": 7343}}}
    p = tmp_path / "spec-box.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    rep = slides_audit.audit_deck(deck, p)
    dec = next(i for i in rep.items if i.name.startswith("装饰 p1"))
    assert "box_in[" in dec.detail


# ---------------------------------------------------------------------------
# backlog 20261019-171344：图槽数量 / 相对顺序（bbox）/ 裁切参数
# ---------------------------------------------------------------------------
def _order_deck(tmp_path):
    """4 张图：左列 1、中列上下 2、右列 1。"""
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    img = _png(tmp_path / "order.png")
    ids = []
    for x, y in ((0.5, 1.0), (4.0, 1.0), (4.0, 3.5), (8.0, 1.0)):
        pic = s.shapes.add_picture(
            str(img), Inches(x), Inches(y), Inches(2), Inches(1.5)
        )
        ids.append(pic.shape_id)
    deck = tmp_path / "order.pptx"
    prs.save(str(deck))
    return deck, ids


def _slots_spec(tmp_path, slots, placeholders=None, name="slots-spec.yaml"):
    page = {"page": 1, "slots": slots}
    if placeholders is not None:
        page["placeholders"] = placeholders
    p = tmp_path / name
    p.write_text(
        yaml.safe_dump({"pages": [page]}, allow_unicode=True), encoding="utf-8"
    )
    return p


def test_slot_count_flags_unregistered_picture_and_ignores_unused_placeholders(
    tmp_path,
):
    deck, ids = _order_deck(tmp_path)
    good = _slots_spec(
        tmp_path,
        [{"shape": sid, "order": i + 1} for i, sid in enumerate(ids)],
        placeholders=[{"md5": f"unused-{i}"} for i in range(5)],
        name="count-good.yaml",
    )
    rep = slides_audit.audit_deck(deck, good)
    row = next(i for i in rep.items if i.name == "图槽数量 p1")
    assert row.status == "pass", row.detail
    assert "成品图 4 / spec 槽 4" in row.detail

    bad = _slots_spec(
        tmp_path,
        [{"shape": ids[0], "order": 1}],
        name="count-bad.yaml",
    )
    rep_bad = slides_audit.audit_deck(deck, bad)
    row_bad = next(i for i in rep_bad.items if i.name == "图槽数量 p1")
    assert row_bad.status == "fail" and "未登记图" in row_bad.detail


def test_slot_order_flags_middle_column_swap(tmp_path):
    deck, ids = _order_deck(tmp_path)
    good = _slots_spec(
        tmp_path,
        [{"shape": sid, "order": i + 1} for i, sid in enumerate(ids)],
        name="order-good.yaml",
    )
    rep = slides_audit.audit_deck(deck, good)
    row = next(i for i in rep.items if i.name == "图槽顺序 p1")
    assert row.status == "pass", row.detail

    # 中列上下对换：bbox y 序应为 order 2→3
    swapped = _slots_spec(
        tmp_path,
        [
            {"shape": ids[0], "order": 1},
            {"shape": ids[1], "order": 3},
            {"shape": ids[2], "order": 2},
            {"shape": ids[3], "order": 4},
        ],
        name="order-swap.yaml",
    )
    rep_bad = slides_audit.audit_deck(deck, swapped)
    row_bad = next(i for i in rep_bad.items if i.name == "图槽顺序 p1")
    assert row_bad.status == "fail" and "列" in row_bad.detail, row_bad.detail


def test_slot_order_flags_reversed_column_order(tmp_path):
    deck, ids = _order_deck(tmp_path)
    spec = _slots_spec(
        tmp_path,
        [
            {"shape": ids[0], "order": 4},
            {"shape": ids[1], "order": 3},
            {"shape": ids[2], "order": 2},
            {"shape": ids[3], "order": 1},
        ],
        name="order-rev.yaml",
    )
    rep = slides_audit.audit_deck(deck, spec)
    row = next(i for i in rep.items if i.name == "图槽顺序 p1")
    assert row.status == "fail" and "首项 order 非递增" in row.detail, row.detail


def test_slot_crop_note_consistency_and_px(tmp_path):
    deck, ids = _deck(tmp_path)

    def spec(note: str, name: str) -> Path:
        return _slots_spec(
            tmp_path,
            [{"shape": ids["pic"], "order": 1, "crop": {"note": note}}],
            name=name,
        )

    ok = slides_audit.audit_deck(
        deck,
        spec("64×48；自 orig.png（128×96）裁 (32,24,96,72)", "crop-ok.yaml"),
    )
    row = next(i for i in ok.items if i.name.startswith("图槽裁切 p1"))
    assert row.status == "pass", row.detail
    assert "px 64×48" in row.detail and "源 128×96 内" in row.detail

    bad = slides_audit.audit_deck(
        deck,
        spec("60×48；自 orig.png（128×96）裁 (32,24,96,72)", "crop-bad.yaml"),
    )
    row_bad = next(i for i in bad.items if i.name.startswith("图槽裁切 p1"))
    assert row_bad.status == "fail" and "≠ note" in row_bad.detail
