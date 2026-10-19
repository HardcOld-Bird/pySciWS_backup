"""``compose slides audit``：按 deck-spec 机械对账（装饰层/字体全槽/图槽/引用框/标题公式）。

deck-spec（YAML）是本轮「要求唯一真源」的载体：**不变量**（装饰层指纹、字体全槽、动画
指针）与**变量槽位**（每页每图槽：占位符指纹/内容锚/原文图号/来源/裁切）。审计逐项输出
PASS/FAIL；无法机械判定的条目降级为「人工确认」行（必须留勾选痕迹，不计 FAIL）。

spec 形态见 `orchestration/skills/document-writing/references/visual-qa.md` §deck-spec。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

from . import slides_fonts, slides_layout

_EMU_IN = 914400.0
DEFAULT_TOL_IN = 0.05


@dataclass
class AuditItem:
    """一条审计读数（status: pass|fail|manual）。"""

    name: str
    status: str
    detail: str


@dataclass
class AuditReport:
    pptx: Path
    spec: Path
    items: list[AuditItem] = field(default_factory=list)

    @property
    def failures(self) -> list[AuditItem]:
        return [i for i in self.items if i.status == "fail"]

    @property
    def manual(self) -> list[AuditItem]:
        return [i for i in self.items if i.status == "manual"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [
            f"[slides audit] {self.pptx.name} × {self.spec.name}："
            + ("PASS" if self.ok else f"FAIL {len(self.failures)} 项")
            + (f"；人工确认 {len(self.manual)} 项" if self.manual else "")
        ]
        for it in self.items:
            mark = {"pass": "√", "fail": "✗", "manual": "?"}.get(it.status, "?")
            lines.append(f"  [{mark}] {it.name}：{it.detail}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pptx": str(self.pptx),
            "spec": str(self.spec),
            "ok": self.ok,
            "failures": len(self.failures),
            "manual": len(self.manual),
            "items": [
                {"name": i.name, "status": i.status, "detail": i.detail}
                for i in self.items
            ],
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _box(shape: Any) -> list[float]:
    return [
        round(shape.left / _EMU_IN, 3),
        round(shape.top / _EMU_IN, 3),
        round((shape.left + shape.width) / _EMU_IN, 3),
        round((shape.top + shape.height) / _EMU_IN, 3),
    ]


def _pages(spec: dict[str, Any], page: int) -> list[dict[str, Any]]:
    return [p for p in (spec.get("pages") or []) if int(p.get("page") or 0) == page]


def _shape_on(slide: Any, sid: int) -> Any | None:
    return next((sh for sh in slide.shapes if int(sh.shape_id) == int(sid)), None)


def _check_decoration(
    slide: Any,
    entry: dict[str, Any],
    page: int,
    *,
    annotate_box: bool = False,
) -> AuditItem:
    name = str(entry.get("name") or "")
    kind = str(entry.get("kind") or "")
    want = entry.get("box_in")
    tol = float(entry.get("tol_in", DEFAULT_TOL_IN))
    label = name or f"kind={kind} box={want}"
    cands = list(slide.shapes)
    if name:
        cands = [sh for sh in cands if getattr(sh, "name", "") == name]
    elif want:
        cands = [
            sh
            for sh in cands
            if all(
                abs(a - b) <= tol for a, b in zip(_box(sh), [float(x) for x in want])
            )
        ]
    if not cands:
        return AuditItem(f"装饰 p{page} {label}", "fail", "未找到匹配形状")
    sh = cands[0]
    box_suffix = f" box_in{_box(sh)}" if annotate_box else ""
    if want:
        got = _box(sh)
        if any(abs(a - float(b)) > tol for a, b in zip(got, want)):
            return AuditItem(
                f"装饰 p{page} {label}",
                "fail",
                f"bbox {got} ≠ spec {want}（tol {tol}）{box_suffix}",
            )
    if kind and kind.lower() not in str(getattr(sh, "shape_type", "")).lower():
        # kind 语义宽松（rectangle/picture/text），仅在显式给出时提示
        actual = str(getattr(sh, "name", "")) + " " + str(getattr(sh, "shape_type", ""))
        if kind.lower() not in actual.lower():
            return AuditItem(
                f"装饰 p{page} {label}",
                "fail",
                f"kind 不匹配（实际 {actual[:60]}）{box_suffix}",
            )
    return AuditItem(
        f"装饰 p{page} {label}",
        "pass",
        f"存在 @ {_box(sh)}（{sh.name}）{box_suffix}",
    )


def _run_slots_ok(run: Any, expected: dict[str, Any]) -> tuple[bool, str]:
    got = slides_fonts.font_slots(run)
    require = [str(t) for t in (expected.get("require_slots") or ["latin", "ea", "cs"])]
    missing = [t for t in require if not got.get(t, {}).get("typeface")]
    if missing:
        return False, f"缺槽 {missing}（现有 {sorted(got)}）"
    for tag, attrs in (expected.get("slots") or {}).items():
        for key, val in (attrs or {}).items():
            actual = got.get(tag, {}).get(key)
            if actual != str(val):
                return False, f"{tag}.{key}={actual!r} ≠ {val!r}"
    return True, "latin/ea/cs 齐" if "ea" in require else "槽位符合"


def _check_fonts(slide: Any, entry: dict[str, Any], page: int) -> list[AuditItem]:
    role = str(entry.get("role") or "字体")
    ids = [int(x) for x in (entry.get("shapes") or [])]
    shapes = [_shape_on(slide, i) for i in ids] if ids else []
    if not ids and entry.get("name_contains"):
        needle = str(entry["name_contains"])
        shapes = [sh for sh in slide.shapes if needle in getattr(sh, "name", "")]
    shapes = [sh for sh in shapes if sh is not None]
    if not shapes:
        return [AuditItem(f"字体 p{page} {role}", "fail", "未找到目标形状")]
    out: list[AuditItem] = []
    any_run = False
    for sh in shapes:
        for pi, para in enumerate(
            getattr(sh, "text_frame", None).paragraphs
            if getattr(sh, "text_frame", None)
            else [],
            1,
        ):
            for ri, run in enumerate(para.runs, 1):
                any_run = True
                ok, why = _run_slots_ok(run, entry)
                if not ok:
                    out.append(
                        AuditItem(
                            f"字体 p{page} {role} {sh.name} P{pi}R{ri}",
                            "fail",
                            f"{why}（{run.text[:20]!r}）",
                        )
                    )
    if not any_run:
        out.append(AuditItem(f"字体 p{page} {role}", "fail", "形状无文本 run"))
    if not out:
        out.append(
            AuditItem(
                f"字体 p{page} {role}", "pass", f"{len(shapes)} 个形状的 run 槽位齐全"
            )
        )
    return out


#: 注记待办词（槽修好后仍留这些词 → 「注记过期」提示）。
_TODO_WORDS: tuple[str, ...] = ("待换", "待补", "判 ✗", "待核对", "待回填", "需核对")

#: 文章种类指纹（source 路径 vs anchor 语义配对提示）。
_ARTICLE_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("RPP", re.compile(r"RepProgPhys|Rep\.?\s*Prog|RPP", re.I)),
    ("NatMater", re.compile(r"NatMater|Nat\.?\s*Mater", re.I)),
    ("PRL", re.compile(r"PhysRevLett|Phys\.?\s*Rev\.?\s*Lett|PRL", re.I)),
    (
        "PRApplied",
        re.compile(r"PhysRevApplied|Phys\.?\s*Rev\.?\s*Appl|PRApplied", re.I),
    ),
    ("CommunPhys", re.compile(r"CommunPhys|Commun\.?\s*Phys", re.I)),
    ("NatCommun", re.compile(r"NatCommun|Nat\.?\s*Commun", re.I)),
    ("NatPhotonics", re.compile(r"NatPhotonics|Nat\.?\s*Photonics", re.I)),
    ("NaturePhys", re.compile(r"NaturePhys|Nature\s*Phys", re.I)),
    ("LightSciAppl", re.compile(r"LightSciAppl|Light\s*Sci", re.I)),
    ("Science", re.compile(r"\bScience\d|Science\s*\d{3}", re.I)),
)


def _article_kind(text: str) -> str:
    for name, pat in _ARTICLE_KINDS:
        if pat.search(text or ""):
            return name
    return ""


def _semantic_mismatch(slot: dict[str, Any]) -> str:
    """source 文章种类与 anchor 文章名冲突时的提示（都判不出种类 → 空串）。"""
    sk = _article_kind(str(slot.get("source") or ""))
    ak = _article_kind(str(slot.get("anchor") or ""))
    if sk and ak and sk != ak:
        return f"source 文章={sk}，anchor 文章={ak}"
    return ""


def _slot_text(shape: Any) -> str:
    try:
        return "".join(r.text for p in shape.text_frame.paragraphs for r in p.runs)
    except Exception:  # noqa: BLE001
        return ""


def _check_slot(
    spec_file: Path, slide: Any, slot: dict[str, Any], page: int
) -> list[AuditItem]:
    """一个图槽的多项判据：内容指纹 / 位置 / 文本 / 注记新鲜度。"""
    sid = int(slot.get("shape") or 0)
    label = f"图槽 p{page} shape={sid} {slot.get('figure_no') or ''}".strip()
    sh = _shape_on(slide, sid)
    if sh is None:
        return [AuditItem(label, "fail", "未找到图槽形状")]
    items: list[AuditItem] = []
    note = str(slot.get("note") or "")

    expect_text = slot.get("expect_text")
    if expect_text:
        try:
            rx = re.compile(str(expect_text))
        except re.error as exc:
            items.append(AuditItem(f"文本 {label}", "fail", f"expect_text 非法：{exc}"))
        else:
            text = _slot_text(sh)
            ok = bool(rx.search(text))
            items.append(
                AuditItem(
                    f"文本 {label}",
                    "pass" if ok else "fail",
                    f"regex {expect_text!r} "
                    + (f"命中 {text[:40]!r}" if ok else f"未命中（{text[:40]!r}）"),
                )
            )

    if slot.get("expect_box"):
        try:
            want = [float(x) for x in slot["expect_box"]]
            tol = float(slot.get("box_tol_in", 0.05))
            got = _box(sh)
        except (TypeError, ValueError):
            items.append(AuditItem(f"位置 {label}", "fail", "expect_box 格式错误"))
        else:
            ok = len(want) == 4 and all(abs(a - b) <= tol for a, b in zip(got, want))
            items.append(
                AuditItem(
                    f"位置 {label}",
                    "pass" if ok else "fail",
                    f"bbox {got} "
                    + ("≈" if ok else "≠")
                    + f" spec {want}（tol {tol}）",
                )
            )

    md5_ok: bool | None = None
    if any(slot.get(k) for k in ("md5", "source", "px")):
        try:
            blob = sh.image.blob
            px = list(sh.image.size)
        except Exception:  # noqa: BLE001
            items.append(AuditItem(label, "fail", "目标不是图片"))
        else:
            want = str(slot.get("md5") or "").lower()
            source = slot.get("source")
            src_md5 = ""
            if source:
                src_path = Path(str(source))
                if not src_path.is_absolute():
                    src_path = spec_file.parent / src_path
                if not src_path.exists():
                    items.append(
                        AuditItem(label, "fail", f"来源文件不存在：{src_path}")
                    )
                    return items
                src_md5 = _md5(src_path.read_bytes())
                want = want or src_md5
            got = _md5(blob)
            md5_ok = (not want or got == want) and (not src_md5 or got == src_md5)
            if not md5_ok:
                why = (
                    f"md5 {got[:12]}… ≠ spec {want[:12]}…"
                    if want and got != want
                    else "内嵌图与来源文件 md5 不一致"
                )
                items.append(AuditItem(label, "fail", why))
            else:
                want_px = slot.get("px")
                if want_px and [int(x) for x in want_px] != [int(x) for x in px]:
                    md5_ok = False
                    items.append(
                        AuditItem(label, "fail", f"像素 {px} ≠ spec {want_px}")
                    )
                else:
                    extra = " / ".join(
                        str(slot[k]) for k in ("anchor", "figure_no") if slot.get(k)
                    )
                    crop = slot.get("crop")
                    if crop:
                        extra += f" / crop={crop}"
                    items.append(
                        AuditItem(
                            label,
                            "pass",
                            f"md5 {got[:12]}… px {px}"
                            + (f"（{extra}）" if extra else ""),
                        )
                    )

    if md5_ok and any(w in note for w in _TODO_WORDS):
        items.append(
            AuditItem(
                f"注记过期 {label}",
                "manual",
                f"note 含待办词但 md5 已与 spec 一致——请刷新注记（{note[:40]}）",
            )
        )
    return items


def _check_formula(slide: Any, entry: dict[str, Any], *, capsule: bool) -> AuditItem:
    sid = int(entry.get("shape") or 0)
    label = f"{'胶囊' if capsule else '标题'}公式 shape={sid}"
    sh = _shape_on(slide, sid)
    if sh is None:
        return AuditItem(label, "fail", "未找到形状")
    text = ""
    try:
        text = "".join(r.text for p in sh.text_frame.paragraphs for r in p.runs)
    except Exception:  # noqa: BLE001
        pass
    n = int(entry.get("n_chars") or len("".join(text.split())))
    if capsule:
        c = int(entry.get("cite_chars") or 0)
        want = 0.27 * n + 0.30 + 0.55 / max(n, 1) + (0.14 * c + 0.08 if c else 0)
    else:
        want = 0.34 * n + 0.55
    got = sh.width / _EMU_IN
    tol = float(entry.get("tol_in", 0.10))
    aligned = True
    try:
        aligned = all(
            p.alignment in (PP_ALIGN.DISTRIBUTE, None)
            or str(p.alignment).endswith("DISTRIBUTE")
            for p in sh.text_frame.paragraphs
        )
    except Exception:  # noqa: BLE001
        pass
    if abs(got - want) > tol:
        return AuditItem(label, "fail", f"宽 {got:.3f} in ≠ 公式 {want:.3f} ±{tol}")
    if not aligned:
        return AuditItem(label, "fail", "对齐不是 algn=dist")
    return AuditItem(label, "pass", f"宽 {got:.3f} ≈ 公式 {want:.3f}；algn=dist")


def _check_animation(slide: Any, page: int) -> AuditItem:
    el = slide.element.find(qn("p:timing"))
    if el is None:
        return AuditItem(f"动画 p{page}", "fail", "缺 <p:timing>")
    onbegin = el.find(f".//{qn('p:cond')}[@evt='onBegin']")
    seek = el.find(f".//{qn('p:seq')}[@nextAc='seek']")
    if onbegin is None:
        return AuditItem(f"动画 p{page}", "fail", "首块未 onBegin")
    if seek is None:
        return AuditItem(f"动画 p{page}", "fail", "缺 nextAc=seek")
    return AuditItem(f"动画 p{page}", "pass", "onBegin + nextAc=seek")


#: 徽标识别：bbox 高 ≥5 in 的图片按水印层处理。
_LOGO_MIN_HEIGHT_IN = 5.0

#: 粗外框识别：描边 ≥5 pt、宽 ≥2 in。
_FRAME_MIN_STROKE_EMU = 5 * 12700
_FRAME_MIN_WIDTH_IN = 2.0

#: roundRect 未显式写 adj 时的 DrawingML 默认值。
_ROUNDRECT_DEFAULT_ADJ = 16667


def _is_picture(shape: Any) -> bool:
    try:
        return int(shape.shape_type) == 13
    except (TypeError, ValueError):
        return False


def _ltwh(shape: Any) -> list[float]:
    """[left, top, width, height] in inches（``logo_bbox_ref`` 口径）。"""
    return [
        round(shape.left / _EMU_IN, 4),
        round(shape.top / _EMU_IN, 4),
        round(shape.width / _EMU_IN, 4),
        round(shape.height / _EMU_IN, 4),
    ]


def _float4(value: Any) -> list[float] | None:
    try:
        out = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return out if len(out) == 4 else None


def _logo_ref_md5s(prs: Any, ref: list[float], tol: float) -> set[str]:
    """首两页徽标指纹：bbox 命中 ``logo_bbox_ref`` 或高度 ≥5 in 的图片。"""
    out: set[str] = set()
    for slide in list(prs.slides)[:2]:
        for sh in slide.shapes:
            if not _is_picture(sh):
                continue
            try:
                ltwh = _ltwh(sh)
                blob = sh.image.blob
            except Exception:  # noqa: BLE001
                continue
            if ltwh[3] >= _LOGO_MIN_HEIGHT_IN or all(
                abs(a - b) <= tol for a, b in zip(ltwh, ref)
            ):
                out.add(_md5(blob))
    return out


def _find_logo(slide: Any, ref_md5s: set[str]) -> Any | None:
    """先按首两页指纹找徽标，退而找 bbox 高 ≥5 in 的水印层图片。"""
    tall: Any | None = None
    for sh in slide.shapes:
        if not _is_picture(sh):
            continue
        try:
            if ref_md5s and _md5(sh.image.blob) in ref_md5s:
                return sh
            if tall is None and sh.height / _EMU_IN >= _LOGO_MIN_HEIGHT_IN:
                tall = sh
        except Exception:  # noqa: BLE001
            continue
    return tall


def _check_logo(
    slide: Any,
    page: int,
    aes: dict[str, Any],
    ref_md5s: set[str],
    *,
    annotate_box: bool = False,
) -> AuditItem:
    label = f"徽标 p{page}"
    ref = _float4(aes.get("logo_bbox_ref"))
    if ref is None:
        return AuditItem(label, "fail", "logo_bbox_ref 需为 [x,y,w,h]（in）")
    sh = _find_logo(slide, ref_md5s)
    if sh is None:
        return AuditItem(
            label, "fail", "未找到徽标图（首两页同指纹或 bbox 高 ≥5 in 的水印层）"
        )
    tol = float(aes.get("logo_bbox_tol_in", 0.02))
    aspect_tol = float(aes.get("logo_aspect_tol", 0.01))
    got = _ltwh(sh)
    want_aspect = ref[2] / ref[3] if ref[3] else 0.0
    got_aspect = got[2] / got[3] if got[3] else 0.0
    bbox_ok = all(abs(a - b) <= tol for a, b in zip(got, ref))
    aspect_ok = abs(got_aspect - want_aspect) <= aspect_tol
    detail = (
        f"bbox {got} {'≈' if bbox_ok else '≠'} {ref}（tol {tol}）；"
        f"宽高比 {got_aspect:.4f} {'≈' if aspect_ok else '≠'} {want_aspect:.4f}"
        f"（tol {aspect_tol}）"
    )
    if annotate_box:
        detail += f" box_in{_box(sh)}"
    return AuditItem(label, "pass" if bbox_ok and aspect_ok else "fail", detail)


def _src_rect(shape: Any) -> dict[str, int]:
    """图片 ``a:srcRect`` 四边裁切（1/1000 %；缺省边为 0）。"""
    el = shape._element.find(".//" + qn("a:srcRect"))
    if el is None:
        return {}
    out: dict[str, int] = {}
    for key in ("l", "t", "r", "b"):
        raw = el.get(key)
        if raw is None:
            continue
        try:
            out[key] = int(raw)
        except ValueError:
            continue
    return out


def _check_logo_crop(
    slide: Any, page: int, aes: dict[str, Any], ref_md5s: set[str]
) -> AuditItem:
    """``aesthetics.expect_srcRect``：徽标 blipFill 裁切逐值 ＋ 裁后源区素比 ≈ 框比。

    配置为字符串 ``none`` 时断言**无 srcRect**（完整图；素比校验自然为 1.0≈1.0）。
    """
    label = f"徽标裁切 p{page}"
    raw_cfg = aes.get("expect_srcRect")
    no_crop = isinstance(raw_cfg, str) and raw_cfg.strip().lower() == "none"
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    tol = int(cfg.get("tol", 100))
    sh = _find_logo(slide, ref_md5s)
    if sh is None:
        return AuditItem(label, "fail", "未找到徽标图")
    rect = _src_rect(sh)
    if no_crop:
        if rect:
            shown = " ".join(f"{k}={v}" for k, v in rect.items())
            return AuditItem(
                label,
                "fail",
                f"存在 srcRect {shown}（期望 none：内容页须完整图） box_in{_box(sh)}",
            )
    elif not rect:
        return AuditItem(
            label, "fail", f"缺 srcRect（期望 l={cfg.get('l')} b={cfg.get('b')}）"
        )
    bad: list[str] = []
    if not no_crop:
        for key in ("l", "t", "r", "b"):
            if key not in cfg:
                continue
            want = int(cfg[key])
            got = rect.get(key, 0)
            if abs(got - want) > tol:
                bad.append(f"{key}={got} ≠ {want}（tol {tol}）")
    try:
        px_w, px_h = [float(v) for v in sh.image.size]
    except Exception:  # noqa: BLE001
        px_w = px_h = 0.0
    crop_w = px_w * (1.0 - (rect.get("l", 0) + rect.get("r", 0)) / 100000.0)
    crop_h = px_h * (1.0 - (rect.get("t", 0) + rect.get("b", 0)) / 100000.0)
    ratio = crop_w / crop_h if crop_h else 0.0
    frame_ratio = sh.width / sh.height if sh.height else 0.0
    ratio_ok = bool(frame_ratio) and abs(ratio - frame_ratio) <= 0.01 * frame_ratio
    shown = (
        "无 srcRect"
        if no_crop
        else "srcRect " + " ".join(f"{k}={v}" for k, v in rect.items())
    )
    detail = (
        f"{shown}；裁后素比 {ratio:.4f} "
        f"{'≈' if ratio_ok else '≠'} 框比 {frame_ratio:.4f}（±1%）"
        f" box_in{_box(sh)}"
    )
    if bad:
        detail = "；".join(bad) + "；" + detail
    return AuditItem(label, "pass" if not bad and ratio_ok else "fail", detail)


def _stroke_width(shape: Any) -> int | None:
    ln = shape._element.find(".//" + qn("a:ln"))
    if ln is None or ln.get("w") is None:
        return None
    try:
        return int(ln.get("w"))
    except ValueError:
        return None


def _prst(shape: Any) -> str:
    pg = shape._element.find(".//" + qn("a:prstGeom"))
    return str(pg.get("prst")) if pg is not None else "custGeom"


def _roundrect_adj(shape: Any) -> int | None:
    pg = shape._element.find(".//" + qn("a:prstGeom"))
    if pg is None:
        return None
    avl = pg.find(qn("a:avLst"))
    if avl is None:
        return None
    for gd in avl.findall(qn("a:gd")):
        if gd.get("name") == "adj":
            m = re.search(r"val\s+(-?\d+)", gd.get("fmla") or "")
            return int(m.group(1)) if m else None
    return None


def _coarse_frames(slide: Any) -> list[Any]:
    out: list[Any] = []
    for sh in slide.shapes:
        if _is_picture(sh):
            continue
        try:
            if sh.width / _EMU_IN < _FRAME_MIN_WIDTH_IN:
                continue
        except (AttributeError, TypeError):
            continue
        stroke = _stroke_width(sh)
        if stroke is not None and stroke >= _FRAME_MIN_STROKE_EMU:
            out.append(sh)
    return out


def _check_frames(slide: Any, page: int, aes: dict[str, Any]) -> list[AuditItem]:
    frames = _coarse_frames(slide)
    if not frames:
        label = (
            f"外框样式 p{page}"
            if aes.get("frame_geom") is not None
            else f"上缘 p{page}"
        )
        return [AuditItem(label, "fail", "未找到粗外框（描边 ≥5 pt 且宽 ≥2 in）")]
    items: list[AuditItem] = []
    geom = aes.get("frame_geom")
    if geom is not None:
        geom = geom if isinstance(geom, dict) else {}
        want_prst = geom.get("prst")
        want_adj = geom.get("adj")
        adj_tol = float(geom.get("adj_tol", 500))
        for sh in frames:
            label = f"外框样式 p{page} shape={sh.shape_id}"
            prst = _prst(sh)
            bad: list[str] = []
            if want_prst is not None and str(want_prst) != prst:
                bad.append(f"prst={prst} ≠ {want_prst}")
            adj = _roundrect_adj(sh)
            if want_adj is not None:
                actual = _ROUNDRECT_DEFAULT_ADJ if adj is None else adj
                if abs(actual - int(want_adj)) > adj_tol:
                    shown = "缺省 16667" if adj is None else str(adj)
                    bad.append(f"adj={shown} ≠ {want_adj}（tol {adj_tol:g}）")
            if bad:
                items.append(AuditItem(label, "fail", "；".join(bad)))
            else:
                extra = f" adj={adj}" if prst == "roundRect" and adj is not None else ""
                items.append(AuditItem(label, "pass", f"prst={prst}{extra}"))
    if aes.get("frame_top") is not None:
        want_top = float(aes["frame_top"])
        tol = float(aes.get("frame_top_tol_in", 0.02))
        for sh in frames:
            label = f"上缘 p{page} shape={sh.shape_id}"
            top = sh.top / _EMU_IN
            ok = abs(top - want_top) <= tol
            items.append(
                AuditItem(
                    label,
                    "pass" if ok else "fail",
                    f"top {top:.3f} {'≈' if ok else '≠'} {want_top} ±{tol}",
                )
            )
    return items


def _check_title_margins(
    page_spec: dict[str, Any], slide: Any, page: int, aes: dict[str, Any]
) -> list[AuditItem]:
    titles = [t for t in (page_spec.get("titles") or []) if t.get("shape")]
    if not titles:
        return [
            AuditItem(
                f"标题边距 p{page}", "manual", "spec 未给 titles.shape，无法定位页标题"
            )
        ]
    left_want = aes.get("title_left")
    formula = aes.get("title_right_formula")
    formula = formula if isinstance(formula, dict) else {}
    slope = float(formula.get("slope", 0.34))
    intercept = float(formula.get("intercept", 1.069))
    right_tol = float(formula.get("tol_in", 0.1))
    left_tol = float(aes.get("title_left_tol_in", 0.02))
    out: list[AuditItem] = []
    for entry in titles:
        sid = int(entry["shape"])
        sh = _shape_on(slide, sid)
        label = f"标题边距 p{page} shape={sid}"
        if sh is None:
            out.append(AuditItem(label, "fail", "未找到标题形状"))
            continue
        n = len("".join(_slot_text(sh).split()))
        left = sh.left / _EMU_IN
        right = (sh.left + sh.width) / _EMU_IN
        problems: list[str] = []
        detail: list[str] = []
        if left_want is not None:
            ok = abs(left - float(left_want)) <= left_tol
            detail.append(
                f"left {left:.3f} {'≈' if ok else '≠'} "
                f"{float(left_want):.3f} ±{left_tol}"
            )
            if not ok:
                problems.append("left")
        if formula:
            want_right = slope * n + intercept
            ok = abs(right - want_right) <= right_tol
            detail.append(
                f"right {right:.3f} {'≈' if ok else '≠'} {want_right:.3f}"
                f"（n={n}, {slope:g}n+{intercept:g}）±{right_tol}"
            )
            if not ok:
                problems.append("right")
        out.append(AuditItem(label, "fail" if problems else "pass", "；".join(detail)))
    return out


def _check_aesthetics(
    page_spec: dict[str, Any],
    slide: Any,
    page: int,
    aes: dict[str, Any],
    logo_ref_md5s: set[str],
) -> list[AuditItem]:
    """``invariants.aesthetics`` 判据：徽标 / 徽标裁切 / 外框几何+上缘 / 标题边距。"""
    items: list[AuditItem] = []
    crop_on = bool(aes.get("expect_srcRect"))
    if aes.get("logo_bbox_ref") is not None:
        items.append(_check_logo(slide, page, aes, logo_ref_md5s, annotate_box=crop_on))
    if crop_on:
        items.append(_check_logo_crop(slide, page, aes, logo_ref_md5s))
    if aes.get("frame_geom") is not None or aes.get("frame_top") is not None:
        items.extend(_check_frames(slide, page, aes))
    if aes.get("title_left") is not None or aes.get("title_right_formula"):
        items.extend(_check_title_margins(page_spec, slide, page, aes))
    return items


def audit_deck(pptx: str | Path, spec: str | Path) -> AuditReport:
    """按 spec 审计 deck；spec 解析失败抛 ``ValueError``。"""
    import yaml
    from pptx import Presentation

    src = Path(pptx)
    spec_path = Path(spec)
    if not src.exists():
        raise FileNotFoundError(f"pptx 不存在：{src}")
    if not spec_path.exists():
        raise FileNotFoundError(f"spec 不存在：{spec_path}")
    try:
        doc = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"spec YAML 解析失败：{exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("spec 必须是映射（invariants/pages/slots…）")
    prs = Presentation(str(src))
    rep = AuditReport(pptx=src, spec=spec_path)

    inv = doc.get("invariants") or {}
    aes = inv.get("aesthetics") if isinstance(inv, dict) else None
    aes = aes if isinstance(aes, dict) else {}
    logo_ref_md5s: set[str] = set()
    if aes.get("logo_bbox_ref") is not None:
        ref = _float4(aes.get("logo_bbox_ref"))
        if ref is not None:
            logo_ref_md5s = _logo_ref_md5s(
                prs, ref, float(aes.get("logo_bbox_tol_in", 0.02))
            )
    # aesthetics 已接管标题几何时，旧 width=0.34n+0.55 判据不再适用。
    titles_superseded = bool(
        aes.get("title_left") is not None or aes.get("title_right_formula")
    )
    pages = doc.get("pages") or []
    for page_spec in pages:
        page = int(page_spec.get("page") or 0)
        if not 1 <= page <= len(prs.slides):
            rep.items.append(AuditItem(f"第 {page} 页", "fail", "页码越界"))
            continue
        slide = prs.slides[page - 1]
        for entry in page_spec.get("decoration") or []:
            rep.items.append(
                _check_decoration(
                    slide,
                    entry,
                    page,
                    annotate_box=bool(aes.get("expect_srcRect")),
                )
            )
        for entry in page_spec.get("fonts") or []:
            rep.items.extend(_check_fonts(slide, entry, page))
        slots = page_spec.get("slots") or []
        placeholders = page_spec.get("placeholders") or []
        draft_count: dict[str, int] = {}
        for ph in placeholders:
            m = str(ph.get("md5") or "")
            if m:
                draft_count[m] = draft_count.get(m, 0) + 1
        bound: dict[str, int] = {}
        for slot in slots:
            m = str((slot.get("placeholder") or {}).get("md5") or "")
            if m:
                bound[m] = bound.get(m, 0) + 1
        for m, n in bound.items():
            allowed = draft_count.get(m, 0)
            if allowed == 0 or n > allowed:
                rep.items.append(
                    AuditItem(
                        f"绑定 p{page}",
                        "fail",
                        f"placeholder.md5 {m[:12]}… 被 {n} 个槽绑定（草稿仅 {allowed} 处）",
                    )
                )
        for slot in slots:
            why = _semantic_mismatch(slot)
            if why:
                rep.items.append(
                    AuditItem(f"绑定存疑 p{page} {slot.get('id')}", "manual", why)
                )
        # 整页循环错位：slot 占位序 == 草稿序的非平凡轮换（k=0 为正常序）
        seq = [
            str((s.get("placeholder") or {}).get("md5") or "")
            for s in slots
            if (s.get("placeholder") or {}).get("md5")
        ]
        draft_seq = [str(p.get("md5") or "") for p in placeholders if p.get("md5")]
        if not page_spec.get("binding_ack") and len(seq) >= 3:
            for k in range(1, len(draft_seq)):
                if draft_seq[k:] + draft_seq[:k] == seq:
                    rep.items.append(
                        AuditItem(
                            f"绑定错位疑似 p{page}",
                            "manual",
                            f"slot 占位序是草稿序的轮换（k={k}）——核对是否整页错位绑定",
                        )
                    )
                    break
        for slot in slots:
            rep.items.extend(_check_slot(spec_path, slide, slot, page))
        for entry in page_spec.get("titles") or []:
            if not titles_superseded:
                rep.items.append(_check_formula(slide, entry, capsule=False))
        for entry in page_spec.get("capsules") or []:
            rep.items.append(_check_formula(slide, entry, capsule=True))
        refs = [
            (int(r["ref"]), int(r["anchor"]))
            for r in (page_spec.get("refs") or [])
            if r.get("ref") and r.get("anchor")
        ]
        if refs:
            try:
                lay = slides_layout.layout_report(src, page=page, refs=refs)
            except ValueError as exc:
                rep.items.append(AuditItem(f"引用框 p{page}", "fail", str(exc)))
            else:
                for chk in lay.checks:
                    rep.items.append(
                        AuditItem(
                            f"引用框 p{page} {chk.name}",
                            "pass" if chk.ok else "fail",
                            chk.detail,
                        )
                    )
        if page_spec.get("animation"):
            rep.items.append(_check_animation(slide, page))
        if aes:
            rep.items.extend(
                _check_aesthetics(page_spec, slide, page, aes, logo_ref_md5s)
            )
        for note in page_spec.get("manual") or []:
            rep.items.append(AuditItem(f"人工确认 p{page}", "manual", str(note)))

    for entry in inv.get("decoration") or []:
        # 顶层 decoration：在所有页找一遍（模板级不变量）
        page = int(entry.get("page") or 0)
        if 1 <= page <= len(prs.slides):
            rep.items.append(
                _check_decoration(
                    prs.slides[page - 1],
                    entry,
                    page,
                    annotate_box=bool(aes.get("expect_srcRect")),
                )
            )
        else:
            rep.items.append(
                AuditItem(
                    f"装饰 {entry.get('name') or '?'}", "fail", "spec 未给有效 page"
                )
            )
    for note in doc.get("manual") or []:
        rep.items.append(AuditItem("人工确认", "manual", str(note)))
    return rep
