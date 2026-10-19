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


def _check_decoration(slide: Any, entry: dict[str, Any], page: int) -> AuditItem:
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
    if want:
        got = _box(sh)
        if any(abs(a - float(b)) > tol for a, b in zip(got, want)):
            return AuditItem(
                f"装饰 p{page} {label}",
                "fail",
                f"bbox {got} ≠ spec {want}（tol {tol}）",
            )
    if kind and kind.lower() not in str(getattr(sh, "shape_type", "")).lower():
        # kind 语义宽松（rectangle/picture/text），仅在显式给出时提示
        actual = str(getattr(sh, "name", "")) + " " + str(getattr(sh, "shape_type", ""))
        if kind.lower() not in actual.lower():
            return AuditItem(
                f"装饰 p{page} {label}", "fail", f"kind 不匹配（实际 {actual[:60]}）"
            )
    return AuditItem(f"装饰 p{page} {label}", "pass", f"存在 @ {_box(sh)}（{sh.name}）")


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
    pages = doc.get("pages") or []
    for page_spec in pages:
        page = int(page_spec.get("page") or 0)
        if not 1 <= page <= len(prs.slides):
            rep.items.append(AuditItem(f"第 {page} 页", "fail", "页码越界"))
            continue
        slide = prs.slides[page - 1]
        for entry in page_spec.get("decoration") or []:
            rep.items.append(_check_decoration(slide, entry, page))
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
        for note in page_spec.get("manual") or []:
            rep.items.append(AuditItem(f"人工确认 p{page}", "manual", str(note)))

    for entry in inv.get("decoration") or []:
        # 顶层 decoration：在所有页找一遍（模板级不变量）
        page = int(entry.get("page") or 0)
        if 1 <= page <= len(prs.slides):
            rep.items.append(_check_decoration(prs.slides[page - 1], entry, page))
        else:
            rep.items.append(
                AuditItem(
                    f"装饰 {entry.get('name') or '?'}", "fail", "spec 未给有效 page"
                )
            )
    for note in doc.get("manual") or []:
        rep.items.append(AuditItem("人工确认", "manual", str(note)))
    return rep
