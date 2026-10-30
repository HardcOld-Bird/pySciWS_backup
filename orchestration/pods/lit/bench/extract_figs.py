# %%
"""Extract high-resolution figure regions from a PDF (PyMuPDF version-safe).

R4 提取器硬化（backlog 20261019-115501/120811/121447）：

- 只认**真题注**（块首 ``FIG. N``），正文里的 "shown in Fig. N" 不再反推图区；
- 候选按**文本占比**过滤：页眉/作者块、整页正文扫描、含正文的全宽版面条直接拒绝；
- 题注反推从题注向上按 ≤40 pt 面板间隙扩展，并纳入邻近短文本（坐标轴刻度/轴标题）；
- 题目覆盖审计：真题注图号 vs manifest、页覆盖、每图文本占比、缓存 DOI/目录标签。
"""

import argparse
import json
import pathlib
import re
import sys

import fitz

DPI = 600
#: 跨面板合并间隙（pt）；题注/正文块是合并屏障，避免把相邻两图并成一块。
MERGE_GAP_PT = 40.0
#: 最终导出图四周留白（pt）；题注反推区域已含轴标签，这里再兜一层。
PAD_PT = 12.0
#: 候选最小尺寸（右栏小图也要能进来）。
MIN_AREA_FRAC = 0.008
MIN_W_FRAC = 0.12
MIN_H_FRAC = 0.03
#: 默认图区上限（0/负数＝不限）。
DEFAULT_MAX_FIGS = 12
#: 题注向上扩展时纳入的短文本（轴刻度/轴标题）半径（pt）。
AXIS_PAD = 35.0
AXIS_TEXT_MAX_CHARS = 25
AXIS_TEXT_MAX_H = 22.0
#: 候选内文本面积占比超过它 → 判为正文/页眉，不是图。
TEXT_HEAVY_FRAC = 0.45
#: 近整页候选；若同时文本占比 > FULL_PAGE_TEXT_FRAC 则拒绝。
FULL_PAGE_FRAC = 0.85
FULL_PAGE_TEXT_FRAC = 0.20

# 期刊页眉/刊头：页面顶部区的全宽细线（PRL 页眉横线）或全宽彩色横幅
# （Nature 系红/蓝刊头条）。检测到后把侵入该区域的候选裁到其下沿——
# 否则 merge(pad) 会把横幅与图区连成一块，导出图顶部带刊头。
HEADER_ZONE_FRAC = 0.12
HEADER_MIN_W_FRAC = 0.60
HEADER_RULE_MAX_H = 2.0
HEADER_BANNER_MAX_H_FRAC = 0.06
HEADER_GAP = 1.0

_CAPTION_START_RE = re.compile(r"^\s*(?:FIG(?:URE)?\.?)\s*(\d+)\b", re.I)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")


def header_bottom(page, drawings):
    """返回页面刊头元素的下沿（pt）；无刊头元素返回 ``None``。

    只看页面顶部 :data:`HEADER_ZONE_FRAC` 内、宽度 ≥ 页宽 60% 的矢量矩形：
    高度 ≤2 pt 视为页眉横线；高度 ≤页高 6% 且非白色填充视为彩色刊头横幅。
    白色全宽块（图区背景）不算刊头，避免误裁图顶。
    """
    pr = page.rect
    zone = pr.y0 + HEADER_ZONE_FRAC * pr.height
    cut = None
    for d in drawings:
        r = fitz.Rect(d["rect"])
        if r.y0 > zone or r.width < HEADER_MIN_W_FRAC * pr.width:
            continue
        fill = d.get("fill")
        white = fill is not None and all(c > 0.95 for c in fill)
        is_rule = r.height <= HEADER_RULE_MAX_H
        is_banner = not white and r.height <= HEADER_BANNER_MAX_H_FRAC * pr.height
        if is_rule or is_banner:
            cut = r.y1 if cut is None else max(cut, r.y1)
    return cut


def _text_blocks(page):
    """页面文本块 ``[(rect, text)]``（去空块）。"""
    out = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[:5]
        t = str(txt).strip()
        if t:
            out.append((fitz.Rect(x0, y0, x1, y1), t))
    return out


def _caption_blocks(page):
    """真题注块 ``[(no, rect, text)]``：文本**块首**必须是 ``FIG. N``。

    正文引用（"as shown in Fig. 2"）不会被误判为题注，从而不再反推出错误图区。
    """
    out = []
    for r, t in _text_blocks(page):
        m = _CAPTION_START_RE.match(t)
        if m:
            out.append((int(m.group(1)), r, t))
    return out


def _text_fraction(rect, blocks):
    """矩形内文本块覆盖面积占比（0–1；重叠文本按叠加计，封顶 1）。"""
    if rect.is_empty:
        return 0.0
    covered = 0.0
    for r, _ in blocks:
        inter = rect & r
        if not inter.is_empty:
            covered += abs(inter.width * inter.height)
    return min(1.0, covered / max(1.0, abs(rect.width * rect.height)))


def merge(rects, pad=MERGE_GAP_PT, barriers=()):
    """按 ``pad`` 合并相交图元；``barriers``（题注/正文块）内的合并被禁止。"""
    rects = [fitz.Rect(r) for r in rects]
    barriers = [fitz.Rect(r) for r in barriers]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            r = fitz.Rect(rects.pop(0))
            grew = True
            while grew:
                grew = False
                rest = []
                for q in rects:
                    if not (r + (-pad, -pad, pad, pad)).intersects(q):
                        rest.append(q)
                        continue
                    if any(b.intersects(r | q) for b in barriers):
                        rest.append(q)
                        continue
                    r = r | q
                    grew = True
                    changed = True
                rects[:] = rest
            out.append(r)
        rects = out
    return rects


def _include_axis_text(union, blocks):
    """把图区附近短文本（轴刻度/轴标题/面板字母）并入 union。"""
    expanded = union + (-AXIS_PAD, -AXIS_PAD, AXIS_PAD, AXIS_PAD)
    out = fitz.Rect(union)
    for r, t in blocks:
        if len(t) > AXIS_TEXT_MAX_CHARS or r.height > AXIS_TEXT_MAX_H:
            continue
        if _CAPTION_START_RE.match(t):
            continue
        if expanded.intersects(r):
            out |= r
    return out


def _caption_regions(page, elements, text_blocks, captions):
    """按真题注反推图区（同底行整行 + 向上 ≤40 pt 面板栈 + 邻近短文本）。

    同一行内并排的面板即使水平间隙 >40 pt，只要题注横跨它们且底边对齐，也算同一图；
    ``MERGE_GAP_PT`` 只约束**向上**堆叠的行间距。
    """
    pr = page.rect
    out = []
    for no, crect, _ in captions:
        stack = [
            fitz.Rect(r)
            for r in elements
            if r.y1 <= crect.y0 + 20 and r.x1 >= crect.x0 - 40 and r.x0 <= crect.x1 + 40
        ]
        if not stack:
            continue
        bottom = max(r.y1 for r in stack)
        row = [r for r in stack if r.y1 >= bottom - 5]
        union = fitz.Rect(row[0])
        for r in row[1:]:
            union |= r
        rest = sorted(
            (r for r in stack if r.y1 < bottom - 5), key=lambda r: r.y1, reverse=True
        )
        for r in rest:
            gap = union.y0 - r.y1
            if gap > MERGE_GAP_PT:
                break
            overlap = min(union.x1, r.x1) - max(union.x0, r.x0)
            if overlap < 0.2 * min(union.width, r.width):
                continue
            union |= r
        union = _include_axis_text(union, text_blocks)
        union = union & pr
        out.append((no, union))
    return out


def _near_caption_no(rect, captions):
    """图元附近真题注的图号（优先下方 160 pt，其次上方）；判不出空串。"""
    below = []
    above = []
    for no, c, _ in captions:
        if c.x1 < rect.x0 - 20 or c.x0 > rect.x1 + 20:
            continue
        if rect.y1 - 5 <= c.y0 <= rect.y1 + 160:
            below.append((c.y0, no))
        elif rect.y0 - 160 <= c.y1 <= rect.y0 + 5:
            above.append((c.y1, no))
    if below:
        return f"Fig. {sorted(below)[0][1]}"
    if above:
        return f"Fig. {sorted(above)[-1][1]}"
    return ""


def _doc_doi(doc):
    """缓存 PDF 的 DOI（元数据 + 前两页文本；判不出空串）。"""
    text = " ".join(
        str(doc.metadata.get(k) or "") for k in ("subject", "title", "keywords")
    )
    for pno in range(min(2, len(doc))):
        text += " " + doc[pno].get_text("text")
    m = _DOI_RE.search(text)
    return m.group(0).rstrip(".,;") if m else ""


def _norm_doi(value):
    return re.sub(r"[\s.]+$", "", str(value or "")).casefold()


def _label_claims(label):
    """目录标签声称的主题 → 论文文本应出现的正则（只查能机械判的强主题）。"""
    claims = []
    low = str(label or "").lower()
    if ("pt" in low or "parity" in low) and ("对称" in str(label) or "symmetry" in low):
        claims.append(("PT/parity-time", re.compile(r"parity[- ]?time|\bPT\b", re.I)))
    if "非厄米" in str(label) or "non-hermitian" in low:
        claims.append(
            (
                "non-Hermitian",
                re.compile(r"non[- ]?hermitian|parity[- ]?time|\bPT\b", re.I),
            )
        )
    return claims


def audit_figures(
    pdf_path, manifest, *, expected_doi=None, label=None, strict_coverage=False
):
    """交付前审计：真题注覆盖、页覆盖、每图文本占比、DOI/目录标签。

    Args:
        strict_coverage: 真为题注覆盖要求**全量**入 manifest（缺失=error）；
            默认只给 warning——deck 常按需取子集，不能让子集目录全红。

    Returns:
        ``{"ok": bool, "issues": [{level, code, detail}], ...}``；``ok`` 只由
        ``level == "error"`` 决定（warning 不阻断，但会打印）。
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    issues = []
    captions_by_no = {}
    caption_pages = set()
    page_blocks = {}
    for pno in range(page_count):
        page = doc[pno]
        blocks = _text_blocks(page)
        page_blocks[pno] = blocks
        for no, _, _ in _caption_blocks(page):
            captions_by_no.setdefault(no, pno + 1)
            caption_pages.add(pno + 1)

    covered_pages = set()
    manifest_nos = set()
    for e in manifest:
        page_no = int(e.get("page") or 0)
        covered_pages.add(page_no)
        rect = fitz.Rect(e.get("bbox") or (0, 0, 0, 0))
        blocks = page_blocks.get(page_no - 1, [])
        tf = _text_fraction(rect, blocks)
        area_frac = (
            abs(rect.width * rect.height)
            / max(1.0, abs(doc[page_no - 1].rect.width * doc[page_no - 1].rect.height))
            if 1 <= page_no <= page_count
            else 0.0
        )
        name = e.get("file") or f"fig{e.get('fig')}"
        if tf > TEXT_HEAVY_FRAC:
            issues.append(
                {
                    "level": "error",
                    "code": "text-heavy",
                    "detail": f"{name} p{page_no} 文本占比 {tf:.0%}（> {TEXT_HEAVY_FRAC:.0%}）",
                }
            )
        if area_frac >= FULL_PAGE_FRAC and tf > FULL_PAGE_TEXT_FRAC:
            issues.append(
                {
                    "level": "error",
                    "code": "full-page-text",
                    "detail": f"{name} p{page_no} 近整页（{area_frac:.0%}）且含正文 {tf:.0%}",
                }
            )
        no_text = str(e.get("figure_no") or "")
        m = re.search(r"(\d+)", no_text)
        if m:
            manifest_nos.add(int(m.group(1)))
        else:
            issues.append(
                {
                    "level": "warning",
                    "code": "no-caption",
                    "detail": f"{name} p{page_no} 未关联到真题注图号",
                }
            )
            if page_no == 1:
                issues.append(
                    {
                        "level": "warning",
                        "code": "page1-uncaptioned",
                        "detail": f"{name} 在首页且无题注图号——多为刊头/作者块/图形摘要",
                    }
                )

    missing = sorted(set(captions_by_no) - manifest_nos)
    if missing:
        issues.append(
            {
                "level": "error" if strict_coverage else "warning",
                "code": "missing-figures",
                "detail": "真题注有、manifest 缺："
                + "、".join(f"Fig. {n}（p{captions_by_no[n]}）" for n in missing),
            }
        )
    no_page = sorted(caption_pages - covered_pages)
    if no_page:
        issues.append(
            {
                "level": "warning",
                "code": "page-coverage",
                "detail": "含题注但无提取图的页：" + "、".join(map(str, no_page)),
            }
        )
    if not captions_by_no:
        issues.append(
            {
                "level": "warning",
                "code": "no-captions",
                "detail": "PDF 未检出 FIG. N 题注（可能是新闻/综述页或扫描件）",
            }
        )

    doi = _doc_doi(doc)
    if expected_doi:
        if not doi:
            issues.append(
                {
                    "level": "warning",
                    "code": "doi-not-found",
                    "detail": f"PDF 未嵌入 DOI（期望 {expected_doi}）——无法机械核对缓存",
                }
            )
        elif _norm_doi(doi) != _norm_doi(expected_doi):
            issues.append(
                {
                    "level": "error",
                    "code": "doi-mismatch",
                    "detail": f"缓存 DOI {doi} ≠ 目标 {expected_doi}",
                }
            )

    if label:
        text = " ".join(str(doc.metadata.get(k) or "") for k in ("title", "subject"))
        for pno in range(min(2, page_count)):
            text += " " + doc[pno].get_text("text")
        for claim, rx in _label_claims(label):
            if not rx.search(text):
                issues.append(
                    {
                        "level": "error",
                        "code": "label-mismatch",
                        "detail": f"目录标签声称 {claim}，但 PDF 前两页文本未见该主题",
                    }
                )

    doc.close()
    return {
        "ok": not any(i["level"] == "error" for i in issues),
        "issues": issues,
        "page_count": page_count,
        "caption_numbers": {str(k): v for k, v in sorted(captions_by_no.items())},
        "manifest_numbers": sorted(manifest_nos),
        "doi": doi,
    }


def extract(
    pdf_path,
    out_dir,
    max_figs=None,
    *,
    crop_header=True,
    bboxes=None,
    merge_gap=MERGE_GAP_PT,
    pad=PAD_PT,
    expected_doi=None,
    label=None,
    strict_coverage=False,
):
    """抽取图区；``crop_header=False`` 关闭刊头裁切（默认开启）。

    刊头**只在导出时裁掉**、不参与候选合并：图区合并与排序保持原样，figN 与
    既有产物一一对应（下游按 figN 引用，重提取不得改编号）。

    ``max_figs``：默认 :data:`DEFAULT_MAX_FIGS`（12）；``<=0`` 不限。``bboxes``：
    ``{figN(1-based): (page(1-based), x0, y0, x1, y1)}`` 用户批注 bbox 直接落参
    （整页扫描型 PDF 用它裁切；覆盖对应 figN 的检测结果）。
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    cands = []
    rejected = []
    header_cuts = {}
    for pno in range(len(doc)):
        page = doc[pno]
        pr = page.rect
        pa = abs(pr.width * pr.height)
        drawings = page.get_drawings()
        cut = header_bottom(page, drawings) if crop_header else None
        header_cuts[pno] = cut
        text_blocks = _text_blocks(page)
        captions = _caption_blocks(page)
        elements = []
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]):
                rr = fitz.Rect(r)
                if abs(rr.width * rr.height) > 0.005 * pa:
                    elements.append(rr)
        for d in drawings:
            rr = fitz.Rect(d["rect"])
            if abs(rr.width * rr.height) > 0.002 * pa:
                elements.append(rr)
        cap_regions = _caption_regions(page, elements, text_blocks, captions)
        generic = merge(
            elements,
            pad=merge_gap,
            barriers=[c for _, c, _ in captions],
        )
        raw = [(no, r, "caption") for no, r in cap_regions]
        raw += [(None, r, "generic") for r in generic]
        cap_pages = {pno for _, _ in cap_regions}
        for no, r, source in raw:
            no_text = f"Fig. {no}" if no is not None else _near_caption_no(r, captions)
            if source == "generic":
                # 题注反推已覆盖本页 → 泛候选多为重复/盒注，直接让位。
                if pno in cap_pages:
                    rejected.append(
                        {
                            "page": pno + 1,
                            "bbox": [
                                round(r.x0, 1),
                                round(r.y0, 1),
                                round(r.x1, 1),
                                round(r.y1, 1),
                            ],
                            "reason": "caption-preferred",
                            "text_frac": round(_text_fraction(r, text_blocks), 3),
                        }
                    )
                    continue
                # 首页无题注的泛候选多为刊头/作者块/图形摘要，不当论文图。
                if pno == 0 and not no_text:
                    rejected.append(
                        {
                            "page": 1,
                            "bbox": [
                                round(r.x0, 1),
                                round(r.y0, 1),
                                round(r.x1, 1),
                                round(r.y1, 1),
                            ],
                            "reason": "page1-uncaptioned",
                            "text_frac": round(_text_fraction(r, text_blocks), 3),
                        }
                    )
                    continue
            w, h = r.width, r.height
            if w <= 1 or h <= 1:
                continue
            if abs(w * h) < MIN_AREA_FRAC * pa:
                continue
            if w < MIN_W_FRAC * pr.width or h < MIN_H_FRAC * pr.height:
                continue
            if w > 0.97 * pr.width and h > 0.97 * pr.height:
                continue
            tf = _text_fraction(r, text_blocks)
            area_frac = abs(w * h) / max(1.0, pa)
            if tf > TEXT_HEAVY_FRAC:
                rejected.append(
                    {
                        "page": pno + 1,
                        "bbox": [
                            round(r.x0, 1),
                            round(r.y0, 1),
                            round(r.x1, 1),
                            round(r.y1, 1),
                        ],
                        "reason": "text-heavy",
                        "text_frac": round(tf, 3),
                    }
                )
                continue
            if area_frac >= FULL_PAGE_FRAC and tf > FULL_PAGE_TEXT_FRAC:
                rejected.append(
                    {
                        "page": pno + 1,
                        "bbox": [
                            round(r.x0, 1),
                            round(r.y0, 1),
                            round(r.x1, 1),
                            round(r.y1, 1),
                        ],
                        "reason": "full-page-text",
                        "text_frac": round(tf, 3),
                    }
                )
                continue
            cands.append(
                {
                    "page": pno,
                    "rect": r,
                    "source": source,
                    "figure_no": no_text,
                    "text_frac": tf,
                    "area": abs(w * h),
                    "hint": "整页图：需裁切参数（--bbox）" if area_frac >= 0.90 else "",
                }
            )

    # 题注反推的图区优先；泛候选若与其高度重叠则丢弃。
    cands.sort(key=lambda c: (0 if c["source"] == "caption" else 1, -c["area"]))
    keep = []
    for c in cands:
        dup = False
        for k in keep:
            if k["page"] != c["page"]:
                continue
            inter = c["rect"] & k["rect"]
            if abs(inter.width * inter.height) / max(1.0, c["area"]) > 0.9:
                dup = True
                break
        if not dup:
            keep.append(c)

    def order(c):
        m = re.search(r"(\d+)", c["figure_no"] or "")
        return (
            int(m.group(1)) if m else 999,
            c["page"],
            round(c["rect"].y0, 1),
            round(c["rect"].x0, 1),
        )

    keep.sort(key=order)
    limit = DEFAULT_MAX_FIGS if max_figs is None else int(max_figs)
    if limit > 0:
        keep = keep[:limit]
    overrides = dict(bboxes or {})
    for idx in sorted(overrides):
        pg, x0, y0, x1, y1 = overrides[idx]
        rr = fitz.Rect(x0, y0, x1, y1) & doc[int(pg) - 1].rect
        item = {
            "page": int(pg) - 1,
            "rect": rr,
            "source": "override",
            "figure_no": "",
            "text_frac": _text_fraction(rr, _text_blocks(doc[int(pg) - 1])),
            "area": abs(rr.width * rr.height),
            "hint": "",
        }
        if 1 <= idx <= len(keep):
            keep[idx - 1] = item
        else:
            keep.append(item)

    manifest = []
    for i, c in enumerate(keep, 1):
        pno = c["page"]
        r = c["rect"]
        override = c["source"] == "override"
        clip = (
            (fitz.Rect(r) + (-pad, -pad, pad, pad)) & doc[pno].rect
            if not override
            else (fitz.Rect(r) & doc[pno].rect)
        )
        cut = header_cuts.get(pno)
        if cut is not None and not override:  # 上扩不得把刊头重新框进来
            clip.y0 = max(clip.y0, cut + HEADER_GAP)
        pix = doc[pno].get_pixmap(
            matrix=fitz.Matrix(DPI / 72, DPI / 72), clip=clip, alpha=False
        )
        fn = out_dir / f"fig{i}.png"
        pix.save(str(fn))
        entry = {
            "fig": i,
            "page": pno + 1,
            "bbox": [
                round(clip.x0, 1),
                round(clip.y0, 1),
                round(clip.x1, 1),
                round(clip.y1, 1),
            ],
            "px": [pix.width, pix.height],
            "file": fn.name,
            "figure_no": c["figure_no"],
            "source": c["source"],
            "text_frac": round(c["text_frac"], 3),
        }
        if override:
            entry["override"] = True
            entry["hint"] = "用户 bbox 落参（整页/批注裁切）"
        elif c["hint"]:
            entry["hint"] = c["hint"]
        manifest.append(entry)

    report = audit_figures(
        pdf_path,
        manifest,
        expected_doi=expected_doi,
        label=label,
        strict_coverage=strict_coverage,
    )
    report["rejected"] = rejected
    (out_dir / "figures.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "figures-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    doc.close()
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PDF 图区提取（600dpi，文本占比过滤 + 题注覆盖审计）"
    )
    ap.add_argument("pdf")
    ap.add_argument("out_dir")
    ap.add_argument(
        "--max-figs", type=int, default=None, help="图区上限（默认 12；<=0 不限）"
    )
    ap.add_argument(
        "--bbox",
        action="append",
        default=[],
        help="用户批注 bbox：N:page:x0,y0,x1,y1（figN 覆盖/追加，可重复）",
    )
    ap.add_argument("--expected-doi", default="", help="期望缓存 DOI（审计用）")
    ap.add_argument("--label", default="", help="目录标签（审计主题一致性）")
    ap.add_argument(
        "--strict-coverage",
        action="store_true",
        help="题注图号必须全量入 manifest（缺失=error；默认 warning）",
    )
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP_PT)
    ap.add_argument("--pad", type=float, default=PAD_PT)
    a = ap.parse_args()
    boxes = {}
    for spec in a.bbox:
        idx, pg, coords = spec.split(":", 2)
        x0, y0, x1, y1 = (float(v) for v in coords.split(","))
        boxes[int(idx)] = (int(pg), x0, y0, x1, y1)
    man = extract(
        a.pdf,
        a.out_dir,
        a.max_figs,
        bboxes=boxes,
        merge_gap=a.merge_gap,
        pad=a.pad,
        expected_doi=a.expected_doi or None,
        label=a.label or None,
        strict_coverage=a.strict_coverage,
    )
    for row in man:
        print(row)
    report = json.loads(
        (pathlib.Path(a.out_dir) / "figures-audit.json").read_text(encoding="utf-8")
    )
    for issue in report["issues"]:
        print(f"[{issue['level']}] {issue['code']}: {issue['detail']}")
    if not report["ok"]:
        sys.exit(1)
