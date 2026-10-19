# %%
"""Extract high-resolution figure regions from a PDF (PyMuPDF version-safe)."""

import argparse
import json
import pathlib
import re

import fitz

DPI = 600
MIN_AREA_FRAC = 0.035
MIN_W_FRAC = 0.22
MIN_H_FRAC = 0.045
#: 默认图区上限（原先写死 6，RPP 86 的 Fig.7 因此从未进入候选；0/负数＝不限）。
DEFAULT_MAX_FIGS = 12

# 期刊页眉/刊头：页面顶部区的全宽细线（PRL 页眉横线）或全宽彩色横幅
# （Nature 系红/蓝刊头条）。检测到后把侵入该区域的候选裁到其下沿——
# 否则 merge(pad=18) 会把横幅与图区连成一块，导出图顶部带刊头。
HEADER_ZONE_FRAC = 0.12
HEADER_MIN_W_FRAC = 0.60
HEADER_RULE_MAX_H = 2.0
HEADER_BANNER_MAX_H_FRAC = 0.06
HEADER_GAP = 1.0


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


def merge(rects, pad=18):
    rects = [fitz.Rect(r) for r in rects]
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
                    if (r + (-pad, -pad, pad, pad)).intersects(q):
                        r = r | q
                        grew = True
                        changed = True
                    else:
                        rest.append(q)
                rects[:] = rest
            out.append(r)
        rects = out
    return rects


_FIG_CAPTION_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)", re.I)


def _caption_regions(page, pa, cut):
    """按 ``FIG. N`` 题注反推图区：题注上方的全部图元并集（救多面板/窄间距图）。

    PRL 123 Fig.2 是三个矢量面板，面板间隙 24pt > merge 的 pad 18 → 单面板宽度
    0.18 < MIN_W_FRAC 0.22，整图被丢弃。题注反推不受面板间距影响。
    """
    pr = page.rect
    rects = []
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            rr = fitz.Rect(r)
            if rr.width > 5 and rr.height > 5:
                rects.append(rr)
    for d in page.get_drawings():
        rr = fitz.Rect(d["rect"])
        if rr.width > 5 and rr.height > 5:
            rects.append(rr)
    out = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[:5]
        if not _FIG_CAPTION_RE.search(str(txt)):
            continue
        top = max(pr.y0, y0 - 260)
        if cut is not None:
            top = max(top, cut + HEADER_GAP)
        band = fitz.Rect(max(pr.x0, x0 - 4), top, min(pr.x1, x1 + 4), y0 - 2)
        members = [
            r
            for r in rects
            if r.intersects(band) and r.y0 >= band.y0 - 2 and r.y1 <= y0 + 4
        ]
        if not members:
            continue
        union = members[0]
        for r in members[1:]:
            union = union | r
        out.append(union)
    return out


def _caption_figure_no(page, rect) -> str:
    """区域内/上下邻近文本里的原文图号（``Fig. 7`` / ``FIGURE 2``）；判不出空串。"""
    pat = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)", re.I)
    below: list[tuple[float, str]] = []
    above: list[tuple[float, str]] = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, txt = b[:5]
        if x1 < rect.x0 - 20 or x0 > rect.x1 + 20:
            continue
        m = pat.search(str(txt))
        if not m:
            continue
        no = f"Fig. {m.group(1)}"
        if rect.y1 - 5 <= y0 <= rect.y1 + 160:
            below.append((y0, no))
        elif rect.y0 - 160 <= y1 <= rect.y0 + 5:
            above.append((y1, no))
    if below:
        return sorted(below)[0][1]
    if above:
        return sorted(above)[-1][1]
    return ""


def extract(pdf_path, out_dir, max_figs=None, *, crop_header=True, bboxes=None):
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
    header_cuts = {}
    for pno in range(len(doc)):
        page = doc[pno]
        pr = page.rect
        pa = abs(pr.width * pr.height)
        drawings = page.get_drawings()
        cut = header_bottom(page, drawings) if crop_header else None
        header_cuts[pno] = cut
        rects = []
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]):
                rr = fitz.Rect(r)
                if abs(rr.width * rr.height) > 0.01 * pa:
                    rects.append(rr)
        for d in drawings:
            rr = fitz.Rect(d["rect"])
            if abs(rr.width * rr.height) > 0.002 * pa:
                rects.append(rr)
        for m in merge(rects):
            w, h = m.width, m.height
            if w <= 1 or h <= 1:
                continue
            if abs(w * h) < MIN_AREA_FRAC * pa:
                continue
            if w < MIN_W_FRAC * pr.width or h < MIN_H_FRAC * pr.height:
                continue
            if w > 0.97 * pr.width and h > 0.97 * pr.height:
                continue
            cands.append((pno, m))
        for m in _caption_regions(page, pa, cut):
            w, h = m.width, m.height
            if w <= 1 or h <= 1:
                continue
            if abs(w * h) < MIN_AREA_FRAC * pa:
                continue
            if w < MIN_W_FRAC * pr.width or h < MIN_H_FRAC * pr.height:
                continue
            if w > 0.97 * pr.width and h > 0.97 * pr.height:
                continue
            cands.append((pno, m))
    keep = []
    for pno, r in sorted(
        cands, key=lambda x: abs(x[1].width * x[1].height), reverse=True
    ):
        dup = False
        for p, q in keep:
            if p == pno:
                inter = r & q
                if (
                    abs(inter.width * inter.height) / max(1.0, abs(r.width * r.height))
                    > 0.9
                ):
                    dup = True
                    break
        if not dup:
            keep.append((pno, r))
    limit = DEFAULT_MAX_FIGS if max_figs is None else int(max_figs)
    if limit > 0:
        keep = keep[:limit]
    overrides = dict(bboxes or {})
    final: list[tuple[int, fitz.Rect]] = list(keep)
    for idx in sorted(overrides):
        pg, x0, y0, x1, y1 = overrides[idx]
        rr = fitz.Rect(x0, y0, x1, y1) & doc[int(pg) - 1].rect
        if 1 <= idx <= len(final):
            final[idx - 1] = (int(pg) - 1, rr)
        else:
            final.append((int(pg) - 1, rr))
    manifest = []
    for i, (pno, r) in enumerate(final, 1):
        override = i in overrides
        clip = (
            fitz.Rect(r) & doc[pno].rect
            if override
            else (fitz.Rect(r) + (-6, -6, 6, 6)) & doc[pno].rect
        )
        cut = header_cuts.get(pno)
        if cut is not None and not override:  # 上扩 6pt 不得把刊头重新框进来
            clip.y0 = max(clip.y0, cut + HEADER_GAP)
        pix = doc[pno].get_pixmap(
            matrix=fitz.Matrix(DPI / 72, DPI / 72), clip=clip, alpha=False
        )
        fn = out_dir / f"fig{i}.png"
        pix.save(str(fn))
        page = doc[pno]
        area_frac = abs(r.width * r.height) / max(
            1.0, abs(page.rect.width * page.rect.height)
        )
        entry = {
            "fig": i,
            "page": pno + 1,
            "bbox": [round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)],
            "px": [pix.width, pix.height],
            "file": fn.name,
            "figure_no": _caption_figure_no(page, r),
        }
        if override:
            entry["override"] = True
            entry["hint"] = "用户 bbox 落参（整页/批注裁切）"
        elif area_frac >= 0.90:
            entry["hint"] = "整页图：需裁切参数（--bbox）"
        manifest.append(entry)
    doc.close()
    (out_dir / "figures.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PDF 图区提取（600dpi，可配上限/批注 bbox）"
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
    a = ap.parse_args()
    boxes = {}
    for spec in a.bbox:
        idx, pg, coords = spec.split(":", 2)
        x0, y0, x1, y1 = (float(v) for v in coords.split(","))
        boxes[int(idx)] = (int(pg), x0, y0, x1, y1)
    for row in extract(a.pdf, a.out_dir, a.max_figs, bboxes=boxes):
        print(row)
