# %%
"""Extract high-resolution figure regions from a PDF (PyMuPDF version-safe)."""

import json
import pathlib
import sys

import fitz

DPI = 600
MIN_AREA_FRAC = 0.035
MIN_W_FRAC = 0.22
MIN_H_FRAC = 0.045

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


def extract(pdf_path, out_dir, max_figs=6, *, crop_header=True):
    """抽取图区；``crop_header=False`` 关闭刊头裁切（默认开启）。

    刊头**只在导出时裁掉**、不参与候选合并：图区合并与排序保持原样，figN 与
    既有产物一一对应（下游按 figN 引用，重提取不得改编号）。
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
    keep = keep[:max_figs]
    manifest = []
    for i, (pno, r) in enumerate(keep, 1):
        clip = (fitz.Rect(r) + (-6, -6, 6, 6)) & doc[pno].rect
        cut = header_cuts.get(pno)
        if cut is not None:  # 上扩 6pt 不得把刊头重新框进来
            clip.y0 = max(clip.y0, cut + HEADER_GAP)
        pix = doc[pno].get_pixmap(
            matrix=fitz.Matrix(DPI / 72, DPI / 72), clip=clip, alpha=False
        )
        fn = out_dir / f"fig{i}.png"
        pix.save(str(fn))
        manifest.append(
            {
                "fig": i,
                "page": pno + 1,
                "bbox": [
                    round(r.x0, 1),
                    round(r.y0, 1),
                    round(r.x1, 1),
                    round(r.y1, 1),
                ],
                "px": [pix.width, pix.height],
                "file": fn.name,
            }
        )
    doc.close()
    (out_dir / "figures.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    for row in extract(sys.argv[1], sys.argv[2]):
        print(row)
