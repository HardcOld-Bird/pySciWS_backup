"""bilingual reorder-suggest —— 源 md 文序订正建议（bilingual-orderfix / B 路径）。

minerU 源 md 文序错乱实锤（样本 ``example/sample_name``：Fig.1/Fig.2
图注句被拆成残段散落、Eq. (6) 与引导句被图块隔开、孤立子图标签逆序散落）。本工具基于
**源 PDF 文本块坐标**推断真实阅读序，输出**订正建议**供 lit 语义校对——只出建议，
**不改源 md 与 cache**（人工校对是设计的一部分；无 ``--apply`` 之类静默改写）。

输出（工作区 ``data/skills/literature_research/bilingual/<stem>/``）::

    reorder-suggested.md   建议订正版源 md（与源 md 同构，便于 diff）
    reorder-report.md      改动清单：原位置→建议位置 + 依据（PDF 页/带/栏/纵坐标）+ 类型

文序推断：PyMuPDF 文本块 bbox → 排版带聚类（同页 y 交叠并簇；图表带按 (y,x)、文本带按
「左栏自上而下 → 右栏自上而下」）→ md 块与 PDF 块按文本相似度映射 → **同一 PDF 块的
md 块归组同位**（图注残段由此自动缝回图注）→ 建议序 = 键排序。映射不上的块挂最近邻居
的键、报告单列「未映射」提示人工注意，不硬排。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

#: 文本相似度映射阈值（低于此值记「未映射」，不硬排）。
MAP_THRESHOLD = 0.45

#: 块类型：标题 / 图片 / 公式 / 表格 / 孤立面板标签 / 图注头 / 图注残段 / 正文。
KINDS = ("heading", "image", "formula", "table", "label", "caption", "fragment", "text")

_HEADING_RE = re.compile(r"^#{1,6}\s")
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")
_LABEL_RE = re.compile(r"^[a-e]$")
_CAPTION_RE = re.compile(r"^\s*(?:FIG(?:URE)?\.?|Fig\.?|图)\s*[\dA-Z]", re.I)
#: 图注残段起头形态（实锤两类：「−100Hz to 100 Hz…」续写、「c Experimental…」子图注）。
#: 数字起头不认「10. Author…」式参考文献条目（整体负向先行，防 \d+ 回溯漏判）。
_FRAGMENT_RE = re.compile(
    r"^(?![-−]?\d+\.\s)(?:[-−]?\d|[a-e](?:\s*,\s*[a-e])*\s+[A-Z])"
)
_SENT_END = (".", "?", "!", "。")


@dataclass
class MdBlock:
    """源 md 的一个结构块（原文逐字保存，建议稿按块搬运不改字）。"""

    kind: str
    text: str
    index: int
    line: int  # 源 md 起始行号（1-based，报告依据）


@dataclass
class PdfBlock:
    """PDF 文本块：坐标 + 阅读序键（页/排版带/栏序/纵坐标）。"""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    band: int = -1
    col: float = 0.0

    @property
    def key(self) -> tuple[int, int, float, float, float]:
        """阅读序键：(页, 带, 栏序, y, x)。"""
        return (self.page, self.band, self.col, self.y0, self.x0)

    @property
    def basis(self) -> str:
        """报告「依据」短语：PDF 页/排版带/栏/纵坐标。"""
        side = (
            "通栏·图表" if self.col == 0.5 else ("左栏" if self.col < 0.5 else "右栏")
        )
        return f"PDF p{self.page} 带{self.band} {side} y={self.y0:.0f}"


def parse_md_blocks(md_text: str) -> list[MdBlock]:
    """把源 md 切成结构块（标题/图片/公式/表格/标签/图注/正文；原文逐字）。"""
    blocks: list[MdBlock] = []
    lines = md_text.split("\n")
    i, n = 0, len(lines)

    def add(kind: str, text: str, start: int) -> None:
        blocks.append(MdBlock(kind=kind, text=text, index=len(blocks), line=start + 1))

    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _HEADING_RE.match(stripped):
            add("heading", stripped, i)
            i += 1
            continue
        if _IMAGE_RE.match(stripped):
            add("image", stripped, i)
            i += 1
            continue
        if stripped.startswith("$$"):
            buf = [lines[i]]
            if stripped.count("$$") < 2:
                i += 1
                while i < n:
                    buf.append(lines[i])
                    if "$$" in lines[i]:
                        i += 1
                        break
                    i += 1
            else:
                i += 1
            add("formula", "\n".join(buf), i - len(buf))
            continue
        if stripped.startswith("<table"):
            buf = [lines[i]]
            i += 1
            while i < n and "</table>" not in buf[-1]:
                buf.append(lines[i])
                i += 1
            add("table", "\n".join(buf), i - len(buf))
            continue
        start = i
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (
                not nxt
                or _HEADING_RE.match(nxt)
                or _IMAGE_RE.match(nxt)
                or nxt.startswith("$$")
                or nxt.startswith("<table")
            ):
                break
            buf.append(nxt)
            i += 1
        text = " ".join(buf)
        if _LABEL_RE.match(text):
            kind = "label"
        elif _CAPTION_RE.match(text):
            kind = "caption"
        elif _FRAGMENT_RE.match(text):
            kind = "fragment"
        else:
            kind = "text"
        add(kind, text, start)
    return blocks


# ---------------------------------------------------------------------------
# PDF 文本块与排版带聚类
# ---------------------------------------------------------------------------
def pdf_text_blocks(pdf_path: Path) -> list[PdfBlock]:
    """PyMuPDF 提取源 PDF 全部文本块（矢量图无文本块，图注/面板标签即文本块）。"""
    import fitz  # type: ignore

    out: list[PdfBlock] = []
    with fitz.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc):
            width = float(page.rect.width)
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, _bno, btype = b
                if btype != 0 or not str(text).strip():
                    continue
                out.append(
                    PdfBlock(
                        page=pno + 1,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        text=" ".join(str(text).split()),
                    )
                )
            for pb in out:
                if pb.page == pno + 1:
                    pb.col = (
                        0.5
                        if (pb.x1 - pb.x0) > 0.55 * width
                        else (0.0 if (pb.x0 + pb.x1) / 2 < width / 2 else 1.0)
                    )
    _assign_bands(out)
    return out


def _assign_bands(blocks: list[PdfBlock]) -> None:
    """同页 y 交叠并簇成排版带（图/整页带自然成簇），带序 = 自上而下。"""
    by_page: dict[int, list[PdfBlock]] = {}
    for pb in blocks:
        by_page.setdefault(pb.page, []).append(pb)
    for page, group in by_page.items():
        group.sort(key=lambda p: (p.y0, p.x0))
        band_of: list[tuple[float, float]] = []  # 每带的 (y0, y1) 并区间
        for pb in group:
            hit = None
            for bi, (top, bottom) in enumerate(band_of):
                overlap = min(bottom, pb.y1) - max(top, pb.y0)
                if overlap > 0.3 * min(bottom - top, pb.y1 - pb.y0):
                    hit = bi
                    break
            if hit is None:
                band_of.append((pb.y0, pb.y1))
                pb.band = len(band_of) - 1
            else:
                top, bottom = band_of[hit]
                band_of[hit] = (min(top, pb.y0), max(bottom, pb.y1))
                pb.band = hit
        # 带序按区间上沿重排（并簇不保证顺序）
        order = {
            bi: rank
            for rank, (bi, _) in enumerate(
                sorted(enumerate(band_of), key=lambda kv: kv[1][0])
            )
        }
        for pb in group:
            pb.band = order[pb.band]
        # 图表带（短文本密布、横向铺开）→ 带内按 (y,x)；文本带按（栏, y）
        for bi in range(len(band_of)):
            members = [p for p in group if p.band == bi]
            short = [p for p in members if len(p.text) < 30]
            span = max(p.x1 for p in members) - min(p.x0 for p in members)
            width_mid = 297.0  # A4/letter 半宽量级；只用作铺开判据
            if len(short) >= max(2, 0.6 * len(members)) and span > width_mid:
                for p in members:
                    p.col = 0.5
    return None


# ---------------------------------------------------------------------------
# md 块 ↔ PDF 块 相似度映射
# ---------------------------------------------------------------------------
_LATEX_RE = re.compile(r"[\\$^_{}~]")
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _norm(text: str) -> str:
    """归一：去 LaTeX 记号/标点、压空白（md 数学与 PDF 字形差异由此消音）。"""
    return " ".join(_TOKEN_RE.findall(_LATEX_RE.sub(" ", str(text)).lower()))


def similarity(md_text: str, pdf_text: str) -> float:
    """md 块与 PDF 块文本相似度（0..1）：归一后序列比 + 词元包含度 + 首锚三路取大。"""
    n1, n2 = _norm(md_text), _norm(pdf_text)
    if not n1 or not n2:
        return 0.0
    ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    t1, t2 = set(n1.split()), set(n2.split())
    contain = len(t1 & t2) / len(t1) if t1 else 0.0
    anchor = 0.9 if n1[:40] and n1[:40] in n2 else 0.0
    return max(ratio, 0.55 * contain + 0.2, anchor)


def map_blocks(
    md_blocks: list[MdBlock], pdf_blocks: list[PdfBlock]
) -> tuple[dict[int, PdfBlock], list[MdBlock]]:
    """逐 md 块找最佳 PDF 块（≥:data:`MAP_THRESHOLD`）；返回 (映射, 未映射清单)。

    公式块走 **编号尾锚**（``\\tag{N}`` ↔ PDF 块以 ``(N)`` 收尾）：数学字形经归一后
    相似度不可靠，而公式编号在两端都铁定在。
    """
    mapped: dict[int, PdfBlock] = {}
    unmapped: list[MdBlock] = []
    tag_re = re.compile(r"\\tag\{(\d+)\}")
    for mb in md_blocks:
        if mb.kind in ("image",):
            continue  # 图片无文本，走「挂图注组」锚定
        best, best_score = None, 0.0
        if mb.kind == "formula":
            m = tag_re.search(mb.text)
            if m:
                # PDF 数学字形的括号常成「ð/Þ」伪字形；编号尾锚按 [(ð]N[)Þ]$ 判。
                tail = re.compile(rf"[(ð]\s*{re.escape(m.group(1))}\s*[)Þ]\s*$")
                for pb in pdf_blocks:
                    if tail.search(pb.text):
                        best, best_score = pb, 0.9
                        break
        if best is None and _LABEL_RE.match(mb.text):
            # 单字符面板标签：以「文本全等」配 PDF 图注/标签块（相似度对单字无感）。
            for pb in pdf_blocks:
                if pb.text.strip().rstrip(".") == mb.text.strip():
                    best, best_score = pb, 0.9
                    break
        if best is None:
            for pb in pdf_blocks:
                s = similarity(mb.text, pb.text)
                if s > best_score:
                    best, best_score = pb, s
        if best is not None and best_score >= MAP_THRESHOLD:
            mapped[mb.index] = best
        else:
            unmapped.append(mb)
    return mapped, unmapped


# ---------------------------------------------------------------------------
# 建议序（键归组）与专项修复建议
# ---------------------------------------------------------------------------
def suggest_order(
    md_blocks: list[MdBlock], mapped: dict[int, PdfBlock]
) -> dict[int, tuple]:
    """给每个 md 块算建议排序键；返回 {块 index: 排序键}。

    图片无文本 → 挂最近图注头块（优先后继 ≤8 块）的键；未映射文本 → 挂最近映射
    邻居的键（仅作定位，报告单列「未映射」，不硬排）。同一 PDF 块的 md 块同键归组
    （图注残段由此自动缝回图注），组内保持原相对序。
    """
    keys: dict[int, tuple] = {}
    for mb in md_blocks:
        if mb.index in mapped:
            keys[mb.index] = mapped[mb.index].key
    caption_idx = [b.index for b in md_blocks if b.kind == "caption"]
    image_idx = [b.index for b in md_blocks if b.kind == "image"]
    for mb in md_blocks:
        if mb.index in keys:
            continue
        if mb.kind == "image":
            near = [c for c in caption_idx if 0 < abs(c - mb.index) <= 16]
            anchor = min(near, key=lambda c: abs(c - mb.index)) if near else None
        elif mb.kind == "label":
            # 面板标签挂**相邻图组**（图前图后 ≤2 块）：「d/c/b」是面板铭牌不是正文
            near = [c for c in image_idx if 0 < abs(c - mb.index) <= 2]
            anchor = min(near, key=lambda c: abs(c - mb.index)) if near else None
        else:
            anchor = None
        if anchor is None:
            mapped_idx = sorted(keys)
            if not mapped_idx:
                continue
            anchor = min(mapped_idx, key=lambda k: abs(k - mb.index))
        keys[mb.index] = keys[anchor]
    return keys


def _stitch_pairs(
    order: list[int],
    md_blocks: list[MdBlock],
    keys: dict[int, tuple],
    mapped: dict[int, PdfBlock],
) -> set[int]:
    """返回「应缝回前块」的块 index：同排版带、纵距 ≤150pt 的图注残段。

    同 PDF 块当然同带；跨栏图注（左栏图注头 + 右栏续段同 y）也由此缝回——实锤的
    Fig.1/Fig.2 残段正是这种「同带不同栏」形态。前块须为图注/残段（正文不吞）。
    """
    stitches: set[int] = set()
    for pos, cur_i in enumerate(order):
        cur = md_blocks[cur_i]
        if cur.kind != "fragment":
            continue
        # 向前找最近的**文本**前块（跨过图片/标签行——残段缝回的是图注文字）
        prev_i = next(
            (
                order[j]
                for j in range(pos - 1, -1, -1)
                if md_blocks[order[j]].kind not in ("image", "label")
            ),
            None,
        )
        if prev_i is None:
            continue
        prev = md_blocks[prev_i]
        if prev.kind not in ("caption", "fragment"):
            continue
        k1, k2 = keys.get(prev_i), keys.get(cur_i)
        if k1 is None or k2 is None or k1[:2] != k2[:2]:
            continue
        pb1, pb2 = mapped.get(prev_i), mapped.get(cur_i)
        if pb1 is not None and pb2 is not None and abs(pb1.y0 - pb2.y0) > 150:
            continue
        stitches.add(cur_i)
    return stitches


def build_report(
    order: list[int],
    md_blocks: list[MdBlock],
    keys: dict[int, tuple],
    mapped: dict[int, PdfBlock],
    unmapped: list[MdBlock],
    stitches: set[int],
) -> str:
    """改动清单：每处「原位置 → 建议位置」+ 依据（PDF 页/带/栏/纵坐标）+ 建议类型。"""
    lines = ["# reorder-report：源 md 文序订正建议", ""]
    moved = [
        (pos, idx)
        for pos, idx in enumerate(order)
        # 缝合与面板标签必进清单（专项建议不受位移为零限制）；其余未映射不硬排只单列
        if idx in stitches
        or md_blocks[idx].kind == "label"
        or (md_blocks[idx].index != pos and idx in mapped)
    ]
    lines.append(
        f"共 {len(md_blocks)} 块；建议移动/缝合 {len(moved)} 处；未映射 {len(unmapped)} 块。"
    )
    lines.append("")
    for pos, idx in moved:
        mb = md_blocks[idx]
        pb = mapped.get(idx)
        if idx in stitches:
            kind = "图注残段缝合"
        elif mb.kind == "label":
            kind = "标签归位或剔出可译单元"
        elif (
            mb.kind == "formula"
            and pos > 0
            and md_blocks[order[pos - 1]].text.rstrip().endswith(":")
        ):
            kind = "移动（公式归位引导句后）"
        else:
            kind = "移动"
        where = f"{mb.line} 行（{mb.kind}）「{mb.text[:40].strip()}…」"
        basis = pb.basis if pb is not None else "未映射（挂最近邻居键定位，人工注意）"
        prev_i = next(
            (
                order[j]
                for j in range(pos - 1, -1, -1)
                if md_blocks[order[j]].kind not in ("image", "label")
            ),
            None,
        )
        target = md_blocks[prev_i] if prev_i is not None else None
        to = f"建议位 {pos}" + (
            f"（紧跟 {target.line} 行「{target.text[:24].strip()}…」）"
            if target
            else ""
        )
        lines.append(f"- [{kind}] {where} → {to}")
        lines.append(f"  依据：{basis}")
    if unmapped:
        lines.append("")
        lines.append("## 未映射（人工注意，不硬排）")
        for mb in unmapped:
            lines.append(f"- {mb.line} 行（{mb.kind}）「{mb.text[:48].strip()}…」")
    lines.append("")
    return "\n".join(lines)


def build_suggested_md(
    order: list[int], md_blocks: list[MdBlock], stitches: set[int]
) -> str:
    """按建议序拼出订正版源 md（块原文逐字；缝合块并回最近文本块，空行分块与源同构）。"""
    out: list[str] = []
    text_at: list[int] = []  # out 中「文本块」的位置下标（缝合并回目标）
    for idx in order:
        mb = md_blocks[idx]
        if idx in stitches and text_at:
            out[text_at[-1]] = out[text_at[-1]].rstrip() + " " + mb.text.strip()
            continue
        out.append(mb.text)
        if mb.kind not in ("image", "label"):
            text_at.append(len(out) - 1)
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 命令面：bilingual reorder-suggest
# ---------------------------------------------------------------------------
def resolve_source_md(target: str) -> tuple[Path, str]:
    """目标 → ``(源 md 路径, stem)``：路径直用；否则在 cache/extracted（含子目录）解析。"""
    p = Path(target)
    if p.is_file():
        return p.resolve(), p.stem
    from .config import settings

    for cand in (settings.cache_extracted / f"{target}.md",):
        if cand.is_file():
            return cand.resolve(), cand.stem
    hits = sorted(settings.cache_extracted.rglob(f"{target}*.md"))
    if len(hits) == 1:
        return hits[0].resolve(), hits[0].stem
    raise FileNotFoundError(
        f"源 md 不存在：{target!r}（cache/extracted 含子目录均未命中）"
    )


def cmd_reorder_suggest(target: str, *, pdf: str | None = None) -> int:
    """``bilingual reorder-suggest``：输出文序订正建议（只出建议，不改源 md 与 cache）。

    Args:
        target: 源 md 路径或 stem。
        pdf: 手工指定源 PDF（反解不到时用；缺省走 bilingual 的 DOI/arXiv/slug 反解）。

    Returns:
        退出码（0=成功；2=前置不满足）。
    """
    from . import bilingual as _bl

    try:
        source_md, stem = resolve_source_md(target)
    except FileNotFoundError as exc:
        print(f"[reorder] {exc}", flush=True)
        return 2
    if pdf:
        src_pdf: Path | None = Path(pdf).expanduser()
        if not src_pdf.is_file():
            print(f"[reorder] --pdf 指定的文件不存在：{src_pdf}", flush=True)
            return 2
        src_pdf = src_pdf.resolve()
    else:
        src_pdf = _bl._find_source_pdf(stem, source_md=source_md)
    if src_pdf is None:
        print(f"[reorder] 缺源 PDF，无法推断文序（用 --pdf 指定）：{stem}", flush=True)
        return 2
    md_blocks = parse_md_blocks(source_md.read_text(encoding="utf-8"))
    pdf_blocks = pdf_text_blocks(src_pdf)
    mapped, unmapped = map_blocks(md_blocks, pdf_blocks)
    keys = suggest_order(md_blocks, mapped)
    order = sorted(keys, key=lambda i: (keys[i], i))
    if len(order) != len(md_blocks):
        print("[reorder] 警告：部分块无排序键，已按原文序兜底", flush=True)
        have = set(order)
        order += [b.index for b in md_blocks if b.index not in have]
    stitches = _stitch_pairs(order, md_blocks, keys, mapped)
    report = build_report(order, md_blocks, keys, mapped, unmapped, stitches)
    suggested = build_suggested_md(order, md_blocks, stitches)
    workdir = _bl.workdir_for(stem)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "reorder-suggested.md").write_text(suggested, encoding="utf-8")
    (workdir / "reorder-report.md").write_text(report, encoding="utf-8")
    n_move = len(
        [
            1
            for pos, idx in enumerate(order)
            if idx in stitches
            or md_blocks[idx].kind == "label"
            or (md_blocks[idx].index != pos and idx in mapped)
        ]
    )
    print(
        f"[reorder] {stem}：块 {len(md_blocks)}（映射 {len(mapped)}/未映射 {len(unmapped)}），"
        f"建议移动/缝合 {n_move} 处，缝合 {len(stitches)} 处",
        flush=True,
    )
    print(f"[reorder] 建议稿：{workdir / 'reorder-suggested.md'}", flush=True)
    print(f"[reorder] 改动清单：{workdir / 'reorder-report.md'}", flush=True)
    return 0
