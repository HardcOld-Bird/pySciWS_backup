"""deck_digest —— 把大型 .pptx「翻译」为图文互证的 Markdown 工作区。

面向的场景：历史汇报 deck 往往几十上百页、图远多于字，直接把图片批量抽出会**切断
图文联系**（图在一处、字在另一处），对后续 LLM 阅读是灾难。本模块因此把翻译拆成两层：

- **原始层（程序化，一次性，零视觉成本）**：逐页扫描形状几何（bbox）、文本、图片 blob，
  按 sha1 去重导出图片，推断每张图的「邻近文字」（候选图注），生成 ASCII 版面布局图，
  检测分节页并据此分块，产出 ``sidecar/slides.jsonl`` + ``images_manifest.json`` +
  每块的**骨架 md**（每张图留「解读：_(待填)_」空位）+ ``progress.json`` 断点账本。
- **理解层（LLM，分批，视觉受限）**：每批读少量页的**整页合成渲染**（``--render``，
  需 LibreOffice：pptx→pdf→png），一次即「看见」该页全部图文的空间关系；仅在分辨率
  不足或多子图歧义时才 zoom 读原图。把解读写回骨架 md，并更新 progress.json。

设计要点：
1. 图文永不分离——md 里每张图自带「所在页 + 版面位置 + 邻近文字 + 解读」。
2. 先看合成页再按需放大——视觉调用从「逐张 N 图」压到「每图页 1 次 + 少量 zoom」。
3. 去重复用——同一 blob 只解读一次，其余处标注「同 Slide X 图 Y」。
4. 原始层机器可读（jsonl），理解层人类/LLM 可读（md），两层都在，可复用于其它 deck。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# python-pptx 是 [writing] 可选依赖；缺失时给出清晰指引而非裸 ImportError。
try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
except ImportError as e:  # pragma: no cover
    raise ImportError("python-pptx 未安装。请运行：uv sync --extra writing") from e

# 复用 pptx_io 的形状树遍历与段落抽取（含 group 递归、项目符号层级）
from .pptx_io import _iter_shapes, _text_frame_paragraphs

EMU_PER_IN = 914400
_MARKER_CHARS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ShapeRec:
    """单个形状的几何 + 语义记录（原始层的最小单元）。"""

    kind: str = ""  # picture / text / table / other
    idx: int = 0  # 图片在本页内的序号（1-based）；非图片为 0
    name: str = ""
    left: float = 0.0
    top: float = 0.0
    w: float = 0.0
    h: float = 0.0
    text: str = ""
    alt: str = ""
    pos_label: str = ""  # 九宫格位置标签，如「左上」「居中」
    # 图片专属
    img_id: str = ""  # sha1（去重键）
    img_rel: str = ""  # 相对 digest 根的图片路径，如 images/s012_p1_ab12cd34.png
    img_bytes: int = 0
    img_px: list[int] = field(default_factory=list)  # 原生像素 [w, h]
    dup_of: str = ""  # 重复图指向首次出现处，如 "Slide 005 图 2"
    near_text: list[str] = field(default_factory=list)  # 邻近文字（候选图注）
    # 不可直读格式（gif 动图 / wmf 矢量）的 PNG 预览
    previews: list[str] = field(default_factory=list)  # 预览 PNG 相对路径
    frame_count: int = 0  # 动图总帧数
    preview_status: str = ""  # needs_libreoffice / gif_failed:... / ok


@dataclass
class SlideRec:
    """单页幻灯片的完整原始记录。"""

    index: int
    layout: str = ""
    title: str = ""
    texts: list[str] = field(default_factory=list)  # 正文要点（含层级前缀）
    notes: str = ""
    pictures: list[ShapeRec] = field(default_factory=list)
    shapes: list[ShapeRec] = field(default_factory=list)  # 全部形状（供布局图/邻近文字）
    layout_map: str = ""  # ASCII 版面布局图
    is_section: bool = False
    render_rel: str = ""  # 整页合成渲染相对路径（含图页才有）


@dataclass
class DigestResult:
    """digest 的结果摘要（供 CLI 打印与测试断言）。"""

    root: Path
    source: Path
    n_slides: int = 0
    n_pictures: int = 0
    n_unique_images: int = 0
    n_duplicates: int = 0
    n_figure_slides: int = 0
    n_text_slides: int = 0
    sections: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[Path] = field(default_factory=list)
    sidecar: Path | None = None
    manifest: Path | None = None
    progress: Path | None = None
    index_md: Path | None = None
    rendered: int = 0
    render_note: str = ""
    vector_previews: int = 0  # 已转 PNG 的矢量图（wmf/emf）预览数
    reused: bool = False  # True 表示复用了既有 digest（未重新提取）
    slide_w_in: float = 0.0
    slide_h_in: float = 0.0

    def summary(self) -> str:
        dup = f"（重复 {self.n_duplicates} 张，去重后 {self.n_unique_images} 张唯一图）"
        lines = [
            f"源文件        : {self.source.name}",
            f"页面          : {self.n_slides} 页（{self.slide_w_in:.2f}×{self.slide_h_in:.2f} in）",
            f"图片          : {self.n_pictures} 张{dup}",
            f"含图页/纯文本页: {self.n_figure_slides} / {self.n_text_slides}",
            f"分节          : {len(self.sections)} 节 → {len(self.chunks)} 个 md 分块",
            f"整页合成渲染  : {self.rendered} 张"
            + (f"（{self.render_note}）" if self.render_note else ""),
            f"输出根目录    : {self.root}",
        ]
        if self.reused:
            lines.insert(0, "[复用] 检测到既有 digest，未重新提取（加 --force 可全量重建）")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 几何 / 版面小工具
# ---------------------------------------------------------------------------
def _inches(v: Any) -> float:
    try:
        return round((v or 0) / EMU_PER_IN, 2)
    except Exception:
        return 0.0


def _pos_label(cx: float, cy: float, sw: float, sh: float) -> str:
    """按中心点归一化到九宫格，返回自然语序标签（左上/居中/右下…）。"""
    if sw <= 0 or sh <= 0:
        return ""
    hor = "左" if cx < sw / 3 else ("右" if cx > 2 * sw / 3 else "中")
    ver = "上" if cy < sh / 3 else ("下" if cy > 2 * sh / 3 else "中")
    if hor == "中" and ver == "中":
        return "居中"
    return hor + ver


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _layout_map(shapes: list[ShapeRec], sw: float, sh: float, cols: int = 64, rows: int = 14) -> str:
    """把一页的形状画成 ASCII 版面图：数字/字母=图片序号，`.`=文本。

    即使没有整页渲染，也能让 LLM 知道「哪张图在左、哪段字在右下」的空间关系。
    """
    if sw <= 0 or sh <= 0:
        return ""
    grid = [[" "] * cols for _ in range(rows)]
    for s in shapes:
        if s.kind == "picture" and s.idx:
            ch = _MARKER_CHARS[(s.idx - 1) % len(_MARKER_CHARS)]
        elif s.kind in ("text", "table") and s.text.strip():
            ch = "."
        else:
            continue
        c0 = _clamp(int(s.left / sw * cols), 0, cols - 1)
        c1 = _clamp(int((s.left + s.w) / sw * cols), 0, cols - 1)
        r0 = _clamp(int(s.top / sh * rows), 0, rows - 1)
        r1 = _clamp(int((s.top + s.h) / sh * rows), 0, rows - 1)
        for r in range(r0, max(r1, r0) + 1):
            for c in range(c0, max(c1, c0) + 1):
                if grid[r][c] == " ":
                    grid[r][c] = ch
    return "\n".join("".join(row).rstrip() for row in grid)


def _near_texts(pic: ShapeRec, texts: list[ShapeRec], gap_in: float = 1.5) -> list[str]:
    """为一张图挑「候选图注」：正上/正下且水平重叠者优先，其次同排水平邻近者。"""
    scored: list[tuple[float, int, ShapeRec]] = []
    for t in texts:
        if not t.text.strip():
            continue
        h_overlap = not (t.left + t.w < pic.left or pic.left + pic.w < t.left)
        v_overlap = not (t.top > pic.top + pic.h or t.top + t.h < pic.top)
        below = t.top >= pic.top + pic.h - 0.05
        above = t.top + t.h <= pic.top + 0.05
        if h_overlap and (below or above):
            dist = (t.top - (pic.top + pic.h)) if below else (pic.top - (t.top + t.h))
            if abs(dist) <= gap_in:
                scored.append((abs(dist), 0, t))
        elif v_overlap:
            dist = max(pic.left - (t.left + t.w), t.left - (pic.left + pic.w))
            if 0 <= dist <= gap_in:
                scored.append((dist, 1, t))
    scored.sort(key=lambda x: (x[1], x[0]))
    out = []
    for _, _, t in scored[:2]:
        s = " ".join(t.text.split())
        out.append(s[:180])
    return out


def _image_payload(shape: Any) -> tuple[str, bytes, list[int], str]:
    """取图片 blob 及元信息：(ext, blob, [px_w, px_h], content_type)。"""
    img = shape.image
    ext = (img.ext or "png").lower()
    blob = img.blob
    ctype = getattr(img, "content_type", "") or ""
    px: list[int] = []
    size = getattr(img, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        px = [int(size[0]), int(size[1])]
    else:
        w, h = getattr(img, "px_width", None), getattr(img, "px_height", None)
        if w and h:
            px = [int(w), int(h)]
    return ext, blob, px, ctype


def _alt_text(shape: Any) -> str:
    try:
        return (shape._element._nvXxPr.cNvPr.get("descr") or "").strip()  # noqa: SLF001
    except Exception:
        return ""


def _render_rel(index: int) -> str:
    """整页合成渲染的相对路径（与 pdf_render 的 prefix='slide' 命名一致）。"""
    return f"renders/slide-{index:03d}.png"


# LLM 的 Read 工具只能看 jpeg/png/webp；gif（动图）与 wmf/emf（矢量）必须先转 PNG
VIEWABLE_EXTS = frozenset({"png", "jpg", "jpeg", "webp"})


def _even_indices(n: int, k: int) -> list[int]:
    """在 n 帧里均匀取 k 帧的下标（必含首末帧）。"""
    if n <= 1 or k <= 1:
        return [0]
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _make_previews(img_path: Path, root: Path, ext: str, gif_frames: int) -> dict[str, Any]:
    """为 LLM 不可直读的图生成 PNG 预览（写入 images/previews/）。

    - **gif**（汇报里多为仿真动画）：用 Pillow 抽均匀分布的若干帧（含首末帧）。
      动画的“变化趋势”靠首/中/末三帧就能大致说清，而成本只是 3 张图。
    - **wmf/emf 等矢量**：Pillow 读不了，给出**确定性**预览路径
      ``images/previews/<stem>.png``，由 LibreOffice 在 --render 阶段补齐（md 链接先写好，
      无需事后改写骨架）。
    """
    prev_dir = root / "images" / "previews"
    if ext == "gif":
        try:
            from PIL import Image

            prev_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(img_path) as im:
                n = int(getattr(im, "n_frames", 1) or 1)
                idxs = _even_indices(n, gif_frames)
                rels: list[str] = []
                for j, fi in enumerate(idxs, 1):
                    out = prev_dir / f"{img_path.stem}_f{j}of{len(idxs)}_frame{fi + 1}.png"
                    if not out.exists():
                        im.seek(fi)
                        im.convert("RGB").save(out)
                    rels.append(f"images/previews/{out.name}")
            return {"frame_count": n, "previews": rels, "preview_frames": idxs}
        except Exception as e:
            return {"preview_status": f"gif_failed:{type(e).__name__}"}
    prev_dir.mkdir(parents=True, exist_ok=True)
    return {
        "previews": [f"images/previews/{img_path.stem}.png"],
        "preview_status": "needs_libreoffice",
    }


def _make_vector_previews(root: Path, manifest: dict[str, dict]) -> tuple[int, str]:
    """用 LibreOffice 把矢量图（wmf/emf）转为 PNG 预览；返回 (成功数, 说明)。"""
    from . import office_convert
    from .config import settings

    pending = [
        m for m in manifest.values() if m.get("preview_status") == "needs_libreoffice"
    ]
    if not pending:
        return 0, ""
    if not settings.find_libreoffice():
        return 0, f"{len(pending)} 张矢量图待 LibreOffice 转 PNG 预览"
    prev_dir = root / "images" / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for m in pending:
        src = root / m["rel"]
        try:
            out = office_convert.convert(src, "png", out_dir=prev_dir, timeout=300)
        except Exception:
            continue
        if out.exists():  # soffice 输出名 = <stem>.png，与骨架里的确定性链接一致
            m["preview_status"] = "ok"
            done += 1
    note = "" if done == len(pending) else f"{len(pending) - done} 张矢量图转换失败"
    return done, note


# ---------------------------------------------------------------------------
# Phase A：扫描 + 导出 + 去重
# ---------------------------------------------------------------------------
def _scan(
    src: Path, root: Path, *, gif_frames: int = 3
) -> tuple[list[SlideRec], dict[str, dict], float, float]:
    """逐页扫描：导出图片（sha1 去重）、不可直读图转预览、记录几何/文本/邻近文字/布局图。"""
    prs = Presentation(str(src))
    sw = _inches(prs.slide_width)
    sh = _inches(prs.slide_height)

    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}  # sha1 -> 记录
    slides: list[SlideRec] = []

    for i, slide in enumerate(prs.slides, 1):
        rec = SlideRec(index=i)
        try:
            rec.layout = slide.slide_layout.name or ""
        except Exception:
            pass
        try:
            if slide.shapes.title is not None:
                rec.title = (slide.shapes.title.text or "").strip()
        except Exception:
            pass

        text_shapes: list[ShapeRec] = []
        pic_count = 0

        for shape in _iter_shapes(slide.shapes):
            sr = ShapeRec(
                name=getattr(shape, "name", "") or "",
                left=_inches(getattr(shape, "left", 0)),
                top=_inches(getattr(shape, "top", 0)),
                w=_inches(getattr(shape, "width", 0)),
                h=_inches(getattr(shape, "height", 0)),
            )
            st = getattr(shape, "shape_type", None)
            try:
                if st == MSO_SHAPE_TYPE.PICTURE:
                    pic_count += 1
                    sr.kind = "picture"
                    sr.idx = pic_count
                    sr.alt = _alt_text(shape)
                    sr.pos_label = _pos_label(
                        sr.left + sr.w / 2, sr.top + sr.h / 2, sw, sh
                    )
                    try:
                        ext, blob, px, _ctype = _image_payload(shape)
                    except Exception:
                        ext, blob, px = "bin", b"", []
                    if blob:
                        sha = hashlib.sha1(blob).hexdigest()
                        sr.img_id = sha
                        sr.img_bytes = len(blob)
                        sr.img_px = px
                        if sha in manifest:
                            m = manifest[sha]
                            sr.img_rel = m["rel"]
                            sr.dup_of = m["first_label"]
                            sr.previews = list(m.get("previews", []))
                            sr.frame_count = m.get("frame_count", 0)
                            sr.preview_status = m.get("preview_status", "")
                            m["occurrences"].append(
                                {"slide": i, "idx": pic_count, "pos": sr.pos_label}
                            )
                        else:
                            fname = f"s{i:03d}_p{pic_count}_{sha[:8]}.{ext}"
                            fpath = img_dir / fname
                            fpath.write_bytes(blob)
                            rel = f"images/{fname}"
                            sr.img_rel = rel
                            entry: dict[str, Any] = {
                                "rel": rel,
                                "ext": ext,
                                "bytes": len(blob),
                                "px": px,
                                "first_label": f"Slide {i:03d} 图 {pic_count}",
                                "first_slide": i,
                                "occurrences": [
                                    {"slide": i, "idx": pic_count, "pos": sr.pos_label}
                                ],
                            }
                            # LLM 不可直读的格式（gif 动图 / wmf 矢量）→ 生成 PNG 预览
                            if ext not in VIEWABLE_EXTS:
                                entry.update(_make_previews(fpath, root, ext, gif_frames))
                                sr.previews = list(entry.get("previews", []))
                                sr.frame_count = entry.get("frame_count", 0)
                                sr.preview_status = entry.get("preview_status", "")
                            manifest[sha] = entry
                    rec.pictures.append(sr)
                    rec.shapes.append(sr)
                    continue

                if getattr(shape, "has_table", False):
                    sr.kind = "table"
                    try:
                        cells = [
                            (c.text or "").strip().replace("\n", " ")
                            for r in shape.table.rows
                            for c in r.cells
                        ]
                        sr.text = " | ".join(x for x in cells if x)[:400]
                    except Exception:
                        pass
                    rec.shapes.append(sr)
                    text_shapes.append(sr)
                    continue

                if getattr(shape, "has_text_frame", False):
                    paras = _text_frame_paragraphs(shape.text_frame)
                    if not paras:
                        continue
                    sr.kind = "text"
                    sr.text = "\n".join(paras)
                    rec.shapes.append(sr)
                    text_shapes.append(sr)
            except Exception:
                continue

        # 标题兜底：无 title 占位符时用首个文本块
        if not rec.title and text_shapes:
            rec.title = text_shapes[0].text.splitlines()[0].lstrip("• ").strip()[:120]

        # 正文要点（排除与标题重复的首行）
        seen_lines: set[str] = set()
        for t in text_shapes:
            if t.kind != "text":
                continue
            for line in t.text.splitlines():
                clean = line.lstrip("• ").strip()
                if not clean or clean == rec.title or clean in seen_lines:
                    continue
                seen_lines.add(clean)
                rec.texts.append(line)

        # 每张图的邻近文字（候选图注）
        for p in rec.pictures:
            p.near_text = _near_texts(p, text_shapes)

        try:
            if slide.has_notes_slide:
                rec.notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        except Exception:
            pass

        rec.layout_map = _layout_map(rec.shapes, sw, sh)
        if rec.pictures:
            rec.render_rel = _render_rel(i)
        slides.append(rec)

    return slides, manifest, sw, sh


# ---------------------------------------------------------------------------
# 分节 / 分块
# ---------------------------------------------------------------------------
def _detect_sections(
    slides: list[SlideRec],
    *,
    section_starts: list[int] | None = None,
    max_sections: int = 60,
) -> list[dict[str, Any]]:
    """检测分节页 → 返回 [{start, end, title}]（1-based，含端点）。

    ``section_starts`` 给定时直接采用（**人工定界优先**，最可靠）。

    自动检测（对“全 deck 共用一个版式”的汇报稿也有效）：
    - **只有标题、没有正文、没有图片**的页 = 典型分隔页（最强信号）；
    - 或：正文极短（≤1 段且 ≤20 字）且版式像分隔页（版式名含 section/节，或该版式的
      页面多数是分隔页）；
    - 首页永不作为分节起点（它是全 deck 标题）。
    检出过多（>max_sections）时收紧为「只认版式名命中者」；一个都没检到则全 deck 归为单节。
    """
    n = len(slides)
    by_index = {s.index: s for s in slides}

    def body_of(s: SlideRec) -> list[str]:
        return [t for t in s.texts if t.strip()]

    def name_hit(s: SlideRec) -> bool:
        lay = s.layout or ""
        return "section" in lay.lower() or "节" in lay

    def title_only(s: SlideRec) -> bool:
        return (not s.pictures) and bool(s.title) and not body_of(s)

    # 版式层面的「分隔页版式」：该版式使用不多，且其页面多数是“只有标题”
    by_layout: dict[str, list[SlideRec]] = {}
    for s in slides:
        if s.layout:
            by_layout.setdefault(s.layout, []).append(s)
    divider_layouts = {
        lay
        for lay, group in by_layout.items()
        if len(group) <= max(2, int(0.2 * n))
        and sum(1 for s in group if title_only(s)) / len(group) >= 0.6
    }

    def is_divider(s: SlideRec) -> bool:
        if s.index == 1 or s.pictures or not s.title:
            return False
        if title_only(s):
            return True
        body = body_of(s)
        short = len(body) <= 1 and sum(len(t) for t in body) <= 20
        return short and (name_hit(s) or s.layout in divider_layouts)

    if section_starts:
        starts = sorted({i for i in section_starts if 1 < i <= n})
    else:
        starts = [s.index for s in slides if is_divider(s)]
        if len(starts) > max_sections:
            starts = [
                s.index for s in slides if s.index > 1 and title_only(s) and name_hit(s)
            ]
    for s in slides:
        s.is_section = s.index in set(starts)

    boundaries = [1, *starts, n + 1]
    sections: list[dict[str, Any]] = []
    for a, b in zip(boundaries, boundaries[1:], strict=False):
        if a >= b:
            continue
        if a == 1:
            title = (slides[0].title if slides else "") or "开篇"
        else:
            head = by_index.get(a)
            title = (head.title if head else "") or f"第 {a} 页起"
        sections.append({"start": a, "end": b - 1, "title": title.strip()[:60]})
    return sections


def _group_chunks(
    sections: list[dict[str, Any]],
    n_slides: int,
    *,
    chunk_by: str = "section",
    max_chunk_slides: int = 40,
    chunk_size: int = 50,
) -> list[dict[str, Any]]:
    """把「节」（语义单元）打包成「块」（md 文件单元）。

    - ``chunk_by="fixed"`` 或未检出分节：按固定页数机械切分。
    - 否则：连续的小节合并进同一块，直到再加一节会超过 ``max_chunk_slides``；
      单节本身超长时先内部切开（标题追加页码范围）。
    每块记录它包含的节，供骨架插入 ``##`` 节标题横幅、index 导航与 progress 使用。
    """
    if chunk_by == "fixed" or not sections:
        size = max(1, chunk_size)
        return [
            {
                "start": a,
                "end": min(a + size - 1, n_slides),
                "title": f"第 {a}–{min(a + size - 1, n_slides)} 页",
                "sections": [],
            }
            for a in range(1, n_slides + 1, size)
        ]

    step = max(1, max_chunk_slides)
    units: list[dict[str, Any]] = []
    for sec in sections:
        a, b = sec["start"], sec["end"]
        if b - a + 1 <= step:
            units.append({"start": a, "end": b, "secs": [sec]})
        else:
            for s0 in range(a, b + 1, step):
                s1 = min(s0 + step - 1, b)
                units.append(
                    {"start": s0, "end": s1, "secs": [{**sec, "start": s0, "end": s1}]}
                )

    grouped: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] | None = None
    cur_start = 0
    for u in units:
        if cur is None:
            cur, cur_start = list(u["secs"]), u["start"]
            continue
        if u["end"] - cur_start + 1 <= step:
            cur.extend(u["secs"])
        else:
            grouped.append(cur)
            cur, cur_start = list(u["secs"]), u["start"]
    if cur:
        grouped.append(cur)

    chunks: list[dict[str, Any]] = []
    for secs in grouped:
        title = (
            secs[0]["title"]
            if len(secs) == 1
            else f"{secs[0]['title']} 等 {len(secs)} 节"
        )
        chunks.append(
            {
                "start": secs[0]["start"],
                "end": secs[-1]["end"],
                "title": title,
                "sections": secs,
            }
        )
    return chunks


def _slug(title: str, maxlen: int = 24) -> str:
    s = re.sub(r"[\s\\/:*?\"<>|，。、；：（）()]+", "_", title.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "part")[:maxlen]


# ---------------------------------------------------------------------------
# 骨架 md / index / progress 写入
# ---------------------------------------------------------------------------
def _picture_block(p: ShapeRec) -> list[str]:
    out = [
        f"- **图 {p.idx}** ｜ 位置：{p.pos_label or '?'} ｜ 版面 {p.w:.1f}×{p.h:.1f} in"
        + (f" ｜ 原生 {p.img_px[0]}×{p.img_px[1]} px" if len(p.img_px) == 2 else "")
        + (f" ｜ {p.img_bytes // 1024} KB" if p.img_bytes else "")
    ]
    if p.img_rel:
        if p.previews:
            if p.preview_status == "needs_libreoffice":
                suf = Path(p.img_rel).suffix.lstrip(".") or "?"
                head = (
                    f"原图为**矢量图**（{suf}），LLM 不能直读；"
                    "PNG 预览待 `--render` 阶段由 LibreOffice 生成（暂可先看本页合成渲染）"
                )
            elif p.frame_count > 1:
                head = (
                    f"原图为**动图**（共 {p.frame_count} 帧），"
                    f"已抽 {len(p.previews)} 帧为 PNG 供查看"
                )
            else:
                head = "原图格式 LLM 不能直读，已转 PNG 预览"
            out.append(f"  - {head}：[原文件]({p.img_rel})")
            for pv in p.previews:
                out.append(f"  ![]({pv})")
        else:
            out.append(f"  ![]({p.img_rel})")
    if p.alt:
        out.append(f"  - 替代文本：{p.alt}")
    for nt in p.near_text:
        out.append(f"  - 邻近文字：「{nt}」")
    if p.dup_of:
        out.append(f"  - 重复图：同 {p.dup_of}（解读可直接复用，无需重看）")
    out.append("  - 解读：_(待填)_")
    return out


def _slide_block(rec: SlideRec) -> str:
    lines = [f"### Slide {rec.index:03d}" + (f" — {rec.title}" if rec.title else "")]
    meta = [f"版式：{rec.layout or '?'}", f"图片：{len(rec.pictures)} 张"]
    if rec.is_section:
        meta.append("**分节页**")
    lines.append("_" + " ｜ ".join(meta) + "_")
    if rec.render_rel:
        lines.append(f"\n![Slide {rec.index:03d} 整页合成渲染]({rec.render_rel})")

    if rec.texts:
        lines.append("\n**要点/正文**\n")
        lines.extend(rec.texts)
    if rec.notes:
        lines.append(f"\n> **演讲者备注：** {rec.notes}")

    if rec.shapes:
        lines.append("\n**版面布局**（数字/字母=图序，`.`=文本）\n")
        lines.append("```")
        lines.append(rec.layout_map)
        lines.append("```")

    if rec.pictures:
        lines.append("\n**图片**\n")
        for p in rec.pictures:
            lines.extend(_picture_block(p))
    lines.append("\n---\n")
    return "\n".join(lines)


def _write_chunk(root: Path, n: int, chunk: dict[str, Any], recs: list[SlideRec]) -> Path:
    name = f"part_{n:02d}_s{chunk['start']:03d}-{chunk['end']:03d}_{_slug(chunk['title'])}.md"
    path = root / name
    secs = chunk.get("sections") or []
    head = [
        f"# Part {n:02d} — {chunk['title']}（Slide {chunk['start']:03d}–{chunk['end']:03d}）",
        "",
    ]
    if len(secs) > 1:
        head.append(
            f"本块含 {len(secs)} 节：" + "、".join(s["title"] for s in secs)
        )
        head.append("")
    head += [
        "> 本文件由 `compose slides digest` 生成**骨架**；每张图的「解读：_(待填)_」需在 Phase B 填写。",
        "> 图片链接相对本文件；`renders/` 为整页合成渲染（`--render` 后生成，缺失属正常待补）。",
        "> 阅读顺序建议：先看该页合成渲染 → 结合要点/邻近文字理解 → 必要时 zoom 原图 → 写解读。",
        "",
        "---",
        "",
    ]
    sec_at = {s["start"]: s for s in secs}
    body: list[str] = []
    for r in recs:
        if not (chunk["start"] <= r.index <= chunk["end"]):
            continue
        s = sec_at.get(r.index)
        if s:
            body.append(
                f"## ▸ {s['title']}（Slide {s['start']:03d}–{s['end']:03d}）\n"
            )
        body.append(_slide_block(r))
    path.write_text("\n".join(head) + "\n".join(body), encoding="utf-8")
    return path


def _write_index(
    root: Path,
    res: DigestResult,
    chunks: list[dict[str, Any]],
    chunk_files: list[Path],
    manifest: dict[str, dict],
    batch_figure: int,
    batch_text: int,
    slides: list[SlideRec],
) -> Path:
    fig_slides = res.n_figure_slides
    top_dup = sorted(
        (m for m in manifest.values() if len(m["occurrences"]) > 1),
        key=lambda m: -len(m["occurrences"]),
    )[:10]

    lines = [
        f"# {res.source.stem} —— 翻译工作区（index）",
        "",
        f"- 源文件：`{res.source}`",
        f"- 规模：{res.n_slides} 页 ｜ 图片 {res.n_pictures} 张（去重后 {res.n_unique_images} 张唯一图，"
        f"重复 {res.n_duplicates} 次）｜ 含图页 {fig_slides} ｜ 纯文本页 {res.n_text_slides}",
        f"- 版面：{res.slide_w_in:.2f}×{res.slide_h_in:.2f} in",
        f"- 整页合成渲染：{res.rendered} 张"
        + (f"（{res.render_note}）" if res.render_note else ""),
        "",
        "## 分块导航",
        "",
        "| Part | Slide 范围 | 节数 | 图页/文本页 | 文件 | 含节 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, (c, f) in enumerate(zip(chunks, chunk_files, strict=False), 1):
        recs = [s for s in slides if c["start"] <= s.index <= c["end"]]
        nfig = sum(1 for s in recs if s.pictures)
        secs = c.get("sections") or []
        sec_txt = "、".join(s["title"] for s in secs) or c["title"]
        if len(sec_txt) > 64:
            sec_txt = sec_txt[:61] + "…"
        lines.append(
            f"| {n:02d} | {c['start']:03d}–{c['end']:03d} | {len(secs)} | "
            f"{nfig}/{len(recs) - nfig} | [{f.name}]({f.name}) | {sec_txt} | 待翻译 |"
        )
    lines += [
        "",
        f"## 分节清单（{len(res.sections)} 节）",
        "",
        "| # | 节标题 | Slide 范围 | 页数 | 图数 |",
        "|---|---|---|---|---|",
    ]
    for i, sec in enumerate(res.sections, 1):
        recs = [s for s in slides if sec["start"] <= s.index <= sec["end"]]
        npic = sum(len(s.pictures) for s in recs)
        lines.append(
            f"| {i:02d} | {sec['title']} | {sec['start']:03d}–{sec['end']:03d} | "
            f"{len(recs)} | {npic} |"
        )
    lines += [
        "",
        "## 高频重复图（解读一次即可复用）",
        "",
    ]
    if top_dup:
        lines.append("| 出现次数 | 首次出现 | 文件 |")
        lines.append("|---|---|---|")
        for m in top_dup:
            lines.append(
                f"| {len(m['occurrences'])} | {m['first_label']} | `{m['rel']}` |"
            )
    else:
        lines.append("_（无重复图）_")
    lines += [
        "",
        "## Phase B 操作约定",
        "",
        f"- 批大小：**{batch_figure} 个含图页 / {batch_text} 个纯文本页**每批（纯文本页无需视觉）。",
        "- 每批流程：读本 index 与对应 part 骨架 → 读该页 `renders/slide-NNN.png` 合成渲染 →",
        "  逐图填「解读」（图展示什么/坐标轴与标注/在本页论证中的角色/与理论或他页的联系）→",
        "  重复图标注「同 Slide X 图 Y」→ 更新 `progress.json` 的 `done_slides`。",
        "- 批末体检：`compose slides digest <pptx> --out <本目录> --lint`（所有图片链接须存在；",
        "  `renders/` 缺失会单列为「待渲染」而非错误）。",
        "- 全部完成后：追加一份**叙事综述版** md（把全 deck 浓缩为研究脉络长文，链接回各 part 与关键图）。",
        "",
        "## 图像语义清单（Phase B 完成后回填）",
        "",
        "| 图像文件 | 语义标签 | 出现页 |",
        "|---|---|---|",
        "| _(待填)_ | | |",
        "",
    ]
    p = root / "index.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _write_sidecar(root: Path, slides: list[SlideRec]) -> Path:
    d = root / "sidecar"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "slides.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for s in slides:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return p


def _write_manifest(root: Path, manifest: dict[str, dict]) -> Path:
    p = root / "images_manifest.json"
    p.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return p


def _write_progress(
    root: Path,
    res: DigestResult,
    chunks: list[dict[str, Any]],
    chunk_files: list[Path],
    slides: list[SlideRec],
    batch_figure: int,
    batch_text: int,
) -> Path:
    fig_by_chunk = []
    for c, f in zip(chunks, chunk_files, strict=False):
        recs = [s for s in slides if c["start"] <= s.index <= c["end"]]
        fig_by_chunk.append(
            {
                "file": f.name,
                "title": c["title"],
                "slide_range": [c["start"], c["end"]],
                "section_titles": [s["title"] for s in c.get("sections", [])],
                "figure_slides": [s.index for s in recs if s.pictures],
                "text_slides": [s.index for s in recs if not s.pictures],
                "status": "pending",
                "done_slides": [],
            }
        )
    prog = {
        "source": str(res.source),
        "root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_slides": res.n_slides,
        "n_pictures": res.n_pictures,
        "n_unique_images": res.n_unique_images,
        "n_duplicates": res.n_duplicates,
        "figure_slides": [s.index for s in slides if s.pictures],
        "text_slides": [s.index for s in slides if not s.pictures],
        "batch_policy": {"figure_slides_per_batch": batch_figure,
                         "text_slides_per_batch": batch_text},
        "render": {
            "rendered": res.rendered,
            "vector_previews": res.vector_previews,
            "note": res.render_note,
            "rel_pattern": "renders/slide-{index:03d}.png",
        },
        "sections": res.sections,
        "chunks": fig_by_chunk,
        "narrative_review": "pending",
    }
    p = root / "progress.json"
    p.write_text(json.dumps(prog, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _write_gitignore(root: Path) -> Path:
    p = root / ".gitignore"
    if not p.exists():
        p.write_text(
            "# 可由 `compose slides digest` 从源 pptx 再生，体积大，不入库\n"
            "images/\nrenders/\n",
            encoding="utf-8",
        )
    return p


# ---------------------------------------------------------------------------
# 整页合成渲染（需 LibreOffice）
# ---------------------------------------------------------------------------
def _render_deck(src: Path, root: Path, pages: list[int], dpi: int) -> tuple[int, str]:
    """pptx → pdf（LibreOffice headless）→ 逐页 PNG；返回 (渲染数, 说明)。"""
    from . import office_convert, pdf_render
    from .config import settings

    if not settings.find_libreoffice():
        return 0, (
            "未检测到 LibreOffice；安装后（或在 .env 设 DOCWRITING_SOFFICE）重跑 "
            "`compose slides digest <pptx> --out <本目录> --render` 即可补渲染"
        )
    renders = root / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    try:
        pdf = office_convert.convert(src, "pdf", out_dir=renders, timeout=1800)
    except Exception as e:
        return 0, f"转换失败：{type(e).__name__}: {e}"
    try:
        pngs = pdf_render.render_pdf_pages(
            pdf, out_dir=renders, dpi=dpi, pages=pages or None, prefix="slide"
        )
    except Exception as e:
        return 0, f"渲染失败：{type(e).__name__}: {e}"
    finally:
        # 中间 PDF 体积巨大且可再生，渲染完即删（保留 PNG）
        try:
            pdf.unlink(missing_ok=True)
        except Exception:
            pass
    return len(pngs), ""


# ---------------------------------------------------------------------------
# 链接体检
# ---------------------------------------------------------------------------
def verify_links(root: str | Path) -> dict[str, Any]:
    """检查所有 md 里的图片链接：存在 / 缺失 / 待渲染（renders/ 未生成）。"""
    root = Path(root)
    md_files = [root / "index.md", *sorted(root.glob("part_*.md"))]
    ok, broken, pending = 0, [], []
    pat = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for md in md_files:
        if not md.is_file():
            continue
        for m in pat.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1)
            if not (md.parent / target).exists():
                pend = target.startswith(("renders/", "images/previews/"))
                (pending if pend else broken).append(f"{md.name}: {target}")
            else:
                ok += 1
    return {"ok": ok, "broken": broken, "pending_renders": pending}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def digest_pptx(
    src: str | Path,
    out_dir: str | Path,
    *,
    render: bool = False,
    dpi: int = 140,
    gif_frames: int = 3,
    chunk_by: str = "section",
    max_chunk_slides: int = 40,
    chunk_size: int = 50,
    batch_figure: int = 5,
    batch_text: int = 15,
    section_starts: list[int] | None = None,
    force: bool = False,
) -> DigestResult:
    """把 .pptx 消化为图文互证的 Markdown 工作区（Phase A）。

    Args:
        src: 源 .pptx。
        out_dir: 输出根目录（产出 index.md / part_*.md / images/ / sidecar/ /
            images_manifest.json / progress.json，可选 renders/）。
        render: 是否同时生成整页合成渲染 PNG（需 LibreOffice；缺失则记录说明并跳过）。
        dpi: 渲染分辨率（版式与图文关系 120–160 足够）。
        gif_frames: gif 动图抽帧预览的帧数（含首末帧；LLM 不能直读 gif）。
        chunk_by: "section"（按分节打包成块，推荐）或 "fixed"（固定页数）。
        max_chunk_slides: 单个 md 块的最大页数；连续小节合并至该上限，超长的节内部再切。
        chunk_size: chunk_by="fixed" 时每块页数。
        batch_figure / batch_text: 写入 progress.json 的 Phase B 批大小建议。
        section_starts: **人工指定**分节起始页（1-based，不含首页）；给定时跳过自动检测。
            适用于全 deck 共用一个版式、自动检测失效的汇报稿。
        force: 忽略既有 digest，全量重建（会覆盖 part_*.md，**会丢失已填解读**）。

    可重复运行：若 out_dir 已有 progress.json 且未 force，则只补做缺失的渲染，
    不重写骨架（保护已填写的解读）。
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"pptx 不存在：{src}")
    if src.suffix.lower() not in (".pptx", ".pptm"):
        raise ValueError(f"仅支持 .pptx（不支持旧版 .ppt）：{src.name}")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"

    # --- 复用既有 digest：只在需要时补渲染 ---
    if progress_path.is_file() and not force:
        prog = json.loads(progress_path.read_text(encoding="utf-8"))
        res = DigestResult(
            root=root,
            source=Path(prog.get("source", src)),
            n_slides=prog.get("total_slides", 0),
            n_pictures=prog.get("n_pictures", 0),
            n_unique_images=prog.get("n_unique_images", 0),
            n_duplicates=prog.get("n_duplicates", 0),
            n_figure_slides=len(prog.get("figure_slides", [])),
            n_text_slides=len(prog.get("text_slides", [])),
            sections=prog.get("sections", []),
            chunks=[root / c["file"] for c in prog.get("chunks", [])],
            sidecar=root / "sidecar" / "slides.jsonl",
            manifest=root / "images_manifest.json",
            progress=progress_path,
            index_md=root / "index.md",
            rendered=prog.get("render", {}).get("rendered", 0),
            render_note=prog.get("render", {}).get("note", ""),
            reused=True,
        )
        if render:
            # 补页渲染（若尚未渲染）
            if not res.rendered:
                pages = prog.get("figure_slides", [])
                n, note = _render_deck(res.source, root, pages, dpi)
                res.rendered, res.render_note = n, note
            # 矢量图（wmf/emf）预览同样依赖 LibreOffice，幂等补齐
            man_path = root / "images_manifest.json"
            if man_path.is_file():
                manifest = json.loads(man_path.read_text(encoding="utf-8"))
                _, vec_note = _make_vector_previews(root, manifest)
                res.vector_previews = sum(
                    1 for m in manifest.values() if m.get("preview_status") == "ok"
                )
                man_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                if vec_note and not res.render_note:
                    res.render_note = vec_note
            prog["render"] = {
                "rendered": res.rendered,
                "vector_previews": res.vector_previews,
                "note": res.render_note,
                "rel_pattern": "renders/slide-{index:03d}.png",
                "dpi": dpi,
            }
            progress_path.write_text(
                json.dumps(prog, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        return res

    # --- 全量提取 ---
    slides, manifest, sw, sh = _scan(src, root, gif_frames=gif_frames)
    n_pics = sum(len(s.pictures) for s in slides)
    n_uniq = len(manifest)
    res = DigestResult(
        root=root,
        source=src.resolve(),
        n_slides=len(slides),
        n_pictures=n_pics,
        n_unique_images=n_uniq,
        n_duplicates=n_pics - n_uniq,
        n_figure_slides=sum(1 for s in slides if s.pictures),
        n_text_slides=sum(1 for s in slides if not s.pictures),
        slide_w_in=sw,
        slide_h_in=sh,
    )

    sections = _detect_sections(slides, section_starts=section_starts)
    res.sections = sections
    chunks = _group_chunks(
        sections,
        len(slides),
        chunk_by=chunk_by,
        max_chunk_slides=max_chunk_slides,
        chunk_size=chunk_size,
    )

    # 渲染（可选）
    if render:
        pages = [s.index for s in slides if s.pictures]
        res.rendered, res.render_note = _render_deck(src, root, pages, dpi)
        # 矢量图（wmf/emf）预览同样依赖 LibreOffice，一并补齐（manifest 稍后落盘）
        _, vec_note = _make_vector_previews(root, manifest)
        res.vector_previews = sum(
            1 for m in manifest.values() if m.get("preview_status") == "ok"
        )
        if vec_note and not res.render_note:
            res.render_note = vec_note
    else:
        res.render_note = "未请求渲染（加 --render 生成整页合成 PNG）"

    # 全量重建：清掉上一次残留、本次不再产出的 part_*.md（文件名随分节标题变化，
    # 若不清理会让 verify_links 重复计数、目录混入旧骨架）。仅重建路径执行；
    # 复用路径（有 progress.json 且未 force）绝不动 part 文件，以保护已填解读。
    for stale in root.glob("part_*.md"):
        stale.unlink(missing_ok=True)

    chunk_files = [
        _write_chunk(root, n, c, slides) for n, c in enumerate(chunks, 1)
    ]
    res.chunks = chunk_files
    res.sidecar = _write_sidecar(root, slides)
    res.manifest = _write_manifest(root, manifest)
    res.index_md = _write_index(
        root, res, chunks, chunk_files, manifest, batch_figure, batch_text, slides
    )
    res.progress = _write_progress(
        root, res, chunks, chunk_files, slides, batch_figure, batch_text
    )
    _write_gitignore(root)
    return res


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="pptx → 图文互证 Markdown 工作区")
    ap.add_argument("pptx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--gif-frames", type=int, default=3, dest="gif_frames")
    ap.add_argument("--chunk-by", default="section", dest="chunk_by")
    ap.add_argument("--section-at", default=None, dest="section_at",
                    help="逗号分隔的分节起始页（人工定界）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lint", action="store_true")
    a = ap.parse_args()
    if a.lint:
        print(json.dumps(verify_links(a.out), ensure_ascii=False, indent=1))
    else:
        starts = (
            [int(x) for x in a.section_at.split(",") if x.strip()]
            if a.section_at
            else None
        )
        r = digest_pptx(
            a.pptx, a.out, render=a.render, dpi=a.dpi, gif_frames=a.gif_frames,
            chunk_by=a.chunk_by, section_starts=starts, force=a.force,
        )
        print(r.summary())
