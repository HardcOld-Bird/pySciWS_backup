"""slides_visual —— PPTX 逐页渲染与视觉验收（机制性视觉检查闸）。

用户硬要求：**不要让组员在不经视觉检查的情况下就提交 ppt 产物**——机制保证优先于自觉。
本模块是那条机制的**生产端与验收端**：

- 生产端 :func:`render_pptx_pages`：固化 pptx→pdf→逐页 PNG 链（LibreOffice + pymupdf），
  产出 `<deck>_render/page-NNN.png`，供组员 Read 做视觉判读、并作为交付**必附**的渲染图。
- 验收端 :func:`verify_pptx_render`：机械校验**渲染存在**、**页数 = 幻灯片数**、**非空白页**；
  任一不过即判 FAIL——把「组员自觉渲染」升级为「缺渲染即打回」。
- 几何端 :func:`geometry_report`（`slides geometry`）：**只读、不渲染**地硬判版面几何硬伤——
  外框边被更高 z 序图片**覆盖**、图元**越界**（画布/外框）、**图片互叠**面积。渲染闸只管
  「渲染出来了没 / 是不是空白」，命中不了「渲染出来了但版面是坏的」；这一端补上后者。

设计取向：验收只读**已落盘的渲染图**（不自行渲染），故「缺渲染」是一个可被机械抓住的
真实状态（组员忘了渲染 → 渲染目录空 → FAIL），而非被本命令静默补齐。渲染与验收分成两条
命令（`compose slides render` / `compose slides verify`），前者产附件、后者做闸门。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import office_convert, pdf_render, pptx_io

#: 逐页 PNG 的文件名前缀（与 :func:`pdf_render.render_pdf_pages` 默认一致）。
PAGE_PREFIX = "page"

#: 判定「非空白页」的最小墨迹占比：与**背景（众数灰度）异色**的像素占全图比例低于此值
#: 即视为空白页。取值极小（0.2%）——只抓真正的空页（渲染失败/整页纯色），不误伤
#: 内容稀少但有文字/图元的正常页。
DEFAULT_MIN_INK_RATIO = 0.002

#: 背景与墨迹的灰度差阈值（0–255）：差 > 此值才算「墨迹」。
_INK_GRAY_DELTA = 24


def default_render_dir(pptx: str | Path) -> Path:
    """逐页渲染图的**约定目录**：``<pptx 同目录>/<stem>_render/``。

    验收端按同一约定定位，故 `compose slides render`（生产）与 `compose slides verify`
    （验收）不必显式互传路径即可对齐——这是「逐页渲染 PNG 为交付必附件」的落点。
    """
    p = Path(pptx)
    return p.parent / f"{p.stem}_render"


@dataclass
class RenderResult:
    """一次 pptx → pdf → 逐页 PNG 渲染的结果。"""

    pptx: Path
    pdf: Path
    out_dir: Path
    pngs: list[Path] = field(default_factory=list)
    slide_count: int = 0

    def summary(self) -> str:
        return (
            f"[slides render] {self.pptx.name}：{self.slide_count} 页 → 渲染 "
            f"{len(self.pngs)} 张 PNG 于 {self.out_dir}"
        )


@dataclass
class VerifyReport:
    """一次 pptx 视觉验收的结果（渲染存在 / 页数一致 / 非空白）。"""

    pptx: Path
    render_dir: Path
    slide_count: int = 0
    png_count: int = 0
    blank_pages: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无任何问题即通过。"""
        return not self.problems

    def summary(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [
            f"[slides verify] {head} {self.pptx.name}：幻灯片 {self.slide_count} 页 / "
            f"渲染 {self.png_count} 张（{self.render_dir}）"
        ]
        lines.extend(f"    ✗ {p}" for p in self.problems)
        return "\n".join(lines)


def render_pptx_pages(
    pptx: str | Path,
    *,
    out_dir: str | Path | None = None,
    dpi: int = 140,
) -> RenderResult:
    """把 .pptx 渲染为逐页 PNG（pptx → pdf → page-NNN.png）。

    Args:
        pptx: 源 .pptx。
        out_dir: 输出目录；默认 :func:`default_render_dir`（``<stem>_render/``）。中间 PDF
            也落此目录（``<stem>.pdf``），供回溯与在 PDF 阅读器里放大核对。
        dpi: 渲染分辨率。

    Returns:
        RenderResult：含输出目录、逐页 PNG 路径与幻灯片页数。

    Raises:
        FileNotFoundError: pptx 不存在。
        office_convert.LibreOfficeNotInstalled: 未检测到 LibreOffice（渲染链依赖它）。
    """
    src = Path(pptx)
    if not src.exists():
        raise FileNotFoundError(f"pptx 不存在：{src}")
    slide_count = len(pptx_io.read_pptx(src))
    out = Path(out_dir) if out_dir else default_render_dir(src)
    out.mkdir(parents=True, exist_ok=True)
    pdf = office_convert.convert(src, "pdf", out_dir=out)
    pngs = pdf_render.render_pdf_pages(pdf, out_dir=out, dpi=dpi, prefix=PAGE_PREFIX)
    return RenderResult(
        pptx=src, pdf=pdf, out_dir=out, pngs=pngs, slide_count=slide_count
    )


def page_ink_ratio(png: str | Path) -> float:
    """一页渲染图里「墨迹」（与背景异色）像素占全图的比例。

    背景取灰度直方图的**众数**（最常见灰度），与背景差 > :data:`_INK_GRAY_DELTA` 的像素
    计为墨迹。空白页返回 ~0。
    """
    from PIL import Image

    with Image.open(png) as im:
        hist = im.convert("L").histogram()
    total = sum(hist) or 1
    bg = max(range(256), key=lambda v: hist[v])
    ink = sum(hist[v] for v in range(256) if abs(v - bg) > _INK_GRAY_DELTA)
    return ink / total


def page_is_blank(
    png: str | Path, min_ink_ratio: float = DEFAULT_MIN_INK_RATIO
) -> bool:
    """该页是否**空白**（墨迹占比低于阈值）。"""
    return page_ink_ratio(png) < min_ink_ratio


def verify_pptx_render(
    pptx: str | Path,
    *,
    render_dir: str | Path | None = None,
    min_ink_ratio: float = DEFAULT_MIN_INK_RATIO,
) -> VerifyReport:
    """机械验收一份 pptx 的**逐页渲染**（渲染存在 / 页数一致 / 非空白）。

    只读已落盘的渲染图（不自行渲染）——故「缺渲染」会被如实抓住。判据三条任一不过即
    计入 :attr:`VerifyReport.problems`，:attr:`VerifyReport.ok` 随之 False。

    Args:
        pptx: 源 .pptx。
        render_dir: 渲染图目录；默认 :func:`default_render_dir`。
        min_ink_ratio: 空白页判定阈值（墨迹占比低于它即空白）。

    Returns:
        VerifyReport。
    """
    src = Path(pptx)
    rdir = Path(render_dir) if render_dir else default_render_dir(src)
    problems: list[str] = []
    if not src.exists():
        return VerifyReport(pptx=src, render_dir=rdir, problems=[f"产物不存在：{src}"])

    slide_count = len(pptx_io.read_pptx(src))
    pngs = sorted(rdir.glob("*.png")) if rdir.is_dir() else []

    if not pngs:
        problems.append(
            f"缺渲染：未找到逐页渲染 PNG（{rdir}）——先跑 `compose slides render` 并在交付中附上"
        )
    elif len(pngs) != slide_count:
        problems.append(f"页数不符：渲染 {len(pngs)} 页 ≠ 幻灯片 {slide_count} 页")

    blank = [p.name for p in pngs if page_is_blank(p, min_ink_ratio)]
    if blank:
        problems.append(f"空白页：{'、'.join(blank)}")

    return VerifyReport(
        pptx=src,
        render_dir=rdir,
        slide_count=slide_count,
        png_count=len(pngs),
        blank_pages=blank,
        problems=problems,
    )


# ---------------------------------------------------------------------------
# 几何体检（render-free、只读）：外框边被覆盖 / 图元越界 / 图片互叠
# ---------------------------------------------------------------------------
# 为什么另立一套：渲染闸（verify）只防「忘了渲染」与「渲染成空白/坏页」，对**版面硬伤**
# （文字/图块压住模板外框、图元越出画布或内容框、两张粘贴截图互相叠住）零覆盖——这些是
# 纯几何事实，从 .pptx 的 bbox + z 序即可硬判，不必也不该让审查员自建脚本。故提供
# `slides geometry`：只读、**不写盘**（受审目录只读时也照跑），逐页输出三类数值。
#
# EMU → 英寸。
_EMU_IN = 914400.0
#: 越界判定容差（英寸）：小于它的超出视为对齐误差，不报。
DEFAULT_TOLERANCE_IN = 0.05
#: 图片互叠的默认面积阈值（平方英寸）：小于它的交叠不报（相邻贴合、圆角边不算）。
DEFAULT_OVERLAP_IN2 = 1.0
#: 每条边被覆盖率的采样点数。
_EDGE_SAMPLES = 400
#: 视作「背景图」的面积占比（≥ 此值不参与互叠判定——满版底图与内容图天然相交）。
_BACKGROUND_AREA_FRAC = 0.8


@dataclass
class ShapeBox:
    """一个形状的包围盒（英寸，左上/右下）+ 页内 z 序（枚举序，越大越靠上）。"""

    z: int
    name: str
    kind: str  # "pic" | "text" | "shape"
    x0: float
    y0: float
    x1: float
    y1: float
    rect: bool = False  # 是否为矩形自选图形（外框自动识别用）

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass
class EdgeCoverage:
    """外框某条边被**更高 z 序图片**覆盖的比例与区间（区间按 400 点采样还原）。"""

    edge: str
    frac: float
    spans: list[str] = field(default_factory=list)


@dataclass
class SlideGeometry:
    """一页的几何体检结果。"""

    index: int
    layout: str = ""
    boxes: list[ShapeBox] = field(default_factory=list)
    frame: ShapeBox | None = None
    frame_source: str = ""  # "explicit" | "auto" | ""
    edges: list[EdgeCoverage] = field(default_factory=list)
    overlaps: list[str] = field(default_factory=list)
    off_canvas: list[str] = field(default_factory=list)
    off_frame: list[str] = field(default_factory=list)

    @property
    def violations(self) -> int:
        """本页违规项计数（有覆盖的边 + 互叠 + 越界）。"""
        return (
            len([e for e in self.edges if e.frac > 0])
            + len(self.overlaps)
            + len(self.off_canvas)
            + len(self.off_frame)
        )


@dataclass
class GeometryReport:
    """一份 pptx 的几何体检报告（只读、render-free）。"""

    pptx: Path
    canvas_in: tuple[float, float]
    slides: list[SlideGeometry] = field(default_factory=list)

    @property
    def violations(self) -> int:
        return sum(s.violations for s in self.slides)

    @property
    def ok(self) -> bool:
        return self.violations == 0

    def summary(self) -> str:
        w, h = self.canvas_in
        lines = [
            f"[slides geometry] {self.pptx.name}：画布 {w:.2f}x{h:.2f} in，{len(self.slides)} 页"
        ]
        for s in self.slides:
            head = f"  第 {s.index} 页"
            if s.frame is not None:
                f = s.frame
                head += (
                    f"：外框 {f.name} ({f.x0:.2f},{f.y0:.2f})-({f.x1:.2f},{f.y1:.2f})"
                    f"（{s.frame_source}）"
                )
            else:
                head += "：未识别外框（用 --frame 指定形状名可启用边覆盖判定）"
            lines.append(head)
            for e in s.edges:
                if e.frac > 0:
                    lines.append(
                        f"      {e.edge}: 被覆盖 {e.frac * 100:.1f}%  ->  {e.spans}"
                    )
            for ov in s.overlaps:
                lines.append(f"      图片互叠: {ov}")
            for oc in s.off_canvas:
                lines.append(f"      越出画布: {oc}")
            for of in s.off_frame:
                lines.append(f"      越出外框: {of}")
        lines.append(
            f"[slides geometry] 合计违规 {self.violations} 项"
            + ("（PASS）" if self.ok else "（需人工判读/返工）")
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        w, h = self.canvas_in
        return {
            "pptx": str(self.pptx),
            "canvas_in": [w, h],
            "violations": self.violations,
            "slides": [
                {
                    "index": s.index,
                    "layout": s.layout,
                    "frame": (
                        {
                            "name": s.frame.name,
                            "box": [round(v, 3) for v in s.frame.box],
                            "source": s.frame_source,
                        }
                        if s.frame is not None
                        else None
                    ),
                    "edge_coverage": [
                        {"edge": e.edge, "frac": round(e.frac, 4), "spans": e.spans}
                        for e in s.edges
                        if e.frac > 0
                    ],
                    "picture_overlaps": s.overlaps,
                    "off_canvas": s.off_canvas,
                    "off_frame": s.off_frame,
                }
                for s in self.slides
            ],
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def _shape_kind(shape: Any) -> str:
    try:
        if int(shape.shape_type) == 13:  # MSO_SHAPE_TYPE.PICTURE
            return "pic"
    except (TypeError, ValueError):
        pass
    if getattr(shape, "has_text_frame", False):
        return "text"
    return "shape"


def _is_rectangle(shape: Any) -> bool:
    """是否为矩形类自选图形（外框自动识别的候选；ROUNDED_RECTANGLE 亦含 RECTANGLE 字样）。"""
    try:
        if int(shape.shape_type) != 1:  # MSO_SHAPE_TYPE.AUTO_SHAPE
            return False
        return "RECTANGLE" in str(shape.auto_shape_type)
    except (AttributeError, TypeError, ValueError):
        return False


def _shape_box(shape: Any, z: int) -> ShapeBox | None:
    try:
        left, top, width, height = shape.left, shape.top, shape.width, shape.height
    except Exception:  # noqa: BLE001 - 个别形状无几何属性，跳过
        return None
    if None in (left, top, width, height):
        return None
    x0, y0 = left / _EMU_IN, top / _EMU_IN
    return ShapeBox(
        z=z,
        name=getattr(shape, "name", "") or f"#{z}",
        kind=_shape_kind(shape),
        x0=x0,
        y0=y0,
        x1=x0 + width / _EMU_IN,
        y1=y0 + height / _EMU_IN,
        rect=_is_rectangle(shape),
    )


def _collect(slide: Any) -> list[ShapeBox]:
    """页内所有**有几何**的形状（z 序取枚举序，含被跳过的无几何形状，保持相对次序）。"""
    boxes: list[ShapeBox] = []
    for z, shape in enumerate(slide.shapes, 1):
        b = _shape_box(shape, z)
        if b is not None:
            boxes.append(b)
    return boxes


def _detect_frame(boxes: list[ShapeBox], cw: float, ch: float) -> ShapeBox | None:
    """自动挑外框：画布面积的 10%–95% 之间、面积最大的矩形（满版底图被排除）。

    启发式，非稳准——命名外框（`--frame`）为首选；识别不出就只报越界/互叠，不臆造。
    """
    canvas_area = cw * ch
    cands = [
        b
        for b in boxes
        if b.rect and 0.10 * canvas_area <= b.area <= 0.95 * canvas_area
    ]
    return max(cands, key=lambda b: b.area) if cands else None


def _edge_coverage(frame: ShapeBox, blockers: list[ShapeBox]) -> list[EdgeCoverage]:
    """外框四条边按点采样，统计被 `blockers`（更高 z 序图片）覆盖的比例与区间。"""
    n = _EDGE_SAMPLES
    edges = {
        "上边": [(frame.x0 + frame.width * k / n, frame.y0) for k in range(n + 1)],
        "下边": [(frame.x0 + frame.width * k / n, frame.y1) for k in range(n + 1)],
        "左边": [(frame.x0, frame.y0 + frame.height * k / n) for k in range(n + 1)],
        "右边": [(frame.x1, frame.y0 + frame.height * k / n) for k in range(n + 1)],
    }
    out: list[EdgeCoverage] = []
    for ename, pts in edges.items():
        hit = [
            any(b.x0 <= px <= b.x1 and b.y0 <= py <= b.y1 for b in blockers)
            for px, py in pts
        ]
        frac = sum(hit) / len(hit)
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for k, h in enumerate(hit):
            if h and start is None:
                start = k
            elif not h and start is not None:
                runs.append((start, k - 1))
                start = None
        if start is not None:
            runs.append((start, len(hit) - 1))
        spans = []
        for a, b in runs:
            if ename in ("上边", "下边"):
                spans.append(f"x {pts[a][0]:.2f}–{pts[b][0]:.2f}")
            else:
                spans.append(f"y {pts[a][1]:.2f}–{pts[b][1]:.2f}")
        out.append(EdgeCoverage(edge=ename, frac=frac, spans=spans))
    return out


def _picture_overlaps(
    pics: list[ShapeBox], cw: float, ch: float, threshold: float
) -> list[str]:
    """图片两两互叠（交叠面积 > ``threshold``）；满版底图（≥80% 画布）不参与。"""
    bg_area = _BACKGROUND_AREA_FRAC * cw * ch
    res: list[str] = []
    for i in range(len(pics)):
        for j in range(i + 1, len(pics)):
            a, b = pics[i], pics[j]
            if a.area >= bg_area or b.area >= bg_area:
                continue
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0 and oy > 0 and ox * oy > threshold:
                res.append(
                    f"{a.name} ∩ {b.name} = {ox:.2f}x{oy:.2f} in ({ox * oy:.2f} in²)"
                    f"  区间 x {max(a.x0, b.x0):.2f}–{min(a.x1, b.x1):.2f}"
                    f" y {max(a.y0, b.y0):.2f}–{min(a.y1, b.y1):.2f}"
                )
    return res


def _off_bounds(
    boxes: list[ShapeBox],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tol: float,
    what: str,
) -> list[str]:
    """盒越出参照框（画布或外框）超过 ``tol`` 的清单。"""
    res: list[str] = []
    for b in boxes:
        sides: list[str] = []
        if b.x0 < x0 - tol:
            sides.append(f"左 {x0 - b.x0:.2f}in")
        if b.x1 > x1 + tol:
            sides.append(f"右 {b.x1 - x1:.2f}in")
        if b.y0 < y0 - tol:
            sides.append(f"上 {y0 - b.y0:.2f}in")
        if b.y1 > y1 + tol:
            sides.append(f"下 {b.y1 - y1:.2f}in")
        if sides:
            res.append(
                f"{b.name}({b.x0:.2f},{b.y0:.2f},{b.x1:.2f},{b.y1:.2f})"
                f" 越出{what}：{'、'.join(sides)}"
            )
    return res


def geometry_report(
    pptx: str | Path,
    *,
    frame: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE_IN,
    overlap_in2: float = DEFAULT_OVERLAP_IN2,
) -> GeometryReport:
    """对 .pptx 做**只读几何体检**：外框边被覆盖率 / 图元越界（画布与外框）/ 图片互叠面积。

    纯 bbox + z 序硬判，不渲染、不写盘——受审目录只读时也照跑（渲染闸 :func:`verify_pptx_render`
    才需要可写目录产 PNG）。

    Args:
        pptx: 源 .pptx。
        frame: 外框形状名（``--frame``）。缺省自动挑「面积最大的矩形」（10%–95% 画布面积）；
            识别不出则只报越界与互叠，不臆造外框。
        tolerance: 越界容差（英寸）。
        overlap_in2: 图片互叠面积阈值（平方英寸）。

    Returns:
        GeometryReport。

    Raises:
        FileNotFoundError: pptx 不存在。
    """
    from pptx import Presentation

    src = Path(pptx)
    if not src.exists():
        raise FileNotFoundError(f"pptx 不存在：{src}")
    prs = Presentation(str(src))
    cw, ch = prs.slide_width / _EMU_IN, prs.slide_height / _EMU_IN

    slides: list[SlideGeometry] = []
    for i, slide in enumerate(prs.slides, 1):
        boxes = _collect(slide)
        try:
            layout = slide.slide_layout.name or ""
        except Exception:  # noqa: BLE001
            layout = ""
        fr: ShapeBox | None = None
        source = ""
        if frame:
            fr = next((b for b in boxes if b.name == frame), None)
            source = "explicit" if fr is not None else ""
        else:
            fr = _detect_frame(boxes, cw, ch)
            source = "auto" if fr is not None else ""

        pics = [b for b in boxes if b.kind == "pic"]
        sg = SlideGeometry(
            index=i,
            layout=layout,
            boxes=boxes,
            frame=fr,
            frame_source=source,
            off_canvas=_off_bounds(boxes, 0.0, 0.0, cw, ch, tolerance, "画布"),
            overlaps=_picture_overlaps(pics, cw, ch, overlap_in2),
        )
        if fr is not None:
            sg.edges = _edge_coverage(fr, [b for b in pics if b.z > fr.z])
            sg.off_frame = _off_bounds(
                pics, fr.x0, fr.y0, fr.x1, fr.y1, tolerance, "外框"
            )
        slides.append(sg)

    return GeometryReport(pptx=src, canvas_in=(cw, ch), slides=slides)
