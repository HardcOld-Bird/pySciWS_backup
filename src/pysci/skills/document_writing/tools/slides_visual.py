"""slides_visual —— PPTX 逐页渲染与视觉验收（机制性视觉检查闸）。

用户硬要求：**不要让组员在不经视觉检查的情况下就提交 ppt 产物**——机制保证优先于自觉。
本模块是那条机制的**生产端与验收端**：

- 生产端 :func:`render_pptx_pages`：固化 pptx→pdf→逐页 PNG 链（LibreOffice + pymupdf），
  产出 `<deck>_render/page-NNN.png`，供组员 Read 做视觉判读、并作为交付**必附**的渲染图。
- 验收端 :func:`verify_pptx_render`：机械校验**渲染存在**、**页数 = 幻灯片数**、**非空白页**；
  任一不过即判 FAIL——把「组员自觉渲染」升级为「缺渲染即打回」。

设计取向：验收只读**已落盘的渲染图**（不自行渲染），故「缺渲染」是一个可被机械抓住的
真实状态（组员忘了渲染 → 渲染目录空 → FAIL），而非被本命令静默补齐。渲染与验收分成两条
命令（`compose slides render` / `compose slides verify`），前者产附件、后者做闸门。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
