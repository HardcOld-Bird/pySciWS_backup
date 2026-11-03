"""中英对照生产线 CLI 管线（``pysci-research bilingual``）——环节 1/3。

设计总原则：**除翻译本身外一切琐事程序化**。翻译组员只逐句把英文替换为中文；
排版、建文件、命名、图片回补、移动全部由本模块机械完成。

**英文与原文逐字一致由构造保证**：``bilingual.md`` 的英文侧从 manifest（源文档切片）
程序化重嵌入，完全不取 work.md 里的英文行——即使翻译组员误改英文行，交付物英文侧
仍逐字一致。

数据布局（工作区 ``data/skills/literature_research/bilingual/<stem>/``）::

    work.md        翻译工作文档（每行 ``ID<TAB>英文原文``）
    manifest.json  单元切片 + 透传块 + 文档元信息
    images/        回补的图（md 引用名与落盘文件名闭合）
    bilingual.md   submit 产物（PDF 渲染归环节 2）

切分契约：**待译**单元 = 标题（H）/ 正文句子（S）/ 图注（C）/ 表注（T）；
**透传**块 = 公式 ``$$`` 块 / 表格 HTML / 图片引用行 / 参考文献条目。句切分先合并段内
软换行再用 pysbd 切句，句末符只认 ``. ? !``，省略号（``...``/``…``）不作句末。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pysbd

from pysci.paths import DATA_ROOT, assert_within_data

from . import cache_manager
from .config import settings

#: 用户 inbox 副本根（submit 末步复制 bilingual.md 到这里，只增不删）。
INBOX_ROOT: Path = DATA_ROOT / "用户的inbox" / "bilingual"

#: 翻译规则（work.md 头部注释行与翻译指引共用一份文本，§3 契约原文）。
RULES_TEXT = (
    "把每行 TAB 后的英文整行替换为中文翻译；保留行首 ID；不增删行；"
    "行内 $公式$、引文标注照抄；术语先读源文档统一；完成后跑 `bilingual submit`"
)

#: 单元类型字母：标题 / 句子 / 图注 / 表注。
TYPE_LETTERS: tuple[str, ...] = ("H", "S", "C", "T")

#: 句切分保护哨兵：省略号与缩写点先换成它，切完再还原（pysbd 别碰这些点）。
_SENTINEL = ""

#: 参考文献节标题（进入后段落按参考文献条目透传）。
_REFERENCES_HEADING = re.compile(r"(?i)^\s*#*\s*(references|bibliography|参考文献)")

#: 行首 ID 前缀（对账与钩子共用）：类型字母 + 4 位序号 + TAB。
ID_PREFIX_RE = re.compile(r"^([A-Z]\d{4})(?:\t|$)")


@dataclass
class Unit:
    """一个待译单元（work.md 的一行）。

    Attributes:
        id: 全文档唯一 ID，形如 ``H0001``/``S0001``/``C0001``/``T0001``。
        type: 类型字母（:data:`TYPE_LETTERS` 之一）。
        en: 原文逐字串（源文档切片，英文侧重嵌入的唯一来源）。
        order: 全文档顺序（与透传块共用计数，submit 按它还原插入位置）。
        para: 段落序号（同段落的句子共享；图注与其图片块同 para，段落边界渲染用）。
        section: 所属章节标题（英文原文；无标题时为空串）。
        level: 标题层级（仅 H 单元有效，渲染 ``#`` 前缀用；其余类型为 0）。
        role: 语义角色标注（``abstract``=摘要段；prepare 摘要识别所置，渲染据此插
            「## 摘要/## 正文」中文小标题（无英文行，2026-11-03 用户裁决）；
            识别错误可手改 manifest 的该字段）。
    """

    id: str
    type: str
    en: str
    order: int
    para: int
    section: str = ""
    level: int = 0
    role: str = ""


@dataclass
class PassBlock:
    """一个透传块（不译，manifest 原样带回）。

    Attributes:
        kind: ``formula``/``table``/``image``/``reference``。
        content: 原样内容（图片行为整行，公式/表格为多行原文）。
        order: 全文档顺序（与单元共用计数）。
        para: 段落序号。
        section: 所属章节标题。
    """

    kind: str
    content: str
    order: int
    para: int
    section: str = ""


@dataclass
class ParsedDoc:
    """解析结果：单元与透传块按文档序交错，外加图片引用名清单。"""

    units: list[Unit] = field(default_factory=list)
    blocks: list[PassBlock] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 句切分（先合并段内软换行 → pysbd；句末符只认 . ? !，省略号不作句末）
# ---------------------------------------------------------------------------
#: 学术缩写（句中点，不是句末）：先换成哨兵防 pysbd 误切。pysbd 自带缩写规则，但
#: ``Eq. (2)`` 这类后接括号/数字的仍会误切（实测），故再加一层机械保护。
_ABBREVIATIONS: tuple[str, ...] = (
    "e.g.",
    "i.e.",
    "et al.",
    "al.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "eq.",
    "vs.",
    "cf.",
    "Ref.",
    "Refs.",
    "Sec.",
    "Ch.",
    "No.",
    "Vol.",
    "pp.",
    "ca.",
    "approx.",
    "Dr.",
    "Prof.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "St.",
    "Jr.",
    "Sr.",
)

#: 省略号（``...`` 或 ``…``）不作句末——学术正文中几乎总是句中省略。
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")


def _protect(text: str) -> str:
    """把缩写点与省略号换成哨兵，避免 pysbd 在这些位置断句。"""
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", _SENTINEL))
    return _ELLIPSIS_RE.sub(lambda m: m.group(0).replace(".", _SENTINEL), text)


def _restore(text: str) -> str:
    """还原哨兵为原文点号/省略号。"""
    return text.replace(_SENTINEL, ".")


def split_sentences(paragraph: str) -> list[str]:
    """把段落文本切成句子（合并段内软换行后 pysbd 切句）。

    句末符只认 ``. ? !``——pysbd 切出的片段若不以此结尾（哨兵还原后判断），
    与后一句合并。返回的句子均为原文逐字串（首尾去空白）。

    Args:
        paragraph: 单个段落的原文（可含软换行）。

    Returns:
        按出现顺序的句子列表；空段落返回空表。
    """
    merged = re.sub(r"\s*\n\s*", " ", paragraph).strip()
    if not merged:
        return []
    seg = pysbd.Segmenter(language="en", clean=False)
    raw = seg.segment(_protect(merged))
    out: list[str] = []
    # cursor = merged 中已消费到的位置：碎片间**分隔按原文取**，不硬插空格——pysbd 会在
    # 「数字紧接单字母面板号」的 ``(`` 前误断句（``Fig. 3``+``(a). ``），无条件插空格会回并出
    # ``Fig. 3 (a)``，与源文不再逐字相等，``verify_slices`` 据此报「切片自检失败」
    # （backlog 20261029-121943-lit）。
    cursor = 0
    for part in raw:
        s = _restore(str(part)).strip()
        if not s:
            continue
        start = merged.find(s, cursor)
        gap = merged[cursor:start] if start >= 0 else " "
        if out and not out[-1].endswith((".", "?", "!")):
            out[-1] = out[-1] + gap + s
        else:
            out.append(s)
        if start >= 0:
            cursor = start + len(s)
    return out


# ---------------------------------------------------------------------------
# markdown 结构解析
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")
_CAPTION_FIG_RE = re.compile(r"^\s*(?:FIG(?:URE)?\.?|Fig\.?|图)\s*[\dA-Z]", re.I)
_CAPTION_TAB_RE = re.compile(r"^\s*(?:TABLE|Tab\.?|表)\s*[\dA-Z]", re.I)


def _classify_paragraph(text: str) -> str:
    """段落首句判图注（C）/表注（T）/正文（S）。

    以图注/表注惯例前缀（``Fig. 1``/``Table 2``/``图 3``）为准；无前缀一律按正文
    句子（S）切分——宁可把图注译成 S，不把正文误标成 C/T。
    """
    head = text.lstrip().lstrip("*").lstrip()
    if _CAPTION_FIG_RE.match(head):
        return "C"
    if _CAPTION_TAB_RE.match(head):
        return "T"
    return "S"


def parse_markdown(md_text: str) -> ParsedDoc:
    """解析 minerU 风格 markdown 为单元 + 透传块序列。

    结构识别：标题（``#``..``######``）、段落（软换行合并）、图片引用行（透传）、
    ``$$...$$`` 公式块（透传，含块内空行）、HTML ``<table>``（透传，可多行）、
    参考文献节（标题命中后段落按条目透传）。

    Args:
        md_text: 源 minerU markdown 全文。

    Returns:
        :class:`ParsedDoc`——单元（H/S/C/T）与透传块按文档序、order 共用计数。
    """
    doc = ParsedDoc()
    order = 0
    para = -1
    section = ""
    in_references = False
    prev_was_image = False
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        m_head = _HEADING_RE.match(stripped)
        if m_head:
            para += 1
            title = m_head.group(2)
            if _REFERENCES_HEADING.match(stripped):
                in_references = True
            else:
                section = title
            doc.units.append(
                Unit(
                    id="",  # 段尾统一编号
                    type="H",
                    en=title,
                    order=order,
                    para=para,
                    section=title,
                    level=len(m_head.group(1)),
                )
            )
            order += 1
            prev_was_image = False
            i += 1
            continue
        if _IMAGE_RE.match(stripped):
            para += 1
            name = stripped[stripped.index("(") + 1 : stripped.rindex(")")]
            doc.images.append(name)
            doc.blocks.append(PassBlock("image", stripped, order, para, section))
            order += 1
            prev_was_image = True
            i += 1
            continue
        if stripped.startswith("$$"):
            # 公式块：$$ 起止（含单行 $$...$$）；块内空行属公式原文。
            buf = [line]
            if stripped.count("$$") < 2:
                i += 1
                while i < len(lines):
                    buf.append(lines[i])
                    if "$$" in lines[i]:
                        i += 1
                        break
                    i += 1
            else:
                i += 1
            para += 1
            doc.blocks.append(
                PassBlock("formula", "\n".join(buf), order, para, section)
            )
            order += 1
            prev_was_image = False
            i += 1
            continue
        if stripped.startswith("<table"):
            # 表格 HTML：到 </table>（可单行可多行）整块透传。
            buf = [line]
            i += 1
            while i < len(lines) and "</table>" not in buf[-1]:
                buf.append(lines[i])
                i += 1
            para += 1
            doc.blocks.append(PassBlock("table", "\n".join(buf), order, para, section))
            order += 1
            prev_was_image = False
            continue
        # 段落：软换行合并到空行/结构行为止。
        buf = [stripped]
        i += 1
        while i < len(lines):
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
        text = re.sub(r"\s*\n\s*", " ", "\n".join(buf)).strip()
        if in_references:
            para += 1
            doc.blocks.append(PassBlock("reference", text, order, para, section))
            order += 1
            prev_was_image = False
            continue
        kind = _classify_paragraph(text)
        # 图注与其图片同 para（渲染要求「图片原位，紧跟其图注句对」，句对不隔 ---）。
        if not (prev_was_image and kind == "C"):
            para += 1
        prev_was_image = False
        for sent in split_sentences(text):
            doc.units.append(
                Unit(id="", type=kind, en=sent, order=order, para=para, section=section)
            )
            order += 1
    _assign_ids(doc)
    return doc


def _assign_ids(doc: ParsedDoc) -> None:
    """按类型字母 + 出现顺序补 ID（H0001/S0001/C0001/T0001，全文档唯一）。"""
    counters: dict[str, int] = {}
    for u in doc.units:
        counters[u.type] = counters.get(u.type, 0) + 1
        u.id = f"{u.type}{counters[u.type]:04d}"


# ---------------------------------------------------------------------------
# 切片自检 + work.md / manifest 生成
# ---------------------------------------------------------------------------
def verify_slices(md_text: str, doc: ParsedDoc) -> None:
    """自检：各单元/透传块按 order 拼接回源文档须无损，防切片错位。

    逐块核对：每块原文（空白归一后）必须能由该块的 en 切片按序拼回（单元用句间
    单空格连接；透传块整块即自身）。任一错位直接抛 :class:`ValueError`，prepare 报错。

    Args:
        md_text: 源 markdown 全文。
        doc: :func:`parse_markdown` 的解析结果。

    Raises:
        ValueError: 切片与源块对不上（切片错位）。
    """
    items: list[Unit | PassBlock] = sorted(
        [*doc.units, *doc.blocks], key=lambda x: x.order
    )
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    # 顺序拼接与源文归一后逐块存在性核对（防 order 乱序/串块/切片错位）。
    whole = norm(md_text)
    pos = 0
    for x in items:
        piece = norm(x.content if isinstance(x, PassBlock) else x.en)
        found = whole.find(piece, pos)
        if found < 0:
            label = x.id if isinstance(x, Unit) else x.kind
            raise ValueError(f"切片自检失败：{label} 的原文在源文档中按序找不到")
        pos = found + len(piece)
    if pos <= 0:
        raise ValueError("切片自检失败：解析结果为空")


def render_work_md(doc: ParsedDoc, translations: dict[str, str] | None = None) -> str:
    """生成 work.md 内容：头部 ``#`` 注释规则 + 每单元 ``ID<TAB>英文原文``。

    段落之间以空行分隔（仅便读；对账按 ID 不按行号）。摘要段（role=abstract）前后
    以 ``#`` 注释行标注（对账/钩子都跳过注释行）；``translations`` 提供时行尾用译文
    （``--reuse`` 回填通道）。
    """
    lines = [
        f"# 翻译规则：{RULES_TEXT}",
        "# 行首 ID 与行数不许改动（pod-guard 钩子硬校验）。",
        "",
    ]
    prev_para: int | None = None
    abs_seen = False
    for u in sorted(doc.units, key=lambda x: x.order):
        if prev_para is not None and u.para != prev_para:
            lines.append("")
        if u.role == "abstract" and not abs_seen:
            lines.append(
                "# ↓摘要段（role=abstract；识别有误改 manifest.json 的 role 字段后重新 submit）"
            )
            abs_seen = True
        if u.role != "abstract" and abs_seen:
            lines.append("# ↑摘要段结束（以下为正文）")
            abs_seen = False
        tail = (translations or {}).get(u.id, u.en)
        lines.append(f"{u.id}\t{tail}")
        prev_para = u.para
    return "\n".join(lines) + "\n"


def build_manifest(doc: ParsedDoc, source_md: Path, no_images: bool) -> dict[str, Any]:
    """构造 manifest 字典：单元切片 + 透传块 + 文档元信息。

    Args:
        doc: 解析结果。
        source_md: 源 markdown 路径（元信息记录）。
        no_images: 本次是否跳过图片回补（submit 据此拒绝出稿）。

    Returns:
        可直接 ``json.dumps`` 的 manifest 字典。
    """
    return {
        "version": 1,
        "source_md": str(source_md),
        "no_images": bool(no_images),
        "images": list(doc.images),
        "units": [asdict(u) for u in sorted(doc.units, key=lambda x: x.order)],
        "blocks": [asdict(b) for b in sorted(doc.blocks, key=lambda x: x.order)],
    }


# ---------------------------------------------------------------------------
# 图片回补（MinerU 云端 → PyMuPDF fallback）
# ---------------------------------------------------------------------------
class ImageBackfillError(RuntimeError):
    """图片回补失败（缺图且无法配对）——prepare 默认据此报错。"""


def _mineru_images(pdf_path: Path) -> dict[str, bytes]:
    """MinerU 云端重抽该 PDF，返回 ``{图名: 字节}``（SDK 的 ExtractResult.images）。"""
    from mineru import MinerU

    client = MinerU(settings.mineru_token or None)
    try:
        result = client.extract(
            str(pdf_path), model="vlm", ocr=True, formula=True, table=True
        )
    finally:
        client.close()
    state = getattr(result, "state", None)
    if state is not None and state != "done":
        raise ImageBackfillError(f"MinerU 解析未完成（state={state}）")
    return {img.name: img.data for img in (getattr(result, "images", None) or [])}


def _pymupdf_images(pdf_path: Path) -> list[bytes]:
    """PyMuPDF 按页序抽嵌入图（fallback：历史缓存图片名对不上旧引用时顺序配对）。"""
    import fitz  # type: ignore

    out: list[bytes] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            for info in page.get_images(full=True):
                xref = info[0]
                base = doc.extract_image(xref)
                out.append(base["image"])
    return out


def backfill_images(
    workdir: Path, refs: list[str], source_pdf: Path | None, *, no_images: bool
) -> None:
    """把 md 引用的图回补到 ``workdir/images/``，核对引用名与落盘文件名闭合。

    先 MinerU 云端重抽按名落盘；引用名对不上（历史缓存重抽）时 fallback 用
    PyMuPDF 从源 PDF 按页序抽图顺序配对并告警。缺图抛
    :class:`ImageBackfillError`；``no_images=True`` 直接跳过（submit 会拒绝出稿）。

    Args:
        workdir: 工作区目录（``bilingual/<stem>/``）。
        refs: md 内图片引用名清单（如 ``images/xxx.jpg``）。
        source_pdf: 源 PDF 路径；``None`` 且需要回补时报错。
        no_images: 跳过回补（测试/调试用）。

    Raises:
        ImageBackfillError: 缺图且无法配对，或缺源 PDF。
    """
    if no_images or not refs:
        return
    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    want = {r.rsplit("/", 1)[-1] for r in refs}
    if source_pdf is None:
        cands = _pdf_candidates(workdir.name)
        listing = (
            "\n".join(f"  - {c}" for c in cands)
            or "  （cache/pdfs 下无按 slug 命中的候选）"
        )
        raise ImageBackfillError(
            "缺源 PDF，无法回补图片（先走既有 pdf_extract/read 抽取）"
            "——候选如下，请用 --pdf 指定：\n"
            f"{listing}\n"
            f"  用法：pysci-research bilingual prepare {workdir.name} --pdf <PDF 路径>"
        )
    saved = _mineru_images(source_pdf)
    for name, data in saved.items():
        (img_dir / name).write_bytes(data)
    missing = sorted(want - set(saved))
    if missing:
        # fallback：按页序顺序配对（extract_figs.py 同款先例），配对结果告警。
        page_imgs = _pymupdf_images(source_pdf)
        if len(page_imgs) < len(missing):
            raise ImageBackfillError(
                f"缺图 {missing}：MinerU 与 PyMuPDF 抽图数（{len(page_imgs)}）不足配对"
            )
        for k, name in enumerate(missing):
            (img_dir / name).write_bytes(page_imgs[k])
        print(
            f"[bilingual] 告警：MinerU 图名与旧引用不闭合，已用 PyMuPDF 按页序顺序配对 {missing}",
            flush=True,
        )
    still = sorted(want - {p.name for p in img_dir.iterdir() if p.is_file()})
    if still:
        raise ImageBackfillError(f"缺图 {still}：回补后引用名仍不闭合")


# ---------------------------------------------------------------------------
# 工作区定位
# ---------------------------------------------------------------------------
def workspace_root() -> Path:
    """工作区根 ``data/skills/literature_research/bilingual/``。"""
    return settings.module_dir / "bilingual"


def workdir_for(stem: str) -> Path:
    """某篇的工作区目录（写入前过 assert_within_data）。"""
    return assert_within_data(workspace_root() / stem, what="bilingual 工作区")


def resolve_source(target: str) -> tuple[Path, str]:
    """把 prepare 目标解析为 ``(源 md 路径, stem)``。

    目标可以是源 minerU markdown 的路径，或 stem（在
    ``data/skills/literature_research/cache/extracted/`` 解析）。

    Args:
        target: 源 md 路径或 stem。

    Returns:
        ``(source_md, stem)``。

    Raises:
        FileNotFoundError: 源 md 不存在（提示先走既有 pdf_extract/read 抽取）。
    """
    p = Path(target)
    if p.is_file():
        return p.resolve(), p.stem
    cand = settings.cache_extracted / f"{target}.md"
    if cand.is_file():
        return cand.resolve(), cand.stem
    hits = sorted(settings.cache_extracted.glob(f"{target}*.md"))
    if len(hits) == 1:
        return hits[0].resolve(), hits[0].stem
    raise FileNotFoundError(
        f"源 md 不存在：{target!r}——先走既有 pdf_extract/read 抽取，本环节不重复实现 PDF→MD"
    )


#: 源 md 头部（标题/DOI 区）的稳定标识：DOI 与 arXiv id（read/browser_fetch 面产物）。
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>\)\]]+")
_ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

#: 元数据行形态（出版信息/DOI/官网链接——摘要识别的前置带判据）。
_META_RE = re.compile(
    r"^(Received|Accepted|Published|Check for updates|Article|ARTICLE|DOI[:.\s]|https?://)",
    re.I,
)


def _is_author_line(text: str) -> bool:
    """作者行判据：≥2 个上标单位标注（``$^{1,6}$`` 形态的作者-机构角标）。"""
    return text.count("$^{") >= 2


def mark_abstract(doc: ParsedDoc) -> int:
    """识别摘要段并置 ``role="abstract"``（返回摘要句数；0=未识别）。

    判据（2026-10-23 用户裁决；官网 Abstract 独立节）：标题（首个 H）之后**先出现
    元数据/作者行**，其后首个段落单元（同 para 全组）即摘要——minerU 丢弃 Abstract/
    Introduction 标题后摘要常是「元数据带之后的第一段」。无元数据带（如纯正文 md）
    不标注；识别错误可手改 manifest 各单元的 ``role`` 字段人工修正。
    """
    units = sorted(doc.units, key=lambda x: x.order)
    title_seen = False
    meta_seen = False
    for u in units:
        if u.type == "H":
            if title_seen:
                break  # 越过第二个标题（首节）即摘要区已过
            title_seen = True
            continue
        if not title_seen:
            continue
        if _META_RE.match(u.en) or _is_author_line(u.en):
            meta_seen = True
            continue
        if not meta_seen:
            return 0  # 无元数据带：不标注（正文 md 形态）
        target_para = u.para
        n = 0
        for v in units:
            if v.para == target_para and v.type in TYPE_LETTERS:
                v.role = "abstract"
                n += 1
        return n
    return 0


#: 头部扫描窗口（参考文献的 DOI 在文末，不进窗口，防误配到别的论文 PDF）。
_MD_HEAD_CHARS = 4096


def extract_pdf_identifiers(md_text: str) -> list[str]:
    """从源 md 头部提取可用于反解源 PDF 的稳定标识（DOI / arXiv id）。

    Args:
        md_text: 源 minerU markdown 全文（只扫头部 :data:`_MD_HEAD_CHARS` 字符）。

    Returns:
        按出现顺序去重的标识列表（DOI 在前、arXiv id 在后）。
    """
    head = md_text[:_MD_HEAD_CHARS]
    out: list[str] = []
    for m in _DOI_RE.finditer(head):
        ident = m.group(0).rstrip(".,;:)]")
        if ident not in out:
            out.append(ident)
    for m in _ARXIV_RE.finditer(head):
        ident = m.group(1)
        if ident not in out:
            out.append(ident)
    return out


def _slug_of(stem: str) -> str:
    """md 的标题 slug（``<slug>_fulltext`` 去后缀；无后缀即 stem）。"""
    return re.sub(r"_fulltext$", "", stem)


def _find_source_pdf(stem: str, *, source_md: Path | None = None) -> Path | None:
    """按 stem（+ 源 md 头部 DOI/arXiv id）反解源 PDF。

    先按 pdf_extract 缓存命名（``<pdfstem>_<hash12>.<backend>`` → ``<pdfstem>.pdf``）
    精确/前缀匹配；再经 :func:`cache_manager.find_cached_pdf` 用源 md 头部提取的
    DOI / arXiv id（及标题 slug）在 ``cache/pdfs`` 与 ``cache/html_fulltext`` 归一解析
    ——read/browser_fetch 面产出的 ``<slug>_fulltext.md`` 对应 PDF 以 DOI/arXiv id 命名
    （20261023-010824-devops）。

    Args:
        stem: 源 md 的 stem。
        source_md: 源 md 路径（用于提取 DOI/arXiv id；None 时跳过元数据反解）。

    Returns:
        反解到的 PDF 路径；解析不到 → ``None``。
    """
    m = re.match(r"^(.*)_[0-9a-f]{12}\.[^.]+$", stem)
    pdf_stem = m.group(1) if m else stem
    exact = settings.cache_pdfs / f"{pdf_stem}.pdf"
    if exact.is_file():
        return exact
    for cand in sorted(settings.cache_pdfs.rglob(f"{pdf_stem}*.pdf")):
        return cand
    slug = _slug_of(stem)
    head = ""
    if source_md is not None and source_md.is_file():
        head = source_md.read_text(encoding="utf-8")[:_MD_HEAD_CHARS]
    for ident in extract_pdf_identifiers(head):
        hit = cache_manager.find_cached_pdf(ident, slug=slug)
        if hit is not None:
            return hit
    return cache_manager.find_cached_pdf(slug, slug=slug)


def _pdf_candidates(stem: str, *, limit: int = 8) -> list[Path]:
    """缺源时的候选 PDF：按 slug 词元粗匹配 ``cache/pdfs``，最新优先（供 ``--pdf`` 指定）。"""
    tokens = [t for t in re.split(r"[-_ ]+", _slug_of(stem).lower()) if len(t) >= 4]
    cands = [
        p
        for p in settings.cache_pdfs.rglob("*.pdf")
        if any(t in p.stem.lower() for t in tokens)
    ]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[:limit]


# ---------------------------------------------------------------------------
# work.md 对账（status / submit 共用）
# ---------------------------------------------------------------------------
def read_work_translations(work_md: Path) -> dict[str, str]:
    """读 work.md 为 ``{ID: 译文}``；``#`` 注释行与空行忽略，行首 ID 为键。

    Args:
        work_md: work.md 路径。

    Returns:
        ID → 该行 TAB 后的全文（译文或未改的英文原文）。
    """
    out: dict[str, str] = {}
    for line in work_md.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ID_PREFIX_RE.match(line)
        if not m:
            continue
        _, _, tail = line.partition("\t")
        out[m.group(1)] = tail
    return out


_CJK_RE = re.compile(r"[一-鿿]")


#: 占位译文的固定前缀族（测试夹具/冒烟假译；backlog 20261023-162903-lit 防复发）。
#: 「假译」「旧译-」两支是实锤夹具族（2026-10-23 事故文本「假译：旧译-S0001-0」等）。
PLACEHOLDER_PREFIXES: tuple[str, ...] = ("假译", "旧译-")

#: 「仅差固定前缀」判定用的更宽前缀族：译文 == 前缀 + 英文原文时即占位（精确判据，
#: 不会误伤以「中译/旧译」等字起头的真实译文——那要求与英文原文逐字只差前缀）。
_PLACEHOLDER_EQ_PREFIXES: tuple[str, ...] = (
    "假译：",
    "假译:",
    "假译",
    "旧译-",
    "旧译 ",
    "旧译",
    "中译 ",
    "中译",
)


def is_placeholder_translation(zh: str, en: str = "") -> bool:
    """占位译文判定（固定前缀启发式 + 「仅差固定前缀」等值判据）。

    测试夹具/冒烟假译（``假译：…``、``旧译-…``）与「前缀+英文原文」形态都算占位——
    它们含 CJK 会被旧口径误判「已译」，从而被 status 计数、被 submit 当交付物出稿
    （2026-10-23 实锤事故）。占位**不计已译**、``--reuse`` **不回填**。
    """
    z = str(zh).strip()
    if not z:
        return False
    if any(z.startswith(p) for p in PLACEHOLDER_PREFIXES):
        return True
    e = str(en).strip()
    return bool(e) and any(z == f"{p}{e}" for p in _PLACEHOLDER_EQ_PREFIXES)


def classify_translation(en: str, zh: str | None) -> str:
    """单行译文状态：``untranslated``/``done``/``doubtful``。

    缺 ID 或内容与 en 相同记「未译」，**占位译文（假译类）也记「未译」**（不计已译），
    含 CJK 字符记「已译」，其余记「存疑」。
    """
    if zh is None or zh.strip() == en.strip():
        return "untranslated"
    if is_placeholder_translation(zh, en):
        return "untranslated"
    return "done" if _CJK_RE.search(zh) else "doubtful"


def placeholder_ids(
    manifest: dict[str, Any], translations: dict[str, str]
) -> list[str]:
    """manifest 单元中译文为占位（假译类）的 ID 清单（status/submit 提示用）。"""
    return [
        str(u["id"])
        for u in manifest["units"]
        if is_placeholder_translation(
            translations.get(u["id"], ""), str(u.get("en", ""))
        )
    ]


def reconcile(
    manifest: dict[str, Any], translations: dict[str, str]
) -> dict[str, list[str]]:
    """按 ID 对账，返回 ``{"untranslated": [...], "doubtful": [...], "done": [...]}``。"""
    buckets: dict[str, list[str]] = {"untranslated": [], "doubtful": [], "done": []}
    for unit in manifest["units"]:
        st = classify_translation(unit["en"], translations.get(unit["id"]))
        buckets[st].append(unit["id"])
    return buckets


def load_workspace(stem: str) -> tuple[Path, dict[str, Any]]:
    """载入工作区的 manifest，返回 ``(workdir, manifest)``。

    Raises:
        FileNotFoundError: 工作区或 manifest 不存在（未跑 prepare）。
    """
    workdir = workdir_for(stem)
    manifest_path = workdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"工作区未就绪（先跑 bilingual prepare）：{manifest_path}"
        )
    return workdir, json.loads(manifest_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# bilingual status
# ---------------------------------------------------------------------------
def cmd_status(stem: str) -> int:
    """按 ID 对账报进度：计数 + 未译 ID 列表。

    Args:
        stem: 工作区 stem。

    Returns:
        退出码（0=成功；2=工作区未就绪）。
    """
    try:
        workdir, manifest = load_workspace(stem)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    translations = read_work_translations(workdir / "work.md")
    b = reconcile(manifest, translations)
    total = len(manifest["units"])
    print(
        f"[bilingual] {stem}：共 {total} 单元——已译 {len(b['done'])}，"
        f"未译 {len(b['untranslated'])}，存疑 {len(b['doubtful'])}",
        flush=True,
    )
    ph = placeholder_ids(manifest, translations)
    if ph:
        print(
            f"占位译文 {len(ph)} 行（假译类，不计已译）——需真实翻译：{' '.join(ph)}",
            flush=True,
        )
    if b["untranslated"]:
        print(f"未译 ID：{' '.join(b['untranslated'])}", flush=True)
    if b["doubtful"]:
        print(f"存疑 ID（无 CJK 字符，请自查）：{' '.join(b['doubtful'])}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# bilingual submit
# ---------------------------------------------------------------------------
def render_bilingual_md(manifest: dict[str, Any], translations: dict[str, str]) -> str:
    """生成 bilingual.md：英文侧从 manifest 重嵌入（逐字一致），中文侧取 work.md。

    输出格式契约（环节 2 渲染依赖）：章节标题 ``## 中文标题``（保留源 ``#`` 层级）
    紧跟一行 ``*English heading*``；句对为中文行 + 斜体英文行，句对之间空一行；
    段落边界 ``---`` 单独成行；图片原位后紧跟图注句对（同 para，不隔 ``---``）；
    公式/表格/参考文献原样原位。摘要段（role=abstract）前插 ``## 摘要`` 中文小标题、
    其后插 ``---`` + ``## 正文`` 正文起始标注（中文独行标题、**无英文行**——2026-11-03
    用户裁决覆盖 10-23 旧裁决：管线不得发明源文没有的英文标注）。
    """
    items: list[dict[str, Any]] = sorted(
        [dict(x, _kind="unit") for x in manifest["units"]]
        + [dict(x, _kind="block") for x in manifest["blocks"]],
        key=lambda x: x["order"],
    )
    lines: list[str] = []
    prev_para: int | None = None
    abs_on = False  # 已进入摘要段
    body_marked = False
    for idx, it in enumerate(items):
        nxt = items[idx + 1] if idx + 1 < len(items) else None
        if prev_para is not None and it["para"] != prev_para:
            lines.append("---")
        is_abs = it["_kind"] == "unit" and it.get("role") == "abstract"
        # 摘要/正文起始标注（2026-11-03 用户裁决：只留中文小标题，去掉管线发明的
        # *Abstract*/*Main text* 英文行——原文没有的英文标注不发明，亦不做换行兜底）。
        if is_abs and not abs_on:
            lines.append("## 摘要")
            abs_on = True
        if not is_abs and abs_on and not body_marked:
            lines.append("---")
            lines.append("## 正文")
            body_marked = True
        if it["_kind"] == "block":
            lines.extend(str(it["content"]).split("\n"))
        else:
            zh = translations[it["id"]].strip()
            if it["type"] == "H":
                level = int(it.get("level") or 2)
                lines.append(f"{'#' * level} {zh}")
            else:
                lines.append(zh)
            lines.append(f"*{it['en']}*")
            # 句对之间空一行（同 para 的后继句对才隔空行；图片后紧跟图注不隔）。
            if nxt is not None and nxt["para"] == it["para"] and nxt["_kind"] == "unit":
                lines.append("")
        prev_para = it["para"]
    return "\n".join(lines).rstrip() + "\n"


#: 用户侧副本文件名最大长度（超长截断；扩展名另加不受影响）。
_NAME_MAX = 60


def output_basename(manifest: dict[str, Any], translations: dict[str, str]) -> str:
    """用户侧副本文件名（不含扩展名）= 文档中文标题（2026-10-23 用户裁决：语义命名）。

    取首个 ``H`` 单元的中文译文；无 H 单元回退首个可译单元中文译文（都缺则英文原文）；
    剔除非法文件名字符、压缩空白、截断至 :data:`_NAME_MAX`；空结果回退 ``bilingual``。
    """
    units = sorted(manifest["units"], key=lambda x: x["order"])
    title = ""
    for u in units:
        if u["type"] == "H":
            title = (translations.get(u["id"]) or "").strip() or u["en"].strip()
            break
    if not title:
        for u in units:
            title = (translations.get(u["id"]) or "").strip()
            if title:
                break
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title)
    base = re.sub(r"\s+", " ", base).strip().strip(".")
    return base[:_NAME_MAX].strip() or "bilingual"


def cmd_submit(stem: str, *, out_dir: str | Path | None = None) -> int:
    """校验全部 ID 有译文后生成 bilingual.md 并复制到用户侧交付目录（只增不删）。

    Args:
        stem: 工作区 stem。
        out_dir: 自定义交付目录（``--out-dir``，须在 ``data/`` 内，不存在则创建）；
            缺省回退 ``data/用户的inbox/bilingual/<stem>/``（与旧行为一致）。

    Returns:
        退出码（0=出稿成功；1=未译拒绝出稿；2=工作区未就绪/图片未闭合/交付目录越界）。
    """
    # 交付目录先行校验（fail-fast：越界时零副作用，不白跑出稿/渲染）——
    # --out-dir 自定义（受 assert_within_data 防 stray 护栏），缺省回退默认位置。
    if out_dir is not None:
        try:
            inbox = assert_within_data(Path(out_dir), what="bilingual 交付目录")
        except ValueError as exc:
            print(f"[bilingual] --out-dir 拒绝：{exc}", file=sys.stderr)
            return 2
        label = "自定义交付目录"
    else:
        inbox = assert_within_data(INBOX_ROOT / stem, what="bilingual inbox 副本")
        label = "用户 inbox"
    try:
        workdir, manifest = load_workspace(stem)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    if manifest.get("no_images") or not manifest.get("images_closed", True):
        print(
            "[bilingual] 图片未回补/未闭合，拒绝出稿（prepare 勿带 --no-images）",
            flush=True,
        )
        return 2
    translations = read_work_translations(workdir / "work.md")
    b = reconcile(manifest, translations)
    ph = placeholder_ids(manifest, translations)
    if ph:
        print(
            f"[bilingual] 警告：占位译文 {len(ph)} 行（假译类，不计已译、拒绝出稿）",
            flush=True,
        )
    if b["untranslated"]:
        print(
            f"[bilingual] 拒绝出稿：{len(b['untranslated'])} 个单元未译——"
            f"{' '.join(b['untranslated'])}",
            flush=True,
        )
        return 1
    text = render_bilingual_md(manifest, translations)
    out = workdir / "bilingual.md"
    out.write_text(text, encoding="utf-8")
    # 串联渲染（任务书 20261023-011225 §二.4）：PDF 失败=警告不阻塞 md 交付。
    pdf = workdir / "bilingual.pdf"
    try:
        from . import bilingual_render

        bilingual_render.render_pdf(out, pdf)
    except Exception as exc:  # noqa: BLE001 —— 渲染层任何故障都不阻塞 md 交付
        print(
            f"[bilingual] 警告：PDF 渲染失败（md 交付不受影响）：{exc}", file=sys.stderr
        )
    inbox.mkdir(parents=True, exist_ok=True)
    # 用户侧副本按文档标题命名（md/pdf 同名成对）；工作区内部名不变（render 契约）。
    base = output_basename(manifest, translations)
    shutil.copy2(out, inbox / f"{base}.md")
    print(f"[bilingual] 已出稿：{out}", flush=True)
    print(f"[bilingual] 已复制到{label}：{inbox / f'{base}.md'}", flush=True)
    if pdf.is_file():
        shutil.copy2(pdf, inbox / f"{base}.pdf")
        print(f"[bilingual] PDF 已复制到{label}：{inbox / f'{base}.pdf'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# bilingual render
# ---------------------------------------------------------------------------
def cmd_render(stem: str) -> int:
    """bilingual render：把 ``bilingual.md`` 渲染为 ``bilingual.pdf``（同目录）。

    Args:
        stem: 工作区 stem。

    Returns:
        退出码（0=成功；2=无出稿/渲染失败）。
    """
    from . import bilingual_render

    workdir = workdir_for(stem)
    md = workdir / "bilingual.md"
    if not md.is_file():
        print(
            f"[bilingual] 无 bilingual.md（先跑 bilingual submit 出稿）：{md}",
            flush=True,
        )
        return 2
    try:
        pdf = bilingual_render.render_pdf(md, workdir / "bilingual.pdf")
    except Exception as exc:  # noqa: BLE001 —— 渲染层故障统一报因
        print(f"[bilingual] 渲染失败：{exc}", file=sys.stderr)
        return 2
    print(f"[bilingual] PDF：{pdf}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# bilingual prepare
# ---------------------------------------------------------------------------
#: 归一化空白（reuse 匹配口径：英文原文归一化空白后等值）。
def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_reuse_map(reuse: str) -> dict[str, str]:
    """``--reuse`` 源 → ``{归一化英文原文: 旧译文}``（旧 work.md + 同目录 manifest.json）。

    旧 work.md 行尾是译文（已译）或英文原文（未译）；英文原文键取旧 manifest 的
    ``en``（无 manifest 时行尾自键——此时已译行无键可用，回填为空）。源可以是旧
    work.md 文件、工作区目录或 stem。

    Raises:
        FileNotFoundError: 源 work.md 不存在。
    """
    p = Path(reuse)
    if p.is_dir():
        work_md = p / "work.md"
    elif p.is_file():
        work_md = p
    else:
        work_md = workdir_for(str(p)) / "work.md"
    if not work_md.is_file():
        raise FileNotFoundError(f"--reuse 源 work.md 不存在：{work_md}")
    old_rows = read_work_translations(work_md)
    en_of: dict[str, str] = {}
    man = work_md.parent / "manifest.json"
    if man.is_file():
        data = json.loads(man.read_text(encoding="utf-8"))
        en_of = {str(u.get("id")): str(u.get("en", "")) for u in data.get("units", [])}
    return {_norm_ws(en_of.get(uid, tail)): tail for uid, tail in old_rows.items()}


def _translated_rows(work_md: Path) -> list[str]:
    """work.md 中已含译文的行 ID（行尾含 CJK 即视为译文——防覆写闸的判据）。"""
    return [
        k for k, tail in read_work_translations(work_md).items() if _CJK_RE.search(tail)
    ]


def guard_translated_workspace(workdir: Path, *, force: bool) -> tuple[int, str]:
    """防覆写闸（backlog 20261023-163700-devops）：含译文工作区再 prepare 的保护。

    工作区即生产资料——旧 work.md 的译文被静默覆盖＝数据丢失口（2026-10-23 实测事故）。
    默认**拒绝**（rc=2）；``--force`` 放行并**自动备份**旧 work.md/manifest 为
    ``*.bak-<ts>``。返回 ``(rc, 输出行)``：rc=0 继续、rc=2 调用方直接返回。
    """
    old_work = workdir / "work.md"
    if not old_work.is_file():
        return 0, ""
    translated = _translated_rows(old_work)
    if not translated:
        return 0, ""  # 无译文：幂等重跑不设门
    if not force:
        return 2, (
            f"[bilingual] 拒绝覆写：工作区已有译文 {len(translated)} 行"
            f"（如 {translated[0]}）——防译文丢失默认拒绝。确认覆写加 --force"
            "（自动备份旧 work.md/manifest）；要保留译文改用 --reuse 回填"
        )
    ts = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    saved: list[str] = []
    for name in ("work.md", "manifest.json"):
        src = workdir / name
        if src.is_file():
            shutil.copy2(src, workdir / f"{name}.bak-{ts}")
            saved.append(name)
    return 0, f"[bilingual] --force：旧 {'/'.join(saved)} 已备份为 *.bak-{ts}"


def cmd_prepare(
    target: str,
    *,
    no_images: bool = False,
    pdf: str | None = None,
    reuse: str | None = None,
    force: bool = False,
) -> int:
    """生成翻译工作文档 work.md + manifest.json + 回补图片，stdout 输出翻译指引。

    Args:
        target: 源 minerU markdown 路径或 stem。
        no_images: 跳过图片回补（测试/调试；submit 会拒绝出稿）。
        pdf: 手工指定的源 PDF（反解失败时按报错里的候选清单指定）。
        reuse: 旧 work.md/工作区目录/stem——按英文原文（归一化空白等值）回填旧译文，
            匹配不上的单元保持未译待补。
        force: 允许覆写**含译文**的工作区（默认拒绝；放行时自动备份旧
            work.md/manifest 为 ``*.bak-<ts>``）。

    Returns:
        退出码（0=成功；2=前置不满足）。
    """
    try:
        source_md, stem = resolve_source(target)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    md_text = source_md.read_text(encoding="utf-8")
    doc = parse_markdown(md_text)
    try:
        verify_slices(md_text, doc)
    except ValueError as exc:
        print(f"[bilingual] prepare 拒绝：{exc}", flush=True)
        return 2
    n_abs = mark_abstract(doc)
    reuse_map: dict[str, str] = {}
    if reuse:
        try:
            reuse_map = load_reuse_map(reuse)
        except FileNotFoundError as exc:
            print(f"[bilingual] --reuse 拒绝：{exc}", flush=True)
            return 2
    if pdf:
        src_pdf: Path | None = Path(pdf).expanduser()
        if not src_pdf.is_file():
            print(f"[bilingual] --pdf 指定的文件不存在：{src_pdf}", flush=True)
            return 2
        src_pdf = src_pdf.resolve()
    else:
        src_pdf = _find_source_pdf(stem, source_md=source_md)
    if src_pdf is not None:
        # 题名一致性（backlog 20261103-234914-lit）：源 PDF 内容题名 vs 抽取件题名 vs
        # DOI 元数据题名——文件名 DOI ≠ 内容的版本错配只告警不阻断。
        ref_title = ""
        doi = next(
            (
                i
                for i in extract_pdf_identifiers(md_text[:_MD_HEAD_CHARS])
                if i.startswith("10.")
            ),
            "",
        )
        if doi:
            try:
                from . import openalex_client

                ref_title = str(
                    (openalex_client.get_work(doi=doi) or {}).get("title") or ""
                )
            except Exception:  # noqa: BLE001 - 离线只降级为「未比对」
                ref_title = ""
        md_title = ""
        m = re.search(r"^#\s+(.+)$", md_text, re.M)
        if m:
            md_title = m.group(1).strip().strip("*` ")
        for line in cache_manager.title_consistency_note(
            pdf=src_pdf,
            md_title=md_title,
            ref_title=ref_title,
            where=f"prepare {stem}",
        ):
            print(line, flush=True)
    workdir = workdir_for(stem)
    workdir.mkdir(parents=True, exist_ok=True)
    rc, note = guard_translated_workspace(workdir, force=force)
    if note:
        print(note, flush=True)
    if rc != 0:
        return rc
    # --reuse 回填：英文原文（归一化空白）等值匹配旧译文；未匹配保持未译待补。
    # 占位译文（假译类）**不回填**（告警跳过——防夹具扩散到新工作区）。
    backfilled: dict[str, str] = {}
    skipped_placeholder = 0
    for u in doc.units:
        zh = reuse_map.get(_norm_ws(u.en), "")
        if not zh or zh.strip() == u.en.strip():
            continue
        if is_placeholder_translation(zh, u.en):
            skipped_placeholder += 1
            continue
        backfilled[u.id] = zh
    (workdir / "work.md").write_text(render_work_md(doc, backfilled), encoding="utf-8")
    manifest = build_manifest(doc, source_md, no_images)
    try:
        backfill_images(workdir, doc.images, src_pdf, no_images=no_images)
    except ImageBackfillError as exc:
        print(f"[bilingual] 图片回补失败：{exc}", flush=True)
        return 2
    manifest["images_closed"] = all(
        (workdir / "images" / r.rsplit("/", 1)[-1]).is_file() for r in doc.images
    )
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    n_h = sum(1 for u in doc.units if u.type == "H")
    n_s = sum(1 for u in doc.units if u.type == "S")
    n_c = sum(1 for u in doc.units if u.type in ("C", "T"))
    print(f"[bilingual] 工作文档：{workdir / 'work.md'}", flush=True)
    print(
        f"[bilingual] 单元 {len(doc.units)}（H {n_h} / S {n_s} / C·T {n_c}）"
        f"，透传块 {len(doc.blocks)}，图 {len(doc.images)}",
        flush=True,
    )
    if n_abs:
        print(
            f"[bilingual] 摘要段已标注（{n_abs} 句 role=abstract；出稿前插「摘要/正文起始」标注）",
            flush=True,
        )
    if reuse:
        print(
            f"[bilingual] --reuse 回填 {len(backfilled)}/{len(doc.units)} 句"
            "（按英文原文匹配；未匹配保持未译）",
            flush=True,
        )
        if skipped_placeholder:
            print(
                f"[bilingual] 警告：跳过占位译文 {skipped_placeholder} 行"
                "（假译类，不回填——请补真实翻译）",
                flush=True,
            )
    print("--- 翻译指引 ---", flush=True)
    print(f"工作文档：{workdir / 'work.md'}", flush=True)
    print(f"规则：{RULES_TEXT}", flush=True)
    print(
        "钩子行为：编辑 work.md 时 pod-guard 机械校验行结构（行数与行首 ID 不得改动）。",
        flush=True,
    )
    print(f"先读源文档统一术语：{source_md}", flush=True)
    print(f"查进度：pysci-research bilingual status {stem}", flush=True)
    print(f"完成后出稿：pysci-research bilingual submit {stem}", flush=True)
    return 0
