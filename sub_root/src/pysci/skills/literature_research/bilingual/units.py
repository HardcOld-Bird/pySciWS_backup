"""句切分与结构解析——manifest 切片契约的生产端（旧 bilingual.py 平移，语义不变）。

两道换接缝（2026-11-17 吸收实施）：

1. **句切分**：pysbd → **wtpsplit**（SaT 小模型，CPU）。缩写哨兵（``Fig.``/``et al.``/
   省略号先换 U+E000 防误断）与**游标逐字拼回**（:func:`reassemble`）原样平移——这两层
   是「切片自检失败」类 bug 的历史根治（backlog 20261029-121943-lit），与切分器无关。
   模型路径约定 ``deps_management/deps/models/wtpsplit/``（环境变量 ``WTPSAT_MODEL_DIR``
   可覆盖）；wtpsplit 未装/模型未 vendor 时报 :class:`SplitDependencyError`，不静默回退。
2. **MD 解析**：正则状态机 → **markdown-it-py**。块结构由 CommonMark token 决定，
   切片用 ``token.map`` 的源行区间**逐字取源文**（构造保证不漂移）；``$$`` 公式块与
   ``<table>`` HTML 的跨空行续块按行补齐（CommonMark 会在此断块——兼容 shim，见
   :func:`parse_markdown`）。

句末符只认 ``. ? !``（学术正文契约）：切分器在别处断出的碎片由 :func:`reassemble`
的合并规则回并，**产物语义不依赖切分器行为**。切片自检（:func:`verify_slices`）是
prepare 的硬闸：各块按 order 拼回源文档必须无损，错位即拒。
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt

from pysci import paths

#: 单元类型字母：标题 / 句子 / 图注 / 表注。
TYPE_LETTERS: tuple[str, ...] = ("H", "S", "C", "T")

#: 句切分保护哨兵（U+E000 私用区）：省略号与缩写点先换成它，切完再还原。
_SENTINEL = "\ue000"

#: 学术缩写（句中点，不是句末）：先换哨兵防切分器误断。``Eq. (2)`` 这类后接括号/数字的
#: 是历史误断重灾区（pysbd 实测），故无论切分器强弱都保留这层机械保护。
_ABBREVIATIONS: tuple[str, ...] = (
    "e.g.", "i.e.", "et al.", "al.", "Fig.", "Figs.", "Eq.", "Eqs.", "eq.",
    "vs.", "cf.", "Ref.", "Refs.", "Sec.", "Ch.", "No.", "Vol.", "pp.",
    "ca.", "approx.", "Dr.", "Prof.", "Mr.", "Ms.", "Mrs.", "St.", "Jr.", "Sr.",
    # 空格变体（MinerU/OCR 提取稿常见「i. e.」「e. g.」，2026-11-17 全链冒烟实测误断）。
    "i. e.", "e. g.",
)

#: 省略号（``...`` / ``…``）不作句末——学术正文中几乎总是句中省略。
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")

#: 参考文献节标题（命中后段落按条目透传，不再切句待译）。
_REFERENCES_RE = re.compile(r"(?i)^\s*(references|bibliography|参考文献)")

#: 图注 / 表注惯例前缀（``Fig. 1`` / ``Table 2`` / ``图 3``）；无前缀一律按正文（宁漏勿错）。
_CAPTION_FIG_RE = re.compile(r"^\s*(?:FIG(?:URE)?\.?|Fig\.?|图)\s*[\dA-Z]", re.I)
_CAPTION_TAB_RE = re.compile(r"^\s*(?:TABLE|Tab\.?|表)\s*[\dA-Z]", re.I)

#: 图片引用行（整行一个图，透传）。
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")

#: 元数据行形态（出版信息/DOI/官网链接——摘要识别的前置带判据）。
_META_RE = re.compile(
    r"^(Received|Accepted|Published|Check for updates|Article|ARTICLE|DOI[:.\s]|https?://)",
    re.I,
)


class SplitDependencyError(RuntimeError):
    """句切分依赖（wtpsplit / vendored SaT 模型）不可用——由依赖代理安装，勿静默回退。"""


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
        role: 语义角色标注（``abstract``=摘要段，prepare 摘要识别所置；识别错误可手改
            manifest 的该字段后重新 submit）。
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
# 句切分（缩写哨兵 → 切分器 → 游标逐字拼回）
# ---------------------------------------------------------------------------
def _protect(text: str) -> str:
    """把缩写点与省略号换成哨兵，避免切分器在这些位置断句。"""
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", _SENTINEL))
    return _ELLIPSIS_RE.sub(lambda m: m.group(0).replace(".", _SENTINEL), text)


def _restore(text: str) -> str:
    """还原哨兵为原文点号/省略号。"""
    return text.replace(_SENTINEL, ".")


#: wtpsplit SaT 实例缓存（模型加载昂贵，进程内单例）。
_SAT_CACHE: list[object] = []


def wtpsplit_model_dir() -> Path:
    """vendored SaT 模型目录（约定 ``deps_management/deps/models/wtpsplit/``）。

    Returns:
        模型目录路径（可能尚不存在——调用方据此报缺件）。
    """
    return Path(
        os.environ.get("WTPSAT_MODEL_DIR")
        or paths.PROJECT_ROOT / "deps_management" / "deps" / "models" / "wtpsplit"
    )


def default_segmenter(protected: str) -> list[str]:
    """默认切分器：wtpsplit SaT（对**已加哨兵**的文本切分，返回原文碎片）。

    模型按约定目录 vendor（``deps_management/deps/models/wtpsplit/sat-3l`` ONNX +
    ``xlm-roberta-base-tokenizer``，由依赖代理落盘）；ONNX CPU 推理、进程内缓存。
    自包含 ADR（2026-11-17）：HF 缓存默认入项目内 ``deps_management/deps/cache/huggingface``
    （本机受限令牌下 ``~/.cache/huggingface`` EACCES，见 vendor 脚本头注）；
    模型与 tokenizer 均为本地路径，全程无联网回退。

    Args:
        protected: :func:`_protect` 处理后的段落文本。

    Returns:
        切分碎片（保留原文间距；:func:`reassemble` 负责逐字拼回）。

    Raises:
        SplitDependencyError: wtpsplit 未安装，或约定目录下模型/tokenizer 缺件。
    """
    if _SAT_CACHE:
        sat = _SAT_CACHE[0]
    else:
        try:
            from wtpsplit import SaT
        except ImportError as exc:  # pragma: no cover - 依赖代理安装后消失
            raise SplitDependencyError(
                "wtpsplit 未安装（依赖代理负责：uv add wtpsplit[onnx-cpu]）"
            ) from exc
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 全本地路径加载，禁联网回退
        os.environ.setdefault(
            "HF_HOME", str(paths.PROJECT_ROOT / "deps_management" / "deps" / "cache" / "huggingface")
        )
        model_root = wtpsplit_model_dir()
        # 约定目录本身或其唯一携带 config.json 的子目录即模型根（vendor 布局两可）。
        if (model_root / "config.json").is_file():
            model_path = model_root
        else:
            subs = [p for p in model_root.glob("*/") if (p / "config.json").is_file()]
            model_path = subs[0] if len(subs) == 1 else model_root
        tok_path = model_root / "xlm-roberta-base-tokenizer"
        if not (model_path / "config.json").is_file() or not tok_path.is_dir():
            raise SplitDependencyError(
                f"SaT 模型/tokenizer 未 vendor：{model_root}（约定 sat-3l/ + "
                "xlm-roberta-base-tokenizer/，由依赖代理落盘；WTPSAT_MODEL_DIR 可覆盖）"
            )
        sat = SaT(
            str(model_path),
            tokenizer_name_or_path=str(tok_path),
            ort_providers=["CPUExecutionProvider"],  # CPU 推理（硬件约束：无 GPU）
        )
        _SAT_CACHE.append(sat)
    return list(sat.split(protected))


def reassemble(merged: str, fragments: Iterable[str]) -> list[str]:
    """游标逐字拼回：把切分碎片还原为「句末只认 . ? !」的句子列表。

    两层语义（旧实现原样平移，切分器无关）：

    1. **分隔按原文取**：碎片在 ``merged`` 中按游标定位，碎片间分隔取原文间隙——
       禁止硬插空格（``Fig. 3(a).`` 面板号回并漂移事故，backlog 20261029-121943-lit）；
    2. **句末合并**：碎片不以句末符（``. ? !。！？``）结尾则与后一碎片合并。传入
       **哨兵域**碎片时缩写点是哨兵、天然不算句末——wtpsplit 在缩写处的误切
       （``Fig␀ ``/``Eq␀ ``）由此回并；传原文域则按原文判定（测试直调口径）。

    Args:
        merged: 段落合并软换行后的原文（与 fragments 同域）。
        fragments: 切分器碎片（可含首尾空白）。

    Returns:
        句子列表，每句为该域的逐字串（首尾去空白）。
    """
    out: list[str] = []
    cursor = 0
    for part in fragments:
        s = str(part).strip()
        if not s:
            continue
        start = merged.find(s, cursor)
        gap = merged[cursor:start] if start >= 0 else " "
        if out and not _SENT_END_RE.search(out[-1]):
            out[-1] = out[-1] + gap + s
        else:
            out.append(s)
        if start >= 0:
            cursor = start + len(s)
    return out


#: 句末符（行尾口径）：`. ? !` 与中文强标点 `。！？`；省略号（``...``/``…``）不作句末。
_SENT_END_RE = re.compile(r"[.?!。！？]$")

#: 括号对（force-split 括号内不切——K2 实测 sat-3l 括号内误切防护）。
_BRACKETS = {"(": ")", "（": "）", "[": "]", "【": "】"}


def _force_split(sentences: Iterable[str]) -> list[str]:
    """句末硬契约补齐：句内残留的句末符边界一律切开（切片取自身，逐字）。

    切分器跨句界合块（wtpsplit 把 ``Yes! It is.`` 并成一块、``Table 1. …`` 不切）时
    由本层规整——「句末只认强标点」是硬契约，不依赖切分器行为。口径：
    ``. ? !`` 后须接空白/行尾；``。！？`` 直接断（中英混排无空格）；**括号内不切**
    （``（deep learning）`` 类内含点号不误断）；省略号不作句末（哨兵/排除集双保险）。
    """
    out: list[str] = []
    for s in sentences:
        stack: list[str] = []
        pos = 0
        for i, ch in enumerate(s):
            if ch in _BRACKETS:
                stack.append(_BRACKETS[ch])
            elif stack and ch == stack[-1]:
                stack.pop()
            elif (
                not stack
                and i + 1 < len(s)  # 边界即句尾不拆；句内边界切开
                and (
                    ch in "。！？"
                    or (ch in ".?!" and s[i + 1].isspace())
                )
            ):
                piece = s[pos : i + 1].strip()
                if piece:
                    out.append(piece)
                pos = i + 1
        tail = s[pos:].strip()
        if tail:
            out.append(tail)
    return out


def split_sentences(
    paragraph: str, *, segmenter: Callable[[str], list[str]] | None = None
) -> list[str]:
    """把段落文本切成句子（合并软换行 → 哨兵保护 → 切分器 → 游标拼回 → 硬契约补齐）。

    语义保证与切分器无关（旧测试期望逐字保持）：缩写/省略号不误切、强标点边界
    必切（含中英混排 ``。！？``）、面板号等无空格形态逐字不漂移、括号内不切。

    Args:
        paragraph: 单个段落的原文（可含软换行）。
        segmenter: 切分器（对加哨兵文本返回碎片）；缺省用 :func:`default_segmenter`
            （wtpsplit）。测试可注入确定性切分器。

    Returns:
        按出现顺序的句子列表（原文逐字串）；空段落返回空表。

    Raises:
        SplitDependencyError: 缺省切分器依赖不可用。
    """
    merged = re.sub(r"\s*\n\s*", " ", paragraph).strip()
    if not merged:
        return []
    protected = _protect(merged)
    segment = segmenter or default_segmenter
    # 合并与定位在哨兵域进行（句末判定=哨兵还原前形态），最后统一还原。
    protected_sentences = reassemble(protected, segment(protected))
    return [_restore(s) for s in _force_split(protected_sentences)]


# ---------------------------------------------------------------------------
# markdown-it-py 结构解析（token.map 逐字切片；$$ / <table> 跨空行续块行补齐）
# ---------------------------------------------------------------------------
def _classify_paragraph(text: str) -> str:
    """段落首句判图注（C）/表注（T）/正文（S）——无前缀一律 S（宁漏勿错）。"""
    head = text.lstrip().lstrip("*").lstrip()
    if _CAPTION_FIG_RE.match(head):
        return "C"
    if _CAPTION_TAB_RE.match(head):
        return "T"
    return "S"


def _raw_lines(lines: list[str], start: int, end: int) -> list[str]:
    """取源文行区间（token.map 语义：``[start, end)``，0 基），逐行去首尾空白。"""
    return [ln.strip() for ln in lines[start:end]]


def _merge_text(raw: list[str]) -> str:
    """段内软换行合并（与旧实现同口径：换行处归一为空格，行内空白保留）。"""
    return re.sub(r"\s*\n\s*", " ", "\n".join(raw)).strip()


def parse_markdown(
    md_text: str, *, segmenter: Callable[[str], list[str]] | None = None
) -> ParsedDoc:
    """解析 MinerU 风格 markdown 为单元（H/S/C/T）+ 透传块序列。

    结构识别（CommonMark token + 行级兼容 shim）：标题、段落（软换行合并后切句）、
    图片引用行（透传）、``$$...$$`` 公式块（透传，含块内空行）、``<table>`` HTML
    （透传，可多行）、参考文献节（标题命中后段落按条目透传）。

    兼容 shim（原地 ADR）：CommonMark 会把含空行的 ``$$`` 块 / ``<table>`` 拆成多个
    token——按源行区间向前补齐到闭合标记，保旧「整块透传、含块内空行」语义。

    Args:
        md_text: 源 MinerU markdown 全文。
        segmenter: 句切分器（透传给 :func:`split_sentences`）。

    Returns:
        :class:`ParsedDoc`——单元与透传块按文档序、order 共用计数。
    """
    lines = md_text.split("\n")
    doc = ParsedDoc()
    order = 0
    para = -1
    section = ""
    in_references = False
    prev_was_image = False
    tokens = MarkdownIt("commonmark", {"html": True}).parse(md_text)
    i = 0
    consumed_to = 0  # 行级续块补齐（$$/<table>）消费到的源行——其内 token 不再重复处理
    while i < len(tokens):
        tok = tokens[i]
        span = tok.map or [0, 0]
        start, end = span[0], span[1]
        if start < consumed_to:
            i += 1
            continue
        if tok.type == "heading_open":
            para += 1
            title = tokens[i + 1].content if i + 1 < len(tokens) else ""
            if _REFERENCES_RE.match(title):
                in_references = True
            else:
                section = title
            doc.units.append(
                Unit(id="", type="H", en=title, order=order, para=para,
                     section=title, level=int((tok.tag or "h1")[1:]))
            )
            order += 1
            prev_was_image = False
            i += 3  # heading_open / inline / heading_close
            continue
        if tok.type == "html_block":
            # 透传块逐字取源行（不 strip——公式/表格内容是渲染契约的原样输入）。
            end_v = end
            if lines[start:end] and lines[start].strip().startswith("<table"):
                # <table> 跨空行续块补齐到 </table>。
                while "</table>" not in "\n".join(lines[start:end_v]) and end_v < len(lines):
                    end_v += 1
                para += 1
                doc.blocks.append(
                    PassBlock("table", "\n".join(lines[start:end_v]), order, para, section)
                )
                order += 1
                prev_was_image = False
                consumed_to = end_v
            else:
                # 非表格 HTML 旧口径按正文行进段落流（MinerU 产物罕见）。
                para += 1
                text = _merge_text(_raw_lines(lines, start, end))
                for sent in split_sentences(text, segmenter=segmenter):
                    doc.units.append(
                        Unit(id="", type="S", en=sent, order=order, para=para,
                             section=section)
                    )
                    order += 1
                prev_was_image = False
            i += 1
            continue
        if tok.type == "paragraph_open":
            raw = _raw_lines(lines, start, end)
            # MinerU 真实产物常见形态：图行与图注行**同段落**（图行后硬换行紧跟图注行）。
            # 段首整行图逐行剥离（透传 + 登记引用名），余下文本走段落流——图注与其图片
            # 同 para（渲染契约：图片原位，紧跟图注句对，不隔 ---）。
            n_img = 0
            while n_img < len(raw) and _IMAGE_RE.match(raw[n_img]):
                line = raw[n_img]
                name = line[line.index("(") + 1 : line.rindex(")")]
                doc.images.append(name)
                if n_img == 0:
                    para += 1
                doc.blocks.append(PassBlock("image", line, order, para, section))
                order += 1
                n_img += 1
            if n_img:
                prev_was_image = True
            raw = raw[n_img:]
            if not any(ln.strip() for ln in raw):
                i += 3  # paragraph_open / inline / paragraph_close
                continue
            stripped = "\n".join(raw)
            if stripped.strip().startswith("$$"):
                # 公式块：$$ 起止（含块内空行）整块透传。
                end_v = end
                while "\n".join(lines[start:end_v]).count("$$") < 2 and end_v < len(lines):
                    end_v += 1
                para += 1
                doc.blocks.append(
                    PassBlock("formula", "\n".join(lines[start:end_v]), order, para, section)
                )
                order += 1
                prev_was_image = False
                consumed_to = end_v
                i += 3
                continue
            text = _merge_text(raw)
            if in_references:
                para += 1
                doc.blocks.append(
                    PassBlock("reference", text, order, para, section)
                )
                order += 1
                prev_was_image = False
                i += 3
                continue
            kind = _classify_paragraph(text)
            # 图注与其图片同 para（渲染契约：图片原位，紧跟其图注句对，不隔 ---）。
            if not (prev_was_image and kind == "C"):
                para += 1
            prev_was_image = False
            for sent in split_sentences(text, segmenter=segmenter):
                doc.units.append(
                    Unit(id="", type=kind, en=sent, order=order, para=para,
                         section=section)
                )
                order += 1
            i += 3
            continue
        i += 1  # 其余结构性 token（list/table 等）：其内层 paragraph_open 会单独处理
    _assign_ids(doc)
    return doc


def _assign_ids(doc: ParsedDoc) -> None:
    """按类型字母 + 出现顺序补 ID（H0001/S0001/C0001/T0001，全文档唯一）。"""
    counters: dict[str, int] = {}
    for u in doc.units:
        counters[u.type] = counters.get(u.type, 0) + 1
        u.id = f"{u.type}{counters[u.type]:04d}"


def verify_slices(md_text: str, doc: ParsedDoc) -> None:
    """切片自检：各单元/透传块按 order 拼接回源文档须无损，防切片错位。

    逐块核对：每块原文（空白归一后）必须能按序在源文档中找到（单元用句、透传块整块）。
    任一错位直接抛 :class:`ValueError`，prepare 据此拒绝。

    Args:
        md_text: 源 markdown 全文。
        doc: :func:`parse_markdown` 的解析结果。

    Raises:
        ValueError: 切片与源块对不上（切片错位）或解析结果为空。
    """
    items: list[Unit | PassBlock] = sorted(
        [*doc.units, *doc.blocks], key=lambda x: x.order
    )
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
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


def _is_author_line(text: str) -> bool:
    """作者行判据：≥2 个上标单位标注（``$^{1,6}$`` 形态的作者-机构角标）。"""
    return text.count("$^{") >= 2


def mark_abstract(doc: ParsedDoc) -> int:
    """识别摘要段并置 ``role="abstract"``（返回摘要句数；0=未识别）。

    判据（2026-10-23 用户裁决）：首个标题之后**先出现元数据/作者行**，其后首个段落
    （同 para 全组）即摘要——MinerU 丢弃 Abstract 标题后摘要常是「元数据带之后的
    第一段」。无元数据带（纯正文 md）不标注；识别错误可手改 manifest 的 ``role`` 字段。
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
