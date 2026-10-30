"""corpus_clean —— rag 语料的删除式规范化（洗掉纯噪声，**绝不改动原始产物**）。

为什么需要这一层（以下数字全部实测于 2026-10-04，语料 = 74 个 .md / 19,528,806 字符）
------------------------------------------------------------------------------------
1. **paperqa 2026.8.12 把 ``.md`` 当代码处理**。``readers.read_doc`` 的后缀分派里没有
   ``.md`` 分支，故落到最后的 else → ``parse_text(split_lines=True)`` + ``chunk_code_text``
   （逐行累加、到 ``chunk_chars`` 就硬切，既不看标题也不看句子边界）。因此「某产物零个
   ``##``」对分块**毫无影响**，治理标题结构是无用功；真正吃掉 chunk 预算的是噪声字符。

2. **单篇论文的噪声占 25%–40%**。语料整体只有 3.0%，那是被 12.4M 字符的 COMSOL 手册
   （占语料 63%，既无参考文献块、图片行占比也低）稀释后的平均值——而手册不是提问对象，
   论文才是。三类噪声：

   ================================  ===========  ====================
   噪声类别                          字符数        占语料
   ================================  ===========  ====================
   MinerU 图片占位行                  272,685      1.40%
   纯文本编号参考文献块（≥3 条连续）   319,995      1.64%（95 块）
   REVTeX 原始书目宏                   35,633      0.18%（1 文件占 58.8%）
   ================================  ===========  ====================

   图片占位行形如 ``![](images/<64 位十六进制>.jpg)``，整行只有内容哈希；参考文献块形如
   ``[1] Schroeder M R 1975 Diffuse sound reflection … J. Acoust. Soc. Am. 57 1``。
   两者对语义检索都是零价值，却会占满 chunk、稀释 embedding、把正文挤出 top-k。

设计约束
--------
- **绝不修改 ``cache/extracted`` 下的原始产物**。那里既有 MinerU 配额换来的抽取件，也有
  人工整理进主题子目录的存量数据（``cpa_ep/``、``ep_theory/`` 等不符合 :mod:`pdf_extract`
  的 ``{stem}_{key}.{backend}.md`` 缓存命名，即不可再生）。洗出的副本写到
  ``pqa_home/corpus/``（= ``cache/rag/corpus/``，已被数据区 .gitignore 的 ``cache/*`` 覆盖）。
- 只做**删除式**规范化：不改写、不重排、不猜测语义结构。每条规则各自独立、可单测。
- 与源头修复互补：:data:`pdf_extract._TEX_DROPPED_ENVS` 已含 ``thebibliography``，管的是
  **将来**的 arXiv LaTeX 抽取；本模块管的是**已经躺在磁盘上**的 74 个产物（含 MinerU 云端
  产物——那是第三方服务的输出，我方无法从源头改）。

用法::

    from pysci.skills.literature_research.tools import corpus_clean
    clean, stats = corpus_clean.clean_markdown(raw_text)
    dest, stats = corpus_clean.materialize_clean(src, dest_root, name="cpa_ep__p1")
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

#: 规范化规则版本。**改任何一条删除规则都要 +1**：rag 的索引台账按此判失效并强制全量
#: 重建，否则「按旧规则洗的语料 + 新规则的代码」会一直共存（增量路径按 mtime 跳过，
#: 新规则永远不生效）。
CLEANER_VERSION = 1

#: 派生副本目录名（``pqa_home`` 之下）。单独取名是为了 :func:`reset_corpus_dir` 能做
#: 越界防御——只肯删这个名字的目录。
CORPUS_DIRNAME = "corpus"

#: LaTeX/REVTeX 的原始书目环境。MinerU 走 PDF 路径时会把它渲染成纯文本编号条目（交给
#: 下面的 :data:`_REF_BLOCK_RE`），但 arXiv LaTeX 源码路径会原样留下整个环境——里面全是
#: ``\\bibitem``/``\\citenamefont``/``\\bibinfo``/``\\BibitemShut`` 宏残渣。缺结束标记时
#: 吞到文件末尾（宁可少一段正文，也不要让半篇宏定义进 embedding）。
_BIB_ENV_RE = re.compile(
    r"\\begin\{thebibliography\}.*?(?:\\end\{thebibliography\}|\Z)", re.S
)

#: MinerU 的图片占位行：alt 恒为空、路径是 64 位内容哈希，整行无任何可检索语义。
#: 只删**独占一行**且 alt 为空的（实测语料里带 alt 的行内图片引用为 0 处，故不会误伤）；
#: 图注是另起一行的 ``FIG. 1. (a) …`` 纯文本，不在此规则内，**予以保留**——图注常含
#: 结论性表述，是正文的一部分。
_IMG_LINE_RE = re.compile(r"^[ \t]*!\[\]\([^)\n]*\)[ \t]*\r?\n?", re.M)

#: 一条编号参考文献：行首 ``[n]`` + 空白 + 非空白起始的正文。
_REF_LINE = r"\[\d{1,3}\][ \t]+\S[^\n]*"

#: 认定「这是一块参考文献表」所需的最少**连续**条目数。取 3 是为了不误伤正文：MinerU 把
#: 行内引用渲染成段中的 ``$[1-8]$`` / ``[9,10]``（不在行首），而正文段落极少连续三行都以
#: ``[n] `` 开头。实测语料命中 95 块 / 319,995 字符，逐块抽查均为真参考文献表。
MIN_REF_BLOCK = 3

#: 连续 ≥:data:`MIN_REF_BLOCK` 条编号参考文献（允许条目间夹空行——MinerU 常在每条之间
#: 插一个空行）。整块删除。
_REF_BLOCK_RE = re.compile(
    rf"(?:^{_REF_LINE}(?:\r?\n|\Z)(?:[ \t]*\r?\n)*){{{MIN_REF_BLOCK},}}", re.M
)

#: 单条参考文献行（用于统计条数，不用于删除）。
_REF_LINE_RE = re.compile(rf"^{_REF_LINE}", re.M)

#: 删除后残留的三连以上空行压回一个空行。``chunk_code_text`` 按**字符**计预算，空行同样
#: 占额；不压的话一篇删掉 30 个图片行的论文会多出几十行空白。
_BLANK_RUN_RE = re.compile(r"(?:[ \t]*\n){3,}")


@dataclass(frozen=True)
class CleanStats:
    """一次规范化的量化结果（供 ``build_index`` 汇总与测试断言）。"""

    chars_in: int = 0
    chars_out: int = 0
    n_bib_envs: int = 0
    n_image_lines: int = 0
    n_ref_blocks: int = 0
    n_ref_lines: int = 0

    @property
    def removed(self) -> int:
        """删掉的字符数（``chars_in - chars_out``，含被压掉的空行）。"""
        return self.chars_in - self.chars_out

    @property
    def removed_ratio(self) -> float:
        """删掉的比例（0–1）；空输入返回 0.0 而不是除零。"""
        return self.removed / self.chars_in if self.chars_in else 0.0


def strip_bibliography_envs(text: str) -> tuple[str, int]:
    """删掉 LaTeX/REVTeX 的 ``thebibliography`` 环境，返回 ``(新文本, 删除块数)``。"""
    hits = _BIB_ENV_RE.findall(text)
    if not hits:
        return text, 0
    return _BIB_ENV_RE.sub("", text), len(hits)


def strip_image_placeholders(text: str) -> tuple[str, int]:
    """删掉独占一行的空 alt 图片占位行，返回 ``(新文本, 删除行数)``。"""
    hits = _IMG_LINE_RE.findall(text)
    if not hits:
        return text, 0
    return _IMG_LINE_RE.sub("", text), len(hits)


def strip_reference_blocks(text: str) -> tuple[str, int, int]:
    """删掉连续编号参考文献表，返回 ``(新文本, 块数, 条数)``。"""
    blocks = _REF_BLOCK_RE.findall(text)
    if not blocks:
        return text, 0, 0
    n_lines = sum(len(_REF_LINE_RE.findall(b)) for b in blocks)
    return _REF_BLOCK_RE.sub("", text), len(blocks), n_lines


def clean_markdown(text: str) -> tuple[str, CleanStats]:
    """对一份抽取产物做删除式规范化，返回 ``(干净文本, 统计)``。

    **幂等**：``clean_markdown(clean_markdown(x)[0])[0] == clean_markdown(x)[0]``。
    这点是 ``materialize_clean`` 「内容未变则不重写」判断的前提。
    """
    chars_in = len(text)
    text, n_bib = strip_bibliography_envs(text)
    text, n_img = strip_image_placeholders(text)
    text, n_blocks, n_refs = strip_reference_blocks(text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text, CleanStats(
        chars_in=chars_in,
        chars_out=len(text),
        n_bib_envs=n_bib,
        n_image_lines=n_img,
        n_ref_blocks=n_blocks,
        n_ref_lines=n_refs,
    )


def materialize_clean(
    src: Path, dest_root: Path, name: str, *, encoding: str = "utf-8"
) -> tuple[Path, CleanStats]:
    """把 ``src`` 洗成 ``dest_root/<name>.md``，返回 ``(副本路径, 统计)``。

    ``name`` 由调用方给（rag 用 ``_docname_for(src)``），本模块不猜命名规则——但要求它
    在 ``dest_root`` 内唯一且不含路径分隔符，否则副本会互相覆盖或写到目录外。

    内容一致时**不重写**：重写会刷新 mtime，让副本看起来比源文件新，白占一次 I/O。
    ``newline="\\n"`` 是刻意的——派生产物必须跨平台字节确定，否则 Windows 的通用换行
    翻译会把 LF 写成 CRLF，同一份语料在两个平台上洗出的副本不一致（这个坑在
    ``journal_metrics.build_scimago_index`` 上实测过一次）。

    无论是否真的删了东西都写副本（不做「没变化就直接索引源文件」的优化）：统一走副本
    让「被索引的到底是哪份文本」只有一个答案，排查时不必先猜这个文件当时走没走优化。
    """
    raw = src.read_text(encoding=encoding, errors="replace")
    clean, stats = clean_markdown(raw)
    dest = dest_root / f"{name}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        unchanged = dest.read_text(encoding=encoding, errors="replace") == clean
    except OSError:
        unchanged = False
    if not unchanged:
        dest.write_text(clean, encoding=encoding, newline="\n")
    return dest, stats


def reset_corpus_dir(dest_root: Path) -> None:
    """清空派生副本目录（``--rebuild`` 时用，免得已删源文件的副本变成孤儿）。

    带越界防御：只肯删**名为** ``CORPUS_DIRNAME`` 的目录。``pqa_home`` 下还住着
    ``index.pkl`` 与 ``index_meta.json``，万一调用方把 ``dest_root`` 传成 ``pqa_home``
    本身，这个守卫能让「顺手删掉整个索引目录」变成一次 no-op 而不是事故。
    """
    if dest_root.name != CORPUS_DIRNAME or not dest_root.is_dir():
        return
    shutil.rmtree(dest_root, ignore_errors=True)
