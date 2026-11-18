"""arXiv e-print LaTeX 源清洗（tex → 轻量 Markdown）。

ADR（自研理由，2026-11-17）：社区无对应薄实现——arxiv-mcp-server / arxiv.py 已因活跃度口径
拒绝（检索/下载由 paper-search-mcp 的 arXiv 连接器承载），而 e-print 源特有的细节清洗
（% 注释、preamble、thebibliography 剔除）无人承接；MinerU/marker 面向 PDF 而非 tex 源。
故自研本纯字符串变换层：无 I/O、无第三方依赖，全部行为可机械断言
（``tests/skills/literature_research/test_tex_clean.py``）。

清洗合同：

- 数学（``$...$``）与 ``\\cite*{...}`` **原样保留**（citecheck 依赖后者，公式回填依赖前者）；
- 转换只做最小映射（section→MD 标题、textbf/it→MD 强调、ref→[ref:x]），不做完整 TeX 语义；
- 无 ``\\begin{document}`` 锚点时视为全文即 body，不猜测 preamble 边界。
"""

from __future__ import annotations

import re

_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_BIBLIO_RE = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)
_HEADING_RES = (
    (re.compile(r"\\section\*?\{([^}]*)\}"), "## "),
    (re.compile(r"\\subsection\*?\{([^}]*)\}"), "### "),
    (re.compile(r"\\subsubsection\*?\{([^}]*)\}"), "#### "),
    (re.compile(r"\\paragraph\*?\{([^}]*)\}"), "##### "),
)
_BOLD_RE = re.compile(r"\\textbf\{([^}]*)\}")
_ITAL_RE = re.compile(r"\\(?:textit|emph)\{([^}]*)\}")
_REF_RE = re.compile(r"\\(?:eq)?ref\{([^}]*)\}")


def strip_comments(src: str) -> str:
    """剔除 ``%`` 行内注释；转义的 ``\\%`` 与换行结构保留。"""
    return "\n".join(_COMMENT_RE.sub("", line) for line in src.splitlines())


def strip_preamble(src: str) -> str:
    """剔除 ``\\begin{document}`` 之前的 preamble 与之后的 ``\\end{document}``。

    无锚点时原样返回（合同见模块 docstring）。
    """
    anchor = src.find(r"\begin{document}")
    if anchor == -1:
        return src
    body = src[anchor + len(r"\begin{document}") :]
    end = body.rfind(r"\end{document}")
    return body[:end] if end != -1 else body


def strip_thebibliography(src: str) -> str:
    """剔除 ``thebibliography`` 环境（书目对账走 refs.bib/frontmatter，不读正文书目）。"""
    return _BIBLIO_RE.sub("", src)


def _heading_repl(prefix: str):
    """返回把 ``\\section{X}`` 映射为 ``<prefix>X`` 的替换函数（避免闭包绑错循环变量）。"""

    def repl(match: re.Match[str]) -> str:
        return f"{prefix}{match.group(1)}"

    return repl


def tex_body_to_markdown(src: str) -> str:
    """轻量 TeX→Markdown 映射：标题层级、强调、引用占位；数学与 cite 原样保留。"""
    out = src
    for pattern, prefix in _HEADING_RES:
        out = pattern.sub(_heading_repl(prefix), out)
    out = _BOLD_RE.sub(r"**\1**", out)
    out = _ITAL_RE.sub(r"*\1*", out)
    return _REF_RE.sub(r"[ref:\1]", out)


def clean_arxiv_tex(src: str) -> str:
    """e-print 清洗全链：去注释 → 去 preamble → 去 thebibliography → 轻量 MD。

    返回去尾随空白/首尾空行的文本；行结构保留。
    """
    out = strip_preamble(strip_comments(src))
    out = tex_body_to_markdown(strip_thebibliography(out))
    return "\n".join(line.rstrip() for line in out.splitlines()).strip("\n")
