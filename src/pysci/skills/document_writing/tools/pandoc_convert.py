"""pandoc_convert —— Markdown → Office（docx/pptx）写转换，基于社区标准 Pandoc。

替代此前手写的 markdown_to_docx / markdown_to_pptx 解析器：Pandoc 原生支持标题层级、
项目符号、管道表、数学公式、图片、演讲者备注（pptx 的 `::: notes` fenced div），并可用
`--reference-doc` 套用自定义 Word/PPT 样式模板，保真度与可维护性均远胜手写解析。

能力边界（重要）：Pandoc 只能「从 markup 生成一份全新文档」，**不能追加到既有
.pptx/.docx，也不读取 .pptx**。因此 `slides add` / `docx add` / `slides new`（增量追加
与结构化构建）仍走 python-pptx/python-docx（见 pptx_io / docx_io），本模块只承担
`from-markdown` 命令。

pandoc 二进制由 pypandoc-binary 自带；缺失时抛 PandocNotAvailable 给清晰指引而非裸报错。
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import settings


class PandocNotAvailable(RuntimeError):
    """未检测到 pandoc（Markdown→Office 写转换后端）。"""

    def __str__(self) -> str:  # noqa: D105
        return (
            "未检测到 pandoc。\n"
            "Markdown→docx/pptx 写转换（from-markdown）需要 Pandoc。\n"
            "正常情况下 `uv sync` 已随 pypandoc-binary 装入 pandoc；若仍缺失：\n"
            "  - 重跑 `uv sync`；或在 .env 设 DOCWRITING_PANDOC 指向系统 pandoc 绝对路径。\n"
            "装好后重开终端跑 `compose doctor` 复核。\n"
            "提示：阅读/提取（read / slides extract / docx read）、LaTeX 编译、以及增量追加\n"
            "（slides add / docx add，走 python-pptx/docx）均不依赖 pandoc，可照常使用。"
        )


def _pypandoc():
    """导入 pypandoc 并把探测到的 pandoc 路径绑定给它；不可用时抛 PandocNotAvailable。"""
    pandoc_path = settings.find_pandoc()
    if not pandoc_path:
        raise PandocNotAvailable()
    try:
        import pypandoc
    except ImportError as e:  # pragma: no cover
        raise PandocNotAvailable() from e
    # 让 pypandoc 使用我们探测到的二进制（尊重 DOCWRITING_PANDOC 覆盖 / 系统 pandoc）
    os.environ["PYPANDOC_PANDOC"] = pandoc_path
    return pypandoc


def _reference_args(reference_doc: str | Path | None) -> list[str]:
    """把可选的 --reference-doc 模板解析为 pandoc extra_args。"""
    if not reference_doc:
        return []
    ref = Path(reference_doc)
    if not ref.is_file():
        raise FileNotFoundError(f"--reference-doc 模板不存在：{ref}")
    return [f"--reference-doc={ref}"]


def md_to_docx(
    md_path: str | Path,
    out_path: str | Path,
    *,
    reference_doc: str | Path | None = None,
) -> Path:
    """Markdown → .docx（Pandoc）。reference_doc 给定时套用 Word 样式模板。"""
    src = Path(md_path)
    if not src.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在：{src}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pypandoc = _pypandoc()
    pypandoc.convert_file(
        str(src),
        to="docx",
        format="markdown",
        extra_args=_reference_args(reference_doc),
        outputfile=str(out),
    )
    return out


def md_to_pptx(
    md_path: str | Path,
    out_path: str | Path,
    *,
    slide_level: int = 2,
    reference_doc: str | Path | None = None,
) -> Path:
    """Markdown → .pptx（Pandoc）。

    slide_level 决定哪级标题开新页（默认 2：`#` = 标题/分节，`##` = 内容页）。
    演讲者备注用 pandoc 约定的 `::: notes` fenced div。reference_doc 给定时套用
    PowerPoint 母版模板。
    """
    src = Path(md_path)
    if not src.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在：{src}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pypandoc = _pypandoc()
    extra = [f"--slide-level={int(slide_level)}", *_reference_args(reference_doc)]
    pypandoc.convert_file(
        str(src),
        to="pptx",
        format="markdown",
        extra_args=extra,
        outputfile=str(out),
    )
    return out


def docx_to_md(path: str | Path, out_path: str | Path) -> Path:
    """可选：.docx → Markdown（Pandoc）。

    默认 docx 读路径仍走 python-docx（docx_io.docx_to_markdown）；此函数为需要 pandoc
    语义（如更完整的样式/公式映射）时的备用后端。
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"docx 不存在：{src}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pypandoc = _pypandoc()
    pypandoc.convert_file(str(src), to="markdown", format="docx", outputfile=str(out))
    return out
