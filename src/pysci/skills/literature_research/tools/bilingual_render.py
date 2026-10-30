"""bilingual render —— 中英对照文档渲染层（bilingual.md → PDF，三环节之 2/3）。

管线：``bilingual.md``（格式契约见 cli 任务书 §二.5，形态已由 pipeline 测试锁死）→
契约预处理 → pandoc HTML 片段（``--mathjax`` 出 ``\\(..\\)``/``\\[..\\]`` 数学语义定界）
→ 自包含 HTML（内联样式与 KaTeX 本地化 bundle）→ playwright 系统 Chrome headless
打印 PDF（A4、常规页边距、printBackground）。

预处理三件事（其余逐行透传）：

- 段落边界 ``---`` 换成 thematic break ``***``——pandoc 会把空行后的 ``---`` 当 YAML
  元数据块吞掉（exit 64，2026-10-23 实测）；
- 斜体英文行 ``*...*`` 改写为 ``<em>...</em>`` 原样内联（行内 markdown 由 pandoc 照常
  解析：``$..$`` 数学、``**粗体**`` 都保真；外层斜体由标签保证）；
- 其余文本行做 markdown 转义（``$...$`` 行内数学原样保留）。

HTML 中间文件保留在工作区内作调试图；图片以 ``images/<name>`` 相对路径原样解析。
KaTeX bundle 与样式模板在 ``data/skills/literature_research/templates/``（含版本来源）。
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pypandoc

from .config import settings

#: 模板区（bilingual.css + katex/ 本地化 bundle，见其中 SOURCE.txt）。
TEMPLATE_DIR: Path = settings.module_dir / "templates"

#: KaTeX 本地化 bundle 目录（katex.min.js / katex.min.css / auto-render.min.js / fonts/）。
KATEX_DIR: Path = TEMPLATE_DIR / "katex"

#: 行内数学（``$...$``）——单行内成对，内容不含换行/美元符。
_INLINE_MATH_RE = re.compile(r"\$[^$\n]+\$")

#: 标题行（保留源 ``#`` 层级）。
_HEADING_RE = re.compile(r"^(#{1,6}) (.*)$")

#: 图片引用行（透传）。
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)$")

#: 斜体英文行（格式契约：``*...*`` 且全行包裹）。
_EN_LINE_RE = re.compile(r"^\*.*\*$")

#: markdown 转义字符（文本行内防误解析；``$...$`` 数学段另行保护）。
_MD_SPECIAL_RE = re.compile(r"([\\`*_\[\]<>])")


def _split_inline_math(text: str) -> list[tuple[bool, str]]:
    """按 ``$...$`` 把一行切成 ``(是否数学, 片段)`` 序列。"""
    out: list[tuple[bool, str]] = []
    pos = 0
    for m in _INLINE_MATH_RE.finditer(text):
        if m.start() > pos:
            out.append((False, text[pos : m.start()]))
        out.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((False, text[pos:]))
    return out or [(False, text)]


def _escape_md_text(text: str) -> str:
    """文本行转义为 pandoc 字面量（``$...$`` 数学段原样保留）。"""
    parts: list[str] = []
    for is_math, seg in _split_inline_math(text):
        if is_math:
            parts.append(seg)
        else:
            seg = seg.replace("&", "&amp;")
            parts.append(_MD_SPECIAL_RE.sub(r"\\\1", seg))
    return "".join(parts)


def to_pandoc_md(md_text: str) -> str:
    """把契约形态的 bilingual.md 预处理成 pandoc 友好的 markdown。

    见模块 docstring「预处理三件事」。公式 ``$$`` 块与 ``<table>`` 块整块透传。

    Args:
        md_text: ``bilingual.md`` 全文（契约形态）。

    Returns:
        可直接喂 pandoc 的 markdown 文本。
    """
    out: list[str] = []
    in_verbatim = False
    for line in md_text.split("\n"):
        stripped = line.strip()
        if in_verbatim:
            out.append(line)
            if stripped == "$$" or "</table>" in stripped:
                in_verbatim = False
                out.append("")
            continue
        if stripped.startswith("$$") or stripped.startswith("<table"):
            if out and out[-1].strip():
                out.append("")
            out.append(line)
            if stripped.startswith("$$"):
                in_verbatim = line.count("$$") < 2  # 多行 $$ 块整块透传到收尾 $$
            else:
                in_verbatim = "</table>" not in stripped
            if not in_verbatim:
                out.append("")
            continue
        if stripped == "---" or stripped == "***":
            if out and out[-1].strip():
                out.append("")
            out.append("***")
            continue
        m_head = _HEADING_RE.match(line)
        if m_head:
            out.append(f"{m_head.group(1)} {_escape_md_text(m_head.group(2))}")
            continue
        if _IMAGE_RE.match(stripped):
            out.append(stripped)
            continue
        if _EN_LINE_RE.match(stripped):
            out.append(f"<em>{stripped[1:-1]}</em>")
            continue
        out.append(_escape_md_text(line) if stripped else "")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 自包含 HTML 组装（内联样式 + KaTeX 本地化 bundle）
# ---------------------------------------------------------------------------
#: 页内收尾脚本：KaTeX 排版（pandoc --mathjax 的语义定界符）+ 图注归组（图片段落
#: 之后、下一个 ``---``（hr）之前的句对组 → ``.caption`` 小字）。
_INIT_JS = """
document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      {left: "\\\\[", right: "\\\\]", display: true},
      {left: "\\\\(", right: "\\\\)", display: false}
    ],
    throwOnError: false
  });
  Array.prototype.slice.call(document.querySelectorAll("p")).forEach(function (p) {
    if (!p.querySelector("img")) return;
    var n = p.nextElementSibling;
    var wrap = document.createElement("div");
    wrap.className = "caption";
    p.after(wrap);
    while (n && n.tagName !== "HR" && !n.querySelector("img") && !n.querySelector(".math.display")) {
      var next = n.nextElementSibling;
      wrap.appendChild(n);
      n = next;
    }
  });
});
"""

#: katex.min.css 里非 woff2 字体源（woff/ttf）一律剔除：打印通道只用 woff2（已内联）。
_NON_WOFF2_RE = re.compile(r",\s*url\([^)]+\.(?:woff|ttf)\)")
_WOFF2_URL_RE = re.compile(r'url\("?(fonts/[^")]+\.woff2)"?\)')


def _inline_katex_css() -> str:
    """katex.min.css 内联化：剔除非 woff2 源，``fonts/*.woff2`` 换 data: URI。

    字体以 data: URI 内联后无跨源字体加载问题（file:// 页面零外部请求）。
    """
    css = (KATEX_DIR / "katex.min.css").read_text(encoding="utf-8")
    css = _NON_WOFF2_RE.sub("", css)
    cache: dict[str, str] = {}

    def repl(m: re.Match[str]) -> str:
        name = m.group(1).rsplit("/", 1)[-1]
        if name not in cache:
            data = (KATEX_DIR / "fonts" / name).read_bytes()
            cache[name] = "data:font/woff2;base64," + base64.b64encode(data).decode()
        return f'url("{cache[name]}")'

    return _WOFF2_URL_RE.sub(repl, css)


def build_html(md_text: str, *, title: str = "bilingual") -> str:
    """契约形态 bilingual.md → 自包含 HTML（pandoc 正文 + 内联 CSS/JS/KaTeX 字体）。

    Args:
        md_text: ``bilingual.md`` 全文。
        title: ``<title>`` 文本。

    Returns:
        完整 HTML 文本；图片仍以 ``images/<name>`` 相对路径解析（与 HTML 同区）。
    """
    body = pypandoc.convert_text(
        to_pandoc_md(md_text), "html", format="markdown", extra_args=["--mathjax"]
    )
    css = (TEMPLATE_DIR / "bilingual.css").read_text(encoding="utf-8")
    katex_js = (KATEX_DIR / "katex.min.js").read_text(encoding="utf-8")
    auto_js = (KATEX_DIR / "auto-render.min.js").read_text(encoding="utf-8")
    # </script 字面量只可能出现在 JS 字符串里，转义防止提前闭合脚本标签。
    esc = lambda s: s.replace("</script", "<\\/script")  # noqa: E731
    return (
        '<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>\n{css}\n</style>\n"
        f"<style>\n{_inline_katex_css()}\n</style>\n</head>\n<body>\n{body}\n"
        f"<script>{esc(katex_js)}</script>\n<script>{esc(auto_js)}</script>\n"
        f"<script>{_INIT_JS}</script>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# PDF 打印（playwright 系统 Chrome headless）
# ---------------------------------------------------------------------------
def render_pdf(
    md_path: str | Path, pdf_path: str | Path, *, keep_html: bool = True
) -> Path:
    """把契约形态的 ``bilingual.md`` 渲染为可打印 PDF（A4、2cm 页边距、printBackground）。

    浏览器通道回落复用 :func:`browser_fetch._launch`（系统 chrome → msedge →
    bundled chromium）；HTML 中间文件默认保留在 md 同目录作调试图。

    Args:
        md_path: 源 ``bilingual.md`` 路径。
        pdf_path: 输出 PDF 路径。
        keep_html: 渲染后是否保留 ``.html`` 中间文件（默认保留）。

    Returns:
        写出的 PDF 路径。

    Raises:
        FileNotFoundError: 源 ``bilingual.md`` 不存在。
        RuntimeError: 浏览器不可用或渲染失败。
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)
    if not md_path.is_file():
        raise FileNotFoundError(f"bilingual.md 不存在：{md_path}")
    html_path = md_path.with_suffix(".html")
    html_path.write_text(
        build_html(md_path.read_text(encoding="utf-8"), title=md_path.parent.name),
        encoding="utf-8",
    )
    from playwright.sync_api import sync_playwright

    from .browser_fetch import BrowserNotAvailable, _launch

    try:
        with sync_playwright() as p:
            browser, _channel = _launch(p, "auto", True)
            try:
                page = browser.new_page()
                page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
                page.wait_for_function(
                    "document.fonts ? document.fonts.status === 'loaded' : true"
                )
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "2cm",
                        "bottom": "2cm",
                        "left": "2cm",
                        "right": "2cm",
                    },
                )
            finally:
                browser.close()
    except BrowserNotAvailable as exc:
        raise RuntimeError(f"浏览器不可用，无法打印 PDF：{exc}") from exc
    if not keep_html:
        html_path.unlink(missing_ok=True)
    return pdf_path
