"""bilingual render —— 中英对照文档渲染链（bilingual.md → PDF，环节 2/3）。

管线：``bilingual.md``（渲染层输入契约，形态由 pipeline 测试锁死）→ 契约预处理 →
pandoc HTML 片段（``--mathjax`` 出 ``\\(..\\)``/``\\[..\\]`` 数学语义定界）→ 自包含
HTML（内联样式与 KaTeX 本地化 bundle，零外部请求）→ playwright 系统浏览器 headless
打印 PDF（A4、0.6cm 四边页边距、双栏、printBackground）→ PyMuPDF 逐页叠加页码
（奇数页右下、偶数页左下，落于下页边距带内）。

预处理三件事（其余逐行透传）：

- 段落边界 ``---`` 换成 thematic break ``***``——pandoc 会把空行后的 ``---`` 当 YAML
  元数据块吞掉（exit 64，2026-10-23 实测）；
- 斜体英文行 ``*...*`` 改写为 ``<em>...</em>`` 原样内联（行内 markdown 由 pandoc 照常
  解析：``$..$`` 数学、``**粗体**`` 都保真；外层斜体由标签保证）；
- 其余文本行做 markdown 转义（``$...$`` 行内数学原样保留）。

HTML 中间文件保留在工作区内作调试图；图片以 ``images/<name>`` 相对路径原样解析。

接缝（2026-11-17 吸收平移时重接，旧 pySciWS ``bilingual_render.py`` 383 行职责不变）：

- **pandoc**：不引 pypandoc（⑤写作域 ADR），subprocess 调 vendored ``pandoc.exe``
  （路径真源 ``pysci.skills.document_writing.toolpaths.pandoc_exe``，跨域复用不复制）；
- **浏览器**：旧 ``browser_fetch._launch`` 耦合剥离为本模块 :func:`launch_browser`
  可配置启动器——显式 exe（环境变量/路径常量）→ 系统 Chrome → Edge → playwright
  自带 chromium；playwright import 延迟到打印时（缺依赖不影响纯函数面）；
- **资产**：bilingual.css + KaTeX 0.19 bundle 随包平移至 ``bilingual/assets/``
  （见 ``assets/katex/SOURCE.txt``），不再经旧 config.module_dir 定位。

版式复杂度热点（测试钉死，勿回退）：宽元素量宽口径 = **min-content**（临时
``width: min-content`` 重排读 rect，公式配 ``nowrap``；历史教训：``scrollWidth``
恒≥屏幕栏宽会把全部公式误判超宽）、判据 = **打印栏宽**（:func:`print_col_width_px`
按 A4/页边距/栏距算出注入 ``__COL_W_PX__``）、超宽 **zoom 等比缩放只缩不放、禁跨栏**
（2026-11-03 用户二轮裁决）；图注粒度 = 图＋**首对**句对同包 ``.figure`` 原子防栏/页
断裂，**余对** ``.caption-cont`` 自由流布（三轮粒度修复定案）。
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import threading
from pathlib import Path

from pysci.skills.document_writing.toolpaths import pandoc_exe

#: 版式资产区（bilingual.css + katex/ 本地化 bundle，随包平移，见其中 SOURCE.txt）。
ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

#: KaTeX 本地化 bundle 目录（katex.min.js / katex.min.css / auto-render.min.js / fonts/）。
KATEX_DIR: Path = ASSETS_DIR / "katex"

#: 页边距（cm，四边；**<1cm 硬线**，2026-11-03 用户裁决「1cm 以下理想」）。
PAGE_MARGIN_CM: float = 0.6

#: 栏距（cm；bilingual.css 的 ``column-gap`` 同值——宽元素判据用）。
COLUMN_GAP_CM: float = 0.7

#: 页码字号（pt，纯阿拉伯数字）。
_PAGE_NUM_PT: float = 8.0

#: A4 纸宽（cm）。
_PAGE_W_CM: float = 21.0

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
#: 之后、下一个 ``---``（hr）之前的句对组：**首对**并入 ``.figure`` 与图同包原子防
#: 栏/页断裂，**余对**包 ``.caption-cont`` 同图注样式自由流布——2026-11-03 三轮粒度
#: 修复：长图注整组原子>栏尾会跳栏首留洞，图＋首对原子即可接续）+ 单栏强制（超栏宽
#: 的展示公式/表格**等比缩放至栏宽**，只缩不放、禁跨栏——2026-11-03 用户二轮裁决：
#: 删除 column-span 跨栏特殊处理）。
#:
#: 栏宽判据用**打印栏宽**（``__COL_W_PX__``，由 :func:`build_html` 按 A4/页边距/栏距
#: 算出后替换）——脚本在屏幕布局下执行，屏幕栏宽≠打印栏宽，用屏幕量宽会漏判。
#: 量宽口径=元素 **min-content**（临时 ``width: min-content`` 重排后读 rect）：表格
#: 按「折行后仍放不下的最小宽度」判（不误伤可折行表格），公式配 ``nowrap`` 取自然宽
#: （KaTeX 内联块间存在折点，nowrap 防 min-content 低估）；对打印栏宽比较。
#: ⚠ 历史教训：早先量宽曾用 ``el.scrollWidth``——块级元素 scrollWidth 恒≥屏幕栏宽
#: （626px > 打印栏宽 361px），**全部公式被误判超宽**而加 ``.wide`` 跨栏，即用户
#: 「大量的公式切断了双栏」事故的根因（2026-11-03 实测 13/13 误标）。
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
    var fig = document.createElement("div");
    fig.className = "figure";
    var wrap = document.createElement("div");
    wrap.className = "caption";
    p.parentNode.insertBefore(fig, p);
    fig.appendChild(p);
    fig.appendChild(wrap);
    var cont = null;
    while (n && n.tagName !== "HR" && !n.querySelector("img") && !n.querySelector(".math.display")) {
      var next = n.nextElementSibling;
      if (!wrap.children.length) {
        wrap.appendChild(n);
      } else {
        if (!cont) {
          cont = document.createElement("div");
          cont.className = "caption-cont";
        }
        cont.appendChild(n);
      }
      n = next;
    }
    if (cont) fig.parentNode.insertBefore(cont, fig.nextSibling);
  });
  function measureMin(el, nowrap) {
    var w0 = el.style.width, m0 = el.style.maxWidth, s0 = el.style.whiteSpace;
    el.style.width = "min-content";
    el.style.maxWidth = "none";
    if (nowrap) el.style.whiteSpace = "nowrap";
    var w = el.getBoundingClientRect().width;
    el.style.width = w0; el.style.maxWidth = m0; el.style.whiteSpace = s0;
    return w;
  }
  var colW = __COL_W_PX__;
  Array.prototype.slice.call(
    document.querySelectorAll(".katex-display, table")
  ).forEach(function (el) {
    var w = measureMin(el, el.classList.contains("katex-display"));
    if (w <= colW + 1) return;
    el.style.zoom = String(colW / w);
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


def print_col_width_px(
    page_w_cm: float = _PAGE_W_CM,
    margin_cm: float = PAGE_MARGIN_CM,
    gap_cm: float = COLUMN_GAP_CM,
) -> float:
    """打印栏宽（px）：(纸宽 − 2×页边距 − 栏距) / 2——宽元素判据的唯一真源。

    屏幕布局下脚本量到的是屏幕栏宽，判据必须用本函数的打印栏宽（注入
    ``__COL_W_PX__``）；口径 = CSS 参考像素（1in = 96px）。
    """
    return (page_w_cm - 2 * margin_cm - gap_cm) / 2 / 2.54 * 96.0


def _escape_script(js: str) -> str:
    """``</script`` 字面量转义（只可能出现在 JS 字符串里），防提前闭合脚本标签。"""
    return js.replace("</script", "<\\/script")


def _pandoc_fragment(md_text: str) -> str:
    """markdown → HTML 正文片段（pandoc ``--mathjax``；subprocess 薄胶水）。

    Raises:
        RuntimeError: pandoc 不可用或非零退出。
    """
    cmd = [pandoc_exe(), "-f", "markdown", "-t", "html", "--mathjax"]
    try:
        result = subprocess.run(
            cmd,
            input=md_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"pandoc 调用失败：{exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"pandoc 失败（exit {result.returncode}）：{result.stderr.strip()}"
        )
    return result.stdout


def build_html(md_text: str, *, title: str = "bilingual") -> str:
    """契约形态 bilingual.md → 自包含 HTML（pandoc 正文 + 内联 CSS/JS/KaTeX 字体）。

    Args:
        md_text: ``bilingual.md`` 全文。
        title: ``<title>`` 文本。

    Returns:
        完整 HTML 文本；图片仍以 ``images/<name>`` 相对路径解析（与 HTML 同区）。
    """
    body = _pandoc_fragment(to_pandoc_md(md_text))
    css = (ASSETS_DIR / "bilingual.css").read_text(encoding="utf-8")
    katex_js = (KATEX_DIR / "katex.min.js").read_text(encoding="utf-8")
    auto_js = (KATEX_DIR / "auto-render.min.js").read_text(encoding="utf-8")
    # 打印栏宽（px）：宽元素判据的唯一真源（见 print_col_width_px）。
    col_w_px = print_col_width_px()
    init_js = _INIT_JS.replace("__COL_W_PX__", f"{col_w_px:.2f}")
    return (
        '<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>\n{css}\n</style>\n"
        f"<style>\n{_inline_katex_css()}\n</style>\n</head>\n<body>\n{body}\n"
        f"<script>{_escape_script(katex_js)}</script>\n"
        f"<script>{_escape_script(auto_js)}</script>\n"
        f"<script>{init_js}</script>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# PDF 打印（playwright 系统浏览器 headless）
# ---------------------------------------------------------------------------
def add_page_numbers(
    pdf_path: str | Path, *, margin_cm: float = PAGE_MARGIN_CM
) -> Path:
    """给 PDF 逐页叠加页码：**奇数页右下、偶数页左下**（1 起算、每页、纯数字）。

    页码约 8pt，落于下页边距带内（``[页高-margin, 页高]``）——不与正文/图片重叠。
    PyMuPDF 后处理，不占浏览器打印正文栏。

    Args:
        pdf_path: 目标 PDF（原地覆盖）。
        margin_cm: 页边距（cm），页码带据此定位。

    Returns:
        同一 PDF 路径（便于链式调用）。

    Raises:
        FileNotFoundError: PDF 不存在。
    """
    import pymupdf as fitz

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")
    m_pt = margin_cm * 72.0 / 2.54
    doc = fitz.open(str(pdf_path))
    tmp = pdf_path.with_suffix(".numbered.pdf")
    try:
        for i, page in enumerate(doc):
            n = i + 1
            text = str(n)
            tw = fitz.get_text_length(text, fontname="helv", fontsize=_PAGE_NUM_PT)
            # 竖向居中于下页边距带（带顶留 2pt 余量不贴正文边界）；横按奇偶右下/左下。
            baseline_y = page.rect.height - m_pt / 2.0 + _PAGE_NUM_PT * 0.3
            x = page.rect.width - m_pt - tw if n % 2 else m_pt
            page.insert_text(
                (x, baseline_y),
                text,
                fontsize=_PAGE_NUM_PT,
                fontname="helv",
                color=(0, 0, 0),
            )
        doc.save(str(tmp))
    finally:
        doc.close()
    tmp.replace(pdf_path)
    return pdf_path


class BrowserNotAvailableError(RuntimeError):
    """系统 Chrome/Edge 与 playwright 自带 chromium 均不可用。"""


#: 显式浏览器可执行文件覆盖（环境变量值 = exe 绝对路径），优先于一切探测。
BROWSER_EXE_ENV = "PYSCI_BROWSER_EXE"

#: 浏览器通道覆盖（环境变量）：``auto`` | ``chrome`` | ``msedge`` | ``chromium``。
BROWSER_CHANNEL_ENV = "PYSCI_BROWSER_CHANNEL"

#: 系统 Chrome 标准安装位（channel 探测之外的显式路径候选，Windows）。
CHROME_EXE_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

#: 系统 Edge 标准安装位（Windows）。
EDGE_EXE_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def _exe_candidates() -> list[tuple[str, str]]:
    """显式 exe 候选：环境变量覆盖优先，其次系统 Chrome/Edge 标准安装位（存在才列）。"""
    out: list[tuple[str, str]] = []
    env = os.environ.get(BROWSER_EXE_ENV, "").strip()
    if env:
        out.append((f"env:{BROWSER_EXE_ENV}", env))
    for exe in (*CHROME_EXE_CANDIDATES, *EDGE_EXE_CANDIDATES):
        if Path(exe).is_file():
            out.append(("exe", exe))
    return out


def launch_browser(pw: object, channel: str = "auto", headless: bool = True):
    """可配置浏览器启动器（旧 ``browser_fetch._launch`` 剥离重写）。

    探测顺序（逐项失败自动下探）：显式 exe（:data:`BROWSER_EXE_ENV` → 系统
    Chrome/Edge 路径常量）→ playwright channel（chrome → msedge）→ playwright
    自带 chromium（``playwright install chromium``）。

    Args:
        pw: ``playwright.sync_api`` 实例（``sync_playwright()`` 上下文产物）。
        channel: ``auto`` | ``chrome`` | ``msedge`` | ``chromium``；环境变量
            :data:`BROWSER_CHANNEL_ENV` 可覆盖缺省 ``auto``。
        headless: 无头模式（打印链固定 True）。

    Returns:
        ``(browser, 标签)``——标签用于日志/测试断言（如 ``exe``/``chrome``/
        ``chromium(bundled)``）。

    Raises:
        BrowserNotAvailableError: 所有候选均启动失败。
    """
    channel = channel or os.environ.get(BROWSER_CHANNEL_ENV, "").strip() or "auto"
    if channel == "chromium":
        chans: list[str] = []
    elif channel != "auto":
        chans = [channel]
    else:
        chans = ["chrome", "msedge"]
    tried: list[str] = []
    for label, exe in _exe_candidates():
        try:
            return pw.chromium.launch(headless=headless, executable_path=exe), label
        except Exception as exc:  # noqa: BLE001 —— 启动失败逐项下探
            tried.append(f"{label}({exe}): {exc}")
    for ch in chans:
        try:
            return pw.chromium.launch(channel=ch, headless=headless), ch
        except Exception as exc:  # noqa: BLE001
            tried.append(f"channel={ch}: {exc}")
    try:
        return pw.chromium.launch(headless=headless), "chromium(bundled)"
    except Exception as exc:  # noqa: BLE001
        tried.append(f"bundled-chromium: {exc}")
    raise BrowserNotAvailableError(
        "浏览器启动失败：\n  "
        + "\n  ".join(tried)
        + "\n提示：确认本机已安装 Chrome/Edge，或运行 `playwright install chromium`；"
        f"也可用 {BROWSER_EXE_ENV} 显式指定可执行文件。"
    )


#: 打印串行化锁：本机仅 CPU/内存 ~5GB，浏览器打印**避免并行多实例**——进程内互斥，
#: 跨进程请由调用方（CLI 状态机）保证串行。
_PRINT_LOCK = threading.Lock()


def render_pdf(
    md_path: str | Path,
    pdf_path: str | Path,
    *,
    keep_html: bool = True,
    channel: str = "auto",
) -> Path:
    """把契约形态的 ``bilingual.md`` 渲染为可打印 PDF。

    A4、:data:`PAGE_MARGIN_CM` 四边页边距（<1cm）、双栏（样式真源 bilingual.css）、
    printBackground；渲染后 :func:`add_page_numbers` 逐页叠加奇偶页码。
    浏览器走 :func:`launch_browser` 探测链；HTML 中间文件默认保留在 md 同目录作调试图。

    Args:
        md_path: 源 ``bilingual.md`` 路径。
        pdf_path: 输出 PDF 路径。
        keep_html: 渲染后是否保留 ``.html`` 中间文件（默认保留）。
        channel: 浏览器通道（透传 :func:`launch_browser`）。

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

    try:
        with _PRINT_LOCK, sync_playwright() as p:
            browser, _used = launch_browser(p, channel, True)
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
                        "top": f"{PAGE_MARGIN_CM}cm",
                        "bottom": f"{PAGE_MARGIN_CM}cm",
                        "left": f"{PAGE_MARGIN_CM}cm",
                        "right": f"{PAGE_MARGIN_CM}cm",
                    },
                )
            finally:
                browser.close()
    except BrowserNotAvailableError as exc:
        raise RuntimeError(f"浏览器不可用，无法打印 PDF：{exc}") from exc
    if not keep_html:
        html_path.unlink(missing_ok=True)
    return add_page_numbers(pdf_path)
