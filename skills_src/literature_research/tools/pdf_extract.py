"""PDF → Markdown 抽取封装。

策略（默认 auto 模式，面向“精读需看公式”的场景）：

**后端优先级**：
1. **mineru-cloud** — MinerU 云端 Open API（vlm 模型，公式→LaTeX、表格质量最好，
   无需本地 GPU）。走 PDF 路径通常就是为了精读看公式，故云端为**首选**。
2. **pymupdf4llm** — 纯本地、极快（<10s/篇），但**公式会乱码**，仅在云端不可用时
   兜底，且会**显式打印 WARNING**（绝不静默回退）。
3. **arXiv LaTeX source** — 若能拿到源码，直接从 .tex 提取（最准，公式原生 LaTeX）。

调用方式::

    from skills_src.literature_research.tools.pdf_extract import extract_pdf

    md = extract_pdf(Path("path/to/paper.pdf"))                   # auto：云端优先
    md = extract_pdf(pdf_path, backend="mineru-cloud")            # 强制云端
    md = extract_pdf(pdf_path, backend="pymupdf4llm")             # 强制本地兜底
    md = extract_pdf(pdf_path, prefer_latex_source="2606.12345")  # 优先 arXiv 源码

结果会缓存到 ``skills_src/literature_research/cache/extracted/{stem}.{backend}.md``，重复调用直接命中缓存。

云端 MinerU 需在项目根 ``.env`` 配置 ``MINERU_TOKEN=``（在 https://mineru.net/apiManage/token
免费申请，Token 有效期约 90 天）。未配置时 auto 会回退 pymupdf4llm 并告警。

依赖：仅需 ``pymupdf4llm``（本地兜底）+ ``requests``（云端调用）。已移除 marker /
本地 magic-pdf（实测本机不可用，云端可完全替代）。
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable

from .config import settings
from .cache_manager import bump_mtime

# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class NoBackendAvailable(RuntimeError):
    pass


class ExtractionFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 后端探测
# ---------------------------------------------------------------------------
def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# auto 模式的后端优先级：云端优先（公式质量最好），本地 pymupdf4llm 兜底
PREFERRED_ORDER: tuple[str, ...] = ("mineru-cloud", "pymupdf4llm")


def _mineru_cloud_ready() -> bool:
    """云端 MinerU 是否可用：需配置 MINERU_TOKEN 且 requests 可导入。"""
    return bool(settings.mineru_token) and _has_module("requests")


def available_backends() -> list[str]:
    """列出当前环境实际可用的后端。"""
    backends: list[str] = []
    if _mineru_cloud_ready():
        backends.append("mineru-cloud")
    if _has_module("pymupdf4llm") or _has_module("fitz"):
        backends.append("pymupdf4llm")
    return backends


def _pick_backend(requested: str) -> str:
    """根据用户请求与实际可用性选择后端。"""
    avail = available_backends()
    if not avail:
        raise NoBackendAvailable(
            "No PDF extraction backend available.\n"
            "  - 云端 MinerU：在 .env 配置 MINERU_TOKEN（https://mineru.net/apiManage/token 免费申请）\n"
            "  - 本地兜底：uv add pymupdf4llm"
        )

    if requested == "auto":
        for pref in PREFERRED_ORDER:
            if pref in avail:
                return pref
        return avail[0]

    if requested not in avail:
        print(
            f"[pdf_extract] 请求的后端 '{requested}' 不可用，回退到 auto",
            file=sys.stderr,
        )
        return _pick_backend("auto")

    return requested


# ---------------------------------------------------------------------------
# 缓存工具
# ---------------------------------------------------------------------------
def _cache_path(pdf_path: Path, backend: str) -> Path:
    """缓存文件名 = PDF 的 sha1 前 12 位 + backend + .md"""
    h = hashlib.sha1()
    h.update(str(pdf_path.resolve()).encode("utf-8"))
    try:
        h.update(str(pdf_path.stat().st_size).encode())
        h.update(str(int(pdf_path.stat().st_mtime)).encode())
    except OSError:
        pass
    key = h.hexdigest()[:12]
    return settings.cache_extracted / f"{pdf_path.stem}_{key}.{backend}.md"


def _read_cache(path: Path) -> str | None:
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        bump_mtime(path)  # 命中 touch，使 mtime≈最近访问（供 prune LRU）
        return text
    return None


def _write_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 后端实现
# ---------------------------------------------------------------------------
# MinerU 云端 Open API 配置（本地 PDF 走“批量上传”流程）
MINERU_API_BASE = "https://mineru.net"
MINERU_MODEL_VERSION = "vlm"      # vlm：公式/表格质量最好（推荐）
MINERU_IS_OCR = True              # 强制 OCR：确保图片/扫描公式也被识别（数字版可改 False）
MINERU_POLL_INTERVAL = 8          # 轮询间隔（秒）
MINERU_POLL_TIMEOUT = 900         # 轮询总超时（秒）


def _extract_with_mineru_cloud(pdf_path: Path) -> str:
    """MinerU 云端 Open API 后端（精准解析，公式→LaTeX，无需本地 GPU）。

    本地 PDF 无法直接给单文件接口（其只收公网 URL），故走“批量上传”流程：
      1. POST /api/v4/file-urls/batch  申请预签名上传链接
      2. PUT  上传 PDF 字节（官方要求：不带 Content-Type）
      3. 轮询 GET /api/v4/extract-results/batch/{batch_id} 直到 state=done
      4. 下载 full_zip_url，解压取 full.md
    """
    token = settings.mineru_token
    if not token:
        raise ExtractionFailed("MinerU 云端未配置：请在项目根 .env 设置 MINERU_TOKEN")
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise ExtractionFailed(f"requests 不可用（云端 MinerU 需要）：{e}") from e

    base = MINERU_API_BASE.rstrip("/")
    auth = {"Authorization": f"Bearer {token}"}
    name = pdf_path.name or "document.pdf"

    # 1. 申请上传链接
    payload = {
        "files": [{"name": name, "is_ocr": MINERU_IS_OCR}],
        "model_version": MINERU_MODEL_VERSION,
        "enable_formula": True,
        "enable_table": True,
        "language": "en",
        "extra_formats": ["latex"],   # 额外产出 LaTeX，便于精读公式
    }
    r = requests.post(
        f"{base}/api/v4/file-urls/batch",
        headers={**auth, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise ExtractionFailed(f"MinerU 申请上传链接失败: {j.get('msg')}")
    data = j.get("data") or {}
    batch_id = data.get("batch_id")
    file_urls = data.get("file_urls") or []
    if not batch_id or not file_urls:
        raise ExtractionFailed(f"MinerU 未返回 batch_id/file_urls: {j}")

    # 2. 上传 PDF 字节（官方要求不带 Content-Type）
    with pdf_path.open("rb") as f:
        up = requests.put(file_urls[0], data=f, timeout=600)
    if up.status_code not in (200, 201):
        raise ExtractionFailed(f"MinerU 上传失败: HTTP {up.status_code} {up.text[:200]}")

    # 3. 轮询解析结果
    poll_url = f"{base}/api/v4/extract-results/batch/{batch_id}"
    deadline = time.time() + MINERU_POLL_TIMEOUT
    full_zip_url: str | None = None
    while time.time() < deadline:
        time.sleep(MINERU_POLL_INTERVAL)
        pr = requests.get(poll_url, headers=auth, timeout=60)
        if pr.status_code != 200:
            continue
        pj = pr.json()
        if pj.get("code") != 0:
            continue
        results = (pj.get("data") or {}).get("extract_result") or []
        if not results:
            continue
        item = results[0]
        state = item.get("state")
        if state == "done":
            full_zip_url = item.get("full_zip_url")
            break
        if state == "failed":
            raise ExtractionFailed(f"MinerU 解析失败: {item.get('err_msg')}")
        prog = item.get("extract_progress") or {}
        if prog.get("total_pages"):
            print(
                f"[pdf_extract] MinerU {state}: "
                f"{prog.get('extracted_pages')}/{prog.get('total_pages')} 页"
            )
    if not full_zip_url:
        raise ExtractionFailed(f"MinerU 轮询超时（>{MINERU_POLL_TIMEOUT}s），未取得结果")

    # 4. 下载 zip，解压取 full.md
    zr = requests.get(full_zip_url, timeout=300)
    zr.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
        names = zf.namelist()
        target = (
            next((n for n in names if n.endswith("full.md")), None)
            or next((n for n in names if n.endswith(".md")), None)
        )
        if target is None:
            raise ExtractionFailed(f"MinerU 结果 zip 内未找到 .md：{names[:10]}")
        return zf.read(target).decode("utf-8", errors="replace")


def _extract_with_pymupdf4llm(pdf_path: Path) -> str:
    """pymupdf4llm 后端：速度最快，公式丢失但文本准确。"""
    try:
        import pymupdf4llm  # type: ignore
        return pymupdf4llm.to_markdown(str(pdf_path))
    except ImportError:
        pass

    # 回退：pymupdf (fitz) + 手写 markdown 化
    try:
        import fitz  # type: ignore
    except ImportError as e:
        raise ExtractionFailed(f"pymupdf4llm/fitz not installed: {e}") from e

    parts: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text")
            parts.append(f"\n\n<!-- page {i} -->\n\n{text}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# arXiv LaTeX 源码路径（物理领域独有优势）
# ---------------------------------------------------------------------------
def _extract_from_arxiv_source(arxiv_id: str) -> str | None:
    """若指定了 arxiv_id，尝试下载源码并从主 .tex 提取正文。

    返回 markdown-like 字符串；失败返回 None。
    """
    try:
        from . import arxiv_client
    except ImportError:
        return None

    tar_path = arxiv_client.download_source(arxiv_id)
    if not tar_path:
        return None
    tex = arxiv_client.extract_tex_from_source(tar_path)
    if not tex:
        return None
    return _tex_to_markdown(tex)


def _tex_to_markdown(tex: str) -> str:
    """极简 LaTeX → Markdown（保留公式、章节、引用）。

    对物理论文，公式应保留为原始 LaTeX，因为 markdown 渲染器（Obsidian、Typora、
    GitHub）都支持 $...$ 与 $$...$$ 语法。
    """
    # 去掉 preamble
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        body = tex[m.end():]
    else:
        body = tex
    m = re.search(r"\\end\{document\}", body)
    if m:
        body = body[:m.start()]

    # 章节标题
    body = re.sub(r"\\section\*?\{(.+?)\}", r"\n## \1\n", body)
    body = re.sub(r"\\subsection\*?\{(.+?)\}", r"\n### \1\n", body)
    body = re.sub(r"\\subsubsection\*?\{(.+?)\}", r"\n#### \1\n", body)
    body = re.sub(r"\\paragraph\*?\{(.+?)\}", r"\n**\1** ", body)

    # 摘要
    body = re.sub(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", r"\n> **Abstract:**\n> \1\n", body, flags=re.S)

    # 图表 caption
    body = re.sub(r"\\caption\{(.+?)\}", r"\n_Figure/Table caption: \1_\n", body, flags=re.S)

    # itemize / enumerate → markdown list
    body = re.sub(r"\\begin\{itemize\}(.+?)\\end\{itemize\}", lambda m: _tex_items_to_md(m.group(1), ordered=False), body, flags=re.S)
    body = re.sub(r"\\begin\{enumerate\}(.+?)\\end\{enumerate\}", lambda m: _tex_items_to_md(m.group(1), ordered=True), body, flags=re.S)

    # \item → -
    body = re.sub(r"\\item\s*", "- ", body)

    # 引用与标签
    body = re.sub(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", r"[\1]", body)
    body = re.sub(r"\\ref\{([^}]+)\}", r"§\1", body)
    body = re.sub(r"\\eqref\{([^}]+)\}", r"(§\1)", body)
    body = re.sub(r"\\label\{[^}]+\}", "", body)

    # 文本样式
    body = re.sub(r"\\textbf\{(.+?)\}", r"**\1**", body)
    body = re.sub(r"\\textit\{(.+?)\}|\\emph\{(.+?)\}", lambda m: f"*{m.group(1) or m.group(2)}*", body)
    body = re.sub(r"\\texttt\{(.+?)\}", r"`\1`", body)

    # 保留 display math（$$...$$ 或 \[...\]）
    body = re.sub(r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}", r"\n$$\1\n$$\n", body, flags=re.S)
    body = re.sub(r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}", r"\n$$\n\\begin{aligned}\1\\end{aligned}\n$$\n", body, flags=re.S)
    body = re.sub(r"\\begin\{gather\*?\}(.+?)\\end\{gather\*?\}", r"\n$$\1$$\n", body, flags=re.S)
    body = re.sub(r"\\\[(.+?)\\\]", r"\n$$\1$$\n", body, flags=re.S)
    body = re.sub(r"\\\((.+?)\\\)", r"$\1$", body, flags=re.S)

    # 移除剩余 LaTeX 命令（保守：只移除已知的排版命令）
    body = re.sub(r"\\(?:noindent|par|clearpage|newpage|medskip|smallskip|bigskip|vspace\*?\{[^}]*\}|hspace\*?\{[^}]*\})", "", body)

    # 表格保留原样（markdown 表格与 LaTeX 表格差异太大，不做转换）

    # 清理多余空行
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _tex_items_to_md(content: str, ordered: bool) -> str:
    items = re.split(r"\\item\s*", content)
    items = [i.strip() for i in items if i.strip()]
    if ordered:
        return "\n" + "\n".join(f"{i+1}. {it}" for i, it in enumerate(items)) + "\n"
    return "\n" + "\n".join(f"- {it}" for it in items) + "\n"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
_BACKEND_IMPL: dict[str, Callable[[Path], str]] = {
    "mineru-cloud": _extract_with_mineru_cloud,
    "pymupdf4llm": _extract_with_pymupdf4llm,
}


def extract_pdf(
    pdf_path: Path | str,
    *,
    backend: str | None = None,
    use_cache: bool = True,
    prefer_latex_source: str | None = None,
    write_cache: bool = True,
) -> str:
    """抽取 PDF 为 Markdown。

    Args:
        pdf_path: PDF 文件路径
        backend: 'mineru-cloud' | 'pymupdf4llm' | 'auto' | None
                 None 表示读 settings.pdf_extract_backend；auto = 云端优先、本地兜底
        use_cache: 是否复用之前的抽取结果
        prefer_latex_source: 若给出 arXiv ID，优先尝试从 LaTeX 源码提取（比 PDF 解析更准）
        write_cache: 是否写入缓存

    Returns:
        markdown 字符串

    Raises:
        NoBackendAvailable: 没有任何可用后端
        ExtractionFailed: 所有尝试的后端都失败
        FileNotFoundError: PDF 不存在
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 1. 优先 arXiv LaTeX 源码
    if prefer_latex_source:
        print(f"[pdf_extract] Trying arXiv LaTeX source for {prefer_latex_source} ...")
        tex_md = _extract_from_arxiv_source(prefer_latex_source)
        if tex_md:
            print(f"[pdf_extract] LaTeX source extracted ({len(tex_md)} chars)")
            if write_cache:
                _write_cache(_cache_path(pdf_path, "arxiv_tex"), tex_md)
            return tex_md
        print("[pdf_extract] LaTeX source unavailable, falling back to PDF extraction")

    # 2. 决定后端
    requested = backend or settings.pdf_extract_backend or "auto"

    # 3. 检查缓存
    if use_cache:
        # 若指定 backend，只查该 backend 的缓存
        if requested != "auto":
            cp = _cache_path(pdf_path, requested)
            cached = _read_cache(cp)
            if cached:
                print(f"[pdf_extract] Cache hit: {cp}")
                return cached
        else:
            # auto 模式：按优先级查所有 backend 缓存
            for pref in ("arxiv_tex", "mineru-cloud", "pymupdf4llm"):
                cp = _cache_path(pdf_path, pref)
                cached = _read_cache(cp)
                if cached:
                    print(f"[pdf_extract] Cache hit: {cp}")
                    return cached

    # 4. 执行抽取（auto：云端优先，本地 pymupdf4llm 兜底且显式告警）
    backends_to_try: list[str]
    if requested == "auto":
        avail = available_backends()
        backends_to_try = [b for b in PREFERRED_ORDER if b in avail]
        if not backends_to_try:
            raise NoBackendAvailable(
                "No PDF extraction backend available. See install hints in module docstring."
            )
        if "mineru-cloud" not in avail:
            print(
                "[pdf_extract] WARNING: 云端 MinerU 不可用（未配置 MINERU_TOKEN 或缺 requests）；"
                "将回退本地 pymupdf4llm，公式可能丢失/乱码。精读请在 .env 配置 MINERU_TOKEN，"
                "或改用 arXiv LaTeX 源码。",
                file=sys.stderr,
            )
    else:
        backends_to_try = [_pick_backend(requested)]

    last_err: Exception | None = None
    for idx, b in enumerate(backends_to_try):
        impl = _BACKEND_IMPL.get(b)
        if not impl:
            continue
        print(f"[pdf_extract] Trying backend: {b} on {pdf_path.name} ...")
        t0 = time.time()
        try:
            md = impl(pdf_path)
            elapsed = time.time() - t0
            print(f"[pdf_extract] {b} succeeded in {elapsed:.1f}s ({len(md)} chars)")
            if write_cache:
                _write_cache(_cache_path(pdf_path, b), md)
            return md
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[pdf_extract] {b} failed after {elapsed:.1f}s: {e}", file=sys.stderr)
            last_err = e
            nxt = backends_to_try[idx + 1] if idx + 1 < len(backends_to_try) else None
            if b == "mineru-cloud" and nxt == "pymupdf4llm":
                print(
                    "[pdf_extract] WARNING: 云端 MinerU 转换失败，正在**回退**到本地 pymupdf4llm；"
                    "公式可能丢失/乱码，不适合精读。请检查 MINERU_TOKEN 是否有效/未过期，"
                    "或改用 arXiv LaTeX 源码。",
                    file=sys.stderr,
                )
            continue

    raise ExtractionFailed(f"All backends failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# 便捷：批量抽取
# ---------------------------------------------------------------------------
def extract_many(
    pdf_paths: list[Path],
    *,
    backend: str | None = None,
    arxiv_ids: dict[Path, str] | None = None,
) -> dict[Path, str | Exception]:
    """批量抽取，返回 {path: markdown_or_exception}。单篇失败不影响其他。"""
    out: dict[Path, str | Exception] = {}
    for p in pdf_paths:
        aid = (arxiv_ids or {}).get(p)
        try:
            out[p] = extract_pdf(p, backend=backend, prefer_latex_source=aid)
        except Exception as e:
            out[p] = e
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF extraction CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("backends", help="列出可用后端")

    p_extract = sub.add_parser("extract", help="抽取单个 PDF")
    p_extract.add_argument("pdf", type=Path)
    p_extract.add_argument("--backend", default=None, help="mineru-cloud|pymupdf4llm|auto")
    p_extract.add_argument("--arxiv-id", default=None, help="若有对应 arXiv ID，优先尝试 LaTeX 源码")
    p_extract.add_argument("--output", type=Path, default=None, help="输出 .md 文件；默认打印到 stdout")
    p_extract.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    if args.cmd == "backends" or args.cmd is None:
        avail = available_backends()
        print(f"Available backends: {avail or '(none)'}")
        print(f"Configured default : {settings.pdf_extract_backend}")
        print(f"MINERU_TOKEN set   : {bool(settings.mineru_token)}")
        if "mineru-cloud" not in avail:
            print("\n云端 MinerU 未就绪：在 .env 配置 MINERU_TOKEN=")
            print("  申请地址：https://mineru.net/apiManage/token （免费，约 90 天有效）")
        if not avail:
            print("\n本地兜底后端也未安装：uv add pymupdf4llm")

    elif args.cmd == "extract":
        md = extract_pdf(
            args.pdf,
            backend=args.backend,
            use_cache=not args.no_cache,
            prefer_latex_source=args.arxiv_id,
        )
        if args.output:
            args.output.write_text(md, encoding="utf-8")
            print(f"[pdf_extract] Written to {args.output} ({len(md)} chars)")
        else:
            print(md)
