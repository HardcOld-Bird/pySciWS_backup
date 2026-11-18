"""浏览器打印链最小冒烟（步骤1·任务三，2026-11-17）。

链路验证（用户裁决 (a) HTML 双栏链的底层）：
playwright sync API -> 系统 Chrome headless（channel="chrome"）-> page.pdf() -> PyMuPDF 验页。
产物：deps_management/scratch/artifacts/{about_blank, content}.pdf（冒烟产物不入仓）。

运行：sub_root/.venv/Scripts/python.exe deps_management/scratch/print_chain_smoke.py
"""

import time
from pathlib import Path

import pymupdf
from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HTML = (
    "<h1>pySci print-chain smoke</h1>"
    "<p>Chrome 打印链冒烟 2026-11-17。</p>"
    "<p>Machine learning (ML) is a branch of AI.</p>"
)


def verify(path: Path) -> None:
    doc = pymupdf.open(path)
    text = doc[0].get_text().strip() if doc.page_count else ""
    print(
        f"  verify {path.name}: pages={doc.page_count} "
        f"rect={doc[0].rect if doc.page_count else '-'} text={text[:40]!r}"
    )
    doc.close()


with sync_playwright() as p:
    t0 = time.perf_counter()
    browser = p.chromium.launch(channel="chrome", headless=True)
    t_launch = time.perf_counter() - t0
    print(f"launch(channel=chrome): {t_launch:.2f}s  version={browser.version}")
    page = browser.new_page()

    # 1) 任务要求的最小冒烟：about:blank -> PDF
    page.goto("about:blank")
    out1 = OUT_DIR / "about_blank.pdf"
    t1 = time.perf_counter()
    page.pdf(path=str(out1), format="A4")
    print(f"about:blank -> pdf: {time.perf_counter() - t1:.2f}s  {out1.stat().st_size} bytes")
    verify(out1)

    # 2) 正文打印（双栏 A4 链的实际负载雏形）
    page.set_content(HTML)
    out2 = OUT_DIR / "content.pdf"
    t2 = time.perf_counter()
    page.pdf(path=str(out2), format="A4")
    print(f"content -> pdf: {time.perf_counter() - t2:.2f}s  {out2.stat().st_size} bytes")
    verify(out2)

    browser.close()
print("PRINT-CHAIN SMOKE OK")
