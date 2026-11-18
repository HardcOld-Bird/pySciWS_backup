"""pysci-doc：⑤文档写作统一 CLI 门面（薄路由，主件）。

职责（选型定稿 2026-11-17）：把任务**分发**到三类入口，不实现其能力——

1. **DSH office 七技能**（office / office-docx / office-pptx / office-xlsx /
   office-render / office-recalculate / office-export-pdf）：Office 原生格式创作、
   编辑、追加、渲染、导 PDF。这类任务由 agent 在 DSH 技能内完成，本 CLI 只负责
   指路（``route``/``doctor``），不越俎；
2. **pandoc 3.14**：格式互转 / 提取 / citeproc 引用（``convert``、``refs`` 的消费端）；
3. **LaTeX 单轨工作流**：latexmk 编译 + 日志解析 + chktex（``latex``）。

加上三个薄胶水入口：``refs``（zotero-mcp 导出→refs.bib）、``formula``（透明 PNG）、
``pdf2png``（视觉校对）。接口设计：每个子命令一个 ``cmd_*`` 函数，只做参数搬运与
人类可读输出，业务在 :mod:`pandoc_wrapper` / :mod:`refs_bridge` / :mod:`latex_workflow`
/ :mod:`formula_png` / :mod:`pdf_to_png`；``main`` 返回退出码（0 成功，1 失败/无路由，
2 参数错误——argparse 自身占用）。

用法::

    pysci-doc route 编辑 pptx 追加一页
    pysci-doc doctor
    pysci-doc convert paper.md -t docx -o paper.docx --reference-doc house.docx
    pysci-doc convert slides.md -t pptx -o slides.pptx --slide-level 2
    pysci-doc refs items.json -o refs.bib
    pysci-doc latex build paper.tex --engine xelatex
    pysci-doc latex lint paper.tex
    pysci-doc formula "E = mc^2" -o eq.png
    pysci-doc pdf2png paper.pdf -o out_dir --dpi 200
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from pysci.skills.document_writing import (
    formula_png,
    latex_workflow,
    pandoc_wrapper,
    pdf_to_png,
    refs_bridge,
    toolpaths,
)


@dataclass(frozen=True)
class RouteRule:
    """一条任务路由规则（``route`` 子命令的数据源）。"""

    keywords: tuple[str, ...]
    target: str  # "dsh-office" | "pandoc" | "latex" | "glue"
    entry: str
    detail: str


#: 任务路由表（与 .agents/skills/document-writing/SKILL.md 的表一一对应）。
ROUTE_TABLE: tuple[RouteRule, ...] = (
    RouteRule(
        keywords=("创建", "新建", "编辑", "追加", "修改", "重算", "docx", "pptx", "xlsx",
                  "word", "excel", "表格", "幻灯片", "office"),
        target="dsh-office",
        entry="skill(office) → office-docx / office-pptx / office-xlsx / office-recalculate",
        detail="Office 原生格式创作/编辑/追加一律走 DSH office 技能（python-docx/pptx/openpyxl 锚定）",
    ),
    RouteRule(
        keywords=("渲染", "目检", "视觉检查", "render"),
        target="dsh-office",
        entry="skill(office-render)",
        detail="Office 文件渲染成图后 read_image 验收（PDF 校对走 pdf2png）",
    ),
    RouteRule(
        keywords=("导出 pdf", "导 pdf", "export-pdf"),
        target="dsh-office",
        entry="skill(office-export-pdf)",
        detail="Office → PDF 用 DSH LibreOffice Kit，不自写",
    ),
    RouteRule(
        keywords=("转换", "互转", "提取", "md", "->docx", "->pptx", "pandoc", "转 docx",
                  "转 pptx", "reference-doc", "家风"),
        target="pandoc",
        entry="pysci-doc convert",
        detail="格式互转/提取走 pandoc；它不能增量追加既有 Office 文件",
    ),
    RouteRule(
        keywords=("引用", "引文", "citeproc", "csl", "参考文献", "bibliography"),
        target="pandoc",
        entry="pysci-doc convert --citeproc --bib refs.bib --csl <style>.csl",
        detail="引文排版走 pandoc citeproc；refs.bib 由 refs 生成",
    ),
    RouteRule(
        keywords=("refs", "bib", "zotero", "文献库"),
        target="glue",
        entry="pysci-doc refs",
        detail="zotero-mcp 导出 JSON → refs.bib 归一化（refs_bridge）",
    ),
    RouteRule(
        keywords=("latex", "排版", "revtex", "论文", "编译", "tex", "beamer"),
        target="latex",
        entry="pysci-doc latex build / lint",
        detail="LaTeX 单轨：latexmk→日志解析→chktex→PDF（期刊投稿无可替代）",
    ),
    RouteRule(
        keywords=("公式", "formula", "透明 png", "equation"),
        target="glue",
        entry="pysci-doc formula",
        detail="公式→透明底 RGBA PNG（docx 内公式走 pandoc OMML，不出图）",
    ),
    RouteRule(
        keywords=("pdf 校对", "pdf2png", "逐页", "pdf 转 png"),
        target="glue",
        entry="pysci-doc pdf2png",
        detail="PDF 逐页渲染 PNG 目检（DSH office-render 不渲染 PDF）",
    ),
)


def match_routes(text: str) -> list[RouteRule]:
    """按关键词给任务文本打分，返回最匹配的路由规则（纯函数，机械可测）。

    Args:
        text: 任务描述（中英文混合均可）。

    Returns:
        命中关键词数最多的规则列表（同分全部返回）；无命中返回空列表。
    """
    lowered = text.lower()
    scored = [
        (sum(1 for kw in rule.keywords if kw.lower() in lowered), rule)
        for rule in ROUTE_TABLE
    ]
    best = max((score for score, _ in scored), default=0)
    return [rule for score, rule in scored if score == best and score > 0]


def cmd_route(args: argparse.Namespace) -> int:
    """``route``：打印任务路由决定（记不准走哪条路时先问它）。"""
    rules = match_routes(args.task)
    if not rules:
        print(f"无路由命中：{args.task!r}（查技能文档任务路由表）")
        return 1
    for rule in rules:
        print(f"[{rule.target}] {rule.entry}\n    {rule.detail}")
    return 0


def doctor_report() -> list[tuple[str, bool, str]]:
    """收集工具链/依赖就位状态（纯数据，供 ``doctor`` 与测试使用）。

    Returns:
        ``(名称, 是否就位, 说明)`` 列表：pandoc/latexmk/chktex 为核心件，
        pymupdf/Pillow 为可选件（formula/pdf2png 需要）。
    """
    rows: list[tuple[str, bool, str]] = []
    for name, exe in (
        ("pandoc", toolpaths.pandoc_exe()),
        ("latexmk", toolpaths.latexmk_exe()),
        ("chktex", toolpaths.chktex_exe()),
    ):
        resolved = Path(exe)
        ok = resolved.is_file() or _on_path(exe)
        rows.append((name, ok, exe))
    for name, module, hint in (
        ("pymupdf", "pymupdf", "uv sync 安装 pymupdf4llm"),
        ("pillow", "PIL", "python-pptx 的传递依赖"),
    ):
        try:
            __import__(module)
            rows.append((name, True, module))
        except ImportError:
            rows.append((name, False, f"缺失（{hint}）"))
    return rows


def _on_path(name: str) -> bool:
    """裸命令名是否可被 PATH 解析（fail-open 场景的兜底探测）。"""
    import shutil

    return shutil.which(name) is not None


def cmd_doctor(args: argparse.Namespace) -> int:
    """``doctor``：打印工具链/依赖状态；核心件缺失退出码 1。"""
    del args
    core_ok = True
    for name, ok, detail in doctor_report():
        mark = "ok " if ok else "MISS"
        print(f"[{mark}] {name}: {detail}")
        if not ok and name in ("pandoc", "latexmk", "chktex"):
            core_ok = False
    return 0 if core_ok else 1


def cmd_convert(args: argparse.Namespace) -> int:
    """``convert``：MD/任意格式 → docx/pptx/…（pandoc 分发）。"""
    try:
        out = pandoc_wrapper.convert(
            args.input,
            to=args.to,
            output=args.output,
            reference_doc=args.reference_doc,
            slide_level=args.slide_level,
            toc=args.toc,
            number_sections=args.number_sections,
            citeproc=args.citeproc,
            bib=args.bib,
            csl=args.csl,
        )
    except (pandoc_wrapper.PandocError, ValueError) as exc:
        print(f"convert 失败：{exc}", file=sys.stderr)
        return 1
    print(out if out is not None else "(stdout)")
    return 0


def cmd_refs(args: argparse.Namespace) -> int:
    """``refs``：zotero-mcp 导出 JSON → refs.bib（全量重写，确定性输出）。"""
    try:
        items = refs_bridge.load_items(args.items)
        out = refs_bridge.write_refs_bib(items, args.output)
    except (refs_bridge.RefsError, OSError, ValueError) as exc:
        print(f"refs 失败：{exc}", file=sys.stderr)
        return 1
    print(f"{out}（{len(items)} 条）")
    return 0


def cmd_latex_build(args: argparse.Namespace) -> int:
    """``latex build``：latexmk 编译 + 日志解析（+ chktex），打印 file:line 报告。"""
    try:
        result = latex_workflow.build(
            args.tex, engine=args.engine, run_lint=not args.no_lint
        )
    except latex_workflow.LatexError as exc:
        print(f"latex build 失败：{exc}", file=sys.stderr)
        return 1
    for issue in result.issues:
        loc = f"{issue.file or '?'}:{issue.line or '?'}"
        print(f"[{issue.severity}] {loc}: {issue.message}")
    if result.ok:
        print(f"ok: {result.pdf}")
    else:
        print(f"failed (exit {result.returncode})", file=sys.stderr)
    return 0 if result.ok else 1


def cmd_latex_lint(args: argparse.Namespace) -> int:
    """``latex lint``：chktex 静态检查（warning 逐条裁决后交付）。"""
    try:
        issues = latex_workflow.lint(args.tex)
    except latex_workflow.LatexError as exc:
        print(f"latex lint 失败：{exc}", file=sys.stderr)
        return 1
    for issue in issues:
        print(f"[warning] {issue.file}:{issue.line}: {issue.message}")
    print(f"{len(issues)} warning(s)")
    return 0


def cmd_formula(args: argparse.Namespace) -> int:
    """``formula``：LaTeX 公式源串 → 透明底裁剪 PNG（pptx/图内公式用）。"""
    try:
        out = formula_png.render_formula(
            args.tex, args.output, dpi=args.dpi, padding=args.padding,
            inline=args.inline,
        )
    except (formula_png.FormulaError, latex_workflow.LatexError) as exc:
        print(f"formula 失败：{exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


def cmd_pdf2png(args: argparse.Namespace) -> int:
    """``pdf2png``：PDF 逐页渲染 PNG（LaTeX/导出 PDF 的视觉校对）。"""
    try:
        paths = pdf_to_png.render_pdf(
            args.pdf, args.output, dpi=args.dpi, prefix=args.prefix
        )
    except pdf_to_png.PdfRenderError as exc:
        print(f"pdf2png 失败：{exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """组装 argparse 声明（子命令结构即接口文档）。"""
    parser = argparse.ArgumentParser(
        prog="pysci-doc", description="⑤文档写作薄路由 CLI（DSH office / pandoc / LaTeX）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("route", help="任务路由查询：该走 DSH office / pandoc / LaTeX 哪条路")
    p.add_argument("task", help="任务描述文本")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("doctor", help="工具链/依赖就位检查")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("convert", help="pandoc 格式互转（MD→docx/pptx 等）")
    p.add_argument("input", help="输入文档")
    p.add_argument("-t", "--to", default="docx", help="目标格式（默认 docx）")
    p.add_argument("-o", "--output", help="输出文件（缺省写 stdout）")
    p.add_argument("--reference-doc", help="--reference-doc 家风样式模板")
    p.add_argument("--slide-level", type=int, help="--slide-level（pptx 分页层级）")
    p.add_argument("--toc", action="store_true", help="--toc")
    p.add_argument("--number-sections", action="store_true", help="--number-sections")
    p.add_argument("--citeproc", action="store_true", help="--citeproc（引文排版）")
    p.add_argument("--bib", help="参考文献库 refs.bib（--bibliography）")
    p.add_argument("--csl", help="CSL 样式（--csl）")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("refs", help="zotero-mcp 导出 JSON → refs.bib 归一化")
    p.add_argument("items", help="导出 JSON（Zotero JSON 或 CSL JSON）")
    p.add_argument("-o", "--output", required=True, help="输出 refs.bib 路径")
    p.set_defaults(func=cmd_refs)

    p = sub.add_parser("latex", help="LaTeX 单轨工作流（latexmk + 日志解析 + chktex）")
    latex_sub = p.add_subparsers(dest="latex_command", required=True)
    pb = latex_sub.add_parser("build", help="latexmk 编译 + 日志解析 + chktex")
    pb.add_argument("tex", help="主 .tex 文件")
    pb.add_argument(
        "--engine", default="pdflatex", choices=("pdflatex", "xelatex", "lualatex")
    )
    pb.add_argument("--no-lint", action="store_true", help="跳过 chktex")
    pb.set_defaults(func=cmd_latex_build)
    pl = latex_sub.add_parser("lint", help="chktex 静态检查")
    pl.add_argument("tex", help="主 .tex 文件")
    pl.set_defaults(func=cmd_latex_lint)

    p = sub.add_parser("formula", help="LaTeX 公式 → 透明底 RGBA PNG（ink 裁剪）")
    p.add_argument("tex", help="公式源串（不含 $ 包裹）")
    p.add_argument("-o", "--output", required=True, help="输出 PNG")
    p.add_argument("--dpi", type=int, default=600, help="渲染分辨率（默认 600）")
    p.add_argument("--padding", type=int, default=2, help="裁剪留白像素")
    p.add_argument("--inline", action="store_true", help="行内模式 $...$（默认显示模式）")
    p.set_defaults(func=cmd_formula)

    p = sub.add_parser("pdf2png", help="PDF 逐页渲染 PNG（视觉校对）")
    p.add_argument("pdf", help="输入 PDF")
    p.add_argument("-o", "--output", required=True, help="PNG 输出目录")
    p.add_argument("--dpi", type=int, default=200, help="渲染分辨率（默认 200）")
    p.add_argument("--prefix", help="输出文件名前缀（默认 PDF stem）")
    p.set_defaults(func=cmd_pdf2png)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（console script ``pysci-doc`` 指向这里）。

    Args:
        argv: 参数列表；None 取 ``sys.argv[1:]``。

    Returns:
        退出码：0 成功，1 失败/无路由，2 参数错误（argparse）。
    """
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
