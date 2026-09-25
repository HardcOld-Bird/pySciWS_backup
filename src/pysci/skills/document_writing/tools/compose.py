"""compose —— 文档写作统一 CLI 门面（document-writing skill 的后端）。

一个入口收敛「LaTeX 论文工作流 + 文档/PPTX 阅读提取」的全部能力，供 skill 文档与
LLM 直接调用，无需了解底层模块实现。

子命令
------
    doctor           环境与能力自检（TeX / LibreOffice / Python 库）
    tex new          从模板脚手架一个写作项目
    tex build        latexmk 编译 → 解析报错（文件:行号）
    tex lint         chktex 静态检查
    tex refs         从 Zotero 导出 refs.bib（复用 literature_research）
    read <file>      统一提取（markitdown）→ Markdown（pptx/docx/pdf/xlsx/...）
    slides extract   结构化提取 .pptx（逐页文本/表格/图片/演讲者备注）
    slides digest    大型 pptx → 图文互证 Markdown 工作区（去重导图/邻近文字/分节分块）
    verify <pdf>     把 PDF 渲染成 PNG，供 LLM「看图」校对版式

用法（在项目根目录）::

    .venv\\Scripts\\python.exe -m pysci.skills.document_writing.tools.compose <cmd> [options]

设计原则：只做编排，重活交给 latex_build / pdf_render / extract / pptx_io / refs_bridge。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from . import latex_build, pdf_render
from .config import settings

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
TEMPLATES_DIR: Path = settings.templates_dir
LATEX_TEMPLATES: Path = TEMPLATES_DIR / "latex"
PROJECTS_DIR: Path = settings.projects_dir


# ===========================================================================
# 通用小工具
# ===========================================================================
def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _list_templates() -> list[str]:
    if not LATEX_TEMPLATES.exists():
        return []
    return sorted(
        d.name for d in LATEX_TEMPLATES.iterdir() if (d / "main.tex").is_file()
    )


def _resolve_main_tex(target: str) -> Path:
    """把 项目名 / 项目目录 / .tex 文件 解析为主 .tex 路径。"""
    p = Path(target)
    # 1) 直接给 .tex 文件
    if p.is_file() and p.suffix.lower() == ".tex":
        return p.resolve()
    # 2) 给目录
    if p.is_dir():
        return _main_in_dir(p)
    # 3) 给项目名（projects/<name>）
    guess = PROJECTS_DIR / target
    if guess.is_dir():
        return _main_in_dir(guess)
    raise FileNotFoundError(
        f"无法解析目标：{target}（可为 .tex 文件、项目目录，或 projects/ 下的项目名）"
    )


def _main_in_dir(d: Path) -> Path:
    main = d / "main.tex"
    if main.is_file():
        return main.resolve()
    cands = [
        f
        for f in sorted(d.glob("*.tex"))
        if "\\documentclass" in f.read_text(encoding="utf-8", errors="ignore")
    ]
    if len(cands) == 1:
        return cands[0].resolve()
    if len(cands) > 1:
        names = ", ".join(c.name for c in cands)
        raise ValueError(f"{d} 下有多个含 \\documentclass 的 .tex：{names}，请直接指定其一")
    raise FileNotFoundError(f"{d} 下未找到 main.tex 或含 \\documentclass 的 .tex")


def _project_dir(main_tex: Path) -> Path:
    return main_tex.parent


# ===========================================================================
# 子命令：doctor
# ===========================================================================
def cmd_doctor(args: argparse.Namespace) -> int:
    print("=== pySciWS document_writing — 能力自检 (doctor) ===\n")
    print(settings.summary())
    print()
    print("【Phase 1 能力清单】")
    tex_ok = settings.tex_ready
    libs = settings.python_libs
    print(f"  LaTeX 编译      : {'✓ 就绪' if tex_ok else '✗ 需安装 TeX Live（见上方指引）'}")
    print(f"  PDF 看图校对    : {'✓ 就绪' if libs['pymupdf'] else '✗ 缺 pymupdf'}")
    print(f"  PPTX 结构化读取 : {'✓ 就绪' if libs['pptx'] else '✗ uv sync --extra writing'}")
    print(f"  文档→Markdown   : {'✓ 就绪' if libs['markitdown'] else '✗ uv sync --extra writing'}")
    print(f"  Zotero→refs.bib : {'✓ 复用 literature_research' if _module_available('pyzotero') else '△ 需 pyzotero'}")
    print()
    print(f"  LaTeX 模板      : {', '.join(_list_templates()) or '（templates/latex/ 下暂无）'}")
    print(f"  写作项目目录    : {PROJECTS_DIR}")
    if not tex_ok:
        print(
            "\n[!] 尚未检测到 TeX。安装 TeX Live 完整版后重开终端，再跑 `compose doctor`。\n"
            "    PPTX 读取 / 文档提取 / PDF 渲染不依赖 TeX，可立即使用。"
        )
    print("=== 自检结束 ===")
    return 0


# ===========================================================================
# 子命令：tex new
# ===========================================================================
def cmd_tex_new(args: argparse.Namespace) -> int:
    tpls = _list_templates()
    if args.template not in tpls:
        print(
            f"[tex new] 未知模板 '{args.template}'。可用：{', '.join(tpls) or '（无）'}",
            file=sys.stderr,
        )
        return 2
    tpl_dir = LATEX_TEMPLATES / args.template

    proj = PROJECTS_DIR / args.slug
    if proj.exists() and any(proj.iterdir()) and not args.force:
        print(
            f"[tex new] 项目已存在且非空：{proj}（加 --force 覆盖模板文件）",
            file=sys.stderr,
        )
        return 2
    proj.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for f in tpl_dir.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(tpl_dir)
        dest = proj / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not args.force:
            continue
        shutil.copy2(f, dest)
        copied.append(str(rel))

    (proj / "figures").mkdir(exist_ok=True)
    (proj / "build").mkdir(exist_ok=True)

    main = proj / "main.tex"
    if args.title and main.exists():
        txt = main.read_text(encoding="utf-8")
        txt = re.sub(
            r"\\title\{[^}]*\}", lambda m: f"\\title{{{args.title}}}", txt, count=1
        )
        main.write_text(txt, encoding="utf-8")

    print(f"[tex new] 项目已创建：{proj}")
    print(f"[tex new] 模板 '{args.template}' → 复制文件：{', '.join(copied)}")
    print("[tex new] 下一步：")
    print(f"    1. 编辑 {main}")
    print(f"    2. compose tex refs {args.slug} --query '<主题>'   # 从 Zotero 刷新 refs.bib")
    print(f"    3. compose tex build {args.slug}                  # 编译")
    print(f"    4. compose verify {proj / 'build' / 'main.pdf'}    # 渲染 PNG 看图校对")
    return 0


# ===========================================================================
# 子命令：tex build
# ===========================================================================
def cmd_tex_build(args: argparse.Namespace) -> int:
    try:
        main = _resolve_main_tex(args.target)
    except (FileNotFoundError, ValueError) as e:
        print(f"[tex build] {e}", file=sys.stderr)
        return 2

    if not settings.tex_ready:
        print(latex_build.TeXNotInstalled())
        return 3

    print(f"[tex build] 编译 {main}（engine={args.engine or 'auto'}）...")
    try:
        res = latex_build.build(
            main,
            engine=args.engine,
            bib=not args.no_bib,
            out_dir=args.out_dir,
            timeout=args.timeout,
            extra_args=["-shell-escape"] if args.shell_escape else None,
        )
    except latex_build.TeXNotInstalled as e:
        print(e, file=sys.stderr)
        return 3

    print(res.summary())
    if not res.ok:
        if res.stdout_tail:
            print("\n--- 编译输出末尾 ---")
            print(res.stdout_tail)
        print("\n[tex build] 修复上述错误后重跑。", file=sys.stderr)
        return 1

    print(f"\n✓ PDF：{res.pdf_path}")
    if args.render and res.pdf_path:
        pngs = pdf_render.render_pdf_pages(res.pdf_path, dpi=args.dpi)
        print(f"[tex build] 已渲染 {len(pngs)} 页 PNG（用 Read 查看版式）：")
        for p in pngs:
            print(f"    {p}")
    else:
        print(f"[tex build] 看图校对：compose verify {res.pdf_path}")
    return 0


# ===========================================================================
# 子命令：tex lint
# ===========================================================================
def cmd_tex_lint(args: argparse.Namespace) -> int:
    try:
        main = _resolve_main_tex(args.target)
    except (FileNotFoundError, ValueError) as e:
        print(f"[tex lint] {e}", file=sys.stderr)
        return 2
    if not settings.find_tex_tools().get("chktex"):
        print("[tex lint] 未检测到 chktex（随 TeX Live 提供）。", file=sys.stderr)
        return 3
    issues = latex_build.lint(main)
    if not issues:
        print(f"[tex lint] {main.name}：未发现问题。")
        return 0
    for it in issues:
        print(it.fmt())
    print(f"[tex lint] 共 {len(issues)} 条提示。")
    return 0


# ===========================================================================
# 子命令：tex refs
# ===========================================================================
def cmd_tex_refs(args: argparse.Namespace) -> int:
    from . import refs_bridge

    try:
        main = _resolve_main_tex(args.target)
    except (FileNotFoundError, ValueError) as e:
        print(f"[tex refs] {e}", file=sys.stderr)
        return 2
    proj = _project_dir(main)
    out = Path(args.out) if args.out else proj / "refs.bib"
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None

    if not (args.collection or args.tag or args.query or keys):
        print(
            "[tex refs] 需指定筛选：--query / --collection / --tag / --keys 之一。",
            file=sys.stderr,
        )
        return 2

    try:
        res = refs_bridge.export_bib(
            collection=args.collection,
            tag=args.tag,
            query=args.query,
            keys=keys,
            limit=args.limit,
            out_path=out,
        )
    except Exception as e:
        print(f"[tex refs] 导出失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"[tex refs] backend={res.backend} 导出 {res.count} 条 → {res.out_path}")
    if res.citekeys:
        preview = ", ".join(res.citekeys[:12])
        more = " ..." if len(res.citekeys) > 12 else ""
        print(f"[tex refs] citekeys：{preview}{more}")
    return 0


# ===========================================================================
# 子命令：read（统一提取）
# ===========================================================================
def cmd_read(args: argparse.Namespace) -> int:
    from . import extract

    src = Path(args.file)
    if not src.exists():
        print(f"[read] 文件不存在：{src}", file=sys.stderr)
        return 2
    try:
        res = extract.to_markdown(src, force=args.force, backend=args.backend)
    except Exception as e:
        print(f"[read] 提取失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(
        f"[read] backend={res.backend}  from_cache={res.from_cache}  "
        f"chars={res.char_count}"
    )
    print(f"[read] Markdown：{res.cache_path}（用 Read 查看全文）")
    if args.preview:
        print("\n--- 预览（前 1500 字符）---")
        print(res.markdown[:1500])
    return 0


# ===========================================================================
# 子命令：slides extract（结构化 pptx）
# ===========================================================================
def cmd_slides_extract(args: argparse.Namespace) -> int:
    from . import pptx_io

    src = Path(args.pptx)
    if not src.exists():
        print(f"[slides] 文件不存在：{src}", file=sys.stderr)
        return 2

    export_dir = args.export_images
    if export_dir is None and args.with_images:
        export_dir = settings.cache_extracted / f"{src.stem}_images"

    try:
        slides = pptx_io.read_pptx(src, export_images_to=export_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[slides] {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"[slides] {e}", file=sys.stderr)
        return 3

    md = pptx_io.slides_to_markdown(
        slides, source_name=src.name, include_notes=not args.no_notes
    )
    out = Path(args.out) if args.out else settings.cache_extracted / f"{src.stem}__slides.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    n_notes = sum(1 for s in slides if s.notes)
    n_imgs = sum(len(s.images) for s in slides)
    n_tables = sum(len(s.tables) for s in slides)
    print(f"[slides] {src.name}：{len(slides)} 页，{n_tables} 表，{n_imgs} 图，{n_notes} 页含备注")
    print(f"[slides] Markdown：{out}（用 Read 查看全文）")
    if export_dir:
        print(f"[slides] 图片已导出到：{export_dir}（可用 Read 查看）")
    if args.preview:
        print("\n--- 预览（前 1500 字符）---")
        print(md[:1500])
    return 0


# ===========================================================================
# 子命令：slides new / add / from-markdown（PPTX 写入）
# ===========================================================================
def cmd_slides_new(args: argparse.Namespace) -> int:
    from . import pptx_io

    out = Path(args.out)
    pptx_io.build_pptx([], out, deck_title=args.title, deck_subtitle=args.subtitle or "")
    print(f"[slides new] 已创建：{out}")
    print(f"[slides new] 追加页面：compose slides add '{out}' --title '...' --bullet '...'")
    return 0


def cmd_slides_add(args: argparse.Namespace) -> int:
    from . import pptx_io

    src = Path(args.pptx)
    if not src.exists():
        print(f"[slides add] 文件不存在：{src}", file=sys.stderr)
        return 2
    spec = pptx_io.SlideSpec(
        title=args.title or "", bullets=args.bullet or [], notes=args.notes or ""
    )
    pptx_io.add_slide_to(src, spec)
    print(f"[slides add] 已追加一页 → {src}")
    return 0


def cmd_slides_from_markdown(args: argparse.Namespace) -> int:
    from . import pptx_io

    md = Path(args.markdown)
    if not md.exists():
        print(f"[slides from-markdown] 文件不存在：{md}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else md.with_suffix(".pptx")
    # utf-8-sig：透明剥离 Windows 工具（如 PowerShell Out-File）写入的 BOM
    pptx_io.markdown_to_pptx(md.read_text(encoding="utf-8-sig"), out)
    print(f"[slides from-markdown] {md.name} → {out}")
    print(f"[slides from-markdown] 回读校验：compose slides extract '{out}'")
    return 0


# ===========================================================================
# 子命令：slides digest（大型 pptx → 图文互证 Markdown 工作区）
# ===========================================================================
def cmd_slides_digest(args: argparse.Namespace) -> int:
    from . import deck_digest

    src = Path(args.pptx)
    if not src.exists():
        print(f"[slides digest] 文件不存在：{src}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else settings.cache_dir / "digests" / src.stem

    if args.lint:
        rep = deck_digest.verify_links(out)
        print(
            f"[slides digest --lint] 链接存在 {rep['ok']} 个；缺失 {len(rep['broken'])} 个；"
            f"待渲染 {len(rep['pending_renders'])} 个"
        )
        for b in rep["broken"][:20]:
            print(f"    ✗ {b}")
        if rep["pending_renders"]:
            print(f"    … 待渲染（跑 --render 后生成）：{len(rep['pending_renders'])} 个")
        return 1 if rep["broken"] else 0

    try:
        res = deck_digest.digest_pptx(
            src,
            out,
            render=args.render,
            dpi=args.dpi,
            gif_frames=args.gif_frames,
            chunk_by=args.chunk_by,
            max_chunk_slides=args.max_chunk_slides,
            chunk_size=args.chunk_size,
            batch_figure=args.batch_figure,
            batch_text=args.batch_text,
            section_starts=(
                [int(x) for x in args.section_at.split(",") if x.strip()]
                if args.section_at
                else None
            ),
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[slides digest] {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"[slides digest] {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[slides digest] 失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(res.summary())
    if res.sections:
        print("\n[slides digest] 检测到的分节：")
        for s in res.sections[:40]:
            print(f"    Slide {s['start']:03d}–{s['end']:03d}  {s['title']}")
        if len(res.sections) > 40:
            print(f"    …共 {len(res.sections)} 节")
    print("\n[slides digest] 下一步：")
    if not res.rendered and res.n_figure_slides:
        print(
            f"    1. 补整页合成渲染（需 LibreOffice）：compose slides digest '{src}' "
            f"--out '{out}' --render"
        )
    print(f"    2. 阅读 {res.index_md}（导航 + Phase B 约定）")
    print(f"    3. 逐批翻译（每批 {args.batch_figure} 图页 / {args.batch_text} 文本页），填写每图「解读」")
    print(f"    4. 批末体检：compose slides digest '{src}' --out '{out}' --lint")
    return 0


# ===========================================================================
# 子命令：docx read / from-markdown / add（DOCX 读写）
# ===========================================================================
def cmd_docx_read(args: argparse.Namespace) -> int:
    from . import docx_io

    src = Path(args.file)
    if not src.exists():
        print(f"[docx read] 文件不存在：{src}", file=sys.stderr)
        return 2
    try:
        md = docx_io.docx_to_markdown(src)
    except (FileNotFoundError, ValueError) as e:
        print(f"[docx read] {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"[docx read] {e}", file=sys.stderr)
        return 3
    out = Path(args.out) if args.out else settings.cache_extracted / f"{src.stem}__docx.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[docx read] {src.name} → {out}（用 Read 查看全文）")
    if args.preview:
        print("\n--- 预览（前 1500 字符）---")
        print(md[:1500])
    return 0


def cmd_docx_from_markdown(args: argparse.Namespace) -> int:
    from . import docx_io

    md = Path(args.markdown)
    if not md.exists():
        print(f"[docx from-markdown] 文件不存在：{md}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else md.with_suffix(".docx")
    docx_io.markdown_to_docx(md.read_text(encoding="utf-8-sig"), out)
    print(f"[docx from-markdown] {md.name} → {out}")
    print(f"[docx from-markdown] 回读校验：compose docx read '{out}'")
    return 0


def cmd_docx_add(args: argparse.Namespace) -> int:
    from . import docx_io

    src = Path(args.docx)
    if not src.exists():
        print(f"[docx add] 文件不存在：{src}", file=sys.stderr)
        return 2
    if args.heading:
        block = docx_io.DocxBlock(kind="heading", text=args.heading, level=args.level)
    elif args.bullet:
        block = docx_io.DocxBlock(kind="bullet", text=args.bullet)
    elif args.text:
        block = docx_io.DocxBlock(kind="paragraph", text=args.text)
    else:
        print("[docx add] 需指定 --heading / --bullet / --text 之一。", file=sys.stderr)
        return 2
    docx_io.add_block_to(src, block)
    print(f"[docx add] 已追加 → {src}")
    return 0


# ===========================================================================
# 子命令：convert（LibreOffice headless 格式转换）
# ===========================================================================
def cmd_convert(args: argparse.Namespace) -> int:
    from . import office_convert

    src = Path(args.file)
    if not src.exists():
        print(f"[convert] 文件不存在：{src}", file=sys.stderr)
        return 2
    try:
        out = office_convert.convert(src, args.to, out_dir=args.out_dir)
    except office_convert.LibreOfficeNotInstalled as e:
        print(e, file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[convert] 失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[convert] {src.name} → {out}")
    if args.to == "pdf" and args.verify:
        pngs = pdf_render.render_pdf_pages(out, dpi=args.dpi)
        print(f"[convert] 已渲染 {len(pngs)} 页 PNG（用 Read 查看）：")
        for p in pngs:
            print(f"    {p}")
    return 0


# ===========================================================================
# 子命令：verify（PDF → PNG）
# ===========================================================================
def cmd_verify(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[verify] PDF 不存在：{pdf}", file=sys.stderr)
        return 2
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    try:
        pngs = pdf_render.render_pdf_pages(
            pdf,
            out_dir=args.out_dir,
            dpi=args.dpi,
            pages=pages,
            max_pages=args.max_pages,
        )
    except Exception as e:
        print(f"[verify] 渲染失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    total = pdf_render.page_count(pdf)
    print(f"[verify] {pdf.name}：共 {total} 页，已渲染 {len(pngs)} 页 PNG（dpi={args.dpi}）")
    print("[verify] 用 Read 工具逐一查看下列 PNG，即可核对真实版式：")
    for p in pngs:
        print(f"    {p}")
    return 0


# ===========================================================================
# 参数解析与入口
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compose",
        description="pySciWS 文档写作统一 CLI（LaTeX 论文工作流 + PPTX/文档阅读提取）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor", help="环境与能力自检")
    sp.set_defaults(func=cmd_doctor)

    # --- tex 子命令组 ---
    tex = sub.add_parser("tex", help="LaTeX 工作流（new/build/lint/refs）")
    texsub = tex.add_subparsers(dest="tex_cmd", required=True)

    sp = texsub.add_parser("new", help="从模板脚手架写作项目")
    sp.add_argument("slug", help="项目名（在 projects/ 下建同名目录）")
    sp.add_argument(
        "--template", default="revtex", help="模板名（见 templates/latex/，默认 revtex）"
    )
    sp.add_argument("--title", default=None, help="替换 \\title{}")
    sp.add_argument("--force", action="store_true", help="覆盖已存在的模板文件")
    sp.set_defaults(func=cmd_tex_new)

    sp = texsub.add_parser("build", help="latexmk 编译 + 报错解析")
    sp.add_argument("target", help="项目名 / 项目目录 / main.tex")
    sp.add_argument(
        "--engine", default=None, help="auto / pdflatex / xelatex / lualatex"
    )
    sp.add_argument("--out-dir", default=None, dest="out_dir", help="构建产物目录（默认 <项目>/build）")
    sp.add_argument("--no-bib", action="store_true", dest="no_bib", help="不跑 bibtex/biber")
    sp.add_argument("--timeout", type=int, default=600)
    sp.add_argument("--shell-escape", action="store_true", dest="shell_escape", help="启用 -shell-escape")
    sp.add_argument("--render", action="store_true", help="编译成功后渲染全部页 PNG")
    sp.add_argument("--dpi", type=int, default=140)
    sp.set_defaults(func=cmd_tex_build)

    sp = texsub.add_parser("lint", help="chktex 静态检查")
    sp.add_argument("target", help="项目名 / 项目目录 / .tex")
    sp.set_defaults(func=cmd_tex_lint)

    sp = texsub.add_parser("refs", help="从 Zotero 导出 refs.bib")
    sp.add_argument("target", help="项目名 / 项目目录 / main.tex")
    sp.add_argument("--query", default=None, help="按关键词搜索 Zotero")
    sp.add_argument("--collection", default=None, help="按分类 key 导出")
    sp.add_argument("--tag", default=None, help="按标签导出")
    sp.add_argument("--keys", default=None, help="逗号分隔的条目 key")
    sp.add_argument("--out", default=None, help="输出 .bib（默认 <项目>/refs.bib）")
    sp.add_argument("--limit", type=int, default=200)
    sp.set_defaults(func=cmd_tex_refs)

    # --- read ---
    sp = sub.add_parser("read", help="统一提取（markitdown）→ Markdown")
    sp.add_argument("file", help="pptx/docx/pdf/xlsx/html/...")
    sp.add_argument("--force", action="store_true", help="忽略缓存重新提取")
    sp.add_argument("--backend", default="auto", help="auto/markitdown/pymupdf4llm/pptx_io")
    sp.add_argument("--preview", action="store_true", help="打印前 1500 字符预览")
    sp.set_defaults(func=cmd_read)

    # --- slides 子命令组 ---
    slides = sub.add_parser("slides", help="幻灯片（extract）")
    slidessub = slides.add_subparsers(dest="slides_cmd", required=True)

    sp = slidessub.add_parser("extract", help="结构化提取 .pptx")
    sp.add_argument("pptx", help=".pptx 文件")
    sp.add_argument("--no-notes", action="store_true", dest="no_notes", help="不含演讲者备注")
    sp.add_argument("--with-images", action="store_true", dest="with_images", help="导出图片到缓存目录")
    sp.add_argument("--export-images", default=None, dest="export_images", help="导出图片到指定目录")
    sp.add_argument("--out", default=None, help="输出 .md（默认 cache/extracted/<名>__slides.md）")
    sp.add_argument("--preview", action="store_true")
    sp.set_defaults(func=cmd_slides_extract)

    sp = slidessub.add_parser("new", help="新建 .pptx（可选标题页）")
    sp.add_argument("out", help="输出 .pptx 路径")
    sp.add_argument("--title", default=None, help="标题页主标题")
    sp.add_argument("--subtitle", default=None, help="标题页副标题")
    sp.set_defaults(func=cmd_slides_new)

    sp = slidessub.add_parser("add", help="向已有 .pptx 追加一页")
    sp.add_argument("pptx", help=".pptx 文件")
    sp.add_argument("--title", default=None, help="该页标题")
    sp.add_argument(
        "--bullet", action="append", default=None, dest="bullet",
        help="项目符号，可重复；前缀 2 空格表示下一级",
    )
    sp.add_argument("--notes", default=None, help="演讲者备注")
    sp.set_defaults(func=cmd_slides_add)

    sp = slidessub.add_parser("from-markdown", help="Markdown 大纲 → .pptx")
    sp.add_argument("markdown", help=".md 大纲文件")
    sp.add_argument("--out", default=None, help="输出 .pptx（默认同名）")
    sp.set_defaults(func=cmd_slides_from_markdown)

    sp = slidessub.add_parser(
        "digest", help="大型 pptx → 图文互证 Markdown 工作区（Phase A）"
    )
    sp.add_argument("pptx", help=".pptx 文件")
    sp.add_argument("--out", default=None, help="输出目录（默认 cache/digests/<名>/）")
    sp.add_argument("--render", action="store_true", help="同时渲染整页合成 PNG（需 LibreOffice）")
    sp.add_argument("--dpi", type=int, default=140)
    sp.add_argument("--gif-frames", type=int, default=3, dest="gif_frames",
                    help="gif 动图抽帧预览的帧数（含首末帧；LLM 不能直读 gif）")
    sp.add_argument(
        "--chunk-by", default="section", choices=["section", "fixed"], dest="chunk_by",
        help="分块策略：section=按检测到的分节（推荐），fixed=固定页数",
    )
    sp.add_argument("--max-chunk-slides", type=int, default=40, dest="max_chunk_slides",
                    help="单个 md 块最大页数（连续小节合并至此上限）")
    sp.add_argument("--section-at", default=None, dest="section_at",
                    help="逗号分隔的分节起始页（人工定界，跳过自动检测）")
    sp.add_argument("--chunk-size", type=int, default=50, dest="chunk_size",
                    help="chunk-by=fixed 时每块页数")
    sp.add_argument("--batch-figure", type=int, default=5, dest="batch_figure",
                    help="Phase B 每批含图页数（写入 progress.json）")
    sp.add_argument("--batch-text", type=int, default=15, dest="batch_text",
                    help="Phase B 每批纯文本页数（写入 progress.json）")
    sp.add_argument("--force", action="store_true", help="全量重建（会覆盖已填解读）")
    sp.add_argument("--lint", action="store_true", help="只对既有输出做图片链接体检")
    sp.set_defaults(func=cmd_slides_digest)

    # --- docx 子命令组 ---
    docx = sub.add_parser("docx", help="DOCX 读写（read/from-markdown/add）")
    docxsub = docx.add_subparsers(dest="docx_cmd", required=True)

    sp = docxsub.add_parser("read", help="结构化读取 .docx → Markdown")
    sp.add_argument("file", help=".docx 文件")
    sp.add_argument("--out", default=None, help="输出 .md（默认 cache/extracted/<名>__docx.md）")
    sp.add_argument("--preview", action="store_true")
    sp.set_defaults(func=cmd_docx_read)

    sp = docxsub.add_parser("from-markdown", help="Markdown → .docx")
    sp.add_argument("markdown", help=".md 文件")
    sp.add_argument("--out", default=None, help="输出 .docx（默认同名）")
    sp.set_defaults(func=cmd_docx_from_markdown)

    sp = docxsub.add_parser("add", help="向已有 .docx 追加一个块")
    sp.add_argument("docx", help=".docx 文件")
    sp.add_argument("--heading", default=None, help="追加标题")
    sp.add_argument("--level", type=int, default=1, help="标题级别（1..9）")
    sp.add_argument("--bullet", default=None, help="追加项目符号")
    sp.add_argument("--text", default=None, help="追加正文段落")
    sp.set_defaults(func=cmd_docx_add)

    # --- convert ---
    sp = sub.add_parser("convert", help="LibreOffice headless 转换（pptx/docx → pdf 等）")
    sp.add_argument("file", help="输入文件（pptx/docx/odt/xlsx/...）")
    sp.add_argument("--to", default="pdf", help="目标格式（pdf/docx/pptx/html/...）")
    sp.add_argument("--out-dir", default=None, dest="out_dir", help="输出目录（默认同 src）")
    sp.add_argument("--verify", action="store_true", help="转 pdf 后渲染 PNG 看图")
    sp.add_argument("--dpi", type=int, default=140)
    sp.set_defaults(func=cmd_convert)

    # --- verify ---
    sp = sub.add_parser("verify", help="PDF → PNG（供 LLM 看图校对版式）")
    sp.add_argument("pdf", help="PDF 文件")
    sp.add_argument("--dpi", type=int, default=140)
    sp.add_argument("--pages", default=None, help="逗号分隔的 1-based 页码")
    sp.add_argument("--max-pages", type=int, default=None, dest="max_pages")
    sp.add_argument("--out-dir", default=None, dest="out_dir")
    sp.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
        return int(rc) if rc else 0
    except KeyboardInterrupt:
        print("\n[compose] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
