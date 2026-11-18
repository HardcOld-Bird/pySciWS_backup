"""CLI 状态机：``python -m pysci.skills.literature_research.bilingual.cli <子命令>``。

五命令工作流（技能文档 ``references/bilingual.md`` 是操作面真源）：

- ``prepare <源md|stem>``   切分 + work.md/manifest + 图片回补 + TM 回填（``--reuse``）；
- ``status <stem>``        随时对账（已译/未译/存疑/占位假译计数）；
- ``submit <stem>``        全译才出稿 bilingual.md + 自动渲染 + 交付副本；
- ``check-work <stem>``    work.md 行结构 × manifest 硬校验（pod-guard 等价护栏）；
- ``render <stem>``        bilingual.md → bilingual.pdf（转接 :mod:`renderer`，并行实施）。

护栏 ADR（2026-11-17）：旧 pod-guard.mjs（DSH PreToolUse 编辑钩子）不平移——submit
**前强制执行 check-work**（结构破损即拒出稿），仅显式 ``--skip-check-work`` 可 override
（输出留痕）。薄胶水规范：路径一律 :mod:`pysci.paths`，无 ``parents[N]``；
状态码：0=成功 / 1=对账未过（未译拒稿）/ 2=前置不满足或结构闸拒绝。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from pysci import paths

from . import images, manifest, quality, reconcile, tm, units

#: 渲染接缝（:mod:`renderer` 归并行实施者；签名 ``render_pdf(md_path, pdf_path) -> Path``）。
RenderFn = Callable[[Path, Path], Path]


def resolve_source(target: str) -> tuple[Path, str]:
    """把 prepare 目标解析为 ``(源 md 路径, stem)``。

    目标可以是源 markdown 的路径，或 stem（在①域 ``papers/`` 提取件区解析——与
    ``mineru_cloud.extract_markdown`` 产物 ``papers/<key>.md`` 衔接）。

    Raises:
        FileNotFoundError: 源 md 不存在（提示先走①域 read 提取）。
    """
    p = Path(target)
    if p.is_file():
        return p.resolve(), p.stem
    papers = paths.LITERATURE_ROOT / "papers"
    cand = papers / f"{target}.md"
    if cand.is_file():
        return cand.resolve(), cand.stem
    hits = sorted(papers.glob(f"{target}*.md"))
    if len(hits) == 1:
        return hits[0].resolve(), hits[0].stem
    raise FileNotFoundError(
        f"源 md 不存在：{target!r}——先走①域 read 提取"
        "（mineru_cloud.extract_markdown 产物 papers/<key>.md）"
    )


def load_renderer() -> RenderFn:
    """载入渲染接缝（:mod:`renderer`，并行实施者维护）。

    Raises:
        RuntimeError: renderer.py 未就绪或渲染依赖缺失。
    """
    try:
        from . import renderer
    except ImportError as exc:  # pragma: no cover - 并行实施落地后消失
        raise RuntimeError(
            "renderer.py 未就绪（并行实施者维护）或渲染依赖缺失"
        ) from exc
    return renderer.render_pdf


def load_reorder() -> Callable[..., object]:
    """载入文序建议接缝（:mod:`reorder` 的 ``suggest_reorder``，并行实施者维护）。"""
    try:
        from . import reorder
    except ImportError as exc:  # pragma: no cover - 并行实施落地后消失
        raise RuntimeError(
            "reorder.py 未就绪（并行实施者维护）或依赖缺失"
        ) from exc
    return reorder.suggest_reorder


# ---------------------------------------------------------------------------
# bilingual prepare
# ---------------------------------------------------------------------------
def cmd_prepare(
    target: str,
    *,
    no_images: bool = False,
    pdf: str | None = None,
    reuse: str | None = None,
    force: bool = False,
) -> int:
    """生成翻译工作文档 work.md + manifest.json + 回补图片，stdout 输出翻译指引。

    Args:
        target: 源 markdown 路径或 stem。
        no_images: 跳过图片回补（测试/调试；submit 会拒绝出稿）。
        pdf: 手工指定的源 PDF（同名反解不到时用）。
        reuse: 旧 work.md/工作区目录/stem——沉淀进 TM 并回填旧译文（精确优先、模糊
            次优；占位假译不入 TM 不回填）。
        force: 允许覆写**含译文**的工作区（默认拒绝；放行时自动备份 ``*.bak-<ts>``）。

    Returns:
        退出码（0=成功；2=前置不满足）。
    """
    try:
        source_md, stem = resolve_source(target)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    md_text = source_md.read_text(encoding="utf-8")
    doc = units.parse_markdown(md_text)
    try:
        units.verify_slices(md_text, doc)
    except ValueError as exc:
        print(f"[bilingual] prepare 拒绝：{exc}", flush=True)
        return 2
    n_abs = units.mark_abstract(doc)
    backfilled: dict[str, str] = {}
    reuse_stats = tm.BackfillResult()
    reuse_source = tm.ReuseSource()
    if reuse:
        try:
            memory = tm.TranslationMemory(tm_db_path())
            reuse_source = memory.add_from_workspace(reuse)
            reuse_stats = memory.backfill((u.id, u.en) for u in doc.units)
            backfilled = dict(reuse_stats.filled)
        except (FileNotFoundError, tm.TmDependencyError) as exc:
            print(f"[bilingual] --reuse 拒绝：{exc}", flush=True)
            return 2
    if pdf:
        try:
            src_pdf = images.resolve_source_pdf(source_md, explicit=pdf)
        except FileNotFoundError as exc:
            print(f"[bilingual] {exc}", flush=True)
            return 2
    else:
        src_pdf = images.resolve_source_pdf(source_md)
    workdir = manifest.workdir_for(stem)
    workdir.mkdir(parents=True, exist_ok=True)
    rc, note = manifest.guard_translated_workspace(workdir, force=force)
    if note:
        print(note, flush=True)
    if rc != 0:
        return rc
    (workdir / "work.md").write_text(
        manifest.render_work_md(doc, backfilled), encoding="utf-8"
    )
    man = manifest.build_manifest(doc, source_md, no_images)
    try:
        images.backfill_images(
            workdir, doc.images, src_pdf, no_images=no_images, source_md=source_md
        )
    except images.ImageBackfillError as exc:
        print(f"[bilingual] 图片回补失败：{exc}", flush=True)
        return 2
    man["images_closed"] = images.images_closed(workdir, doc.images)
    (workdir / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    n_h = sum(1 for u in doc.units if u.type == "H")
    n_s = sum(1 for u in doc.units if u.type == "S")
    n_c = sum(1 for u in doc.units if u.type in ("C", "T"))
    print(f"[bilingual] 工作文档：{workdir / 'work.md'}", flush=True)
    print(
        f"[bilingual] 单元 {len(doc.units)}（H {n_h} / S {n_s} / C·T {n_c}）"
        f"，透传块 {len(doc.blocks)}，图 {len(doc.images)}",
        flush=True,
    )
    if n_abs:
        print(
            f"[bilingual] 摘要段已标注（{n_abs} 句 role=abstract；出稿前插「摘要/正文起始」标注）",
            flush=True,
        )
    if reuse:
        print(
            f"[bilingual] --reuse 回填 {len(backfilled)}/{len(doc.units)} 句"
            f"（精确 {reuse_stats.exact}、模糊 {reuse_stats.fuzzy}；未匹配保持未译）",
            flush=True,
        )
        skipped = reuse_source.skipped_placeholder + reuse_stats.skipped_placeholder
        if skipped:
            print(
                f"[bilingual] 警告：跳过占位译文 {len(skipped)} 行"
                "（假译类，不入 TM 不回填——请补真实翻译）",
                flush=True,
            )
    print("--- 翻译指引 ---", flush=True)
    print(f"工作文档：{workdir / 'work.md'}", flush=True)
    print(f"规则：{manifest.RULES_TEXT}", flush=True)
    print(
        "行结构护栏：submit 前强制 check-work（行数与行首 ID 不得改动）。",
        flush=True,
    )
    print(f"先读源文档统一术语：{source_md}", flush=True)
    print(f"查进度：bilingual status {stem}", flush=True)
    print(f"完成后出稿：bilingual submit {stem}", flush=True)
    return 0


def tm_db_path() -> Path:
    """TM 持久库路径（``bilingual/tm.sqlite3``，跨会话沉淀）。"""
    return manifest.workspace_root() / tm.TM_DB_NAME


# ---------------------------------------------------------------------------
# bilingual status / check-work
# ---------------------------------------------------------------------------
def cmd_status(stem: str) -> int:
    """按 ID 对账报进度：计数 + 未译/存疑/占位 ID 列表。

    Returns:
        退出码（0=成功；2=工作区未就绪）。
    """
    try:
        workdir, man = manifest.load_workspace(stem)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    translations = manifest.read_work_translations(workdir / "work.md")
    b = reconcile.reconcile(man, translations)
    total = len(man["units"])
    print(
        f"[bilingual] {stem}：共 {total} 单元——已译 {len(b['done'])}，"
        f"未译 {len(b['untranslated'])}，存疑 {len(b['doubtful'])}",
        flush=True,
    )
    ph = reconcile.placeholder_ids(man, translations)
    if ph:
        print(
            f"占位译文 {len(ph)} 行（假译类，不计已译）——需真实翻译：{' '.join(ph)}",
            flush=True,
        )
    if b["untranslated"]:
        print(f"未译 ID：{' '.join(b['untranslated'])}", flush=True)
    if b["doubtful"]:
        print(f"存疑 ID（无 CJK 字符，请自查）：{' '.join(b['doubtful'])}", flush=True)
    return 0


def cmd_check_work(stem: str) -> int:
    """check-work 护栏：work.md 行结构 × manifest 硬校验（pod-guard 等价物）。

    Returns:
        退出码（0=通过；2=结构破损/工作区未就绪）。
    """
    try:
        workdir, man = manifest.load_workspace(stem)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    errors = manifest.check_work_structure(workdir, man)
    if errors:
        print(f"[bilingual] check-work 未过（{len(errors)} 项）：", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        return 2
    print(f"[bilingual] check-work 通过：{stem}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# bilingual submit
# ---------------------------------------------------------------------------
def cmd_submit(
    stem: str,
    *,
    out_dir: str | Path | None = None,
    skip_check_work: bool = False,
    render_fn: RenderFn | None = None,
) -> int:
    """校验全部 ID 有真译文后生成 bilingual.md 并复制到交付目录（只增不删）。

    Args:
        stem: 工作区 stem。
        out_dir: 自定义交付目录（须在 ``data/`` 或 ``用户的inbox/`` 内，不存在则
            创建）；缺省回退 ``用户的inbox/bilingual/<stem>/``（用户全权管理区，
            仅最终交付物、只增不删，见 AGENTS.md 内容目录政策）。
        skip_check_work: 显式 override check-work 护栏（默认强制执行）。
        render_fn: 渲染函数注入（测试用；缺省转接 :mod:`renderer`）。

    Returns:
        退出码（0=出稿成功；1=未译拒绝出稿；2=前置不满足/结构闸拒绝）。
    """
    # 交付目录先行校验（fail-fast：越界时零副作用，不白跑出稿/渲染）。
    if out_dir is not None:
        try:
            inbox = paths.assert_deliverable_dir(
                Path(out_dir), what="bilingual 交付目录"
            )
        except ValueError as exc:
            print(f"[bilingual] --out-dir 拒绝：{exc}", file=sys.stderr)
            return 2
        label = "自定义交付目录"
    else:
        try:
            inbox = paths.assert_deliverable_dir(
                manifest.inbox_root() / stem, what="bilingual inbox 副本"
            )
        except ValueError as exc:
            print(f"[bilingual] 交付目录越界：{exc}", file=sys.stderr)
            return 2
        label = "用户 inbox"
    try:
        workdir, man = manifest.load_workspace(stem)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    if man.get("no_images") or not man.get("images_closed", True):
        print(
            "[bilingual] 图片未回补/未闭合，拒绝出稿（prepare 勿带 --no-images）",
            flush=True,
        )
        return 2
    errors = manifest.check_work_structure(workdir, man)
    if errors:
        if skip_check_work:
            print(
                f"[bilingual] 警告：--skip-check-work override，check-work 未过（{len(errors)} 项）",
                flush=True,
            )
        else:
            print(
                "[bilingual] check-work 未过，拒绝出稿"
                "（行数与行首 ID 不得改动；确认人工裁决可 --skip-check-work 显式 override）：",
                flush=True,
            )
            for e in errors:
                print(f"  - {e}", flush=True)
            return 2
    translations = manifest.read_work_translations(workdir / "work.md")
    b = reconcile.reconcile(man, translations)
    ph = reconcile.placeholder_ids(man, translations)
    if ph:
        print(
            f"[bilingual] 警告：占位译文 {len(ph)} 行（假译类，不计已译、拒绝出稿）",
            flush=True,
        )
    if b["untranslated"]:
        print(
            f"[bilingual] 拒绝出稿：{len(b['untranslated'])} 个单元未译——"
            f"{' '.join(b['untranslated'])}",
            flush=True,
        )
        return 1
    text = manifest.render_bilingual_md(man, translations)
    out = workdir / "bilingual.md"
    out.write_text(text, encoding="utf-8")
    # 串联渲染（旧任务书 20261023-011225 §二.4）：PDF 失败=警告不阻塞 md 交付。
    pdf = workdir / "bilingual.pdf"
    try:
        render = render_fn or load_renderer()
        render(out, pdf)
    except Exception as exc:  # noqa: BLE001 —— 渲染层任何故障都不阻塞 md 交付
        print(
            f"[bilingual] 警告：PDF 渲染失败（md 交付不受影响）：{exc}", file=sys.stderr
        )
    inbox.mkdir(parents=True, exist_ok=True)
    # 用户侧副本按文档标题命名（md/pdf 同名成对）；工作区内部名不变（渲染契约）。
    base = manifest.output_basename(man, translations)
    shutil.copy2(out, inbox / f"{base}.md")
    print(f"[bilingual] 已出稿：{out}", flush=True)
    print(f"[bilingual] 已复制到{label}：{inbox / f'{base}.md'}", flush=True)
    if pdf.is_file():
        shutil.copy2(pdf, inbox / f"{base}.pdf")
        print(f"[bilingual] PDF 已复制到{label}：{inbox / f'{base}.pdf'}", flush=True)
    return 0


def cmd_reorder_suggest(
    target: str, *, pdf: str | None = None, suggest_fn: Callable[..., object] | None = None
) -> int:
    """reorder-suggest：源 md 文序订正**建议**（转接 :mod:`reorder`，只出建议不改源）。

    Returns:
        退出码（0=出建议；2=前置不满足/失败）。
    """
    try:
        source_md, stem = resolve_source(target)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    try:
        src_pdf = images.resolve_source_pdf(source_md, explicit=pdf)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    if src_pdf is None:
        print("[bilingual] 缺源 PDF，无法推断文序（先走①域 read 提取，或用 --pdf 指定）", flush=True)
        return 2
    workdir = manifest.workdir_for(stem)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        suggest = suggest_fn or load_reorder()
        result = suggest(source_md, src_pdf, workdir)
    except Exception as exc:  # noqa: BLE001 —— 接缝层故障统一报因
        print(f"[bilingual] reorder-suggest 失败：{exc}", flush=True)
        return 2
    print(f"[bilingual] 文序建议已出：{workdir}/reorder-suggested.md + reorder-report.md", flush=True)
    print(f"[bilingual] 摘要：{result}", flush=True)
    print("[bilingual] 只出建议——源 md 零改动，订正请人工校对后落盘", flush=True)
    return 0


# ---------------------------------------------------------------------------
# bilingual render / quality
# ---------------------------------------------------------------------------
def cmd_render(stem: str, *, render_fn: RenderFn | None = None) -> int:
    """bilingual render：``bilingual.md`` → ``bilingual.pdf``（同目录）。

    Returns:
        退出码（0=成功；2=无出稿/渲染失败）。
    """
    workdir = manifest.workdir_for(stem)
    md = workdir / "bilingual.md"
    if not md.is_file():
        print(
            f"[bilingual] 无 bilingual.md（先跑 bilingual submit 出稿）：{md}",
            flush=True,
        )
        return 2
    try:
        render = render_fn or load_renderer()
        pdf = render(md, workdir / "bilingual.pdf")
    except Exception as exc:  # noqa: BLE001 —— 渲染层故障统一报因
        print(f"[bilingual] 渲染失败：{exc}", file=sys.stderr)
        return 2
    print(f"[bilingual] PDF：{pdf}", flush=True)
    return 0


def cmd_quality(
    stem: str,
    *,
    glossary: Path | None = None,
    back_check: bool = False,
    back_translator: Callable[[str], str] | None = None,
    min_ratio: float = 0.5,
) -> int:
    """质量闸门：术语一致性核对（``--glossary``）与回译校验（``--back-check``）。

    回译引擎优雅降级（K2 约定）：无 ``DEEPL_AUTH_KEY``/deepl 未装时跳过并提示，
    不阻塞出稿（主通道译者不受影响）。

    Returns:
        退出码（0=无 fail 级发现；1=有 fail 级发现；2=工作区未就绪/输入错误）。
    """
    try:
        workdir, man = manifest.load_workspace(stem)
    except FileNotFoundError as exc:
        print(f"[bilingual] {exc}", flush=True)
        return 2
    translations = manifest.read_work_translations(workdir / "work.md")
    findings: list[quality.QualityFinding] = []
    if glossary is not None:
        try:
            terms = quality.load_glossary(Path(glossary))
        except (OSError, ValueError) as exc:
            print(f"[bilingual] 术语表载入失败：{exc}", flush=True)
            return 2
        findings += quality.term_consistency_findings(man, translations, terms)
    if back_check and back_translator is None:
        back_translator = quality.make_back_translator()
        if back_translator is None:
            print(
                "[bilingual] 跳过回译校验（.env 无 DEEPL_AUTH_KEY 或 deepl 未装；"
                "主通道翻译不受影响）",
                flush=True,
            )
    if back_translator is not None:
        findings += quality.backtranslation_findings(
            man, translations, back_translator, min_ratio=min_ratio
        )
    for f in findings:
        print(f"[quality] {f.severity} {f.kind} {f.unit_id}：{f.detail}", flush=True)
    if any(f.severity == "fail" for f in findings):
        return 1
    print(f"[bilingual] 质量检查完成：{len(findings)} 项发现", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """argparse 入口（``python -m pysci.skills.literature_research.bilingual.cli``）。"""
    parser = argparse.ArgumentParser(
        prog="bilingual",
        description="中英对照文档生产线（prepare/status/submit/check-work/render/quality）",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare", help="切分 + work.md/manifest + 图片回补 + TM 回填")
    p_prep.add_argument("target", help="源 markdown 路径或 stem（papers/ 提取件）")
    p_prep.add_argument("--pdf", help="手工指定源 PDF")
    p_prep.add_argument("--no-images", action="store_true", help="跳过图片回补（调试）")
    p_prep.add_argument("--reuse", help="旧 work.md/目录/stem——TM 回填旧译文")
    p_prep.add_argument("--force", action="store_true", help="覆写含译文工作区（自动备份）")
    p_stat = sub.add_parser("status", help="对账报进度")
    p_stat.add_argument("stem")
    p_sub = sub.add_parser("submit", help="出稿 + 自动渲染 + 交付副本")
    p_sub.add_argument("stem")
    p_sub.add_argument("--out-dir", help="自定义交付目录（须在 data/ 内）")
    p_sub.add_argument(
        "--skip-check-work",
        action="store_true",
        help="显式 override check-work 护栏（默认强制执行）",
    )
    p_chk = sub.add_parser("check-work", help="work.md 行结构硬校验")
    p_chk.add_argument("stem")
    p_ren = sub.add_parser("render", help="bilingual.md → bilingual.pdf")
    p_ren.add_argument("stem")
    p_reo = sub.add_parser("reorder-suggest", help="源 md 文序订正建议（只出建议）")
    p_reo.add_argument("target", help="源 markdown 路径或 stem（papers/ 提取件）")
    p_reo.add_argument("--pdf", help="手工指定源 PDF")
    p_qua = sub.add_parser("quality", help="术语一致性/回译校验（质量闸门）")
    p_qua.add_argument("stem")
    p_qua.add_argument("--glossary", help="术语表 JSON（{英文术语: 中文译名}）")
    p_qua.add_argument(
        "--back-check", action="store_true", help="回译校验（deepl；无 key 跳过并提示）"
    )
    p_qua.add_argument("--min-ratio", type=float, default=0.5, help="回译相似度下限")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(
            args.target,
            no_images=args.no_images,
            pdf=args.pdf,
            reuse=args.reuse,
            force=args.force,
        )
    if args.command == "status":
        return cmd_status(args.stem)
    if args.command == "submit":
        return cmd_submit(
            args.stem, out_dir=args.out_dir, skip_check_work=args.skip_check_work
        )
    if args.command == "check-work":
        return cmd_check_work(args.stem)
    if args.command == "render":
        return cmd_render(args.stem)
    if args.command == "reorder-suggest":
        return cmd_reorder_suggest(args.target, pdf=args.pdf)
    return cmd_quality(
        args.stem,
        glossary=Path(args.glossary) if args.glossary else None,
        back_check=args.back_check,
        min_ratio=args.min_ratio,
    )


if __name__ == "__main__":
    raise SystemExit(main())
