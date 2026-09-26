"""comsol_simulation 统一 CLI 入口（对齐 compose / research 的门面模式）。

对 ``tools/`` 下各模块（config/session/inspect/build/run/export/postprocess/docs）做薄编排，
让我（Agent）与用户都能用一条命令驱动闭环的每一步::

    uv run python -m pysci.skills.comsol_simulation.tools.simulation doctor
    ... simulation inspect tree --mph model.mph
    ... simulation run solve --mph model.mph --study std1
    ... simulation export image --mph model.mph --plotgroup pg10 --out out.png
    ... simulation render mesh.vtk --out render.png --scalars p
    ... simulation docs search "perfectly matched layer"
    ... simulation build recipes

子命令分组：doctor / inspect / run / export / render / post / docs / build。
需要活体 COMSOL 的子命令会按需起 standalone 会话并在结束自动释放（见 session.session）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import build as _build
from . import docs as _docs
from . import export as _export
from . import inspect as _inspect
from . import postprocess as _post
from . import run as _run
from .config import settings
from .session import session


# ---------------------------------------------------------------------------
# 通用：按需开会话加载模型
# ---------------------------------------------------------------------------
def _with_model(mph: str, cores: int, fn) -> Any:
    with session(mode="standalone", cores=cores) as client:
        model = client.load(str(mph))
        return fn(model)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print(settings.summary())
    from .session import check_available

    avail = check_available()
    print(f"\nmph importable & COMSOL discovered : {avail}")
    docs = _docs.list_docs()
    print(f"indexed manuals                    : {len(docs)}")
    for d in docs:
        print(f"    - {d['doc']} ({d['n_sections']} sections, {d['built_at']})")
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
def cmd_inspect_tree(args: argparse.Namespace) -> int:
    print(_with_model(args.mph, args.cores, lambda m: _inspect.dump_tree(m)))
    return 0


def cmd_inspect_params(args: argparse.Namespace) -> int:
    def fn(m):
        params = _inspect.list_parameters(m)
        return "\n".join(f"{k} = {v['value']}  # {v['descr']}" for k, v in params.items())

    print(_with_model(args.mph, args.cores, fn))
    return 0


def cmd_inspect_inventory(args: argparse.Namespace) -> int:
    def fn(m):
        inv = _inspect.inventory(m)
        return "\n".join(f"{k}: {v}" for k, v in inv.items())

    print(_with_model(args.mph, args.cores, fn))
    return 0


def cmd_inspect_java(args: argparse.Namespace) -> int:
    info = _inspect.summarize_java(args.file)
    print(_inspect.format_java_summary(info))
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run_solve(args: argparse.Namespace) -> int:
    def fn(m):
        if args.set_param:
            for kv in args.set_param:
                k, _, v = kv.partition("=")
                _build.set_parameters(m, {k: v})
        sr = _run.solve(m, study=args.study, clear=args.clear)
        print(sr.report())
        return sr.ok

    ok = _with_model(args.mph, args.cores, fn)
    return 0 if ok else 1


def cmd_run_batch(args: argparse.Namespace) -> int:
    br = _run.run_batch(
        args.input,
        args.output,
        study=args.study,
        cores=args.cores,
        tempdir=args.tempdir,
        timeout=args.timeout,
    )
    print(f"batch ok={br.ok} rc={br.returncode} elapsed={br.elapsed:.1f}s out={br.output_file}")
    if br.stdout:
        print("--- stdout ---")
        print(br.stdout[-4000:])
    if br.stderr:
        print("--- stderr ---")
        print(br.stderr[-4000:])
    return 0 if br.ok else 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def cmd_export_image(args: argparse.Namespace) -> int:
    def fn(m):
        r = _export.export_image(m, args.plotgroup, args.out, size=args.size and tuple(args.size))
        print(r.report())
        return r.ok

    return 0 if _with_model(args.mph, args.cores, fn) else 1


def cmd_export_data(args: argparse.Namespace) -> int:
    def fn(m):
        r = _export.export_data(m, args.source, args.out, fmt=args.fmt, expr=args.expr)
        print(r.report())
        return r.ok

    return 0 if _with_model(args.mph, args.cores, fn) else 1


def cmd_export_table(args: argparse.Namespace) -> int:
    def fn(m):
        r = _export.export_table(m, args.table, args.out)
        print(r.report())
        return r.ok

    return 0 if _with_model(args.mph, args.cores, fn) else 1


# ---------------------------------------------------------------------------
# render (pyvista 离屏)
# ---------------------------------------------------------------------------
def cmd_render(args: argparse.Namespace) -> int:
    grid = _post.load_grid(args.file)
    out = _post.render_grid(
        grid,
        args.out,
        scalars=args.scalars,
        cmap=args.cmap,
        show_edges=args.edges,
        clip_normal=args.clip_normal and tuple(args.clip_normal),
        title=args.title,
    )
    print(f"rendered -> {out}")
    return 0


# ---------------------------------------------------------------------------
# post
# ---------------------------------------------------------------------------
def cmd_post_stats(args: argparse.Namespace) -> int:
    if args.csv:
        t = _post.read_comsol_csv(args.csv)
        print(f"columns: {t.columns}; shape: {t.shape}; meta: {t.meta}")
        return 0
    grid = _post.load_grid(args.file)
    print(_post.field_stats(grid, args.scalars))
    return 0


def cmd_post_quality(args: argparse.Namespace) -> int:
    grid = _post.load_grid(args.file)
    for k, v in _post.mesh_quality(grid).items():
        print(f"{k}: {v}")
    return 0


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------
def cmd_docs_convert(args: argparse.Namespace) -> int:
    out = _docs.convert(args.manual, force=args.force)
    print(f"converted -> {out}")
    return 0


def cmd_docs_convert_all(args: argparse.Namespace) -> int:
    res = _docs.convert_all(force=args.force)
    bad = {k: str(v) for k, v in res.items() if isinstance(v, Exception)}
    print(f"ok={[k for k, v in res.items() if not isinstance(v, Exception)]} failed={bad}")
    return 0 if not bad else 1


def cmd_docs_index(args: argparse.Namespace) -> int:
    n = _docs.build_index(rebuild=not args.no_rebuild)
    print(f"indexed sections: {n}")
    return 0


def cmd_docs_search(args: argparse.Namespace) -> int:
    hits = _docs.search(args.query, limit=args.limit, doc=args.doc)
    if not hits:
        print("(no hits)")
        return 0
    for h in hits:
        print(h.report())
    return 0


def cmd_docs_read(args: argparse.Namespace) -> int:
    print(_docs.read(args.doc, heading=args.heading))
    return 0


def cmd_docs_list(args: argparse.Namespace) -> int:
    for d in _docs.list_docs():
        print(f"{d['doc']}: {d['n_sections']} sections @ {d['built_at']}")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def cmd_build_recipes(args: argparse.Namespace) -> int:
    for r in _build.list_recipes():
        print(f"{r.name}: {r.description}  params={r.params}  tags={r.tags}")
    return 0


def cmd_build_apply(args: argparse.Namespace) -> int:
    params = {}
    for kv in args.param or []:
        k, _, v = kv.partition("=")
        params[k] = v

    def fn(m):
        _build.run_recipe(m, args.recipe, params)
        if args.save:
            _run.save_model(m, args.save)
            print(f"saved -> {args.save}")
        return True

    _with_model(args.mph, args.cores, fn)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pysci-simulation",
        description="pySciWS COMSOL 仿真统一 CLI（会话/自省/建模/求解/导出/渲染/后处理/文档）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    p_model = {"type": Path, "help": ".mph 模型文件"}

    sp = sub.add_parser("doctor", help="环境与能力自检")
    sp.set_defaults(func=cmd_doctor)

    # --- inspect ---
    ins = sub.add_parser("inspect", help="模型自省（tree/params/inventory/java）")
    inssub = ins.add_subparsers(dest="ins_cmd", required=True)
    for name, func, needs_model in (
        ("tree", cmd_inspect_tree, True),
        ("params", cmd_inspect_params, True),
        ("inventory", cmd_inspect_inventory, True),
    ):
        s = inssub.add_parser(name, help=f"dump {name}")
        if needs_model:
            s.add_argument("--mph", **p_model, required=True)
            s.add_argument("--cores", type=int, default=None)
        s.set_defaults(func=func)
    s = inssub.add_parser("java", help="解析 GUI 导出的 .java 摘要（不加载模型）")
    s.add_argument("file", type=Path)
    s.set_defaults(func=cmd_inspect_java)

    # --- run ---
    run = sub.add_parser("run", help="求解与批处理（solve/batch）")
    runsub = run.add_subparsers(dest="run_cmd", required=True)
    s = runsub.add_parser("solve", help="加载并求解（可改参/清解）")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument("--study", default=None, help="study/sol tag；默认求解全部")
    s.add_argument("--clear", action="store_true", help="求解前 model.clear()")
    s.add_argument("--set-param", action="append", dest="set_param", help="name=value，可重复")
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_run_solve)
    s = runsub.add_parser("batch", help="comsolbatch.exe 批处理")
    s.add_argument("input", type=Path)
    s.add_argument("--output", type=Path, default=None)
    s.add_argument("--study", default=None)
    s.add_argument("--cores", type=int, default=None)
    s.add_argument("--tempdir", type=Path, default=None)
    s.add_argument("--timeout", type=float, default=None)
    s.set_defaults(func=cmd_run_batch)

    # --- export ---
    ex = sub.add_parser("export", help="导出（image/data/table）")
    exsub = ex.add_subparsers(dest="ex_cmd", required=True)
    s = exsub.add_parser("image", help="绘图组 → PNG")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument("--plotgroup", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--size", type=int, nargs=2, default=None, help="宽 高（像素）")
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_export_image)
    s = exsub.add_parser("data", help="数据集/绘图组 → CSV/VTK")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument("--source", required=True, help="dataset 或 plotgroup tag")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--fmt", default=None, help="csv/txt/vtk")
    s.add_argument("--expr", default=None, help="场表达式（含场值；默认仅坐标）")
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_export_data)
    s = exsub.add_parser("table", help="结果表 → CSV")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument("--table", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_export_table)

    # --- render ---
    s = sub.add_parser("render", help="pyvista 离屏渲染网格/场文件")
    s.add_argument("file", type=Path, help="VTK/VTU/STL/PLY")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--scalars", default=None)
    s.add_argument("--cmap", default="coolwarm")
    s.add_argument("--edges", action="store_true", help="画网格边")
    s.add_argument("--clip-normal", type=float, nargs=3, default=None, dest="clip_normal")
    s.add_argument("--title", default=None)
    s.set_defaults(func=cmd_render)

    # --- post ---
    po = sub.add_parser("post", help="后处理（stats/quality）")
    posub = po.add_subparsers(dest="post_cmd", required=True)
    s = posub.add_parser("stats", help="场统计（VTK 标量或 COMSOL CSV）")
    s.add_argument("--file", type=Path, default=None, help="VTK/VTU 场文件")
    s.add_argument("--csv", type=Path, default=None, help="COMSOL 导出 CSV")
    s.add_argument("--scalars", default=None)
    s.set_defaults(func=cmd_post_stats)
    s = posub.add_parser("quality", help="网格质量统计")
    s.add_argument("file", type=Path)
    s.set_defaults(func=cmd_post_quality)

    # --- docs ---
    do = sub.add_parser("docs", help="手册知识管线（convert/convert-all/index/search/read/list）")
    dosub = do.add_subparsers(dest="docs_cmd", required=True)
    s = dosub.add_parser("convert", help="转换单本手册")
    s.add_argument("manual", help="PRIORITY_MANUALS 相对路径或绝对 PDF 路径")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_docs_convert)
    s = dosub.add_parser("convert-all", help="批量转换核心手册")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_docs_convert_all)
    s = dosub.add_parser("index", help="构建 FTS5 索引")
    s.add_argument("--no-rebuild", action="store_true", dest="no_rebuild")
    s.set_defaults(func=cmd_docs_index)
    s = dosub.add_parser("search", help="FTS5 全文检索")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--doc", default=None)
    s.set_defaults(func=cmd_docs_search)
    s = dosub.add_parser("read", help="读取手册/章节 Markdown")
    s.add_argument("doc")
    s.add_argument("--heading", default=None)
    s.set_defaults(func=cmd_docs_read)
    s = dosub.add_parser("list", help="列出已索引手册")
    s.set_defaults(func=cmd_docs_list)

    # --- build ---
    bu = sub.add_parser("build", help="建模 recipe（recipes/apply）")
    busub = bu.add_subparsers(dest="build_cmd", required=True)
    s = busub.add_parser("recipes", help="列出已注册 recipe")
    s.set_defaults(func=cmd_build_recipes)
    s = busub.add_parser("apply", help="在模型上应用 recipe（可选保存）")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument("--recipe", required=True)
    s.add_argument("--param", action="append", help="name=value，可重复（覆盖默认）")
    s.add_argument("--save", type=Path, default=None, help="应用后保存到此 .mph")
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_build_apply)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # 统一 cores 默认值（None → settings 上限）
    if getattr(args, "cores", None) is None:
        args.cores = settings.comsol_max_cores
    try:
        rc = args.func(args)
        return int(rc) if rc else 0
    except KeyboardInterrupt:
        print("\n[simulation] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
