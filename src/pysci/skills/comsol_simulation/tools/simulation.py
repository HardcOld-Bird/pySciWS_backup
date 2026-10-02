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

定位（大型重构后）：通用 COMSOL 驱动（load/param/solve/evaluate）主力走已注册的社区 ``comsol`` MCP；
本 CLI 承载差异化层——出版级导出精修 / pyvista 渲染 / 物理校验 / 节点自省 / 配方 / 常驻会话——
并在 MCP 不可用时兜底（自身独立完成整条 load→solve→export 管线；单 license 下二者勿同时运行）。

子命令分组：doctor / inspect / run / export / render / post / docs / build。
需要活体 COMSOL 的子命令会按需起 standalone 会话并在结束自动释放（见 session.session）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import build as _build
from . import docs as _docs
from . import export as _export
from . import inspect as _inspect
from . import mcp_server as _mcp
from . import postprocess as _post
from . import run as _run
from . import session as _session
from .config import settings
from .session import session

# ---------------------------------------------------------------------------
# 通用：按需开会话加载模型
# ---------------------------------------------------------------------------
#: 由 main() 从顶层 --connect-port 写入；非空时所有 _with_model 复用同一常驻 server。
#: 用模块全局避免逐个 cmd_* 调用点传参（十余处）。
_CLI_CONNECT_PORT: int | None = None


def _with_model(mph: str, cores: int, fn) -> Any:
    """开会话加载模型并执行 ``fn(model)``。

    默认 standalone（进程内自启自释）。若 main() 记录了 ``--connect-port``，改连该常驻
    server（复用同一 JVM，省每次 ~30s 冷启动与 license 抖动）。
    """
    if _CLI_CONNECT_PORT:
        with session(mode="connect", port=_CLI_CONNECT_PORT) as client:
            model = client.load(str(mph))
            return fn(model)
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
    print("\n=== 社区 comsol MCP 服务端（path C：HTTP transport）===")
    _print_mcp_status(_mcp.mcp_server_status())
    return 0


# ---------------------------------------------------------------------------
# diagnose（一键聚合诊断）
# ---------------------------------------------------------------------------
#: 高频踩坑检查清单（与 knowledge/java_api_pitfalls.md 同步的精简版）。
_PITFALL_CHECKLIST: tuple[str, ...] = (
    "Surface 色标统一：rangecoloractive=on + rangecolormin/max（勿靠自动标度逐 case 漂移）",
    "PolarGroup 极径统一：axislimits=on + rmin/rmax",
    "PlotGroup2D 无轴限属性：--extent 对 2D 无效，改用 --geom-bbox 反演 extent_recovered",
    "重求解后导出空白：result(pg).set('data', dset) 重绑（run solve 已默认自动做）",
    "硬墙对称面远场：RadiationPattern 开 anglerestr 限 0-180°，防下半平面伪瓣",
    "BPF/Box 选择集为空会静默零场：显式 selection 校验 entities 数",
    "JPype 整数属性须 JInt 包裹（node set 已自动处理）",
)


def cmd_diagnose(args: argparse.Namespace) -> int:
    """单次调用聚合 doctor + tree + inventory + 数据集绑定状态 + pitfalls 清单。

    把原本需多次往返（doctor/inspect tree/inspect inventory + 手工核对 pitfalls）的
    诊断压缩为一次 JVM 会话，减少诊断往返 Token。
    """
    from .session import check_available

    print(settings.summary())
    print(f"\nmph importable & COMSOL discovered : {check_available()}")
    print(f"indexed manuals                    : {len(_docs.list_docs())}")

    def fn(m):
        jm = _inspect._jmodel(m)
        rn = _inspect._safe(getattr(jm, "result", None), default=None)
        dsets = _inspect._tags(_inspect._call_if(rn, "dataset", default=None))
        binds: dict[str, Any] = {}
        for pg in _inspect._tags(rn):
            node = _inspect._safe(jm.result, pg, default=None)
            binds[pg] = (
                _inspect._safe(node.getString, "data", default=None)
                if node is not None
                else None
            )
        return _inspect.dump_tree(m), _inspect.inventory(m), dsets, binds

    tree, inv, dsets, binds = _with_model(args.mph, args.cores, fn)
    print("\n=== model tree ===")
    print(tree)
    print("\n=== inventory ===")
    for k, v in inv.items():
        print(f"{k}: {v}")
    print("\n=== datasets / plotgroup bindings ===")
    print(f"datasets: {dsets}")
    for pg, d in binds.items():
        flag = "OK" if d in dsets else "!! 未绑定或失效（导出将空白）"
        print(f"  {pg}.data = {d!r}  [{flag}]")
    print("\n=== pitfalls 检查清单 ===")
    for line in _PITFALL_CHECKLIST:
        print(f"  - {line}")
    return 0


# ---------------------------------------------------------------------------
# server（跨进程常驻会话：start/stop/status）
# ---------------------------------------------------------------------------
def cmd_server_start(args: argparse.Namespace) -> int:
    info = _session.start_persistent_server(port=args.port, cores=args.cores)
    print(
        f"[server] 常驻 COMSOL server 已就绪：host={info['host']} port={info['port']} pid={info['pid']}"
    )
    print(
        "[server] 后续命令加 --connect-port "
        + str(info["port"])
        + " 即可复用此 JVM（免冷启动）。"
    )
    print(
        "[server] ⚠ 单 license：勿与交互式 COMSOL GUI 同时运行；用完执行 `simulation server stop`。"
    )
    return 0


def cmd_server_stop(args: argparse.Namespace) -> int:
    stopped = _session.stop_persistent_server()
    print(
        "[server] 已终止常驻 server 并清理状态文件。"
        if stopped
        else "[server] 无常驻 server 状态文件（未运行）。"
    )
    return 0


def cmd_server_status(args: argparse.Namespace) -> int:
    info = _session.server_status(port=args.port)
    print(f"state_file : {info['state_file']}")
    print(f"running    : {info['running']}")
    print(f"pid        : {info['pid']}")
    print(f"host:port  : {info['host']}:{info['port']}")
    return 0


# ---------------------------------------------------------------------------
# mcp（社区 comsol MCP 服务端托管：path C / HTTP transport）
# ---------------------------------------------------------------------------
def _print_mcp_status(info: dict[str, Any], *, show_json: bool = False) -> None:
    """把 :func:`mcp_server.mcp_server_status` 的结果渲染成人能快速判断的形式。"""
    if show_json:
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        return
    pid_note = f"alive={info['pid_alive']}"
    if info.get("launcher_pid"):
        pid_note += f", launcher_pid={info['launcher_pid']}"
    if info.get("externally_started"):
        pid_note += ", 非本模块派生"
    detached = info.get("detached")
    if detached is True:
        detach_note = f"已脱离终端（派生方式 {info.get('spawn_method') or '外部'}；本会话结束后仍存活）"
    elif detached is False:
        # 只在 WMI 不可用而降级到 CreateProcess 时才会出现；这种实例活不过终端拆除。
        detach_note = (
            f"⚠ 未脱离（派生方式 {info.get('spawn_method') or '外部'}）——终端拆除时会被静默回收，"
            "请重跑 `mcp ensure --restart`"
        )
    else:
        detach_note = "无法判定"
    print(f"transport     : {info['transport']}  (endpoint {info['endpoint_path']})")
    print(f"url           : {info['url']}")
    print(f"running       : {info['running']}  -- {info['probe_verdict']}")
    print(f"pid           : {info['pid']}  ({pid_note})")
    print(f"detached      : {detach_note}")
    print(
        f"holds_license : {info['holds_license']}  -- 该进程是否已加载 jvm.dll（= COMSOL 是否已被惰性启动）"
    )
    holders = info.get("license_holders") or []
    if holders:
        who = ", ".join(f"PID {h['pid']} ({h['exe']})" for h in holders)
        print(f"license       : 已被占用 -- {who}")
    else:
        print("license       : 空闲（本机无任何进程加载 jvm.dll）")
    print(f"exe           : {info['exe'] or '(未发现)'}")
    print(f"launcher      : {info.get('launcher') or '(未生成)'}")
    print(f"log           : {info['log_file']}")


def _mcp_license_hint(info: dict[str, Any]) -> None:
    """给出单 license 下的下一步建议——这是本项目最容易踩的坑，故每次都说清楚。"""
    if info["holds_license"] or info.get("license_holders"):
        print(
            "[mcp] ⚠ 唯一 license 已被占用：跑 CLI standalone 求解或开 COMSOL GUI 之前，"
            "先 `pysci-simulation mcp stop`（或停掉上面列出的进程）。"
        )
    else:
        print("[mcp] COMSOL 尚未启动，license 空闲：可安全使用 CLI / GUI。")


def cmd_mcp_ensure(args: argparse.Namespace) -> int:
    """幂等地保证社区 comsol MCP 服务端在跑（调用任何 comsol MCP 工具前的第 0 步）。"""
    try:
        info = _mcp.ensure_mcp_server(timeout=args.timeout, restart=args.restart)
    except _mcp.McpServerError as e:
        print(f"[mcp] 失败：{e}", file=sys.stderr)
        return 1
    action = {"started": "已派生新服务端", "reused": "复用已在运行的服务端"}.get(
        str(info.get("action")), str(info.get("action"))
    )
    print(f"[mcp] {action}：{info['url']}")
    _print_mcp_status(info, show_json=args.json)
    if args.json:
        return 0
    if info.get("action") == "started":
        print(
            "[mcp] Qoder 的 mcp.json 应把 comsol 注册为 URL 型（否则 Qoder 仍会按 stdio 派生并抢 license）："
        )
        print(info["registration_json"])
    _mcp_license_hint(info)
    return 0


def cmd_mcp_status(args: argparse.Namespace) -> int:
    info = _mcp.mcp_server_status()
    _print_mcp_status(info, show_json=args.json)
    if not args.json:
        _mcp_license_hint(info)
    return 0


def cmd_mcp_stop(args: argparse.Namespace) -> int:
    stopped = _mcp.stop_mcp_server()
    print(
        "[mcp] 已终止服务端并清理状态文件。"
        if stopped
        else "[mcp] 没有发现运行中的服务端（端口无监听）。"
    )
    holders = _mcp.license_holders()
    if holders:
        who = ", ".join(f"PID {h['pid']} ({h['exe']})" for h in holders)
        print(
            f"[mcp] ⚠ license 仍被占用：{who}（可能是 GUI / comsolmphserver / 其它 MCP 实例）。"
        )
    else:
        print("[mcp] license 已空闲。")
    return 0


def cmd_license(args: argparse.Namespace) -> int:
    """谁在占用唯一 COMSOL license（jvm.dll 判据，不启动任何进程）。"""
    holders = _mcp.license_holders()
    if args.json:
        print(json.dumps(holders, indent=2, ensure_ascii=False))
        return 0
    if not holders:
        print("license 空闲：本机无任何进程加载 jvm.dll。")
        return 0
    print(f"license 被 {len(holders)} 个进程占用（单机只有 1 个）：")
    for h in holders:
        print(f"  PID {h['pid']:>7}  {h['exe']}")
        print(f"           {h['jvm']}")
    print(
        "判据：只认 jvm.dll。_jpype.pyd 在 import mph 时就加载，与 license 无关；"
        "MPh standalone 的 JVM 在 python.exe 进程内，所以“看不到 COMSOL 进程”不等于“license 空闲”。"
    )
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
        return "\n".join(
            f"{k} = {v['value']}  # {v['descr']}" for k, v in params.items()
        )

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


def cmd_inspect_node(args: argparse.Namespace) -> int:
    def fn(m):
        node = _inspect.resolve_path(m, args.path)
        info = _inspect.dump_node(node, include_methods=args.methods)
        return _inspect.format_node_dump(info, path=args.path)

    print(_with_model(args.mph, args.cores, fn))
    return 0


# ---------------------------------------------------------------------------
# node（通用属性 setter，与 inspect node 对称）
# ---------------------------------------------------------------------------
def cmd_node_set(args: argparse.Namespace) -> int:
    def fn(m):
        res = _inspect.set_node_props(m, args.path, args.set)
        print(res.report())
        if args.save:
            _run.save_model(m, args.save)
            print(f"saved -> {args.save}")
        return res.all_ok

    return 0 if _with_model(args.mph, args.cores, fn) else 1


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
    print(
        f"batch ok={br.ok} rc={br.returncode} elapsed={br.elapsed:.1f}s out={br.output_file}"
    )
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
        r = _export.export_image(
            m,
            args.plotgroup,
            args.out,
            size=args.size and tuple(args.size),
            extent=tuple(args.extent) if args.extent else None,
            clean=args.clean,
            sidecar=not args.no_sidecar,
            color_range=tuple(args.color_range) if args.color_range else None,
            polar_rmax=(0.0, args.polar_rmax) if args.polar_rmax is not None else None,
            geom_bbox=tuple(args.geom_bbox) if args.geom_bbox else None,
        )
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


def cmd_post_framebox(args: argparse.Namespace) -> int:
    box = _post.detect_frame_box(args.image)
    if box is None:
        print("framebox: 未检测到轴框像素框", file=sys.stderr)
        return 1
    print(f"crop_box_px (x0,y0,x1,y1, origin 顶左): {list(box)}")
    if args.sidecar:
        import json

        p = Path(args.sidecar)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data["crop_box_px"] = list(box)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"merged -> {p}")
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
    print(
        f"ok={[k for k, v in res.items() if not isinstance(v, Exception)]} failed={bad}"
    )
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
    p.add_argument(
        "--connect-port",
        type=int,
        default=None,
        dest="connect_port",
        metavar="PORT",
        help="连接已运行的常驻 COMSOL server（`server start` 拉起）复用同一 JVM，免每次冷启动/license 抖动",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    p_model = {"type": Path, "help": ".mph 模型文件"}

    sp = sub.add_parser("doctor", help="环境与能力自检")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser(
        "diagnose", help="一键聚合诊断（doctor+tree+inventory+绑定状态+pitfalls）"
    )
    sp.add_argument("--mph", **p_model, required=True)
    sp.add_argument("--cores", type=int, default=None)
    sp.set_defaults(func=cmd_diagnose)

    # --- server（跨进程常驻会话）---
    sv = sub.add_parser(
        "server",
        help="常驻 server 生命周期（start/stop/status）——多步 CLI 复用同一 JVM",
    )
    svsub = sv.add_subparsers(dest="server_cmd", required=True)
    s = svsub.add_parser(
        "start", help="拉起常驻 server 并记录 pid/port（勿与 GUI 同跑）"
    )
    s.add_argument("--port", type=int, default=2036)
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_server_start)
    s = svsub.add_parser("stop", help="终止常驻 server 并清理状态文件")
    s.set_defaults(func=cmd_server_stop)
    s = svsub.add_parser("status", help="查询常驻 server 状态（running/pid/port）")
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_server_status)

    # --- mcp（社区 comsol MCP 服务端托管）---
    mc = sub.add_parser(
        "mcp",
        help="社区 comsol MCP 服务端托管（ensure/status/stop）——path C：HTTP transport，COMSOL 惰性启动",
    )
    mcsub = mc.add_subparsers(dest="mcp_cmd", required=True)
    s = mcsub.add_parser(
        "ensure",
        help="幂等：不在跑就脱离终端派生一个，在跑就复用（调用任何 comsol MCP 工具前的第 0 步）",
    )
    s.add_argument(
        "--timeout", type=float, default=90.0, help="等待就绪的秒数（默认 90）"
    )
    s.add_argument(
        "--restart",
        action="store_true",
        help="先停掉已有实例再派生（改了 transport/port 后用）",
    )
    s.add_argument(
        "--json", action="store_true", help="输出原始 JSON（供脚本/agent 解析）"
    )
    s.set_defaults(func=cmd_mcp_ensure)
    s = mcsub.add_parser(
        "status", help="查询服务端状态 + license 占用（只读，不派生任何进程）"
    )
    s.add_argument("--json", action="store_true", help="输出原始 JSON")
    s.set_defaults(func=cmd_mcp_status)
    s = mcsub.add_parser(
        "stop", help="终止服务端并清理状态文件（同时释放它可能已占的 license）"
    )
    s.set_defaults(func=cmd_mcp_stop)

    # --- license（只读）---
    s = sub.add_parser(
        "license", help="谁在占用唯一 COMSOL license（jvm.dll 判据，只读）"
    )
    s.add_argument("--json", action="store_true", help="输出原始 JSON")
    s.set_defaults(func=cmd_license)

    # --- inspect ---
    ins = sub.add_parser("inspect", help="模型自省（tree/params/inventory/node/java）")
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
    s = inssub.add_parser(
        "node", help="按路径 dump 节点状态（类型/属性值/枚举允许值/选择实体数）"
    )
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument(
        "--path",
        required=True,
        help="点分路径，段语法 name 或 name(tag)，如 component(comp1).physics(acpr).feature(bpf1)",
    )
    s.add_argument(
        "--methods", action="store_true", help="附带反射方法签名（活体 javadoc）"
    )
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_inspect_node)

    # --- node ---
    nd = sub.add_parser("node", help="通用节点写入（set）")
    ndsub = nd.add_subparsers(dest="node_cmd", required=True)
    s = ndsub.add_parser("set", help="按路径逐条设置节点属性（与 inspect node 对称）")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument(
        "--path",
        required=True,
        help="点分路径，段语法 name 或 name(tag)，如 result(pg_field).feature(surf1)",
    )
    s.add_argument(
        "--set",
        action="append",
        dest="set",
        required=True,
        metavar="K=V",
        help="属性 key=value，可重复；整数值自动用 JInt 包裹",
    )
    s.add_argument(
        "--save", type=Path, default=None, help="设置后另存到此 .mph（不传则不落盘）"
    )
    s.add_argument("--cores", type=int, default=None)
    s.set_defaults(func=cmd_node_set)

    # --- run ---
    run = sub.add_parser("run", help="求解与批处理（solve/batch）")
    runsub = run.add_subparsers(dest="run_cmd", required=True)
    s = runsub.add_parser("solve", help="加载并求解（可改参/清解）")
    s.add_argument("--mph", **p_model, required=True)
    s.add_argument("--study", default=None, help="study/sol tag；默认求解全部")
    s.add_argument("--clear", action="store_true", help="求解前 model.clear()")
    s.add_argument(
        "--set-param", action="append", dest="set_param", help="name=value，可重复"
    )
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
    s.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=None,
        help="显式轴限 x0 x1 y0 y1（数据坐标；对 2D 绘图组无效，改用 --geom-bbox）",
    )
    s.add_argument(
        "--color-range",
        type=float,
        nargs=2,
        default=None,
        dest="color_range",
        metavar="MIN MAX",
        help="统一 Surface 色标（多 case 共享 colorbar 标度）",
    )
    s.add_argument(
        "--polar-rmax",
        type=float,
        default=None,
        dest="polar_rmax",
        metavar="R",
        help="统一极坐标绘图组极径上限（rmin=0）",
    )
    s.add_argument(
        "--geom-bbox",
        type=float,
        nargs=4,
        default=None,
        dest="geom_bbox",
        metavar="X0 X1 Y0 Y1",
        help="几何包围盒；反演 auto-zoom 窗口写入 sidecar 的 extent_recovered",
    )
    s.add_argument("--clean", action="store_true", help="隐藏 colorbar/图例/标题")
    s.add_argument(
        "--no-sidecar",
        action="store_true",
        dest="no_sidecar",
        help="不写 .sidecar.json",
    )
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
    s.add_argument(
        "--clip-normal", type=float, nargs=3, default=None, dest="clip_normal"
    )
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
    s = posub.add_parser(
        "framebox", help="检测渲染 PNG 轴框像素框（pixel↔data 映射/裁剪用）"
    )
    s.add_argument("--image", type=Path, required=True)
    s.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="可选：把 crop_box_px 合并进该 sidecar json",
    )
    s.set_defaults(func=cmd_post_framebox)

    # --- docs ---
    do = sub.add_parser(
        "docs", help="手册知识管线（convert/convert-all/index/search/read/list）"
    )
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
    global _CLI_CONNECT_PORT
    parser = build_parser()
    args = parser.parse_args(argv)
    # 顶层 --connect-port：非空时所有活体命令改连常驻 server（复用同一 JVM）
    _CLI_CONNECT_PORT = getattr(args, "connect_port", None)
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
