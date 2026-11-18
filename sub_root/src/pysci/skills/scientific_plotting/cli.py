"""figures：③科研绘图统一 CLI 门面（薄路由；入口 ``pysci-figures``）。

职责（选型定稿 + 用户裁决 2026-11-17）：**路由**到薄胶水与社区件，不实现绘图能力——
画图本体在 ultraplot/matplotlib 的图脚本里，本 CLI 只管样式查询、导出件自检、
端标取整换算与依赖探活。旧 figures 全集壳（doctor/styles/new/build/preview/audit/
inspect/overlap/raster-panel/compose-grid/field-panel/list）**不保留**：build/preview
形态约定进 SKILL.md，重判据砍，inspect/imgcheck 随旧 COMSOL 流程废弃。

子命令：

- ``styles``   — 样式资产 + 期刊尺寸契约速查；
- ``audit``    — 对导出件做轻量自检（宽度/字体嵌入），退出码 0=通过（na 不连坐）；
- ``field-tick`` — colorbar 端标取整 v2 换算（ceil_label）；
- ``doctor``   — 绘图栈依赖探活（ultraplot/matplotlib/SciencePlots/pyvista）。

用法::

    pysci-figures styles
    pysci-figures audit out/fig1.pdf out/fig1.eps --journal aps --span double
    pysci-figures field-tick 65.48
    pysci-figures doctor
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec

from pysci.skills.scientific_plotting import audit, field_panel, styles

#: doctor 探活清单（包名，显示名，是否按需件）。
_STACK: tuple[tuple[str, str, bool], ...] = (
    ("ultraplot", "ultraplot（写作面）", False),
    ("matplotlib", "matplotlib（底座）", False),
    ("scienceplots", "SciencePlots（样式资产源）", False),
    ("pyvista", "pyvista（3D/场，按需）", True),
)


def cmd_styles(_args: argparse.Namespace) -> int:
    """``styles``：列出样式资产与期刊尺寸契约。"""
    print("样式资产（叠加顺序：审美层在前，journal-* 契约层最后覆盖）：")
    for name in styles.list_styles():
        print(f"  {name}")
    print("期刊尺寸契约（mm，audit 容差 ±2mm）：")
    for journal, spans in styles.JOURNAL_WIDTHS.items():
        single = spans["single_mm"]
        double = spans["double_mm"]
        print(f"  {journal}: single {single:g} / double {double:g}")
    print(f"家风色循环（{len(styles.HOUSE_CYCLE)} 色，Okabe-Ito 去黄）：")
    print(f"  {' '.join(styles.HOUSE_CYCLE)}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """``audit``：逐文件字体嵌入判定 + 可选宽度契约判定。"""
    findings: list[audit.Finding] = []
    expected_mm = None
    if args.width_mm is not None:
        expected_mm = args.width_mm
    elif args.journal and args.span:
        expected_mm = styles.resolve_width_mm(args.journal, args.span)
    for path in args.files:
        for f in audit.audit_bytes(path):
            findings.append(audit.Finding(f.check, f.status, f"{path}: {f.detail}"))
    if expected_mm is not None:
        findings.append(audit.check_width(args.actual_width_mm, expected_mm))
    report = audit.AuditReport(tuple(findings))
    print(report.format())
    return 0 if report.passed else 1


def cmd_field_tick(args: argparse.Namespace) -> int:
    """``field-tick``：端标取整 v2 换算（逐个入参打印）。"""
    for value in args.values:
        try:
            print(f"{value:g} -> {field_panel.ceil_label(value):g}")
        except ValueError as exc:
            print(f"{value:g} -> 错误：{exc}", file=sys.stderr)
            return 1
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """``doctor``：绘图栈依赖探活（按需件缺失只提示，不判失败）。"""
    missing_required = False
    for module, label, optional in _STACK:
        ok = find_spec(module) is not None
        state = "ok" if ok else ("缺失（按需加装）" if optional else "缺失")
        if not ok and not optional:
            missing_required = True
        print(f"  {label}: {state}")
    print(f"  样式资产目录: {styles.style_dir()}")
    return 1 if missing_required else 0


def build_parser() -> argparse.ArgumentParser:
    """组装 argparse（纯函数，机械可测）。"""
    parser = argparse.ArgumentParser(
        prog="pysci-figures",
        description="③科研绘图薄路由 CLI（画图本体在图脚本 + ultraplot/matplotlib）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("styles", help="样式资产 + 期刊尺寸契约速查").set_defaults(func=cmd_styles)

    p_audit = sub.add_parser("audit", help="导出件轻量自检（字体嵌入/宽度契约）")
    p_audit.add_argument("files", nargs="+", help="导出文件（pdf/eps/svg/png）")
    p_audit.add_argument("--journal", choices=sorted(styles.JOURNAL_WIDTHS), help="期刊族")
    p_audit.add_argument("--span", choices=("single", "double"), help="栏宽")
    p_audit.add_argument("--width-mm", type=float, help="显式契约宽度（mm）")
    p_audit.add_argument(
        "--actual-width-mm", type=float, help="成品实测宽度（mm，与契约比对）"
    )
    p_audit.set_defaults(func=cmd_audit)

    p_tick = sub.add_parser("field-tick", help="colorbar 端标取整 v2 换算")
    p_tick.add_argument("values", nargs="+", type=float, help="端标量值")
    p_tick.set_defaults(func=cmd_field_tick)

    sub.add_parser("doctor", help="绘图栈依赖探活").set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（返回退出码：0 成功/通过，1 失败/缺失，2 参数错误——argparse 占用）。"""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
