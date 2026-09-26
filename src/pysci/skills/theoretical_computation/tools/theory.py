"""theoretical_computation 统一 CLI 入口（对齐 figures / compose / research / simulation 的门面模式）。

对 ``tools/`` 下各模块（config/cas/numerical/eigen/topology/visualize/session）做薄编排，
让我（Agent）与用户都能用一条命令驱动理论计算闭环的每一步::

    uv run pysci-theory doctor
    ... theory new gain_ep ep_band_analysis
    ... theory run src/pysci/research/gain_ep/theory/ep_band_analysis.py --session ep_band_analysis
    ... theory plot data/research/1_gain_ep/theory/ep_band_analysis/results/data.npz
    ... theory list gain_ep

子命令分组：doctor / new / run / plot / list。
run 会执行计算脚本并将产物导出到 session 数据目录；plot 快速可视化已有结果。
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

# CLI 始终无头运行：在导入任何 pyplot 相关模块前锁定 Agg 后端，避免弹窗/阻塞。
import matplotlib

matplotlib.use("Agg")

from .config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    """环境自检：报告核心依赖版本、后端状态。"""
    print(settings.summary())
    print()

    # 核心计算库
    print("--- computation backends ---")
    for mod in ("sympy", "numpy", "scipy"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12}: {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<12}: MISSING ({e!r})")

    # 可视化库
    print("\n--- visualization backends ---")
    for mod in ("matplotlib", "pyvista"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12}: {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<12}: MISSING ({e!r})")

    # matplotlib 后端
    print(f"\n  matplotlib backend : {matplotlib.get_backend()}")

    # pyvista 离屏能力
    try:
        import pyvista as pv

        print(f"  pyvista off_screen : {pv.OFF_SCREEN}")
    except ImportError:
        print("  pyvista off_screen : N/A (not installed)")

    # 路径检查
    print("\n--- paths ---")
    print(f"  project_root : {settings.project_root}")
    print(f"  module_dir   : {settings.module_dir}")
    print(f"  templates    : {settings.templates_dir}")
    print(f"  cache        : {settings.cache_dir}")
    print(f"  recipes      : {settings.recipes_dir}")

    return 0


# ---------------------------------------------------------------------------
# new (scaffold)
# ---------------------------------------------------------------------------
_SCRIPT_TEMPLATE = textwrap.dedent('''\
    """{title}

    计算目标：（请在此描述本次理论计算要解决的问题）
    物理模型：（简述哈密顿量/传递矩阵/色散关系等）
    """

    from pathlib import Path

    import numpy as np
    import sympy as sp

    from pysci.skills.theoretical_computation.tools import cas, eigen, numerical, visualize
    from pysci.skills.theoretical_computation.tools.config import settings
    from pysci.skills.theoretical_computation.tools.session import ensure_session


    def main(session_dir: Path | None = None) -> None:
        """计算入口（由 pysci-theory run 调用，也可直接 python 执行）。"""
        # --- 会话初始化 ---
        if session_dir is None:
            session_dir = ensure_session("{research}", "{slug}")
        print(f"[session] {{session_dir}}")

        # === 1. 定义符号模型 ===
        # TODO: 定义物理参数符号、构建哈密顿量/传递矩阵
        # 示例：
        # omega, kappa, gamma = sp.symbols("omega kappa gamma", real=True)
        # H = sp.Matrix([[omega, kappa], [kappa, -omega]])

        # === 2. 符号推导 ===
        # TODO: 本征值求解、化简、级数展开等
        # eigenvals = H.eigenvals()

        # === 3. 数值化 + 参数空间扫描 ===
        # TODO: 定义 ParamSpace，lambdify，网格求值
        # param_space = numerical.ParamSpace(...)
        # func = numerical.lambdify_expr(expr, param_space)
        # result = numerical.evaluate_on_grid(func, param_space)

        # === 4. 分析与可视化 ===
        # TODO: 分析结果，绘制探索图
        # fig = visualize.quick_plot_2d(x, y, labels=("param", "value"))
        # visualize.export_exploration(fig, session_dir / "plots" / "overview.png")

        print("[done] 计算完成（请填充上方 TODO）")


    if __name__ == "__main__":
        main()
''')


def cmd_new(args: argparse.Namespace) -> int:
    """脚手架新计算会话：在 src/ 下创建脚本骨架，在 data/ 下创建结果目录。"""
    research = args.research
    slug = args.slug

    # 代码目录
    code_dir = settings.research_theory_code_dir(research)
    if not code_dir.exists():
        print(f"[theory] 代码目录不存在，自动创建: {code_dir}", file=sys.stderr)
        code_dir.mkdir(parents=True, exist_ok=True)
        # 确保有 __init__.py
        init_file = code_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                f'"""{research} 理论计算子包。"""\n', encoding="utf-8"
            )

    script_path = code_dir / f"{slug}.py"
    if script_path.exists() and not args.force:
        print(f"[theory] 脚本已存在: {script_path}（用 --force 覆盖）", file=sys.stderr)
        return 1

    # 数据目录
    data_dir = settings.research_theory_data_dir(research) / slug
    (data_dir / "results").mkdir(parents=True, exist_ok=True)
    (data_dir / "plots").mkdir(parents=True, exist_ok=True)

    # 生成脚本
    title = args.title or f"{research}/{slug} 理论计算"
    content = _SCRIPT_TEMPLATE.format(research=research, slug=slug, title=title)
    script_path.write_text(content, encoding="utf-8")

    # 初始化 notes.md
    notes_path = data_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            f"# {title}\n\n计算日志（由 Agent 自动维护）。\n\n---\n\n",
            encoding="utf-8",
        )

    print(f"已脚手架计算会话：{slug}")
    print(f"  代码脚本 : {script_path}")
    print(f"  数据目录 : {data_dir}")
    print(f"  计算日志 : {notes_path}")
    print("\n下一步：")
    print(f"  1) 编辑 {script_path}（填入物理模型与计算逻辑）")
    print(f"  2) uv run pysci-theory run '{script_path}' --session {slug}")
    print(f"  3) Read 产出的 plots/*.png 做视觉校验")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    """执行计算脚本，捕获输出。"""
    script = Path(args.script)
    if not script.exists():
        print(f"[theory] 脚本不存在: {script}", file=sys.stderr)
        return 1

    # 确定 session 目录
    session_dir: Path | None = None
    if args.session:
        # 如果是绝对路径直接用，否则在 research 数据目录下查找
        session_path = Path(args.session)
        if session_path.is_absolute():
            session_dir = session_path
        elif args.research:
            session_dir = settings.research_theory_data_dir(args.research) / args.session
        else:
            # 尝试从脚本路径推断 research
            session_dir = Path(args.session)

    # 构建执行命令
    cmd = [sys.executable, str(script)]
    if session_dir:
        cmd.extend(["--session-dir", str(session_dir)])
        session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[theory] 执行: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(
        cmd,
        capture_output=not args.stream,
        text=True,
        cwd=str(settings.project_root),
    )

    if not args.stream and result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\n[theory] 脚本退出码: {result.returncode}", file=sys.stderr)
        return result.returncode

    print("-" * 60)
    print("[theory] 执行完成。")
    if session_dir:
        plots_dir = session_dir / "plots"
        if plots_dir.is_dir():
            pngs = sorted(plots_dir.glob("*.png"))
            if pngs:
                print(f"\n视觉校验（{len(pngs)} 张图）：")
                for p in pngs[-5:]:  # 最多显示最近 5 张
                    print(f"  Read '{p}'")
    return 0


# ---------------------------------------------------------------------------
# plot (quick visualization of result files)
# ---------------------------------------------------------------------------
def cmd_plot(args: argparse.Namespace) -> int:
    """快速可视化已有结果数据文件。"""
    data_file = Path(args.file)
    if not data_file.exists():
        print(f"[theory] 数据文件不存在: {data_file}", file=sys.stderr)
        return 1

    from . import visualize as _viz

    # 确定输出路径
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = data_file.parent.parent / "plots" / f"{data_file.stem}_quick.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data_file.suffix == ".npz":
        import numpy as np

        data = dict(np.load(str(data_file), allow_pickle=True))
        print(f"[theory] 加载 .npz: keys={list(data.keys())}")
        # 尝试智能绘图：找到数组键并绘制
        fig = _viz.quick_plot_npz(data, title=data_file.stem)
        _viz.export_exploration(fig, out_path, dpi=settings.default_plot_dpi)
    elif data_file.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(data_file)
        print(f"[theory] 加载 .csv: columns={list(df.columns)}, shape={df.shape}")
        fig = _viz.quick_plot_dataframe(df, title=data_file.stem)
        _viz.export_exploration(fig, out_path, dpi=settings.default_plot_dpi)
    else:
        print(f"[theory] 不支持的文件格式: {data_file.suffix}", file=sys.stderr)
        return 1

    print(f"\n视觉校验：Read '{out_path}'")
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    """列出某研究线的计算会话。"""
    research = args.research
    data_dir = settings.research_theory_data_dir(research)

    if not data_dir.is_dir():
        print(f"[theory] 尚无理论计算数据目录: {data_dir}", file=sys.stderr)
        print("  用 `pysci-theory new` 脚手架一个计算会话。")
        return 0

    print(f"=== theory sessions under {research} ===")
    found = False
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if d.name.startswith(".") or d.name == "参考资料":
            continue
        found = True
        results_dir = d / "results"
        plots_dir = d / "plots"
        n_results = len(list(results_dir.glob("*"))) if results_dir.is_dir() else 0
        n_plots = len(list(plots_dir.glob("*.png"))) if plots_dir.is_dir() else 0
        print(f"  {d.name:<32} results={n_results}  plots={n_plots}")

    if not found:
        print("  (未发现任何计算会话；用 `pysci-theory new` 脚手架一个)")

    # 也列出代码侧
    code_dir = settings.research_theory_code_dir(research)
    if code_dir.is_dir():
        scripts = sorted(code_dir.glob("*.py"))
        scripts = [s for s in scripts if s.name != "__init__.py"]
        if scripts:
            print(f"\n=== theory scripts in src/pysci/research/{research}/theory/ ===")
            for s in scripts:
                print(f"  {s.name}")

    return 0


# ---------------------------------------------------------------------------
# argparse 构建
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysci-theory",
        description="pySciWS theoretical_computation CLI — 理论计算闭环门面",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    p_doctor = sub.add_parser("doctor", help="环境自检")
    p_doctor.set_defaults(func=cmd_doctor)

    # new
    p_new = sub.add_parser("new", help="脚手架新计算会话")
    p_new.add_argument("research", help="研究线名称（如 gain_ep）")
    p_new.add_argument("slug", help="计算会话标识（如 ep_band_analysis）")
    p_new.add_argument("--title", help="计算标题（用于脚本 docstring）")
    p_new.add_argument("--force", action="store_true", help="覆盖已有脚本")
    p_new.set_defaults(func=cmd_new)

    # run
    p_run = sub.add_parser("run", help="执行计算脚本")
    p_run.add_argument("script", help="脚本路径")
    p_run.add_argument("--session", help="会话名称或绝对路径")
    p_run.add_argument("--research", help="研究线名称（用于解析相对 session）")
    p_run.add_argument(
        "--stream", action="store_true", help="实时流式输出（不捕获 stdout）"
    )
    p_run.set_defaults(func=cmd_run)

    # plot
    p_plot = sub.add_parser("plot", help="快速可视化结果数据")
    p_plot.add_argument("file", help="数据文件路径（.npz / .csv）")
    p_plot.add_argument("--out", help="输出 PNG 路径")
    p_plot.set_defaults(func=cmd_plot)

    # list
    p_list = sub.add_parser("list", help="列出计算会话")
    p_list.add_argument("research", help="研究线名称")
    p_list.set_defaults(func=cmd_list)

    return parser


# ---------------------------------------------------------------------------
# main 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
