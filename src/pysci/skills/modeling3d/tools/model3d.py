"""modeling3d 统一 CLI 入口（对齐 theory / figures / imagine 的门面模式）。

对 ``tools/`` 下各模块（config/check/blender/tripo_client）做薄编排，
用一条命令驱动双管线闭环的每一步::

    uv run pysci-model3d doctor
    ... model3d new gain_ep sample_holder --kind print
    ... model3d build src/pysci/research/gain_ep/models/sample_holder.py --strict-volume
    ... model3d check data/research/1_gain_ep/models/sample_holder/stl/sample_holder.stl --expect-bbox 40,20,8
    ... model3d render --mesh <stl/glb> --out <png>
    ... model3d tripo bench_photo.jpg --out scene.glb --dry-run

子命令分组：doctor / new / build / check / render / tripo / list。
build 执行 build123d 脚本并**自动串跑验收门**；render 走 Blender headless。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import tripo_client
from .blender import (
    RENDER_STUDIO_SCRIPT,
    blender_exe,
    blender_version,
    run_blender_script,
)
from .check import check_mesh
from .config import settings

# 建模脚本与产物之间的协议标记：build123d 脚本模板在导出后打印这些行，
# `build` 子命令据此定位 STL 并串跑验收门。改标记须同步 data 模板。
_MARKER_STL = "[model3d] stl: "
_MARKER_VOLUME = "[model3d] volume_mm3: "


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    """环境自检：配置摘要、Blender 可用性、核心库版本、MCP/Tripo 状态。"""
    print(settings.summary())
    print()

    print("--- geometry / mesh backends ---")
    for mod in ("build123d", "trimesh"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12}: {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<12}: MISSING ({e!r})")

    print("\n--- blender ---")
    try:
        print(f"  executable   : {blender_exe()}")
        print(f"  version      : {blender_version()}")
    except Exception as e:  # noqa: BLE001
        print(f"  UNAVAILABLE  : {e}")
    print(
        f"  mcp addon    : {settings.mcp_addon_path or '(未安装，见 SKILL.md MCP 节)'}"
    )
    print(
        f"  studio script: {RENDER_STUDIO_SCRIPT} "
        f"({'OK' if RENDER_STUDIO_SCRIPT.exists() else 'MISSING'})"
    )

    print("\n--- tripo (图生 3D) ---")
    if settings.tripo_ready:
        try:
            balance = tripo_client.check_key()
            print(f"  key          : 有效，balance={balance.get('data')}")
        except tripo_client.TripoError as e:
            print(f"  key          : 已配置但校验失败（{e}）")
            print(f"  hint         : {e.hint()}")
    else:
        print(
            "  key          : (unset) — 在 .env 配 TRIPO_API_KEY 后启用；"
            "`tripo --dry-run` 可先行演练流程"
        )

    print("\n--- paths ---")
    print(f"  project_root : {settings.project_root}")
    print(f"  module_dir   : {settings.module_dir}")
    print(f"  templates    : {settings.templates_dir}")
    return 0


# ---------------------------------------------------------------------------
# new (scaffold)
# ---------------------------------------------------------------------------
def _session_dirs(research: str, slug: str) -> Path:
    from pysci.paths import research_model_dir

    session_dir = research_model_dir(research, slug=slug)
    for sub in ("stl", "step", "renders"):
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    notes = session_dir / "notes.md"
    if not notes.exists():
        notes.write_text(
            f"# {research}/{slug} 建模日志\n\n创建时间：{_now()}\n\n---\n\n",
            encoding="utf-8",
        )
    return session_dir


def _render_template(name: str, mapping: dict[str, str]) -> str:
    """读取 data 模板并做 %TOKEN% 替换（模板含大量花括号，不用 str.format）。"""
    tmpl = settings.templates_dir / name
    if not tmpl.exists():
        raise FileNotFoundError(f"脚手架模板缺失: {tmpl}")
    text = tmpl.read_text(encoding="utf-8")
    for token, value in mapping.items():
        text = text.replace(f"%{token}%", value)
    return text


def cmd_new(args: argparse.Namespace) -> int:
    """脚手架新建模会话：src/ 下生成脚本骨架，data/ 下建产物目录。"""
    research, slug = args.research, args.slug
    session_dir = _session_dirs(research, slug)

    code_dir = settings.research_model_code_dir(research)
    code_dir.mkdir(parents=True, exist_ok=True)
    init_file = code_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(f'"""{research} 3D 建模子包。"""\n', encoding="utf-8")

    script_path = code_dir / f"{slug}.py"
    if script_path.exists() and not args.force:
        print(
            f"[model3d] 脚本已存在: {script_path}（用 --force 覆盖）", file=sys.stderr
        )
        return 1

    title = args.title or f"{research}/{slug} 3D 建模"
    tmpl_name = "print_part.py.tmpl" if args.kind == "print" else "scene.py.tmpl"
    content = _render_template(
        tmpl_name,
        {
            "TITLE": title,
            "RESEARCH": research,
            "SLUG": slug,
            "SESSION_DIR": session_dir.as_posix(),
        },
    )
    script_path.write_text(content, encoding="utf-8")

    print(f"已脚手架建模会话：{slug}（kind={args.kind}）")
    print(f"  代码脚本 : {script_path}")
    print(f"  产物目录 : {session_dir}")
    print("\n下一步：")
    if args.kind == "print":
        print(f"  1) 编辑 {script_path}（填入设计参数与几何，单位毫米）")
        print(f"  2) uv run pysci-model3d build '{script_path}' --strict-volume")
        print("  3) build 会自动串跑验收门；通过后再 `render --mesh` 出概念图")
    else:
        print(f"  1) 编辑 {script_path}（Blender 场景脚本：材质/灯光/相机）")
        print(f"  2) uv run pysci-model3d render '{script_path}'")
    return 0


# ---------------------------------------------------------------------------
# build (CAD 脚本执行 + 自动验收门)
# ---------------------------------------------------------------------------
def _parse_size(raw: str) -> tuple[float, float, float]:
    parts = [float(x) for x in raw.replace("，", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"尺寸应为 L,W,H 三个数：{raw!r}")
    return (parts[0], parts[1], parts[2])


def cmd_build(args: argparse.Namespace) -> int:
    """执行 build123d 建模脚本（项目 .venv），解析产物标记并自动跑验收门。"""
    script = Path(args.script)
    if not script.exists():
        print(f"[model3d] 脚本不存在: {script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script)]
    print(f"[model3d] 执行: {' '.join(cmd)}")
    print("-" * 60)
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(settings.project_root)
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"[model3d] 建模脚本退出码: {result.returncode}", file=sys.stderr)
        return result.returncode
    print("-" * 60)

    # 解析协议标记，定位 STL 并串跑验收门
    stl_path: Path | None = None
    for line in (result.stdout or "").splitlines():
        if line.startswith(_MARKER_STL):
            stl_path = Path(line[len(_MARKER_STL) :].strip())
    if stl_path is None:
        print(
            f"[model3d] 警告：stdout 中没有 '{_MARKER_STL}<path>' 标记，"
            "跳过验收门（脚本模板会自动打印该标记）",
            file=sys.stderr,
        )
        return 0

    from pysci.paths import assert_within_data

    try:
        assert_within_data(stl_path, what="STL 产物")
    except ValueError as e:
        print(f"[model3d] {e}", file=sys.stderr)
        return 1

    expect_volume = args.expect_volume
    if expect_volume is None and args.strict_volume:
        # 从标记行取脚本报告的 B-rep 解析体积作为期望值（三角化后略小，给 1% 容差）
        for line in (result.stdout or "").splitlines():
            if line.startswith(_MARKER_VOLUME):
                expect_volume = float(line[len(_MARKER_VOLUME) :].strip())
    report = check_mesh(
        stl_path,
        expect_volume=expect_volume,
        expect_bbox=_parse_size(args.expect_bbox) if args.expect_bbox else None,
    )
    print(report.format())
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# check (独立验收门)
# ---------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """对既有网格文件跑打印验收门。"""
    report = check_mesh(
        args.mesh,
        expect_volume=args.expect_volume,
        volume_rtol=args.volume_rtol,
        expect_bbox=_parse_size(args.expect_bbox) if args.expect_bbox else None,
        bbox_atol=args.bbox_atol,
        require_watertight=not args.no_watertight,
    )
    print(report.format())
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# render (Blender headless)
# ---------------------------------------------------------------------------
def cmd_render(args: argparse.Namespace) -> int:
    """渲染概念图：``render <scene.py>`` 跑自定义脚本；``render --mesh m.stl`` 快速出图。"""
    if args.script:
        # 自定义 Blender 场景脚本模式：-- 之后的参数原样透传
        return run_blender_script(
            args.script, extra_args=args.blender_args, stream=True
        )

    if not args.mesh:
        print("[model3d] 需要 <scene.py> 或 --mesh <网格文件>", file=sys.stderr)
        return 2
    mesh = Path(args.mesh).resolve()  # Blender 以自身 CWD 解析相对路径，必须传绝对路径
    if not mesh.exists():
        print(f"[model3d] 网格文件不存在: {mesh}", file=sys.stderr)
        return 2

    out = (
        Path(args.out).resolve()
        if args.out
        else mesh.parent
        / "renders"
        / f"{mesh.stem}_{settings.render_engine.lower()}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    from pysci.paths import assert_within_data

    assert_within_data(out, what="渲染图")

    extra = [
        "--mesh",
        str(mesh),
        "--out",
        str(out),
        "--engine",
        args.engine or settings.render_engine,
        "--samples",
        str(args.samples or settings.render_samples),
        "--width",
        str(args.width or settings.render_size[0]),
        "--height",
        str(args.height or settings.render_size[1]),
    ]
    if args.bg:
        extra += ["--bg", args.bg]
    rc = run_blender_script(RENDER_STUDIO_SCRIPT, extra_args=extra, stream=True)
    if rc == 0 and out.exists():
        print(f"\n[model3d] render: {out}")
        print(f"视觉校验：Read '{out}'")
    return rc


# ---------------------------------------------------------------------------
# tripo (图生 3D / 文生 3D)
# ---------------------------------------------------------------------------
def cmd_tripo(args: argparse.Namespace) -> int:
    """Tripo 云生成：参考图（默认）或文本 prompt → GLB，落盘后可进验收门/渲染。"""
    source = args.source
    is_file = Path(source).exists()
    image = source if (is_file and not args.text) else None
    prompt = None if image else source

    out = (
        Path(args.out)
        if args.out
        else (
            settings.cache_dir
            / f"tripo_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.glb"
        )
    )

    if args.dry_run:
        print("=== tripo dry-run（不联网、不计费）===")
        print(f"  mode         : {'image_to_model' if image else 'text_to_model'}")
        print(f"  source       : {image or repr(prompt)}")
        print(f"  out          : {out}")
        print(f"  pbr/quad     : {args.pbr}/{args.quad}")
        print(f"  face_limit   : {args.face_limit or '(默认)'}")
        print(
            f"  key          : {'已配置' if settings.tripo_ready else '(未配置 — .env 设 TRIPO_API_KEY)'}"
        )
        print(f"  base_url     : {settings.tripo_base_url}")
        print(
            "  流程         : upload → create_task → poll(queued/running→success) → download GLB+预览图"
        )
        return 0

    try:
        glb = tripo_client.generate_to_file(
            out=out,
            image=image,
            prompt=prompt,
            pbr=args.pbr,
            quad=args.quad,
            face_limit=args.face_limit,
            model_version=args.model_version,
            timeout=args.timeout,
        )
    except tripo_client.TripoError as e:
        print(f"[tripo] 失败: {e}", file=sys.stderr)
        print(f"[tripo] hint: {e.hint()}", file=sys.stderr)
        return 1

    print(f"\n[model3d] tripo GLB: {glb}")
    print(
        "下一步（打印用途须过验收门，云生成网格通常**非流形**，需 Blender 重拓/修网格）："
    )
    print(f"  uv run pysci-model3d check '{glb}' --no-watertight   # 渲染用途仅体检")
    print(f"  uv run pysci-model3d render --mesh '{glb}'            # 快速预览")
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    """列出某研究线的建模会话。"""
    from pysci.paths import research_model_dir

    data_dir = research_model_dir(args.research)
    if not data_dir.is_dir():
        print(f"[model3d] 尚无模型数据目录: {data_dir}")
        print("  用 `pysci-model3d new` 脚手架一个建模会话。")
        return 0

    print(f"=== modeling sessions under {args.research} ===")
    found = False
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if d.name.startswith("."):
            continue
        found = True
        n_stl = len(list((d / "stl").glob("*"))) if (d / "stl").is_dir() else 0
        n_png = (
            len(list((d / "renders").glob("*.png"))) if (d / "renders").is_dir() else 0
        )
        print(f"  {d.name:<32} stl={n_stl}  renders={n_png}")
    if not found:
        print("  (未发现任何建模会话；用 `pysci-model3d new` 脚手架一个)")

    code_dir = settings.research_model_code_dir(args.research)
    if code_dir.is_dir():
        scripts = [s for s in sorted(code_dir.glob("*.py")) if s.name != "__init__.py"]
        if scripts:
            print(
                f"\n=== model scripts in src/pysci/research/{args.research}/models/ ==="
            )
            for s in scripts:
                print(f"  {s.name}")
    return 0


# ---------------------------------------------------------------------------
# argparse 构建
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pysci-model3d",
        description="pySciWS modeling3d CLI — 打印建模 + 概念渲染双管线门面",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="环境自检")
    p_doctor.set_defaults(func=cmd_doctor)

    p_new = sub.add_parser("new", help="脚手架新建模会话")
    p_new.add_argument("research", help="研究线名称（如 gain_ep）")
    p_new.add_argument("slug", help="会话标识（如 sample_holder）")
    p_new.add_argument(
        "--kind",
        choices=("print", "scene"),
        default="print",
        help="print=build123d 打印件脚本（默认）；scene=Blender 场景脚本",
    )
    p_new.add_argument("--title", help="标题（用于脚本 docstring）")
    p_new.add_argument("--force", action="store_true", help="覆盖已有脚本")
    p_new.set_defaults(func=cmd_new)

    p_build = sub.add_parser("build", help="执行 CAD 脚本 + 自动验收门")
    p_build.add_argument("script", help="build123d 建模脚本路径")
    p_build.add_argument("--expect-volume", type=float, help="期望体积 mm³（断言）")
    p_build.add_argument(
        "--strict-volume",
        action="store_true",
        help="以脚本报告的 B-rep 解析体积为期望值自动断言（1%% 容差）",
    )
    p_build.add_argument("--expect-bbox", help="期望包围盒 L,W,H（mm，断言）")
    p_build.set_defaults(func=cmd_build)

    p_check = sub.add_parser("check", help="网格验收门（打印交付前必过）")
    p_check.add_argument("mesh", help="网格文件（.stl/.glb/.obj/.ply）")
    p_check.add_argument("--expect-volume", type=float, help="期望体积 mm³")
    p_check.add_argument("--volume-rtol", type=float, default=0.01, help="体积相对容差")
    p_check.add_argument("--expect-bbox", help="期望包围盒 L,W,H（mm）")
    p_check.add_argument("--bbox-atol", type=float, default=0.1, help="包围盒容差 mm")
    p_check.add_argument(
        "--no-watertight",
        action="store_true",
        help="不强制流形闭合（渲染用途/云生成网格体检）",
    )
    p_check.set_defaults(func=cmd_check)

    p_render = sub.add_parser("render", help="Blender headless 渲染")
    p_render.add_argument(
        "script", nargs="?", help="自定义 Blender 场景脚本（与 --mesh 二选一）"
    )
    p_render.add_argument("--mesh", help="快速模式：直接渲染该网格（studio 布光模板）")
    p_render.add_argument("--out", help="输出 PNG 路径（默认 <mesh>/../renders/）")
    p_render.add_argument("--engine", help="EEVEE | CYCLES（默认取配置）")
    p_render.add_argument("--samples", type=int, help="采样数（Cycles）")
    p_render.add_argument("--width", type=int, help="出图宽 px")
    p_render.add_argument("--height", type=int, help="出图高 px")
    p_render.add_argument("--bg", help="背景：hex 如 0x223344 | studio | transparent")
    p_render.add_argument(
        "blender_args",
        nargs="*",
        help="自定义脚本模式下透传给脚本的参数（置于 -- 之后）",
    )
    p_render.set_defaults(func=cmd_render)

    p_tripo = sub.add_parser("tripo", help="Tripo 图生 3D / 文生 3D → GLB")
    p_tripo.add_argument(
        "source", help="参考图路径（默认）或文本 prompt（配 --text 或图不存在时）"
    )
    p_tripo.add_argument("--out", help="GLB 输出路径（默认技能 cache 目录）")
    p_tripo.add_argument(
        "--text", action="store_true", help="强制按文生 3D 处理 source"
    )
    p_tripo.add_argument("--pbr", action="store_true", help="请求 PBR 材质模型")
    p_tripo.add_argument("--quad", action="store_true", help="请求四边面拓扑")
    p_tripo.add_argument("--face-limit", type=int, help="面数上限")
    p_tripo.add_argument("--model-version", help="模型档位（默认账号缺省档）")
    p_tripo.add_argument("--timeout", type=float, help="轮询超时秒数")
    p_tripo.add_argument(
        "--dry-run", action="store_true", help="只打印请求计划，不联网不计费"
    )
    p_tripo.set_defaults(func=cmd_tripo)

    p_list = sub.add_parser("list", help="列出建模会话")
    p_list.add_argument("research", help="研究线名称")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
