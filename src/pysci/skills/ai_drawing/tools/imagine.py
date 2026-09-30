"""ai_drawing 统一 CLI 入口（对齐 figures / compose / research / simulation 的门面模式）。

对 ``tools/`` 下各模块（config/postprocess/ledger，及后续的 comfy_*/workflows/providers/bridge）
做薄编排，让我（Agent）与用户都能用一条命令驱动"AI 出图 → 入库记账 → 后处理 → 视觉校验"的闭环::

    uv run pysci-imagine doctor
    ... imagine ingest --src <ImageGen 产物> --slug cover --research gain_ep
    ... imagine adjust data/skills/ai_drawing/assets/cover.png --out ... --resize 1024 0
    ... imagine gallery --sheet
    ... imagine ledger --backend comfyui --limit 10
    ... imagine list

Phase 1 子命令（零外部依赖，纯 Pillow + 账本）：
``doctor / ingest / adjust / sheet / gallery / ledger / list``。
Phase 2 子命令（Tier 1，经 ComfyUI→Ark/即梦）：
``comfy {doctor,nodes,server start|stop|status} / gen / i2i / run / workflows``。
Phase 3 子命令（审美参考 → 数据复现桥）：``palette / bridge``。
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from pysci.paths import assert_within_data, research_artwork_dir

from . import bridge as _bridge
from . import comfy_client as _comfy_client
from . import comfy_session as _comfy_session
from . import ledger as _ledger
from . import postprocess as _pp
from . import providers as _providers
from . import workflows as _workflows
from .config import settings

# 允许的图像后缀（入库/后处理的白名单，避免误收非图文件）。
_IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------
def _unique(dest: Path) -> Path:
    """若 dest 已存在则追加 ``_1/_2/...`` 直到不冲突（避免静默覆盖）。"""
    if not dest.exists():
        return dest
    stem, suffix, parent = dest.stem, dest.suffix, dest.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _probe_url(url: str, timeout: float = 1.0) -> bool:
    """廉价 TCP 探活：仅测试 host:port 是否可连（不发 HTTP、不启服务）。"""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_image(path: Path) -> Path:
    """校验源图存在且后缀受支持，返回解析后的路径。"""
    if not path.is_file():
        raise FileNotFoundError(f"源图不存在：{path}")
    if path.suffix.lower() not in _IMG_SUFFIXES:
        raise ValueError(
            f"不支持的图像后缀 {path.suffix!r}；支持 {sorted(_IMG_SUFFIXES)}"
        )
    return path


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print(settings.summary())
    print()

    print("--- libraries ---")
    for mod, required in (("PIL", True), ("requests", True), ("websocket", False)):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "ok")
            print(f"  {mod:<12}: {ver}")
        except Exception as e:  # noqa: BLE001
            tag = "MISSING" if required else "(not installed, optional)"
            print(f"  {mod:<12}: {tag}" + (f" ({e!r})" if required else ""))

    print("\n--- comfyui orchestrator ---")
    print(f"  configured     : {settings.comfy.found} (source={settings.comfy.source})")
    print(f"  root           : {settings.comfy.root or '(none)'}")
    reachable = _probe_url(settings.comfy_server_url)
    print(f"  server_url     : {settings.comfy_server_url}")
    print(f"  server_reachable: {reachable}" + ("" if reachable else "  (未运行/未配置)"))
    if not reachable:
        print("  提示：ComfyUI 为外部编排器，Phase 2 才需要；用 comfy-cli 独立安装后")
        print("        `comfy launch --background -- --cpu` 起服务，并把 COMFY_ROOT/COMFY_SERVER_URL 写入 .env。")

    print("\n--- provider（火山方舟 / 即梦）---")
    print(f"  default_model  : {settings.default_model}")
    print(f"  default_size   : {settings.default_size}")
    print(f"  ark_api_key    : {'已配置' if settings.ark_ready else '(未配置)'}")
    if not settings.ark_ready:
        print("  提示：ARK_API_KEY 仅用于探测存在性；实际密钥存 ComfyUI-Jimeng-API 节点的 api_keys.json。")

    print("\n--- tier 0（Qoder 内置 ImageGen，零安装兜底）---")
    print("  由 Agent 直接调用 ImageGen 出图，再 `imagine ingest` 入库记账。")
    return 0


# ---------------------------------------------------------------------------
# ingest：把一张已生成的图（ImageGen 产物 / 外部图）入库 + 记账
# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    src = Path(args.src).expanduser()
    try:
        src = _check_image(src)
    except (FileNotFoundError, ValueError) as e:
        print(f"[imagine] ingest 失败：{e}", file=sys.stderr)
        return 1

    slug = args.slug or src.stem
    ext = src.suffix.lower()

    # 目标目录：给了 --research 落研究线 artwork/，否则落技能 assets/
    if args.research:
        dest_dir = research_artwork_dir(args.research)
    else:
        dest_dir = settings.assets_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique(dest_dir / f"{slug}{ext}")
    assert_within_data(dest, what="AI 绘图产物")
    shutil.copy2(src, dest)

    # 可选：同时精选进 gallery
    gallery_copy: Path | None = None
    if args.gallery:
        settings.gallery_dir.mkdir(parents=True, exist_ok=True)
        gallery_copy = _unique(settings.gallery_dir / f"{slug}{ext}")
        shutil.copy2(src, gallery_copy)

    entry = _ledger.record(
        prompt=args.prompt or "",
        seed=args.seed,
        model=args.model or "",
        backend=args.backend or "imagegen",
        workflow=args.workflow or "",
        ref=args.ref or "",
        out=dest,
        notes=args.notes or "",
        extra={"ingested_from": str(src), "slug": slug},
    )

    print(f"已入库：{dest}")
    if gallery_copy:
        print(f"已精选进 gallery：{gallery_copy}")
    print(f"已记账（manifest 第 {len(_ledger.load_all())} 条）：backend={entry.backend} model={entry.model or '-'}")
    print(f"\n✓ 视觉校验：Read '{dest}'")
    return 0


# ---------------------------------------------------------------------------
# adjust：Pillow 单图后处理（crop/resize/rotate/convert/pad/format）
# ---------------------------------------------------------------------------
def cmd_adjust(args: argparse.Namespace) -> int:
    try:
        src = _check_image(Path(args.src).expanduser())
    except (FileNotFoundError, ValueError) as e:
        print(f"[imagine] adjust 失败：{e}", file=sys.stderr)
        return 1
    out = Path(args.out).expanduser()
    assert_within_data(out, what="后处理产物")
    try:
        written = _pp.adjust_image(
            src,
            out,
            crop=tuple(args.crop) if args.crop else None,
            resize=tuple(args.resize) if args.resize else None,
            rotate=args.rotate,
            mode=args.mode,
            pad=tuple(args.pad) if args.pad else None,
            pad_color=tuple(args.pad_color) if args.pad_color else (255, 255, 255),
            quality=args.quality,
            fmt=args.format,
        )
    except ValueError as e:
        print(f"[imagine] adjust 失败：{e}", file=sys.stderr)
        return 1
    print(f"后处理完成：{written}")
    print(f"  ✓ 视觉校验：Read '{written}'")
    return 0


# ---------------------------------------------------------------------------
# sheet：多图拼合 contact sheet（供 Agent 一眼看全）
# ---------------------------------------------------------------------------
def cmd_sheet(args: argparse.Namespace) -> int:
    images = [Path(p).expanduser() for p in args.images]
    out = Path(args.out).expanduser() if args.out else settings.cache_dir / "contact_sheet.png"
    assert_within_data(out, what="contact sheet")
    try:
        written = _pp.contact_sheet(
            images,
            out,
            cols=args.cols,
            thumb=args.thumb,
            pad=args.pad,
            label=not args.no_label,
            bg=tuple(args.bg) if args.bg else (245, 245, 245),
        )
    except ValueError as e:
        print(f"[imagine] sheet 失败：{e}", file=sys.stderr)
        return 1
    print(f"contact sheet：{written}（{len(images)} 张）")
    print(f"  ✓ 视觉校验：Read '{written}'")
    return 0


# ---------------------------------------------------------------------------
# gallery：列出/精选审美范本 + 生成画廊 contact sheet
# ---------------------------------------------------------------------------
def cmd_gallery(args: argparse.Namespace) -> int:
    settings.gallery_dir.mkdir(parents=True, exist_ok=True)

    # --add：把一张图精选进 gallery
    if args.add:
        try:
            src = _check_image(Path(args.add).expanduser())
        except (FileNotFoundError, ValueError) as e:
            print(f"[imagine] gallery --add 失败：{e}", file=sys.stderr)
            return 1
        slug = args.slug or src.stem
        dest = _unique(settings.gallery_dir / f"{slug}{src.suffix.lower()}")
        assert_within_data(dest, what="gallery 范本")
        shutil.copy2(src, dest)
        print(f"已精选进 gallery：{dest}")
        return 0

    imgs = sorted(
        p for p in settings.gallery_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
    )

    # --sheet：把画廊拼成 contact sheet
    if args.sheet:
        if not imgs:
            print("[imagine] gallery 为空，无法生成 contact sheet。", file=sys.stderr)
            return 1
        out = settings.cache_dir / "gallery_sheet.png"
        written = _pp.contact_sheet(imgs, out, cols=args.cols, thumb=args.thumb)
        print(f"画廊 contact sheet：{written}（{len(imgs)} 张）")
        print(f"  ✓ 视觉校验：Read '{written}'")
        return 0

    # 默认：列出画廊
    print(f"=== gallery（{settings.gallery_dir}）===")
    if not imgs:
        print("  (空；用 `imagine gallery --add <img> --slug S` 精选范本)")
        return 0
    for p in imgs:
        print(f"  {p.name}")
    print(f"\n共 {len(imgs)} 张。`imagine gallery --sheet` 生成缩略图总览。")
    return 0


# ---------------------------------------------------------------------------
# ledger：查询/过滤生成账本
# ---------------------------------------------------------------------------
def cmd_ledger(args: argparse.Namespace) -> int:
    if args.render:
        lp = _ledger.render_markdown()
        print(f"已重渲染账本：{lp}")
        return 0
    if args.stats:
        st = _ledger.stats()
        print(f"总记录数：{st['total']}（含 seed：{st['with_seed']}）")
        if st["by_backend"]:
            print("按后端：" + ", ".join(f"{k}={v}" for k, v in sorted(st["by_backend"].items())))
        if st["by_model"]:
            print("按模型：" + ", ".join(f"{k}={v}" for k, v in sorted(st["by_model"].items())))
        return 0

    entries = _ledger.query(
        backend=args.backend,
        model=args.model,
        contains=args.contains,
        limit=args.limit,
    )
    print(_ledger.format_table(entries))
    print(f"\n账本文件：{_ledger.ledger_path()}")
    print(f"机器可读：{_ledger.manifest_path()}")
    return 0


# ---------------------------------------------------------------------------
# list：技能数据区/研究线效果图的产物清单
# ---------------------------------------------------------------------------
def _list_dir_images(d: Path, title: str) -> int:
    if not d.is_dir():
        print(f"=== {title} ===\n  (目录不存在：{d})")
        return 0
    imgs = sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
    )
    print(f"=== {title}（{d}）===")
    if not imgs:
        print("  (空)")
        return 0
    for p in imgs[:200]:
        try:
            rel = p.relative_to(d).as_posix()
        except ValueError:
            rel = p.name
        print(f"  {rel}")
    if len(imgs) > 200:
        print(f"  …（共 {len(imgs)} 张，仅显示前 200）")
    else:
        print(f"  共 {len(imgs)} 张。")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    if args.research:
        return _list_dir_images(research_artwork_dir(args.research), f"artwork: {args.research}")
    _list_dir_images(settings.assets_dir, "assets（生成图入库）")
    print()
    _list_dir_images(settings.gallery_dir, "gallery（精选范本）")
    print()
    wf = sorted(settings.workflows_dir.glob("*.json")) if settings.workflows_dir.is_dir() else []
    print(f"=== workflows（{settings.workflows_dir}）===")
    if not wf:
        print("  (空；Phase 2 保存的 ComfyUI API 格式工作流配方落此)")
    else:
        for p in wf:
            print(f"  {p.name}")
    return 0


# ---------------------------------------------------------------------------
# comfy：编排器子命令（doctor / nodes / server start|stop|status）
# ---------------------------------------------------------------------------
def _client_or_die() -> _comfy_client.ComfyClient:
    """要求服务器可达并返回客户端；不可达则打印指引并 SystemExit(1)。"""
    try:
        _comfy_session.require_available()
    except _comfy_session.SessionError as e:
        print(f"[imagine] {e}", file=sys.stderr)
        raise SystemExit(1) from e
    return _comfy_client.ComfyClient()


def _fmt_spec(spec: object) -> str:
    """把 /object_info 的输入 spec 渲染成紧凑一行（类型 + 选项/元信息）。"""
    if not isinstance(spec, (list, tuple)) or not spec:
        return str(spec)
    typ = spec[0]
    meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(typ, list):
        shown = typ[:8]
        s = f"COMBO {shown}{'…' if len(typ) > 8 else ''}"
    else:
        s = str(typ)
    if meta:
        m = {k: meta[k] for k in ("default", "min", "max", "multiline") if k in meta}
        if m:
            s += f" {m}"
    return s


def cmd_comfy_doctor(args: argparse.Namespace) -> int:
    print("=== ComfyUI 编排器自检 ===")
    avail = _comfy_session.check_available()
    for k, v in avail.items():
        print(f"  {k:<22}: {v}")
    if not avail["port_alive"]:
        print("\n✗ 服务器不可达。启动指引见上 / references/comfyui.md。")
        return 1

    c = _comfy_client.ComfyClient()
    try:
        stats = c.system_stats()
    except _comfy_client.ComfyError as e:
        print(f"\n✗ /system_stats 失败：{e}", file=sys.stderr)
        return 1
    sysinfo = stats.get("system", {})
    print(f"\n  comfyui_version : {sysinfo.get('comfyui_version', '?')}")
    for d in stats.get("devices", []) or []:
        print(f"  device          : {d.get('name')} ({d.get('type')})")

    print("\n--- JimengAI 节点（/object_info 真值）---")
    want = ["JimengAPIClient", "JimengSeedream4", "JimengSeedream5",
            "JimengSeedream3", "JimengQuotaSettings"]
    present: dict[str, bool] = {}
    for cls in want:
        try:
            present[cls] = cls in c.object_info(cls)
        except _comfy_client.ComfyError:
            present[cls] = False
        print(f"  {cls:<20}: {'OK' if present[cls] else 'MISSING'}")

    if present.get("JimengAPIClient"):
        try:
            info = c.object_info("JimengAPIClient")
            kn = info["JimengAPIClient"]["input"]["required"].get("key_name", [[]])[0]
            real = [k for k in kn if k != "Custom"]
            print(f"\n  key_name 选项        : {kn}")
            print(f"  api_keys.json 配置   : {f'已配置 {len(real)} 个' if real else '未配置（仅 Custom）'}")
            if real:
                print(f"  提示：gen/i2i 用 --key-name '{real[0]}'（或 .env 设 COMFY_JIMENG_KEY_NAME）")
        except (KeyError, IndexError, _comfy_client.ComfyError) as e:
            print(f"  key_name 探测失败：{e!r}")
    return 0


def cmd_comfy_nodes(args: argparse.Namespace) -> int:
    c = _client_or_die()
    cls = args.class_type
    if not cls:
        try:
            info = c.object_info()
        except _comfy_client.ComfyError as e:
            print(f"[imagine] comfy nodes 失败：{e}", file=sys.stderr)
            return 1
        names = sorted(info.keys())
        if not args.all:
            names = [n for n in names if "jimeng" in n.lower()]
        print(f"=== 节点 class_type（{len(names)}）===")
        for n in names:
            print(f"  {n}")
        if not args.all:
            print("\n（默认只列 Jimeng 节点；加 --all 列全部）")
        return 0
    try:
        info = c.object_info(cls)
    except _comfy_client.ComfyError as e:
        print(f"[imagine] comfy nodes 失败：{e}", file=sys.stderr)
        return 1
    node = info.get(cls)
    if not node:
        print(f"[imagine] /object_info 无节点 {cls!r}（未安装该自定义节点？）", file=sys.stderr)
        return 1
    print(f"=== {cls} ===")
    print(f"  display_name : {node.get('display_name')}")
    print(f"  category     : {node.get('category')}")
    inp = node.get("input", {})
    for group in ("required", "optional"):
        g = inp.get(group) or {}
        if g:
            print(f"  [{group}]")
            for name, spec in g.items():
                print(f"    {name}: {_fmt_spec(spec)}")
    print(f"  outputs      : {node.get('output_name', node.get('output'))}")
    return 0


def cmd_comfy_server_start(args: argparse.Namespace) -> int:
    try:
        info = _comfy_session.launch(cpu=args.cpu, port=args.port)
    except _comfy_session.SessionError as e:
        print(f"[imagine] comfy server start 失败：{e}", file=sys.stderr)
        return 1
    print(f"ComfyUI 已就绪：{info['url']}  (via {info['launched_via']})")
    print(f"  GUI 协作：浏览器开 {info['url']}")
    return 0


def cmd_comfy_server_stop(args: argparse.Namespace) -> int:
    res = _comfy_session.stop()
    print(f"stop：{res['method']}  (stopped={res['stopped']})")
    return 0


def cmd_comfy_server_status(args: argparse.Namespace) -> int:
    info = _comfy_session.status()
    for k, v in info.items():
        print(f"  {k:<14}: {v}")
    return 0


# ---------------------------------------------------------------------------
# gen / i2i / run：Tier 1 出图（构造图 → /prompt → 等待 → /view 存盘 → 记账）
# ---------------------------------------------------------------------------
def _resolve_dest_dir(args: argparse.Namespace) -> Path:
    """产物落点：--out 目录 > 研究线 artwork/ > 技能 assets/。"""
    out = getattr(args, "out", None)
    if out:
        d = Path(out).expanduser()
    elif getattr(args, "research", None):
        d = research_artwork_dir(args.research)
    else:
        d = settings.assets_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_graph_and_save(
    graph: dict,
    args: argparse.Namespace,
    *,
    prompt: str,
    seed: int | None,
    model: object,
    provider_name: str,
    ref: str,
    workflow_name: str,
    notes: str,
) -> int:
    """提交图、等待、存盘、逐张记账，并打印视觉校验提示。"""
    try:
        _comfy_session.require_available()
    except _comfy_session.SessionError as e:
        print(f"[imagine] {e}", file=sys.stderr)
        return 1
    c = _comfy_client.ComfyClient()
    dest_dir = _resolve_dest_dir(args)
    assert_within_data(dest_dir, what="AI 绘图产物目录")

    print("提交工作流并等待出图（云端生成，约数十秒）…")
    try:
        images = c.run_to_images(
            graph, timeout=args.timeout, progress=not args.no_progress
        )
    except _comfy_client.ComfyError as e:
        print(f"[imagine] 生成失败：{e}", file=sys.stderr)
        return 1

    stem = getattr(args, "slug", None)
    written = _comfy_client.save_images(images, dest_dir, stem=stem)
    api_id = getattr(model, "api_id", "") or (model if isinstance(model, str) else "")
    ui_ver = getattr(model, "ui_version", "")
    size = getattr(args, "size", None)
    for i, w in enumerate(written):
        # 节点内 current_seed = seed + idx（seed==-1 时为随机）；逐张记录以便复现。
        seed_i = (seed + i) if (isinstance(seed, int) and seed >= 0) else seed
        _ledger.record(
            prompt=prompt,
            seed=seed_i,
            model=str(api_id),
            backend="comfyui",
            workflow=workflow_name,
            ref=ref,
            out=w,
            notes=notes,
            extra={
                "provider": provider_name,
                "model_version": ui_ver,
                "size": size,
                "batch": len(written),
                "prompt_id": images[i].get("prompt_id") if i < len(images) else None,
            },
        )
    print(f"已出图 {len(written)} 张 → {dest_dir}")
    for w in written:
        print(f"  {w}")
    print(f"\n✓ 视觉校验：Read '{written[0]}'")
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    try:
        prov = _providers.get_provider(args.provider)
        m = prov.resolve_model(args.model)
    except KeyError as e:
        print(f"[imagine] gen 失败：{e}", file=sys.stderr)
        return 1
    graph = _workflows.txt2img_seedream(
        args.prompt,
        provider=prov,
        model=m,
        size=args.size,
        width=args.width,
        height=args.height,
        seed=args.seed,
        n=args.n,
        key_name=args.key_name,
        filename_prefix=args.filename_prefix or _workflows.DEFAULT_FILENAME_PREFIX,
        thinking=not args.no_thinking,
        watermark=args.watermark,
        group=args.group,
        max_images=args.max_images,
        image_quota=args.quota,
    )
    if args.save_workflow:
        print(f"已存工作流配方：{_workflows.save_workflow(graph, args.save_workflow)}")
    if args.dry_run:
        print(_workflows.graph_summary(graph))
        print("\n（--dry-run：仅构造，未提交。去掉即实际出图）")
        return 0
    return _run_graph_and_save(
        graph, args, prompt=args.prompt, seed=args.seed, model=m,
        provider_name=prov.name, ref="", workflow_name=args.save_workflow or "",
        notes=args.notes or "",
    )


def cmd_i2i(args: argparse.Namespace) -> int:
    try:
        prov = _providers.get_provider(args.provider)
        m = prov.resolve_model(args.model)
    except KeyError as e:
        print(f"[imagine] i2i 失败：{e}", file=sys.stderr)
        return 1
    imgs = [Path(p).expanduser() for p in args.image]
    if not args.dry_run:
        for p in imgs:
            if not p.is_file():
                print(f"[imagine] i2i 源图不存在：{p}", file=sys.stderr)
                return 1

    common = dict(
        provider=prov, model=m, size=args.size, width=args.width, height=args.height,
        seed=args.seed, n=args.n, key_name=args.key_name,
        filename_prefix=args.filename_prefix or _workflows.DEFAULT_FILENAME_PREFIX,
        thinking=not args.no_thinking, watermark=args.watermark, image_quota=args.quota,
    )
    if args.dry_run:
        graph = _workflows.img2img_seedream(
            args.prompt, [p.name for p in imgs], **common
        )
        if args.save_workflow:
            print(f"已存工作流配方：{_workflows.save_workflow(graph, args.save_workflow)}")
        print(_workflows.graph_summary(graph))
        print("\n（--dry-run：用本地文件名占位，未上传/未提交）")
        return 0

    c = _client_or_die()
    names: list[str] = []
    for p in imgs:
        try:
            up = c.upload_image(p)
        except _comfy_client.ComfyError as e:
            print(f"[imagine] 上传源图失败：{e}", file=sys.stderr)
            return 1
        names.append(up.get("name", p.name))
    graph = _workflows.img2img_seedream(args.prompt, names, **common)
    if args.save_workflow:
        print(f"已存工作流配方：{_workflows.save_workflow(graph, args.save_workflow)}")
    return _run_graph_and_save(
        graph, args, prompt=args.prompt, seed=args.seed, model=m,
        provider_name=prov.name, ref=",".join(str(p) for p in imgs),
        workflow_name=args.save_workflow or "", notes=args.notes or "",
    )


def cmd_run(args: argparse.Namespace) -> int:
    try:
        graph = _workflows.load_workflow(args.workflow)
    except (FileNotFoundError, ValueError) as e:
        print(f"[imagine] run 失败：{e}", file=sys.stderr)
        return 1
    overrides = None
    if args.args:
        try:
            overrides = json.loads(args.args)
        except json.JSONDecodeError as e:
            print(f"[imagine] --args 不是合法 JSON：{e}", file=sys.stderr)
            return 1
        if not isinstance(overrides, dict):
            print("[imagine] --args 应为 JSON 对象", file=sys.stderr)
            return 1
    _workflows.apply_args(graph, overrides)
    if args.dry_run:
        print(_workflows.graph_summary(graph))
        return 0
    gen_id = _workflows._find_gen_node(graph)  # noqa: SLF001
    gi = graph.get(gen_id, {}).get("inputs", {}) if gen_id else {}
    return _run_graph_and_save(
        graph, args, prompt=str(gi.get("prompt", "")), seed=gi.get("seed"),
        model=str(gi.get("model_version", "")), provider_name="", ref="",
        workflow_name=str(args.workflow), notes=args.notes or "",
    )


def cmd_workflows(args: argparse.Namespace) -> int:
    wfs = _workflows.list_workflows()
    print(f"=== workflows（{settings.workflows_dir}）===")
    if not wfs:
        print("  (空；gen/i2i --save-workflow NAME 可保存配方)")
        return 0
    for p in wfs:
        try:
            g = json.loads(p.read_text(encoding="utf-8"))
            n = len(g) if isinstance(g, dict) else "?"
        except (json.JSONDecodeError, OSError):
            n = "?"
        print(f"  {p.name}  ({n} 节点)")
    return 0


# ---------------------------------------------------------------------------
# palette / bridge（Phase 3：审美参考 → 数据复现桥）
# ---------------------------------------------------------------------------
def cmd_palette(args: argparse.Namespace) -> int:
    try:
        info = _bridge.analyze_reference(Path(args.src).expanduser(), n_colors=args.n)
    except FileNotFoundError as e:
        print(f"[imagine] palette 失败：{e}", file=sys.stderr)
        return 1
    pal = info["palette"]
    print(f"范本：{info['path']}")
    print(f"  尺寸：{info['width']}×{info['height']}（{info['orientation']}，w/h={info['aspect']}）")
    print(f"  平均亮度：{info['mean_brightness']}/255")
    print(f"  主色（{len(pal)}，按占比降序）：")
    for i, hx in enumerate(pal):
        role = "主色/背景" if i == 0 else f"强调色 {i}"
        print(f"    {i + 1}. {hx}  {role}")
    if pal:
        print(f"\n用法：管线里 palette.color(\"{pal[0]}\") 直接注入（color() 接受 hex）")
    print("接入绘图复现：imagine bridge --ref <此范本> --research R --slug S")
    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    try:
        res = _bridge.bridge(
            args.ref, args.research, args.slug,
            style=args.style, width=args.width, template=args.template,
            n_colors=args.n, palette_name=args.palette_name,
            copy_ref=not args.no_copy_ref, overwrite=args.overwrite,
        )
    except (FileNotFoundError, KeyError, FileExistsError, ValueError) as e:
        print(f"[imagine] bridge 失败：{e}", file=sys.stderr)
        return 1
    info = res["reference"]
    print("已桥接审美范本 → 绘图管线骨架（待填真实数据）：")
    print(f"  图目录      ：{res['figdir']}")
    print(f"  管线代码    ：{res['pipeline']}")
    print(f"  设计规格    ：{res['design_spec']}")
    if res["ref_copy"]:
        print(f"  范本副本    ：{res['ref_copy']}")
    print(f"  抽取主色    ：{', '.join(info['palette']) or '(无)'}")
    if res["palette_name"]:
        print(f"  已注册调色板：{res['palette_name']}（已绑定图目录 STYLE.yaml）")
    print("\n下一步（视觉闭环）：")
    print(f"  1. 编辑管线填真实数据：{res['pipeline']}")
    print(f"  2. uv run pysci-figures build '{res['figdir']}'")
    print(f"  3. Read 预览对照范本迭代：{res['figdir']}/out/{args.slug}_preview.png")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def _add_gen_opts(s: argparse.ArgumentParser) -> None:
    """gen / i2i 共享的出图选项（provider/model/size/seed/n/落点/护栏/dry-run）。

    ``--group/--max-images``（仅 gen）与 ``--image``（仅 i2i）在各自 parser 里另加。
    """
    s.add_argument("--prompt", required=True, help="正向提示词")
    s.add_argument("--provider", default=None,
                   help=f"provider 名（默认 {_providers.DEFAULT_PROVIDER}）")
    s.add_argument("--model", default=None, help="模型 UI 版本或 API ID（默认 provider 首选）")
    s.add_argument("--size", default=None,
                   help="尺寸选项（如 '2K (adaptive)' / '2048x2048' / 'Custom'；默认 provider 首选）")
    s.add_argument("--width", type=int, default=None, help="仅 --size Custom 时生效")
    s.add_argument("--height", type=int, default=None, help="仅 --size Custom 时生效")
    s.add_argument("--seed", type=int, default=0,
                   help="随机种子（0 为节点默认；-1 触发节点内随机；逐张 seed+i 记账）")
    s.add_argument("--n", type=int, default=1, help="生成张数（受成本护栏 max_images 夹取）")
    s.add_argument("--key-name", dest="key_name", default=None,
                   help="JimengAPIClient.key_name（默认 .env COMFY_JIMENG_KEY_NAME 或 provider 默认）")
    s.add_argument("--filename-prefix", dest="filename_prefix", default=None,
                   help="SaveImage 文件名前缀（落 ComfyUI output/ 子目录）")
    s.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                   help="关闭提示词优化（仅 Seedream 4.0 生效）")
    s.add_argument("--watermark", action="store_true", help="加水印（默认否）")
    s.add_argument("--quota", type=int, default=0,
                   help=">0 时附加 JimengQuotaSettings 护栏（按张数限额）")
    # 落点
    s.add_argument("--out", default=None, help="产物目录（默认研究线 artwork/ 或技能 assets/）")
    s.add_argument("--research", default=None, help="研究线名（落 article/artwork/）")
    s.add_argument("--slug", default=None, help="产物命名前缀（默认用 ComfyUI 返回的文件名）")
    s.add_argument("--notes", default=None, help="自由备注（记账）")
    s.add_argument("--save-workflow", dest="save_workflow", default=None,
                   help="把构造的 API 图另存为工作流配方名（落 workflows/，供 run 复用）")
    # 执行控制
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="仅构造/打印 API 图，不提交（不耗额度、无需服务器）")
    s.add_argument("--timeout", type=float, default=300.0, help="等待出图超时（秒）")
    s.add_argument("--no-progress", dest="no_progress", action="store_true",
                   help="关闭实时进度（回退轮询 /history）")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pysci-imagine",
        description="pySciWS AI 绘图技能：ComfyUI 编排云端图像模型出图 + 后处理 + 生成账本 + 审美→数据桥接。",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # doctor
    s = sub.add_parser("doctor", help="自检：配置/依赖库/ComfyUI 可达性/provider/Tier 0")
    s.set_defaults(func=cmd_doctor)

    # ingest
    s = sub.add_parser("ingest", help="把一张已生成的图入库（assets 或研究线 artwork）并记账")
    s.add_argument("--src", required=True, help="源图路径（如 ImageGen 产物）")
    s.add_argument("--slug", default=None, help="入库命名（默认取源图文件名）")
    s.add_argument("--research", default=None, help="研究线名（给则落 article/artwork/，否则落技能 assets/）")
    s.add_argument("--gallery", action="store_true", help="同时精选进 gallery")
    s.add_argument("--prompt", default=None, help="生成用 prompt（记账）")
    s.add_argument("--seed", type=int, default=None, help="随机种子（记账）")
    s.add_argument("--model", default=None, help="图像模型 ID（记账）")
    s.add_argument("--backend", default="imagegen", help="出图后端（imagegen/comfyui/manual；默认 imagegen）")
    s.add_argument("--workflow", default=None, help="ComfyUI 工作流配方名（记账）")
    s.add_argument("--ref", default=None, help="图生图源图（记账）")
    s.add_argument("--notes", default=None, help="自由备注（记账）")
    s.set_defaults(func=cmd_ingest)

    # adjust
    s = sub.add_parser("adjust", help="Pillow 单图后处理（crop/resize/rotate/convert/pad/format）")
    s.add_argument("src", help="源图路径")
    s.add_argument("--out", required=True, help="目标路径（后缀决定格式，除非给 --format）")
    s.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None)
    s.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="目标尺寸；某边为 0 时按原图长宽比自动")
    s.add_argument("--rotate", type=float, default=None, help="逆时针旋转角度（expand）")
    s.add_argument("--mode", default=None, help="色彩模式转换（RGB/RGBA/L…）")
    s.add_argument("--pad", type=int, nargs=2, metavar=("W", "H"), default=None, help="加边到 WxH 画布并居中")
    s.add_argument("--pad-color", type=int, nargs=3, metavar=("R", "G", "B"), default=None, dest="pad_color")
    s.add_argument("--format", default=None, help="显式输出格式（覆盖后缀推断）")
    s.add_argument("--quality", type=int, default=95, help="JPEG/WEBP 质量（1-100）")
    s.set_defaults(func=cmd_adjust)

    # sheet
    s = sub.add_parser("sheet", help="多图拼合 contact sheet（缩略图网格 + 文件名标签）")
    s.add_argument("images", nargs="+", help="源图路径列表")
    s.add_argument("--out", default=None, help="输出路径（默认 cache/contact_sheet.png）")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--thumb", type=int, default=256)
    s.add_argument("--pad", type=int, default=8)
    s.add_argument("--no-label", action="store_true", dest="no_label")
    s.add_argument("--bg", type=int, nargs=3, metavar=("R", "G", "B"), default=None)
    s.set_defaults(func=cmd_sheet)

    # gallery
    s = sub.add_parser("gallery", help="审美范本画廊：列出 / --add 精选 / --sheet 总览")
    s.add_argument("--add", default=None, help="把一张图精选进 gallery")
    s.add_argument("--slug", default=None, help="精选命名（配合 --add）")
    s.add_argument("--sheet", action="store_true", help="生成画廊 contact sheet")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--thumb", type=int, default=256)
    s.set_defaults(func=cmd_gallery)

    # ledger
    s = sub.add_parser("ledger", help="查询/过滤生成账本（--stats 汇总 / --render 重渲染 MD）")
    s.add_argument("--backend", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--contains", default=None, help="在 prompt/notes/out 中子串匹配")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--stats", action="store_true", help="打印汇总统计")
    s.add_argument("--render", action="store_true", help="由 manifest 重渲染 LEDGER.md")
    s.set_defaults(func=cmd_ledger)

    # list
    s = sub.add_parser("list", help="列出技能数据区（assets/gallery/workflows）或某研究线 artwork 的产物")
    s.add_argument("--research", default=None, help="研究线名（给则列其 article/artwork/）")
    s.set_defaults(func=cmd_list)

    # ------------------------------------------------------------------
    # Phase 2（Tier 1，经 ComfyUI→Ark/即梦）：comfy / gen / i2i / run / workflows
    # ------------------------------------------------------------------
    # comfy（嵌套子命令：doctor / nodes / server）
    s = sub.add_parser("comfy", help="ComfyUI 编排器：doctor / nodes / server {start,stop,status}")
    csub = s.add_subparsers(dest="comfy_cmd", required=True)

    cs = csub.add_parser("doctor", help="编排器自检（/system_stats + Jimeng 节点存在性 + key 配置）")
    cs.set_defaults(func=cmd_comfy_doctor)

    cs = csub.add_parser("nodes", help="内省节点 schema（/object_info 真值）；不带参数列 Jimeng 节点")
    cs.add_argument("class_type", nargs="?", default=None,
                    help="节点 class_type（如 JimengSeedream4）；给则详列其输入 spec")
    cs.add_argument("--all", action="store_true", help="列出全部节点（默认只列含 'jimeng' 的）")
    cs.set_defaults(func=cmd_comfy_nodes)

    cs = csub.add_parser("server", help="无头服务器生命周期（经 comfy-cli）")
    ssub = cs.add_subparsers(dest="server_cmd", required=True)
    ss = ssub.add_parser("start", help="拉起无头 ComfyUI（comfy launch --background -- --cpu）")
    ss.add_argument("--cpu", dest="cpu", action="store_true", default=True,
                    help="传 --cpu（本机无 CUDA，默认开）")
    ss.add_argument("--no-cpu", dest="cpu", action="store_false", help="不传 --cpu（有 CUDA 时）")
    ss.add_argument("--port", type=int, default=None, help="覆盖端口（默认取 COMFY_SERVER_URL 的端口）")
    ss.set_defaults(func=cmd_comfy_server_start)
    ss = ssub.add_parser("stop", help="停止服务器（comfy stop，其次 taskkill 记录的 pid）")
    ss.set_defaults(func=cmd_comfy_server_stop)
    ss = ssub.add_parser("status", help="服务器状态（state file + 端口存活 + CLI 可用性）")
    ss.set_defaults(func=cmd_comfy_server_status)

    # gen（文生图 t2i）
    s = sub.add_parser("gen", help="文生图（t2i）：构造 API 图 → 云端出图 → 存盘记账")
    _add_gen_opts(s)
    s.add_argument("--group", action="store_true", help="启用组图生成（enable_group_generation）")
    s.add_argument("--max-images", dest="max_images", type=int, default=1,
                   help="单组张数（1-15；配合 --group）")
    s.set_defaults(func=cmd_gen)

    # i2i（图生图，参考图式）
    s = sub.add_parser("i2i", help="图生图（参考图式）：先 /upload/image 上传源图再出图")
    s.add_argument("--image", action="append", required=True, metavar="PATH",
                   help="参考图路径（可多次给以传多张，按序连 image_1..image_N）")
    _add_gen_opts(s)
    s.set_defaults(func=cmd_i2i)

    # run（万能注参：跑已存工作流配方）
    s = sub.add_parser("run", help="万能注参：跑已存工作流配方（GUI 'Save (API Format)' 导出最稳）")
    s.add_argument("--workflow", required=True,
                   help="工作流配方名/路径（workflows/ 下，可省 .json）")
    s.add_argument("--args", default=None,
                   help='覆盖参数 JSON 对象（友好键如 {"seed":42,"prompt":"..."}，或点号 {"3.seed":42}）')
    s.add_argument("--out", default=None, help="产物目录")
    s.add_argument("--research", default=None, help="研究线名（落 article/artwork/）")
    s.add_argument("--slug", default=None, help="产物命名前缀")
    s.add_argument("--notes", default=None, help="备注（记账）")
    s.add_argument("--dry-run", dest="dry_run", action="store_true", help="仅打印图，不提交")
    s.add_argument("--timeout", type=float, default=300.0, help="等待出图超时（秒）")
    s.add_argument("--no-progress", dest="no_progress", action="store_true", help="关闭实时进度")
    s.set_defaults(func=cmd_run)

    # workflows（列出配方）
    s = sub.add_parser("workflows", help="列出已存工作流配方（workflows/*.json）")
    s.set_defaults(func=cmd_workflows)

    # ------------------------------------------------------------------
    # Phase 3（审美参考 → 数据复现桥）：palette / bridge
    # ------------------------------------------------------------------
    # palette（离线抽色）
    s = sub.add_parser("palette", help="从一张图抽取主色板（hex，按占比降序）——桥接的离线取色")
    s.add_argument("--src", required=True, help="源图路径（审美范本）")
    s.add_argument("--n", type=int, default=6, help="主色数量（1-16，默认 6）")
    s.set_defaults(func=cmd_palette)

    # bridge（脚手架绘图管线 + 写 design_spec.md）
    s = sub.add_parser("bridge", help="审美范本 → 数据复现桥：脚手架绘图管线 + 写 design_spec.md")
    s.add_argument("--ref", required=True, help="审美范本图路径")
    s.add_argument("--research", required=True, help="研究线名（不含数字前缀）")
    s.add_argument("--slug", required=True, help="图目录名（如 fig1_cover）")
    s.add_argument("--style", default="aps", help="期刊风格预设（aps/nature）")
    s.add_argument("--width", default="double", help="设计宽度 single/double 或毫米数")
    s.add_argument("--template", default=None,
                   help="脚手架模板（single/multi_panel/concept_plus_data；默认 multi_panel）")
    s.add_argument("--n", type=int, default=6, help="抽取主色数量（默认 6）")
    s.add_argument("--palette-name", dest="palette_name", default=None,
                   help="给则把抽取配色注册为该名调色板（持久化）并绑定图目录 STYLE.yaml")
    s.add_argument("--no-copy-ref", dest="no_copy_ref", action="store_true",
                   help="不把范本图复制进图目录")
    s.add_argument("--overwrite", action="store_true", help="图目录/管线已存在时覆盖")
    s.set_defaults(func=cmd_bridge)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
        return int(rc) if rc else 0
    except KeyboardInterrupt:
        print("\n[imagine] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
