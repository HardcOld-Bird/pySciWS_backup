"""ai_drawing 统一 CLI 入口（对齐 figures / compose / research / simulation 的门面模式）。

对 ``tools/`` 下各模块（config / ark_client / imaging / postprocess / ledger / bridge）
做薄编排，让我（Agent）与用户都能用一条命令驱动
"云端出图 → 入库记账 → 位图加工 → 视觉校验 → 桥接数据复现"的闭环::

    uv run pysci-imagine doctor
    ... imagine gen --prompt "..." --research gain_ep --slug cover
    ... imagine i2i --image ref.png --prompt "..." --slug v2
    ... imagine mark --src cover.jpg --rect 120,80,400,300 --arrow 900,200,700,400
    ... imagine edit --image marked.png --prompt "将框选区域 A 改为..."
    ... imagine layers --image poster.png --prompt "拆分为 N 个透明图层：..."
    ... imagine img fuse --src 素材.png --base 实拍.jpg --out fused.png
    ... imagine gallery --add fused.png --slug hero
    ... imagine bridge --ref hero.png --research gain_ep --slug fig1_cover

子命令分组：
- **自检/清单**：``doctor`` / ``list`` / ``models`` / ``ledger`` / ``gallery``
- **云端出图**（火山方舟 Seedream，直连 OpenAI 兼容 API）：``gen`` / ``i2i`` / ``edit`` / ``layers``
- **本地位图加工**：``adjust``（Pillow，核心依赖）/ ``img``（skimage+OpenCV，可选依赖）/
  ``mark``（PIL 画编辑标记，零依赖）/ ``sheet``（contact sheet）
- **入库与桥接**：``ingest`` / ``palette`` / ``bridge``

设计取舍：本技能**不经 ComfyUI 编排**（本机无 CUDA GPU，节点图的本地推理优势无法发挥；
且方舟 API 已内置图层拆分/交互编辑/组图，能力面宽于第三方节点）。将来若接入 GPU，
可零代码注册官方 ``comfy-mcp``，详见 ``references/backends.md``。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from pysci.paths import assert_within_data, research_artwork_dir

from . import ark_client as _ark
from . import bridge as _bridge
from . import imaging as _img
from . import ledger as _ledger
from . import postprocess as _pp
from .config import settings

# 允许的图像后缀（入库/后处理的白名单，避免误收非图文件）。
_IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

#: 图层拆分 / 交互编辑需要 5.0-pro（或 flash）；其余能力 4.0 即可且更便宜。
PRO_MODEL_HINT: str = "doubao-seedream-5-0-pro-260628"

#: 交互编辑标记的默认配色（红框最醒目，且在 AI 图上对比度足够）。
_MARK_DEFAULT_COLOR = (230, 40, 40)


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


def _check_image(path: Path) -> Path:
    """校验源图存在且后缀受支持，返回解析后的路径。"""
    if not path.is_file():
        raise FileNotFoundError(f"源图不存在：{path}")
    if path.suffix.lower() not in _IMG_SUFFIXES:
        raise ValueError(
            f"不支持的图像后缀 {path.suffix!r}；支持 {sorted(_IMG_SUFFIXES)}"
        )
    return path


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


def _resolve_out_path(args: argparse.Namespace, default_name: str) -> Path:
    """单文件产物落点：--out 给全路径则用之，否则落进 _resolve_dest_dir。"""
    out = getattr(args, "out", None)
    if out:
        p = Path(out).expanduser()
        if p.suffix.lower() in _IMG_SUFFIXES:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
    return _resolve_dest_dir(args) / default_name


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print(settings.summary())
    print()

    print("--- libraries ---")
    for mod, required in (("PIL", True), ("requests", True), ("numpy", True)):
        try:
            m = __import__(mod)
            print(f"  {mod:<14}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            tag = "MISSING" if required else "(optional)"
            print(f"  {mod:<14}: {tag}" + (f" ({e!r})" if required else ""))
    for name, ver in _img.available().items():
        print(
            f"  {name:<14}: {ver or '(not installed, optional — uv pip install -e ".[imaging]")'}"
        )

    print("\n--- provider（火山方舟 / 即梦 Seedream）---")
    print(f"  endpoint       : {settings.images_endpoint}")
    print(f"  default_model  : {settings.default_model}")
    print(f"  default_size   : {settings.default_size}")
    print(f"  ark_api_key    : {'已配置' if settings.ark_ready else '(未配置)'}")
    if not settings.ark_ready:
        print(
            "  ✗ 云端出图需要 ARK_API_KEY：控制台「API Key 管理」创建（Secret 仅显示一次），"
        )
        print("    并到「开通管理」→「图像生成」逐个开通模型，然后写入项目根 .env。")
    else:
        print(
            "  ✓ 已配置。首跑建议：imagine gen --prompt 'a red circle' --size 1K --dry-run"
        )
    print(
        f"\n  成本护栏 max_images = {settings.max_images}（--n/--max-images 会被此值夹取）"
    )

    print("\n--- tier 0（Qoder 内置 ImageGen，零安装兜底）---")
    print("  由 Agent 直接调用 ImageGen 出图，再 `imagine ingest` 入库记账。")
    return 0


# ---------------------------------------------------------------------------
# models：模型能力矩阵（导航用；--live 直接读方舟 /models 核对真值）
# ---------------------------------------------------------------------------
_MODEL_MATRIX: tuple[tuple[str, str, str], ...] = (
    # 本账号已开通且 2026-10 真实出图验证过的两档放最前；其余按“能否开通”分组标注。
    (
        "doubao-seedream-5-0-flash-260915",
        "文生图/图生图(≤10参考图)；不支持组图/png",
        "**默认主力**（已开通，实测出图）",
    ),
    (
        "doubao-seedream-5-0-pro-260628",
        "文生图/图生图(≤10)/**图层拆分**/**交互编辑**；不支持组图/png",
        "edit/layers 用（已开通，实测出图）",
    ),
    (
        "doubao-seedream-4-5-251128",
        "文生图/图生图(≤14)/组图",
        "本账号未开通；是否仍可开通未核实",
    ),
    (
        "doubao-seedream-3-0-t2i-250415",
        "仅文生图；**唯一 seed 生效**的档位",
        "本账号未开通；是否仍可开通未核实",
    ),
    (
        "doubao-seedream-5-0-260128",
        "4.5 能力 + **output_format=png** + 联网检索 tools",
        "5.0-lite：关闭订阅、即将下架，本账号不可开通",
    ),
    (
        "doubao-seedream-4-0-250828",
        "文生图/图生图(≤14)/组图",
        "4.0：已下架，本账号不可开通",
    ),
    (
        "doubao-seedream-4-0-20260415",
        "4.0 的较新日期快照",
        "4.0：已下架，本账号不可开通",
    ),
)


def cmd_models(args: argparse.Namespace) -> int:
    print("=== 方舟 Seedream 图像模型能力矩阵 ===")
    print(
        "（本表是人工整理的导航，**会滞后**；Model ID 的唯一真值源是方舟控制台「模型列表」，"
    )
    print("  本地用 `imagine models --live` 可直接读账号实际可见的 ID）\n")
    for mid, caps, note in _MODEL_MATRIX:
        mark = " *" if mid == settings.default_model else "  "
        print(f"{mark}{mid}")
        print(f"    能力：{caps}")
        print(f"    说明：{note}")
    print("\n通用约束（方舟侧硬规定）：")
    print(
        f"  seed            : 范围 {list(_ark.SEED_RANGE)}；**仅 3.0-t2i 生效**，且官方声明"
    )
    print(
        "                    '相同 seed 生成类似结果但不保证完全一致' → **不可作复现依据**"
    )
    print(
        "  watermark       : 方舟默认 true；本 CLI 默认显式传 false（--watermark 才开）"
    )
    print("  output_format   : png 仅 5.0 主档支持，其余恒输出 jpeg")
    print("  返回 url        : **24 小时失效** → 本 CLI 一律立即下载落盘")
    print("  参考图上限      : 5.0-pro/flash ≤10，其余 ≤14；参考图数+产出数 ≤15")
    print(f"\n当前默认模型（.env AI_DRAWING_MODEL）：{settings.default_model}")

    if args.live:
        print("\n--- 账号实际可见的图像模型 ID（GET /models，只读免费）---")
        try:
            rows = _ark.list_models()
        except _ark.ArkError as e:
            print(f"[imagine] models --live 失败：{e}", file=sys.stderr)
            return 1
        ids = [str(it.get("id") or "") for it in rows]
        # /models 返回账号**全部**模型（LLM/视频/嵌入都在内）；本技能只关心图像族，
        # 默认过滤，否则一百多行噪音会把真正要看的几行淹掉。
        img_ids = [
            i for i in ids if i.startswith(("doubao-seedream", "doubao-seededit"))
        ]
        shown = ids if args.all else img_ids
        known = {mid for mid, _, _ in _MODEL_MATRIX}
        for mid in shown:
            tag = "" if mid in known else "   ← 本地矩阵未收录，能力请查控制台文档"
            print(f"  {mid}{tag}")
        tail = "" if args.all else "（加 --all 连 LLM/视频/嵌入一起看）"
        print(
            f"\n图像模型 {len(img_ids)} 个 / 账号全部模型 {len(ids)} 个{tail}；"
            "矩阵未收录的 ID 一律以控制台文档为准。"
        )
        print(
            "注意：/models 只反映**目录可见**，不反映开通状态——已下架的 ID 也可能在列。"
            "开通与否只能看控制台「开通管理」，或用一次真实调用探测"
            "（授权阶段被拒即未开通，该次不计费）。"
        )
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
    # 与云端出图路径一致：产物内嵌 provenance（PNG tEXt；非 PNG 自动跳过且不改字节）。
    # Tier 0（ImageGen）与外部来源的图正是从这里进库的，同样应该自带出处。
    _ledger.embed_metadata(
        dest,
        {
            "prompt": args.prompt or "",
            "model": args.model or "",
            "backend": args.backend or "imagegen",
        },
    )

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
        recipe=args.recipe or "",
        ref=args.ref or "",
        out=dest,
        notes=args.notes or "",
        extra={"ingested_from": str(src), "slug": slug},
    )
    _ledger.render_markdown()  # record 只追加 JSONL，渲染在操作结束时做一次

    print(f"已入库：{dest}")
    if gallery_copy:
        print(f"已精选进 gallery：{gallery_copy}")
    print(
        f"已记账（manifest 第 {len(_ledger.load_all())} 条）："
        f"backend={entry.backend} model={entry.model or '-'}"
    )
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
    out = (
        Path(args.out).expanduser()
        if args.out
        else settings.cache_dir / "contact_sheet.png"
    )
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
        p
        for p in settings.gallery_dir.iterdir()
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
            print(
                "按后端："
                + ", ".join(f"{k}={v}" for k, v in sorted(st["by_backend"].items()))
            )
        if st["by_model"]:
            print(
                "按模型："
                + ", ".join(f"{k}={v}" for k, v in sorted(st["by_model"].items()))
            )
        return 0

    # 查询前先渲染一次，保证用户 Read LEDGER.md 看到的与 manifest 同步
    _ledger.render_markdown()
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
        p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
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
        return _list_dir_images(
            research_artwork_dir(args.research), f"artwork: {args.research}"
        )
    _list_dir_images(settings.assets_dir, "assets（生成图入库）")
    print()
    _list_dir_images(settings.gallery_dir, "gallery（精选范本）")
    print()
    rc = (
        sorted(settings.recipes_dir.glob("*.md"))
        if settings.recipes_dir.is_dir()
        else []
    )
    print(f"=== recipes（{settings.recipes_dir}）===")
    if not rc:
        print("  (空；流水线配方知识卡片落此，与其余技能同构)")
    else:
        for p in rc:
            print(f"  {p.name}")
    return 0


# ---------------------------------------------------------------------------
# 云端出图（火山方舟直连）：gen / i2i / edit / layers 共用内核
# ---------------------------------------------------------------------------
def _cloud_kwargs(args: argparse.Namespace) -> dict:
    """把 CLI 参数收敛成 ark_client.generate() 的 kwargs。"""
    kw: dict = {
        "model": args.model or settings.default_model,
        "size": args.size
        or (None if (args.width or args.height) else settings.default_size),
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "watermark": args.watermark,
        "output_format": args.output_format,
        "response_format": args.response_format,
        "timeout": args.timeout,
    }
    if args.web_search:
        # 联网检索仅 5.0 主档支持；方舟侧不支持的模型会自行报错，这里不做本地白名单
        kw["tools"] = [{"type": "web_search"}]
    if args.extra:
        try:
            extra = json.loads(args.extra)
        except json.JSONDecodeError as e:
            raise _ark.ArkError(f"--extra 不是合法 JSON：{e}") from e
        if not isinstance(extra, dict):
            raise _ark.ArkError("--extra 应为 JSON 对象")
        kw["extra"] = extra
    return kw


def _cloud_generate(
    args: argparse.Namespace, *, images: list[str] | None, kind: str
) -> int:
    """统一的云端出图执行体：构造 → （dry-run 或）调用 → 落盘 → 记账 → 提示视觉校验。"""
    try:
        kw = _cloud_kwargs(args)
        if images:
            kw["images"] = images
        sequential = bool(getattr(args, "group", False))
        n = int(getattr(args, "n", 1) or 1)
        max_images = getattr(args, "max_images", None)
        calls = 1
        if sequential or max_images:
            kw["sequential"] = sequential
            kw["max_images"] = max_images or n
        else:
            # 非组图：方舟**单次请求只出一张** → 多张靠多次独立请求（各自不同构图），
            # 次数按成本护栏夹取。这与 --group 语义不同：group 是一套连贯组图。
            calls = max(1, min(n, settings.max_images))
            kw["max_images"] = 1
        payload = _ark.build_payload(
            args.prompt, **{k: v for k, v in kw.items() if k != "timeout"}
        )
    except _ark.ArkError as e:
        print(f"[imagine] {kind} 参数校验失败：{e}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, OSError) as e:
        # 参考图读不到 / --extra JSON 非法等：给一句人话，不吐 traceback
        print(f"[imagine] {kind} 参数校验失败：{e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"=== 请求体（{settings.images_endpoint}）===")
        print(json.dumps(_ark._redact_payload(payload), ensure_ascii=False, indent=2))  # noqa: SLF001
        if calls > 1:
            print(
                f"（非组图：将独立发起 {calls} 次请求；--n 已被成本护栏 "
                f"max_images={settings.max_images} 夹取）"
            )
        print("\n（--dry-run：仅构造请求体，未发出。去掉即实际出图并计费）")
        return 0

    dest_dir = _resolve_dest_dir(args)
    assert_within_data(dest_dir, what="AI 绘图产物目录")
    stem = getattr(args, "slug", None) or f"{kind}_{int(time.time())}"
    ref_str = ",".join(str(p) for p in images) if images else ""

    print(f"调用方舟出图（{payload['model']}，约 30–60 s/次，共 {calls} 次）…")
    written: list[Path] = []
    snaps: list[Path] = []
    billed = 0
    tokens = 0
    for i in range(calls):
        tag = f"第 {i + 1}/{calls} 次" if calls > 1 else ""
        try:
            # timeout 已在 kw 里（_cloud_kwargs 统一收）——不要再显式传一次，否则重名
            resp = _ark.generate(args.prompt, **kw)
        except (_ark.ArkError, OSError) as e:
            print(f"[imagine] {kind} {tag}失败：{e}", file=sys.stderr)
            if not written:
                return 1
            break  # 已有产出：保住它们，不把整批判失败
        for im in resp.failed_images:
            print(f"  ! {tag}第 {im.index} 张失败：{im.error}", file=sys.stderr)
        if not resp.ok_images:
            # 逐张的失败原因上面已经打过；这里只决定“要不要再试下一次请求”。
            # 报错文案统一交给循环后的 ``if not written`` 出口，避免两处措辞不一致。
            if written:
                continue
            break
        stem_i = stem if calls == 1 else f"{stem}_{i}"
        try:
            got = _ark.save_images(resp, dest_dir, stem=stem_i, timeout=args.timeout)
        except (_ark.ArkError, OSError) as e:
            # OSError 覆盖 requests.RequestException（下载 url 失败）与磁盘写入失败
            print(f"[imagine] 落盘失败：{e}", file=sys.stderr)
            if not written:
                return 1
            break
        snap = _ark.save_snapshot(resp, settings.runs_dir, stem=stem_i)
        snaps.append(snap)
        billed += resp.generated_images
        tokens += int(resp.usage.get("total_tokens") or 0)
        for w in got:
            # 产物内嵌 prompt/model 元数据（PNG tEXt），脱离账本也可溯源
            _ledger.embed_metadata(
                w, {"prompt": args.prompt, "model": resp.model, "kind": kind}
            )
            _ledger.record(
                prompt=args.prompt,
                seed=args.seed,
                model=resp.model or payload.get("model", ""),
                backend="ark",
                recipe=getattr(args, "recipe", "") or "",
                ref=ref_str,
                out=w,
                notes=args.notes or "",
                extra={
                    "kind": kind,
                    "size": payload.get("size")
                    or f"{payload.get('width')}x{payload.get('height')}",
                    "generated_images": resp.generated_images,
                    "usage": resp.usage,
                    "snapshot": str(snap),
                },
            )
        written.extend(got)

    if not written:
        print(f"[imagine] {kind} 无任何成功产出。", file=sys.stderr)
        return 1
    _ledger.render_markdown()  # 批量记账结束，渲染一次（而非每张全量重渲染）
    print(f"已出图 {len(written)} 张 → {dest_dir}")
    for w in written:
        print(f"  {w}")
    print(f"  用量：{billed} 张计费，tokens={tokens or '?'}")
    print(f"  请求快照：{snaps[-1] if snaps else '-'}")
    print(f"\n✓ 视觉校验：Read '{written[0]}'")
    print("  满意就立刻 `imagine gallery --add <path> --slug S` 归档——")
    print("  方舟不保证同 prompt 复现同图，归档是唯一可靠的留存手段。")
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    return _cloud_generate(args, images=None, kind="gen")


def cmd_i2i(args: argparse.Namespace) -> int:
    return _cloud_generate(args, images=list(args.image), kind="i2i")


def cmd_edit(args: argparse.Namespace) -> int:
    """交互编辑：i2i 的语义化预设（默认切 5.0-pro，输入应是 `imagine mark` 产出的带标记图）。"""
    if not args.model:
        args.model = PRO_MODEL_HINT
    print("[imagine] edit：交互编辑需 5.0-pro/flash；prompt 应描述图中每个标记的意图")
    print("           （如「将框选区域 A 改为…，在箭头所指位置添加…」）。")
    return _cloud_generate(args, images=list(args.image), kind="edit")


def cmd_layers(args: argparse.Namespace) -> int:
    """图层拆分：i2i 的语义化预设（默认 5.0-pro + 组图关闭；产出为 1 底图 + ≤16 带 alpha 图层）。"""
    if not args.model:
        args.model = PRO_MODEL_HINT
    print("[imagine] layers：prompt 需**明确列出要拆哪些层**，例如")
    print(
        '           "将海报拆分为四个透明图层：主标题、说明文字、主体物、纯色背景。'
        '保持所有元素的原始尺寸、位置和光影不变。"'
    )
    return _cloud_generate(args, images=list(args.image), kind="layers")


# ---------------------------------------------------------------------------
# mark：在图上画编辑标记（框/箭头/点/文字）—— 交互编辑的前置工具，纯 PIL 零依赖
# ---------------------------------------------------------------------------
def _draw_arrow(
    draw, xy0: tuple[float, float], xy1: tuple[float, float], color, width: int
) -> None:
    """画带箭头的线段（PIL 无原生箭头，用 polygon 补箭头头部）。"""
    import math

    x0, y0 = xy0
    x1, y1 = xy1
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    head = max(12.0, width * 4.0)
    spread = math.radians(24)
    for s in (ang + math.pi - spread, ang + math.pi + spread):
        draw.line(
            [x1, y1, x1 + head * math.cos(s), y1 + head * math.sin(s)],
            fill=color,
            width=width,
        )


def cmd_mark(args: argparse.Namespace) -> int:
    """在源图上叠加带 A/B/C 标签的框、箭头、点、文字，产出交互编辑的输入图。

    标签按 ``--rect`` → ``--arrow`` → ``--point`` 的出现顺序自动编号 A, B, C…，
    与方舟交互编辑的 prompt 写法（"将框选区域 A 改为…"）直接对应。
    """
    from PIL import Image, ImageDraw

    try:
        src = _check_image(Path(args.src).expanduser())
    except (FileNotFoundError, ValueError) as e:
        print(f"[imagine] mark 失败：{e}", file=sys.stderr)
        return 1

    im = Image.open(src).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = tuple(args.color) if args.color else _MARK_DEFAULT_COLOR
    w = int(args.width)
    labels: list[tuple[str, str, tuple[float, float]]] = []  # (标签, 类型, 文字锚点)
    seq = 0

    def next_label() -> str:
        nonlocal seq
        lab = chr(ord("A") + seq) if seq < 26 else f"M{seq + 1}"
        seq += 1
        return lab

    for spec in args.rect or []:
        try:
            x, y, rw, rh = (float(v) for v in spec.split(","))
        except ValueError:
            print(f"[imagine] --rect 格式应为 x,y,w,h，实得 {spec!r}", file=sys.stderr)
            return 1
        draw.rectangle([x, y, x + rw, y + rh], outline=color + (255,), width=w)
        lab = next_label()
        draw.rectangle([x, max(0.0, y - 26), x + 26, y], fill=color + (235,))
        draw.text((x + 6, max(2.0, y - 24)), lab, fill=(255, 255, 255, 255))
        labels.append((lab, "框选区域", (x, y)))

    for spec in args.arrow or []:
        try:
            x0, y0, x1, y1 = (float(v) for v in spec.split(","))
        except ValueError:
            print(
                f"[imagine] --arrow 格式应为 x1,y1,x2,y2，实得 {spec!r}",
                file=sys.stderr,
            )
            return 1
        _draw_arrow(draw, (x0, y0), (x1, y1), color + (255,), w)
        lab = next_label()
        draw.text((x1 + 8, y1 + 8), lab, fill=color + (255,))
        labels.append((lab, "箭头所指位置", (x1, y1)))

    for spec in args.point or []:
        try:
            x, y = (float(v) for v in spec.split(","))
        except ValueError:
            print(f"[imagine] --point 格式应为 x,y，实得 {spec!r}", file=sys.stderr)
            return 1
        r = max(6.0, w * 2.0)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color + (255,), width=w)
        lab = next_label()
        draw.text((x + r + 4, y - r), lab, fill=color + (255,))
        labels.append((lab, "标记点", (x, y)))

    for spec in args.text or []:
        parts = spec.split(",", 2)
        if len(parts) != 3:
            print(f"[imagine] --text 格式应为 x,y,内容，实得 {spec!r}", file=sys.stderr)
            return 1
        x, y, txt = float(parts[0]), float(parts[1]), parts[2]
        draw.text((x, y), txt, fill=color + (255,))

    out = _resolve_out_path(args, f"{src.stem}_marked.png")
    assert_within_data(out, what="标记图")
    Image.alpha_composite(im, overlay).convert("RGB").save(out)

    print(f"已生成标记图：{out}（{len(labels)} 个标记）")
    for lab, kindname, (x, y) in labels:
        print(f"  {lab}: {kindname} @ ({x:.0f}, {y:.0f})")
    print(f"\n✓ 视觉校验：Read '{out}'  ← 确认标记落在你想编辑的位置上")
    if labels:
        hints = "，".join(f"将{kn} {lab} …" for lab, kn, _ in labels[:3])
        print("\n下一步（交互编辑，需 5.0-pro）：")
        print(f"  imagine edit --image '{out}' \\")
        print(
            f'    --prompt "根据图中标记进行修改：{hints}。保持整体透视、光影与风格不变。"'
        )
    return 0


# ---------------------------------------------------------------------------
# img：本地位图算子（scikit-image + OpenCV，optional extra）
# ---------------------------------------------------------------------------
def cmd_img(args: argparse.Namespace) -> int:
    """``imagine img <op>`` 分发到 :mod:`imaging`。依赖缺失时给安装命令而非崩溃。"""
    op = args.img_cmd
    try:
        if op == "split":
            # split/measure 不注册 --out/--slug（_add_img_common(out=False)）→ 用 getattr 兜
            res = _img.split_alpha(
                args.src,
                args.dest or settings.cache_dir,
                stem=getattr(args, "slug", None),
            )
            print(f"已分离：RGB → {res['rgb']}\n         alpha → {res['alpha']}")
            print(f"\n✓ 视觉校验：Read '{res['alpha']}'  ← alpha 可直接当 mask 用")
            return 0

        if op == "composite":
            out = _resolve_out_path(args, "composite.png")
            positions = None
            if args.pos:
                positions = [tuple(int(v) for v in p.split(",")) for p in args.pos]
            p = _img.composite(
                args.base,
                args.layer,
                out,
                opacities=args.opacity,
                positions=positions,
            )
            print(f"已合成 {len(args.layer)} 层 → {p}")
            print(f"\n✓ 视觉校验：Read '{p}'")
            return 0

        if op == "fuse":
            out = _resolve_out_path(args, "fused.png")
            center = (
                tuple(int(v) for v in args.center.split(",")) if args.center else None
            )
            p = _img.fuse(
                args.src, args.base, out, center=center, mask=args.mask, mode=args.mode
            )
            print(f"泊松融合完成 → {p}")
            print(f"\n✓ 视觉校验：Read '{p}'  ← 重点看边界过渡是否自然")
            return 0

        if op == "inpaint":
            out = _resolve_out_path(args, "inpainted.png")
            p = _img.inpaint(
                args.src, out, mask=args.mask, radius=args.radius, method=args.method
            )
            print(f"inpaint 完成 → {p}")
            print(f"\n✓ 视觉校验：Read '{p}'")
            return 0

        if op == "mask":
            out = _resolve_out_path(args, "mask.png")
            p = _img.make_mask(
                args.src,
                out,
                method=args.method,
                thresh=args.thresh,
                invert=args.invert,
                blur=args.blur,
                rect=args.grabcut_rect,
                iterations=args.iterations,
            )
            print(f"mask 已生成 → {p}")
            print(
                f"\n✓ 视觉校验：Read '{p}'  ← 白色为前景；不对就换 --method 或调 --thresh"
            )
            return 0

        if op == "morph":
            out = _resolve_out_path(args, "morph.png")
            p = _img.morphology(
                args.src,
                out,
                op=args.op_name,
                radius=args.radius,
                iterations=args.iterations,
            )
            print(f"形态学 {args.op_name} 完成 → {p}")
            print(f"\n✓ 视觉校验：Read '{p}'")
            return 0

        if op == "warp":
            out = _resolve_out_path(args, "warped.png")
            size = tuple(args.size) if args.size else None
            p = _img.perspective(
                args.src,
                out,
                src_pts=args.src_pts,
                dst_pts=args.dst_pts,
                size=size,
                rotate=args.rotate,
            )
            print(f"几何校正完成 → {p}")
            print(f"\n✓ 视觉校验：Read '{p}'  ← 确认主体已拉正")
            return 0

        if op == "measure":
            regions = _img.measure_regions(
                args.src, min_area=args.min_area, thresh=args.thresh, mask=args.mask
            )
            if not regions:
                print("未检出连通域（试降低 --min-area 或换 --thresh / --mask）")
                return 0
            print(f"=== 连通域测量（{len(regions)} 个，按面积降序）===")
            print(
                f"  {'label':>5} {'area_px':>9} {'centroid':>18} {'aspect':>7} {'solidity':>9}"
            )
            for r in regions:
                print(
                    f"  {r['label']:>5} {r['area_px']:>9} "
                    f"{str(r['centroid']):>18} {str(r['aspect'] or '-'):>7} {r['solidity']:>9}"
                )
            print("\n像素→物理量换算：先用已知尺寸的标尺求出 px/mm，再乘 area_px。")
            return 0

        if op == "align":
            out = _resolve_out_path(args, "aligned.png") if not args.no_apply else None
            res = _img.align(args.src, args.ref, out, upsample=args.upsample)
            print(
                f"配准结果：平移 (dx, dy) = {res['shift_xy']} px，RMS 误差 {res['rms_error']}"
            )
            if res.get("out"):
                print(f"已写出对齐图 → {res['out']}")
                print(f"\n✓ 视觉校验：Read '{res['out']}'")
            return 0

        print(f"[imagine] img：未知子命令 {op!r}", file=sys.stderr)
        return 1
    except _img.ImagingError as e:
        print(f"[imagine] img {op} 失败：{e}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as e:
        print(f"[imagine] img {op} 失败：{e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# palette / bridge（审美参考 → 数据复现桥）
# ---------------------------------------------------------------------------
def cmd_palette(args: argparse.Namespace) -> int:
    try:
        info = _bridge.analyze_reference(Path(args.src).expanduser(), n_colors=args.n)
    except FileNotFoundError as e:
        print(f"[imagine] palette 失败：{e}", file=sys.stderr)
        return 1
    pal = info["palette"]
    print(f"范本：{info['path']}")
    print(
        f"  尺寸：{info['width']}×{info['height']}（{info['orientation']}，w/h={info['aspect']}）"
    )
    print(f"  平均亮度：{info['mean_brightness']}/255")
    print(f"  主色（{len(pal)}，按占比降序）：")
    for i, hx in enumerate(pal):
        role = "主色/背景" if i == 0 else f"强调色 {i}"
        print(f"    {i + 1}. {hx}  {role}")
    if pal:
        print(f'\n用法：管线里 palette.color("{pal[0]}") 直接注入（color() 接受 hex）')
    print("接入绘图复现：imagine bridge --ref <此范本> --research R --slug S")
    return 0


def cmd_bridge(args: argparse.Namespace) -> int:
    try:
        res = _bridge.bridge(
            args.ref,
            args.research,
            args.slug,
            style=args.style,
            width=args.width,
            template=args.template,
            n_colors=args.n,
            palette_name=args.palette_name,
            copy_ref=not args.no_copy_ref,
            overwrite=args.overwrite,
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
def _add_cloud_opts(s: argparse.ArgumentParser) -> None:
    """gen / i2i / edit / layers 共享的云端出图选项。

    ``--image``（i2i/edit/layers）与 ``--group``（gen）在各自 parser 里另加。
    """
    s.add_argument(
        "--prompt",
        required=True,
        help="提示词。图层拆分/交互编辑的**意图也写在这里**（无专用 API 参数）",
    )
    s.add_argument(
        "--model",
        default=None,
        help=f"方舟 Model ID（默认 .env AI_DRAWING_MODEL 或 {settings.default_model}）；"
        "ID 真值源是控制台「模型列表」",
    )
    s.add_argument(
        "--size", default=None, help="1K/2K/3K/4K 或 宽x高（默认 .env 或 2K）"
    )
    s.add_argument(
        "--width", type=int, default=None, help="自定义宽（与 --size 二选一）"
    )
    s.add_argument(
        "--height", type=int, default=None, help="自定义高（与 --size 二选一）"
    )
    s.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"随机种子，范围 {list(_ark.SEED_RANGE)}。**仅 3.0-t2i 生效**，且方舟声明"
        "'相同 seed 也不保证完全一致' → 别拿它当复现手段，满意就 gallery --add",
    )
    s.add_argument(
        "--n",
        type=int,
        default=1,
        help="生成张数（受成本护栏 max_images 夹取）。非组图时方舟单请求只出一张，"
        "故 --n>1 会发起多次**独立**请求（各自不同构图，适合挑图）；"
        "要一套连贯组图请用 --group",
    )
    s.add_argument(
        "--watermark",
        action="store_true",
        help="加'AI 生成'水印（方舟默认加；本 CLI 默认显式关闭，给此开关才加）",
    )
    s.add_argument(
        "--output-format",
        dest="output_format",
        default=None,
        choices=("png", "jpeg"),
        help="输出格式；**png 仅 5.0 主档支持**，其余模型恒为 jpeg",
    )
    s.add_argument(
        "--response-format",
        dest="response_format",
        default="b64_json",
        choices=("b64_json", "url"),
        help="默认 b64_json（直接回传字节，规避 url 24h 失效风险）",
    )
    s.add_argument(
        "--web-search",
        dest="web_search",
        action="store_true",
        help="启用联网检索 tools（仅 5.0 主档支持）",
    )
    s.add_argument(
        "--extra",
        default=None,
        help='追加/覆盖请求体字段的 JSON 对象（方舟上新参数时用，如 {"thinking":true}）',
    )
    # 落点与记账
    s.add_argument(
        "--out", default=None, help="产物目录（默认研究线 artwork/ 或技能 assets/）"
    )
    s.add_argument("--research", default=None, help="研究线名（落 article/artwork/）")
    s.add_argument("--slug", default=None, help="产物命名前缀")
    s.add_argument("--notes", default=None, help="自由备注（记账）")
    s.add_argument(
        "--recipe", default=None, help="所属流水线配方名（记账，对应 recipes/*.md）"
    )
    # 执行控制
    s.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="仅构造并打印请求体，不发请求（不计费、无需 API Key）",
    )
    s.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"请求超时秒数（默认 {settings.timeout}）",
    )


def _add_img_common(
    s: argparse.ArgumentParser, *, src: bool = True, out: bool = True
) -> None:
    """img 子命令共享的 src / 落点参数。"""
    if src:
        s.add_argument("src", help="源图路径")
    if out:
        s.add_argument(
            "--out", default=None, help="输出路径（后缀决定格式；默认落 cache/）"
        )
        s.add_argument(
            "--research", default=None, help="研究线名（落 article/artwork/）"
        )
        s.add_argument("--slug", default=None, help="产物命名前缀")


def _build_img_subparsers(sub: argparse._SubParsersAction) -> None:
    """``imagine img <op>``：本地位图算子（需 optional extra ``imaging``）。"""
    s = sub.add_parser(
        "img",
        help="本地位图算子（scikit-image + OpenCV；uv pip install -e '.[imaging]'）",
    )
    isub = s.add_subparsers(dest="img_cmd", required=True)

    o = isub.add_parser(
        "split", help="把带 alpha 的图拆成 _rgb.png + _alpha.png（图层拆分下游）"
    )
    _add_img_common(o, out=False)
    o.add_argument("--dest", default=None, help="输出目录（默认 cache/）")
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "composite", help="多图层 alpha 合成到底图（消费方舟图层拆分输出）"
    )
    o.add_argument("--base", required=True, help="底图路径")
    o.add_argument(
        "--layer",
        action="append",
        required=True,
        help="图层路径（可多次；按给出顺序自下而上叠）",
    )
    o.add_argument(
        "--opacity",
        type=float,
        action="append",
        default=None,
        help="每层不透明度 0-1（可多次，与 --layer 一一对应）",
    )
    o.add_argument(
        "--pos",
        action="append",
        default=None,
        help='每层左上角偏移 "x,y"（可多次，与 --layer 对应）',
    )
    _add_img_common(o, src=False)
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "fuse",
        help="泊松融合：把素材无缝嵌入底图（cv2.seamlessClone）",
        description="泊松融合迁移的是**梯度**而非绝对颜色：纯色素材融进均匀底图会完全“消失”，"
        "要贴一块图请用 `img composite`。",
    )
    _add_img_common(o)
    o.add_argument("--base", required=True, help="底图路径（素材融进它）")
    o.add_argument(
        "--mask", default=None, help="融合区域 mask（默认取素材 alpha 通道）"
    )
    o.add_argument("--center", default=None, help='落点 "x,y"（默认底图中心）')
    o.add_argument(
        "--mode",
        default="mixed",
        choices=("mixed", "normal"),
        help="mixed（默认）取 max(|∇src|,|∇dst|)，底图纹理透出；normal 只用 |∇src|，"
        "底图纹理被抹成调和过渡。二者仅在底图有纹理时可区分",
    )
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "inpaint", help="算法级修补（Telea/NS，**不需要 GPU**）：抹瑕疵/水印残留"
    )
    _add_img_common(o)
    o.add_argument("--mask", required=True, help="待修补区域 mask（非零处被修补）")
    o.add_argument("--radius", type=float, default=3.0, help="邻域半径（默认 3）")
    o.add_argument(
        "--method", default="telea", choices=("telea", "ns"), help="算法（默认 telea）"
    )
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "mask", help="生成二值 mask（otsu/manual/canny/grabcut），供 fuse/inpaint"
    )
    _add_img_common(o)
    o.add_argument(
        "--method", default="otsu", choices=("otsu", "manual", "canny", "grabcut")
    )
    o.add_argument(
        "--thresh", type=int, default=None, help="manual 的阈值 / canny 的低阈值"
    )
    o.add_argument("--invert", action="store_true", help="反转前景背景")
    o.add_argument("--blur", type=int, default=0, help="先高斯模糊抑噪的半径（0 关闭）")
    o.add_argument(
        "--grabcut-rect",
        dest="grabcut_rect",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="grabcut 的主体框",
    )
    o.add_argument("--iterations", type=int, default=5, help="grabcut 迭代次数")
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "morph", help="形态学开/闭/腐蚀/膨胀（skimage.morphology）：去碎斑/填孔"
    )
    _add_img_common(o)
    o.add_argument(
        "--op-name",
        dest="op_name",
        default="open",
        choices=("open", "close", "erode", "dilate"),
        help="运算（默认 open）",
    )
    o.add_argument("--radius", type=int, default=2, help="结构元半径（默认 2）")
    o.add_argument("--iterations", type=int, default=1, help="重复次数")
    o.set_defaults(func=cmd_img)

    o = isub.add_parser("warp", help="几何校正：透视变换拉正斜拍照片，或纯旋转")
    _add_img_common(o)
    o.add_argument(
        "--src-pts",
        dest="src_pts",
        default=None,
        help='原图 4 角点 "x1,y1;x2,y2;x3,y3;x4,y4"（顺时针）',
    )
    o.add_argument(
        "--dst-pts", dest="dst_pts", default=None, help="目标 4 角点（默认拉成矩形）"
    )
    o.add_argument(
        "--size", type=int, nargs=2, metavar=("W", "H"), default=None, help="输出尺寸"
    )
    o.add_argument(
        "--rotate", type=float, default=None, help="只做旋转的角度（度，逆时针）"
    )
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "measure", help="连通域测量：面积/质心/长短轴/离心率（科研量化）"
    )
    _add_img_common(o, out=False)
    o.add_argument(
        "--min-area", dest="min_area", type=int, default=50, help="最小面积过滤（px）"
    )
    o.add_argument(
        "--thresh", type=int, default=None, help="固定阈值（默认 Otsu 自动）"
    )
    o.add_argument("--mask", default=None, help="直接给二值 mask（跳过阈值化）")
    o.set_defaults(func=cmd_img)

    o = isub.add_parser(
        "align", help="相位相关配准：求 src 相对 ref 的亚像素平移（可写出对齐图）"
    )
    _add_img_common(o)
    o.add_argument("--ref", required=True, help="参考图（尺寸须与 src 一致）")
    o.add_argument(
        "--upsample", type=int, default=10, help="亚像素上采样倍数（默认 10）"
    )
    o.add_argument(
        "--no-apply",
        dest="no_apply",
        action="store_true",
        help="只报平移量，不写对齐图",
    )
    o.set_defaults(func=cmd_img)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pysci-imagine",
        description="pySciWS AI 绘图技能：方舟 Seedream 云端出图 + 本地位图加工 + 生成账本 + 审美→数据桥接。",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- 自检 / 清单 ---
    s = sub.add_parser(
        "doctor", help="自检：配置/依赖库/方舟端点与密钥/成本护栏/Tier 0"
    )
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser(
        "models",
        help="打印方舟 Seedream 模型能力矩阵与通用约束（--live 读账号真实可见 ID）",
    )
    s.add_argument(
        "--live",
        action="store_true",
        help="额外 GET /models 列出账号实际可见的图像模型 ID（只读免费；需 ARK_API_KEY）",
    )
    s.add_argument(
        "--all",
        action="store_true",
        help="配合 --live：不过滤，连 LLM/视频/嵌入模型一起列",
    )
    s.set_defaults(func=cmd_models)

    s = sub.add_parser(
        "list", help="列出技能数据区（assets/gallery/recipes）或某研究线 artwork 的产物"
    )
    s.add_argument(
        "--research", default=None, help="研究线名（给则列其 article/artwork/）"
    )
    s.set_defaults(func=cmd_list)

    s = sub.add_parser(
        "ledger", help="查询/过滤生成账本（--stats 汇总 / --render 重渲染 MD）"
    )
    s.add_argument("--backend", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--contains", default=None, help="在 prompt/notes/out 中子串匹配")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--stats", action="store_true", help="打印汇总统计")
    s.add_argument("--render", action="store_true", help="由 manifest 重渲染 LEDGER.md")
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("gallery", help="审美范本画廊：列出 / --add 精选 / --sheet 总览")
    s.add_argument("--add", default=None, help="把一张图精选进 gallery")
    s.add_argument("--slug", default=None, help="精选命名（配合 --add）")
    s.add_argument("--sheet", action="store_true", help="生成画廊 contact sheet")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--thumb", type=int, default=256)
    s.set_defaults(func=cmd_gallery)

    # --- 云端出图 ---
    s = sub.add_parser("gen", help="文生图（可 --group 组图）：方舟直连 → 落盘 → 记账")
    _add_cloud_opts(s)
    s.add_argument(
        "--group",
        action="store_true",
        help="组图生成（sequential_image_generation=auto；仅 4.0/4.5/5.0 主档）",
    )
    s.add_argument(
        "--max-images",
        dest="max_images",
        type=int,
        default=None,
        help="单组张数（配合 --group；参考图数+产出数 ≤15）",
    )
    s.set_defaults(func=cmd_gen)

    s = sub.add_parser("i2i", help="图生图/多参考图：本地图自动转 base64（无需先上传）")
    s.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="PATH_OR_URL",
        help="参考图路径或 URL（可多次；5.0-pro ≤10 张，其余 ≤14 张）",
    )
    _add_cloud_opts(s)
    s.set_defaults(func=cmd_i2i)

    s = sub.add_parser(
        "edit", help="交互编辑（5.0-pro）：对带标记图按 prompt 做局部编辑"
    )
    s.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="PATH_OR_URL",
        help="带标记的输入图（先用 `imagine mark` 生成）",
    )
    _add_cloud_opts(s)
    s.set_defaults(func=cmd_edit)

    s = sub.add_parser(
        "layers", help="图层拆分（5.0-pro）：1 底图 + ≤16 个带 alpha 图层"
    )
    s.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="PATH_OR_URL",
        help="待拆分的图",
    )
    _add_cloud_opts(s)
    s.set_defaults(func=cmd_layers)

    # --- 本地位图加工 ---
    s = sub.add_parser(
        "mark", help="在图上画带 A/B/C 标签的框/箭头/点（交互编辑前置，纯 PIL）"
    )
    s.add_argument("--src", required=True, help="源图路径")
    s.add_argument(
        "--rect",
        action="append",
        default=None,
        metavar="X,Y,W,H",
        help="框选区域（可多次；自动编号 A/B/C…）",
    )
    s.add_argument(
        "--arrow",
        action="append",
        default=None,
        metavar="X1,Y1,X2,Y2",
        help="箭头（可多次）",
    )
    s.add_argument(
        "--point", action="append", default=None, metavar="X,Y", help="标记点（可多次）"
    )
    s.add_argument(
        "--text",
        action="append",
        default=None,
        metavar="X,Y,内容",
        help="自由文字标注（可多次）",
    )
    s.add_argument(
        "--color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="标记颜色（默认红 230 40 40）",
    )
    s.add_argument("--width", type=int, default=4, help="线宽（默认 4）")
    s.add_argument("--out", default=None, help="输出路径（默认 <src>_marked.png）")
    s.add_argument("--research", default=None, help="研究线名")
    s.set_defaults(func=cmd_mark)

    s = sub.add_parser(
        "adjust", help="Pillow 单图后处理（crop/resize/rotate/convert/pad/format）"
    )
    s.add_argument("src", help="源图路径")
    s.add_argument(
        "--out", required=True, help="目标路径（后缀决定格式，除非给 --format）"
    )
    s.add_argument(
        "--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None
    )
    s.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="目标尺寸；某边为 0 时按原图长宽比自动",
    )
    s.add_argument(
        "--rotate", type=float, default=None, help="逆时针旋转角度（expand）"
    )
    s.add_argument("--mode", default=None, help="色彩模式转换（RGB/RGBA/L…）")
    s.add_argument(
        "--pad",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="加边到 WxH 画布并居中",
    )
    s.add_argument(
        "--pad-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        dest="pad_color",
    )
    s.add_argument("--format", default=None, help="显式输出格式（覆盖后缀推断）")
    s.add_argument("--quality", type=int, default=95, help="JPEG/WEBP 质量（1-100）")
    s.set_defaults(func=cmd_adjust)

    s = sub.add_parser(
        "sheet", help="多图拼合 contact sheet（缩略图网格 + 文件名标签）"
    )
    s.add_argument("images", nargs="+", help="源图路径列表")
    s.add_argument(
        "--out", default=None, help="输出路径（默认 cache/contact_sheet.png）"
    )
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--thumb", type=int, default=256)
    s.add_argument("--pad", type=int, default=8)
    s.add_argument("--no-label", action="store_true", dest="no_label")
    s.add_argument("--bg", type=int, nargs=3, metavar=("R", "G", "B"), default=None)
    s.set_defaults(func=cmd_sheet)

    _build_img_subparsers(sub)

    # --- 入库与桥接 ---
    s = sub.add_parser(
        "ingest", help="把一张已生成的图入库（assets 或研究线 artwork）并记账"
    )
    s.add_argument("--src", required=True, help="源图路径（如 ImageGen 产物）")
    s.add_argument("--slug", default=None, help="入库命名（默认取源图文件名）")
    s.add_argument(
        "--research",
        default=None,
        help="研究线名（给则落 article/artwork/，否则落技能 assets/）",
    )
    s.add_argument("--gallery", action="store_true", help="同时精选进 gallery")
    s.add_argument("--prompt", default=None, help="生成用 prompt（记账）")
    s.add_argument("--seed", type=int, default=None, help="随机种子（记账）")
    s.add_argument("--model", default=None, help="图像模型 ID（记账）")
    s.add_argument(
        "--backend",
        default="imagegen",
        help="出图后端（ark/imagegen/manual；默认 imagegen）",
    )
    s.add_argument("--recipe", default=None, help="所属流水线配方名（记账）")
    s.add_argument("--ref", default=None, help="图生图源图（记账）")
    s.add_argument("--notes", default=None, help="自由备注（记账）")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser(
        "palette", help="从一张图抽取主色板（hex，按占比降序）——桥接的离线取色"
    )
    s.add_argument("--src", required=True, help="源图路径（审美范本）")
    s.add_argument("--n", type=int, default=6, help="主色数量（1-16，默认 6）")
    s.set_defaults(func=cmd_palette)

    s = sub.add_parser(
        "bridge", help="审美范本 → 数据复现桥：脚手架绘图管线 + 写 design_spec.md"
    )
    s.add_argument("--ref", required=True, help="审美范本图路径")
    s.add_argument("--research", required=True, help="研究线名（不含数字前缀）")
    s.add_argument("--slug", required=True, help="图目录名（如 fig1_cover）")
    s.add_argument("--style", default="aps", help="期刊风格预设（aps/nature）")
    s.add_argument("--width", default="double", help="设计宽度 single/double 或毫米数")
    s.add_argument(
        "--template",
        default=None,
        help="脚手架模板（single/multi_panel/concept_plus_data；默认 multi_panel）",
    )
    s.add_argument("--n", type=int, default=6, help="抽取主色数量（默认 6）")
    s.add_argument(
        "--palette-name",
        dest="palette_name",
        default=None,
        help="给则把抽取配色注册为该名调色板（持久化）并绑定图目录 STYLE.yaml",
    )
    s.add_argument(
        "--no-copy-ref",
        dest="no_copy_ref",
        action="store_true",
        help="不把范本图复制进图目录",
    )
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
