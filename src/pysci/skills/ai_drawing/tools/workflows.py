"""程序化构造 ComfyUI **API 格式**工作流图（dict builders）。

ComfyUI 有两种工作流序列化：**UI 格式**（GUI 存的 ``nodes``/``links``，含画布坐标）与
**API 格式**（``POST /prompt`` 吃的 ``{node_id: {class_type, inputs}}``，连线表示为
``[源节点id, 源输出口序号]``）。本模块只产 **API 格式**——它是提交执行的真值形态。

已联网核实的契约（ComfyUI-Jimeng-API v2.5.0，菜单 JimengAI）见 :mod:`providers` 与项目记忆：
``JimengAPIClient`` → ``JimengSeedream4``（``model_version``/``prompt``/``size``/``width``/
``height``/``seed``/``enable_group_generation``/``max_images``/``generation_count``/``thinking``/
``watermark`` + 可选 autogrow ``images``）→ ``SaveImage``；``JimengQuotaSettings`` 作可选护栏。

.. warning::
   ``class_type`` 与输入名**禁止猜**：本模块登记的是核实值，但服务器真值以
   ``GET /object_info`` 为准（``imagine comfy nodes JimengSeedream4``）。尤其 i2i 的 autogrow
   ``images`` 输入在不同 ComfyUI 版本下的 API 序列化可能为嵌套 dict（本模块采用）或点号扁平键；
   若 ``/prompt`` 回报 node_errors，请用 ``imagine comfy nodes`` 核实后改 ``run --workflow``
   （GUI「Save (API Format)」导出的图保证正确）。

用法::

    from pysci.skills.ai_drawing.tools.workflows import txt2img_seedream

    graph = txt2img_seedream("a journal cover ...", seed=42, size="2K (adaptive)")
    # → 交给 comfy_client.queue_prompt(graph)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import settings
from .providers import ImageModel, Provider, get_provider

# ---------------------------------------------------------------------------
# class_type 常量（集中管理；真值以 /object_info 为准）
# ---------------------------------------------------------------------------
#: comfy-core 节点。
NODE_LOAD_IMAGE = "LoadImage"
NODE_SAVE_IMAGE = "SaveImage"

#: ComfyUI-Jimeng-API 节点（菜单 JimengAI）。
NODE_JIMENG_CLIENT = "JimengAPIClient"
NODE_JIMENG_QUOTA = "JimengQuotaSettings"
NODE_JIMENG_SEEDREAM3 = "JimengSeedream3"
NODE_JIMENG_SEEDREAM4 = "JimengSeedream4"
NODE_JIMENG_SEEDREAM5 = "JimengSeedream5"

#: SaveImage 默认文件名前缀（落 ComfyUI output/ 下的子目录，便于归类）。
DEFAULT_FILENAME_PREFIX = "Jimeng/Image/Seedream4"


def _link(node_id: str, slot: int = 0) -> list[Any]:
    """API 格式连线：``[源节点id, 源输出口序号]``。"""
    return [node_id, slot]


# ---------------------------------------------------------------------------
# 单节点构造
# ---------------------------------------------------------------------------
def api_client_node(key_name: str = "Custom") -> dict[str, Any]:
    """``JimengAPIClient`` 节点（所有工作流的起点，读节点侧 api_keys.json）。

    ``key_name`` 须匹配 ``api_keys.json`` 里某条目的 ``customName``（或 ``"Custom"`` 配合在
    GUI 内填入的 ``new_api_key``）。pysci **不**经此传原始密钥——密钥只存节点侧文件。
    """
    return {
        "class_type": NODE_JIMENG_CLIENT,
        "inputs": {"key_name": key_name, "new_api_key": "", "new_key_name": ""},
    }


def quota_node(
    client_id: str,
    *,
    image_model: str = "None",
    image_limit: int = 0,
    video_model: str = "None",
    video_limit: int = 0,
) -> dict[str, Any]:
    """``JimengQuotaSettings`` 成本护栏节点（``image_limit=0`` 表示不设限）。

    .. note::
       该节点把限额写进**进程级单例**（按 api_key），生成节点执行前 ``check_quota``。它的
       ``status`` 输出若不连到任何 OUTPUT_NODE，可能被 ComfyUI 剪枝而不在本图内执行——故
       pysci 侧的**硬护栏**是 :func:`txt2img_seedream` 的 ``max_images`` 上限 + 默认单张；
       本节点是可选的服务器侧二次保险，建议单独跑一次以注册限额。
    """
    return {
        "class_type": NODE_JIMENG_QUOTA,
        "inputs": {
            "client": _link(client_id, 0),
            "image_model": image_model,
            "image_limit": int(image_limit),
            "video_model": video_model,
            "video_limit": int(video_limit),
        },
    }


def load_image_node(image_name: str) -> dict[str, Any]:
    """``LoadImage`` 节点（i2i 源图；``image_name`` 为经 ``POST /upload/image`` 上传后的名字）。"""
    return {"class_type": NODE_LOAD_IMAGE, "inputs": {"image": image_name}}


def save_image_node(images_ref: list[Any], filename_prefix: str = DEFAULT_FILENAME_PREFIX) -> dict[str, Any]:
    """``SaveImage`` 节点（OUTPUT_NODE，触发整图执行并把结果落 ComfyUI output/）。"""
    return {
        "class_type": NODE_SAVE_IMAGE,
        "inputs": {"images": images_ref, "filename_prefix": filename_prefix},
    }


def seedream_gen_node(
    model: ImageModel,
    client_id: str,
    *,
    prompt: str,
    size: str,
    width: int,
    height: int,
    seed: int,
    generation_count: int = 1,
    enable_group_generation: bool = False,
    max_images: int = 1,
    thinking: bool = True,
    watermark: bool = False,
    images_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``JimengSeedream4`` 生成节点（t2i / i2i 由 ``images_ref`` 有无决定）。

    ``images_ref`` 为 autogrow ``images`` 输入的 API 序列化（如 ``{"image_1": ["2", 0]}``）；
    None → 纯文生图。
    """
    inputs: dict[str, Any] = {
        "client": _link(client_id, 0),
        "model_version": model.ui_version,
        "prompt": prompt,
        "size": size,
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "enable_group_generation": bool(enable_group_generation),
        "max_images": int(max_images),
        "generation_count": int(generation_count),
        "thinking": bool(thinking),
        "watermark": bool(watermark),
    }
    if images_ref:
        inputs["images"] = images_ref
    return {"class_type": model.node_class, "inputs": inputs}


# ---------------------------------------------------------------------------
# 尺寸解析：把 size 选项还原成 width/height（Custom 之外的预设也给出对应像素，供占位）
# ---------------------------------------------------------------------------
def _size_to_wh(size: str, width: int | None, height: int | None) -> tuple[int, int]:
    """决定最终 width/height。

    节点仅在 ``size == "Custom"`` 时才真正使用 width/height；预设尺寸下 width/height 仅作占位。
    故：用户显式给了 width/height 就用之；否则给一个安全占位（2048×2048）。
    """
    w = int(width) if width else 2048
    h = int(height) if height else 2048
    return w, h


# ---------------------------------------------------------------------------
# 高层图构造：t2i / i2i
# ---------------------------------------------------------------------------
def txt2img_seedream(
    prompt: str,
    *,
    provider: str | Provider | None = None,
    model: str | ImageModel | None = None,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int = 0,
    n: int = 1,
    key_name: str | None = None,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    thinking: bool = True,
    watermark: bool = False,
    group: bool = False,
    max_images: int = 1,
    image_quota: int = 0,
) -> dict[str, Any]:
    """构造**文生图** API 格式工作流图。

    Args:
        prompt: 正向提示词。
        provider: provider 名或实例（默认 jimeng-ark / Seedream4）。
        model: 模型 UI 版本或 API ID（默认 provider 首选）。
        size: 尺寸选项（默认 provider.default_size；经 normalize_size 规范化）。
        width/height: 仅 ``size="Custom"`` 时生效。
        seed: 随机种子（0 为节点默认；-1 触发节点内随机）。
        n: 生成张数（generation_count）；受成本护栏夹到 [1, settings.max_images]。
        key_name: JimengAPIClient.key_name（默认取 provider/配置）。
        filename_prefix: SaveImage 前缀。
        thinking: 提示词优化（仅 Seedream 4.0 生效）。
        watermark: 是否加水印（默认否）。
        group/max_images: 组图生成开关与单组张数。
        image_quota: >0 时附加 JimengQuotaSettings 护栏（按张数限额）。

    Returns:
        API 格式图 ``{node_id: {class_type, inputs}}``。
    """
    prov = provider if isinstance(provider, Provider) else get_provider(provider)
    m = model if isinstance(model, ImageModel) else prov.resolve_model(model)
    size_norm = prov.normalize_size(size)
    w, h = _size_to_wh(size_norm, width, height)
    n_cap = max(1, min(int(n), max(1, settings.max_images)))
    key = key_name or _default_key_name(prov)

    graph: dict[str, Any] = {"1": api_client_node(key)}
    gen_id = "2"
    if image_quota and int(image_quota) > 0:
        graph["2"] = quota_node("1", image_model=m.ui_version, image_limit=int(image_quota))
        gen_id = "3"
    graph[gen_id] = seedream_gen_node(
        m,
        "1",
        prompt=prompt,
        size=size_norm,
        width=w,
        height=h,
        seed=seed,
        generation_count=n_cap,
        enable_group_generation=group,
        max_images=max(1, min(int(max_images), 15)),
        thinking=thinking,
        watermark=watermark,
    )
    save_id = str(int(gen_id) + 1)
    graph[save_id] = save_image_node(_link(gen_id, 0), filename_prefix)
    return graph


def img2img_seedream(
    prompt: str,
    image_names: list[str],
    *,
    provider: str | Provider | None = None,
    model: str | ImageModel | None = None,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int = 0,
    n: int = 1,
    key_name: str | None = None,
    filename_prefix: str = DEFAULT_FILENAME_PREFIX,
    thinking: bool = True,
    watermark: bool = False,
    image_quota: int = 0,
) -> dict[str, Any]:
    """构造**图生图** API 格式工作流图（1+ 张参考图，经 autogrow ``images`` 输入）。

    ``image_names`` 为经 ``POST /upload/image`` 上传到 ComfyUI 后的图片名（可多张，按序连
    ``image_1..image_N``）。其余参数同 :func:`txt2img_seedream`。

    .. warning::
       autogrow ``images`` 输入的 API 序列化（本函数用嵌套 dict ``{"image_1": [id,0]}``）
       在不同 ComfyUI 版本下可能不同；若 ``/prompt`` 报 node_errors，用 ``imagine comfy nodes
       JimengSeedream4`` 核实，或改用 GUI 导出的 ``run --workflow``。
    """
    if not image_names:
        raise ValueError("img2img_seedream 需要至少一张参考图（image_names 为空）")
    prov = provider if isinstance(provider, Provider) else get_provider(provider)
    if not prov.supports_i2i:
        raise ValueError(f"provider {prov.name!r} 不支持图生图")
    m = model if isinstance(model, ImageModel) else prov.resolve_model(model)
    size_norm = prov.normalize_size(size)
    w, h = _size_to_wh(size_norm, width, height)
    n_cap = max(1, min(int(n), max(1, settings.max_images)))
    key = key_name or _default_key_name(prov)

    graph: dict[str, Any] = {"1": api_client_node(key)}
    next_id = 2
    # 参考图 LoadImage 节点 + autogrow images 序列化
    images_ref: dict[str, Any] = {}
    for i, name in enumerate(image_names[:14], start=1):
        lid = str(next_id)
        graph[lid] = load_image_node(name)
        images_ref[f"image_{i}"] = _link(lid, 0)
        next_id += 1
    gen_id = str(next_id)
    next_id += 1
    if image_quota and int(image_quota) > 0:
        graph[gen_id] = quota_node("1", image_model=m.ui_version, image_limit=int(image_quota))
        gen_id = str(next_id)
        next_id += 1
    graph[gen_id] = seedream_gen_node(
        m,
        "1",
        prompt=prompt,
        size=size_norm,
        width=w,
        height=h,
        seed=seed,
        generation_count=n_cap,
        thinking=thinking,
        watermark=watermark,
        images_ref=images_ref,
    )
    save_id = str(next_id)
    graph[save_id] = save_image_node(_link(gen_id, 0), filename_prefix)
    return graph


def _default_key_name(prov: Provider) -> str:
    """JimengAPIClient.key_name 默认值：优先 .env 的 COMFY_JIMENG_KEY_NAME，否则 provider 默认。"""
    import os

    env = os.environ.get("COMFY_JIMENG_KEY_NAME")
    if env and env.strip():
        return env.strip()
    return prov.key_name_default


# ---------------------------------------------------------------------------
# 工作流配方：保存 / 加载 / 列出（data/skills/ai_drawing/workflows/*.json）
# ---------------------------------------------------------------------------
def workflow_path(name: str) -> Path:
    """解析工作流配方路径（``name`` 可省 ``.json``）。"""
    fname = name if name.lower().endswith(".json") else f"{name}.json"
    p = Path(fname)
    if p.is_absolute() or p.exists():
        return p
    return settings.workflows_dir / fname


def save_workflow(graph: dict[str, Any], name: str) -> Path:
    """把 API 格式图存为配方文件。"""
    dest = workflow_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def load_workflow(name: str) -> dict[str, Any]:
    """加载配方文件为 API 格式图。"""
    src = workflow_path(name)
    if not src.is_file():
        raise FileNotFoundError(f"工作流配方不存在：{src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"工作流配方格式错误（应为 dict）：{src}")
    return data


def list_workflows() -> list[Path]:
    """列出已存配方（workflows/*.json）。"""
    d = settings.workflows_dir
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


# ---------------------------------------------------------------------------
# run --workflow 的注参：把 args 覆盖进已存图
# ---------------------------------------------------------------------------
#: 友好键 → 生成节点输入名（run --args 用，自动定位图里的生成节点）。
_FRIENDLY_INPUTS = {
    "prompt": "prompt",
    "seed": "seed",
    "size": "size",
    "width": "width",
    "height": "height",
    "model": "model_version",
    "model_version": "model_version",
    "n": "generation_count",
    "generation_count": "generation_count",
    "watermark": "watermark",
    "thinking": "thinking",
}

#: 视作"生成节点"的 class_type 集合（用于友好键定位）。
_GEN_CLASSES = {NODE_JIMENG_SEEDREAM3, NODE_JIMENG_SEEDREAM4, NODE_JIMENG_SEEDREAM5}


def _find_gen_node(graph: dict[str, Any]) -> str | None:
    for nid, node in graph.items():
        if isinstance(node, dict) and node.get("class_type") in _GEN_CLASSES:
            return nid
    return None


def apply_args(graph: dict[str, Any], args: dict[str, Any] | None) -> dict[str, Any]:
    """把 ``args`` 覆盖进图（**就地修改并返回**）。

    支持两种键：
    - **点号定位** ``"<node_id>.<input>"``：精确覆盖某节点某输入（如 ``"3.seed"``）。
    - **友好键** ``prompt/seed/size/width/height/model/n/watermark/thinking``：自动定位图里的
      生成节点并覆盖对应输入。

    未知友好键原样忽略（打印告警交由调用方）。
    """
    if not args:
        return graph
    gen_id = _find_gen_node(graph)
    for key, val in args.items():
        if "." in key:
            nid, _, inp = key.partition(".")
            if nid in graph and isinstance(graph[nid], dict):
                graph[nid].setdefault("inputs", {})[inp] = val
            continue
        inp = _FRIENDLY_INPUTS.get(key)
        if inp and gen_id:
            graph[gen_id].setdefault("inputs", {})[inp] = val
    return graph


# ---------------------------------------------------------------------------
# 图摘要（供 CLI 打印 / 调试）
# ---------------------------------------------------------------------------
def graph_summary(graph: dict[str, Any]) -> str:
    """把 API 格式图渲染成紧凑的人读摘要（节点 → class_type + 关键输入）。"""
    lines = [f"API 工作流图（{len(graph)} 节点）:"]
    for nid in sorted(graph, key=lambda x: (len(x), x)):
        node = graph[nid]
        ct = node.get("class_type", "?")
        inputs = node.get("inputs", {})
        keys = []
        for k, v in inputs.items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                keys.append(f"{k}<-#{v[0]}:{v[1]}")
            elif isinstance(v, dict):
                keys.append(f"{k}{{...}}")
            else:
                sv = str(v)
                keys.append(f"{k}={sv[:24] + '…' if len(sv) > 24 else sv}")
        lines.append(f"  #{nid} {ct}: " + ", ".join(keys))
    return "\n".join(lines)
