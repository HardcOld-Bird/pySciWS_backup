"""可插拔图像 provider 抽象（即梦 / 火山方舟优先，预留 DashScope / 智谱）。

本技能**不直连**任何云 API——实际出图由无头 ComfyUI 经社区节点完成。故这里的"provider"
描述的是**一组可用的 ComfyUI 生成节点 + 其模型版本 / 尺寸选项**，供 :mod:`workflows`
据此程序化构造 API 格式工作流图，并供 CLI 做参数校验与默认值解析。

设计意图：新增一个 provider（如通义万相 / 智谱 CogView 的 ComfyUI 节点）只需在此登记一条
:class:`Provider`，``workflows`` 与 ``imagine gen/i2i`` 无需改动即可复用。

已联网核实的契约（ComfyUI-Jimeng-API，菜单 JimengAI，最低 ComfyUI 0.25.1）：
- ``JimengSeedream4`` 的 ``model_version`` COMBO 用 **UI 版本号**（``doubao-seedream-4.0`` /
  ``doubao-seedream-4.5``），节点内部再映射到 **Ark API 模型 ID**
  （``doubao-seedream-4-0-250828`` / ``doubao-seedream-4-5-251128``）。两者都要认得。
- ``size`` COMBO 取 ``RECOMMENDED_SIZES_V4``；节点 execute 用 ``size.split(" ")[0]`` 还原成
  Ark 的 ``size`` 字段（``"2K (adaptive)"`` → ``"2K"``，``"2048x2048 (1:1)"`` → ``"2048x2048"``）。

.. note::
   ``class_type`` 与输入名**禁止猜**：真值以服务器 ``GET /object_info`` 为准
   （``imagine comfy nodes JimengSeedream4``）。本模块登记的是截至核实时的稳定值。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageModel:
    """一个图像模型版本（ComfyUI 节点视角）。

    Attributes:
        ui_version: ComfyUI 节点 ``model_version`` COMBO 的取值（如 ``doubao-seedream-4.0``）。
        api_id: 对应的 Ark API 模型 ID（如 ``doubao-seedream-4-0-250828``），用于账本/直连参考。
        node_class: 承载该模型的 ComfyUI 节点 ``class_type``（如 ``JimengSeedream4``）。
        label: 人类可读标签。
    """

    ui_version: str
    api_id: str
    node_class: str
    label: str


@dataclass(frozen=True)
class Provider:
    """一个图像生成 provider（= 一族 ComfyUI 节点 + 其模型/尺寸选项）。

    Attributes:
        name: provider 标识（kebab，如 ``jimeng-ark``）。
        label: 人类可读名。
        node_class: 该 provider 主生成节点的 ``class_type``。
        client_node: 必需的客户端节点 ``class_type``（读 api_keys.json）。
        models: 可用模型版本（按推荐序）。
        size_options: ``size`` COMBO 的合法取值（来自节点 RECOMMENDED_SIZES）。
        default_size: 默认尺寸。
        supports_i2i: 是否支持图生图（连图像输入）。
        supports_group: 是否支持组图生成（enable_group_generation/max_images）。
        key_name_default: ``JimengAPIClient.key_name`` 的默认值（须匹配 api_keys.json 的 customName）。
        notes: 使用备注。
        wired: 是否已接线（False 表示预留、尚未实现对应节点构造）。
    """

    name: str
    label: str
    node_class: str
    client_node: str
    models: tuple[ImageModel, ...]
    size_options: tuple[str, ...]
    default_size: str
    supports_i2i: bool = True
    supports_group: bool = True
    key_name_default: str = "Custom"
    notes: str = ""
    wired: bool = True

    # ---------- 便捷查询 ----------
    @property
    def default_model(self) -> ImageModel:
        """推荐默认模型（首个）。"""
        return self.models[0]

    def model_versions(self) -> list[str]:
        return [m.ui_version for m in self.models]

    def resolve_model(self, model: str | None) -> ImageModel:
        """把用户给的 ``--model``（UI 版本号或 API ID）解析为 :class:`ImageModel`。

        容忍三种写法：完整 UI 版本（``doubao-seedream-4.0``）、API ID
        （``doubao-seedream-4-0-250828``）、或简写（``4.0`` / ``4`` / ``seedream-4``）。
        None / 空 → 默认模型。

        Raises:
            KeyError: 无法匹配到任何已知模型。
        """
        if not model:
            return self.default_model
        q = str(model).strip().lower()
        for m in self.models:
            if q in (m.ui_version.lower(), m.api_id.lower()):
                return m
        # 简写匹配：去掉常见前缀后比对版本尾号
        norm = q.replace("doubao-", "").replace("seedream-", "").replace("seedream", "")
        norm = norm.strip("-. ")
        for m in self.models:
            tail = m.ui_version.lower().replace("doubao-seedream-", "")
            if norm and (norm == tail or norm.replace(".", "-") == tail.replace(".", "-")):
                return m
            if norm and norm in m.api_id.lower():
                return m
        raise KeyError(
            f"provider {self.name!r} 无模型 {model!r}；可选 UI 版本 {self.model_versions()} "
            f"或对应 API ID {[m.api_id for m in self.models]}"
        )

    def normalize_size(self, size: str | None) -> str:
        """把用户给的 ``--size`` 规范成节点 COMBO 合法取值。

        接受 ``"2K"`` / ``"2k (adaptive)"`` / ``"2048x2048"`` / ``"Custom"`` 等；尽量匹配到
        ``size_options`` 里的完整串（节点 execute 会再 ``split(" ")[0]``）。无法匹配时原样返回
        （交由服务器校验，报错更直观）。
        """
        if not size:
            return self.default_size
        q = str(size).strip()
        low = q.lower()
        for opt in self.size_options:
            if opt.lower() == low:
                return opt
        # 前缀匹配：如 "2K" → "2K (adaptive)"，"2048x2048" → "2048x2048 (1:1)"
        for opt in self.size_options:
            if opt.lower().startswith(low) or opt.lower().split(" ")[0] == low:
                return opt
        return q


# ---------------------------------------------------------------------------
# 已核实的尺寸选项（来自节点 RECOMMENDED_SIZES_V4 / V3）
# ---------------------------------------------------------------------------
_SIZES_V4: tuple[str, ...] = (
    "2K (adaptive)",
    "4K (adaptive)",
    "2048x2048 (1:1)",
    "2304x1728 (4:3)",
    "1728x2304 (3:4)",
    "2848x1600 (16:9)",
    "1600x2848 (9:16)",
    "2496x1664 (3:2)",
    "1664x2496 (2:3)",
    "3136x1344 (21:9)",
    "4096x4096 (1:1)",
    "Custom",
)

_SIZES_V3: tuple[str, ...] = (
    "1024x1024 (1:1)",
    "864x1152 (3:4)",
    "1152x864 (4:3)",
    "1280x720 (16:9)",
    "720x1280 (9:16)",
    "Custom",
)


# ---------------------------------------------------------------------------
# provider 注册表
# ---------------------------------------------------------------------------
_JIMENG_ARK = Provider(
    name="jimeng-ark",
    label="即梦 / 火山方舟 Seedream（经 ComfyUI-Jimeng-API）",
    node_class="JimengSeedream4",
    client_node="JimengAPIClient",
    models=(
        ImageModel("doubao-seedream-4.0", "doubao-seedream-4-0-250828", "JimengSeedream4", "Seedream 4.0（默认，支持 thinking 提示词优化）"),
        ImageModel("doubao-seedream-4.5", "doubao-seedream-4-5-251128", "JimengSeedream4", "Seedream 4.5（更高细节）"),
    ),
    size_options=_SIZES_V4,
    default_size="2K (adaptive)",
    supports_i2i=True,
    supports_group=True,
    notes="主用 provider。i2i 经 autogrow 输入 images.image_N 连 LoadImage；组图用 enable_group_generation+max_images。",
)

_JIMENG_ARK_S3 = Provider(
    name="jimeng-ark-s3",
    label="即梦 Seedream 3.0（文生图，已弃用但可用）",
    node_class="JimengSeedream3",
    client_node="JimengAPIClient",
    models=(
        ImageModel("doubao-seedream-3.0-t2i", "doubao-seedream-3-0-t2i-250415", "JimengSeedream3", "Seedream 3.0 t2i"),
    ),
    size_options=_SIZES_V3,
    default_size="1024x1024 (1:1)",
    supports_i2i=False,
    supports_group=False,
    notes="schema 与 4 不同（有 guidance_scale，无 model_version/group/thinking）；建议用 run --workflow 走手存图。",
    wired=False,
)

_JIMENG_ARK_S5 = Provider(
    name="jimeng-ark-s5",
    label="即梦 Seedream 5（Pro/Lite）",
    node_class="JimengSeedream5",
    client_node="JimengAPIClient",
    models=(
        ImageModel("doubao-seedream-5.0-pro", "doubao-seedream-5-0-pro-260628", "JimengSeedream5", "Seedream 5 Pro（提示词优化，参考图须开 thinking）"),
        ImageModel("doubao-seedream-5.0-lite", "doubao-seedream-5-0-260128", "JimengSeedream5", "Seedream 5 Lite（组图/联网搜索/种子）"),
    ),
    size_options=_SIZES_V4,
    default_size="2K (adaptive)",
    supports_i2i=True,
    supports_group=True,
    notes="用 DynamicCombo，API 格式序列化与 4 不同；建议用 run --workflow 走手存图。",
    wired=False,
)

#: 预留（尚未接线）：通义万相 DashScope / 智谱 CogView——需各自的 ComfyUI 节点，登记后即可复用。
_DASHSCOPE_RESERVED = Provider(
    name="dashscope",
    label="通义万相 DashScope（预留）",
    node_class="",
    client_node="",
    models=(),
    size_options=(),
    default_size="",
    supports_i2i=False,
    supports_group=False,
    notes="预留位：接入对应 ComfyUI 节点后在此登记 models/size_options 即可。",
    wired=False,
)

_PROVIDERS: dict[str, Provider] = {
    _JIMENG_ARK.name: _JIMENG_ARK,
    _JIMENG_ARK_S3.name: _JIMENG_ARK_S3,
    _JIMENG_ARK_S5.name: _JIMENG_ARK_S5,
    _DASHSCOPE_RESERVED.name: _DASHSCOPE_RESERVED,
}

#: 默认 provider 名（即梦/方舟 Seedream 4）。
DEFAULT_PROVIDER = _JIMENG_ARK.name


def list_providers(*, wired_only: bool = False) -> list[Provider]:
    """列出已登记 provider（``wired_only=True`` 只返回已接线、可直接构造图的）。"""
    out = list(_PROVIDERS.values())
    if wired_only:
        out = [p for p in out if p.wired]
    return out


def get_provider(name: str | None = None) -> Provider:
    """按名取 provider；None/空 → 默认。未知名字抛 KeyError（附可选项）。"""
    key = (name or DEFAULT_PROVIDER).strip().lower()
    if key not in _PROVIDERS:
        raise KeyError(
            f"未知 provider {name!r}；可选 {sorted(_PROVIDERS)}（默认 {DEFAULT_PROVIDER}）"
        )
    return _PROVIDERS[key]


def resolve_model(provider_name: str | None, model: str | None) -> tuple[Provider, ImageModel]:
    """一步解析 (provider, model)。"""
    p = get_provider(provider_name)
    return p, p.resolve_model(model)
