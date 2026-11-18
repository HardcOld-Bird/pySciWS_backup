"""出图护栏：watermark 显式关闭 / seed 兼容档 / b64 规避 URL 失效 / 成本上限。

每条护栏对应一个真实事故面（清单详见
``.agents/skills/image-generation/references/generation-guardrails.md``）：

- **watermark**：方舟默认 ``watermark=true``（右下角"AI 生成"水印），科研交付图
  不能带 → 请求恒**显式** ``watermark=false``，除非调用方明确 ``allow_watermark=True``；
- **seed 兼容档**：两家 seed 语义不同——方舟 seed ∈ [0, 65535] 且**仅 3.0-t2i 生效**、
  官方声明"类似结果不保证一致"→ 只记账不作复现依据；硅基流动 seed ∈ [0, 9999999999]
  **全档生效**（可复现）。统一按 provider 校验取值范围，越界报错不静默截断；
- **b64 规避 URL 失效**：方舟 ``response_format=b64_json``（url 24h 失效）、硅基流动
  只出 url（**1h 失效**）→ 方舟默认 b64，副通道必须立即下载（:func:`url_ttl_note`）；
- **成本上限**：单次请求张数 clamp（默认 :data:`DEFAULT_MAX_IMAGES`，可用环境变量
  ``IMAGEGEN_MAX_IMAGES`` 覆盖）；方舟组图另有硬上限 15 张（含参考图）。

用法（providers.build_payload 内部已接线，也可单独调用判定）::

    from pysci.skills.image_generation import guardrails

    guardrails.check_seed(777, "siliconflow")   # OK
    guardrails.check_seed(777, "ark")           # OK（≤65535）
    guardrails.clamp_count(10)                  # → 4（成本护栏）
"""

from __future__ import annotations

import os

#: 通道名 → seed 合法区间（官方文档契约；方舟仅 3.0-t2i 认 seed）。
SEED_RANGE: dict[str, tuple[int, int]] = {
    "ark": (0, 65535),
    "siliconflow": (0, 9999999999),
}

#: 单次请求默认最大出图张数（成本护栏；环境变量 IMAGEGEN_MAX_IMAGES 可覆盖）。
DEFAULT_MAX_IMAGES: int = 4

#: 方舟组图硬上限：参考图数 + 产出数 ≤ 15（官方契约）。
ARK_ABS_MAX_IMAGES: int = 15

#: URL 时效备注（写进账本 extra，提醒"立即下载"）。
_URL_TTL: dict[str, str] = {
    "ark": "url 24h 失效（默认走 b64_json）",
    "siliconflow": "url 1h 失效（仅出 url，必须立即下载）",
}


class GuardrailError(ValueError):
    """出图护栏拒绝：参数会触发水印/复现/成本/契约风险时抛出。"""


def max_images_cap() -> int:
    """当前生效的单次出图张数上限（成本护栏）。

    环境变量 ``IMAGEGEN_MAX_IMAGES`` 优先；非法值回退 :data:`DEFAULT_MAX_IMAGES`。
    """
    raw = os.environ.get("IMAGEGEN_MAX_IMAGES", "").strip()
    if not raw:
        return DEFAULT_MAX_IMAGES
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_IMAGES


def clamp_count(n: int, *, cap: int | None = None) -> int:
    """把请求张数 clamp 到 ``[1, cap]``（成本护栏；cap 默认 :func:`max_images_cap`）。"""
    limit = cap if cap is not None else max_images_cap()
    return max(1, min(int(n), max(1, int(limit))))


def check_seed(seed: int | None, provider: str, *, model: str = "") -> int | None:
    """按 provider 校验 seed 兼容档（区间 + 生效性备注），越界抛 :class:`GuardrailError`。

    Args:
        seed: 请求的随机种子；None 表示不固定（直接放行）。
        provider: ``"ark"`` | ``"siliconflow"``。
        model: 模型 ID，仅用于报错提示（方舟 seed 仅 3.0-t2i 生效）。

    Returns:
        原样返回 ``seed``（None 亦然），便于链式调用。
    """
    if seed is None:
        return None
    if provider not in SEED_RANGE:
        raise GuardrailError(f"未知 provider {provider!r}；支持 {sorted(SEED_RANGE)}")
    lo, hi = SEED_RANGE[provider]
    if not isinstance(seed, int) or isinstance(seed, bool) or not (lo <= seed <= hi):
        raise GuardrailError(
            f"seed={seed!r} 超出 {provider} 合法范围 [{lo}, {hi}]"
            + (f"（model={model}）" if model else "")
            + ("；方舟 seed 仅 3.0-t2i 生效且非确定，不可作复现依据" if provider == "ark" else "")
        )
    return seed


def apply_watermark(payload: dict, *, allow_watermark: bool = False) -> dict:
    """把 ``watermark`` 显式写入请求体（默认 ``False``——方舟默认加水印）。

    Args:
        payload: 待发送的请求体（原地修改并返回）。
        allow_watermark: True 时显式保留水印（如需可溯源的草稿图）；默认 False。

    Returns:
        ``payload`` 自身（便于链式调用）。
    """
    payload["watermark"] = bool(allow_watermark)
    return payload


def check_images_budget(
    n_images: int,
    *,
    provider: str,
    n_refs: int = 0,
    sequential: bool = False,
    cap: int | None = None,
) -> int:
    """校验并 clamp 一次请求的出图张数（成本护栏 + 方舟组图硬上限）。

    Args:
        n_images: 想要的产出张数。
        provider: ``"ark"`` | ``"siliconflow"``。
        n_refs: 参考图张数（方舟组图计"参考+产出 ≤ 15"）。
        sequential: 是否方舟组图（sequential_image_generation）。
        cap: 覆盖成本上限（默认 :func:`max_images_cap`）。

    Returns:
        clamp 后的张数。

    Raises:
        GuardrailError: 方舟组图"参考+产出"超过 15 张硬上限时。
    """
    if provider == "ark" and sequential:
        total = int(n_refs) + int(n_images)
        if total > ARK_ABS_MAX_IMAGES:
            raise GuardrailError(
                f"参考图 {n_refs} 张 + 产出 {n_images} 张 = {total}，超过方舟组图硬上限 "
                f"{ARK_ABS_MAX_IMAGES} 张（含参考图）"
            )
        return max(1, int(n_images))
    return clamp_count(n_images, cap=cap)


def url_ttl_note(provider: str) -> str:
    """该通道产出 URL 的时效备注（写账本 extra 用）。"""
    return _URL_TTL.get(provider, "url 时效未知，立即下载")


def preferred_response_format(provider: str) -> str:
    """默认取图方式：方舟 ``b64_json``（规避 url 24h 失效）；硅基流动仅支持 url。"""
    return "b64_json" if provider == "ark" else "url"
