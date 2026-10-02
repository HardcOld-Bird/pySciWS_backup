"""火山方舟（即梦 Seedream）图片生成 API 客户端。

这是本技能**唯一的对外通信面**：直接以 OpenAI 兼容协议调用方舟
``POST {base}/images/generations``，用项目已有的 ``requests``，**零新增依赖**。

为什么不经 ComfyUI 编排：方舟图像 API 本身已内置图层拆分、交互编辑、组图生成等
复合能力，而社区第三方节点只暴露了其中一个字段子集；且本机无 CUDA GPU，节点图
的本地推理优势无法发挥。详见 ``.qoder/skills/ai-drawing/references/backends.md``。

已核实的 API 契约（2026-09，方舟官方文档 + 三家镜像文档交叉验证）：

请求字段
    ``model``        必需，Model ID 字符串（控制台「模型列表」为唯一真值源）
    ``prompt``       必需
    ``image``        可选，URL 或 base64 data URI 数组；5.0-pro ≤10 张，其余 ≤14 张
    ``size``         可选，``1K``/``2K``/``3K``/``4K`` 或 ``宽x高``
    ``width``/``height``  可选，自定义像素（与 size 二选一）
    ``response_format``   ``url``（默认）| ``b64_json``
    ``output_format``     ``png`` | ``jpeg``（默认 jpeg）；**仅 5.0 主档支持**
    ``watermark``         **方舟默认 true**，故本客户端总是显式传 false
    ``seed``              取值 [0, 65535]；**仅 3.0-t2i 生效**，且官方声明
                          "相同 seed 生成类似结果，但不保证完全一致" → 不可作复现依据
    ``sequential_image_generation``  ``auto`` 开启组图；仅 4.0/4.5/5.0 主档
    ``max_images``        组图张数；参考图数 + 产出数 ≤ 15
    ``tools``             ``[{"type": "web_search"}]`` 联网检索；仅 5.0 主档

响应字段
    ``data[i].url``       response_format=url 时返回，**24 小时后失效** → 必须立即下载
    ``data[i].b64_json``  response_format=b64_json 时返回
    ``data[i].size``      ``"<宽>x<高>"``
    ``data[i].error``     ``{code, message}``，**单张失败**也走这里（其余张仍成功）
    ``usage.generated_images``  成功张数（仅成功图片计费）
    ``error``             ``{code, message}``，整请求级错误

用法::

    from pysci.skills.ai_drawing.tools import ark_client

    resp = ark_client.generate("a red circle", model="doubao-seedream-5-0-flash-260915")
    paths = ark_client.save_images(resp, dest_dir, stem="circle")
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import settings

#: 图片生成端点相对路径（拼在 ``ark_base_url`` 之后）。
IMAGES_PATH: str = "/images/generations"

#: ``seed`` 合法取值范围（方舟硬约束）。
SEED_RANGE: tuple[int, int] = (0, 65535)

#: 参考图数量上限：5.0-pro/flash 为 10，其余（5.0 / 4.5 / 4.0）为 14。
REF_LIMIT_PRO: int = 10
REF_LIMIT_DEFAULT: int = 14

#: 档位标记。Model ID 用**中划线**分档（``doubao-seedream-5-0-260128``）；文档里的
#: "5.0-lite" 只是昵称——2026-10 用 ``GET /models`` 实测，真实 ID **没有** ``lite``
#: 后缀，把昵称写进匹配串会误拒合法请求。
TIER_50_MARKER: str = "5-0"
TIER_4X_MARKERS: tuple[str, ...] = ("4-0", "4-5")

#: 组图（sequential_image_generation）支持 4.0/4.5/5.0 **主档**；pro/flash 只能出单图。
GROUP_CAPABLE_MARKERS: tuple[str, ...] = TIER_4X_MARKERS + (TIER_50_MARKER,)

#: 参考图/组图张数硬上限（方舟：参考图数 + 产出数 ≤ 15）。
ABS_MAX_IMAGES: int = 15

#: 模型列表端点（只读、免费、不消耗额度）——账号实际可见的 Model ID 才是唯一真值源。
MODELS_PATH: str = "/models"


class ArkError(RuntimeError):
    """方舟 API 调用失败。携带 HTTP 状态码与方舟返回的 code/message 以便精准提示。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = body

    def hint(self) -> str:
        """把常见错误码翻译成可执行的下一步（省掉一轮盲目排查）。"""
        text = f"{self.code or ''} {self}".lower()
        if (
            self.status_code in (401, 403)
            or "unauthorized" in text
            or "invalidapikey" in text
        ):
            return (
                "ARK_API_KEY 无效或未配置：确认 .env 中 ARK_API_KEY 已粘贴完整"
                "（前后无空格），且创建 Key 时的项目空间与当前一致。"
            )
        if "notfound" in text or "不存在" in text or "invalidmodel" in text:
            return (
                "模型未开通或 ID 写错：到方舟控制台「开通管理」→「图像生成」逐个点开通，"
                "并以「模型列表」中的 Model ID 为准（注意 ID 不是模型昵称）。"
            )
        if "quota" in text or "limit" in text or "余额" in text or "欠费" in text:
            return "额度/限流问题：检查账户余额、免费额度是否用尽，或降低 max_images 后重试。"
        if "authentication" in text or "serviceunavailable" in text:
            return "方舟服务侧异常：稍后重试；若持续，检查网络能否访问 ark.cn-beijing.volces.com。"
        return ""


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------
@dataclass
class ArkImage:
    """单张产出。``error`` 非空表示该张失败（其余张可能仍成功）。"""

    index: int
    url: str | None = None
    b64_json: str | None = None
    size: str | None = None  # "<宽>x<高>"
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.url or self.b64_json)

    def to_bytes(self, *, timeout: float | None = None) -> bytes:
        """取回图像字节：优先本地已有的 b64_json，否则下载 url（**24h 内**）。"""
        if self.b64_json:
            return _decode_b64(self.b64_json)
        if not self.url:
            raise ArkError(f"第 {self.index} 张既无 b64_json 也无 url，无法取回")
        resp = requests.get(self.url, timeout=timeout or settings.timeout)
        if resp.status_code != 200:
            raise ArkError(
                f"下载产出图失败：HTTP {resp.status_code}", status_code=resp.status_code
            )
        return resp.content


@dataclass
class ArkResponse:
    """一次生成请求的完整结果。"""

    model: str = ""
    created: int = 0
    images: list[ArkImage] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(
        default_factory=dict
    )  # 发出时的请求体（脱敏，不含 key）
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok_images(self) -> list[ArkImage]:
        return [im for im in self.images if im.ok]

    @property
    def failed_images(self) -> list[ArkImage]:
        return [im for im in self.images if not im.ok]

    @property
    def generated_images(self) -> int:
        """成功张数（方舟按此计费）。"""
        v = self.usage.get("generated_images")
        return int(v) if isinstance(v, int) else len(self.ok_images)


# ---------------------------------------------------------------------------
# 输入编码与参数校验
# ---------------------------------------------------------------------------
def encode_image(path: str | Path) -> str:
    """把本地图编码成方舟接受的 data URI（``data:image/<fmt>;base64,<...>``，fmt 须小写）。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"参考图不存在：{p}")
    mime = mimetypes.guess_type(p.name)[0]
    if not mime or not mime.startswith("image/"):
        # 方舟仅接受 jpg/jpeg/png；按后缀兜底推断
        ext = p.suffix.lower().lstrip(".")
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext or 'png'}"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def normalize_images(images: Any, *, model: str) -> list[str]:
    """把 --image 参数（路径 / URL / data URI 混合）规整成方舟 image 数组。

    本地路径转 data URI（省掉"先上传再引用"这一步）；URL 与 data URI 原样透传。
    同时按模型校验张数上限。
    """
    if images is None:
        return []
    if isinstance(images, (str, Path)):
        images = [images]
    out: list[str] = []
    for it in images:
        s = str(it).strip()
        if not s:
            continue
        if s.startswith(("http://", "https://", "data:")):
            out.append(s)
        else:
            out.append(encode_image(s))
    limit = REF_LIMIT_PRO if _is_pro(model) else REF_LIMIT_DEFAULT
    if len(out) > limit:
        raise ArkError(
            f"参考图 {len(out)} 张超过 {model} 的上限 {limit} 张"
            f"（5.0-pro/flash ≤{REF_LIMIT_PRO}，其余 ≤{REF_LIMIT_DEFAULT}）"
        )
    return out


def check_seed(seed: int | None) -> None:
    """校验 seed 落在方舟合法区间；越界直接报错，不静默截断。"""
    if seed is None:
        return
    lo, hi = SEED_RANGE
    if not isinstance(seed, int) or not (lo <= seed <= hi):
        raise ArkError(f"seed={seed!r} 超出方舟合法范围 [{lo}, {hi}]")


def _norm_model(model: str) -> str:
    """Model ID 归一：小写 + 点转中划线（文档写 ``5.0``，真实 ID 是 ``5-0``）。"""
    return model.lower().replace(".", "-")


def _is_pro(model: str) -> bool:
    """pro/flash 档：参考图 ≤10，且不支持组图与 ``output_format``。"""
    m = model.lower()
    return "pro" in m or "flash" in m


def _is_png_capable(model: str) -> bool:
    """``output_format=png`` 目前仅 5.0 **主档**支持（ID 无 pro/flash 后缀）。"""
    return TIER_50_MARKER in _norm_model(model) and not _is_pro(model)


def _is_group_capable(model: str) -> bool:
    """组图仅 4.0/4.5/5.0 **主档**支持；pro/flash 只能出单图。"""
    m = _norm_model(model)
    return any(k in m for k in GROUP_CAPABLE_MARKERS) and not _is_pro(model)


def check_output_format(fmt: str | None, *, model: str) -> None:
    """``output_format=png`` 仅 5.0 主档支持；其余传 png 会失败，提前拦下并给替代路径。

    本地能力表会滞后于方舟上新，故报错里始终指一条核对真值的路（``models --live``）
    与一条绕过本检查的路（``--extra``），不让一张过期的表挡住合法请求。
    """
    if not fmt:
        return
    if fmt.lower() == "jpeg":
        return
    if not _is_png_capable(model):
        raise ArkError(
            f"模型 {model} 按本地能力表不支持 output_format={fmt}（png 目前仅 5.0 主档，"
            "如 doubao-seedream-5-0-260128）；其余模型恒输出 jpeg。要 PNG 请：① 换 5.0 主档，"
            "或 ② 出图后 `imagine adjust <jpg> --format png` 本地转（仅换容器）。"
            "能力表可能滞后：先 `imagine models --live` 核对真值，确已支持则用 "
            '`--extra \'{"output_format":"png"}\'` 绕过本检查。'
        )


def check_group(*, model: str, sequential: bool, max_images: int, n_refs: int) -> None:
    """校验组图参数：模型是否支持、总张数是否 ≤15。"""
    if not sequential:
        return
    if not _is_group_capable(model):
        raise ArkError(
            f"模型 {model} 按本地能力表不支持组图生成（sequential_image_generation）；"
            "仅 4.0 / 4.5 / 5.0 主档支持，pro/flash 只能出单图。"
            "能力表可能滞后，用 `imagine models --live` 核对账号实际可见的 Model ID。"
        )
    total = n_refs + max_images
    if total > ABS_MAX_IMAGES:
        raise ArkError(
            f"参考图 {n_refs} 张 + 产出 {max_images} 张 = {total}，超过方舟上限 "
            f"{ABS_MAX_IMAGES} 张（含参考图）。"
        )


def _decode_b64(s: str) -> bytes:
    """解码 b64_json，容忍 ``data:image/...;base64,`` 前缀与空白。"""
    payload = s.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload.strip(), validate=False)
    except (binascii.Error, ValueError) as e:
        raise ArkError(f"b64_json 解码失败：{e}") from e


def sniff_suffix(data: bytes, *, fallback: str = ".jpg") -> str:
    """用 Pillow 探测真实图像格式决定后缀。

    方舟 4.x 恒输出 jpeg，而代码/文档多处曾按 PNG 假设——这里一律以**实际字节**为准，
    不再猜。Pillow 不可用时按 magic number 粗判，再不行用 fallback。
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").lower()
        if fmt == "jpeg":
            return ".jpg"
        if fmt:
            return f".{fmt}"
    except Exception:  # noqa: BLE001
        pass
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return fallback


# ---------------------------------------------------------------------------
# 主调用
# ---------------------------------------------------------------------------
def build_payload(
    prompt: str,
    *,
    model: str | None = None,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    images: Any = None,
    seed: int | None = None,
    n: int = 1,
    sequential: bool = False,
    max_images: int | None = None,
    watermark: bool = False,
    output_format: str | None = None,
    response_format: str = "b64_json",
    tools: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造请求体（不发网络请求，便于 --dry-run 与单测）。

    所有方舟契约护栏都在此集中校验：seed 范围、参考图上限、output_format 支持度、
    组图张数、watermark 显式化。
    """
    mdl = model or settings.default_model
    if not prompt or not prompt.strip():
        raise ArkError("prompt 不能为空")

    refs = normalize_images(images, model=mdl)
    check_seed(seed)
    check_output_format(output_format, model=mdl)

    # 组图张数：显式 max_images > n > 成本护栏 clamp 后的 1
    group_count = max_images if max_images is not None else max(1, int(n or 1))
    group_count = max(
        1,
        min(
            int(group_count), settings.max_images if not sequential else ABS_MAX_IMAGES
        ),
    )
    check_group(
        model=mdl, sequential=sequential, max_images=group_count, n_refs=len(refs)
    )

    payload: dict[str, Any] = {
        "model": mdl,
        "prompt": prompt.strip(),
        # 方舟 watermark 默认 true，必须显式传 false，否则每张图右下角都有"AI 生成"水印
        "watermark": bool(watermark),
        "response_format": response_format,
    }
    if refs:
        payload["image"] = refs
    if size:
        payload["size"] = str(size)
    if width:
        payload["width"] = int(width)
    if height:
        payload["height"] = int(height)
    if seed is not None:
        payload["seed"] = int(seed)
    if output_format:
        payload["output_format"] = output_format.lower()
    if sequential:
        payload["sequential_image_generation"] = "auto"
        payload["max_images"] = group_count
    if tools:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    return payload


def generate(
    prompt: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> ArkResponse:
    """调用方舟图片生成 API 并解析响应。

    任何非 2xx 都抛 :class:`ArkError`（带可执行提示）；单张失败体现在
    ``ArkResponse.failed_images``，不视为整请求失败。
    """
    key = api_key if api_key is not None else settings.ark_api_key
    if not key:
        raise ArkError(
            "未配置 ARK_API_KEY：请在项目根 .env 中填入方舟 API Key"
            "（控制台「API Key 管理」创建；Secret 仅显示一次）。"
        )
    base = (base_url or settings.ark_base_url).rstrip("/")
    url = f"{base}{IMAGES_PATH}"
    payload = build_payload(prompt, **kwargs)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    http = session or requests
    try:
        resp = http.post(
            url, headers=headers, json=payload, timeout=timeout or settings.timeout
        )
    except requests.RequestException as e:
        raise ArkError(f"请求方舟失败（网络层）：{e}") from e

    # 请求体里的 image 可能是数 MB 的 data URI，快照前先截断，避免 runs/ 爆盘
    snapshot_payload = _redact_payload(payload)

    if resp.status_code != 200:
        code, msg, body = _parse_error_body(resp)
        err = ArkError(
            msg or f"HTTP {resp.status_code}",
            status_code=resp.status_code,
            code=code,
            body=body,
        )
        hint = err.hint()
        if hint:
            err.args = (f"{err} | 提示：{hint}",)
        raise err

    try:
        body = resp.json()
    except ValueError as e:
        raise ArkError(f"方舟返回非 JSON（HTTP {resp.status_code}）：{e}") from e

    if isinstance(body, dict) and body.get("error"):
        e = body["error"]
        err = ArkError(
            str(e.get("message") or "方舟返回 error"),
            status_code=resp.status_code,
            code=str(e.get("code") or "") or None,
            body=body,
        )
        hint = err.hint()
        if hint:
            err.args = (f"{err} | 提示：{hint}",)
        raise err

    return _parse_response(body, request=snapshot_payload)


def list_models(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """``GET {base}/models``：列出账号**实际可见**的 Model ID（只读、免费、不消耗额度）。

    本地能力矩阵（``imagine models``）是人工整理的导航，会随方舟上新而滞后；
    本函数是核对真值的唯一入口（``imagine models --live``）。返回 OpenAI 兼容
    响应里的 ``data`` 列表（每项含 ``id`` / ``created`` 等字段）。
    """
    key = api_key if api_key is not None else settings.ark_api_key
    if not key:
        raise ArkError(
            "未配置 ARK_API_KEY：models --live 需要真实密钥才能向方舟询问模型列表"
            "（离线时请直接看 `imagine models` 的本地矩阵）。"
        )
    base = (base_url or settings.ark_base_url).rstrip("/")
    http = session or requests
    try:
        resp = http.get(
            f"{base}{MODELS_PATH}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout or settings.timeout,
        )
    except requests.RequestException as e:
        raise ArkError(f"请求方舟模型列表失败（网络层）：{e}") from e
    if resp.status_code != 200:
        code, msg, body = _parse_error_body(resp)
        raise ArkError(
            msg or f"HTTP {resp.status_code}",
            status_code=resp.status_code,
            code=code,
            body=body,
        )
    try:
        body = resp.json()
    except ValueError as e:
        raise ArkError(
            f"方舟模型列表返回非 JSON（HTTP {resp.status_code}）：{e}"
        ) from e
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise ArkError(f"方舟模型列表形状异常（缺 data 数组）：{str(body)[:200]}")
    return [it for it in data if isinstance(it, dict)]


def _parse_response(body: dict[str, Any], *, request: dict[str, Any]) -> ArkResponse:
    data = body.get("data") or []
    images: list[ArkImage] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        images.append(
            ArkImage(
                index=i,
                url=item.get("url"),
                b64_json=item.get("b64_json"),
                size=item.get("size"),
                error=item.get("error"),
            )
        )
    return ArkResponse(
        model=str(body.get("model") or request.get("model") or ""),
        created=int(body.get("created") or 0),
        images=images,
        usage=body.get("usage") or {},
        request=request,
        raw=body,
    )


def _parse_error_body(resp: requests.Response) -> tuple[str | None, str | None, Any]:
    try:
        body = resp.json()
    except ValueError:
        return None, (resp.text or "")[:500] or None, None
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            return (
                str(e.get("code") or "") or None,
                str(e.get("message") or "") or None,
                body,
            )
        return None, str(body.get("message") or "") or None, body
    return None, None, body


def _redact_payload(payload: dict[str, Any], *, keep: int = 64) -> dict[str, Any]:
    """快照用：把 data URI 截断为前缀 + 长度，其余原样保留。"""
    out = dict(payload)
    imgs = out.get("image")
    if isinstance(imgs, list):
        out["image"] = [
            (f"{s[:keep]}…<data-uri {len(s)} chars>" if s.startswith("data:") else s)
            for s in imgs
            if isinstance(s, str)
        ]
    return out


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def save_images(
    resp: ArkResponse,
    dest_dir: str | Path,
    *,
    stem: str | None = None,
    timeout: float | None = None,
) -> list[Path]:
    """把响应中的图像写到 dest_dir，**立即下载**（url 24h 失效）。

    后缀由 :func:`sniff_suffix` 按真实字节判定，不依赖 output_format 假设。
    返回成功写入的路径列表（按 index 升序）。
    """
    d = Path(dest_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = stem or f"ark_{resp.created or int(time.time())}"
    ok = resp.ok_images
    written: list[Path] = []
    for im in ok:
        data = im.to_bytes(timeout=timeout)
        suffix = sniff_suffix(data)
        name = f"{base}{suffix}" if len(ok) == 1 else f"{base}_{im.index:02d}{suffix}"
        dest = _unique(d / name)
        dest.write_bytes(data)
        written.append(dest)
    return written


def save_snapshot(
    resp: ArkResponse, runs_dir: str | Path, *, stem: str | None = None
) -> Path:
    """把请求体 + 响应元信息（不含图像字节）存成 JSON，便于排障与复算参数。"""
    d = Path(runs_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = stem or f"ark_{resp.created or int(time.time())}"
    raw = dict(resp.raw)
    # 响应里的 b64_json 可能数 MB，快照中剥离
    for item in raw.get("data") or []:
        if isinstance(item, dict) and item.get("b64_json"):
            item["b64_json"] = f"<{len(item['b64_json'])} chars, stripped>"
    snap = {
        "request": resp.request,
        "response": raw,
        "usage": resp.usage,
        "failed": [{"index": im.index, "error": im.error} for im in resp.failed_images],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = _unique(d / f"{base}.json")
    p.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _unique(dest: Path) -> Path:
    """若已存在则追加 ``_1/_2/...``（避免静默覆盖既有产物）。"""
    if not dest.exists():
        return dest
    i = 1
    while True:
        cand = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        if not cand.exists():
            return cand
        i += 1
