"""双云出图统一客户端：火山方舟（主）+ 硅基流动（副）。

两者都是 OpenAI 兼容 ``POST {base}/images/generations``（文生图 / 图生图），
故共用一个 requests 客户端 + 双 provider 参数映射，**零新增依赖**。

ADR（2026-11-17，按官方文档实现、真实行为未实测）：API 契约取自方舟图片生成教程
（2026-09 三方交叉验证 + Seedream 5.5 教程）与硅基流动 OpenAPI schema（2026-11-17
抓取）；ARK_API_KEY 已过期，真实出图冒烟挂起（``tests/skills/image_generation/
smoke_real.py``，key 更新后跑）。模型 ID 一律**字符串直传**（控制台/模型广场为唯一
真值源），本地不维护模型注册表。

契约要点速查（护栏细则见 guardrails 模块与 references/generation-guardrails.md）：

- **方舟** ``ark``：``size`` 1K/2K/3K/4K 或 宽x高；watermark 默认 true 须显式 false；
  ``response_format=b64_json`` 规避 url 24h 失效；seed [0,65535] 仅 3.0-t2i 生效；
  组图 ``sequential_image_generation="auto"``（4.0/4.5/5.0 主档，参考+产出 ≤15）；
  单张失败走 ``data[i].error``；5.5 透明底直出 ``background="transparent"`` +
  ``output_format="png"``；5.0-pro 图层拆分/交互编辑走 ``extra`` 透传。
- **硅基流动** ``siliconflow``：``image_size`` "宽x高"；``batch_size`` ≤4；seed
  [0,9999999999] 全档生效（可复现）；图生图字段 ``image``（Qwen-Image-Edit /
  FLUX.1-Kontext-dev）或 ``input_image``（FLUX.1-Kontext-max/pro）；响应只出
  ``images[].url``（**1h 失效**）→ 立即下载。

key 读项目根 ``.env``：``ARK_API_KEY`` / ``SILICONFLOW_API_KEY`` /
``SILICONFLOW_BASE_URL``。**401/403 抛 :class:`ProviderError`（带可执行提示），
不裸崩**——key 过期是本项目的常态（冒烟挂起），错误必须可读可处置。

用法::

    from pysci.skills.image_generation import providers

    resp = providers.generate(
        "ark", "a minimal schematic of a lithium cell, flat vector style",
        model="doubao-seedream-5-5", size="2K", n=1,
    )
    paths = providers.save_images(resp, "data/research/1_demo/article/artwork")
"""

from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from pysci.paths import PROJECT_ROOT

from . import guardrails

#: 图片生成端点相对路径（拼在各 provider base_url 之后）。
IMAGES_PATH: str = "/images/generations"

#: 方舟默认推理端点与默认模型（控制台「模型列表」为唯一真值源，模型字符串直传）。
ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
ARK_DEFAULT_MODEL: str = "doubao-seedream-5-5"

#: 硅基流动默认端点与默认模型（OpenAPI enum 之一）。
SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"
SILICONFLOW_DEFAULT_MODEL: str = "Qwen/Qwen-Image"

#: 单次请求默认超时（秒）：云端同步出图，2K 通常 30–60s。
DEFAULT_TIMEOUT: float = 180.0


@dataclass(frozen=True)
class ProviderSpec:
    """一个云通道的静态配置。

    Attributes:
        name: 通道名（``ark`` | ``siliconflow``）。
        base_url: OpenAI 兼容 base（拼 ``/images/generations``）。
        key_env: API key 的环境变量名（项目根 .env）。
        base_env: 覆盖 base_url 的环境变量名。
        default_model: 无参时的默认模型 ID。
    """

    name: str
    base_url: str
    key_env: str
    base_env: str
    default_model: str


#: 双云通道注册（薄胶水只认这两家；扩通道在此加一行 + guardrails.SEED_RANGE）。
PROVIDERS: dict[str, ProviderSpec] = {
    "ark": ProviderSpec("ark", ARK_BASE_URL, "ARK_API_KEY", "ARK_BASE_URL", ARK_DEFAULT_MODEL),
    "siliconflow": ProviderSpec(
        "siliconflow",
        SILICONFLOW_BASE_URL,
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_BASE_URL",
        SILICONFLOW_DEFAULT_MODEL,
    ),
}


class ProviderError(RuntimeError):
    """云 API 调用失败（网络/鉴权/契约/响应形状）。

    携带 HTTP 状态码与上游 code/message；:meth:`hint` 把常见错误翻译成可执行的
    下一步——尤其 401/403（key 过期/无效）要能一眼看出去哪换 key。
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        code: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.code = code
        self.body = body

    def hint(self) -> str:
        """常见错误 → 可执行处置建议（无建议返回空串）。"""
        spec = PROVIDERS.get(self.provider)
        key_env = spec.key_env if spec else "对应 API key"
        text = f"{self.code or ''} {self}".lower()
        if self.status_code in (401, 403) or "unauthorized" in text or "invalid" in text:
            return (
                f"{key_env} 无效/过期/未配置：到服务商控制台重新创建 key，更新项目根 .env"
                "（前后无空格），再重试；本项目对 401 的口径是「优雅报错不裸崩」。"
            )
        if "notfound" in text or "not found" in text or "invalidmodel" in text:
            return "模型未开通或 ID 写错：以控制台/模型广场的 Model ID 为准（不是模型昵称）。"
        if "quota" in text or "limit" in text or "余额" in text or "429" in str(self.status_code):
            return "额度/限流问题：检查余额与免费额度，或降低出图张数（成本护栏）后重试。"
        if "overloaded" in text or "503" in str(self.status_code) or "504" in str(self.status_code):
            return "服务侧过载/超时：稍后重试；持续失败再查网络连通性。"
        return ""


# ---------------------------------------------------------------------------
# .env 加载与通道解析
# ---------------------------------------------------------------------------
def _load_env_file() -> None:
    """把项目根 .env 的键值并入 os.environ（已设的不覆盖；不硬依赖 python-dotenv）。"""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def resolve(provider: str) -> ProviderSpec:
    """取通道配置（未知通道报错，不猜）。

    Args:
        provider: ``"ark"`` | ``"siliconflow"``。

    Raises:
        ProviderError: 未知通道名。
    """
    spec = PROVIDERS.get(str(provider).strip().lower())
    if spec is None:
        raise ProviderError(f"未知 provider {provider!r}；支持 {sorted(PROVIDERS)}")
    return spec


def base_url(provider: str) -> str:
    """该通道的 base URL（环境变量 ``*_BASE_URL`` 可覆盖）。"""
    spec = resolve(provider)
    _load_env_file()
    return (os.environ.get(spec.base_env, "").strip() or spec.base_url).rstrip("/")


def api_key(provider: str) -> str | None:
    """该通道的 API key（.env / 环境变量；未配置返回 None）。"""
    spec = resolve(provider)
    _load_env_file()
    v = os.environ.get(spec.key_env, "").strip()
    return v or None


# ---------------------------------------------------------------------------
# 输入编码
# ---------------------------------------------------------------------------
def encode_image(path: str | Path) -> str:
    """把本地图编码成 data URI（``data:image/<fmt>;base64,...``，两家通吃）。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"参考图不存在：{p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    if not mime.startswith("image/"):
        mime = "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def normalize_images(images: Any) -> list[str]:
    """把参考图参数（路径 / URL / data URI 混合）规整成 data URI/URL 列表。"""
    if images is None:
        return []
    if isinstance(images, (str, Path)):
        images = [images]
    out: list[str] = []
    for it in images:
        s = str(it).strip()
        if not s:
            continue
        out.append(s if s.startswith(("http://", "https://", "data:")) else encode_image(s))
    return out


# ---------------------------------------------------------------------------
# 请求体拼装（不发网络，便于 mock 单测）
# ---------------------------------------------------------------------------
def build_payload(
    provider: str,
    prompt: str,
    *,
    model: str | None = None,
    images: Any = None,
    size: str | None = None,
    n: int = 1,
    seed: int | None = None,
    sequential: bool = False,
    transparent: bool = False,
    response_format: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造该通道的请求体；护栏在此集中生效（watermark/seed/成本/格式）。

    Args:
        provider: ``"ark"`` | ``"siliconflow"``。
        prompt: 文生图提示词（图生图时为编辑意图）。
        model: 模型 ID 字符串（默认取通道默认模型）。
        images: 参考图（路径 / URL / data URI 或其列表）→ 图生图。
        size: 尺寸——方舟 ``"2K"`` 或 ``"1024x1024"``；硅基 ``"1024x1024"``。
        n: 产出张数（成本护栏 clamp；方舟组图走 ``sequential``）。
        seed: 随机种子（兼容档校验见 :mod:`guardrails`）。
        sequential: 方舟组图（``sequential_image_generation="auto"``）。
        transparent: 透明底直出（方舟 5.5 ``background="transparent"`` + png；
            硅基无此参数，由 postprocess.rembg 兜底）。
        response_format: 取图方式（默认按通道取 :func:`guardrails.preferred_response_format`）。
        extra: 上游参数透传（图层拆分/mark/负面词/步数等，覆盖同名键）。

    Returns:
        可直接 POST JSON 的请求体。

    Raises:
        guardrails.GuardrailError: 护栏拒绝（seed 越界、组图超 15 张等）。
        ProviderError: 未知通道 / prompt 为空。
    """
    spec = resolve(provider)
    if not prompt or not prompt.strip():
        raise ProviderError("prompt 不能为空", provider=spec.name)
    refs = normalize_images(images)
    guardrails.check_seed(seed, spec.name, model=model or spec.default_model)

    payload: dict[str, Any] = {
        "model": (model or spec.default_model),
        "prompt": prompt.strip(),
    }
    # 护栏：watermark 恒显式 false（方舟默认加水印）。
    guardrails.apply_watermark(payload)

    if spec.name == "ark":
        fmt = response_format or guardrails.preferred_response_format("ark")
        payload["response_format"] = fmt
        if refs:
            payload["image"] = refs
        if size:
            payload["size"] = str(size)
        if transparent:
            # Seedream 5.5 透明底直出（background + png）；5.0 仅图生图保透明。
            payload["background"] = "transparent"
            payload["output_format"] = "png"
        if sequential:
            count = guardrails.check_images_budget(
                n, provider="ark", n_refs=len(refs), sequential=True
            )
            payload["sequential_image_generation"] = "auto"
            payload["max_images"] = count
        if seed is not None:
            payload["seed"] = int(seed)
    else:  # siliconflow
        count = guardrails.check_images_budget(n, provider=spec.name, n_refs=len(refs))
        if refs:
            payload[_sf_image_key(payload["model"])] = refs[0]
        if size:
            payload["image_size"] = str(size)
        if count > 1:
            payload["batch_size"] = count
        if seed is not None:
            payload["seed"] = int(seed)
        if response_format and response_format != "url":
            raise ProviderError(
                "硅基流动 OpenAPI schema 无 response_format 字段（仅出 url，1h 失效）——"
                "请省略本参数并在收到响应后立即下载。",
                provider=spec.name,
            )

    if extra:
        payload.update(extra)
    return payload


def _sf_image_key(model: str) -> str:
    """硅基图生图字段：Kontext max/pro 用 ``input_image``，其余（Qwen-Image-Edit /
    FLUX.1-Kontext-dev）用 ``image``。ADR：按 OpenAPI schema 映射，未实测。"""
    m = model.lower()
    return "input_image" if "kontext" in m and not m.endswith("-dev") else "image"


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------
@dataclass
class GeneratedImage:
    """单张产出。``error`` 非空表示该张失败（其余张可能仍成功）。"""

    index: int
    url: str | None = None
    b64_json: str | None = None
    size: str | None = None
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        """该张是否成功（有图且无 error）。"""
        return not self.error and bool(self.url or self.b64_json)

    def to_bytes(self, *, timeout: float | None = None) -> bytes:
        """取回图像字节：优先本地 b64_json，否则**立即**下载 url（1h/24h 内）。"""
        if self.b64_json:
            return _decode_b64(self.b64_json)
        if not self.url:
            raise ProviderError(f"第 {self.index} 张既无 b64_json 也无 url，无法取回")
        resp = requests.get(self.url, timeout=timeout or DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            raise ProviderError(
                f"下载产出图失败：HTTP {resp.status_code}", status_code=resp.status_code
            )
        return resp.content


@dataclass
class GenerationResult:
    """一次生成请求的统一结果（方舟/硅基响应形状归一）。"""

    provider: str
    model: str = ""
    created: int = 0
    images: list[GeneratedImage] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    request: dict[str, Any] = field(default_factory=dict)  # 脱敏请求快照（无 key）
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok_images(self) -> list[GeneratedImage]:
        """成功产出列表。"""
        return [im for im in self.images if im.ok]

    @property
    def failed_images(self) -> list[GeneratedImage]:
        """单张失败列表（``data[i].error`` 语义）。"""
        return [im for im in self.images if not im.ok]


# ---------------------------------------------------------------------------
# 主调用
# ---------------------------------------------------------------------------
def generate(
    provider: str,
    prompt: str,
    *,
    model: str | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> GenerationResult:
    """调用该通道的图片生成 API 并解析为统一结果。

    任何非 2xx 抛 :class:`ProviderError`（带 :meth:`ProviderError.hint` 提示）；
    **单张失败**体现在 ``GenerationResult.failed_images``，不算整请求失败。
    401/403（key 过期）同样走 :class:`ProviderError`——优雅报错不裸崩。
    """
    spec = resolve(provider)
    key = api_key(spec.name)
    if not key:
        raise ProviderError(
            f"未配置 {spec.key_env}：请在项目根 .env 填入该通道 API Key",
            provider=spec.name,
            status_code=401,
        )
    payload = build_payload(spec.name, prompt, model=model, **kwargs)
    url = f"{base_url(spec.name)}{IMAGES_PATH}"
    http = session or requests
    try:
        resp = http.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout or DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise ProviderError(f"请求 {spec.name} 失败（网络层）：{e}", provider=spec.name) from e

    snapshot = _redact_payload(payload)
    code, msg, body = _parse_body(resp)
    if resp.status_code != 200:
        err = ProviderError(
            msg or f"HTTP {resp.status_code}",
            provider=spec.name,
            status_code=resp.status_code,
            code=code,
            body=body,
        )
        hint = err.hint()
        if hint:
            err.args = (f"{err} | 提示：{hint}",)
        raise err
    return _parse_response(spec.name, body, model=str(payload.get("model") or ""), request=snapshot)


def _parse_body(resp: requests.Response) -> tuple[str | None, str | None, Any]:
    """解析响应体：兼容方舟 ``error{code,message}``、硅基 ``{code,message}`` 与纯字符串。"""
    try:
        body = resp.json()
    except ValueError:
        return None, (resp.text or "")[:500] or None, None
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            return str(e.get("code") or "") or None, str(e.get("message") or "") or None, body
        return (
            str(body.get("code") or "") or None,
            str(body.get("message") or "") or None,
            body,
        )
    return None, str(body)[:500] if body else None, body


def _parse_response(
    provider: str, body: Any, *, model: str, request: dict[str, Any]
) -> GenerationResult:
    """把两家响应形状归一成 :class:`GenerationResult`。

    方舟：``data[i].{url,b64_json,size,error}`` + ``usage.generated_images``；
    硅基：``images[i].{url}`` + 顶层 ``seed``（timings.inference）。
    """
    if not isinstance(body, dict):
        raise ProviderError(f"{provider} 响应非 JSON 对象：{str(body)[:200]}", provider=provider)
    items: list[dict[str, Any]] = []
    if isinstance(body.get("data"), list):
        items = [it for it in body["data"] if isinstance(it, dict)]
    elif isinstance(body.get("images"), list):
        items = [it for it in body["images"] if isinstance(it, dict)]
    images = [
        GeneratedImage(
            index=i,
            url=it.get("url"),
            b64_json=it.get("b64_json"),
            size=it.get("size"),
            error=it.get("error") if isinstance(it.get("error"), dict) else None,
        )
        for i, it in enumerate(items)
    ]
    seed_raw = body.get("seed")
    return GenerationResult(
        provider=provider,
        model=str(body.get("model") or model),
        created=int(body.get("created") or 0),
        images=images,
        usage=body.get("usage") or {},
        seed=int(seed_raw) if isinstance(seed_raw, int) else None,
        request=request,
        raw=body,
    )


def _redact_payload(payload: dict[str, Any], *, keep: int = 64) -> dict[str, Any]:
    """账本/快照用：data URI 截断为前缀 + 长度（防数 MB 参考图进账本）。"""
    out = dict(payload)
    for key in ("image", "input_image"):
        v = out.get(key)
        if isinstance(v, list):
            out[key] = [
                f"{s[:keep]}…<data-uri {len(s)} chars>" if s.startswith("data:") else s
                for s in v
                if isinstance(s, str)
            ]
        elif isinstance(v, str) and v.startswith("data:"):
            out[key] = f"{v[:keep]}…<data-uri {len(v)} chars>"
    return out


def _decode_b64(s: str) -> bytes:
    """解码 b64_json（容忍 ``data:image/...;base64,`` 前缀与空白）。"""
    payload = s.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload.strip(), validate=False)
    except (binascii.Error, ValueError) as e:
        raise ProviderError(f"b64_json 解码失败：{e}") from e


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def sniff_suffix(data: bytes, *, fallback: str = ".png") -> str:
    """按**真实字节**判定图像格式后缀（Pillow 探测，magic number 兜底）。"""
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
    return fallback


def save_images(
    resp: GenerationResult,
    dest_dir: str | Path,
    *,
    stem: str | None = None,
    timeout: float | None = None,
) -> list[Path]:
    """把产出写到 dest_dir（url 立即下载，规避时效）；后缀按真实字节判定。

    Returns:
        成功写入的路径列表（按 index 升序）。已存在同名时追加 ``_1/_2``，不覆盖。
    """
    d = Path(dest_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = stem or f"{resp.provider}_{resp.created or int(time.time())}"
    written: list[Path] = []
    ok = resp.ok_images
    for im in ok:
        data = im.to_bytes(timeout=timeout)
        name = f"{base}{sniff_suffix(data)}" if len(ok) == 1 else f"{base}_{im.index:02d}{sniff_suffix(data)}"
        dest = _unique(d / name)
        dest.write_bytes(data)
        written.append(dest)
    return written


def _unique(dest: Path) -> Path:
    """若已存在则追加 ``_1/_2/...``（不静默覆盖既有产物）。"""
    if not dest.exists():
        return dest
    i = 1
    while True:
        cand = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        if not cand.exists():
            return cand
        i += 1
