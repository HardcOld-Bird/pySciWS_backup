"""Tripo 开放平台（图生 3D / 文生 3D）API 客户端。

本技能图生 3D 的**唯一直连通信面**（复杂模型不便手动建模时：参考照片 → GLB）。
仅用项目已有的 ``requests``，零新增依赖。交互通道 blender-mcp 亦内置 Tripo 工具，
但交付管线走本客户端（可无 Blender 会话、可无人值守、产物直接进验收门）——
两通道共享同一 ``TRIPO_API_KEY``，不构成重复实现（本模块是代码面，MCP 是会话面）。

API 契约（2026-10 依据 platform.tripo3d.ai 官方文档整理；**本机尚无 key，
未经实调验证**——首次真实调用时如有出入，以 HTTP 响应为准并回写本 docstring）：

鉴权
    所有请求带 ``Authorization: Bearer <TRIPO_API_KEY>``。

上传参考图
    ``POST {base}/v2/openapi/upload``，multipart 字段 ``file``
    → ``{"code": 0, "data": {"file_token": "..."}}``

创建任务
    ``POST {base}/v2/openapi/task``，JSON：
    - 图生 3D：``{"type": "image_to_model", "file": {"type": "<扩展名如 jpg>",
      "file_token": "..."}, ...选项}``
    - 文生 3D：``{"type": "text_to_model", "prompt": "...", ...选项}``
    - 常用选项：``pbr_model``（bool，PBR 材质）、``quad``（bool，四边面拓扑）、
      ``face_limit``（int，面数上限）、``model_version``（str，默认账号缺省档）
    → ``{"code": 0, "data": {"task_id": "..."}}``

轮询任务
    ``GET {base}/v2/openapi/task/{task_id}``
    → ``data.status``：``queued`` | ``running`` | ``success`` | ``failed`` | ``banned``
    → 成功时 ``data.output``：``model``（GLB 下载 URL）、``pbr_model``、
      ``rendered_image``（预览图）；**URL 有时效，成功后必须立即下载**。

余额
    ``GET {base}/v2/openapi/user/balance``（doctor 可用来验证 key 有效性）。

用法::

    from pysci.skills.modeling3d.tools import tripo_client

    glb = tripo_client.generate_to_file(
        image="bench_photo.jpg", out=Path("scene.glb"), pbr=True)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from .config import TRIPO_POLL_INTERVAL, settings


class TripoError(RuntimeError):
    """Tripo API 调用失败。携带 HTTP 状态码与平台 code/message 以便精准提示。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = body

    def hint(self) -> str:
        """把常见错误翻译成可执行的下一步。"""
        text = f"{self}".lower()
        if self.status_code in (401, 403) or "unauthorized" in text or "token" in text:
            return (
                "TRIPO_API_KEY 无效或未配置：到 platform.tripo3d.ai 创建 API Key，"
                "粘贴进项目根 .env 的 TRIPO_API_KEY=（前后无空格）。"
            )
        if "balance" in text or "credit" in text or "arrearage" in text:
            return "Tripo 积分不足或已欠费：到开放平台控制台充值。"
        if "task" in text and "failed" in text:
            return (
                "生成任务失败：参考图可能不满足要求（建议单物体、白底、分辨率≥512，"
                "避免多视角拼图），或换 --text 文生 3D 重试。"
            )
        return "查看上方 HTTP 状态码与响应体；契约详见 tripo_client.py 模块 docstring。"


def _headers() -> dict[str, str]:
    if not settings.tripo_api_key:
        raise TripoError("TRIPO_API_KEY 未配置", status_code=None)
    return {"Authorization": f"Bearer {settings.tripo_api_key}"}


def _raise_for_response(resp: requests.Response) -> dict[str, Any]:
    """非 2xx 或业务 code≠0 时抛 TripoError；否则返回 JSON。"""
    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text[:500]}
    if resp.status_code >= 400 or data.get("code", 0) != 0:
        raise TripoError(
            data.get("message", resp.reason or "unknown error"),
            status_code=resp.status_code,
            code=data.get("code"),
            body=data,
        )
    return data


def check_key() -> dict[str, Any]:
    """查询账户余额——最轻量的 key 有效性验证（doctor 用）。"""
    resp = requests.get(
        f"{settings.tripo_base_url}/v2/openapi/user/balance",
        headers=_headers(),
        timeout=30,
    )
    return _raise_for_response(resp)


def upload_image(image: str | Path) -> tuple[str, str]:
    """上传参考图。

    Args:
        image: 本地图片路径（jpg/png/webp）。

    Returns:
        (file_token, 扩展名)——创建 image_to_model 任务需要二者。
    """
    p = Path(image)
    ext = p.suffix.lstrip(".").lower()
    with p.open("rb") as f:
        resp = requests.post(
            f"{settings.tripo_base_url}/v2/openapi/upload",
            headers=_headers(),
            files={"file": (p.name, f)},
            timeout=120,
        )
    data = _raise_for_response(resp)
    token = data["data"]["file_token"]
    return token, ext


def create_task(
    *,
    image: str | Path | None = None,
    prompt: str | None = None,
    pbr: bool = False,
    quad: bool = False,
    face_limit: int | None = None,
    model_version: str | None = None,
) -> str:
    """创建生成任务（image 与 prompt 二选一，image 优先）。

    Returns:
        task_id。
    """
    payload: dict[str, Any] = {}
    if image is not None:
        token, ext = upload_image(image)
        payload.update(
            type="image_to_model", file={"type": ext or "jpg", "file_token": token}
        )
    elif prompt is not None:
        payload.update(type="text_to_model", prompt=prompt)
    else:
        raise TripoError("image 与 prompt 至少提供一个")
    if pbr:
        payload["pbr_model"] = True
    if quad:
        payload["quad"] = True
    if face_limit is not None:
        payload["face_limit"] = face_limit
    if model_version:
        payload["model_version"] = model_version

    resp = requests.post(
        f"{settings.tripo_base_url}/v2/openapi/task",
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    data = _raise_for_response(resp)
    return str(data["data"]["task_id"])


def poll_task(task_id: str, *, timeout: float | None = None) -> dict[str, Any]:
    """轮询任务直至 success/failed。

    Args:
        task_id: 任务 ID。
        timeout: 超时秒数（默认 settings.tripo_timeout）。

    Returns:
        终态任务的 ``data`` 字典（含 ``output``）。

    Raises:
        TripoError: 任务失败/被禁，或轮询超时。
    """
    deadline = time.monotonic() + (timeout or settings.tripo_timeout)
    last_status = ""
    while True:
        resp = requests.get(
            f"{settings.tripo_base_url}/v2/openapi/task/{task_id}",
            headers=_headers(),
            timeout=30,
        )
        data = _raise_for_response(resp)["data"]
        status = data.get("status", "unknown")
        if status != last_status:
            print(f"[tripo] task {task_id}: {status}")
            last_status = status
        if status == "success":
            return data
        if status in ("failed", "banned", "unknown"):
            raise TripoError(f"任务终态 {status}", body=data)
        if time.monotonic() > deadline:
            raise TripoError(f"轮询超时（{timeout or settings.tripo_timeout}s）")
        time.sleep(TRIPO_POLL_INTERVAL)


def download(url: str, dest: str | Path) -> Path:
    """下载产物（GLB/预览图）到本地。URL 有时效，成功后应立即下载。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def generate_to_file(
    *,
    out: str | Path,
    image: str | Path | None = None,
    prompt: str | None = None,
    pbr: bool = False,
    quad: bool = False,
    face_limit: int | None = None,
    model_version: str | None = None,
    timeout: float | None = None,
    keep_preview: bool = True,
) -> Path:
    """端到端：创建任务 → 轮询 → 下载 GLB（可选同时下载预览图 ``<out>.preview.png``）。

    Returns:
        GLB 本地路径。

    Raises:
        TripoError: 任一环节失败（调用方用 ``hint()`` 展示下一步）。
    """
    task_id = create_task(
        image=image,
        prompt=prompt,
        pbr=pbr,
        quad=quad,
        face_limit=face_limit,
        model_version=model_version,
    )
    data = poll_task(task_id, timeout=timeout)
    output = data.get("output") or {}
    model_url = output.get("pbr_model") if pbr else None
    model_url = model_url or output.get("model")
    if not model_url:
        raise TripoError("任务成功但 output 中没有 model URL", body=data)
    out_path = download(model_url, out)
    preview = output.get("rendered_image")
    if keep_preview and preview:
        download(preview, out_path.with_suffix(".preview.png"))
    return out_path
