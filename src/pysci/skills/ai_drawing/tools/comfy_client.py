"""ComfyUI REST/WS 客户端：提交工作流、轮询/监听进度、取回产物、内省节点。

pysci 与 ComfyUI 的**唯一**通信面（纯 HTTP，``requests`` 已在主依赖；``websocket-client`` 为
可选 extra ``[comfy]``，缺失时自动回退到轮询 ``/history``，功能不减、只少流式进度）。

ComfyUI HTTP API 契约（已核实）：
- ``GET  /system_stats``                —— 系统/版本信息（兼作廉价探活）。
- ``GET  /object_info[/{class_type}]``  —— 节点 schema 真值（**禁止猜** class_type/输入名）。
- ``POST /upload/image``                —— multipart 上传源图（i2i 用），返回 ``{name,subfolder,type}``。
- ``POST /prompt``  ``{"prompt":graph,"client_id":cid}`` —— 提交 API 格式图 → ``{prompt_id,...}``。
- ``GET  /history/{prompt_id}``         —— 执行结果（``status`` + ``outputs``）。
- ``GET  /view?filename=&subfolder=&type=`` —— 取回产物字节。
- ``GET  /queue``                       —— 队列状态。
- ``POST /interrupt``                   —— 中断当前执行。
- ``WS   /ws?clientId=cid``             —— 实时进度（executing/progress/executed/status）。

用法::

    from pysci.skills.ai_drawing.tools.comfy_client import ComfyClient

    c = ComfyClient()
    if c.is_reachable():
        pid = c.queue_prompt(graph)
        images = c.run_to_images(graph)   # 提交 + 等待 + 取回字节
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from .config import settings


class ComfyError(RuntimeError):
    """ComfyUI 通信 / 执行错误。"""


class ComfyClient:
    """ComfyUI HTTP(+WS) 客户端。

    Args:
        base_url: 服务器地址（默认取配置 ``COMFY_SERVER_URL``）。
        client_id: WS 会话标识（默认随机 UUID；``/prompt`` 与 ``/ws`` 用它关联进度）。
        timeout: 单次 HTTP 请求超时（秒）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        client_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or settings.comfy_server_url).rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # 探活 / 系统信息
    # ------------------------------------------------------------------
    def is_reachable(self, timeout: float = 2.0) -> bool:
        """廉价探活：``GET /system_stats`` 能否在超时内 200。"""
        try:
            r = self._session.get(f"{self.base_url}/system_stats", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def system_stats(self) -> dict[str, Any]:
        """``GET /system_stats`` → dict。不可达抛 :class:`ComfyError`。"""
        return self._get_json("/system_stats")

    # ------------------------------------------------------------------
    # 节点内省（schema 真值）
    # ------------------------------------------------------------------
    def object_info(self, class_type: str | None = None) -> dict[str, Any]:
        """``GET /object_info[/{class_type}]`` → 节点 schema。

        不带 ``class_type`` 返回全部节点（较大）；带则只返回该节点（推荐，用于核实输入名）。
        """
        path = "/object_info" if not class_type else f"/object_info/{class_type}"
        return self._get_json(path)

    def node_input_names(self, class_type: str) -> list[str]:
        """取某节点的输入名列表（``required`` + ``optional`` 合并）。schema 缺失抛错。"""
        info = self.object_info(class_type)
        node = info.get(class_type)
        if not node:
            raise ComfyError(f"/object_info 无节点 {class_type!r}（未安装该自定义节点？）")
        names: list[str] = []
        inp = node.get("input", {})
        for group in ("required", "optional"):
            for k in (inp.get(group) or {}):
                names.append(k)
        return names

    # ------------------------------------------------------------------
    # 上传源图（i2i）
    # ------------------------------------------------------------------
    def upload_image(
        self,
        path: str | Path,
        *,
        overwrite: bool = True,
        image_type: str = "input",
        subfolder: str = "",
    ) -> dict[str, Any]:
        """``POST /upload/image`` 上传一张源图，返回 ``{name, subfolder, type}``。"""
        p = Path(path)
        if not p.is_file():
            raise ComfyError(f"待上传源图不存在：{p}")
        with p.open("rb") as f:
            files = {"image": (p.name, f, "image/png")}
            data = {"overwrite": str(overwrite).lower(), "type": image_type, "subfolder": subfolder}
            try:
                r = self._session.post(
                    f"{self.base_url}/upload/image", files=files, data=data, timeout=self.timeout
                )
            except requests.RequestException as e:
                raise ComfyError(f"上传源图失败：{e!r}") from e
        if r.status_code != 200:
            raise ComfyError(f"上传源图失败 HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as e:
            raise ComfyError(f"上传源图响应非 JSON：{r.text[:200]}") from e

    # ------------------------------------------------------------------
    # 提交 / 队列 / 中断
    # ------------------------------------------------------------------
    def queue_prompt(self, graph: dict[str, Any]) -> str:
        """``POST /prompt`` 提交 API 格式图 → ``prompt_id``。

        服务器会先校验图；校验失败时 ``/prompt`` 返回 ``node_errors`` / ``error``，本方法抛
        :class:`ComfyError` 并附上详情（这是诊断 class_type/输入名错误的第一现场）。
        """
        payload = {"prompt": graph, "client_id": self.client_id}
        try:
            r = self._session.post(
                f"{self.base_url}/prompt", json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise ComfyError(f"提交工作流失败（服务器不可达？）：{e!r}") from e
        if r.status_code != 200:
            detail = r.text[:800]
            try:
                j = r.json()
                detail = json.dumps(j, ensure_ascii=False)[:800]
            except ValueError:
                pass
            raise ComfyError(f"提交工作流被拒 HTTP {r.status_code}: {detail}")
        j = r.json()
        pid = j.get("prompt_id")
        if not pid:
            raise ComfyError(f"提交工作流响应无 prompt_id：{json.dumps(j, ensure_ascii=False)[:400]}")
        if j.get("node_errors"):
            # 有 prompt_id 但同时报节点错误：仍抛出，避免静默出错图
            raise ComfyError(
                f"工作流节点错误（用 `imagine comfy nodes` 核实 class_type/输入名）："
                f"{json.dumps(j['node_errors'], ensure_ascii=False)[:600]}"
            )
        return pid

    def queue(self) -> dict[str, Any]:
        """``GET /queue`` → 队列状态。"""
        return self._get_json("/queue")

    def interrupt(self) -> None:
        """``POST /interrupt`` 中断当前执行。"""
        try:
            self._session.post(f"{self.base_url}/interrupt", timeout=self.timeout)
        except requests.RequestException as e:
            raise ComfyError(f"中断失败：{e!r}") from e

    # ------------------------------------------------------------------
    # 历史 / 产物
    # ------------------------------------------------------------------
    def history(self, prompt_id: str) -> dict[str, Any]:
        """``GET /history/{prompt_id}`` → ``{prompt_id: {status, outputs}}``（未完成时可能为空）。"""
        return self._get_json(f"/history/{prompt_id}")

    def get_view(
        self, filename: str, *, subfolder: str = "", view_type: str = "output"
    ) -> bytes:
        """``GET /view`` 取回一张产物的字节。"""
        params = {"filename": filename, "subfolder": subfolder, "type": view_type}
        try:
            r = self._session.get(f"{self.base_url}/view", params=params, timeout=self.timeout)
        except requests.RequestException as e:
            raise ComfyError(f"取回产物失败：{e!r}") from e
        if r.status_code != 200:
            raise ComfyError(f"取回产物失败 HTTP {r.status_code}（{filename}）")
        return r.content

    # ------------------------------------------------------------------
    # 等待完成（WS 优先，回退轮询）
    # ------------------------------------------------------------------
    def wait_until_done(
        self,
        prompt_id: str,
        *,
        timeout: float = 300.0,
        poll_interval: float = 1.5,
        progress: bool = True,
    ) -> dict[str, Any]:
        """阻塞直到 ``prompt_id`` 执行完成，返回其 history 条目。

        优先用 WebSocket 监听实时进度（``websocket-client`` 可用时）；否则/失败时回退到轮询
        ``/history``。超时抛 :class:`ComfyError`。执行报错（status.error）也抛错。
        """
        if progress and self._wait_via_ws(prompt_id, timeout):
            pass  # WS 已确认完成（或已尽力），下面统一读 history
        else:
            self._wait_via_polling(prompt_id, timeout, poll_interval)

        hist = self.history(prompt_id)
        entry = hist.get(prompt_id)
        if not entry:
            raise ComfyError(f"完成后 /history 无 {prompt_id}（可能被清理）")
        status = entry.get("status", {})
        if status.get("status_str") == "error" or status.get("completed") is False:
            msgs = status.get("messages", [])
            raise ComfyError(f"工作流执行报错：{json.dumps(msgs, ensure_ascii=False)[:600]}")
        return entry

    def _wait_via_polling(
        self, prompt_id: str, timeout: float, poll_interval: float
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self.history(prompt_id)
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                return
            if entry and entry.get("status", {}).get("status_str") == "error":
                return  # 交给 wait_until_done 统一抛错
            time.sleep(poll_interval)
        raise ComfyError(f"等待 {prompt_id} 完成超时（{timeout}s）")

    def _wait_via_ws(self, prompt_id: str, timeout: float) -> bool:
        """用 WS 监听直到本 prompt 完成。返回 True 表示 WS 路径已处理（含确认完成）。

        ``websocket-client`` 不可用或连接失败 → 返回 False（调用方回退轮询）。
        """
        try:
            import websocket  # type: ignore
        except ImportError:
            return False
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={self.client_id}"
        try:
            ws = websocket.create_connection(ws_url, timeout=timeout)
        except Exception:  # noqa: BLE001 - WS 不可用则回退轮询
            return False
        try:
            ws.settimeout(timeout)
            while True:
                try:
                    msg = ws.recv()
                except Exception:  # noqa: BLE001 - 超时/断开 → 回退轮询兜底
                    return False
                if isinstance(msg, bytes):
                    continue  # 预览帧（二进制），忽略
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                mtype = data.get("type")
                mdata = data.get("data", {})
                if mdata.get("prompt_id") and mdata["prompt_id"] != prompt_id:
                    continue
                if mtype == "executing" and mdata.get("node") is None:
                    return True  # 本 prompt 执行结束
                if mtype == "execution_error":
                    return True  # 交给 history 读取错误详情
                if mtype == "status":
                    continue
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 一站式：提交 + 等待 + 取回产物字节
    # ------------------------------------------------------------------
    def run_to_images(
        self,
        graph: dict[str, Any],
        *,
        timeout: float = 300.0,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """提交图、等待完成、取回所有产物，返回 ``[{filename, subfolder, type, data(bytes)}]``。

        这是 ``imagine gen/i2i/run`` 的核心驱动。无产物抛 :class:`ComfyError`。
        """
        pid = self.queue_prompt(graph)
        entry = self.wait_until_done(pid, timeout=timeout, progress=progress)
        outputs = entry.get("outputs", {})
        results: list[dict[str, Any]] = []
        for _nid, node_out in outputs.items():
            for img in node_out.get("images", []) or []:
                fn = img.get("filename")
                if not fn:
                    continue
                sub = img.get("subfolder", "")
                typ = img.get("type", "output")
                data = self.get_view(fn, subfolder=sub, view_type=typ)
                results.append(
                    {"filename": fn, "subfolder": sub, "type": typ, "data": data, "prompt_id": pid}
                )
        if not results:
            raise ComfyError(f"工作流 {pid} 完成但无图像产物（检查是否接了 SaveImage）")
        return results

    # ------------------------------------------------------------------
    # 内部：GET JSON
    # ------------------------------------------------------------------
    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            r = self._session.get(f"{self.base_url}{path}", timeout=self.timeout)
        except requests.RequestException as e:
            raise ComfyError(f"GET {path} 失败（服务器不可达？{self.base_url}）：{e!r}") from e
        if r.status_code != 200:
            raise ComfyError(f"GET {path} 失败 HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError as e:
            raise ComfyError(f"GET {path} 响应非 JSON：{r.text[:200]}") from e


def save_images(
    images: list[dict[str, Any]],
    dest_dir: str | Path,
    *,
    stem: str | None = None,
) -> list[Path]:
    """把 :meth:`ComfyClient.run_to_images` 的产物字节写到 ``dest_dir``，返回路径列表。

    多张时用 ``<stem>_<i>_<原文件名>`` 避免覆盖；``stem`` 为 None 时用原文件名。
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    multi = len(images) > 1
    for i, img in enumerate(images):
        fn = img.get("filename") or f"comfy_{i}.png"
        name = f"{stem}_{i}_{fn}" if (multi and stem) else (f"{stem}_{fn}" if stem else fn)
        # 兜住非法字符（ComfyUI 文件名一般安全，这里只替换路径分隔符）
        name = name.replace("/", "_").replace("\\", "_")
        out = dest / name
        out.write_bytes(img.get("data", b""))
        written.append(out)
    return written
