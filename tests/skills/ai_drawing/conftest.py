"""ai_drawing 测试的共享 fixture 与守卫（对齐 ``tests/skills/comsol_simulation/conftest.py``）。

约定：
- 纯 Python 单测不依赖 ComfyUI 服务器 / 火山方舟 key，始终运行——因此这里**不**用
  ``collect_ignore_glob`` 整目录忽略收集。
- 需要活体 ComfyUI 编排器或真实云端生成的 hardware 冒烟用例，通过 ``comfy_server`` /
  ``ark_key`` fixture 守卫：服务器不可达、或 key 未配置时 ``pytest.skip``（而非 collection error）。
  这些用例另标 ``hardware`` + ``slow``，默认 ``-m "not hardware"`` 下不收集。
"""

from __future__ import annotations

import pytest

from pysci.skills.ai_drawing.tools import comfy_session
from pysci.skills.ai_drawing.tools.config import settings


def comfy_available() -> bool:
    """ComfyUI 编排器服务器当前是否可达（端口存活；不启服务、不占额度）。"""
    return comfy_session.is_available()


@pytest.fixture(scope="session")
def comfy_server():
    """可达的 ComfyUI 客户端；服务器不可达时 skip。

    返回一个已确认可达的 :class:`ComfyClient`（供 hardware 用例做 ``/system_stats``、
    ``/object_info`` 内省等只读操作）。
    """
    if not comfy_available():
        pytest.skip(
            f"需要可达的 ComfyUI 编排器（{settings.comfy_server_url}）；"
            "请先 `comfy launch --background -- --cpu` 或设 COMFY_SERVER_URL（hardware）"
        )
    from pysci.skills.ai_drawing.tools.comfy_client import ComfyClient

    client = ComfyClient()
    if not client.is_reachable():
        pytest.skip("ComfyUI 端口存活但 /system_stats 不可达（hardware）")
    return client


@pytest.fixture(scope="session")
def ark_key() -> str:
    """火山方舟 API Key（存在性）；未配置时 skip。

    .. note::
       实际密钥存 ComfyUI-Jimeng-API 节点的 ``api_keys.json``；这里只探测 ``.env`` 的
       ``ARK_API_KEY`` 存在性，用作"真实 t2i/i2i 会花钱"用例的门槛。
    """
    if not settings.ark_ready:
        pytest.skip("需要 ARK_API_KEY（真实云端生成会消耗额度；hardware）")
    return settings.ark_api_key or ""
