"""活体 ComfyUI 冒烟（hardware + slow）：/system_stats、Jimeng 节点 schema 内省、真实出图。

默认 ``-m "not hardware"`` 下**不收集**；即便显式 ``-m hardware``，也经 conftest 的
``comfy_server`` / ``ark_key`` fixture 在服务器不可达 / 无 key 时 ``pytest.skip``。

真实 t2i 会**消耗火山方舟额度**（约 0.2 元/张），故再加一道显式开关：仅当环境变量
``AI_DRAWING_LIVE_GEN=1`` 时才实跑，避免任何意外花费。
"""

from __future__ import annotations

import os

import pytest

from pysci.skills.ai_drawing.tools import workflows as wf

pytestmark = [pytest.mark.hardware, pytest.mark.slow]


def test_live_system_stats(comfy_server):
    stats = comfy_server.system_stats()
    assert isinstance(stats, dict)


def test_live_jimeng_client_node_present(comfy_server):
    """/object_info 里应有 JimengAPIClient（未装 ComfyUI-Jimeng-API 则 skip）。"""
    info = comfy_server.object_info("JimengAPIClient")
    if "JimengAPIClient" not in info:
        pytest.skip("服务器未安装 ComfyUI-Jimeng-API（无 JimengAPIClient 节点）")
    names = comfy_server.node_input_names("JimengAPIClient")
    assert "key_name" in names


def test_live_seedream4_schema_matches_registry(comfy_server):
    """核实 JimengSeedream4 的输入名与本仓库登记的契约一致（class_type 禁止猜的护栏）。"""
    info = comfy_server.object_info("JimengSeedream4")
    if "JimengSeedream4" not in info:
        pytest.skip("服务器无 JimengSeedream4 节点")
    names = set(comfy_server.node_input_names("JimengSeedream4"))
    # workflows.seedream_gen_node 依赖的必需输入名
    for expected in ("client", "model_version", "prompt", "size", "seed"):
        assert expected in names, f"JimengSeedream4 缺输入 {expected}（schema 漂移？）"


def test_live_txt2img_real_generation(comfy_server, ark_key, tmp_path):
    """真实文生图一张（耗额度）——仅在 AI_DRAWING_LIVE_GEN=1 时实跑。"""
    if os.environ.get("AI_DRAWING_LIVE_GEN") != "1":
        pytest.skip("设 AI_DRAWING_LIVE_GEN=1 才实跑真实出图（会消耗方舟额度）")
    graph = wf.txt2img_seedream(
        "a minimalist scientific journal cover, abstract wave interference pattern",
        seed=12345, size="2K", n=1,
    )
    images = comfy_server.run_to_images(graph, timeout=300.0, progress=False)
    assert images and images[0]["data"]
    from pysci.skills.ai_drawing.tools.comfy_client import save_images

    written = save_images(images, tmp_path / "live", stem="live_t2i")
    assert written and written[0].stat().st_size > 0
