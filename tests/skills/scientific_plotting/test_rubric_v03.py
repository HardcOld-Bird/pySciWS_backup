"""rubric v0.3 的六处校准修订 + `figures inspect` 的在场性（backlog 20261015-204912）。

来源：I.i3 校准审查报告 §4.1–4.7（首份实审对 v0.2 的可执行性观察，组长转立单）。本文件把
「六条各有一处对应措辞（可逐条 grep 命中）」钉成机械判据，并顺带交叉核对工具面在场。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from pysci.paths import ORCHESTRATION_ROOT

RUBRIC = ORCHESTRATION_ROOT / "pods" / "reviewer" / "rubrics" / "scientific-plotting.md"


def _rubric() -> str:
    return RUBRIC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 六处校准（§4.1 / §4.2 / §4.3 / §4.4工具 / §4.6 / §4.7）
# ---------------------------------------------------------------------------
def test_version_bumped_and_changelog():
    md = _rubric()
    assert "scientific-plotting）v0.3" in md
    assert "v0.3（2026-10-15）" in md, "缺 v0.3 修订说明"


def test_item_41_physical_expectation_input():
    """§4.1：任务书可随附最小物理预期陈述；H1/D1 以之为判据。"""
    md = _rubric()
    assert md.count("最小物理预期陈述") >= 2, "H1 与 D1 都应点明该输入机制"
    assert "135°" in md, "应给一个可照抄的示例措辞"


def test_item_42_script_as_third_party_evidence():
    """§4.2：H2 三方对照显式加入生成脚本的绘图表达式，且脚本仅作佐证。"""
    md = _rubric()
    assert "生成脚本的绘图表达式" in md
    assert "直接证据" in md and "佐证" in md, "须写明硬判定优先直接证据、脚本仅佐证"


def test_item_43_external_renderer_ui_elements():
    """§4.3：H3/D4 增列外部渲染器品牌/UI 元素是否已清除。"""
    md = _rubric()
    assert md.count("外部渲染器品牌/UI 元素") >= 2, "H3 与 D4 都应增列"


def test_item_44_inspect_command_documented():
    """§4.4：工具面补 `pysci-figures inspect`（不依赖管线的图像体检）。"""
    md = _rubric()
    assert "pysci-figures inspect" in md
    assert "边缘水印·logo 检测" in md


def test_item_46_pasted_raster_traceability():
    """§4.6：粘贴型栅格子图的口径 = 图像级可追溯（不可数值复算）。"""
    md = _rubric()
    assert "图像级可追溯（不可数值复算）" in md
    assert "粘贴型栅格子图" in md


def test_item_47_colorblind_executable_method():
    """§4.7：色觉安全给可执行判定法（deuteranopia 模拟 + 区分度阈值）。"""
    md = _rubric()
    assert "simulate_cvd" in md and "0.12" in md


# ---------------------------------------------------------------------------
# 交叉核对：文档所述的工具确实在场（防「文档写了命令但没实现」）
# ---------------------------------------------------------------------------
def test_inspect_command_is_registered_in_figures_cli(capsys):
    from pysci.skills.scientific_plotting.tools import figures

    with pytest.raises(SystemExit):
        figures.main(["-h"])
    out = capsys.readouterr().out
    assert re.search(r"^\s+inspect\b", out, re.M), "figures CLI 未注册 inspect 子命令"


def test_audit_cvd_threshold_matches_documented_value():
    """文档写的 0.12 就是 `audit._check_colorblind` 的判据阈值（口径同源）。"""
    import inspect as _inspect

    from pysci.skills.scientific_plotting.tools import audit

    sig = _inspect.signature(audit._check_colorblind)
    assert sig.parameters["thresh"].default == 0.12


def test_rubric_within_budget():
    assert RUBRIC.stat().st_size <= 8192, "references/量表层 8KB 预算被破"


def test_rubric_still_generic_no_task_specific_recipe():
    """通用性护栏：v0.3 的修订不得把本任务的具体返工指引搬进标准。"""
    md = _rubric()
    for token in ("正方形", "菱形", "管槽", "行式", "列式"):
        assert token not in md, f"量表混入了任务特定措辞：{token}"


def test_inspect_detects_brand_like_mark_on_synthetic(tmp_path):
    """末环：文档承诺的体检能力确实能报出「蓝黄成对」标记。"""
    from PIL import Image

    from pysci.skills.scientific_plotting.tools import imgcheck

    a = np.full((300, 400, 3), 255, dtype=np.uint8)
    a[4:17, 382:398] = (100, 144, 200)
    a[4:17, 368:382] = (240, 240, 96)
    p = Path(tmp_path) / "l.png"
    Image.fromarray(a).save(p)
    health, rc = imgcheck.inspect_image(p)
    assert rc == 1 and any(m["brand_like"] for m in health["edge_marks"])
