"""rubric v0.3 六处校准 + v0.4 三处强化 + v0.4.1 两处可执行性 的在场性。

v0.3 来源：I.i3 校准审查报告 §4.1–4.7（首份实审对 v0.2 的可执行性观察，组长转立单）。
v0.4 来源：返工1 成因分析（a1）§环3/§4——「数字对但语义错」全链无环可拦、常识排版纪律
未入判定词表。v0.4.1 来源：b3 终审（reviewer-20261016-003734）v0.4 首战校准——H4① 未直书
「彩色数据区」、遮挡判法未给像素级口径（bbox 8 报 8 假阳）。本文件把「各条各有一处对应
措辞（可逐条 grep 命中）」钉成机械判据，并交叉核对工具面在场与 8KB 预算。
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


def _lines_with(md: str, token: str) -> list[str]:
    return [ln for ln in md.splitlines() if token in ln]


# ---------------------------------------------------------------------------
# 版本与修订史
# ---------------------------------------------------------------------------
def test_version_bumped_and_changelog():
    md = _rubric()
    assert "scientific-plotting）v0.5" in md
    assert "v0.5（2026-10-18）" in md, "缺 v0.5 修订说明（H6/对照表/布局节奏）"
    assert "v0.4.2（2026-10-17）" in md, "v0.4.2 修订史应保留"
    assert "v0.4.1（2026-10-16）" in md, "v0.4.1 修订史应保留"
    assert "v0.4（2026-10-15）" in md, "v0.4 修订史应保留"
    assert "v0.3（2026-10-15）" in md, "v0.3 修订史应保留"


# ---------------------------------------------------------------------------
# v0.4.2（backlog 20261016-005506-reviewer）：判法升级为**可机械复跑**——审查者照做即可，
# 不必自建差分渲染脚本；几何声明可由图面反算，不依赖生产者 notes。
# ---------------------------------------------------------------------------
def test_item_v042_occlusion_method_is_executable_spec():
    """工具与命令节给出**实现范式＋阈值**（照做即可，不必自建脚本）。"""
    md = _rubric()
    assert "ink = |A" in md and "A_text→white| > 30" in md, "缺墨迹蒙版实现范式"
    assert "重叠 > 0 px" in md and "10 px @600 dpi" in md, (
        "缺重叠/间距阈值（硬项/告警分级）"
    )
    assert "差分白字渲染" in md and "求交" in md, "缺指定判法两步"


def test_item_v042_geometry_self_calibration_paradigm():
    """自标定范式：标注几何**可由图面反算**（G2/D3 独立复核，不依赖 notes）。"""
    md = _rubric()
    assert "自标定" in md and "反算" in md, "缺几何自标定范式"
    assert "像素→物理尺寸" in md, "缺换算方向（像素坐标 → 物理尺寸）"


# ---------------------------------------------------------------------------
# v0.5（tb-20261017-212141-3-3 a5）：H6 处置/指令对齐、参考图对照、布局节奏
# ---------------------------------------------------------------------------
def test_item_v05_h6_disposal_and_user_instruction():
    md = _rubric()
    h6 = " ".join(_lines_with(md, "| H6 |"))
    assert "处置分支" in h6 and "指定分支" in h6
    assert "用户显式指令" in h6 and "上报留痕" in h6
    assert "FAIL" in h6


def test_item_v05_g2_reference_table_hook():
    md = _rubric()
    g2 = md.split("### G2", 1)[1].split("## 硬项", 1)[0]
    assert "参考图逐要素对照表" in g2
    assert "照搬/适配/偏离" in g2 and "缺理解痕迹" in g2


def test_item_v05_d5_layout_rhythm_four_rules():
    md = _rubric()
    d5 = " ".join(_lines_with(md, "| D5 "))
    assert "布局节奏/编号锚定四条" in d5
    for token in (
        "标题与上方外缘间距 > 与下方子图间距",
        "图例置于数据图下方、不压主体",
        "子图组与脚注段留清晰间距",
        "整体 bbox（含图例/colorbar/标注）左上角",
    ):
        assert token in d5, f"D5 缺布局准则：{token}"


# ---------------------------------------------------------------------------
# v0.4.1 两处可执行性（b3 终审校准）：H4① 彩色数据区 / 像素级判法 + bbox 警示
# ---------------------------------------------------------------------------
def test_item_v041_h4_covers_colour_data_area():
    """①H4① 直书「彩色数据区」（此前只写「图线」，压场图色块需类推才命中）。"""
    md = _rubric()
    h4 = " ".join(_lines_with(md, "| H4 |"))
    assert "彩色数据区" in h4, "H4① 未直书彩色数据区"
    assert "18px" in h4, "缺 b3 的 18px 回归样例（判法须能复现其结论）"


def test_item_v041_h4_pixel_method_and_bbox_warning():
    """②H4 判法注明像素级口径（真源自足、可照做）与 bbox 假阳性警示。"""
    md = _rubric()
    h4 = " ".join(_lines_with(md, "| H4 |"))
    assert "像素级为真源" in h4
    for token in ("差分白字渲染", "蒙版求交", "墨迹蒙版", "交叠即遮挡"):
        assert token in h4, f"缺像素级判法要素：{token}"
    assert "bbox" in h4 and ("粗筛" in h4 or "假阳" in h4), "缺 bbox 假阳性警示"


def test_item_v041_d5_wordlist_includes_colour_data_area():
    md = _rubric()
    d5 = " ".join(_lines_with(md, "| D5 "))
    assert "彩色数据区" in d5


def test_item_v041_verdict_quantifies_what():
    """顺带：evidence 量值须注明量的是什么（H4 由 bbox 转像素级后尤需）。"""
    md = _rubric()
    assert "量值须注明量的是什么" in md


# ---------------------------------------------------------------------------
# v0.4 三处强化（返工1 M3）：G2 意图对齐核对 / H4 零遮挡铁律四条 / D5 检查点词表
# ---------------------------------------------------------------------------
def test_item_v04_g2_intent_alignment_method():
    """①G2 增「意图对齐核对」判定方法：产物元素↔参考语义↔物理意图三方核对。"""
    md = _rubric()
    assert "意图对齐核对" in md
    for token in ("产物元素", "参考语义", "物理意图", "三方一致"):
        assert token in md, f"G2 意图对齐核对缺三方要素：{token}"
    assert "数字对但语义错" in md, "须能抓出「数字对但语义错」"


def test_item_v04_h4_zero_occlusion_four_rules():
    """②H4 零遮挡铁律四条（硬伤级）逐条可 grep。"""
    md = _rubric()
    h4 = " ".join(_lines_with(md, "| H4 |"))
    assert "零遮挡铁律" in h4 and "硬伤级" in h4
    for token in (
        "文字/刻度值不压主图图线",  # ①
        "图体与标题留间距",  # ②
        "colorbar 标题↔色标、色标↔子图留边距",  # ③
        "任何元素重叠",  # ④
    ):
        assert token in h4, f"H4 零遮挡铁律缺一条：{token}"
    assert "FAIL 无例外" in h4


def test_item_v04_d5_checkpoint_wordlist():
    """③零遮挡铁律词表入 D5 检查点。"""
    md = _rubric()
    d5 = " ".join(_lines_with(md, "| D5 "))
    assert "零遮挡铁律词表" in d5
    for token in ("文字压图线", "刻度压图线", "标题贴身", "色标压子图", "任何元素重叠"):
        assert token in d5, f"D5 词表缺：{token}"


# ---------------------------------------------------------------------------
# v0.3 六处校准（§4.1 / §4.2 / §4.3 / §4.4工具 / §4.6 / §4.7）
# ---------------------------------------------------------------------------
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
    """通用性护栏：v0.3/v0.4 的修订不得把本任务的具体返工指引搬进标准。"""
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
