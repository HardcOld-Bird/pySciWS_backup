"""ink-gap 统一测距：基准 32px + 三种错法反例（硬切/单标签/1D 并块）。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pysci.paths import ORCHESTRATION_ROOT, PROJECT_ROOT, research_asset_dir
from pysci.skills.scientific_plotting.tools import figures, ink_gap

RUBRIC = ORCHESTRATION_ROOT / "pods" / "reviewer" / "rubrics" / "scientific-plotting.md"
SKILL = ORCHESTRATION_ROOT / "skills" / "scientific-plotting" / "SKILL.md"


def _gain_ep_root() -> Path:
    """worktree 不含 gitignored 成品 → 只读回退到主树（git common dir 的父级）。"""
    root = research_asset_dir("gain_ep")
    if (
        root / "article" / "figures" / "beam_scatter" / "beam_scatter_combined_r5.png"
    ).exists():
        return root
    common = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).resolve().parent / "data" / "research" / "1_gain_ep"


BASELINE = (
    _gain_ep_root()
    / "article"
    / "figures"
    / "beam_scatter"
    / "beam_scatter_combined_r5.png"
)
ROI_A = (3700, 1200, 3920, 2300)  # cb3 刻度标签列
ROI_B = (3925, 1200, 4015, 2300)  # 轴题 Re p_t (Pa)


def test_baseline_axis_title_to_cb3_label_is_32px():
    """基准图：轴题块 ↔ cb3 标签块 = 32 px ±2（复审独立实测）。"""
    img = ink_gap.load_image(BASELINE)
    rep = ink_gap.gap_between_rois(img, ROI_A, ROI_B, mode="region", dpi=600)
    assert abs(rep.distance_px - 32.0) <= 2.0, rep
    assert rep.distance_mm is not None and abs(rep.distance_mm - 1.35) < 0.15


def test_counterexample_hard_cut_yields_1px():
    """错法①：把标签组件按 x=3897 硬切 → 同一字形两半相距 1 px 假象。"""
    img = ink_gap.load_image(BASELINE)
    regions = ink_gap.components(ink_gap.ink_mask(img))
    target = next(r for r in regions if r.bbox_px[0] == 3884 and r.bbox_px[2] == 3908)
    left = target.pixels_xy[target.pixels_xy[:, 0] < 3897]
    right = target.pixels_xy[target.pixels_xy[:, 0] >= 3897]
    d = float(cKDTree(left).query(right, k=1)[0].min())
    assert d == pytest.approx(1.0, abs=1.0)
    assert d < 5 and d != pytest.approx(32, abs=2)


def test_counterexample_single_label_pixel_yields_157px():
    """错法②：只拿顶部单个标签做字形距（pixel）→ 157 px 虚高。"""
    img = ink_gap.load_image(BASELINE)
    rep = ink_gap.gap_between_rois(img, ROI_A, ROI_B, mode="pixel", dpi=600)
    assert abs(rep.distance_px - 157.0) <= 5.0, rep
    assert rep.distance_px > 100


def _projection_1d_gap(mask: np.ndarray, threshold: int = 0) -> int:
    """错法③：x 投影 run 合并后取 run 间空隙（1D，忽略 y）。"""
    proj = np.asarray(mask, dtype=bool).any(axis=0)
    runs: list[list[int]] = []
    for x, on in enumerate(proj):
        if on and (not runs or runs[-1][1] != x - 1):
            runs.append([x, x])
        elif on:
            runs[-1][1] = x
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= threshold:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return min(b[0] - a[1] for a, b in zip(merged, merged[1:], strict=False))


def test_counterexample_1d_merge_reports_15px_on_separated_blocks():
    """错法③：1D 并块把不同 y 的文字行并排 → 15 px，2D 真距 ≥50 px。"""
    mask = np.zeros((140, 200), dtype=bool)
    mask[0:11, 0:51] = True  # 标签上行
    mask[100:111, 0:86] = True  # 标签下行（右侧突出）
    mask[0:11, 100:121] = True  # 轴题（与上行同 y）
    assert _projection_1d_gap(mask, threshold=0) == 15
    regions = ink_gap.components(mask)
    top = [r for r in regions if r.bbox_px[1] <= 10 and r.bbox_px[0] < 60]
    title = [r for r in regions if r.bbox_px[0] >= 100]
    lower = [r for r in regions if 90 <= r.bbox_px[1] <= 110]
    d = ink_gap.pixel_gap_px(top + lower, title)
    assert d >= 45, d
    assert abs(d - 15) > 20


def test_pairwise_gaps_table(tmp_path):
    mask = np.zeros((40, 120), dtype=bool)
    mask[5:15, 5:25] = True
    mask[5:15, 35:55] = True
    rows = ink_gap.pairwise_gaps(mask, dpi=600)
    assert rows and rows[0].distance_px == pytest.approx(10.0, abs=1.0)
    assert rows[0].distance_mm is not None


def test_cli_ink_gap_region_and_json(capsys):
    rc = figures.main(
        [
            "ink-gap",
            str(BASELINE),
            "--roi-a",
            *map(str, ROI_A),
            "--roi-b",
            *map(str, ROI_B),
            "--mode",
            "region",
            "--dpi",
            "600",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "31.0 px" in out or "32.0 px" in out
    assert "mm" in out

    rc = figures.main(
        [
            "ink-gap",
            str(BASELINE),
            "--roi-a",
            *map(str, ROI_A),
            "--roi-b",
            *map(str, ROI_B),
            "--mode",
            "region",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["readings"][0]["distance_px"] == pytest.approx(32.0, abs=2.0)


def test_rubric_and_skill_point_to_same_tool():
    rubric = RUBRIC.read_text(encoding="utf-8")
    skill = SKILL.read_text(encoding="utf-8")
    assert "ink-gap" in rubric
    assert "pysci-figures ink-gap" in skill
    assert RUBRIC.stat().st_size <= 8192 and SKILL.stat().st_size <= 8192
