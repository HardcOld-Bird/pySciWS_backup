"""任务书拆解指导的文档面（backlog 20261016-040451-writing）。

任务书给「右缘 2000→≈1900」这类**固定像素提示值**时，照搬会与既定目标（内移到行界、消除
半截字）冲突——实测两个提示值都会竖切正文末字（「将/并/以及」/「流」）。指导面须写明：落点
需实测校准的改动给**目标 + 复现锚点**，数值只标「仅量级参考」，避免执行端在「照做 vs 达标」
间二选一。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT

GUIDE = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "taskbook-template.md"
)


def test_decomposition_guide_requires_calibration_anchor():
    md = GUIDE.read_text(encoding="utf-8")
    assert "复现锚点" in md, "缺「给目标 + 复现锚点」纪律"
    assert "仅量级参考" in md, "缺「数值须标仅量级参考」要求"
    # 量级参考与目标须同段出现，别只留一个裸像素数
    assert "须同段出现" in md
