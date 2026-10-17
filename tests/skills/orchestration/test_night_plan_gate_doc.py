"""夜航 × Plan 两环节的调研结论与政策修正（backlog 20261017-112800-devops）。

文档核查：`EnterPlanMode`/`ExitPlanMode` 是内建工具，但 Plan 模式官方定性为「独立工作状态、
不是权限策略」，**未证**其批准走 PreToolUse/PermissionRequest 管道 → 按「不可接管」处理、
不实施；政策文⑥ 的「硬阻塞、不写钩子接管」过时定性已修正。
"""

from __future__ import annotations


def test_policy_six_no_longer_claims_hard_block():
    from pysci.skills.orchestration.tools import night

    p = night.NIGHT_POLICY
    assert "不写钩子接管" not in p, "过时定性残留"
    assert "PermissionRequest" in p, "⑥ 须写明 hook 可代替用户批准"
    assert "EnterPlanMode" in p and "ExitPlanMode" in p
    assert "auto" in p, "须给离机替代路径（auto 档）"


def test_investigation_record_on_disk():
    from pysci.paths import ORCHESTRATION_ROOT

    md = (
        ORCHESTRATION_ROOT
        / "skills"
        / "orchestration"
        / "references"
        / "night-plan-gate.md"
    ).read_text(encoding="utf-8")
    assert "不可接管" in md and "不实施" in md
    assert "plan-mode.md" in md or "独立的工作状态" in md, "缺官方定性依据"
    assert "PermissionRequest" in md and '"behavior":"allow"' in md, "缺钩子输出形状"
    assert "PreToolUse" in md, "缺实测配方"
