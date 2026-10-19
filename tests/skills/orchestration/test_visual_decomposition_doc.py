"""复杂视觉任务拆解教义：主册指针 + 分册内容 + leader-only 旁注（tb-5 返工 2 轮）。"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT, PROJECT_ROOT

REF = ORCHESTRATION_ROOT / "skills" / "orchestration" / "references"


def test_main_doc_points_to_visual_volume_and_checklist():
    md = (REF / "plan-decomposition.md").read_text(encoding="utf-8")
    assert "plan-decomposition-visual.md" in md, "主册缺分册指针"
    assert "视觉检验单元" in md and "单叶检验量" in md, "自查表缺视觉条目"


def test_visual_volume_has_split_rules():
    md = (REF / "plan-decomposition-visual.md").read_text(encoding="utf-8")
    for token in ("每幅子图一单", "每页一单", "每幅一单", "缺陷清单", "视觉检验量"):
        assert token in md, token
    assert len(md.encode("utf-8")) <= 8192


def test_leader_only_dispatch_table_points_to_volume():
    md = (PROJECT_ROOT / ".qoder" / "rules" / "leader-only.md").read_text(
        encoding="utf-8"
    )
    assert "plan-decomposition-visual.md" in md
    assert "视觉检验单元" in md
