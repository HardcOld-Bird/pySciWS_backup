"""inspector 巡检补「信息架构」判据（返工监察改进 c4；jury B:3 全票批准）。

背景：``findings`` 历史上**零**信息组织类捕获——监察面存在盲区，同类缺陷只能靠用户驳回
暴露（案卷 tb-20261015-091655-1-discoverability）。c2 方案定了**不新造规则**：菜单补一项、
判据引既有 §4。

本文件钉菜单面两件事（行为面「抽查→立案进既有分流」由
``test_inspect_engine.py::test_info_arch_patrol_finding_takes_the_existing_funnel`` 覆盖——
那里有现成的 ``iso_engine`` mock 范式，不在本文件再造一份夹具）：

① 菜单项在场且**只引用**验收纪律 §4（判据唯一维护点），不复制条款文字；
② 抽中该项后的任务书自包含（组员看不到引擎会话上下文）——含 §4 的真本路径。
"""

from __future__ import annotations

import random

from pysci.skills.orchestration.tools import inspectd

#: 新菜单项 id（引擎侧固定，组员不自选）。
ENTRY_ID = "info-architecture"
LABEL = "交付信息架构抽查"

#: 判据唯一维护点：验收纪律 §4 的节标题与真本路径（菜单项只许引用这两样）。
SECTION = "高频查阅维度集中呈现"
REF_PATH = "orchestration/skills/orchestration/references/acceptance-discipline.md"

#: §4 的**判别性条款措辞**——菜单项若把这些抄一遍，"判据唯一维护点"就成了幻觉。
#: 注：本判据是**对单条 brief 的缺席检查**（不扫全树），故字面量无需运行期拼接——
#: backlog 20261014-075230-devops 的禁词拼接约定管的是全树负向 grep。
CLAUSE_PHRASES = (
    "用户拿到后会反复查什么",
    "使用场景",
    "集中章节",
    "原文未提",
    "索引视图",
)


# ---------------------------------------------------------------------------
# ① 菜单项在场 + 只引用不复制
# ---------------------------------------------------------------------------
def test_menu_entry_present_with_label_and_brief():
    entry = inspectd.PATROL_BY_ID[ENTRY_ID]
    assert entry["label"] == LABEL
    assert entry["brief"].strip(), "缺 brief ＝ 组员拿到空任务书"
    assert not entry.get("self_cover"), "该项不是 §0.2 自覆盖项"
    assert not entry.get("rubric"), "该项不吃设计审查三维槽位"


def test_menu_entry_references_section_without_duplicating_clause():
    """判据唯一维护点：引用 §4（节标题 + 真本路径），且**不含** §4 的判别性措辞。"""
    brief = inspectd.PATROL_BY_ID[ENTRY_ID]["brief"]
    assert SECTION in brief, "须点名 §4 节标题（否则抽中后不知按哪条判据）"
    assert "§4" in brief and REF_PATH in brief, "须给出判据真本路径"
    for phrase in CLAUSE_PHRASES:
        assert phrase not in brief, f"菜单项复制了 §4 条款文字：{phrase!r}"


def test_patrol_brief_for_entry_is_self_contained():
    """抽中后的任务书自包含：类别/要做的事/§4 路径/收尾纪律（组员看不到引擎上下文）。"""
    txt = inspectd.patrol_brief(inspectd.PATROL_BY_ID[ENTRY_ID])
    assert ENTRY_ID in txt and LABEL in txt
    assert SECTION in txt and REF_PATH in txt
    assert "pysci-inspect finding add" in txt, "收尾纪律（§5.2）须照旧带上"


# ---------------------------------------------------------------------------
# ② 随机抽取池：该项由引擎纳入（组员不自选）
# ---------------------------------------------------------------------------
def test_entry_is_in_the_random_pick_pool():
    rng = random.Random(11)
    seen = {inspectd.pick_patrol(rng=rng)["id"] for _ in range(300)}
    assert ENTRY_ID in seen
