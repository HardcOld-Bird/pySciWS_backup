"""验收纪律 §4「高频查阅维度集中呈现」的落盘断言（返工 tb-20261015-091655-1 A.a5）。

背景：「总结类交付信息散而不聚」缺陷（用户驳回：硬性时间节点散落各节）的成因是**判据缺失**
——验收纪律无信息架构条款，交付物组织全凭临场自觉。本套用例把补上的判据钉在盘上：

① 新节在场且三要素齐全（原则 / 任务书判据化 / 反推法）——判据化那条必须**点名**
   `success_criteria` 与可机械识别的判据串，否则「显式含一条」只是口号；
② 拆解指导只留**指针**、不留第二份条款（同一能力不留两份：复制一份就等着两处漂移）；
③ 两文件在 harness 预算内（≤8192B，落盘字节口径同 `pysci-dev doctor`）。

**为什么用文档内容断言而不是 grep 判据**：本单不是「禁词」类判据（无全树负向搜索），而是
「正本是否声明了 X」——读已知路径的那份文件即可，无需 backlog 20261014-075230-devops 的
禁词运行期拼接约定。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.devops.tools import budget as budget_mod

#: 真本根（部署副本 gitignored、worktree 里不存在，故断言只打真本）。
_REF = ORCHESTRATION_ROOT / "skills" / "orchestration" / "references"
DISCIPLINE = _REF / "acceptance-discipline.md"
DECOMPOSITION = _REF / "plan-decomposition.md"

#: 判据条目串——任务书 success_criteria 里要显式出现的那一条（唯一真源在本节）。
CRITERION = "高频查阅维度集中呈现"

#: 预算（basic.md §1：references 单文件 ≤8192B，落盘字节）。
BUDGET_B = 8192


def _text(p) -> str:
    return p.read_text(encoding="utf-8")


def test_section_exists_with_heading():
    """① 新节在场：`## 4. 高频查阅维度集中呈现`（章节号 + 判据名）。"""
    text = _text(DISCIPLINE)
    assert f"## 4. {CRITERION}" in text, "验收纪律缺 §4 节标题"


def test_section_has_three_required_elements():
    """① 三要素齐全：原则（使用场景为纲）/ 判据化（success_criteria 显式条目）/ 反推法。"""
    text = _text(DISCIPLINE)
    body = text[text.index(f"## 4. {CRITERION}") :]
    # 原则：以用户使用场景为纲 + 高频维度须有集中章节（原文未提者如实声明）
    for marker in ("使用场景", "集中章节", "反复查询", "原文未提"):
        assert marker in body, f"§4 原则缺要素：{marker}"
    # 判据化：判据串与 success_criteria 同段出现（可机械识别的前提）
    para = next(
        (
            ln
            for ln in body.split("\n\n")
            if CRITERION in ln and "success_criteria" in ln
        ),
        "",
    )
    assert para, f"§4 判据化段须同时点名 {CRITERION} 与 success_criteria"
    # 反推法：设计流程里的那一问
    assert "用户拿到后会反复查什么" in body, "§4 缺反推法（设计流程的那一问）"


def test_decomposition_has_pointer_without_duplicating_clause():
    """② 拆解指导留指针、不复制条款：指向 §4，但原则正文只留在验收纪律一份。"""
    text = _text(DECOMPOSITION)
    assert "acceptance-discipline.md" in text, "拆解指导未指向验收纪律"
    assert CRITERION in text and "§4" in text, "指针未点名验收纪律 §4"
    # 条款本体不得复制过来（这三条是原则/反推法的判别性措辞）
    for dup in ("用户拿到后会反复查什么", "使用场景", "集中章节"):
        assert dup not in text, f"拆解指导复制了验收纪律条款：{dup}"


def test_both_files_within_budget():
    """③ 两文件 ≤8192B（CRLF 等价字节；超限按纪律拆分，不放宽预算）。"""
    for p in (DISCIPLINE, DECOMPOSITION):
        size = budget_mod.disk_size(p)
        assert size <= BUDGET_B, f"{p.name} 超预算：{size}B > {BUDGET_B}B"
