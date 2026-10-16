"""`command-surface.md` 的**顶层命令覆盖**与建议单口径（backlog 20261015-205905）。

现场：`requeue`/`resolve`/`watchdog`/`devops`/`refresh-models`/`twin` 六个**真实存在**
的顶层命令在任何文档面都零命中——只能靠别的报错偶然看到 argparse 的 usage 行。本文件把
「顶层命令全部有档」钉成机械判据（同类缺口此后再现即红），并钉建议单的两条口径。
"""

from __future__ import annotations

import re

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import orch

CMDSURF = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "command-surface.md"
)

#: 205905 点名的六个（本轮补齐）。
NAMED = (
    "requeue",
    "resolve",
    "watchdog",
    "devops",
    "refresh-models",
    "twin",
)

_USAGE_CHOICES = re.compile(r"\{(.*?)\}", re.S)


def _doc() -> str:
    return CMDSURF.read_text(encoding="utf-8")


def _top_level_commands(capsys) -> list[str]:
    """从 `orch -h` 的 usage 里取顶层命令集合（argparse 会把花括号组**折行**，故跨行匹配）。"""
    with pytest.raises(SystemExit):
        orch.main(["-h"])
    out = capsys.readouterr().out
    m = _USAGE_CHOICES.search(out)
    assert m, "未能从 orch -h 的 usage 解析出顶层命令集合"
    cmds = [c.strip() for c in m.group(1).replace("\n", " ").split(",")]
    cmds = [c for c in cmds if re.fullmatch(r"[a-z][a-z0-9-]*", c)]
    assert len(cmds) >= 10, f"顶层命令集合看着不对：{cmds}"
    return cmds


def _documented(doc: str, cmd: str) -> bool:
    """命令在文档里出现——**含 `a|b` 合并写法**（如 `inspector enable|disable` 即覆盖 `inspector disable`）。"""
    return cmd in doc or f"|{cmd}" in doc or f"{cmd}|" in doc


def test_every_top_level_command_is_documented(capsys):
    """机械护栏：`pysci-orch -h` 的每个顶层命令都要在 command-surface.md 里有档。"""
    doc = _doc()
    missing = [c for c in _top_level_commands(capsys) if not _documented(doc, c)]
    assert not missing, f"顶层命令缺档：{missing}（command-surface.md）"


@pytest.mark.parametrize("cmd", NAMED)
def test_named_commands_present(cmd):
    assert _documented(_doc(), cmd), f"{cmd} 未在 command-surface.md 出现"


def test_named_commands_have_semantics_not_just_the_word():
    """不只是「出现过」：各自带一句语义/作用域。"""
    doc = _doc()
    assert "needs_leader → pending" in doc  # requeue
    assert "确认已达标的 needs_leader 条目 → done" in doc  # resolve
    assert "广播纪元停全部" in doc  # watchdog
    assert "协作式停止" in doc  # devops stop
    assert "模型映射" in doc  # refresh-models
    assert "复核 twin" in doc  # twin review


def test_requeue_resolve_scope_is_documented():
    """两者作用域写清（backlog 条目 + needs_leader），免被误当建议单销账口。"""
    doc = _doc()
    assert "**不收建议单 id**" in doc


def test_duplicate_suggestion_policy_documented():
    """重复建议单的处置口径 + 历史面审计读法（criteria ②）。"""
    doc = _doc()
    assert "重复建议单" in doc
    assert "合并至哪单" in doc
    assert "以 `--override` 正文为准" in doc


def test_suggestion_member_column_semantics_documented():
    """组长代补单在 `suggestions` 列表的 `[member]` 显示口径（criteria ③）。"""
    doc = _doc()
    assert "`[member]` 列" in doc
    assert "代补单" in doc and "目标" in doc


# ---------------------------------------------------------------------------
# 立单样例与源文件命名（backlog 20261016-215944 / -215946 判据②）
# ---------------------------------------------------------------------------
def test_suggestion_minimal_example_documented():
    """立单块给出**可整体复制的最小样例**＋判据标记的接受形态（含 markdown 标题形态）。

    缺陷（20261016-203600-devops 首写实锤）：文档只说「`success_criteria:` 段」，读者自然写
    `## success_criteria` 标题段，而旧解析器不认 → 首写必返工一轮。
    """
    doc = _doc()
    assert "最小样例" in doc
    assert "`success_criteria:`" in doc
    assert "接受形态" in doc and "标题形态" in doc, "须写明标题形态已被支持"


def test_plan_new_source_file_naming_documented():
    """`plan new` 行补源文件命名提示（勿与目标同名——同名行为未定义）。"""
    doc = _doc()
    assert "勿与目标同名" in doc


# ---------------------------------------------------------------------------
# `plan adhoc` 的阻塞语义（backlog 20261015-211200）
# ---------------------------------------------------------------------------
SKILL = ORCHESTRATION_ROOT / "skills" / "orchestration" / "SKILL.md"


def test_plan_adhoc_blocking_semantics_in_command_surface():
    """命令面明示：`plan adhoc` **同步阻塞到完成**（与 `plan advance` 入队即返回**相反**）。"""
    doc = _doc()
    assert "同步阻塞到完成" in doc
    assert "入队即返回" in doc
    assert "run_in_background" in doc  # 后台纪律
    # 对照须点名 advance（否则读者仍可能按 advance 的直觉放前台）
    adhoc_at = doc.index("同步阻塞到完成")
    tail = doc[adhoc_at : adhoc_at + 400]
    assert "plan advance" in tail, "adhoc 的阻塞语义旁未对照 plan advance"


def test_skill_iron_rule_3_lists_plan_adhoc():
    """SKILL 铁律 3「长任务必须后台运行」的命令枚举含 `plan adhoc`，并写明与 advance 的差异。"""
    md = SKILL.read_text(encoding="utf-8")
    start = md.index("**长任务必须后台运行**")
    block = md[start : md.index("4. **", start)]
    assert "plan adhoc" in block
    assert "run_in_background" in block
    assert "同步阻塞" in block and "入队即返回" in block


def test_watched_docs_within_budget():
    """本册与 SKILL 都在 8KB 真本预算内（落盘字节口径）。"""
    for p in (CMDSURF, SKILL):
        assert p.stat().st_size <= 8192, f"{p.name} 超预算：{p.stat().st_size}B"


# ---------------------------------------------------------------------------
# 命令面集中整备（backlog 20261016-221800）：子树化 / 内部命令隐藏 / 旧名拒收
# ---------------------------------------------------------------------------
#: 旧顶层名（**运行期拼接**：防常量折叠把字面量折进 .pyc 使含二进制的 grep 误命中）。
_OLD_NAMES = tuple(
    a + "-" + b
    for a, b in (
        ("consult", "twin"),
        ("twin", "review"),
        ("bench", "add"),
        ("bench", "up"),
        ("bench", "queue"),
        ("bench", "seen"),
        ("bench", "stop"),
        ("inspector", "up"),
        ("inspector", "status"),
        ("inspector", "patrol"),
        ("inspector", "assign"),
        ("inspector", "on"),
        ("inspector", "off"),
        ("drain", "stop"),
    )
)

#: 旧「命令 x 前缀」措辞（说话人对称化前）——运行期拼接同理。
_OLD_PHRASES = ("orch " + "consult", "orch " + "taste", "orch " + "eval")

#: 新子树与其子命令（映射表〔默认〕行的机械面）。
_SUBGROUPS = {
    "deputy": ["consult"],
    "twin": ["consult", "taste", "review"],
    "bench": ["add", "start", "queue", "seen", "stop", "eval", "baseline"],
    "inspector": ["start", "status", "patrol", "assign", "enable", "disable"],
    "devops": ["stop"],
}

_INTERNAL_CMDS = (
    "_watchdog-loop",
    "_queue-worker",
    "_drain-devops",
    "_bench-worker",
    "_inspect-worker",
)


def _sub_choices(capsys, argv: list[str]) -> list[str]:
    from pysci.skills.orchestration.tools import orch as _orch

    capsys.readouterr()
    with pytest.raises(SystemExit) as ei:
        _orch.main(argv)
    assert ei.value.code == 0
    out = capsys.readouterr().out
    m = re.search(r"\{([^}]*)\}", out, re.S)
    assert m, out
    return [c.strip() for c in m.group(1).replace("\n", " ").split(",")]


@pytest.mark.parametrize("old", _OLD_NAMES)
def test_old_top_level_names_rejected(capsys, old):
    """旧名**彻底移除**（用户明示无需向后兼容）：argparse 直接拒收。"""
    from pysci.skills.orchestration.tools import orch as _orch

    with pytest.raises(SystemExit) as ei:
        _orch.main([old])
    assert ei.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("group", sorted(_SUBGROUPS))
def test_subtree_subcommands_available(capsys, group):
    """映射表〔默认〕行：新命令可用（子面 `-h` 列出全部子命令，不多不少）。"""
    assert _sub_choices(capsys, [group, "-h"]) == _SUBGROUPS[group]


def test_internal_commands_hidden_from_usage(capsys):
    """5 个 `_` 内部命令不进 `orch -h`（功能保留，钩子/派生照常调用）。

    注意：argparse 对子命令不认 ``help=SUPPRESS``（会打出字面量 ``==SUPPRESS==``）——
    故这里同时钉「字面量不漏」与「名字不出现」两面。
    """
    from pysci.skills.orchestration.tools import orch as _orch

    capsys.readouterr()
    with pytest.raises(SystemExit):
        _orch.main(["-h"])
    out = capsys.readouterr().out
    assert "==SUPPRESS==" not in out, "argparse 的 SUPPRESS 字面量漏进帮助"
    for name in _INTERNAL_CMDS:
        assert name not in out, f"内部命令漏进 usage：{name}"


def test_internal_commands_still_invokable(capsys):
    """隐藏 ≠ 移除：内部命令仍可被钩子/派生进程调用（此处只验 argparse 认它）。"""
    from pysci.skills.orchestration.tools import orch as _orch

    with pytest.raises(SystemExit) as ei:
        _orch.main(["_queue-worker", "-h"])
    assert ei.value.code == 0
    assert "worker" in capsys.readouterr().out


def test_inspector_start_and_enable_are_distinguishable(capsys):
    """`start`（拉起 worker）与 `enable`（全局开关）help 一眼可辨（用户点名 up/on 两字母之差）。"""
    from pysci.skills.orchestration.tools import orch as _orch

    capsys.readouterr()
    with pytest.raises(SystemExit):
        _orch.main(["inspector", "-h"])
    out = capsys.readouterr().out
    rows = {
        parts[0]: ln
        for ln in out.splitlines()
        if (parts := ln.strip().split()) and parts[0] in ("start", "enable", "disable")
    }
    assert "拉起" in rows["start"] and "worker" in rows["start"], rows
    assert "开关" in rows["enable"], rows
    assert "开关" in rows["disable"], rows


def test_twin_family_is_symmetric(capsys):
    """`deputy consult` / `twin consult` 同为咨询动词，参数形态对称（各自 case/question）。"""
    from pysci.skills.orchestration.tools import orch as _orch

    capsys.readouterr()
    with pytest.raises(SystemExit):
        _orch.main(["deputy", "consult", "-h"])
    d = capsys.readouterr().out
    with pytest.raises(SystemExit):
        _orch.main(["twin", "consult", "-h"])
    t = capsys.readouterr().out
    assert "question" in d and "consult" in d, d
    assert "case_file" in t and "案卷" in t, t


def test_live_docs_have_no_old_command_names():
    """活文档面（真本技能全文 + README + 现行设计引用）零旧名/旧措辞残留。

    历史留档（`references/legacy-readme-v2.2.md`／`decisions.md`／`history-and-risks.md`）
    如实记录当时状态，**不在**扫描面内（其更名不回溯）。
    """
    faces = sorted((ORCHESTRATION_ROOT / "skills").rglob("*.md")) + [
        ORCHESTRATION_ROOT / "README.md",
        ORCHESTRATION_ROOT / "references" / "plan-and-routing.md",
        ORCHESTRATION_ROOT / "references" / "wake-and-events.md",
        ORCHESTRATION_ROOT / "references" / "quality-and-improvement.md",
    ]
    bad: dict[str, list[str]] = {}
    for p in faces:
        txt = p.read_text(encoding="utf-8")
        hit = [t for t in (*_OLD_NAMES, *_OLD_PHRASES) if t in txt]
        if hit:
            bad[p.relative_to(ORCHESTRATION_ROOT).as_posix()] = hit
    assert not bad, f"活文档面旧名残留：{bad}"
