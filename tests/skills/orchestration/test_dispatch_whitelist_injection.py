"""任务书「白名单（写权限）」段**声明即注入** pod-guard（backlog 20261015-132939-fix3）。

实案 b2：任务书明明声明了白名单目录，guard 却照拦成 ``whitelist_conflict``——**声明面**
（lint 的 ``whitelist`` 规则读它、只告警）与**注入面**（``PYSCI_TASK_DIRS`` 只来自
``--dirs``/plan step ``dirs``）脱节。本单把两面接成单点：派发时用 lint 的**同一**解析器
（:func:`dispatch._whitelist_entries`）读出声明，并入 ``PYSCI_TASK_DIRS``。

末两例跑**真 pod-guard**（``node`` + stdin JSON + env，同 ``test_pod_guard_blocks`` 的
方法学）：声明目录放行、声明外目录照旧拦——证明注入既生效又不过宽。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import dispatch

from .test_pod_guard_blocks import NODE_REQUIRED, run_guard

TB = """# 任务书 demo

## 要求

改 `src/pysci/skills/orchestration/tools/` 下的东西，测试放 `tests/`。

## 白名单（写权限）

- src/pysci/skills/orchestration/tools/
- tests/

（白名单外功能必需才动、最小改动、显式声明。）
"""


def _taskbook(tmp_path: Path) -> Path:
    p = tmp_path / "task-demo.md"
    p.write_text(TB, encoding="utf-8")
    return p


def test_parses_declared_dirs_from_the_taskbook(tmp_path):
    assert dispatch.taskbook_whitelist_dirs(_taskbook(tmp_path)) == [
        "src/pysci/skills/orchestration/tools",
        "tests",
    ]


def test_missing_or_absent_taskbook_fails_soft(tmp_path):
    assert dispatch.taskbook_whitelist_dirs(tmp_path / "nope.md") == []
    assert dispatch.taskbook_whitelist_dirs(None) == []


def test_compose_keeps_explicit_dirs_first_and_dedupes(tmp_path):
    out = dispatch.dirs_with_taskbook_whitelist(
        ["tests", "data/x"], _taskbook(tmp_path)
    )
    assert out == ["tests", "data/x", "src/pysci/skills/orchestration/tools"]


@NODE_REQUIRED
def test_declared_dir_is_allowed_by_the_real_guard(tmp_path):
    """b2 同款场景：只有任务书声明、没有 ``--dirs`` → 注入后真 guard **放行**。"""
    dirs = dispatch.dirs_with_taskbook_whitelist([], _taskbook(tmp_path))
    pod = tmp_path / "pod"
    pod.mkdir()
    target = PROJECT_ROOT / "tests" / "_probe_declared.md"

    rc, err = run_guard(
        file_path=str(target),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
        task_dirs=dirs,
    )

    assert rc == 0, err


@NODE_REQUIRED
def test_undeclared_dir_is_still_blocked(tmp_path):
    """注入不过宽：白名单没声明的路径照旧拦（否则等于把 guard 关掉）。"""
    dirs = dispatch.dirs_with_taskbook_whitelist([], _taskbook(tmp_path))
    pod = tmp_path / "pod"
    pod.mkdir()
    target = PROJECT_ROOT / "src" / "pysci" / "paths.py"

    rc, err = run_guard(
        file_path=str(target),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
        task_dirs=dirs,
    )

    assert rc == 2 and "白名单目录之外" in err


def test_env_carries_the_injected_dirs(tmp_path):
    """声明面 → env 面：``build_env`` 把同一列表写进 ``PYSCI_TASK_DIRS``（guard 的唯一输入）。"""
    from pysci.skills.orchestration.tools import registry, runner

    dirs = dispatch.dirs_with_taskbook_whitelist(["data/x"], _taskbook(tmp_path))
    env = runner.build_env(
        registry.Registry.load().member("devops"), dirs, deployed_skills=[]
    )

    assert env["PYSCI_TASK_DIRS"].split(";") == dirs


def test_injection_point_is_wired_into_dispatch():
    """接线在场：``do_dispatch`` 组装 ``all_dirs`` 时走的是这条通路（改回旧式即红）。"""
    src = Path(dispatch.__file__).read_text(encoding="utf-8")
    assert "dirs_with_taskbook_whitelist(" in src
    assert "all_dirs = list(dirs or []) + extra_dirs\n" not in src, "旧式裸拼接已退役"


# ---------------------------------------------------------------------------
# 中文路径条目 + 括号注记 + 「声明了却解析不出」告警（backlog 20261015-130838-writing）
# ---------------------------------------------------------------------------
TB_CJK = """# 任务书 demo

## 白名单（写权限）

- data/大任务/
- tests/

（白名单外功能必需才动、最小改动、显式声明。）
"""


def test_cjk_whitelist_entry_is_kept_whole(tmp_path):
    """含中文的目录名**整条保留**——原实现截成上级 `data`，**比声明更宽**（等于放行整棵
    data/），既是漏读也是越权（实案：writing 单声明 `data/大任务/…`）。"""
    p = tmp_path / "tb-cjk.md"
    p.write_text(TB_CJK, encoding="utf-8")

    assert dispatch.taskbook_whitelist_dirs(p) == ["data/大任务", "tests"]


def test_bracket_annotation_is_not_glued_onto_the_path(tmp_path):
    """`- data/taskbooks/**（新建）` → 路径不带注记（取 token 时括号即终止符）。"""
    p = tmp_path / "tb-anno.md"
    p.write_text(
        "## 白名单（写权限）\n\n- data/taskbooks/**（新建）\n- tests/\n",
        encoding="utf-8",
    )

    assert dispatch.taskbook_whitelist_dirs(p) == ["data/taskbooks", "tests"]


@NODE_REQUIRED
def test_declared_cjk_dir_is_allowed_but_its_parent_is_not(tmp_path):
    """端到端真 guard：声明的**中文目录**放行；其**父目录**照旧拦（注入既生效也不过宽）。"""
    p = tmp_path / "tb-cjk.md"
    p.write_text(TB_CJK, encoding="utf-8")
    dirs = dispatch.dirs_with_taskbook_whitelist([], p)
    pod = tmp_path / "pod"
    pod.mkdir()

    inside = PROJECT_ROOT / "data" / "大任务" / "1_文档阅读" / "x.md"
    rc_in, err_in = run_guard(
        file_path=str(inside),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
        task_dirs=dirs,
    )
    outside = PROJECT_ROOT / "data" / "others" / "y.md"
    rc_out, _err_out = run_guard(
        file_path=str(outside),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
        task_dirs=dirs,
    )

    assert rc_in == 0, err_in
    assert rc_out == 2, "父目录不得被顺带放行（原截断版会放行整棵 data/）"


def test_warns_when_whitelist_section_yields_nothing(tmp_path, capsys):
    """判据：「声明了白名单段但 dirs 为空」在派发口**可见**（不再静默拦死组员）。"""
    p = tmp_path / "tb-none.md"
    p.write_text("## 白名单（写权限）\n\n（无条目）\n", encoding="utf-8")

    assert dispatch.dirs_with_taskbook_whitelist([], p) == []

    out = capsys.readouterr().out
    assert "解析不出任何目录" in out and "whitelist_conflict" in out, out


# ---------------------------------------------------------------------------
# 声明面「写了」却解析不出目录 → **可见失败**（backlog 20261015-222249）
# ---------------------------------------------------------------------------
#: 白名单段存在，但条目**不是路径样**（无 `/`、无已知后缀）→ 解析得空。
TB_MISPARSE = """# 任务书 demo

## 白名单（写权限）
- 只动该模块内的实现文件
- 其它一律不改
"""


def test_misparsed_section_warns_even_with_explicit_dirs(tmp_path, capsys):
    """有显式 dirs 时，声明面被静默丢弃**也要出声**（旧实现只在「两者皆空」时告警）。"""
    p = tmp_path / "tb-misparse.md"
    p.write_text(TB_MISPARSE, encoding="utf-8")

    out = dispatch.dirs_with_taskbook_whitelist(["src"], p)
    msg = capsys.readouterr().out

    assert out == ["src"]  # 仍按显式 dirs 注入
    assert "解析不出任何目录" in msg
    assert "声明面被静默丢弃" in msg


def test_misparsed_section_warns_when_nothing_else(tmp_path, capsys):
    """两者皆空的老分支语义保留（组员会被拦成 whitelist_conflict）。"""
    p = tmp_path / "tb-misparse2.md"
    p.write_text(TB_MISPARSE, encoding="utf-8")

    out = dispatch.dirs_with_taskbook_whitelist([], p)
    msg = capsys.readouterr().out

    assert out == []
    assert "whitelist_conflict" in msg


def test_normal_taskbook_emits_no_new_warning(tmp_path, capsys):
    """正常任务书（能解析出目录）**无新告警**（criteria：不误报）。"""
    assert dispatch.dirs_with_taskbook_whitelist([], _taskbook(tmp_path))
    assert "解析不出任何目录" not in capsys.readouterr().out


def test_no_whitelist_section_emits_no_warning(tmp_path, capsys):
    """压根没写白名单段 → 不是格式失配，不告警。"""
    p = tmp_path / "tb-nosec.md"
    p.write_text("# 任务书\n\n## 目标\n做事。\n", encoding="utf-8")
    assert dispatch.dirs_with_taskbook_whitelist(["src"], p) == ["src"]
    assert "解析不出任何目录" not in capsys.readouterr().out
