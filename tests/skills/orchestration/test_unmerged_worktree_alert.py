"""「已提交未合并」检出与告警（backlog 20261015-145045 f4a）。

实案（2026-10-15 两起，用户征询后裁定必修）：devops 尾段「提交→合并→推送」被进程杀死/
跳截断/重启时停在**已提交未合并**——成果卡在 worktree 里静默躺平（6304 卡 3 小时），零告警。

判据**复用** ``worktree_salvage_state`` 的 ``unmerged``（``git rev-list --count main..branch``），
不新造第二份；**只检出出声，不自动合并**（组长裁决：patrol 自动 merge 进 main 是护栏级写操作）。
"""

from __future__ import annotations

import re
from pathlib import Path

from pysci.skills.orchestration.tools import drain, events, workflow


def _fake_git(
    monkeypatch, *, unmerged: str, worktrees: list[str], head="abc123456789", fresh=True
):
    """假 git：只回答本判据要用的那几条命令。

    **批量口径**（2026-10-15 加急 adhoc-storm-fix）：面清单走一次 ``worktree list --porcelain``、
    各面 head 走一次 ``for-each-ref``；领先数 ``rev-list --count`` 与分叉点 ``merge-base``
    只在该 (branch, head) 未缓存时才问。``fresh=False`` 保留跨调用缓存（供「空闲 tick」用例）。
    """
    calls: list[tuple[str, ...]] = []

    def _run(*args: str):
        calls.append(tuple(args))
        if args[:3] == ("worktree", "list", "--porcelain"):
            return 0, "\n\n".join(f"worktree {p}" for p in worktrees) + "\n"
        if args[:1] == ("for-each-ref",):
            return 0, "".join(f"wt-{Path(p).name} {head}\n" for p in worktrees)
        if args[:2] == ("rev-list", "--count"):
            return 0, unmerged
        if args[:1] == ("merge-base",):
            return 0, "deadbeef1234"
        if args[:1] == ("rev-parse",) or args[:1] == ("-C",):
            return 0, head
        return 1, ""

    monkeypatch.setattr(drain, "_run_git", _run)
    if fresh:
        drain._UNMERGED_CACHE.clear()  # 缓存是进程内全局：跨用例会串味
    return calls


def _face(name: str) -> str:
    return str(workflow.WORKTREES_ROOT / name)


def test_unmerged_face_is_detected_with_readings(iso_state, monkeypatch):
    """分支领先 main → 检出并给出读数（分支/提交数/落后点/现成命令）。"""
    _fake_git(monkeypatch, unmerged="3", worktrees=[_face("face-a")])

    alerts = drain.unmerged_worktree_alerts()

    assert len(alerts) == 1
    a = alerts[0]
    assert a["worktree"] == "face-a" and a["branch"] == "wt-face-a"
    assert a["unmerged"] == 3
    assert a["head"] == "abc123456789"
    assert a["behind_point"] == "deadbeef1234", "落后点＝与 main 的分叉点"
    assert a["merge_cmd"] == "pysci-dev worktree merge face-a", "须附现成命令供一键执行"
    assert "3 个提交未合并" in a["title"]


def test_merged_face_is_silent(iso_state, monkeypatch):
    """已合并（领先 0）→ 不误报。"""
    _fake_git(monkeypatch, unmerged="0", worktrees=[_face("face-a")])
    assert drain.unmerged_worktree_alerts() == []


def test_no_worktree_is_silent(iso_state, monkeypatch):
    _fake_git(monkeypatch, unmerged="3", worktrees=[])
    assert drain.unmerged_worktree_alerts() == []


def test_non_orchestration_worktrees_are_ignored(iso_state, monkeypatch):
    """仓外/主树的 worktree 不算本编排的面（面根之外一律忽略）。"""
    _fake_git(monkeypatch, unmerged="3", worktrees=[str(iso_state / "elsewhere")])
    assert drain.unmerged_worktree_alerts() == []


def test_announce_emits_once_per_branch_head(iso_state, monkeypatch):
    """出声：同面同 head 只喊一次；head 变了（又有新提交）再喊。"""
    _fake_git(monkeypatch, unmerged="2", worktrees=[_face("face-b")])
    sent: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        events,
        "emit",
        lambda kind, **kw: sent.append(
            (kind, kw.get("subject", ""), kw.get("action", ""))
        ),
    )
    seen: set = set()

    first = drain.announce_unmerged_worktrees(seen)
    second = drain.announce_unmerged_worktrees(seen)

    assert len(first) == 1 and second == [], "同一 (branch, head) 不得每分钟重复喊"
    assert len(sent) == 1
    kind, subject, action = sent[0]
    assert kind == events.NEEDS_LEADER and subject == "wt-face-b"
    assert action == "pysci-dev worktree merge face-b", "出声须点名现成合并命令"

    _fake_git(
        monkeypatch, unmerged="5", worktrees=[_face("face-b")], head="cafe00001111"
    )
    third = drain.announce_unmerged_worktrees(seen)
    assert len(third) == 1 and len(sent) == 2, "head 变了＝又有新提交，应再喊一次"


def test_watchdog_tick_calls_the_alert(iso_state, monkeypatch):
    """接线：检出挂在既有巡检 tick（不新造定时器）——改回不挂即红。"""
    from pathlib import Path

    from pysci.skills.orchestration.tools import watchdog

    src = Path(watchdog.__file__).read_text(encoding="utf-8")
    assert "_drain.announce_unmerged_worktrees(unmerged_seen)" in src
    assert "unmerged_seen" in src


# ---------------------------------------------------------------------------
# 空终端风暴：① 隐藏窗口口径 ② 空闲 tick 体量（2026-10-15 加急 adhoc-storm-fix）
# ---------------------------------------------------------------------------
def test_run_hidden_sets_create_no_window_on_win32(monkeypatch):
    """判据①：win32 上一切 spawn 都带 ``CREATE_NO_WINDOW``（父进程有无控制台都一样）。"""
    import subprocess
    import sys as _sys

    from pysci.skills.orchestration.tools import procs

    seen: list[dict] = []

    class _P:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(argv, **kw):
        seen.append(kw)
        return _P()

    monkeypatch.setattr(procs.subprocess, "run", _fake_run)
    procs.run_hidden(["git", "status"], capture_output=True, text=True)

    assert len(seen) == 1
    if _sys.platform == "win32":
        assert seen[0].get("creationflags", 0) & subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in seen[0], "POSIX 不加旗标（行为不变）"


def test_no_bare_spawn_outside_the_single_landing_point():
    """判据③（防再漏）：**无控制台可达面**（orchestration/tools + devops/tools）除口径实现
    与显式豁免外**不得有裸 spawn**。

    DEBT 清单已于 adhoc-debt-spawns **清零**；wake-bind-20261018 s1 把 devops/tools 的
    低频遗留站点（mdgen/dev/probe/skilltax/keyscan/guard_probe）也并入同一口径——这些
    工具会在 pythonw tick / DETACHED worker 的孙进程层级被调，裸 spawn 同样闪窗。
    扫描面递归到子目录（tui 也在内）；新出现的裸 spawn 即红，把「再漏一处」从靠人盯
    变成机械拦截。

    **豁免**必须双证：文件在 ``EXEMPT`` 里给出理由，且源码含 ``spawn-hidden-exempt:``
    标记（理由写在调用点旁边，代码移动/删除时一并可见）。
    """

    from pysci.paths import PROJECT_ROOT

    # 入口形态不止 ``subprocess.run/Popen``：``check_output``/``call``/``os.system``/
    # 未限定 ``Popen(`` 同样是裸 spawn（只是本树当前一处都没有——扫描面先收严）。
    patterns = (
        "subprocess.run(",
        "subprocess.Popen(",
        "subprocess.call(",
        "subprocess.check_output(",
        "subprocess.check_call(",
        "os.system(",
        "os.popen(",
    )
    faces = (
        PROJECT_ROOT / "src" / "pysci" / "skills" / "orchestration" / "tools",
        PROJECT_ROOT / "src" / "pysci" / "skills" / "devops" / "tools",
    )
    # 显式豁免：rel 路径 → 理由（源码里必须同时有 spawn-hidden-exempt: 标记）。
    exempt = {
        "tui/app.py": "interactive-editor：VS Code 由用户在 TUI 显式拉起，可见属预期",
    }

    offenders: list[str] = []
    for tools in faces:
        if not tools.is_dir():
            continue
        for p in sorted(tools.rglob("*.py")):
            if p.name == "procs.py":  # 唯一落点本体
                continue
            rel = p.relative_to(tools).as_posix()
            text = p.read_text(encoding="utf-8")
            has_bare = any(pat in text for pat in patterns) or re.search(
                r"(?<![\w.])Popen\(", text
            )
            if rel in exempt:
                assert "spawn-hidden-exempt:" in text, (
                    f"{rel} 在豁免名单里，但源码缺 spawn-hidden-exempt: 标记"
                )
                continue
            if has_bare:
                offenders.append(rel)
    assert "tui/app.py" in exempt and "spawn-hidden-exempt:" in (
        faces[0] / "tui" / "app.py"
    ).read_text(encoding="utf-8"), "交互式启动器的豁免理由必须留在源码里"

    assert offenders == [], (
        f"新出现裸 spawn（请改走 procs.run_hidden/popen_hidden）：{offenders}"
    )


def test_idle_tick_costs_at_most_two_git_calls(iso_state, monkeypatch):
    """判据②：**空闲 tick 的 git 派生 ≤ 2**（一次面清单 + 一次 for-each-ref；同 head 全走缓存）。"""
    _fake_git(monkeypatch, unmerged="3", worktrees=[_face("face-a"), _face("face-b")])

    first_alerts = drain.unmerged_worktree_alerts()
    assert len(first_alerts) == 2, first_alerts

    calls2 = _fake_git(
        monkeypatch,
        unmerged="3",
        worktrees=[_face("face-a"), _face("face-b")],
        fresh=False,  # 保留缓存＝「同一个长驻 tick 的下一次」
    )
    drain.unmerged_worktree_alerts()

    assert len(calls2) == 2, f"无变化的 tick 必须只花 2 条 git：{calls2}"
    assert {c[0] for c in calls2} == {"worktree", "for-each-ref"}


def test_head_change_recomputes_only_that_face(iso_state, monkeypatch):
    """有变化才多花：head 变了 → 该面重算（rev-list+merge-base），其余仍走缓存。"""
    _fake_git(monkeypatch, unmerged="3", worktrees=[_face("face-a")])
    drain.unmerged_worktree_alerts()

    calls = _fake_git(
        monkeypatch, unmerged="4", worktrees=[_face("face-a")], head="cafe00001111"
    )
    alerts = drain.unmerged_worktree_alerts()

    assert len(calls) == 4, calls  # 2 批量 + 该面的 rev-list + merge-base
    assert alerts and alerts[0]["unmerged"] == 4 and alerts[0]["head"] == "cafe00001111"
