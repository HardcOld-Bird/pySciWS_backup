"""声明「入 git」的内容面的**落盘即提交**（backlog 20261015-085823-devops）。

**为什么需要它**：``orchestration/state/deliveries/`` 与 ``data/taskbooks/`` 都在源头声明
「入 git」（``queueworker.DELIVERIES_DIR``——交付全文是黄金任务回归集唯一入库落点；用户裁决
「用户可见可编辑」），但两个写入方**只写文件、从不 ``git add``**：承诺只在本地成立，新克隆
与远端都取不到任务书与交付全文，而 TUI 详情页正把这些路径当权威引用呈现给用户（2026-10-15
实发：两目录全部未跟踪）。

**形状**：与技能部署台账同一形状（:func:`pysci.skills.orchestration.tools.sync._commit_ledger`，
backlog 20261013-203724-devops 已合入）——跟踪面由工具改写，工具就负责随即提交，不给 main
留脏面。差别只在触发点：本模块只在**新文件诞生**时提交（新建任务书 / 新交付留档 / 新报告），
已跟踪文件的**改写**仍是普通的 `` M`` 面上改动、交给既有的 state 收尾扫描。判据是「声明入库
的面不留 ``??``」，不是「每次改动即一笔提交」。

**fail-soft**：非 git 工作树 / 面未被跟踪（有意不入库或尚未首提）/ add 或 commit 失败（钩子
改写、索引锁竞争、无身份）→ 各返回一行说明、**绝不抛**。首投被 pre-commit 钩子改写中止时
**重新 add 再试一次**（同一纪律：被钩子改写不算失败；交付全文常带 markdown 行尾双空格，
实测首投必栽在这一步），重试仍不过才算真失手。失败不额外告警也不隐藏：面仍是未跟踪或改动的，
本就是 ``git status`` 与 ``pysci-dev doctor`` 上的可见观测量，下一次状态收尾照常兜住。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pysci.paths import PROJECT_ROOT


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """在项目根跑一条 git（记账用；``PROJECT_ROOT`` 是模块级，便于测试注入）。"""
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def rel_to_root(path: str | Path) -> str | None:
    """面 → 项目根相对的 POSIX 路径；不在项目根下（测试 tmp 面等）→ ``None``。"""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def commit_face(path: str | Path, message: str) -> str:
    """把 ``path``（文件或目录面）的改动落成一笔提交；返回一行人类可读结果。

    只在 ``path`` **已被 git 跟踪**时提交——未被跟踪＝有意不入库（或尚未首提），不擅自
    把它纳入版本控制（与 ``sync._commit_ledger`` 同判据）。无改动则不产生空提交。

    Args:
        path: 面（``data/taskbooks`` 目录、``state/deliveries`` 目录或单个文件）。
        message: 提交信息（cz 规范；钩子 commitizen 会校验）。
    """
    rel = rel_to_root(path)
    if rel is None:
        return f"[i] {path} 不在项目根下——跳过自动提交"
    try:
        if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
            return f"[i] 非 git 工作树——{rel} 未自动提交"
        if _git("ls-files", "--error-unmatch", "--", rel).returncode != 0:
            return f"[i] {rel} 未被跟踪——跳过自动提交（有意不入库或尚未首提）"
        if not _git("status", "--porcelain", "--", rel).stdout.strip():
            return ""
        if _git("add", "--", rel).returncode != 0:
            return f"[i] {rel} 自动提交 add 失败——请手动提交"
        done = _git("commit", "-m", message, "--", rel)
        if done.returncode != 0:
            # pre-commit 钩子会**改写**文件（trailing-ws / end-of-file-fixer）并中止本次提交。
            # 这正是本仓纪律里的「被钩子改写不算失败」：重新 add 再试一次（2026-10-15 实测
            # 首投失败即此因——交付全文常带 markdown 行尾双空格），重试仍不过才算真失手。
            _git("add", "--", rel)
            done = _git("commit", "-m", message, "--", rel)
        if done.returncode != 0:
            # 退回暂存，让面停在可辨认的未跟踪/改动态，别卡住下一个人的提交。
            _git("reset", "-q", "--", rel)
            tail = (done.stderr or done.stdout or "").strip().splitlines()
            why = tail[-1][:120] if tail else ""
            return f"[i] {rel} 自动提交失败（{why}）——已退回未暂存，待下次收尾"
        sha = _git("rev-parse", "--short", "HEAD").stdout.strip()
        return f"[↑] {rel} 已提交 {sha}"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"[i] {rel} 自动提交异常（{type(exc).__name__}）——请手动提交"
