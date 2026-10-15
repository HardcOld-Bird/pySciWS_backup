"""声明「入 git」的内容面的**落盘即提交**（backlog 20261015-085823-devops）。

``orchestration/state/deliveries/`` 与 ``data/taskbooks/`` 声明入库却只写不 add → 新留档
永远停在 ``??``、远端取不到（2026-10-15 实发）。这里钉住 :mod:`statecommit` 的四条护栏
（跟踪才提交 / 无变更不空提交 / 面外跳过 / 失败 fail-soft）与两处接线。

用**真 git 仓库**（tmp_path）验行为——桩 git 只能验调用，验不了「提交真发生了」。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.skills.orchestration.tools import queueworker, statecommit, taskbooks

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="需真 git（验提交真发生，桩替不了）"
)


def _repo(root: Path) -> None:
    """在 tmp_path 造一个自足 git 仓库并设置身份（不依赖开发者全局 git config）。"""
    subprocess.run(["git", "init", "-q", "."], cwd=str(root), check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """临时 git 仓库 + 身份；``statecommit.PROJECT_ROOT`` 指向它。"""
    _repo(tmp_path)
    for k, v in (
        ("GIT_AUTHOR_NAME", "t"),
        ("GIT_AUTHOR_EMAIL", "t@example.com"),
        ("GIT_COMMITTER_NAME", "t"),
        ("GIT_COMMITTER_EMAIL", "t@example.com"),
    ):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(statecommit, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, encoding="utf-8"
    )


def _seed_tracked_dir(root: Path, rel: str) -> Path:
    """把一个目录连同一份既有文件提交进仓库（＝「该面已被声明入库」）。"""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "seed.md").write_text("既有\n", encoding="utf-8")
    assert _git(root, "add", "--", rel).returncode == 0
    assert _git(root, "commit", "-q", "-m", "chore: seed").returncode == 0
    return d


# ---------------------------------------------------------------------------
# ① 跟踪面：新文件落盘即提交（判据：新留档在 git log --stat 可见、不再 ??）
# ---------------------------------------------------------------------------
def test_commits_new_file_on_tracked_face(repo):
    d = _seed_tracked_dir(repo, "data/taskbooks")
    (d / "tb-1.taskbook.md").write_text("新任务书\n", encoding="utf-8")

    out = statecommit.commit_face(d, "chore(state): 任务书入库")

    assert out.startswith("[↑]"), out
    status = _git(repo, "status", "--porcelain", "--", "data/taskbooks").stdout
    assert "??" not in status, f"落盘即提交后不该留未跟踪项：{status!r}"
    show = _git(repo, "log", "-1", "--stat", "--format=%s").stdout
    assert "任务书入库" in show and "tb-1.taskbook.md" in show


def test_no_change_makes_no_empty_commit(repo):
    d = _seed_tracked_dir(repo, "data/taskbooks")
    head = _git(repo, "rev-parse", "HEAD").stdout

    assert statecommit.commit_face(d, "chore(state): 任务书入库") == ""
    assert _git(repo, "rev-parse", "HEAD").stdout == head, "无变更不该产生空提交"


# ---------------------------------------------------------------------------
# ② 未被跟踪的面：不擅自纳入版本控制（有意不入库/尚未首提）
# ---------------------------------------------------------------------------
def test_skips_untracked_face(repo):
    d = repo / "data" / "taskbooks"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tb-1.taskbook.md").write_text("x\n", encoding="utf-8")

    out = statecommit.commit_face(d, "chore(state): 任务书入库")

    assert "未被跟踪" in out, out
    assert _git(repo, "status", "--porcelain").stdout.startswith("??"), "不该被自动纳入"


# ---------------------------------------------------------------------------
# ③ 面在项目根之外（测试 tmp 面等）：直接跳过，不去猜相对路径
# ---------------------------------------------------------------------------
def test_skips_path_outside_root(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "taskbooks"
    outside.mkdir()
    assert "不在项目根下" in statecommit.commit_face(outside, "chore(state): x")


# ---------------------------------------------------------------------------
# ④ 提交失手 fail-soft：先按「被钩子改写不算失败」重试一次，再退回暂存态
# ---------------------------------------------------------------------------
def test_retries_once_after_hook_rewrites(repo, monkeypatch):
    """首投被 pre-commit 钩子改写中止 → 重新 add 再试一次即成功（实测首投必栽此步）。"""
    d = _seed_tracked_dir(repo, "data/taskbooks")
    (d / "tb-2.taskbook.md").write_text("新\n", encoding="utf-8")
    real = statecommit._git
    attempts: list[str] = []

    def fake(*args: str):
        if args and args[0] == "commit":
            attempts.append("commit")
            if len(attempts) == 1:
                return subprocess.CompletedProcess(args, 1, "", "hook modified files")
        return real(*args)

    monkeypatch.setattr(statecommit, "_git", fake)
    out = statecommit.commit_face(d, "chore(state): 任务书入库")

    assert out.startswith("[↑]"), out
    assert len(attempts) == 2, "改写中止后须重试一次（重试仍不过才算失手）"


def test_commit_failure_fails_soft_and_unstages(repo, monkeypatch):
    d = _seed_tracked_dir(repo, "data/taskbooks")
    (d / "tb-1.taskbook.md").write_text("新\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    real = statecommit._git

    def fake(*args: str):
        calls.append(args)
        if args and args[0] == "commit":
            return subprocess.CompletedProcess(args, 1, "", "boom: 钩子改写")
        return real(*args)

    monkeypatch.setattr(statecommit, "_git", fake)
    out = statecommit.commit_face(d, "chore(state): 任务书入库")

    assert "自动提交失败" in out and "boom" in out, out
    assert [c[0] for c in calls].count("commit") == 2, "重试一次后仍失败才算真失手"
    assert ("reset", "-q", "--", "data/taskbooks") in calls, "失手须退回暂存态"


# ---------------------------------------------------------------------------
# ⑤ 接线：两条写入路径都真的叫了它（改回「只写不提交」即红）
# ---------------------------------------------------------------------------
def test_taskbooks_create_commits_face(tmp_path, monkeypatch):
    from pysci import paths as _paths

    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tmp_path / "data" / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", tmp_path / "data")
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        statecommit, "commit_face", lambda p, m: seen.append((str(p), m)) or ""
    )

    taskbooks.create("标题", "正文")

    assert [m for _p, m in seen] == [taskbooks.TASKBOOK_COMMIT_MSG], seen


def test_archive_delivery_commits_face(tmp_path, monkeypatch):
    monkeypatch.setattr(queueworker, "DELIVERIES_DIR", tmp_path / "deliveries")
    seen: list[str] = []
    monkeypatch.setattr(
        statecommit, "commit_face", lambda p, m: seen.append(str(p)) or ""
    )

    rel = queueworker.archive_delivery({"id": "j1", "member": "sim"}, {}, body="正文")

    assert rel and seen == [str(tmp_path / "deliveries")], seen


# ---------------------------------------------------------------------------
# ⑥ 任务书**状态改写**也即写即提交（backlog 20261015-153541-tier-a：
#    该面的 state.json 曾以改动态滞留 main，拦下别人的合并交集闸 → merge_dirty_face）
# ---------------------------------------------------------------------------
def test_taskbook_state_rewrite_leaves_the_face_clean(repo, monkeypatch):
    from pysci import paths as _paths
    from pysci.skills.orchestration.tools import taskbooks

    d = _seed_tracked_dir(
        repo, "data/taskbooks"
    )  # 面已入册（真实场景：面本就有跟踪文件）
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", d)
    monkeypatch.setattr(_paths, "DATA_ROOT", repo / "data")

    tid = taskbooks.create("演示大任务", "正文")["id"]  # 创建：新文件即提交
    taskbooks.link_plan(tid, "p1")  # **改写** state.json：此前会留 ` M` 脏面

    status = _git(repo, "status", "--porcelain", "--", "data/taskbooks").stdout
    assert status.strip() == "", (
        f"改写须即写即提交、不留脏面（曾因此拦下别人合并）：{status!r}"
    )


def test_taskbook_state_save_is_wired_to_commit_face():
    """接线在场（改回「只写不提交」即红）。"""
    from pathlib import Path

    from pysci.skills.orchestration.tools import taskbooks

    src = Path(taskbooks.__file__).read_text(encoding="utf-8")
    assert "statecommit.commit_face(TASKBOOKS_DIR, TASKBOOK_COMMIT_MSG)" in src
