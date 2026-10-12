"""worktree merge 机械化（backlog 20261010-worktree-merge-cp）。

--no-ff 只带 git 面，sessions-runtime.json/registry.presplit.json/watch.json 等
gitignored 运行态不随迁——registry-runtime-split 那次靠人工 cp 才保住 5 成员 6 会话。
`pysci-dev worktree merge` 把「cp → merge → 会话数不减自检」固化，本组钉住其纯函数面：
cp 语义（new vs bak 覆盖）、fail-open、会话总数读数、cmd 主体在倒退时中止且不起 git。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pysci.skills.devops.tools import dev


def _mk_state(root: Path, name: str, text: str) -> Path:
    sd = root / "orchestration" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    p = sd / name
    p.write_text(text, encoding="utf-8")
    return p


def _runtime_bytes(n: int) -> str:
    """n 条会话的运行态 JSON（单成员 all）。"""
    sessions = [{"sid": f"s{i}", "hops": i, "status": "active"} for i in range(n)]
    return json.dumps({"version": 1, "members": {"all": {"sessions": sessions}}})


# ---------------------------------------------------------------------------
# cp 语义
# ---------------------------------------------------------------------------
def test_cp_ignored_state_new_target(tmp_path):
    """main 侧无同名 → 纯 cp，tag=new，不产备份。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(2))
    log = dev._cp_ignored_state(wt, main, ts="20261010T2300")
    assert ("sessions-runtime.json", "new") in log
    dst = main / "orchestration" / "state" / "sessions-runtime.json"
    assert dst.read_text(encoding="utf-8") == _runtime_bytes(2)


def test_cp_ignored_state_backs_up_existing(tmp_path):
    """main 侧已存在同名 → 先 .bak-<ts> 再覆盖。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", "NEW")
    _mk_state(main, "sessions-runtime.json", "OLD")
    log = dev._cp_ignored_state(wt, main, ts="TSTAMP")
    tags = dict(log)
    assert tags["sessions-runtime.json"].startswith("bak=")
    sd = main / "orchestration" / "state"
    assert (sd / "sessions-runtime.json").read_text(encoding="utf-8") == "NEW"
    assert (sd / "sessions-runtime.json.bak-TSTAMP").read_text(
        encoding="utf-8"
    ) == "OLD"


def test_cp_ignored_state_skips_missing_source(tmp_path):
    """worktree 侧无该文件 → 不搬、不报错（fail-open）。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(wt, "watch.json", "{}")  # 只有 watch.json
    log = dev._cp_ignored_state(wt, main, ts="T")
    names = [nm for nm, _ in log]
    assert "watch.json" in names
    assert "sessions-runtime.json" not in names  # 源缺 → 不出现


def test_cp_includes_all_merge_state_candidates(tmp_path):
    """MERGE_STATE_FILES 三件全在 worktree → 全搬（保底清单生效，不依赖 git）。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    for name in dev.MERGE_STATE_FILES:
        _mk_state(wt, name, f"x-{name}")
    log = dev._cp_ignored_state(wt, main, ts="T")
    names = {nm for nm, _ in log}
    assert set(dev.MERGE_STATE_FILES) <= names


# ---------------------------------------------------------------------------
# 会话总数读数（fail-open）
# ---------------------------------------------------------------------------
def test_runtime_session_total_counts_sum(tmp_path):
    sessions = {
        "version": 1,
        "members": {"a": {"sessions": [1, 2]}, "b": {"sessions": [3]}},
    }
    _mk_state(tmp_path, "sessions-runtime.json", json.dumps(sessions))
    assert dev._runtime_session_total(tmp_path) == 3


def test_runtime_session_total_missing_is_none(tmp_path):
    """运行态文件缺席返回 None（区别于「存在但 0 条」——供合并防护分辨「无」与「少」）。"""
    assert dev._runtime_session_total(tmp_path) is None


def test_runtime_session_total_corrupt_is_zero(tmp_path):
    _mk_state(tmp_path, "sessions-runtime.json", "{not json")
    assert dev._runtime_session_total(tmp_path) == 0


# ---------------------------------------------------------------------------
# git check-ignore 发现（真 git init，秒级）
# ---------------------------------------------------------------------------
def _git_init(path: Path) -> None:
    """在 path 建独立 git 仓（check-ignore 需要仓库上下文）。"""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(path), "init", "-q"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=True,
    )


def test_ignored_state_names_discovers_via_git(tmp_path):
    """非保底清单里的 ignored 文件也被 git check-ignore 发现并纳入搬运。"""
    wt = tmp_path / "wt"
    _git_init(wt)
    (wt / "orchestration" / "state").mkdir(parents=True)
    (wt / ".gitignore").write_text(
        "/orchestration/state/sessions-runtime.json\n/orchestration/state/custom-heartbeat.json\n",
        encoding="utf-8",
    )
    _mk_state(wt, "sessions-runtime.json", "a")
    _mk_state(wt, "custom-heartbeat.json", "b")  # 非保底、被 ignored
    _mk_state(wt, "tracked-config.json", "c")  # 非 ignored
    names = dev._ignored_state_names(wt)
    assert "custom-heartbeat.json" in names
    assert "sessions-runtime.json" in names
    assert "tracked-config.json" not in names


def test_cp_picks_up_non_baseline_ignored_file(tmp_path):
    """自定义 gitignored 运行态文件也能被 cp 搬运（发现集 ∪ 保底清单）。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _git_init(wt)
    (wt / ".gitignore").write_text(
        "/orchestration/state/custom-heartbeat.json\n", encoding="utf-8"
    )
    _mk_state(wt, "custom-heartbeat.json", "HB")
    log = dev._cp_ignored_state(wt, main, ts="T")
    assert ("custom-heartbeat.json", "new") in log
    assert (main / "orchestration" / "state" / "custom-heartbeat.json").read_text(
        encoding="utf-8"
    ) == "HB"


# ---------------------------------------------------------------------------
# cmd 主体：倒退中止、正常路径、merge 失败透传
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_git(monkeypatch):
    """拦截 _run_git，记录调用；返回 (calls, set_rc)。"""
    calls: list[tuple] = []
    state = {"rc": 0, "out": "Merge made by the 'ort' strategy."}

    def _fake(*args):
        calls.append(args)
        return state["rc"], state["out"]

    monkeypatch.setattr(dev, "_run_git", _fake)
    return calls, state


def test_cmd_merge_cps_then_merges_and_self_check(tmp_path, monkeypatch, fake_git):
    """正常路径：worktree 会话更多 → cp 到 main → git merge → 会话数不减 → rc 0。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(3))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(1))
    calls, _ = fake_git
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 0
    assert ("merge", "--no-ff", "wt-foo") in calls
    # 合并后 main 的运行态=worktree 的 3 条会话（cp 生效）
    assert dev._runtime_session_total(main) == 3


def test_cmd_merge_aborts_when_sessions_would_shrink(tmp_path, monkeypatch, fake_git):
    """worktree 会话数 < main baseline → 硬中止，绝不起 git merge（防数据丢失）。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(2))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(5))
    calls, _ = fake_git
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 2
    assert not any(c[0] == "merge" for c in calls), "倒退防护必须早于 cp 与 merge"
    # 未 cp：main 仍是原来的 5 条
    assert dev._runtime_session_total(main) == 5


def test_cmd_merge_proceeds_when_worktree_has_no_runtime(
    tmp_path, monkeypatch, fake_git
):
    """worktree 从未加载/派发 → 无运行态（None），不该被误判为「比 main 少」而挡合并。

    这是防护精化的核心：main 有会话、worktree 无运行态文件时，cp 对 sessions-runtime.json
    是 no-op（源缺），会话数不减，应正常放行 merge。
    """
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(5))  # main 有，worktree 无
    calls, _ = fake_git
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 0
    assert ("merge", "--no-ff", "wt-foo") in calls
    assert dev._runtime_session_total(main) == 5  # 未被 worktree 空态覆盖


def test_cmd_merge_propagates_git_failure(tmp_path, monkeypatch, fake_git):
    """git merge 非零 → 透传 rc（cp 已落，冲突交人工）。"""
    wt = tmp_path / "wt"
    main = tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(2))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(2))
    calls, state = fake_git
    state["rc"] = 1
    state["out"] = "CONFLICT"
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 1


def test_action_merge_routes_to_command(tmp_path, monkeypatch, fake_git):
    """cmd_worktree(action=merge) 分派到 cmd_worktree_merge，wt 不存在则 rc 2。"""
    monkeypatch.setattr(dev, "WORKTREES_ROOT", tmp_path / "worktrees")
    ns = dev.argparse.Namespace(action="merge", name="nope")
    assert dev.cmd_worktree(ns) == 2

    (tmp_path / "worktrees" / "ok").mkdir(parents=True)
    monkeypatch.setattr(dev, "PROJECT_ROOT", tmp_path / "main")
    ns2 = dev.argparse.Namespace(action="merge", name="ok")
    assert dev.cmd_worktree(ns2) == 0  # 两侧无运行态 → 0>=0，merge rc 0


# ---------------------------------------------------------------------------
# 合并前交集检查（backlog 20261012-082932-devops）
# 「本次合并触碰哪些文件」的判据必须是 merge-base 起算的提交面：`main..HEAD` 在分支
# 落后 main 时把**别人新提交**的文件也算进来，实测造成过一次错误 <blocked>。
# ---------------------------------------------------------------------------
def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@e",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
    )
    return proc.stdout.strip()


def _repo_with_branch(tmp_path: Path) -> tuple[Path, Path]:
    """真 git 仓 + 一个 wt-foo 分支的 worktree（两侧都有 a/b/registry 三个跟踪文件）。"""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    for name in ("a.txt", "b.txt"):
        (main / name).write_text("base\n", encoding="utf-8")
    sd = main / "orchestration" / "state"
    sd.mkdir(parents=True)
    (sd / "registry.json").write_text('{"v":1}\n', encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "wt-foo", str(wt))
    return main, wt


def test_porcelain_paths_parsing():
    """重命名取新路径、未跟踪目录原样留尾斜杠、非 status 形态的行跳过。"""
    out = (
        " M orchestration/state/backlog.json\n"
        "?? data/others/\n"
        "R  old.md -> new.md\n"
        "A  added.txt\n"
        "Merge made by the 'ort' strategy.\n"  # 别的命令混进来的输出
    )
    assert dev.porcelain_paths(out) == {
        "orchestration/state/backlog.json",
        "data/others/",
        "new.md",
        "added.txt",
    }


def test_intersect_faces_covers_untracked_dir_prefix():
    """未跟踪**目录**脏项要命中其下的提交文件（git 会因「would be overwritten」拒合并）。"""
    committed = {"data/x.json", "src/a.py"}
    dirty = {"data/", "src/b.py"}
    assert dev.intersect_faces(committed, dirty) == ["data/x.json"]


def test_committed_face_excludes_main_advances(tmp_path, monkeypatch):
    """本单核心：分支落后 main 时，merge-base 起算的提交面不含别人的提交。

    同一情形下 `git diff --name-only main..wt-foo` 会把 main 新提交的 registry.json 也列出
    ——那正是假阳性 blocked 的来源，故一并断言（把误报形态钉成反例）。
    """
    main, wt = _repo_with_branch(tmp_path)
    (wt / "b.txt").write_text("branch work\n", encoding="utf-8")
    _git(wt, "add", "b.txt")
    _git(wt, "commit", "-q", "-m", "branch work")
    # main 前进一次（别人改了 registry.json）
    (main / "orchestration" / "state" / "registry.json").write_text(
        '{"v":2}\n', encoding="utf-8"
    )
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "main advance")
    # 且 main 工作区在 registry.json 上还有未提交改动
    (main / "orchestration" / "state" / "registry.json").write_text(
        '{"v":3} uncommitted\n', encoding="utf-8"
    )

    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    face = dev.committed_face("wt-foo")
    dirty = dev.dirty_face()

    assert face == {"b.txt"}, "本次提交面只该含我自己提交的文件"
    assert dirty == {"orchestration/state/registry.json"}
    assert dev.intersect_faces(face or set(), dirty) == []
    naive = _git(main, "diff", "--name-only", "main..wt-foo").splitlines()
    assert "orchestration/state/registry.json" in naive, "反例前提：main..HEAD 确实误报"


def test_committed_face_fail_open_when_git_errors(monkeypatch, tmp_path):
    """求交算不出（merge-base/diff 非零）→ None，调用方跳过检查而不是堵死一切合并。"""

    def _boom(*args):
        return 128, "fatal: not a git repository"

    monkeypatch.setattr(dev, "_run_git", _boom)
    assert dev.committed_face("wt-foo") is None


def test_dirty_face_reads_unstripped_porcelain(monkeypatch):
    """回归护栏：porcelain 首行的前导空格是**判位符**，不许被 ``_run_git`` 的 strip() 吃掉。

    首项恰好是「已改未暂存」（``" M path"``）时，strip 后变成 ``"M path"``——解析器视其为
    非 status 行，交集检查会**静默漏掉第一个脏文件**。
    """

    class _Proc:
        stdout = " M a.txt\n?? data/\n"

    monkeypatch.setattr(dev, "_git_proc", lambda *a: _Proc())
    assert dev.dirty_face() == {"a.txt", "data/"}


def test_merge_faces_none_when_project_root_absent(monkeypatch, tmp_path):
    """git 根本跑不起来（PROJECT_ROOT 不存在）→ merge_faces None → 检查跳过、不误挡合并。"""
    monkeypatch.setattr(dev, "PROJECT_ROOT", tmp_path / "nope")
    monkeypatch.setattr(dev, "_run_git", lambda *a: (0, "deadbeef"))
    assert dev.merge_faces("wt-foo") is None


def test_cmd_merge_blocks_on_true_overlap(tmp_path, monkeypatch, capsys):
    """真交集（我提交的 registry.json 在 main 侧未提交地脏着）→ rc 2，且 cp 与 merge 都没发生。"""
    main, wt = _repo_with_branch(tmp_path)
    (wt / "orchestration" / "state" / "registry.json").write_text(
        '{"v":2} from-branch\n', encoding="utf-8"
    )
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "branch touches registry")
    (main / "orchestration" / "state" / "registry.json").write_text(
        '{"v":9} uncommitted\n', encoding="utf-8"
    )
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(4))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(1))
    before = _git(main, "rev-parse", "HEAD")
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 2
    out = capsys.readouterr().out
    assert "orchestration/state/registry.json" in out and "交集" in out
    assert _git(main, "rev-parse", "HEAD") == before, "不该产生 merge commit"
    assert (main / "orchestration" / "state" / "registry.json").read_text(
        encoding="utf-8"
    ) == '{"v":9} uncommitted\n', "工作区未提交改动必须原样保住"
    assert dev._runtime_session_total(main) == 1, "检查先于 cp——运行态不该被搬"


def test_cmd_merge_proceeds_when_branch_behind_main(tmp_path, monkeypatch, capsys):
    """误报场景的正解：main 前进且脏在别人的文件上 → 交集空 → 照常合并（不再错误 blocked）。"""
    main, wt = _repo_with_branch(tmp_path)
    (wt / "b.txt").write_text("branch work\n", encoding="utf-8")
    _git(wt, "add", "b.txt")
    _git(wt, "commit", "-q", "-m", "branch work")
    (main / "a.txt").write_text("main advance\n", encoding="utf-8")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "main advance")
    (main / "orchestration" / "state" / "registry.json").write_text(
        '{"v":3} uncommitted\n', encoding="utf-8"
    )
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 0
    out = capsys.readouterr().out
    assert "[√] 交集为空" in out
    assert (main / "b.txt").read_text(encoding="utf-8") == "branch work\n"
    assert (main / "orchestration" / "state" / "registry.json").read_text(
        encoding="utf-8"
    ) == '{"v":3} uncommitted\n', "无关脏项不受影响"
