"""worktree merge 机械化（backlog 20261010-worktree-merge-cp）。

--no-ff 只带 git 面，sessions-runtime.json/registry.presplit.json/watch.json 等
gitignored 运行态不随迁——registry-runtime-split 那次靠人工 cp 才保住 5 成员 6 会话。
`pysci-dev worktree merge` 把「cp → merge → 会话数不减自检」固化，本组钉住其纯函数面：
cp 语义（new vs bak 覆盖）、fail-open、会话总数读数、cmd 主体在倒退时中止且不起 git，
以及 **锁面不跨树**（``*.lock`` 属排除集——backlog 20261012-191512-devops）。
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
    _mk_state(wt, "registry.presplit.json", "{}")  # 只有它
    log = dev._cp_ignored_state(wt, main, ts="T")
    names = [nm for nm, _ in log]
    assert "registry.presplit.json" in names
    assert "sessions-runtime.json" not in names  # 源缺 → 不出现


def test_cp_never_moves_wake_chain_files(tmp_path, monkeypatch):
    """**唤醒链文件永不跨树 cp**（backlog 20261016-203600-devops 实锤事故）。

    这些是 main 的活体观察面，worktree 侧那份是它出生时的旧快照：cp 过去即把事件流/水位/
    消费账/模块水位**覆盖回卷**（``events.jsonl`` 45 条被 6 条覆盖 → seq 重新编号；消费账
    ``JOB-DONE|#6`` 与回卷后的新 ``#6`` 撞车 → 真事件被静默吞掉、组长零唤醒）。
    本用例模拟「git 把四个唤醒链文件都判为 ignored」的真实场景：一个都不许搬。
    """
    wt, main = tmp_path / "wt", tmp_path / "main"
    for name in sorted(dev.MERGE_STATE_EXCLUDE_NAMES):
        _mk_state(wt, name, f"WT-{name}")
        _mk_state(main, name, f"MAIN-{name}")
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(1))
    # 模拟 git 侧 ignored 发现（真实仓库里这些文件正是 ignored）
    monkeypatch.setattr(
        dev, "_ignored_state_names", lambda w: set(dev.MERGE_STATE_EXCLUDE_NAMES)
    )

    log = dev._cp_ignored_state(wt, main, ts="T")

    moved = {nm for nm, _ in log}
    sd = main / "orchestration" / "state"
    assert "sessions-runtime.json" in moved, "会话池照搬（排除面只限唤醒链）"
    for name in sorted(dev.MERGE_STATE_EXCLUDE_NAMES):
        assert name not in moved, f"{name} 被 cp 进 main——唤醒链会被覆盖回卷"
        assert (sd / name).read_text(encoding="utf-8") == f"MAIN-{name}", (
            f"{name} 的内容被 worktree 旧快照覆盖（回卷根因）"
        )
        assert not (sd / f"{name}.bak-T").exists(), "排除面连备份都不该产生"


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
# 锁面不跨树（backlog 20261012-191512-devops）
# ---------------------------------------------------------------------------
def _gitignored_locks(wt: Path) -> None:
    """造一个「锁被 git 判为 ignored」的 worktree 面（真实仓库里正是如此）。"""
    _git_init(wt)
    (wt / ".gitignore").write_text("/orchestration/state/*.lock\n", encoding="utf-8")
    _mk_state(
        wt,
        "inspector.lock",
        '{"pid": 26520, "run_log": "…/inspector-runs/x.log"}',
    )
    _mk_state(wt, "devops.lock", '{"pid": 14956, "run_log": "…/devops-runs/y.log"}')


def test_cp_never_moves_locks(tmp_path):
    """``*.lock`` 被 git 发现为 ignored，也**绝不**搬——锁不是数据而是「谁在跑」的活体
    读数（pid + run_log 双锚）。cp 进 main 后 main 认到一把两头都不属于自己的锁：
    ``orch status`` 误报 worker 在跑，真在跑的 drain 反被这把假锁挡住不唤醒。
    """
    wt, main = tmp_path / "wt", tmp_path / "main"
    _gitignored_locks(wt)
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(1))
    found = dev._ignored_state_names(wt)
    assert {"inspector.lock", "devops.lock"} <= found, "前提：git 确实把锁判为 ignored"

    log = dev._cp_ignored_state(wt, main, ts="T")

    moved = {nm for nm, _ in log}
    assert "sessions-runtime.json" in moved, "状态文件照搬，排除面只限锁"
    sd = main / "orchestration" / "state"
    for lock in ("inspector.lock", "devops.lock"):
        assert lock not in moved
        assert not (sd / lock).exists(), f"{lock} 被 cp 进了 main——假锁会骗过判活面"


def test_cmd_merge_output_has_no_lock_line(tmp_path, monkeypatch, fake_git, capsys):
    """判据的读数面：``worktree merge`` 输出里不出现锁行（cp 日志即搬运事实）。"""
    wt, main = tmp_path / "wt", tmp_path / "main"
    _gitignored_locks(wt)
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(3))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(1))
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    assert dev.cmd_worktree_merge("foo", wt) == 0

    out = capsys.readouterr().out
    assert ".lock" not in out, f"合并面把锁当运行态搬了：\n{out}"
    assert "sessions-runtime.json" in out


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


def test_cmd_merge_thin_runtime_offers_no_cp_state_channel(
    tmp_path, monkeypatch, fake_git, capsys
):
    """会话数倒退中止时，[NEXT] 必须给出「不搬运行态」通道（backlog 20261012-194305-devops）
    ——不再逼使用者绕开工具手敲 `git merge --no-ff`。"""
    wt, main = tmp_path / "wt", tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(1))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(41))
    calls, _ = fake_git
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    assert rc == 2
    out = capsys.readouterr().out
    assert "[NEXT]" in out and "--no-cp-state" in out
    assert not any(c[0] == "merge" for c in calls), "中止早于 merge"


def test_cmd_merge_no_cp_state_merges_and_keeps_main_runtime(
    tmp_path, monkeypatch, fake_git, capsys
):
    """`--no-cp-state`：不搬运行态直接合并——跳过 cp 与会话数自检，main 版运行态原样保留。

    这正是「分支已含 main 全部提交、只是运行态薄」的孤儿回收形状的合规通道。
    """
    wt, main = tmp_path / "wt", tmp_path / "main"
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(1))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(41))
    calls, _ = fake_git
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt, no_cp_state=True)

    assert rc == 0
    assert ("merge", "--no-ff", "wt-foo") in calls  # 真走了 merge
    assert dev._runtime_session_total(main) == 41  # main 运行态未被薄面覆盖
    assert "保留 main 版" in capsys.readouterr().out


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


def _repo_with_multiline_registry(tmp_path: Path) -> tuple[Path, Path]:
    """同 ``_repo_with_branch``，但 registry.json 是**多行且够长**——让分支与 main 侧改
    相隔够远的行（git 视作不同 hunk），三路应回得以「干净合并 vs 冲突」两条路径分别可测。"""
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    sd = main / "orchestration" / "state"
    sd.mkdir(parents=True)
    (sd / "registry.json").write_text(
        '{\n  "v": 1,\n  "k1": "a",\n  "k2": "b",\n  "k3": "c",\n'
        '  "k4": "d",\n  "k5": "e",\n  "note": "base"\n}\n',
        encoding="utf-8",
    )
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "wt-foo", str(wt))
    return main, wt


def _reg(v: str, note: str) -> str:
    """registry.json 同形内容（v=第 2 行、note=末行），便于两条测试各改一处。"""
    return (
        f'{{\n  "v": {v},\n  "k1": "a",\n  "k2": "b",\n  "k3": "c",\n'
        f'  "k4": "d",\n  "k5": "e",\n  "note": "{note}"\n}}\n'
    )


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


def test_cmd_merge_no_shelve_blocks_on_true_overlap(tmp_path, monkeypatch, capsys):
    """--no-shelve（旧行为）：真交集（我提交的 registry.json 在 main 侧未提交地脏着）→ rc 2，
    且 cp 与 merge 都没发生。"""
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

    rc = dev.cmd_worktree_merge("foo", wt, shelve=False)

    assert rc == 2
    out = capsys.readouterr().out
    assert "orchestration/state/registry.json" in out and "交集" in out
    assert _git(main, "rev-parse", "HEAD") == before, "不该产生 merge commit"
    assert (main / "orchestration" / "state" / "registry.json").read_text(
        encoding="utf-8"
    ) == '{"v":9} uncommitted\n', "工作区未提交改动必须原样保住"
    assert dev._runtime_session_total(main) == 1, "检查先于 cp——运行态不该被搬"


def test_cmd_merge_auto_shelves_and_reapplies(tmp_path, monkeypatch, capsys):
    """缺省（自动搁置）：真交集不再 rc=2——合并成功，且 main 侧在飞改动三路应回、内容保住。"""
    main, wt = _repo_with_multiline_registry(tmp_path)
    # 分支改第 2 行（"v"）；main 侧未提交改末行（"note"）——相隔够远 → 应回可干净合并
    (wt / "orchestration" / "state" / "registry.json").write_text(
        _reg(2, "base"), encoding="utf-8"
    )
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "branch bumps v")
    (main / "orchestration" / "state" / "registry.json").write_text(
        _reg(1, "main-uncommitted"), encoding="utf-8"
    )
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(4))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(1))
    before = _git(main, "rev-parse", "HEAD")
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    rc = dev.cmd_worktree_merge("foo", wt)

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "自动搁置" in out and "已三路应回" in out
    assert _git(main, "rev-parse", "HEAD") != before, "应产生 merge commit（不再中止）"
    # 合并后：分支内容（v=2）与 main 侧在飞改动（note）**都在**（不丢）
    merged = (main / "orchestration" / "state" / "registry.json").read_text(
        encoding="utf-8"
    )
    assert '"v": 2' in merged and "main-uncommitted" in merged, merged


def test_cmd_merge_shelve_conflict_keeps_stash(tmp_path, monkeypatch, capsys):
    """应回冲突：不丢改动——stash 条目保留、冲突如实报（合并本身仍已完成）。"""
    main, wt = _repo_with_multiline_registry(tmp_path)
    # 两侧都改末行（note）且内容不同 → 合并后应回必冲突
    (wt / "orchestration" / "state" / "registry.json").write_text(
        _reg(1, "branch-note"), encoding="utf-8"
    )
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "branch edits note")
    (main / "orchestration" / "state" / "registry.json").write_text(
        _reg(1, "main-note"), encoding="utf-8"
    )
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(4))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(1))
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    dev.cmd_worktree_merge("foo", wt)

    out = capsys.readouterr().out
    assert "应回冲突" in out and "未丢失" in out
    # stash 条目仍在（改动可找回）
    rc, listing = dev._run_git("stash", "list")
    assert rc == 0 and "pysci-merge-shelve" in listing, listing


def test_unmerged_entries_parsing(monkeypatch):
    """porcelain 未合并条目解析：按 XY 码筛选、按路径排序、去重（脏/未跟踪行不得混入）。"""

    class _Proc:
        stdout = (
            "UU src/pysci/skills/orchestration/tools/plans.py\n"
            "AA docs/x.md\n"
            " M unrelated.md\n"
            "?? tests/decoy.py\n"
            "UU src/pysci/skills/orchestration/tools/plans.py\n"
            "Merge made by the 'ort' strategy.\n"
        )

    monkeypatch.setattr(dev, "_git_proc", lambda *a: _Proc())
    assert dev.unmerged_entries() == [
        ("AA", "docs/x.md"),
        ("UU", "src/pysci/skills/orchestration/tools/plans.py"),
    ]


def test_cmd_merge_conflict_summary_names_real_unmerged_file(
    tmp_path, monkeypatch, capsys
):
    """回归（backlog 20261017-223552-devops）：摘要列真正的 unmerged 文件。

    复现原始事故：main 侧未跟踪文件在应回时与合并结果碰撞（git 输出末行即它）；
    真正的三路冲突是 registry.json。摘要不得把末行未跟踪文件当成冲突成因。
    """
    main, wt = _repo_with_multiline_registry(tmp_path)
    (wt / "orchestration" / "state" / "registry.json").write_text(
        _reg(1, "branch-note"), encoding="utf-8"
    )
    test_rel = "tests/skills/orchestration/test_taskbook_settings.py"
    (wt / test_rel).parent.mkdir(parents=True, exist_ok=True)
    (wt / test_rel).write_text("branch version\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "branch edits note + adds test")
    (main / "orchestration" / "state" / "registry.json").write_text(
        _reg(1, "main-note"), encoding="utf-8"
    )
    (main / test_rel).parent.mkdir(parents=True, exist_ok=True)
    (main / test_rel).write_text("main untracked version\n", encoding="utf-8")
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(4))
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(1))
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)

    dev.cmd_worktree_merge("foo", wt)

    out = capsys.readouterr().out
    assert "应回冲突" in out and "未合并文件" in out, out
    conflict_block = out.split("未合并文件", 1)[1].split("处置：", 1)[0]
    assert "orchestration/state/registry.json" in conflict_block, conflict_block
    assert "test_taskbook_settings.py" not in conflict_block, (
        "与冲突无关的未跟踪文件被误列为冲突成因：\n" + conflict_block
    )
    # 真实冲突仍待解（工作区状态与摘要一致）、stash 条目保留
    conflicts = dev.unmerged_entries()
    assert any(
        path.endswith("orchestration/state/registry.json") for _, path in conflicts
    ), conflicts
    rc, listing = dev._run_git("stash", "list")
    assert rc == 0 and "pysci-merge-shelve" in listing, listing


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


# ---------------------------------------------------------------------------
# worktree add 的世代守卫（backlog 20261012-184000-leader）
# ---------------------------------------------------------------------------
def _add_guard_env(tmp_path, monkeypatch, *, listed_for=None):
    """把 WORKTREES_ROOT 与 git 桩好；``listed_for(root)`` 给 ``git worktree list`` 的行。"""
    root = tmp_path / "worktrees"
    monkeypatch.setattr(dev, "WORKTREES_ROOT", root)
    calls: list[tuple] = []

    def _fake(*args):
        calls.append(args)
        if args[:2] == ("worktree", "list"):
            return 0, "\n".join(listed_for(root) if listed_for else [])
        return 0, ""

    monkeypatch.setattr(dev, "_run_git", _fake)
    return root, calls


def test_add_on_registered_face_says_reuse(tmp_path, monkeypatch, capsys):
    """目录存在**且登记在册** = 自己的面还在 → 说明白「继续用」，但 add 不重复起（rc=2）。"""
    root, calls = _add_guard_env(
        tmp_path,
        monkeypatch,
        # 真 git 在 Windows 上输出的正是正斜杠形态（实测 ``git worktree list``），
        # 桩照此写——解析器若改了归一化方式，这里就会失配
        listed_for=lambda r: [
            f"{tmp_path / 'main'}  8e3d83b [main]",
            f"{(r / 'gen').as_posix()}  abc1234 [wt-gen]",
        ],
    )
    (root / "gen").mkdir(parents=True)

    rc = dev.cmd_worktree(dev.argparse.Namespace(action="add", name="gen"))

    assert rc == 2
    assert "登记在册" in capsys.readouterr().out
    assert not any(c[:2] == ("worktree", "add") for c in calls), "同名面不重复建"


def test_add_on_lingering_dir_points_to_next_generation(tmp_path, monkeypatch, capsys):
    """目录存在却**不在登记面里**（remove --force 后旧句柄没放）= 残骸 → 必须指路新一代名，
    绝不让人把第二个写者请回同一个房间。"""
    root, calls = _add_guard_env(
        tmp_path, monkeypatch, listed_for=lambda r: [f"{tmp_path / 'main'}  abc1234"]
    )
    (root / "gen").mkdir(parents=True)

    rc = dev.cmd_worktree(dev.argparse.Namespace(action="add", name="gen"))

    assert rc == 2
    out = capsys.readouterr().out
    assert "残骸" in out and "worktree add gen-r2" in out
    assert "rm -rf" in out, "旧目录归组长处置，不能让组员就地删"
    assert not any(c[:2] == ("worktree", "add") for c in calls)


# ---------------------------------------------------------------------------
# add 播种运行态（backlog 20261013-131505-devops）
# ---------------------------------------------------------------------------
def test_seed_runtime_from_main_into_fresh_worktree(tmp_path, monkeypatch):
    """fresh worktree（运行态空壳）+ main richer → 播种，首次 merge 无需人工 cp。

    复现的坑：`git worktree add` 只带跟踪面，gitignored 的 sessions-runtime.json 不过来，
    于是新面是空壳（0 会话），merge 的「会话数不减」闸（base=N）必中止——实测每单都要
    先人工 cp 一次才过闸。
    """
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(45))
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(0))

    line = dev._seed_runtime_from_main(wt)

    assert line.startswith("[√]"), line
    assert dev._runtime_session_total(wt) == dev._runtime_session_total(main) == 45
    assert "45" in line, "读数要报出播种了多少会话"


def test_seed_is_idempotent_and_never_thins_a_worktree(tmp_path, monkeypatch):
    """第二次调用不再覆盖（worktree 已不稀疏）——播种不许把推进过的运行态打回去。"""
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(3))
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(0))
    assert dev._seed_runtime_from_main(wt).startswith("[√]")

    # worktree 侧此后推进到比 main 多（它才是 richer 的一方）
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(9))
    line = dev._seed_runtime_from_main(wt)
    assert line.startswith("[=]"), line
    assert dev._runtime_session_total(wt) == 9, "更富的一面不许被播种覆盖"


def test_seed_skips_when_main_has_no_runtime(tmp_path, monkeypatch):
    """main 自己没有运行态 → 跳过且不造文件（别凭空写一个空壳进 worktree）。"""
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    (main / "orchestration" / "state").mkdir(parents=True, exist_ok=True)
    (wt / "orchestration" / "state").mkdir(parents=True, exist_ok=True)

    line = dev._seed_runtime_from_main(wt)

    assert "无运行态可播种" in line
    assert not (wt / "orchestration" / "state" / "sessions-runtime.json").exists()


def test_seed_fail_open_on_unwritable_target(tmp_path, monkeypatch):
    """播种失败不许把 add 炸掉（fail-open）：返回一行提示，merge 侧仍会出声。"""
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    _mk_state(main, "sessions-runtime.json", _runtime_bytes(2))
    _mk_state(wt, "sessions-runtime.json", _runtime_bytes(0))

    def _boom(self, data, **kw):  # pragma: no cover - 只在注入时走
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    line = dev._seed_runtime_from_main(wt)
    assert line.startswith("[!]") and "播种失败" in line
