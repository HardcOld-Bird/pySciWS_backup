"""orch-tests 机械验收 check 的注册与路由（backlog 20261010-163347-devops）。

基础设施类交付（orch CLI / guards / pod 只读层）以往只能声明 ``check="none"``，
验收完全依赖 devops 自述测试证据 + 组长人肉跑 pytest——「能代码硬保证的不靠纪律」
同一形状的缺口。此处在 registry.json 的 checks 块注册 ``orch-tests``，本测试钉住
两条不变量：随附 registry 含此 check 且 cmd 形状正确；run_checks 能路由且不因
模板里没有 ``{path}`` 而崩（pytest 命令不吃产物路径）。
"""

from __future__ import annotations

import json
from pathlib import Path

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import checks as checks_mod
from pysci.skills.orchestration.tools import watch
from pysci.skills.orchestration.tools.checks import run_checks
from pysci.skills.orchestration.tools.delivery import Artifact
from pysci.skills.orchestration.tools.registry import REGISTRY_PATH, Registry


def _fake_run(monkeypatch, rc: int = 0, out: str = "253 passed"):
    """把 subprocess.run 打桩；记录每次调用的 (argv, cwd)。"""
    calls: list[dict] = []

    class _Proc:
        def __init__(self, code: int, stdout: str):
            self.returncode = code
            self.stdout = stdout
            self.stderr = ""

    def _run(
        argv,
        *,
        cwd=None,
        capture_output=True,
        text=True,
        encoding=None,
        errors=None,
        timeout=None,
    ):
        calls.append({"argv": list(argv), "cwd": cwd, "timeout": timeout})
        return _Proc(rc, out)

    monkeypatch.setattr(checks_mod.subprocess, "run", _run)
    return calls


def test_repo_shipped_registry_has_orch_tests_check():
    """随附 registry.json 必含 orch-tests，cmd 为 uv run --no-sync pytest tests/skills/orchestration -q。

    护栏：devops 类交付声明 check="orch-tests" 时若随附注册缺失，run_checks 会走
    unrouted 分支静默放行——本测试在 CI 层面钉住「随附必须真注册」。--no-sync 是
    纪律面：机械验收由派发进程逐跳执行，裸 uv run 的隐式 sync 在 Windows 上会撞
    运行中 exe 的锁并先删光其余入口（backlog 20261012-125455-devops）。
    """
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    spec = data.get("checks", {}).get("orch-tests")
    assert spec is not None, "随附 registry.json 应注册 orch-tests"
    assert spec["cmd"] == [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "tests/skills/orchestration",
        "-q",
    ]


def test_repo_shipped_registry_keeps_figure_audit_check():
    """新增 orch-tests 不该顶掉旧 figure-audit（两条 check 并存）。"""
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert "figure-audit" in data.get("checks", {})
    assert "orch-tests" in data.get("checks", {})


def test_repo_shipped_registry_devops_turns_above_80():
    """夜航实锤：多部分大单在 80 轮处三连 run_failed（台账 num_turns 恒 80）——腰斩的是
    参数不是规格（backlog 20261012-073700-leader 要求 1）。随附 registry 必须已抬到 150
    且留着注记，防止下一次有人把它改回 80。timeout 维持 10800（3h 内 150 轮有余量）。"""
    devops = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["members"]["devops"]
    assert devops["max_turns"] == 150
    assert devops["timeout_s"] == 10800
    assert "num_turns" in devops.get("turns_note", "")


def test_manual_fallback_warning_lives_in_the_docs():
    """手动兜底的硬警示在**文档**（不再在输出）：SKILL 报「唯一顶层命令」，
    drain-watch.md §2 载完整警示（`&`/父任务）；code 侧输出不再背它
    （自动挂载为主，backlog 20261013-222533-leader）。"""
    skill = (ORCHESTRATION_ROOT / "skills" / "orchestration" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    drain_watch = (
        ORCHESTRATION_ROOT
        / "skills"
        / "orchestration"
        / "references"
        / "drain-watch.md"
    ).read_text(encoding="utf-8")
    assert "唯一顶层命令" in skill
    assert "`&`" in drain_watch and "父任务" in drain_watch
    assert "唯一顶层命令" not in "\n".join(watch.autowatch_lines())


def test_run_checks_routes_orch_tests_without_path_placeholder(monkeypatch):
    """orch-tests 模板不吃 {path}——run_checks 仍能拼出可执行 argv 且 verdict=pass。

    {path} 替换是 str.replace（无匹配即原样），路径不存在也不该抛。断言 argv 与
    cwd（PROJECT_ROOT）都正确，且产物 path 只做记录不进 argv。
    """
    calls = _fake_run(monkeypatch, rc=0)
    reg = Registry.load()
    art = Artifact(path="arbitrary-not-a-real-file.md", check="orch-tests")
    results = run_checks([art], reg, member_pod=Path("."))
    assert len(results) == 1 and results[0]["verdict"] == "pass"
    assert len(calls) == 1
    assert calls[0]["argv"] == [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "tests/skills/orchestration",
        "-q",
    ]
    assert "{path}" not in "".join(calls[0]["argv"])


def test_run_checks_orch_tests_fail_verdict(monkeypatch):
    """pytest 非零退出 → fail（verdict 只反映 exit code；不吞异常）。"""
    _fake_run(monkeypatch, rc=1, out="3 failed")
    reg = Registry.load()
    art = Artifact(path="x", check="orch-tests")
    results = run_checks([art], reg, member_pod=Path("."))
    assert results[0]["verdict"] == "fail"
    assert "3 failed" in results[0]["output_tail"]


# ---------------------------------------------------------------------------
# 判据防污染（backlog 20261015-132939-fix3；实案 A.a5 误杀）
# ---------------------------------------------------------------------------
def test_output_tail_is_anchored_by_the_exit_code(monkeypatch):
    """摘要把退出码摆在**首位**，且 ``returncode`` 作为字段单列——读 tail 的人不必从
    尾部文本里猜判据（实案 A.a5 的误读土壤）。"""
    _fake_run(monkeypatch, rc=0, out="253 passed")
    results = run_checks(
        [Artifact(path="x", check="orch-tests")], Registry.load(), member_pod=Path(".")
    )
    assert results[0]["output_tail"].startswith("[check exit=0]")
    assert results[0]["returncode"] == 0


def test_sentinel_note_in_stdout_cannot_flip_a_green_run(monkeypatch):
    """A.a5 同款：stdout 末尾是「防写穿哨兵豁免留痕」块、进程退出码 0 → 判据仍 **pass**。

    （留痕本身已改走 stderr，见 ``test_state_write_sentinel``；此处再钉一层「判据只认
    退出码」——即便摘要里出现留痕字样也不得动摇 verdict。）
    """
    out = (
        "1621 passed, 3 skipped in 114.80s\n"
        + "=" * 25
        + " 防写穿哨兵豁免留痕（allow_real_state） "
        + "=" * 25
        + "\n  - tests/skills/orchestration/test_state_write_sentinel.py::x\n"
    )
    _fake_run(monkeypatch, rc=0, out=out)
    results = run_checks(
        [Artifact(path="x", check="orch-tests")], Registry.load(), member_pod=Path(".")
    )
    assert results[0]["verdict"] == "pass"
    assert results[0]["returncode"] == 0


def test_registry_load_save_roundtrip_preserves_orch_tests(tmp_path, monkeypatch):
    """Registry.load→save→load 往返保留 checks 块两条（save 走原子写 tmp→replace）。"""
    src = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    isolated = tmp_path / "registry.json"
    isolated.write_text(json.dumps(src, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        "pysci.skills.orchestration.tools.registry.REGISTRY_PATH", isolated
    )
    # Registry.path 在类实例化时以 REGISTRY_PATH 为默认，需构造后手动绑定 path
    reg = Registry(data=json.loads(isolated.read_text(encoding="utf-8")), path=isolated)
    reg.save()
    after = json.loads(isolated.read_text(encoding="utf-8"))
    assert set(after["checks"]) >= {"figure-audit", "orch-tests"}
    assert after["checks"]["orch-tests"]["cmd"] == [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "tests/skills/orchestration",
        "-q",
    ]
