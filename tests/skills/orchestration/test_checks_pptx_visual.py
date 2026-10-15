"""pptx 视觉验收 check（`pptx-visual`）的注册与路由。

用户硬要求「不要让组员在不经视觉检查的情况下提交 ppt 产物」的**机械面**：把
`compose slides verify` 注册为随附 check（`pptx-visual`），交付声明
``<artifact check="pptx-visual">…deck.pptx</artifact>`` 即由 orch 自动跑——校验逐页渲染
存在、页数 = 幻灯片数、非空白页，缺渲染/页数不符即退出码非 0 判 FAIL。

本测试钉两条不变量（形状同 `test_checks_orch_tests.py`）：随附 registry.json 含此 check
且 cmd 形状正确；run_checks 能把产物路径喂给 `{path}` 并据退出码判 pass/fail。
"""

from __future__ import annotations

import json
from pathlib import Path

from pysci.skills.orchestration.tools.checks import run_checks
from pysci.skills.orchestration.tools.delivery import Artifact
from pysci.skills.orchestration.tools.registry import REGISTRY_PATH, Registry


def _fake_run(monkeypatch, rc: int = 0, out: str = ""):
    calls: list[dict] = []

    class _Proc:
        def __init__(self, code: int, stdout: str):
            self.returncode = code
            self.stdout = stdout
            self.stderr = ""

    def _run(argv, **kw):
        calls.append({"argv": list(argv), "cwd": kw.get("cwd")})
        return _Proc(rc, out)

    from pysci.skills.orchestration.tools import procs as _procs

    monkeypatch.setattr(_procs.subprocess, "run", _run)
    return calls


def test_repo_shipped_registry_has_pptx_visual_check():
    """随附 registry.json 必含 pptx-visual，cmd 走 `pysci-compose slides verify {path}`。

    `--no-sync` 是纪律面（同 orch-tests/figure-audit）：机械验收由派发进程逐跳执行，
    裸 `uv run` 的隐式 sync 在 Windows 上会撞运行中 exe 的锁。
    """
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    spec = data.get("checks", {}).get("pptx-visual")
    assert spec is not None, "随附 registry.json 应注册 pptx-visual"
    assert spec["cmd"] == [
        "uv",
        "run",
        "--no-sync",
        "pysci-compose",
        "slides",
        "verify",
        "{path}",
    ]
    # 与旧 check 并存（新增不该顶掉谁）
    assert {"figure-audit", "orch-tests", "pptx-visual"} <= set(data["checks"])


def test_run_checks_feeds_artifact_path_to_pptx_visual(monkeypatch):
    """{path} 替换成产物绝对路径，cwd=PROJECT_ROOT，pass/fail 只认退出码。"""
    calls = _fake_run(monkeypatch, rc=0, out="[slides verify] PASS deck.pptx")
    art = Artifact(
        path="data/research/1_gain_ep/汇报PPT/20260918/20260918.pptx",
        check="pptx-visual",
    )
    results = run_checks([art], Registry.load(), member_pod=Path("."))

    assert results[0]["verdict"] == "pass"
    argv = calls[0]["argv"]
    assert argv[:5] == ["uv", "run", "--no-sync", "pysci-compose", "slides"]
    assert argv[-1].endswith("20260918.pptx"), argv
    assert "{path}" not in "".join(argv)


def test_run_checks_pptx_visual_fail_on_nonzero_exit(monkeypatch):
    """缺渲染/页数不符 → 验收命令非零退出 → verdict=fail。"""
    _fake_run(monkeypatch, rc=1, out="[slides verify] FAIL 缺渲染")
    art = Artifact(path="data/x/deck.pptx", check="pptx-visual")
    results = run_checks([art], Registry.load(), member_pod=Path("."))
    assert results[0]["verdict"] == "fail"
    assert results[0]["returncode"] == 1
