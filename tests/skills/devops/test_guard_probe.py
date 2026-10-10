"""`pysci.skills.devops.tools.guard_probe` 的活体探针回归。

**这里跑的是探针本身**，用真 guard（node 直调）+ Temp fixture 端到端。三档断言：

1. **正常态**：双 guard 都在执法 + fail-open 都留痕 → status='ok'；
2. **退化态**：把 guard 复制一份、把 catch 改回静默 → 探针应报 bad（本用例锁死
   「留痕纪律不可回退」）；
3. **空壳态**：把 guard 改回裸 exit 0 → 探针应报 bad 且 `in_force=False`
   （本用例锁死「全绿不代表在执法」的可观测面）。

方法学：`monkeypatch` 把 `guard_probe.DELIVERY_GATE`/`POD_GUARD` 指到临时改写版
guard；`subprocess.run` 走真实 `node`（skipif 缺 node）。这样退化态/空壳态可复现
且不污染受治理的 guard 源文件。
"""

from __future__ import annotations

import shutil

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.devops.tools import guard_probe as gp

NODE_REQUIRED = pytest.mark.skipif(
    shutil.which("node") is None, reason="需要 node 运行 guard"
)

REAL_DELIVERY_GATE = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"
REAL_POD_GUARD = ORCHESTRATION_ROOT / "guards" / "pod-guard.mjs"


@NODE_REQUIRED
def test_probe_delivery_gate_ok_on_current_source():
    r = gp.probe_delivery_gate()
    assert r.name == "delivery-gate"
    assert r.status == "ok", r.detail
    assert r.in_force and r.fail_open_traced


@NODE_REQUIRED
def test_probe_pod_guard_ok_on_current_source():
    r = gp.probe_pod_guard()
    assert r.name == "pod-guard"
    assert r.status == "ok", r.detail
    assert r.in_force and r.fail_open_traced


@NODE_REQUIRED
def test_run_all_reports_both_guards_and_ok():
    rs = gp.run_all()
    assert [r.name for r in rs] == ["delivery-gate", "pod-guard"]
    assert all(r.status == "ok" for r in rs), [r.detail for r in rs]
    lines = gp.report(rs)
    assert all(ln.strip().startswith("[√]") for ln in lines)


@NODE_REQUIRED
def test_probe_flags_silent_catch_regression(monkeypatch, tmp_path):
    """把 delivery-gate 的 failOpen 打成空函数 → 探针应报 bad（fail_open_traced=False）。

    模拟「未来重构把 `[delivery-gate fail-open]` 留痕改成裸 catch」，本用例是留痕纪律
    的机械防线。
    """
    mutated = tmp_path / "silent-catch.mjs"
    src = REAL_DELIVERY_GATE.read_text(encoding="utf-8")
    mutated.write_text(
        src.replace(
            "console.error(`[delivery-gate fail-open] ${where} 异常，本次放行：${msg}`);",
            "/* 静默退化（模拟 2026-10-10 首版事故） */",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gp, "DELIVERY_GATE", mutated)
    r = gp.probe_delivery_gate()
    assert r.status == "bad", r.detail
    assert r.in_force, "违规仍应被拦（此处只改了 fail-open 分支）"
    assert not r.fail_open_traced, "静默 catch 必须被探针捕获"


@NODE_REQUIRED
def test_probe_flags_hollow_guard_regression(monkeypatch, tmp_path):
    """把 delivery-gate 主执行流整体短路（永远 exit 0）→ 探针应报 bad 且 in_force=False。

    模拟「guard 变成空壳、手动看 rc 一切正常」的观测盲区——`guards-in-force` 元哨兵
    与本用例是这种退化的唯一机械化拦截面。
    """
    mutated = tmp_path / "hollow.mjs"
    mutated.write_text(
        "// 空壳：吞一切，全 exit 0（模拟首版 TDZ 事故被外层 catch 静默掩盖）\n"
        'let raw = "";\n'
        "process.stdin.setEncoding('utf8');\n"
        "for await (const c of process.stdin) raw += c;\n"
        "process.exit(0);\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gp, "DELIVERY_GATE", mutated)
    r = gp.probe_delivery_gate()
    assert r.status == "bad"
    assert not r.in_force
    assert "违规 AGENTS.md 未被拦" in r.detail


@NODE_REQUIRED
def test_pod_guard_probe_flags_readonly_bypass(monkeypatch, tmp_path):
    """把 pod-guard 的 readonly 循环整段删掉 → 探针应报 bad（写 charter.md 不再被拦）。"""
    mutated = tmp_path / "no-readonly.mjs"
    src = REAL_POD_GUARD.read_text(encoding="utf-8")
    assert "命中只读层" in src, "guard 源已漂移：探针 fixture 需同步更新"
    mutated.write_text(
        src.replace("命中只读层", "命中只读层-DEADCODE"), encoding="utf-8"
    )
    # 更严格：直接把 block() 里 readonly 分支改软
    # 用 regex 替换 readonly 循环内的 block(...) 为 process.exit(0)
    weakened = src.replace(
        "if (rel === rn || rel.startsWith(rn + '/')) block(`命中只读层 ${rn}`);",
        "if (rel === rn || rel.startsWith(rn + '/')) process.exit(0);",
    )
    assert weakened != src, "替换未生效"
    mutated.write_text(weakened, encoding="utf-8")
    monkeypatch.setattr(gp, "POD_GUARD", mutated)
    r = gp.probe_pod_guard()
    assert r.status == "bad", r.detail
    assert not r.in_force, "写 charter.md 未再被拦（正是我们要捕获的退化）"
    assert "写只读层未被拦" in r.detail


def test_probe_inconclusive_when_guard_missing(monkeypatch, tmp_path):
    """guard 文件不存在（新克隆仓库缺 orchestration/guards/）→ status=inconclusive，不假绿。"""
    monkeypatch.setattr(gp, "DELIVERY_GATE", tmp_path / "does-not-exist.mjs")
    r = gp.probe_delivery_gate()
    assert r.status == "inconclusive"
    assert "guard 文件缺失" in r.detail


def test_probe_inconclusive_when_node_missing(monkeypatch):
    """node 不在 PATH → status=inconclusive（探针环境异常不能伪装成执法失败）。"""
    monkeypatch.setattr(gp, "_which_node", lambda: None)
    r = gp.probe_pod_guard()
    assert r.status == "inconclusive"
    assert "node" in r.detail


def test_report_inconclusive_does_not_mark_fail():
    """报告行首 [!] 表 advisory；`bad` 才用 [✗]——doctor rc 计数按此分档。"""
    from dataclasses import replace

    ok = gp.GuardProbeResult("a", "ok", True, True, "…")
    bad = gp.GuardProbeResult("b", "bad", False, True, "…")
    inc = gp.GuardProbeResult("c", "inconclusive", False, False, "…")
    lines = gp.report([ok, bad, inc])
    assert lines[0].strip().startswith("[√]")
    assert lines[1].strip().startswith("[✗]")
    assert lines[2].strip().startswith("[!]")
    # 显式覆盖 replace 未使用警告
    assert replace(ok, status="ok") == ok
