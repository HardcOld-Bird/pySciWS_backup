"""副本 A/B 空壳闸与 done 读数证据回归（backlog 20261012-170000-leader）。

实锤形状（run 20261012-165247，1.3s done 无台账行）：A/B 两端 ref 早于 E4——副本树里
没有 ``pysci.skills.bencher``，``-m`` 入口在副本内 ModuleNotFoundError → 零读数 →
占位 fail 行进 compare → 伪判决「未改善」→ 队列项照常 done。三块闸各一条阳性面：

- 入口重锚：副本子进程按**主树 bench.py 文件**直跑（协议稳定、被测 harness 仍来自副本）；
- 空壳闸：任一侧零真实读数 → run_ab 置 error、不出判决、不写台账；
- done 机械判据：ab 判决必须有台账读数、capture 必有归档、run 必有判决行——证据缺失
  一律 failed（静默完成比失败危险，REFACTOR §14）。

真派发、真 worktree 一律桩掉；``_hop`` 的老 ref eval 兼容单独钉（协议层能力探测）。
"""

from __future__ import annotations

import argparse
import json
import types

from pysci.skills.bencher.tools import bench as bench_tool
from pysci.skills.orchestration.tools import benchd, replica
from pysci.skills.orchestration.tools import eval as ev


def _iso_replica(monkeypatch, tmp_path):
    """把 run_ab 的落盘/建树/台账面全钉到 tmp，返回 (AB_DIR, evals.jsonl 路径)。"""
    monkeypatch.setattr(replica, "AB_DIR", tmp_path / "ab")
    monkeypatch.setattr(
        replica,
        "build_replica",
        lambda name, ref, *, cases=None, log=None: {
            "name": name,
            "path": str(tmp_path / name),
            "ref": ref,
            "commit": ref.ljust(40, "0")[:40],
            "restored": 0,
        },
    )
    monkeypatch.setattr(replica, "destroy_replica", lambda name, *, log=None: True)
    monkeypatch.setattr(replica, "list_replicas", lambda: [])
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    monkeypatch.setattr(ev, "HARNESS_DIR", harness)
    ledger = tmp_path / "evals.jsonl"
    monkeypatch.setattr(ev, "EVAL_LEDGER_PATH", ledger)
    return tmp_path / "ab", ledger


def _hop_stdout(side: str, turns: int) -> str:
    row = {
        "case": "fig-basic",
        "verdict": "pass",
        "reasons": [],
        "num_turns": turns,
        "duration_ms": 1000,
        "runner": "bencher",
        "label": f"l-{side}",
    }
    return (
        "（副本正文噪声）\nROWS_BEGIN\n"
        + json.dumps(row, ensure_ascii=False)
        + "\nROWS_END\n"
    )


def _ab_item(**kw):
    kw.setdefault("cases", ["fig-basic"])
    return benchd.enqueue("ab", ref_a="OLD", ref_b="NEW", stamp="T1", **kw)


# ===========================================================================
#  1. 空壳闸：零读数 → error + failed（1.3s-done 形状必须绝迹）
# ===========================================================================


def test_ab_hollow_replica_marks_failed(iso_bench, monkeypatch, tmp_path):
    """复刻 run 165247：副本 ModuleNotFoundError → 无读数 → failed 非 done。"""
    _, ledger = _iso_replica(monkeypatch, tmp_path)
    monkeypatch.setattr(
        replica,
        "_replica_call",
        lambda *a, **k: (
            1,
            "",
            "ModuleNotFoundError: No module named 'pysci.skills.bencher'",
        ),
    )
    it = _ab_item()
    res = benchd.run_item(it, tmp_path / "run.log")

    assert res["status"] == "failed"
    assert "空壳" in res["error"] and res["verdict"] == ""
    assert res["evidence"]["ledger_rows"] == 0
    assert not ledger.exists(), "零读数不许写台账行（假装测过）"
    doc = json.loads((tmp_path / "ab" / f"{it['id']}.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "" and doc["error"], "判决文件同样留 error 而非伪判决"
    assert doc["sides"]["a"]["cases"]["fig-basic"]["synthetic"] is True, (
        "占位行必须可辨识——它不是判决证据"
    )


def test_ab_no_cases_marks_failed(iso_bench, monkeypatch, tmp_path):
    """空 cases（HARNESS_DIR 无案）的 ab：无案可跑 → failed，绝不 done。"""
    _iso_replica(monkeypatch, tmp_path)
    monkeypatch.setattr(replica, "_replica_call", lambda *a, **k: (1, "", "空队列"))
    it = benchd.enqueue("ab", cases=[], ref_a="OLD", ref_b="NEW", stamp="T1")
    res = benchd.run_item(it, tmp_path / "run.log")

    assert res["status"] == "failed"
    assert "空壳" in res["error"]


# ===========================================================================
#  2. 正常通路：mock 派发 → 判决行落账 → done 携带判决
# ===========================================================================


def test_ab_full_flow_writes_verdict_rows(iso_bench, monkeypatch, tmp_path):
    """b 端 turns 减半 → 改善；两侧各一行 runner=bencher 入台账，done 有据。"""
    _, ledger = _iso_replica(monkeypatch, tmp_path)

    def fake_call(replica_path, bench_args, *, timeout_s, log=None):
        label = bench_args[bench_args.index("--label") + 1]
        side = label.rsplit("-", 1)[-1]
        return 0, _hop_stdout(side, 10 if side == "a" else 5), ""

    monkeypatch.setattr(replica, "_replica_call", fake_call)
    it = _ab_item(label="l")
    res = benchd.run_item(it, tmp_path / "run.log")

    assert res["status"] == "done" and res["verdict"] == "改善"
    assert res["evidence"]["ledger_rows"] == 2
    assert res["headline"].startswith("改善")
    lines = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and all(r["runner"] == "bencher" for r in lines)
    assert all(r["case"] == "fig-basic" for r in lines)


def test_ab_judge_rejects_verdict_without_ledger(iso_bench, monkeypatch, tmp_path):
    """done 机械判据独立于 run_ab 实现：判决在而台账零行 → failed。"""
    _iso_replica(monkeypatch, tmp_path)
    monkeypatch.setattr(
        replica,
        "run_ab",
        lambda *a, **k: {"verdict": "改善", "label": k["label"], "evidence": {}},
    )
    res = benchd.run_item(_ab_item(), tmp_path / "run.log")
    assert res["status"] == "failed" and "零新增行" in res["error"]


def test_ab_pollution_stays_failed_with_inconclusive(iso_bench, monkeypatch, tmp_path):
    """窗口期污染护栏不回归：inconclusive+留痕 → failed（145147 正确形状）。"""
    _iso_replica(monkeypatch, tmp_path)
    monkeypatch.setattr(
        replica,
        "run_ab",
        lambda *a, **k: {
            "verdict": "inconclusive",
            "error": "窗口期污染",
            "escalate": "twin",
            "label": k["label"],
        },
    )
    res = benchd.run_item(_ab_item(), tmp_path / "run.log")
    assert res["status"] == "failed"
    assert res["headline"] == "窗口期污染" or "escalate" in res["headline"]


# ===========================================================================
#  3. 入口重锚：主树文件直跑 + rc≠0 可见
# ===========================================================================


def test_replica_call_entry_is_main_tree_bench_file(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(replica, "subprocess", types.SimpleNamespace(run=fake_run))
    rc, out, err = replica._replica_call(
        tmp_path, ["_hop", "--cases", "x"], timeout_s=5
    )
    argv = seen["argv"]
    assert rc == 0 and "-m" not in argv, "不再用 -m：入口必须来自主树"
    assert argv[1] == str(replica.BENCH_ENTRY)
    assert argv[1].endswith("bench.py") and replica.BENCH_ENTRY.exists(), (
        "入口=主树 bench.py 文件（pre-E4 ref 也有它）"
    )


def test_replica_call_logs_nonzero_rc(monkeypatch, tmp_path):
    """rc≠0 必须留痕到调用方日志——静默失败正是空壳 done 的温床。"""
    monkeypatch.setattr(
        replica,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(
                returncode=7, stdout="", stderr="boom: 模块缺失"
            )
        ),
    )
    logged: list[str] = []
    rc, _, err = replica._replica_call(
        tmp_path, ["_replica-prep"], timeout_s=5, log=logged.append
    )
    assert rc == 7 and "boom" in err
    assert any("rc=7" in m for m in logged)


# ===========================================================================
#  4. ``_hop`` 对老 ref eval 的协议兼容（能力探测 + runner 分账补齐）
# ===========================================================================


def test_cmd_hop_compat_old_eval_without_runner(monkeypatch, capsys):
    """pre-E4 的 eval 无 RUNNER_BENCHER、run_case 不收 runner：照跑且行带 runner。"""

    class OldEval:
        @staticmethod
        def load_cases(case_id: str = ""):
            return [argparse.Namespace(case_id="fig-basic")]

        @staticmethod
        def run_case(case, *, tier: str = "", label: str = ""):  # 无 runner 形参
            return ev.Row(
                ts="2026-10-12T00:00:00+08:00",
                case=case.case_id,
                member="figure",
                tier="minimum",
                model="",
                verdict="pass",
                reasons=[],
                num_turns=3,
                duration_ms=100,
                max_turns=12,
            )

    monkeypatch.setattr(bench_tool, "_eval", OldEval)
    assert bench_tool.main(["_hop", "--cases", "fig-basic", "--label", "l"]) == 0
    rows = replica.rows_from_stdout(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["runner"] == "bencher" and rows[0]["verdict"] == "pass", (
        "台账分账口径不随被测树年代漂移"
    )


# ===========================================================================
#  5. capture / run 的 done 机械判据
# ===========================================================================


def test_capture_zero_files_marks_failed(iso_bench, monkeypatch, tmp_path):
    monkeypatch.setattr(
        replica,
        "capture_baseline",
        lambda case, *, ref="HEAD", extra_pods=None: {
            "case": case,
            "ref": ref,
            "files": [],
        },
    )
    it = benchd.enqueue("capture", cases=["fig-basic"], stamp="T1")
    res = benchd.run_item(it, tmp_path / "run.log")
    assert res["status"] == "failed" and "归档 0 文件" in res["error"]


def test_capture_with_files_stays_done(iso_bench, monkeypatch, tmp_path):
    monkeypatch.setattr(
        replica,
        "capture_baseline",
        lambda case, *, ref="HEAD", extra_pods=None: {
            "case": case,
            "ref": ref,
            "commit": "abc123",
            "files": [{"path": "orchestration/pods/figure/AGENTS.md", "bytes": 10}],
        },
    )
    it = benchd.enqueue("capture", cases=["fig-basic"], stamp="T1")
    res = benchd.run_item(it, tmp_path / "run.log")
    assert res["status"] == "done"
    assert res["headline"] == "基线已捕获 fig-basic @HEAD"


def test_run_zero_rows_marks_failed(iso_bench, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "load_cases", lambda case_id="": [])
    it = benchd.enqueue("run", cases=[], stamp="T1")  # 全套：load_cases 未拦 → 零行
    res = benchd.run_item(it, tmp_path / "run.log")
    assert res["status"] == "failed" and "零判决行" in res["error"]
