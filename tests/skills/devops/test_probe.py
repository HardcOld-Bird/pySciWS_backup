"""agentsMdExcludes 机械探针（probe.py）确定性回归（backlog 20261010-165408-devops）。

只测**不需真跑 headless** 的部分：token 解析、staleness 键、因果 A/B 的裁决逻辑
（喂 canned 观测覆盖 ok / exclusion_failed / inconclusive 三分支）、台账写读与
doctor 分级。真 CLI 探针是集成面，由 `pysci-dev probe` 手工跑、结论落台账。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.devops.tools import probe


def test_token_re_grabs_only_unguessable_tokens():
    text = "TOK-0123456789abcdef 和 TOK-fedcba9876543210 是标记；TOK-XYZ 不是，token 也不是 secret。"
    got = probe.TOKEN_RE.findall(text)
    assert got == ["TOK-0123456789abcdef", "TOK-fedcba9876543210"]


def test_front_rule_carries_always_on_and_token():
    body = probe._front_rule("TOK-abcdef0123456789", "应可见")
    assert "trigger: always_on" in body and "TOK-abcdef0123456789" in body


def test_pod_exclude_patterns_normalizes_to_basenames(tmp_path, monkeypatch):
    pods = tmp_path / "pods"
    for name, excl in (
        ("lit", ["**/leader-only.md"]),
        ("reviewer", ["rules/leader-only.md"]),
        ("deputy", ["leader-only.md", "**/other.md"]),
        ("broken", []),
    ):
        q = pods / name / ".qoder"
        q.mkdir(parents=True)
        if name == "broken":
            (q / "settings.json").write_text("{ not json", encoding="utf-8")
        else:
            (q / "settings.json").write_text(
                json.dumps({"agentsMdExcludes": excl}), encoding="utf-8"
            )
    monkeypatch.setattr(probe, "PODS_ROOT", pods)
    # 三种 leader-only 形态归一化到同一 basename，去重排序后与 other.md 并列
    assert probe.pod_exclude_patterns() == ["leader-only.md", "other.md"]


def test_pod_exclude_patterns_empty_when_no_pods(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "PODS_ROOT", tmp_path / "ghost")
    assert probe.pod_exclude_patterns() == []


def test_config_hash_deterministic_and_sensitive(monkeypatch):
    monkeypatch.setattr(probe, "cli_version", lambda: "CLI-1.2.3")
    monkeypatch.setattr(probe, "pod_exclude_patterns", lambda: ["leader-only.md"])
    h1 = probe.config_hash()
    assert h1 == probe.config_hash()  # 同配置 → 稳定
    monkeypatch.setattr(
        probe, "pod_exclude_patterns", lambda: ["leader-only.md", "x.md"]
    )
    assert probe.config_hash() != h1  # 排除面变 → 哈希变
    monkeypatch.setattr(probe, "pod_exclude_patterns", lambda: ["leader-only.md"])
    monkeypatch.setattr(probe, "cli_version", lambda: "CLI-9.9.9")
    assert probe.config_hash() != h1  # CLI 升级 → 哈希变


# ---------------------------------------------------------------------------
# 因果 A/B 裁决逻辑（canned 观测，不跑 CLI）
# ---------------------------------------------------------------------------
# build_fixture 内部固定的两枚 token（hide 恒为 b*16）；测试据此构造 canned 复述集。
KEEP = "TOK-" + "a" * 16
HIDE = "TOK-" + "b" * 16


def _fake_probe(monkeypatch, tmp_path, control_tokens, withexclude_tokens):
    """把 build_fixture/_run_session 换成 canned 观测（按 pod 路径区分 control/withexclude）。"""
    root = tmp_path / "fx"
    wpath, cpath = root / "withexclude", root / "control"
    monkeypatch.setattr(
        probe, "build_fixture", lambda: (root, KEEP, HIDE, wpath, cpath)
    )

    def fake_run(pod_cwd):
        toks = control_tokens if pod_cwd == cpath else withexclude_tokens
        return {
            "tokens": set(toks),
            "text": " ".join(toks) or "NONE",
            "ratio": 0.2,
            "turns": 1,
            "rc": 0,
        }

    monkeypatch.setattr(probe, "_run_session", fake_run)


def test_verdict_ok_when_control_sensitive_and_withexclude_hides(monkeypatch, tmp_path):
    # control 复述 keep+hide（灵敏）；withexclude 只见 keep（hide 被排除生效）
    _fake_probe(monkeypatch, tmp_path, [KEEP, HIDE], [KEEP])
    v = probe.probe_mechanism()
    assert v["status"] == "ok" and v["mechanism_ok"] is True
    assert v["sensitive"] is True and v["hide_leaked_in_withexclude"] is False
    assert v["keep_injected"] is True  # 未被排除的 keep 仍在场 → 非「全盘没注入」


def test_verdict_exclusion_failed_when_hide_leaks(monkeypatch, tmp_path):
    # withexclude 仍见 hide → 排除没生效，真越权回归
    _fake_probe(monkeypatch, tmp_path, [KEEP, HIDE], [KEEP, HIDE])
    v = probe.probe_mechanism()
    assert v["status"] == "exclusion_failed" and v["mechanism_ok"] is False
    assert v["hide_leaked_in_withexclude"] is True


def test_verdict_inconclusive_when_control_not_sensitive(monkeypatch, tmp_path):
    # control 都没复述 hide → 探针不灵敏，绝不误判为绿
    _fake_probe(monkeypatch, tmp_path, [KEEP], [])
    v = probe.probe_mechanism(retries=2)
    assert v["status"] == "inconclusive" and v["mechanism_ok"] is False
    assert v["sensitive"] is False


def test_inconclusive_retries_until_sensitive(monkeypatch, tmp_path):
    # 首跑模型没复述，重试第二次复述 → 应得 ok（不因单次抖动误判）
    root = tmp_path / "fx"
    wpath, cpath = root / "withexclude", root / "control"
    monkeypatch.setattr(
        probe, "build_fixture", lambda: (root, KEEP, HIDE, wpath, cpath)
    )
    calls = {"control": 0}

    def fake_run(pod_cwd):
        if pod_cwd == cpath:
            calls["control"] += 1
            toks = [KEEP, HIDE] if calls["control"] >= 2 else [KEEP]
        else:
            toks = [KEEP]
        return {
            "tokens": set(toks),
            "text": " ".join(toks),
            "ratio": 0.2,
            "turns": 1,
            "rc": 0,
        }

    monkeypatch.setattr(probe, "_run_session", fake_run)
    v = probe.probe_mechanism(retries=3)
    assert v["status"] == "ok" and calls["control"] == 2


# ---------------------------------------------------------------------------
# 台账写读 + doctor 分级
# ---------------------------------------------------------------------------
@pytest.fixture
def ledger_at(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(probe, "LEDGER", path)
    monkeypatch.setattr(probe, "cli_version", lambda: "CLI-1.0.0")
    monkeypatch.setattr(probe, "pod_exclude_patterns", lambda: ["leader-only.md"])
    return path


def test_write_read_ledger_roundtrip_deterministic(ledger_at):
    rec = probe.write_ledger(
        {
            "status": "ok",
            "mechanism_ok": True,
            "sensitive": True,
            "keep_injected": True,
            "hide_leaked_in_withexclude": False,
            "model": probe.PROBE_MODEL,
            "control_ratio": 0.3,
            "withexclude_ratio": 0.2,
        }
    )
    assert rec["schema"] == 1 and rec["mechanism_ok"] is True
    assert rec["config_hash"] == probe.config_hash()
    assert "ts" not in rec and "time" not in rec  # 无时间戳 → 跨合并零漂移
    assert probe.read_ledger() == rec
    ledger_at.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    assert probe.read_ledger() == rec  # 落盘再读一致


def test_read_ledger_none_when_missing_or_broken(ledger_at):
    assert probe.read_ledger() is None  # 文件不存在
    ledger_at.write_text("{ nope", encoding="utf-8")
    assert probe.read_ledger() is None  # 非法 JSON
    ledger_at.write_text("[1, 2]", encoding="utf-8")
    assert probe.read_ledger() is None  # 非 dict


def test_doctor_status_advisory_when_never_run(ledger_at):
    level, msg = probe.doctor_status()
    assert level == "advisory" and "未跑过" in msg


def test_doctor_status_fail_when_mechanism_broken(ledger_at):
    probe.write_ledger(
        {
            "status": "exclusion_failed",
            "mechanism_ok": False,
            "sensitive": True,
            "keep_injected": True,
            "hide_leaked_in_withexclude": True,
            "model": probe.PROBE_MODEL,
            "control_ratio": 0.3,
            "withexclude_ratio": 0.3,
        }
    )
    level, msg = probe.doctor_status()
    assert level == "fail" and "exclusion_failed" in msg


def test_doctor_status_advisory_when_hash_stale(ledger_at):
    rec = probe.write_ledger(
        {
            "status": "ok",
            "mechanism_ok": True,
            "sensitive": True,
            "keep_injected": True,
            "hide_leaked_in_withexclude": False,
            "model": probe.PROBE_MODEL,
            "control_ratio": 0.3,
            "withexclude_ratio": 0.2,
        }
    )
    # 台账机制成立，但排除面随后变化 → staleness advisory（不硬失败）
    probe.pod_exclude_patterns = lambda: ["leader-only.md", "extra.md"]
    level, msg = probe.doctor_status()
    assert level == "advisory" and "过期" in msg
    assert rec["mechanism_ok"] is True


def test_doctor_status_ok_when_fresh(ledger_at):
    probe.write_ledger(
        {
            "status": "ok",
            "mechanism_ok": True,
            "sensitive": True,
            "keep_injected": True,
            "hide_leaked_in_withexclude": False,
            "model": probe.PROBE_MODEL,
            "control_ratio": 0.3,
            "withexclude_ratio": 0.2,
        }
    )
    level, msg = probe.doctor_status()
    assert level == "ok" and "实测确认" in msg


# ---------------------------------------------------------------------------
# main() 返回码契约
# ---------------------------------------------------------------------------
def _verdict(status, mechanism_ok):
    return {
        "status": status,
        "mechanism_ok": mechanism_ok,
        "sensitive": True,
        "keep_injected": True,
        "hide_leaked_in_withexclude": not mechanism_ok,
        "model": probe.PROBE_MODEL,
        "control_ratio": 0.3,
        "withexclude_ratio": 0.2,
    }


def test_main_rc0_when_mechanism_ok(monkeypatch):
    monkeypatch.setattr(probe, "probe_mechanism", lambda *a, **k: _verdict("ok", True))
    monkeypatch.setattr(probe, "write_ledger", lambda v: {"config_hash": "x"})
    assert probe.main(["--json"]) == 0


@pytest.mark.parametrize("status", ["exclusion_failed", "inconclusive"])
def test_main_rc2_when_mechanism_not_ok(monkeypatch, status):
    monkeypatch.setattr(
        probe, "probe_mechanism", lambda *a, **k: _verdict(status, False)
    )
    monkeypatch.setattr(probe, "write_ledger", lambda v: {"config_hash": "x"})
    assert probe.main(["--json"]) == 2


def test_main_no_ledger_skips_write(monkeypatch):
    monkeypatch.setattr(probe, "probe_mechanism", lambda *a, **k: _verdict("ok", True))
    called = {"w": 0}
    monkeypatch.setattr(
        probe, "write_ledger", lambda v: called.__setitem__("w", called["w"] + 1)
    )
    assert probe.main(["--no-ledger", "--json"]) == 0
    assert called["w"] == 0  # --no-ledger 绝不落台账
