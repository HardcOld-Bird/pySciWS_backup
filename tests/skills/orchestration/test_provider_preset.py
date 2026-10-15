"""提供商预设三面契约（backlog 20261012-225500-leader：档位退役，改 qwen/mimo/deepseek 预设）。

success_criteria 落点：
- 三预设切换 = 改 ``data/pysci_settings.toml`` **一行** → 席位表/status 换血 → 派发取到
  该预设的席（台账 model 字段即证据）；pro 席成员按预设名单取 pro 席；
- 仲裁团构成随预设换血且排除同族（家族名面板，配置核验 + 读数行证据）；
- deepseek 预设**已回填激活**（BYOK UUID 首席与内置同名 → UUID 消歧；激活闸=席表非空：
  整链覆盖为空仍拒发/拒开工/status 出声，不静默回落）；
- minimum 独立链不随预设走；#80 粘滞降级在预设链上照常（机制不推翻）；
- inspector on/off 改写 toml 且 status 读数一致。

额度路径与 #80 机制面的既有覆盖仍在 ``test_quota_awareness``/``test_model_fallback_chain``；
本组只钉**预设这个新决策面**。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import (
    dispatch,
    drain,
    inspectd,
    jury,
    orch,
    presets,
    registry,
    settings,
)

from .conftest import (
    cmd_model,
    envelope,
    fake_headless,
    read_backlog,
    seed_backlog,
    set_jury_panel,
    set_models,
    set_presets,
    write_preset,
)

A = "seat-a"
B = "seat-b"
P1 = "seat-pro"
BUILTIN = "Qwen3.8-Flash"
#: deepseek 回填（用户 2026-10-13 供 Key）：BYOK 与内置同名 "DeepSeek-Flash" → UUID 首席消歧。
DS_UUID = "1a54cc57-e614-4521-80cc-447de31dc896"
#: 与 BYOK **同名**的内置回退席（正主，别拿 Qwen 的 BUILTIN 顶——两席同名才是要消歧的点）。
DS_BUILTIN = "DeepSeek-Flash"
OK_BODY = "<result>完工，commit abc1234 已合并。</result>"
QUOTA_TEXT = (
    "You've reached your credit usage limit. Please upgrade your "
    "subscription plan to get more resources. Report Issue (input /feedback)"
)


# ---------------------------------------------------------------------------
# ① 真源读取：toml 的 preset 名（精确串 / fail-open / 警告）
# ---------------------------------------------------------------------------
def test_settings_preset_exact_string(iso_state):
    assert settings.preset_name() == ("mimo", [])


def test_settings_unknown_preset_falls_back_with_warning(iso_state):
    """非法串（大小写敏感）→ 出厂预设 + 一条指路警告，不静默吞掉。"""
    write_preset("QWEN")
    name, warnings = settings.preset_name()
    assert name == "mimo" and warnings and "不是合法预设" in warnings[0]
    assert "qwen/mimo/deepseek" in warnings[0]


def test_settings_missing_file_fails_open(iso_state):
    settings.SETTINGS_PATH.unlink()
    name, warnings = settings.preset_name()
    assert name == "mimo" and warnings and "未配置" in warnings[0]


def test_inspector_flag_reads_toml_and_false_forms(iso_state):
    assert settings.inspector_flag() is True
    settings.SETTINGS_PATH.write_text(
        '[model]\npreset = "mimo"\n\n[inspector]\nenabled = false\n',
        encoding="utf-8",
    )
    assert settings.inspector_flag() is False
    settings.SETTINGS_PATH.unlink()
    assert settings.inspector_flag() is None, "toml 缺席 → None（回落 legacy 文件）"


# ---------------------------------------------------------------------------
# ② 预设表与角色解析
# ---------------------------------------------------------------------------
def test_three_preset_seat_tables():
    qwen = presets.PRESETS["qwen"]
    mimo = presets.PRESETS["mimo"]
    deepseek = presets.PRESETS["deepseek"]
    assert qwen.main == ("Qwen-3.8-Flash",)
    assert qwen.pro == ("Qwen-3.8-Max",) and qwen.pro_members == ("twin",)
    assert mimo.main == ("Mimo-V2.6-Flash",)
    assert mimo.pro == ("Mimo-V2.6-Pro",)
    assert mimo.pro_members == ("reviewer", "deputy", "inspector", "twin")
    assert deepseek.main == (DS_UUID,), (
        "用户明令 2026-10-15：仅 BYOK 席，禁一切 Qoder 内置渠道"
    )
    assert deepseek.pro == (), "deepseek 全员同档无 pro 席（用户设计）"
    assert all(
        BUILTIN not in chain and DS_BUILTIN not in chain
        for chain in (qwen.main, qwen.pro, mimo.main, mimo.pro, deepseek.main)
    ), "三预设链内不得出现任何内置席（用户明令 2026-10-15：BYOK 失败即告警不落内置）"
    assert qwen.jury == ("glm", "kimi", "mimo") and mimo.jury == ("glm", "kimi", "qwen")
    assert deepseek.jury == ("glm", "kimi", "mimo"), "三预设都排除预设同族"


def test_resolve_role_preflist_and_override(iso_state):
    # mimo 预设（默认）：pro 名单四人组取 pro，其余 main
    assert presets.resolve_role("reviewer") == "pro"
    assert presets.resolve_role("twin") == "pro"
    assert presets.resolve_role("devops") == "main"
    # 改一行 → qwen 预设：名单缩到仅 twin
    write_preset("qwen")
    assert presets.resolve_role("reviewer") == "main"
    assert presets.resolve_role("twin") == "pro"
    # 覆盖优先于名单（--model-tier / 案卷 tier:）
    assert presets.resolve_role("devops", "pro") == "pro"
    assert presets.resolve_role("reviewer", "main") == "main"


def test_legacy_tier_names_normalize_but_bogus_rejected(iso_state):
    assert presets.normalize_role("flash") == "main"
    assert presets.normalize_role("max") == "pro"
    assert presets.normalize_role("minimum") == "minimum"
    assert presets.normalize_role("bogus") == "bogus"  # 交 argparse choices 判 rc=2
    try:
        presets.resolve_role("devops", "bogus")
    except ValueError as exc:
        assert "合法值" in str(exc)
    else:
        raise AssertionError("非法覆盖必须硬拦，不许静默回落烧错档")


def test_status_lines_show_preset_and_chains(iso_registry):
    reg = registry.Registry.load()
    out = "\n".join(presets.status_lines(reg))
    assert "预设 mimo" in out and "data/pysci_settings.toml" in out
    assert "Mimo-V2.6-Flash" in out and BUILTIN not in out, (
        "各席解析值要给人看；内置席被硬过滤不得出现（用户明令 2026-10-15）"
    )
    assert "仲裁团构成：glm+kimi+qwen" in out
    assert "minimum（独立，不随预设走）" in out
    assert "未激活" not in out


def test_status_lines_deepseek_activated(iso_state, iso_registry):
    """deepseek 出声面：席链读数含 BYOK UUID、**不含任何内置席**，无「未激活」。"""
    write_preset("deepseek")
    reg = registry.Registry.load()
    out = "\n".join(presets.status_lines(reg))
    assert "预设 deepseek" in out and "未激活" not in out
    assert DS_UUID in out and DS_BUILTIN not in out, (
        "用户明令 2026-10-15：禁内置 DeepSeek——读数不得出现内置席"
    )


def test_status_lines_warn_when_seats_emptied(iso_state, iso_dispatch):
    """整链覆盖为空仍=未配置：status 出声（与 preset_gate 同一判据）。"""
    set_presets(iso_dispatch, deepseek={"main": [], "pro": []})
    write_preset("deepseek")
    reg = registry.Registry.load()
    out = "\n".join(presets.status_lines(reg))
    assert "[!] 预设 deepseek 未激活" in out and "席表为空" in out


# ---------------------------------------------------------------------------
# ③ 派发面：改一行换席 / 角色覆盖 / 预设不可用拒发
# ---------------------------------------------------------------------------
def _seed_dispatch_presets(state):
    set_presets(
        state,
        mimo={"main": [A, B], "pro": [P1, B]},
        qwen={"main": [B, A], "pro": [P1, A]},
        deepseek={"main": [], "pro": []},
    )


def test_switching_preset_changes_dispatch_seat(iso_dispatch, monkeypatch):
    """改一行 toml → 下一次派发换席（success_criteria 的切换实测面）。"""
    _seed_dispatch_presets(iso_dispatch)
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)
    assert o.kind == "result" and cmd_model(calls[0]) == A

    write_preset("qwen")  # ← 唯一改动：一行
    o = dispatch.do_dispatch("quotamember", text="再出图", quiet=True, no_checks=True)
    assert o.kind == "result" and cmd_model(calls[1]) == B, "预设换了，席必须跟着换"


def test_minimum_chain_does_not_follow_preset(iso_dispatch, monkeypatch):
    _seed_dispatch_presets(iso_dispatch)
    set_models(iso_dispatch, minimum="min-seat")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    for preset in ("mimo", "qwen"):
        write_preset(preset)
        dispatch.do_dispatch(
            "quotamember",
            text="压测",
            quiet=True,
            no_checks=True,
            model_tier="minimum",
        )
    assert [cmd_model(c) for c in calls] == ["min-seat", "min-seat"], (
        "minimum 独立链不随预设走"
    )


def test_role_override_selects_pro_chain(iso_dispatch, monkeypatch):
    _seed_dispatch_presets(iso_dispatch)
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    dispatch.do_dispatch(
        "quotamember", text="升档", quiet=True, no_checks=True, model_tier="max"
    )
    assert cmd_model(calls[0]) == P1, "旧档名 max 归一到 pro 席"
    dispatch.do_dispatch(
        "quotamember", text="回主档", quiet=True, no_checks=True, model_tier="main"
    )
    assert cmd_model(calls[1]) == A


def test_unknown_role_refuses_without_dispatch(iso_dispatch, monkeypatch):
    _seed_dispatch_presets(iso_dispatch)
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(OK_BODY, rc=0))
    o = dispatch.do_dispatch(
        "quotamember", text="x", quiet=True, no_checks=True, model_tier="bogus"
    )
    assert o.code == 2 and o.kind == "run_failed"
    assert calls == [], "非法角色在派发前硬拦，不白烧一跳"


def test_deepseek_preset_refuses_dispatch_when_seats_emptied(
    iso_dispatch, monkeypatch, capsys
):
    """整链覆盖为空 → 拒绝激活：不派发、不落台账、原因可读（不静默换别的预设）。"""
    _seed_dispatch_presets(iso_dispatch)
    write_preset("deepseek")
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(OK_BODY, rc=0))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=False, no_checks=True)
    out = capsys.readouterr().out
    assert o.code == 2 and o.kind == "run_failed"
    assert "未激活" in o.error and "席表为空" in o.error
    assert "未激活" in out and "pysci_settings.toml" in out
    assert calls == []
    assert not (iso_dispatch / "ledger.jsonl").exists(), "拒发不写无意义台账行"


def test_deepseek_activation_uses_uuid_first_for_same_name(iso_dispatch, monkeypatch):
    """回填激活面：registry presets 块在但不含 deepseek 键 → 走代码出厂表；
    BYOK 与内置同名，派发必须取 UUID 首席（消歧），status 席链两席俱在。"""
    set_presets(iso_dispatch, mimo={"main": ["M1", BUILTIN], "pro": ["P1", BUILTIN]})
    write_preset("deepseek")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    o = dispatch.do_dispatch("quotamember", text="自检", quiet=True, no_checks=True)
    assert o.kind == "result" and cmd_model(calls[0]) == DS_UUID, (
        "同名消歧=UUID 首席命中"
    )
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8")
    assert f'"model": "{DS_UUID}"' in line, (
        "台账 model 记 UUID 读数（success_criteria 那条）"
    )
    out = "\n".join(presets.status_lines(registry.Registry.load()))
    assert DS_UUID in out and DS_BUILTIN not in out and "未激活" not in out


def test_deepseek_byok_failure_fails_loud_no_builtin_fallback(
    iso_dispatch, monkeypatch
):
    """用户明令 2026-10-15：BYOK 失败**直接失败告警**，绝不回退到 Qoder 内置渠道
    （内置扣 Qoder 积分且用户额度已耗尽——2026-10-15 事故根因之一）。"""
    set_presets(iso_dispatch, mimo={"main": ["M1", BUILTIN], "pro": ["P1", BUILTIN]})
    write_preset("deepseek")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(QUOTA_TEXT),
    )
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)
    assert o.kind != "result", "BYOK 不可用必须失败，不得静默降级"
    assert [cmd_model(c) for c in calls] == [DS_UUID], (
        "只打 BYOK 席——任何情况下不得尝试内置席（含同名 DeepSeek-Flash）"
    )


def test_preset_gate_seats_nonempty_rule(iso_dispatch):
    """激活闸口径（dispatch/drain 共读）：生效链非空=放行；整链覆盖为空=拒绝；minimum 恒放行。

    判据读的是**合并后的生效链**而非出厂表：跟踪面把 deepseek 整链置空，照样拒——
    否则「代码表已回填」会把配置面的清空事故盖过去。
    """
    _seed_dispatch_presets(iso_dispatch)  # mimo/qwen 有席，deepseek 两链皆空
    write_preset("mimo")
    assert presets.preset_gate(registry.Registry.load(), "main") == "", "有席即放行"
    write_preset("deepseek")
    reason = presets.preset_gate(registry.Registry.load(), "main")
    assert "deepseek" in reason and "席表为空" in reason
    assert presets.preset_gate(registry.Registry.load(), "minimum") == "", (
        "minimum 独立链不随预设走"
    )


def test_gate_no_pro_seat_is_design_not_fault(iso_dispatch):
    """回填后的 deepseek：main 有席、pro 无席（用户刻意「全员同档」）。

    显式 ``--model-tier pro`` 仍该拒（不静默回落烧到别的档），但读数必须说「本预设无该档席」
    而不是「预设未激活」——把设计报成故障会把处置指到「去回填席位」上去。
    """
    set_presets(
        iso_dispatch, qwen={"main": ["Q1"], "pro": ["Q2"]}
    )  # 无 deepseek 键=用出厂表
    write_preset("deepseek")
    reg = registry.Registry.load()
    assert presets.preset_gate(reg, "main") == "", "回填后 main 席放行（激活成功）"
    pro = presets.preset_gate(reg, "pro")
    assert pro and "未激活" not in pro and "刻意无该档席" in pro


def test_drain_refuses_start_under_empty_seats(iso_state, iso_dispatch, monkeypatch):
    """drain 在**拿锁之前**拒开工：整队列不被标成失败，错误读数淹没不了处置动作。

    场景用「跟踪面把 deepseek 整链置空」构造（回填后出厂表本非空，只有覆盖能置空）——
    与派发面共读 :func:`presets.preset_gate`，两侧判据不许分叉。
    """
    seed_backlog(
        iso_state,
        [
            {"id": "d1", "status": "pending", "summary": "s", "member": "devops"},
            {"id": "d2", "status": "pending", "summary": "s", "member": "devops"},
        ],
    )
    set_presets(iso_dispatch, deepseek={"main": [], "pro": []})
    write_preset("deepseek")
    dispatched = []
    monkeypatch.setattr(dispatch, "do_dispatch", lambda *a, **k: dispatched.append(a))
    summary = drain.drain_backlog(drain.new_run_log())
    assert summary.get("skipped") == "preset_unavailable"
    assert "未激活" in summary.get("error", "")
    assert dispatched == []
    assert not drain.LOCK_PATH.exists(), "拒开工发生在拿锁之前"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert all(i["status"] == "pending" for i in by_id.values()), "条目无罪不连坐"


# ---------------------------------------------------------------------------
# ④ #80 粘滞降级在预设链上照常（机制复用，不推翻）
# ---------------------------------------------------------------------------
def test_preset_chain_sticky_demotion_and_alert(
    iso_state, iso_dispatch, monkeypatch, capsys
):
    _seed_dispatch_presets(iso_dispatch)
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope(QUOTA_TEXT)
            if n == 1
            else envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)
        ),
    )
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=False, no_checks=True)
    out = capsys.readouterr().out
    assert o.kind == "result" and len(calls) == 2
    assert cmd_model(calls[0]) == A and cmd_model(calls[1]) == B, (
        "预设链的第二席=下一 BYOK 席（跳过故障席→换 BYOK；内置席已硬过滤，用户明令 2026-10-15）"
    )
    assert "[!] 档位回退：main 席seat-a不可用" in out
    events = [
        json.loads(line)
        for line in (iso_dispatch / "channel-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert events and events[0]["tier"] == "main" and events[0]["to"] == B
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8")
    assert B in line, "台账 model 记实际所用席位（BYOK 席，内置已过滤）"


# ---------------------------------------------------------------------------
# ⑤ 仲裁团构成随预设换血（家族面板）
# ---------------------------------------------------------------------------
def _family_panel(iso_jury, monkeypatch, preset: str, seats):
    write_preset(preset)
    set_jury_panel(iso_jury.parent / "registry.json", seats)
    return jury.seat_readiness()


def test_jury_panel_swaps_with_preset(iso_jury, monkeypatch):
    seats = [
        {
            "name": "deepseek",
            "model": "deepseek-v4-pro",
            "base_url": "https://ds.invalid/v1",
        },
        {"name": "glm", "model": "glm-5.3", "base_url": "https://glm.invalid/v1"},
        {"name": "kimi", "model": "kimi-k2.6", "base_url": "https://kimi.invalid/v1"},
        {
            "name": "mimo",
            "model": "Mimo-V2.6-Pro",
            "base_url": "https://mimo.invalid/v1",
        },
    ]
    mimo_plan = _family_panel(iso_jury, monkeypatch, "mimo", seats)
    assert mimo_plan["panel"] == ["glm", "kimi", "qwen"]
    assert [s.name for s in mimo_plan["out_of_panel"]] == ["deepseek", "mimo"], (
        "mimo 预设下同族 mimo 席与 deepseek 席都不入面板（换血证据）"
    )
    assert mimo_plan["missing_panel"] == ["qwen"], "面板缺席要点名让用户补配置"

    qwen_plan = _family_panel(iso_jury, monkeypatch, "qwen", seats)
    assert qwen_plan["panel"] == ["glm", "kimi", "mimo"]
    assert [s.name for s in qwen_plan["out_of_panel"]] == ["deepseek"]
    assert [s.name for s in qwen_plan["seats"] if s.role == "primary"] == [
        "glm",
        "kimi",
        "mimo",
    ]
    assert qwen_plan["missing_panel"] == []


def test_jury_summary_reports_panel_swap(iso_jury, monkeypatch):
    seats = [
        {
            "name": "deepseek",
            "model": "deepseek-v4-pro",
            "base_url": "https://ds.invalid/v1",
        },
        {"name": "glm", "model": "glm-5.3", "base_url": "https://glm.invalid/v1"},
        {"name": "kimi", "model": "kimi-k2.6", "base_url": "https://kimi.invalid/v1"},
    ]
    plan = _family_panel(iso_jury, monkeypatch, "mimo", seats)
    line = jury.readiness_summary(plan)
    assert "面板 glm+kimi+qwen（预设 mimo）" in line
    assert "预设外未入面板：deepseek" in line
    assert "面板缺席：qwen" in line, "缺席家族必须出声，否则庭里少一席没人知道"


# ---------------------------------------------------------------------------
# ⑥ inspector 开关真源迁移（toml 外科式改写，注释保全）
# ---------------------------------------------------------------------------
def test_inspector_off_on_rewrites_toml_preserving_comments(iso_state):
    before = settings.SETTINGS_PATH.read_text(encoding="utf-8")
    assert inspectd.set_enabled(False) == "false"
    assert settings.inspector_flag() is False and inspectd.enabled() is False
    after = settings.SETTINGS_PATH.read_text(encoding="utf-8")
    assert 'preset = "mimo"' in after, "[model] 段与注释不得被 round-trip 吃掉"
    assert "false" in after
    assert inspectd.set_enabled(True) == "true" and inspectd.enabled() is True


def test_inspector_enabled_falls_back_to_legacy_when_toml_absent(iso_state):
    settings.SETTINGS_PATH.unlink()
    state_file = inspectd.ENABLE_PATH
    state_file.write_text("false\n", encoding="utf-8")
    assert inspectd.enabled() is False, "toml 缺席才回落 legacy 文件"
    state_file.write_text("true\n", encoding="utf-8")
    assert inspectd.enabled() is True


def test_inspector_cli_switch_writes_toml(iso_state, capsys):
    assert orch.main(["inspector-off"]) == 0
    assert settings.inspector_flag() is False
    assert "开关=false" in capsys.readouterr().out
    assert orch.main(["inspector-on"]) == 0
    assert settings.inspector_flag() is True
