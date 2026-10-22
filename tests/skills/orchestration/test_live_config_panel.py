"""配置实时化 + 队列面板校核 + 空终端根治（用户实测反馈 2026-10-14）。

判据面：
- **生效键**：``[model] leadtime_s`` / ``boundary_hold_s`` 改值即生效、非法回落**出声**，
  且真源标注（``toml 显式`` / ``出厂默认``）跟的是**生效值**的来源（回落后标出厂，
  不许把用户「以为生效」的假象当真）；
- **面板**：倒计时每秒跳动（注入假时钟验证）、边界避让窗口与配置同面板展示；
- **口径**：成员队列与支撑席存量分列；事件水位读数与 ``events-seen.json`` 同帧一致；
- **空终端**：headless 派生与计划任务 action 都用**隐藏宿主**。
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    events,
    routing,
    runner,
    settings,
)
from pysci.skills.orchestration.tools.tui import modules


# ---------------------------------------------------------------------------
# ① 生效键：改值即生效 + 真源标注 + 非法回落告警
# ---------------------------------------------------------------------------
def test_leadtime_effective_key_wins_and_is_labelled():
    fv = routing.face({"model": {"leadtime_s": 900}})
    assert fv.leadtime_s == 900
    assert fv.source("leadtime_s") == "toml 显式"


def test_missing_key_reads_factory_default_and_says_so():
    fv = routing.face({"model": {}})
    assert fv.leadtime_s == routing.DEFAULT_LEADTIME_S
    assert fv.source("leadtime_s") == "出厂默认"


def test_illegal_leadtime_falls_back_and_source_is_factory():
    """回落后的生效值是**出厂值**，真源就该标出厂——否则用户以为自己改成功了。"""
    fv = routing.face({"model": {"leadtime_s": 99999}})
    assert fv.leadtime_s == routing.DEFAULT_LEADTIME_S
    assert fv.source("leadtime_s") == "出厂默认"
    assert any("leadtime_s" in w for w in fv.warnings)


def test_boundary_hold_toml_key_effective_and_capped():
    fv = routing.face({"model": {"boundary_hold_s": 120}})
    assert fv.boundary_hold_s == 120
    assert fv.source("boundary_hold_s") == "toml 显式"
    capped = routing.face({"model": {"boundary_hold_s": 999}})
    assert capped.boundary_hold_s == 300
    assert any("boundary_hold_s" in w and "超上限" in w for w in capped.warnings)


def test_boundary_hold_unset_means_legacy_fallback_sentinel():
    fv = routing.face({"model": {}})
    assert fv.boundary_hold_s is None, "未配置 → 调用方回退 registry 旧位"
    assert fv.source("boundary_hold_s") == "出厂默认"


def test_boundary_hold_registry_fallback_speaks_up(monkeypatch, capsys):
    """toml 未配置 → 回退 registry 旧位**并出声提示迁移**（沿 leadtime 旧位兼容语义）。"""
    monkeypatch.setattr(
        routing,
        "face",
        lambda *a, **k: routing.Face(enabled=False, boundary_hold_s=None),
    )
    reg = SimpleNamespace(routing_hold_s=180)
    line, wait = dispatch._boundary_hold(reg, no_hold=False)
    out = capsys.readouterr().out
    assert "registry 旧位" in out and "data/用户的inbox/pysci_settings.toml" in out
    assert (line, wait) == ("", 0)


def test_boundary_hold_from_toml_makes_no_migration_noise(monkeypatch, capsys):
    monkeypatch.setattr(
        routing,
        "face",
        lambda *a, **k: routing.Face(enabled=False, boundary_hold_s=300),
    )
    dispatch._boundary_hold(SimpleNamespace(routing_hold_s=180), no_hold=False)
    assert capsys.readouterr().out == ""


def test_shipped_settings_file_has_effective_keys():
    """**事故根治的机械哨兵**：真源文件里这两键必须是**生效键**（不许写成 `#` 注释示例）。"""
    from pysci.paths import PROJECT_ROOT

    text = (PROJECT_ROOT / "data" / "用户的inbox" / "pysci_settings.toml").read_text(
        encoding="utf-8"
    )
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip().startswith("leadtime_s")
        or ln.strip().startswith("boundary_hold_s")
    ]
    for key in ("leadtime_s", "boundary_hold_s"):
        assert any(ln.startswith(f"{key} =") for ln in lines), f"{key} 必须是生效键"
    # 三处矛盾统一：prose 注释里的提前分钟数必须与生效值一致（900s = 15 分钟）
    assert "入峰提前 15 分钟" in text
    fv = routing.face({"model": {"leadtime_s": 900}})
    assert f"入峰提前 {fv.leadtime_s // 60} 分钟" in text


# ---------------------------------------------------------------------------
# ② 面板：配置生效值 + 真源 + 每秒倒计时 + 边界避让状态
# ---------------------------------------------------------------------------
def test_countdown_line_ticks_with_fake_clock(monkeypatch):
    monkeypatch.setattr(
        routing, "as_cn", lambda *a, **k: datetime(2026, 10, 14, 12, 30)
    )
    line = modules._countdown_line("2026-10-14 14:00", "入谷")
    assert "距下次切换还有" in line
    # polish-c：时长走人性化格式（天/小时/分/秒）——90 分钟读作「1 小时 30 分 0 秒」
    assert "1 小时 30 分 0 秒" in line
    assert "2026-10-14 14:00" in line and "入谷" in line
    # 跳动：往前 1 秒 → 少 1 秒（每秒重算的语义）
    monkeypatch.setattr(
        routing, "as_cn", lambda *a, **k: datetime(2026, 10, 14, 12, 30, 1)
    )
    assert "1 小时 29 分 59 秒" in modules._countdown_line("2026-10-14 14:00", "入谷")


def test_countdown_line_handles_unfindable_switch():
    assert "判不出" in modules._countdown_line("", "")


def test_system_detail_shows_effective_config_and_sources(tmp_path, monkeypatch):
    toml = tmp_path / "pysci_settings.toml"
    toml.write_text(
        '[model]\npreset = "mimo"\nleadtime_s = 900\nboundary_hold_s = 180\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "SETTINGS_PATH", toml)
    lines = modules._system_detail()
    joined = "\n".join(lines)
    assert "入峰提前量 leadtime_s = 900s" in joined
    assert "〔真源：toml 显式〕" in joined
    assert "边界避让：窗口 180s" in joined
    assert "倒计时：距下次切换还有" in joined


def test_system_detail_label_factory_when_key_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "none.toml")
    lines = modules._system_detail()
    assert any("leadtime_s = 3600s" in ln for ln in lines)
    assert any("〔真源：出厂默认〕" in ln for ln in lines)


def test_countdown_excluded_from_badge_fingerprint(monkeypatch):
    """倒计时行**不进徽标指纹**——否则每秒一次变化会让徽标常亮（§2.2）。"""
    tick = ["  倒计时：距下次切换还有 90 分 0 秒（x → 入谷）"]
    monkeypatch.setattr(modules, "_system_detail", lambda: tick)
    monkeypatch.setattr(modules, "_system_summary", lambda: ["s"])
    mod = modules._fingerprint_module(
        "system", "系统与路由", modules._system_summary, modules._system_stable
    )
    first = mod.cursor()
    tick[0] = "  倒计时：距下次切换还有 89 分 59 秒（x → 入谷）"
    assert mod.cursor() == first, "倒计时跳动不该改变徽标水位"
    assert mod.updates_since(first) == 0


# ---------------------------------------------------------------------------
# ③ 队列面板口径拆分 + 事件水位真源一致性
# ---------------------------------------------------------------------------
def test_queues_detail_splits_member_and_support_seat(monkeypatch):
    from pysci.skills.orchestration.tools import inspectd

    monkeypatch.setattr(inspectd, "enabled", lambda: False)
    lines = modules._queues_detail()
    joined = "\n".join(lines)
    assert "支撑席存量" in joined and "成员队列皆空 ≠ 支撑席也空" in joined
    assert "7 专职 + deputy + reviewer" in joined
    assert "口径：9 成员队列 pending=" in joined
    assert "不拉起巡检" in joined and "只报不告警" in joined


def test_water_lines_empty_states_scope(monkeypatch):
    from pysci.skills.orchestration.tools import queues

    monkeypatch.setattr(queues, "QUEUES_DIR", Path("no-such-dir"))
    out = queues.water_lines()
    joined = "\n".join(out)
    assert "pending=0 in_progress=0" in joined, "在途合计读数必须在第一行"
    assert "双零" in joined and "支撑席" in joined


def test_event_watermark_reflects_ack_seq_same_frame(iso_events):
    """用户曾见「已排空至 0 待排空 3」的滞后快照——排空后读数必须**同帧**反映 ack_seq。"""
    events.emit(events.JOB_DONE, "figure-1", "收尾", "advance")
    events.emit(events.QUEUE_EMPTY, "run-1", "消化完", "drain")
    assert any("待排空 2" in ln for ln in events.status_lines())
    assert events.drain()
    lines = events.status_lines()
    assert any("待排空 0" in ln for ln in lines), lines
    assert any(f"已排空至 {events.latest_seq()}" in ln for ln in lines), lines


# ---------------------------------------------------------------------------
# ⑤ 空终端根治：headless 派生隐藏窗口 + 计划任务隐藏宿主
# ---------------------------------------------------------------------------
def test_headless_spawn_hides_window_on_windows(monkeypatch):
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="stop here"):
        runner.run_headless(["echo", "x"], cwd=Path("."), env={}, timeout_s=1)
    if sys.platform == "win32":
        assert captured.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW, (
            "headless 派生必须带 CREATE_NO_WINDOW——否则由分离进程派起时会开空终端"
        )


def test_scheduled_task_action_uses_hidden_host():
    ps1 = (
        Path(__file__).resolve().parents[3]
        / "orchestration"
        / "scripts"
        / "watchdog-scheduler.ps1"
    )
    text = ps1.read_text(encoding="utf-8")
    assert "pythonw.exe" in text, "计划任务 action 须换隐藏宿主（pythonw）"
    assert "empty-terminal" in text or "no console window" in text
