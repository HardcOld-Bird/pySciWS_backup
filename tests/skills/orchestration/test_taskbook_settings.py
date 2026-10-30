"""``[taskbooks]`` 设置节读数：执行模式（含热切换保证）与 TUI 表单默认值。

用户 2026-10-17 裁决：execution_mode 支持**热切换**（手改 toml 保存即下一回合生效）——
本组用例钉住该保证的根基：读数每调用现算、无任何缓存。
"""

from __future__ import annotations

import pytest

from pysci.skills.orchestration.tools import settings


@pytest.fixture
def iso_settings(tmp_path, monkeypatch):
    """把 ``SETTINGS_PATH`` 指向 tmp（不读开发者真源 toml）。"""
    path = tmp_path / "pysci_settings.toml"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    return path


def test_factory_defaults_are_pinned():
    """出厂常量钉死在**代码**里（用户手改 toml 不得漂移它）。"""
    form = settings.DEFAULT_TASKBOOK_FORM
    assert (
        form.new_priority,
        form.new_leader,
        form.reject_priority,
        form.reject_infra,
        form.reject_leader,
        form.reject_inspector,
    ) == ("normal", False, "normal", True, False, True)
    assert settings.DEFAULT_TASKBOOK_EXECUTION_MODE == "parallel"
    assert settings.TASKBOOK_EXECUTION_MODES == ("parallel", "serial")


def test_shipped_config_stays_valid_after_hand_edits():
    """随附 toml 是**手改面**（用户实测即改）——只钉**合法性**，不钉数值。

    数值漂移是设计行为（「保存即生效」）；出厂值由
    :func:`test_factory_defaults_are_pinned` 在代码面钉。此处只拦「改出非法值」。
    """
    mode, _warns = settings.taskbook_execution_mode()
    assert mode in settings.TASKBOOK_EXECUTION_MODES
    form, _warns = settings.taskbook_form_defaults()
    assert form.new_priority in settings.TASKBOOK_PRIORITIES
    assert form.reject_priority in settings.TASKBOOK_PRIORITIES
    assert isinstance(form.new_leader, bool)
    for flag in (form.reject_infra, form.reject_leader, form.reject_inspector):
        assert isinstance(flag, bool)


def test_execution_mode_serial_accepted():
    mode, warnings = settings.taskbook_execution_mode(
        {"taskbooks": {"execution_mode": "serial"}}
    )
    assert (mode, warnings) == ("serial", [])


def test_execution_mode_missing_and_invalid_fall_back_with_warning():
    mode, warnings = settings.taskbook_execution_mode({})
    assert mode == "parallel" and len(warnings) == 1
    assert "execution_mode" in warnings[0]

    for bad in ("queue", "SERIAL", "", 3):
        mode, warnings = settings.taskbook_execution_mode(
            {"taskbooks": {"execution_mode": bad}}
        )
        assert mode == "parallel", f"{bad!r} 应回落出厂"
        assert len(warnings) == 1 and "合法：parallel/serial" in warnings[0]


def test_execution_mode_hot_switch_takes_effect_next_read(iso_settings):
    """热切换保证：改 toml 下一次读即生效——读数无缓存，串行谓词据此每命令现算。"""
    iso_settings.write_text(
        '[taskbooks]\nexecution_mode = "parallel"\n', encoding="utf-8"
    )
    assert settings.taskbook_execution_mode()[0] == "parallel"
    iso_settings.write_text(
        '[taskbooks]\nexecution_mode = "serial"\n', encoding="utf-8"
    )
    assert settings.taskbook_execution_mode()[0] == "serial"


def test_taskbook_form_defaults_per_key_fallback():
    """逐键回落：缺席/非法各出一条警告，其余键照常生效。"""
    form, warnings = settings.taskbook_form_defaults(
        {
            "taskbooks": {
                "new_default_priority": "high",
                "new_default_leader": "yes",
                "reject_default_priority": "urgent",
                "reject_default_infra": False,
                "reject_default_leader": "maybe",
                # reject_default_inspector 缺席
            }
        }
    )
    assert form.new_priority == "high"
    assert form.new_leader is True
    assert form.reject_priority == "normal", "非法优先级回落出厂"
    assert form.reject_infra is False
    assert form.reject_leader is False, "非法开关回落出厂"
    assert form.reject_inspector is True, "缺席键回落出厂"
    assert len(warnings) == 3
    assert any("reject_default_priority" in w for w in warnings)
    assert any("reject_default_leader" in w for w in warnings)
    assert any("reject_default_inspector" in w for w in warnings)


def test_taskbook_form_defaults_missing_section():
    form, warnings = settings.taskbook_form_defaults({})
    assert form == settings.DEFAULT_TASKBOOK_FORM
    assert len(warnings) == 6
