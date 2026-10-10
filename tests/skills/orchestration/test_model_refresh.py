"""模型映射自动刷新（backlog 20261010-model-map-refresh）。

--list-models 实测格式（2026-10-10）::

    MODEL
    Qwen3.8-Flash                                          ← 内置，无 UUID
    Qwen-3.8-Flash (ed6f5996-0992-479f-86a8-3f29ce0e794c)  ← BYOK
    Qwen-3.8-Max (982ae403-bc5e-4f74-996d-04a9cd66f933)    ← BYOK

内置 ``Qwen3.8-Flash`` 与 BYOK ``Qwen-3.8-Flash`` 仅差一个连字符 → 模式必须**精确匹配**。
刷新语义：命中且不同才写回；未命中档保持现值（宁旧勿空）；空目录抛错由调用方 fail-open。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import drain, orch, registry

from .conftest import seed_backlog

CATALOG = (
    "MODEL\n"
    "Qwen3.8-Flash\n"
    "Qwen-3.8-Flash (ed6f5996-0992-479f-86a8-3f29ce0e794c)\n"
    "Qwen-3.8-Max (982ae403-bc5e-4f74-996d-04a9cd66f933)\n"
)
MAX_UUID = "982ae403-bc5e-4f74-996d-04a9cd66f933"
FB_UUID = "ed6f5996-0992-479f-86a8-3f29ce0e794c"


# ---------------------------------------------------------------------------
# parse_model_catalog
# ---------------------------------------------------------------------------
def test_parse_catalog_real_format():
    assert registry.parse_model_catalog(CATALOG) == {
        "Qwen3.8-Flash": "Qwen3.8-Flash",  # 内置：值即名自身（-m 直接可用）
        "Qwen-3.8-Flash": FB_UUID,
        "Qwen-3.8-Max": MAX_UUID,
    }


def test_parse_catalog_empty():
    assert registry.parse_model_catalog("") == {}
    assert registry.parse_model_catalog("MODEL\n\n  \n") == {}


# ---------------------------------------------------------------------------
# refresh_models（tmp registry + 假 list_models_output）
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_registry(tmp_path, monkeypatch):
    """隔离 registry：返回 (reg_file, saves)；save 写 tmp 并记录调用。"""
    reg_file = tmp_path / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "models": {
                    "max": "old-max-uuid",
                    "flash": "Qwen3.8-Flash",
                    "flash_fallback": "",
                },
                "members": {},
                "checks": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    saves: list = []

    def fake_save(self):
        saves.append(self.data)
        reg_file.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    monkeypatch.setattr(registry.Registry, "save", fake_save)
    return reg_file, saves


def _fake_output(monkeypatch, text=CATALOG):
    monkeypatch.setattr(registry, "list_models_output", lambda *, timeout_s=60: text)


def test_refresh_updates_changed_tiers(iso_registry, monkeypatch):
    reg_file, saves = iso_registry
    _fake_output(monkeypatch)
    res = registry.refresh_models()
    assert res["changed"]["max"] == {"old": "old-max-uuid", "new": MAX_UUID}
    assert res["changed"]["flash_fallback"] == {"old": "", "new": FB_UUID}
    assert "flash" not in res["changed"], "内置档值未变（名自身）不应写"
    assert res["missing"] == [] and res["catalog_size"] == 3
    models = json.loads(reg_file.read_text(encoding="utf-8"))["models"]
    assert models["max"] == MAX_UUID and models["flash_fallback"] == FB_UUID
    assert len(saves) == 1


def test_refresh_all_current_no_save(iso_registry, monkeypatch):
    reg_file, saves = iso_registry
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["models"] = {
        "max": MAX_UUID,
        "flash": "Qwen3.8-Flash",
        "flash_fallback": FB_UUID,
    }
    reg_file.write_text(json.dumps(data), encoding="utf-8")
    _fake_output(monkeypatch)
    res = registry.refresh_models()
    assert res["changed"] == {} and saves == [], "无变化不得写回（免 churn）"


def test_refresh_missing_name_keeps_current(iso_registry, monkeypatch):
    """目录缺 BYOK 名（账户瞬时状态）→ 该档保持现值，绝不清空。"""
    reg_file, saves = iso_registry
    _fake_output(monkeypatch, "MODEL\nQwen3.8-Flash\n")
    res = registry.refresh_models()
    assert res["changed"] == {} and saves == []
    assert res["missing"] == ["max←Qwen-3.8-Max", "flash_fallback←Qwen-3.8-Flash"]
    models = json.loads(reg_file.read_text(encoding="utf-8"))["models"]
    assert models["max"] == "old-max-uuid", "未命中不清空"


def test_refresh_empty_catalog_raises(iso_registry, monkeypatch):
    _fake_output(monkeypatch, "MODEL\n")
    with pytest.raises(ValueError):
        registry.refresh_models()


def test_refresh_custom_patterns_override(iso_registry, monkeypatch):
    reg_file, _ = iso_registry
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["model_patterns"] = {"max": "Qwen3.8-Flash"}  # 覆盖 max 档匹配名
    reg_file.write_text(json.dumps(data), encoding="utf-8")
    _fake_output(monkeypatch)
    res = registry.refresh_models()
    assert res["changed"]["max"]["new"] == "Qwen3.8-Flash"


# ---------------------------------------------------------------------------
# drain 启动自愈（fail-open）
# ---------------------------------------------------------------------------
def test_drain_startup_refreshes_and_logs(iso_state, monkeypatch):
    seed_backlog(iso_state, [])
    calls = []

    def fake_refresh():
        calls.append(1)
        return {
            "changed": {"max": {"old": "a", "new": MAX_UUID}},
            "missing": ["flash_fallback←Qwen-3.8-Flash"],
            "catalog_size": 2,
        }

    monkeypatch.setattr(drain, "refresh_models", fake_refresh)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log, dry=True)
    assert calls == [1]
    assert summary["models_refreshed"] == {"max": {"old": "a", "new": MAX_UUID}}
    log = run_log.read_text(encoding="utf-8")
    assert "模型映射已刷新" in log and "未命中" in log


def test_drain_startup_refresh_fail_open(iso_state, monkeypatch):
    """刷新抛任何异常只告警，不阻塞消化（宁旧勿空）。"""
    seed_backlog(iso_state, [{"id": "b1", "status": "pending"}])

    def boom():
        raise OSError("exe not found")

    monkeypatch.setattr(drain, "refresh_models", boom)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log, dry=True)
    assert summary["count"] == 1
    assert "models_refreshed" not in summary
    assert "模型映射刷新失败" in run_log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# orch refresh-models CLI
# ---------------------------------------------------------------------------
def test_cli_refresh_models_success(iso_registry, monkeypatch, capsys):
    _fake_output(monkeypatch)
    assert orch.main(["refresh-models"]) == 0
    out = capsys.readouterr().out
    assert "[max]" in out and MAX_UUID in out and "已刷新" in out


def test_cli_refresh_models_failure_rc2(iso_registry, monkeypatch, capsys):
    def boom(*, timeout_s=60):
        raise FileNotFoundError("exe")

    monkeypatch.setattr(registry, "list_models_output", boom)
    assert orch.main(["refresh-models"]) == 2
    assert "刷新失败" in capsys.readouterr().out
