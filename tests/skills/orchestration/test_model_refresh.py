"""模型映射自动刷新（backlog 20261010-model-map-refresh）。

--list-models 实测格式（2026-10-10）::

    MODEL
    Qwen3.8-Flash                                          ← 内置，无 UUID
    Qwen-3.8-Flash (ed6f5996-0992-479f-86a8-3f29ce0e794c)  ← BYOK
    Qwen-3.8-Max (982ae403-bc5e-4f74-996d-04a9cd66f933)    ← BYOK

内置 ``Qwen3.8-Flash`` 与 BYOK ``Qwen-3.8-Flash`` 仅差一个连字符 → 模式必须**精确匹配**。
刷新语义（backlog 20261012-080410-devops 方案 A）：命中且跟踪面未写该档 → 解析值落
``sessions-runtime.json`` 的 ``models_resolved``（gitignored）；跟踪面写了该档 → 人值优先、
只报 ``shadowed`` 不改；未命中的档保持现值（宁旧勿空）；空目录抛错由调用方 fail-open。
**跟踪 registry.json 在任何分支下都不被写。**
"""

from __future__ import annotations

import json
from pathlib import Path

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
BYOK_FLASH_UUID = "ed6f5996-0992-479f-86a8-3f29ce0e794c"


# ---------------------------------------------------------------------------
# parse_model_catalog
# ---------------------------------------------------------------------------
def test_parse_catalog_real_format():
    assert registry.parse_model_catalog(CATALOG) == {
        "Qwen3.8-Flash": "Qwen3.8-Flash",  # 内置：值即名自身（-m 直接可用）
        "Qwen-3.8-Flash": BYOK_FLASH_UUID,
        "Qwen-3.8-Max": MAX_UUID,
    }


def test_parse_catalog_empty():
    assert registry.parse_model_catalog("") == {}
    assert registry.parse_model_catalog("MODEL\n\n  \n") == {}


# ---------------------------------------------------------------------------
# refresh_models（tmp 两面 + 假 list_models_output）
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_reg(tmp_path, monkeypatch):
    """隔离配置面与运行态（同目录，随 ``REGISTRY_PATH`` 一起搬）；返回 (reg, runtime)。"""
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    return reg_file, tmp_path / registry.RUNTIME_NAME


def _seed(reg_file: Path, models: dict) -> bytes:
    """写跟踪配置面并返回其字节（后续断言「一个字节没变」的基准）。"""
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "models": models,
                "members": {},
                "checks": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return reg_file.read_bytes()


def _runtime(runtime_file: Path) -> dict:
    return (
        json.loads(runtime_file.read_text(encoding="utf-8"))
        if runtime_file.exists()
        else {}
    )


def _fake_output(monkeypatch, text=CATALOG):
    monkeypatch.setattr(registry, "list_models_output", lambda *, timeout_s=60: text)


def test_refresh_writes_runtime_face_never_tracked(iso_reg, monkeypatch):
    """跟踪面空（四档都交机器）→ 解析值全落 models_resolved，registry.json 字节不变。"""
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, {})
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before, "自愈绝不写 git 跟踪的配置面"
    face = _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY]
    assert face == {
        "max": MAX_UUID,
        "flash": BYOK_FLASH_UUID,
        "flash_fallback": "Qwen3.8-Flash",
        "minimum": "Qwen3.8-Flash",
    }, "渠道互换（用户裁决 2026-10-12）：flash 解析到 BYOK，内置名降到兜底档"
    assert set(res["changed"]) == {"max", "flash", "flash_fallback", "minimum"}
    assert res["changed"]["max"] == {"old": None, "new": MAX_UUID}
    assert res["missing"] == [] and res["shadowed"] == {}
    assert res["catalog_size"] == 3


def test_tracked_pinned_tier_wins_and_reports_shadowed(iso_reg, monkeypatch):
    """跟踪面 pin 住的档：不写运行态、不改跟踪值，只在 shadowed 里点名（人不写则不换）。"""
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, {"max": "old-max-uuid"})
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before
    assert res["shadowed"] == {"max": {"tracked": "old-max-uuid", "resolved": MAX_UUID}}
    assert "max" not in _runtime(runtime_file).get(registry.RUNTIME_MODELS_KEY, {}), (
        "被遮蔽的档不进运行态面（免得存一个永不生效的值）"
    )
    assert set(res["changed"]) == {"flash", "flash_fallback", "minimum"}


def test_three_level_fallback_order(iso_reg):
    """读侧三级回退：跟踪值 > 运行态解析值 > 出厂默认（有意写空也是裁决）。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {"max": "", "flash": "TrackedFlash"})

    assert registry.Registry.load().models == {
        **registry.DEFAULT_MODELS,
        "max": "",
        "flash": "TrackedFlash",
    }, "两面皆空的档回落出厂默认"

    runtime_file.write_text(
        json.dumps(
            {
                "version": 1,
                "members": {},
                registry.RUNTIME_MODELS_KEY: {
                    "flash": "RuntimeFlash",
                    "minimum": "RuntimeMin",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    models = registry.Registry.load().models

    assert models["max"] == "", "跟踪面写空=不传 -m 的裁决，不被运行态值顶掉"
    assert models["flash"] == "TrackedFlash", "跟踪值优先于运行态值"
    assert models["minimum"] == "RuntimeMin", "运行态值优先于出厂默认"


def test_refresh_no_change_does_not_rewrite_runtime(iso_reg, monkeypatch):
    """运行态面已是最新 → 一个字节不写（免 churn，也免心跳文件无谓重写）。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    _fake_output(monkeypatch)
    registry.refresh_models()
    before = runtime_file.read_bytes()

    res = registry.refresh_models()

    assert res["changed"] == {}
    assert runtime_file.read_bytes() == before


def test_refresh_missing_name_keeps_current(iso_reg, monkeypatch):
    """目录缺 BYOK 名（账户瞬时状态）→ 该档保持现值，绝不清空。"""
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, {"max": "old-max-uuid"})
    _fake_output(monkeypatch, "MODEL\nQwen3.8-Flash\n")

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before
    assert res["missing"] == ["max←Qwen-3.8-Max", "flash←Qwen-3.8-Flash"]
    assert "max" not in _runtime(runtime_file).get(registry.RUNTIME_MODELS_KEY, {})
    assert registry.Registry.load().models["max"] == "old-max-uuid", "未命中不清空"


def test_refresh_empty_catalog_raises(iso_reg, monkeypatch):
    _seed(iso_reg[0], {})
    _fake_output(monkeypatch, "MODEL\n")
    with pytest.raises(ValueError):
        registry.refresh_models()


def test_refresh_custom_patterns_override(iso_reg, monkeypatch):
    """registry.model_patterns 覆盖默认匹配名（BYOK 改名后不必改代码）。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {},
                "model_patterns": {"max": "Qwen3.8-Flash"},
                "members": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _fake_output(monkeypatch)

    registry.refresh_models()

    face = _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY]
    assert face["max"] == "Qwen3.8-Flash", "覆盖名生效：max 解析到内置名而非 UUID"


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
            "missing": ["flash←Qwen-3.8-Flash"],
            "shadowed": {
                "minimum": {"tracked": "HumanMin", "resolved": "Qwen3.8-Flash"}
            },
            "catalog_size": 2,
        }

    monkeypatch.setattr(drain, "refresh_models", fake_refresh)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log, dry=True)
    assert calls == [1]
    assert summary["models_refreshed"] == {"max": {"old": "a", "new": MAX_UUID}}
    log = run_log.read_text(encoding="utf-8")
    assert "模型映射已刷新（运行态面）" in log and "未命中" in log
    assert "跟踪面遮蔽解析值" in log and "HumanMin" in log, "遮蔽要留痕，别静默吞掉"


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
def test_cli_refresh_models_success(iso_reg, monkeypatch, capsys):
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    _fake_output(monkeypatch)
    assert orch.main(["refresh-models"]) == 0
    out = capsys.readouterr().out
    assert "[max]" in out and MAX_UUID in out and "已刷新" in out
    assert "models_resolved" in out, "打印落点：写运行态面，不是跟踪 registry"
    assert runtime_file.exists() and not _runtime(runtime_file).get("members")


def test_cli_refresh_models_reports_shadowed(iso_reg, monkeypatch, capsys):
    reg_file, _runtime_file = iso_reg
    _seed(reg_file, {"max": "old-max-uuid"})
    _fake_output(monkeypatch)
    assert orch.main(["refresh-models"]) == 0
    out = capsys.readouterr().out
    assert "遮蔽解析值" in out and "old-max-uuid" in out and MAX_UUID in out
    assert "护栏级" in out, "指出换档的正确通路（改跟踪值须经授权）"


def test_cli_refresh_models_failure_rc2(iso_reg, monkeypatch, capsys):
    _seed(iso_reg[0], {})

    def boom(*, timeout_s=60):
        raise FileNotFoundError("exe")

    monkeypatch.setattr(registry, "list_models_output", boom)
    assert orch.main(["refresh-models"]) == 2
    assert "刷新失败" in capsys.readouterr().out
