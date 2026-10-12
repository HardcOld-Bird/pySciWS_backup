"""模型映射自动刷新（backlog 20261010-model-map-refresh；链形态 20261012-211500-leader）。

--list-models 实测格式（2026-10-12 22:xx 真实账户目录）::

    MODEL
    Mimo-V2.6-Pro   (99127e47-…)                               ← 用户新增自定义模型
    Mimo-V2.6-Flash (017036af-…)
    Qwen-3.8-Max    (982ae403-…)                               ← BYOK 按量
    Qwen-3.8-Flash  (ed6f5996-…)
    Qwen3.8-Flash                                              ← 内置，无 UUID

内置 ``Qwen3.8-Flash`` 与 BYOK ``Qwen-3.8-Flash`` 仅差一个连字符 → 模式必须**精确匹配**。
刷新语义（backlog 20261012-080410-devops 方案 A + 20261012-211500-leader 链化）：按**链模板**
（``DEFAULT_MODEL_CHAINS``）逐席解析，该档在跟踪面**无人写值** → 整条解析链落
``sessions-runtime.json`` 的 ``models_resolved``（gitignored）；跟踪面写了该档 → 人值优先、
只报 ``shadowed`` 不改（合并粒度是整条链，机器不往人写的链中间插席）；目录里缺的名**跳过该席、
其余照旧**（宁旧勿空）；空目录抛错由调用方 fail-open。**跟踪 registry.json 任何分支都不被写。**
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysci.skills.orchestration.tools import drain, orch, registry

from .conftest import seed_backlog

MIMO_PRO_UUID = "99127e47-626d-421f-b0cb-02ffdbf1ada7"
MIMO_FLASH_UUID = "017036af-b630-498a-bfea-883a20c67e33"
MAX_UUID = "982ae403-bc5e-4f74-996d-04a9cd66f933"
BYOK_FLASH_UUID = "ed6f5996-0992-479f-86a8-3f29ce0e794c"

CATALOG = (
    "MODEL\n"
    f"Mimo-V2.6-Pro ({MIMO_PRO_UUID})\n"
    f"Mimo-V2.6-Flash ({MIMO_FLASH_UUID})\n"
    f"Qwen-3.8-Max ({MAX_UUID})\n"
    f"Qwen-3.8-Flash ({BYOK_FLASH_UUID})\n"
    "Qwen3.8-Flash\n"
)

#: 解析后的三档链（席位形态，与 registry.json 跟踪面同值不同形：那边是人类名模板）
CHAINS = {
    "max": [MIMO_PRO_UUID, MAX_UUID, ""],
    "flash": [MIMO_FLASH_UUID, BYOK_FLASH_UUID, "Qwen3.8-Flash"],
    "minimum": [MIMO_FLASH_UUID, "Qwen3.8-Flash"],
}


# ---------------------------------------------------------------------------
# parse_model_catalog
# ---------------------------------------------------------------------------
def test_parse_catalog_real_format():
    assert registry.parse_model_catalog(CATALOG) == {
        "Qwen3.8-Flash": "Qwen3.8-Flash",  # 内置：值即名自身（-m 直接可用）
        "Qwen-3.8-Flash": BYOK_FLASH_UUID,
        "Qwen-3.8-Max": MAX_UUID,
        "Mimo-V2.6-Pro": MIMO_PRO_UUID,
        "Mimo-V2.6-Flash": MIMO_FLASH_UUID,
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
    """跟踪面空（三档都交机器）→ 解析链全落 models_resolved，registry.json 字节不变。"""
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, {})
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before, "自愈绝不写 git 跟踪的配置面"
    face = _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY]
    assert face == CHAINS, "三档都是**整条链**（首席在前），不是单值"
    assert set(res["changed"]) == {"max", "flash", "minimum"}
    assert res["changed"]["max"] == {"old": None, "new": CHAINS["max"]}
    assert res["missing"] == [] and res["shadowed"] == {} and res["conflicts"] == []
    assert res["catalog_size"] == 5


def test_refresh_writes_seat_name_table_for_alerts(iso_reg, monkeypatch):
    """规格 4 告警面可读性：自愈顺带把「席（UUID/内置名）→ 人类名」写进运行态反查表。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    _fake_output(monkeypatch)

    registry.refresh_models()

    names = _runtime(runtime_file)[registry.RUNTIME_SEAT_NAMES_KEY]
    assert names[MIMO_PRO_UUID] == "Mimo-V2.6-Pro"
    assert names[MIMO_FLASH_UUID] == "Mimo-V2.6-Flash"
    assert names[MAX_UUID] == "Qwen-3.8-Max"
    reg = registry.Registry.load()
    assert reg.seat_label(MIMO_FLASH_UUID) == "Mimo-V2.6-Flash"
    assert reg.seat_label("") == "(用户级默认)", (
        "不传 -m 的一席也要有名字，告警面不能空转"
    )
    assert reg.seat_label("unseen-seat") == "unseen-seat", "反查不到就原样返回"


def test_tracked_pinned_tier_wins_and_reports_shadowed(iso_reg, monkeypatch):
    """跟踪面 pin 住的档：不写运行态、不改跟踪值，只在 shadowed 里点名（人不写则不换）。"""
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, {"max": "old-max-uuid"})
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before
    assert res["shadowed"] == {
        "max": {"tracked": "old-max-uuid", "resolved": CHAINS["max"]}
    }
    assert "max" not in _runtime(runtime_file).get(registry.RUNTIME_MODELS_KEY, {}), (
        "被遮蔽的档不进运行态面（免得存一个永不生效的值）"
    )
    assert set(res["changed"]) == {"flash", "minimum"}


def test_machine_does_not_insert_seats_into_human_chain(iso_reg, monkeypatch):
    """合并粒度是**整条链**：跟踪面写了两席的链，机器不得往里插第三席（#74 的链形态版本）。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {"flash": [MIMO_FLASH_UUID, "Qwen3.8-Flash"]})
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert res["shadowed"]["flash"] == {
        "tracked": [MIMO_FLASH_UUID, "Qwen3.8-Flash"],
        "resolved": CHAINS["flash"],
    }
    assert "flash" not in _runtime(runtime_file).get(registry.RUNTIME_MODELS_KEY, {})
    assert registry.Registry.load().model_chain("flash") == [
        MIMO_FLASH_UUID,
        "Qwen3.8-Flash",
    ], "读侧就是人写的整条链，顺序也不动"


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
                    "flash": ["RuntimeFlash", "RuntimeBackup"],
                    "minimum": "RuntimeMin",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    reg = registry.Registry.load()

    assert reg.models["max"] == "", "跟踪面写空=不传 -m 的裁决，不被运行态值顶掉"
    assert reg.models["flash"] == "TrackedFlash", "跟踪值优先于运行态值"
    assert reg.models["minimum"] == "RuntimeMin", "运行态值优先于出厂默认"
    assert reg.model_chain("minimum") == ["RuntimeMin"], "单值旧写＝一席链"


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


def test_refresh_legacy_shape_counts_as_no_change(iso_reg, monkeypatch):
    """运行态面是旧单值写法、归一后同值 → 判「无变化」不 churn；真名全没命中的档保持现值。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    legacy = {"flash": "Qwen3.8-Flash", "minimum": "Qwen3.8-Flash"}
    runtime_file.write_text(
        json.dumps({registry.RUNTIME_MODELS_KEY: legacy}, ensure_ascii=False),
        encoding="utf-8",
    )
    _fake_output(monkeypatch, "MODEL\nQwen3.8-Flash\n")

    res = registry.refresh_models()

    assert res["changed"] == {}, (
        f"旧写归一后同值不该被当成变化；max 只剩占位席也不该写成机器裁决：{res['changed']}"
    )
    assert _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY] == legacy, (
        "models_resolved 面一字未动（无变化即不写）"
    )


def test_refresh_missing_name_keeps_current(iso_reg, monkeypatch):
    """目录缺链里某个名（账户瞬时状态）→ 该席跳过、其余照旧，绝不清空该档。"""
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, {"max": "old-max-uuid"})
    _fake_output(monkeypatch, "MODEL\nQwen3.8-Flash\n")

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before
    assert set(res["missing"]) == {
        "flash←Mimo-V2.6-Flash",
        "flash←Qwen-3.8-Flash",
        "max←Mimo-V2.6-Pro",
        "max←Qwen-3.8-Max",
        "minimum←Mimo-V2.6-Flash",
    }
    assert "max" not in _runtime(runtime_file).get(registry.RUNTIME_MODELS_KEY, {})
    assert registry.Registry.load().models["max"] == "old-max-uuid", "未命中不清空"
    assert res["shadowed"] == {}, (
        "跟踪链含账户已摘除的席 → 只报 missing 不报 shadowed："
        "那不是人能改的漂移，报了会淹没真信号"
    )


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
                "model_patterns": {"max": ["Qwen3.8-Flash"]},
                "members": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _fake_output(monkeypatch)

    registry.refresh_models()

    face = _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY]
    assert face["max"] == ["Qwen3.8-Flash"], (
        "覆盖名生效：max 整条链被替换成内置名一席，而非默认模板的三席"
    )


def test_legacy_inverted_patterns_are_rules_not_tiers(iso_reg, monkeypatch):
    """历史倒表 ``{"人类名": "档"}`` 是**名→档规则**，不能被当成档位名模板。

    旧实现把两张表直接 merge ⇒ 倒表的键被当成档位名，每个 drain 报一堆
    ``Qwen3.8-Flash←flash`` 的假「未命中」（backlog 20261012-211500-leader 收口）。
    """
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {},
                "model_patterns": {
                    "Mimo-V2.6-Pro": "max",
                    "Qwen3.8-Flash": "flash",
                },
                "members": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert res["missing"] == [], "倒表键是模型名，不该被当成档去目录里找"
    assert res["conflicts"] == [], (
        "一席同属多档（Qwen3.8-Flash 在 flash 与 minimum 两条链里）不算串档"
    )
    face = _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY]
    assert "Mimo-V2.6-Pro" not in face, "倒表键没变成一个以模型名命名的档"
    assert face["max"] == CHAINS["max"], "档→名模板未被倒表污染"


def test_inverted_rule_contradicting_chain_is_named(iso_reg, monkeypatch):
    """真串档要有牙：规则把一个名归到它**根本不在其中**的档 → conflicts 点名（#74 哨兵）。"""
    reg_file, _ = iso_reg
    _seed(reg_file, {})
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {},
                "model_patterns": {"Mimo-V2.6-Pro": "minimum"},
                "members": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert res["conflicts"] == ["Mimo-V2.6-Pro∈max，规则说 minimum"], (
        f"串档必须被点名，否则自愈会安静地把组长档的席位塞进压测档：{res['conflicts']}"
    )


def test_model_chains_overrides_template(iso_reg, monkeypatch):
    """``registry.model_chains``（新写形态）覆盖链模板：换档/加席不必改代码。"""
    reg_file, runtime_file = iso_reg
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {},
                "model_chains": {"flash": ["Qwen3.8-Flash", "Qwen-3.8-Flash"]},
                "members": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _fake_output(monkeypatch)

    registry.refresh_models()

    face = _runtime(runtime_file)[registry.RUNTIME_MODELS_KEY]
    assert face["flash"] == ["Qwen3.8-Flash", BYOK_FLASH_UUID], (
        "人写的链序即解析后的席序（内置兜底在前、BYOK 在后）"
    )


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
    assert "[max]" in out and "Mimo-V2.6-Pro" in out and "已刷新" in out
    assert "(空)" in out and "→" in out, "打印的是整条链（首席→次席），不是单值"
    assert "models_resolved" in out, "打印落点：写运行态面，不是跟踪 registry"
    assert runtime_file.exists() and not _runtime(runtime_file).get("members")


def test_cli_refresh_models_reports_shadowed(iso_reg, monkeypatch, capsys):
    reg_file, _runtime_file = iso_reg
    _seed(reg_file, {"max": "old-max-uuid"})
    _fake_output(monkeypatch)
    assert orch.main(["refresh-models"]) == 0
    out = capsys.readouterr().out
    assert "遮蔽解析链" in out and "old-max-uuid" in out and "Mimo-V2.6-Pro" in out
    assert "护栏级" in out, "指出换档的正确通路（改跟踪值须经授权）"


def test_cli_refresh_models_failure_rc2(iso_reg, monkeypatch, capsys):
    _seed(iso_reg[0], {})

    def boom(*, timeout_s=60):
        raise FileNotFoundError("exe")

    monkeypatch.setattr(registry, "list_models_output", boom)
    assert orch.main(["refresh-models"]) == 2
    assert "刷新失败" in capsys.readouterr().out
