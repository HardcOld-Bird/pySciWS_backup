"""registry 配置面磁盘权威（backlog 20261011-220621-leader，REFACTOR 首战缺陷）。

实锤形状：21:05:57 drain worker 启动并 ``Registry.load()``（deputy/reviewer=max）→
21:42:41 E0-C 批次合并降档 flash+tier_note（HEAD 正确）→ 22:04 实测工作树**回退为 max、
tier_note 消失**。长驻进程收尾时把启动快照整个 ``save()`` 回去，后写覆盖，用户已裁决的
配置改动在运行态被无声撤销。本组钉住修复后的不变量：**任何进程都不得用陈旧全量副本
覆盖 state 配置面**（save 前重读磁盘、只覆盖本进程声明的键）。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import registry as reg_mod
from pysci.skills.orchestration.tools.registry import Registry

MAX_OLD = "OLD-MAX-UUID"
MAX_NEW = "982ae403-bc5e-4f74-996d-04a9cd66f933"


def _cfg() -> dict:
    """干净形状的配置面（配置无 sessions、运行态另存）。"""
    return {
        "version": 1,
        "exe": None,
        "models": {"max": MAX_OLD, "flash": "Qwen3.8-Flash", "flash_fallback": ""},
        "members": {
            "devops": {"pod": "orchestration/pods/devops", "model_tier": "flash"},
            "deputy": {"pod": "orchestration/pods/deputy", "model_tier": "max"},
        },
        "checks": {
            "orch-tests": {"cmd": ["uv", "run", "pytest", "-q"], "cwd": "project"}
        },
    }


def _runtime(sessions: dict[str, list[dict]]) -> dict:
    return {
        "version": 1,
        "members": {mid: {"sessions": s} for mid, s in sessions.items()},
    }


def _seed(iso_registry, cfg: dict, runtime: dict) -> tuple:
    reg_file, runtime_file, _backup = iso_registry
    reg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_file.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reg_file, runtime_file


def _session(sid: str, hops: int, offset: int) -> dict:
    return {
        "sid": sid,
        "name": f"任务{sid}",
        "hops": hops,
        "last_active": f"2026-10-11T21:{hops:02d}:00+08:00",
        "chars_offset": offset,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# 主回归：复刻首战事故序列（旧快照 → 期间合并 → 写运行态）
# ---------------------------------------------------------------------------
def test_stale_worker_runtime_write_keeps_merged_config(iso_registry):
    """worker 拿 21:05 的快照记心跳，绝不把 21:42 合并进来的降档回退成 max。"""
    reg_file, runtime_file = _seed(
        iso_registry,
        _cfg(),
        _runtime({"devops": [_session("s1", 1, 10)]}),
    )
    worker = Registry.load()  # ← 21:05:57 drain worker 启动
    m = worker.member("devops")  # ← 一跳开始（此后不再重读配置）

    merged = _cfg()  # ← 21:42:41 E0-C 批次合并进磁盘
    merged["members"]["deputy"]["model_tier"] = "flash"
    merged["members"]["deputy"]["tier_note"] = "2026-10-11 REFACTOR v1.0 §1.1 定稿降档"
    merged["members"]["devops"]["effort"] = "high"
    merged_text = json.dumps(merged, ensure_ascii=False, indent=2)
    reg_file.write_text(merged_text, encoding="utf-8")

    m.sessions = [_session("s1", 2, 999)]  # ← 一跳收尾：心跳与水位入账
    worker.write_sessions(m)
    worker.save()

    assert reg_file.read_text(encoding="utf-8") == merged_text, (
        "心跳类写回根本不该碰配置面（磁盘权威：无声明则一字不写）"
    )
    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert cfg["members"]["deputy"]["model_tier"] == "flash"
    assert cfg["members"]["deputy"]["tier_note"].startswith("2026-10-11")
    assert cfg["members"]["devops"]["effort"] == "high"
    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    sess = rt["members"]["devops"]["sessions"][0]
    assert (sess["hops"], sess["chars_offset"]) == (2, 999), "运行态该写的照样写"


def test_runtime_write_keeps_undeclared_members_pool(iso_registry):
    """并发第二写者：本进程只声明 devops，他人（lit）会话池原样保留。"""
    reg_file, runtime_file = _seed(
        iso_registry,
        _cfg(),
        _runtime(
            {
                "devops": [_session("d1", 1, 10)],
                "lit": [_session("l1", 5, 777)],
            }
        ),
    )
    worker = Registry.load()
    m = worker.member("devops")
    m.sessions = [_session("d1", 2, 20)]
    worker.write_sessions(m)
    worker.save()

    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    assert rt["members"]["lit"]["sessions"][0]["chars_offset"] == 777
    assert rt["members"]["devops"]["sessions"][0]["chars_offset"] == 20


def test_runtime_merge_keeps_other_process_new_session(iso_registry):
    """同成员并发：本进程按 sid 合并，不抹掉他方期间新建的会话。"""
    reg_file, runtime_file = _seed(
        iso_registry, _cfg(), _runtime({"devops": [_session("d1", 1, 10)]})
    )
    worker = Registry.load()
    m = worker.member("devops")
    # 他方进程在期间为同一成员新开了会话 d2（磁盘已有）
    other = _runtime({"devops": [_session("d1", 1, 10), _session("d2", 1, 5)]})
    runtime_file.write_text(
        json.dumps(other, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    m.sessions = [_session("d1", 2, 20)]
    worker.write_sessions(m)
    worker.save()

    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    by_sid = {s["sid"]: s for s in rt["members"]["devops"]["sessions"]}
    assert by_sid["d1"]["chars_offset"] == 20, "本进程改过的 sid 以本进程为准"
    assert by_sid["d2"]["sid"] == "d2", "他方新建的会话不丢"


# ---------------------------------------------------------------------------
# 声明式配置写：只叠加声明的键
# ---------------------------------------------------------------------------
def test_new_member_default_does_not_override_midlife_merge(iso_registry):
    """ensure_member_defaults 的默认条目是「新增」语义：磁盘已有真条目则不覆盖。"""
    reg_file, _runtime_file = _seed(iso_registry, _cfg(), _runtime({}))
    worker = Registry.load()
    worker.ensure_member_defaults("lit")  # 内存无 lit → 声明默认条目

    merged = _cfg()
    merged["members"]["lit"] = {
        "pod": "orchestration/pods/lit",
        "model_tier": "max",
        "effort": "high",
    }
    reg_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    worker.write_sessions(worker.member("lit"))
    worker.save()

    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert cfg["members"]["lit"]["model_tier"] == "max", (
        "默认值不得盖掉合并进来的真条目"
    )
    assert cfg["members"]["lit"]["effort"] == "high"


def test_new_member_default_lands_when_disk_also_absent(iso_registry):
    """磁盘也无该成员 → 声明的新增正常落盘（新增能力没被合并语义挡掉）。"""
    reg_file, runtime_file = _seed(iso_registry, _cfg(), _runtime({}))
    worker = Registry.load()
    worker.ensure_member_defaults("newpod")
    worker.save()

    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert cfg["members"]["newpod"]["model_tier"] == "flash"
    assert "sessions" not in cfg["members"]["newpod"]
    assert cfg["members"]["devops"] == _cfg()["members"]["devops"]
    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    assert rt["members"]["newpod"]["sessions"] == []


def test_declare_config_overlays_only_declared_keys(iso_registry):
    """declare_config 只把声明的 models 键叠上去，同块兄弟档与成员配置保持磁盘值。"""
    reg_file, _runtime_file = _seed(iso_registry, _cfg(), _runtime({}))
    worker = Registry.load()

    merged = _cfg()
    merged["models"]["flash"] = "HumanFlash"
    merged["members"]["deputy"]["model_tier"] = "flash"
    reg_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    worker.declare_config(models={"max": MAX_NEW})
    worker.save()

    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert cfg["models"] == {
        "max": MAX_NEW,
        "flash": "HumanFlash",
        "flash_fallback": "",
    }
    assert cfg["members"]["deputy"]["model_tier"] == "flash"
    assert worker.models["max"] == MAX_NEW, "内存副本同步（同进程后续读一致）"


def test_refresh_models_from_stale_snapshot_keeps_merged_config(
    iso_registry, monkeypatch
):
    """drain 启动自愈：即便拿的是旧快照，也只叠 models，不整体回写配置面。"""
    reg_file, _runtime_file = _seed(iso_registry, _cfg(), _runtime({}))
    stale = Registry.load()  # ← worker 启动时的旧副本

    merged = _cfg()
    merged["members"]["deputy"]["model_tier"] = "flash"
    merged["members"]["devops"]["tier_note"] = "E0-C 合并附注"
    merged["checks"]["figure-audit"] = {
        "cmd": ["uv", "run", "pysci-fig", "audit"],
        "cwd": "pod",
    }
    reg_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, *a, **k: stale))
    catalog = f"Qwen-3.8-Max ({MAX_NEW})\nQwen3.8-Flash\n"
    monkeypatch.setattr(reg_mod, "list_models_output", lambda *a, **k: catalog)

    res = reg_mod.refresh_models()

    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert res["changed"]["max"]["new"] == MAX_NEW, "变更档要落"
    assert cfg["models"]["max"] == MAX_NEW
    assert cfg["members"]["deputy"]["model_tier"] == "flash", (
        "期间合并的成员改动不被回退"
    )
    assert cfg["members"]["devops"]["tier_note"] == "E0-C 合并附注"
    assert "figure-audit" in cfg["checks"], "期间合并的 checks 不丢"


def test_corrupt_config_file_fails_loud_not_overwritten(iso_registry):
    """坏 JSON 硬失败：绝不用骨架/内存副本覆盖一份「可能还能救」的 state 文件。"""
    reg_file, _runtime_file, _backup = iso_registry
    reg_file.write_text("{ not json", encoding="utf-8")
    worker = Registry(
        data=_cfg(),
        runtime={"version": 1, "members": {}},
        path=reg_file,
        cfg_overlay={"models": {"max": MAX_NEW}},
    )

    with pytest.raises(json.JSONDecodeError):
        worker.save()
    assert reg_file.read_text(encoding="utf-8") == "{ not json"
