"""registry.json 运行态迁出（backlog 20261010-registry-runtime-split）。

根治 163347/220711 两笔同构合并阻塞：sessions/hops/last_active/chars_offset 留在配置面
会让 main 工作区每派发必 dirty。本组钉住迁移机制的不变量——首次 load 一次性搬旧数据 +
备份（幂等、备份不覆盖）、运行态成为会话池唯一落点、配置面永不再携带 sessions。
"""

from __future__ import annotations

import json

import pytest

from pysci.skills.orchestration.tools import registry as reg_mod
from pysci.skills.orchestration.tools.registry import (
    BACKUP_NAME,
    RUNTIME_NAME,
    Member,
    Registry,
)


def _legacy_data(*, with_sessions: bool = True) -> dict:
    """一份「旧形状」registry：成员内嵌 sessions（with_sessions=False 时已剥离）。"""
    sess = [
        {
            "sid": "aaaa1111",
            "name": "任务A",
            "hops": 3,
            "last_active": "2026-10-10T09:00:00+08:00",
            "chars_offset": 1234,
            "status": "active",
        }
    ]
    entry = {"pod": "orchestration/pods/devops", "model_tier": "flash"}
    if with_sessions:
        entry["sessions"] = sess
    return {
        "version": 1,
        "exe": None,
        "models": {"max": "", "flash": "Qwen3.8-Flash"},
        "members": {"devops": entry},
        "checks": {"orch-tests": {"cmd": ["uv", "run", "pytest", "-q"]}},
    }


@pytest.fixture
def iso_registry(tmp_path, monkeypatch):
    """把 REGISTRY_PATH 指向 tmp，返回 (reg_file, runtime_file, backup_file)。"""
    reg_file = tmp_path / "registry.json"
    runtime_file = tmp_path / RUNTIME_NAME
    backup_file = tmp_path / BACKUP_NAME
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", reg_file)
    return reg_file, runtime_file, backup_file


def test_first_load_migrates_sessions_to_runtime(iso_registry):
    """旧形状（配置内嵌 sessions + 运行态缺席）→ 首次 load 抬高到运行态、配置剥离。"""
    reg_file, runtime_file, backup_file = iso_registry
    original = _legacy_data()
    reg_file.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

    reg = Registry.load()

    # 运行态落盘，且承载原会话
    assert runtime_file.exists()
    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    assert (
        rt["members"]["devops"]["sessions"] == original["members"]["devops"]["sessions"]
    )
    # 配置面已剥离 sessions 并原子写回磁盘
    on_disk = json.loads(reg_file.read_text(encoding="utf-8"))
    assert "sessions" not in on_disk["members"]["devops"]
    # 配置面其余低频项（checks/models/pod）保持不变
    assert on_disk["checks"]["orch-tests"]["cmd"] == ["uv", "run", "pytest", "-q"]
    # 原文备份已创建，保留 pre-split 的完整形状
    assert backup_file.exists()
    backed = json.loads(backup_file.read_text(encoding="utf-8"))
    assert (
        backed["members"]["devops"]["sessions"]
        == original["members"]["devops"]["sessions"]
    )


def test_migration_is_idempotent(iso_registry):
    """迁移后再 load：无运行态文件变化、配置不再被改写、备份不被覆盖（幂等）。"""
    reg_file, runtime_file, backup_file = iso_registry
    reg_file.write_text(
        json.dumps(_legacy_data(), ensure_ascii=False), encoding="utf-8"
    )

    Registry.load()  # 首次迁移
    runtime_after_first = runtime_file.read_text(encoding="utf-8")
    config_after_first = reg_file.read_text(encoding="utf-8")
    backup_after_first = backup_file.read_text(encoding="utf-8")

    reg2 = Registry.load()  # 二次 load 不得再搬、再写、再备份
    assert runtime_file.read_text(encoding="utf-8") == runtime_after_first
    assert reg_file.read_text(encoding="utf-8") == config_after_first
    assert backup_file.read_text(encoding="utf-8") == backup_after_first
    assert reg2.member("devops").sessions[0]["chars_offset"] == 1234


def test_backup_not_overwritten_when_already_present(iso_registry):
    """备份文件已存在（真·pre-split 快照）→ 迁移绝不覆盖它。"""
    reg_file, _runtime, backup_file = iso_registry
    backup_file.write_text('{"sentinel": "do-not-overwrite"}', encoding="utf-8")
    reg_file.write_text(
        json.dumps(_legacy_data(), ensure_ascii=False), encoding="utf-8"
    )

    Registry.load()

    assert json.loads(backup_file.read_text(encoding="utf-8")) == {
        "sentinel": "do-not-overwrite"
    }


def test_no_cutover_when_runtime_already_exists(iso_registry):
    """干净形状（配置无 sessions + 运行态在盘）→ 不触发迁移、不创建备份。"""
    reg_file, runtime_file, backup_file = iso_registry
    reg_file.write_text(
        json.dumps(_legacy_data(with_sessions=False), ensure_ascii=False),
        encoding="utf-8",
    )
    runtime_file.write_text(
        json.dumps(
            {
                "version": 1,
                "members": {"devops": {"sessions": [{"sid": "z", "hops": 1}]}},
            }
        ),
        encoding="utf-8",
    )

    reg = Registry.load()

    assert not backup_file.exists(), "无旧数据可迁 → 不该有备份"
    assert reg.member("devops").sessions[0]["sid"] == "z"


def test_no_cutover_for_fresh_registry_without_sessions(iso_registry):
    """全新配置（成员无 sessions、运行态缺席）→ 不迁移、不写运行态/备份。"""
    reg_file, runtime_file, backup_file = iso_registry
    reg_file.write_text(
        json.dumps(_legacy_data(with_sessions=False), ensure_ascii=False),
        encoding="utf-8",
    )

    Registry.load()

    assert not backup_file.exists()
    assert not runtime_file.exists()


def test_write_member_and_save_route_sessions_to_runtime(iso_registry):
    """write_member 把会话写进运行态、配置面永不含 sessions（save 原子双写）。"""
    reg_file, runtime_file, backup_file = iso_registry
    reg_file.write_text(
        json.dumps(_legacy_data(), ensure_ascii=False), encoding="utf-8"
    )
    reg = Registry.load()

    m = reg.member("devops")
    m.sessions.append(
        {"sid": "bbbb2222", "name": "任务B", "hops": 0, "status": "active"}
    )
    reg.write_member(m)
    reg.save()

    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert "sessions" not in cfg["members"]["devops"]
    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    sids = [s["sid"] for s in rt["members"]["devops"]["sessions"]]
    assert sids == ["aaaa1111", "bbbb2222"]


def test_save_strips_any_embedded_sessions(iso_registry):
    """即便内存里配置仍带旧内嵌 sessions，save 也会剥离（护栏：配置面永不携带运行态）。"""
    reg_file, runtime_file, _backup = iso_registry
    reg_file.write_text(
        json.dumps(_legacy_data(), ensure_ascii=False), encoding="utf-8"
    )
    # 手工构造带内嵌 sessions 的 data + 空运行态，绕过 load 迁移
    reg = Registry(
        data=_legacy_data(), runtime={"version": 1, "members": {}}, path=reg_file
    )
    reg.save()
    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert "sessions" not in cfg["members"]["devops"]


def test_ensure_member_defaults_seeds_runtime_not_config(iso_registry):
    """新成员默认条目：配置面只落低频项，运行态落空会话池。"""
    reg_file, runtime_file, _backup = iso_registry
    reg_file.write_text(
        json.dumps(_legacy_data(with_sessions=False), ensure_ascii=False),
        encoding="utf-8",
    )
    runtime_file.write_text(json.dumps({"version": 1, "members": {}}), encoding="utf-8")
    reg = Registry.load()

    reg.ensure_member_defaults("newpod")
    reg.save()

    cfg = json.loads(reg_file.read_text(encoding="utf-8"))
    assert "sessions" not in cfg["members"]["newpod"]
    assert cfg["members"]["newpod"]["model_tier"] == "flash"
    rt = json.loads(runtime_file.read_text(encoding="utf-8"))
    assert rt["members"]["newpod"]["sessions"] == []


def test_runtime_path_tracks_patched_registry(iso_registry):
    """runtime_path/backup_path 随 self.path 父目录移动（测试只 patch 一处即两面隔离）。"""
    reg_file, runtime_file, backup_file = iso_registry
    reg = Registry(data={}, path=reg_file)
    assert reg.runtime_path == runtime_file
    assert reg.backup_path == backup_file


def test_member_config_fields_untouched_by_split(iso_registry):
    """extra_dirs/readonly 等配置项仍从配置面读（迁移只搬 sessions，不碰其它）。"""
    reg_file, _runtime, _backup = iso_registry
    data = _legacy_data()
    data["members"]["devops"]["extra_dirs"] = ["bench"]
    data["members"]["devops"]["readonly_extra"] = ["refs"]
    reg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    m = Registry.load().member("devops")

    assert m.raw["extra_dirs"] == ["bench"]
    assert m.raw["readonly_extra"] == ["refs"]
    assert isinstance(m, Member)
