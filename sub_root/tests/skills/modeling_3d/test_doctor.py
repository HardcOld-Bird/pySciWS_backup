"""doctor 机械断言：版本/tasklist 纯解析、Blender 定位注入、内存预算常量。"""

from __future__ import annotations

import pytest

from pysci.skills.modeling_3d import doctor


def test_parse_blender_version():
    assert doctor.parse_blender_version("Blender 4.5.3\n\tbuild date: 2025-01-01") == "4.5.3"
    assert doctor.parse_blender_version("Blender 5.1.0 Alpha") == "5.1.0"
    assert doctor.parse_blender_version("garbage output") is None


def test_parse_tasklist_pids():
    out = (
        '"blender.exe","1234","Console","1","120,000 K"\r\n'
        '"blender.exe","5678","Console","1","80,000 K"\r\n'
        '"python.exe","9012","Console","1","10,000 K"\r\n'
        'INFO: 没有运行的任务匹配指定标准。\r\n'
    )
    assert doctor.parse_tasklist_pids(out) == [1234, 5678]
    assert doctor.parse_tasklist_pids('"blender.exe","42",...') == [42]


def test_find_blender_env_points_to_exe(tmp_path):
    exe = tmp_path / "blender.exe"
    exe.write_bytes(b"stub")
    assert doctor.find_blender(env=str(exe)) == exe


def test_find_blender_env_points_to_dir(tmp_path):
    sub = tmp_path / "blender-4.5"
    sub.mkdir()
    exe = sub / "blender.exe"
    exe.write_bytes(b"stub")
    found = doctor.find_blender(env=str(tmp_path))
    assert found == exe


def test_find_blender_vendor_layouts(tmp_path):
    # 布局 1：vendor 根直接放 exe（env="" 显式绕过真实环境变量）
    direct = tmp_path / "a" / "blender.exe"
    direct.parent.mkdir()
    direct.write_bytes(b"stub")
    assert doctor.find_blender(env="", vendor_dir=direct.parent) == direct
    # 布局 2：单层版本子目录（并行代理 vendor 的常见形态）
    root = tmp_path / "b"
    sub = root / "blender-4.5.3-windows-x64"
    sub.mkdir(parents=True)
    exe = sub / "blender.exe"
    exe.write_bytes(b"stub")
    assert doctor.find_blender(env="", vendor_dir=root) == exe


def test_find_blender_not_found(tmp_path):
    assert doctor.find_blender(env="", vendor_dir=tmp_path / "empty") is None


def test_ensure_single_instance_raises_when_busy(monkeypatch):
    monkeypatch.setattr(doctor, "blender_processes", lambda: [4242])
    with pytest.raises(RuntimeError, match="禁并行"):
        doctor.ensure_single_instance()
    monkeypatch.setattr(doctor, "blender_processes", lambda: [])
    doctor.ensure_single_instance()  # 无实例不抛


def test_memory_budget_constant():
    assert doctor.MEMORY_BUDGET_GB == 5.0  # 16GB 机型 ~5GB 约束（计划口径）


def test_available_memory_gb_shape():
    val = doctor.available_memory_gb()
    assert val is None or val > 0


def test_doctor_report_without_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "BLENDER_VENDOR_DIR", tmp_path / "none")
    monkeypatch.delenv(doctor.BLENDER_ENV, raising=False)
    rep = doctor.doctor(probe_version=False)
    assert rep.blender_path is None
    assert not rep.ok
    assert any("NOT FOUND" in n or "未找到" in n for n in rep.notes)
    assert "CYCLES" in rep.format()
