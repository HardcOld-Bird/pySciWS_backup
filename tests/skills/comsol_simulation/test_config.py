"""config 纯 Python 单测：安装发现的路径推导、.env 覆盖、资源护栏、目录创建、摘要脱敏。

不启动 JVM、不占 license —— 只验证 discover/_derive/_get_env*/build_settings 的确定性行为。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.comsol_simulation.tools import config


def _fake_comsol_root(tmp_path: Path, version: str = "6.4") -> Path:
    """搭一个最小可识别的 COMSOL 安装树（bin/win64 + doc/pdf + java jvm.dll）。"""
    root = tmp_path / "comsol" / version / "base"
    bindir = root / "bin" / "win64"
    bindir.mkdir(parents=True)
    (bindir / "comsolmphserver.exe").write_bytes(b"")
    (bindir / "comsolbatch.exe").write_bytes(b"")
    (root / "doc" / "pdf" / "COMSOL_Multiphysics").mkdir(parents=True)
    jvm_dir = root / "java" / "win64" / "jre" / "bin" / "server"
    jvm_dir.mkdir(parents=True)
    (jvm_dir / "jvm.dll").write_bytes(b"")
    return root


def test_derive_from_root_finds_paths(tmp_path: Path):
    root = _fake_comsol_root(tmp_path)
    inst = config._derive_from_root(root, source="test")
    assert inst.found is True
    assert inst.version == "6.4"
    assert inst.root == root
    assert inst.server_exe is not None and inst.server_exe.name == "comsolmphserver.exe"
    assert inst.batch_exe is not None and inst.batch_exe.name == "comsolbatch.exe"
    assert inst.jvm is not None and inst.jvm.name == "jvm.dll"
    assert inst.doc_pdf_dir is not None and inst.doc_pdf_dir.exists()
    assert inst.source == "test"


def test_derive_from_root_missing(tmp_path: Path):
    inst = config._derive_from_root(tmp_path / "nope", source="test")
    assert inst.found is False
    assert inst.root is None
    assert inst.server_exe is None
    assert inst.doc_pdf_dir is None


def test_discover_comsol_env_override(tmp_path: Path):
    root = _fake_comsol_root(tmp_path)
    inst = config.discover_comsol(install_dir=str(root))
    assert inst.found is True
    assert inst.source == "env"
    assert inst.root == root


def test_discover_comsol_nonexistent_dir_falls_back(tmp_path: Path, capsys):
    inst = config.discover_comsol(install_dir=str(tmp_path / "does_not_exist"))
    assert isinstance(inst, config.ComsolInstall)
    # 显式目录不存在 → 绝不用 "env"，回退自动发现（mph-discovery 或 none）
    assert inst.source in ("mph-discovery", "none")
    assert "COMSOL_INSTALL_DIR" in capsys.readouterr().err


def test_manual_path_and_priority(tmp_path: Path):
    root = _fake_comsol_root(tmp_path)
    prog = (
        root / "doc" / "pdf" / "COMSOL_Multiphysics" / "COMSOL_ProgrammingReferenceManual.pdf"
    )
    prog.write_bytes(b"%PDF-1.4 fake")
    inst = config._derive_from_root(root, source="test")
    assert inst.manual_path("COMSOL_Multiphysics/COMSOL_ProgrammingReferenceManual.pdf") == prog
    assert inst.manual_path("COMSOL_Multiphysics/DoesNotExist.pdf") is None
    prios = inst.priority_manuals()
    assert prog in prios
    assert all(p.exists() for p in prios)


def test_manual_path_none_when_no_doc_dir(tmp_path: Path):
    inst = config.ComsolInstall(found=False, doc_pdf_dir=None)
    assert inst.manual_path("anything.pdf") is None
    assert inst.priority_manuals() == []


def test_get_env_int(monkeypatch):
    monkeypatch.setenv("COMSOL_MAX_CORES", "8")
    assert config._get_env_int("COMSOL_MAX_CORES", 4) == 8
    monkeypatch.setenv("COMSOL_MAX_CORES", "not-an-int")
    assert config._get_env_int("COMSOL_MAX_CORES", 4) == 4  # 非整数 → 回退默认
    monkeypatch.delenv("COMSOL_MAX_CORES", raising=False)
    assert config._get_env_int("COMSOL_MAX_CORES", 4) == 4  # 未设置 → 默认


def test_get_env_blank_is_unset(monkeypatch):
    monkeypatch.setenv("COMSOL_INSTALL_DIR", "   ")
    assert config._get_env("COMSOL_INSTALL_DIR") is None
    monkeypatch.setenv("COMSOL_INSTALL_DIR", "x")
    assert config._get_env("COMSOL_INSTALL_DIR") == "x"


def test_settings_dirs_exist():
    s = config.settings
    for d in (
        s.docs_dir,
        s.cache_dir,
        s.recipes_dir,
        s.templates_dir,
        s.knowledge_dir,
        s.runs_dir,
        s.tempdir,
    ):
        assert d.is_dir(), d
    assert s.module_dir == config.MODULE_DIR
    assert s.doc_index_db == s.cache_dir / "doc_index.db"
    assert s.tempdir == s.runs_dir / "tmp"
    assert s.comsol_max_cores >= 1


def test_settings_summary_masks_token():
    s = config.settings
    txt = s.summary()
    assert "comsol_found" in txt
    assert "comsol_max_cores" in txt
    if s.mineru_token:
        assert s.mineru_token not in txt  # token 脱敏，绝不整串出现
        assert s.mineru_ready is True
