"""office_convert 测试：输入校验 + 缺失 LibreOffice 时优雅降级。"""

from __future__ import annotations

import pytest

from pysci.skills.document_writing.tools import office_convert
from pysci.skills.document_writing.tools.config import settings


def test_missing_src_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        office_convert.convert(tmp_path / "nope.pptx", "pdf")


def test_not_installed_graceful(tmp_path):
    if settings.find_libreoffice():
        pytest.skip("本机已装 LibreOffice，降级路径不适用")
    src = tmp_path / "x.pptx"
    src.write_bytes(b"pk")
    with pytest.raises(office_convert.LibreOfficeNotInstalled):
        office_convert.convert(src, "pdf")


def test_find_libreoffice_env_override(tmp_path, monkeypatch):
    """.env 的 DOCWRITING_SOFFICE 应优先于一切探测（自定义 D 盘安装路径）。"""
    fake = tmp_path / "soffice.exe"
    fake.write_bytes(b"MZ")
    monkeypatch.setenv("DOCWRITING_SOFFICE", str(fake))
    assert settings.find_libreoffice() == str(fake)
