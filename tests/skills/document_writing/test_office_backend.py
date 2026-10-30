"""office_convert 的后端选择：**wpscli 优先，LibreOffice 兜底**（backlog 20261017-105200）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.document_writing.tools import config, office_convert


class _Proc:
    def __init__(self, rc: int = 0):
        self.returncode = rc
        self.stdout = ""
        self.stderr = ""


@pytest.fixture
def bins(monkeypatch):
    """可调的后端探测：返回 setter(bins=(wps|None, soffice|None))。"""

    def _set(*, wps: str | None, soffice: str | None):
        monkeypatch.setattr(config.Settings, "find_wpscli", lambda self: wps)
        monkeypatch.setattr(config.Settings, "find_libreoffice", lambda self: soffice)

    return _set


def _src(tmp_path: Path) -> Path:
    p = tmp_path / "deck.pptx"
    p.write_bytes(b"x")
    return p


def _fake_run(recorder: list, *, wps_rc: int = 0):
    """假 subprocess：记录命令；wps 按 ``wps_rc`` 决定是否产出，soffice 总产出。"""

    def run(cmd, timeout=None, **kw):
        recorder.append(list(cmd))
        if Path(cmd[0]).name.lower().startswith("wpscli"):
            if wps_rc != 0:
                return _Proc(wps_rc)
            out = Path(cmd[2]).parent / f"{Path(cmd[2]).stem}.pdf"
        else:
            out = Path(cmd[cmd.index("--outdir") + 1]) / f"{Path(cmd[-1]).stem}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"%PDF-1.4")
        return _Proc(0)

    return run


def test_auto_prefers_wpscli(tmp_path, monkeypatch, bins):
    bins(wps="wpscli.exe", soffice="soffice.exe")
    calls: list = []
    monkeypatch.setattr(office_convert, "_run", _fake_run(calls))

    out, backend = office_convert.convert_ex(_src(tmp_path), "pdf")

    assert backend == "wpscli"
    assert out.name == "deck.pdf" and out.exists()
    assert calls[0][0] == "wpscli.exe" and calls[0][1] == "ppt2pdf"


def test_fallback_when_wpscli_missing(tmp_path, monkeypatch, bins):
    bins(wps=None, soffice="soffice.exe")
    calls: list = []
    monkeypatch.setattr(office_convert, "_run", _fake_run(calls))

    _, backend = office_convert.convert_ex(_src(tmp_path), "pdf")
    assert backend == "libreoffice"
    assert calls[0][0] == "soffice.exe"


def test_wpscli_failure_falls_back(tmp_path, monkeypatch, bins):
    bins(wps="wpscli.exe", soffice="soffice.exe")
    calls: list = []
    monkeypatch.setattr(office_convert, "_run", _fake_run(calls, wps_rc=1))

    _, backend = office_convert.convert_ex(_src(tmp_path), "pdf")
    assert backend == "libreoffice", "wpscli 失败必须回落"
    assert [c[0] for c in calls] == ["wpscli.exe", "soffice.exe"]


def test_explicit_backend_skips_wps(tmp_path, monkeypatch, bins):
    bins(wps="wpscli.exe", soffice="soffice.exe")
    calls: list = []
    monkeypatch.setattr(office_convert, "_run", _fake_run(calls))

    _, backend = office_convert.convert_ex(_src(tmp_path), "pdf", backend="libreoffice")
    assert backend == "libreoffice"
    assert calls[0][0] == "soffice.exe" and len(calls) == 1


def test_no_backend_raises(tmp_path, monkeypatch, bins):
    bins(wps=None, soffice=None)
    with pytest.raises(office_convert.LibreOfficeNotInstalled):
        office_convert.convert_ex(_src(tmp_path), "pdf")
