"""figures CLI 薄路由机械断言：子命令分发 / audit 退出码 / field-tick 换算。"""

from __future__ import annotations

import pytest

from pysci.skills.scientific_plotting import cli


def test_styles_command_lists_assets_and_contract(capsys):
    assert cli.main(["styles"]) == 0
    out = capsys.readouterr().out
    assert "journal-aps" in out
    assert "journal-nature" in out
    assert "single 85" in out and "double 170" in out  # APS 尺寸契约
    assert "89" in out and "183" in out  # Nature 尺寸契约
    assert "0072B2" in out  # 家风色循环


def test_audit_command_pass_on_embedded_pdf(tmp_path, capsys):
    pdf = tmp_path / "fig.pdf"
    pdf.write_bytes(b"%PDF /FontFile2")
    assert cli.main(["audit", str(pdf)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_audit_command_fail_on_unembedded_pdf(tmp_path, capsys):
    pdf = tmp_path / "fig.pdf"
    pdf.write_bytes(b"%PDF /Font /Foo")
    assert cli.main(["audit", str(pdf)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_audit_command_width_contract(tmp_path, capsys):
    pdf = tmp_path / "fig.pdf"
    pdf.write_bytes(b"%PDF /FontFile2")
    code = cli.main(
        [
            "audit",
            str(pdf),
            "--journal",
            "aps",
            "--span",
            "double",
            "--actual-width-mm",
            "170.5",
        ]
    )
    assert code == 0
    assert "width" in capsys.readouterr().out


def test_field_tick_command(capsys):
    assert cli.main(["field-tick", "65.48", "0.6436"]) == 0
    out = capsys.readouterr().out
    assert "65.48 -> 70" in out
    assert "0.6436 -> 0.65" in out


def test_field_tick_command_bad_value(capsys):
    assert cli.main(["field-tick", "0"]) == 1
    assert "错误" in capsys.readouterr().err


def test_doctor_command_runs(capsys):
    # 缺失项只提示；退出码仅当必需件（ultraplot/matplotlib）缺失才 1
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "ultraplot" in out and "pyvista" in out
    assert code in (0, 1)


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["bogus"])
    assert exc.value.code == 2
