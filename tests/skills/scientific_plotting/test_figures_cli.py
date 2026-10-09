"""cmd_new 打印路径的回归测试（backlog 20261009-fig-new-path-mismatch）。

`figures new` 曾在 stdout 用数据侧 figdir 拼出并不存在的「管线模块」路径，
误导组员与任务书白名单声明。此处断言打印的管线模块路径即真实写入位置
（代码侧 figures_code_root/<slug>.py），且附带白名单提示。
"""

from __future__ import annotations

import pytest

from pysci.skills.scientific_plotting.tools import figures


@pytest.fixture
def isolated_roots(tmp_path, monkeypatch):
    """把数据侧/代码侧 figures 根重定向到 tmp，避免写真实项目路径。"""
    import pysci.paths
    from pysci.skills.scientific_plotting.tools import runner, scaffold

    data_root = tmp_path / "data" / "figures"
    code_root = tmp_path / "code" / "figures"
    monkeypatch.setattr(pysci.paths, "research_fig_dir", lambda name: data_root)
    monkeypatch.setattr(runner, "figures_code_root", lambda research: code_root)
    monkeypatch.setattr(scaffold, "figures_root", lambda research: data_root)
    monkeypatch.setattr(scaffold, "figures_code_root", lambda research: code_root)
    return {"data_root": data_root, "code_root": code_root}


def test_new_prints_real_pipeline_path(isolated_roots, capsys):
    rc = figures.main(
        ["new", "gain_ep", "fig9_test", "--style", "aps", "--width", "single"]
    )
    assert rc == 0
    out = capsys.readouterr().out

    pipeline = isolated_roots["code_root"] / "fig9_test.py"
    figdir = isolated_roots["data_root"] / "fig9_test"

    # 真实写入位置与打印一致
    assert pipeline.is_file()
    assert str(pipeline) in out
    # 不再把管线模块误标到数据侧
    assert str(figdir / "fig9_test.py") not in out
    # 白名单提示
    assert "白名单" in out
    # 产物目录（build 目标）仍打印数据侧真实路径
    assert str(figdir) in out
    assert figdir.is_dir()
