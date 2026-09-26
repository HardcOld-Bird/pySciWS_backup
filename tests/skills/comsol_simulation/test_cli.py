"""simulation 统一 CLI 的纯 Python 冒烟：不启动 COMSOL 的子命令（doctor / build recipes / docs list）
与 argparse 行为。用 monkeypatch 把 docs.settings 指到临时目录，绝不触碰真实 doc_index.db。

需要活体 COMSOL 的子命令（inspect/run/export/build apply）由 hardware 冒烟覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pysci.skills.comsol_simulation.tools import docs, simulation


@pytest.fixture(autouse=True)
def _isolate_docs(tmp_path, monkeypatch):
    """把 docs 的数据区指到 tmp，避免 CLI 测试读写真实索引库。"""
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(
        docs,
        "settings",
        SimpleNamespace(
            docs_dir=tmp_path / "docs",
            cache_dir=cache,
            doc_index_db=cache / "doc_index.db",
        ),
    )


def test_doctor_returns_zero(capsys):
    rc = simulation.main(["doctor"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "comsol_simulation configuration" in out
    assert "mph importable" in out


def test_build_recipes_lists_builtin(capsys):
    rc = simulation.main(["build", "recipes"])
    assert rc == 0
    assert "waveguide2d_acpr" in capsys.readouterr().out


def test_docs_list_empty_ok(capsys):
    # 空索引也算成功（返回 0，不报错）
    assert simulation.main(["docs", "list"]) == 0


def test_docs_search_empty_index(capsys):
    assert simulation.main(["docs", "search", "anything"]) == 0
    assert "no hits" in capsys.readouterr().out


def test_no_subcommand_exits():
    with pytest.raises(SystemExit):
        simulation.main([])


def test_bad_subcommand_exits():
    with pytest.raises(SystemExit):
        simulation.main(["__nope__"])
