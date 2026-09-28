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


# ---------------------------------------------------------------------------
# 新增子命令的 argparse 接线（不启 COMSOL，仅解析）
# ---------------------------------------------------------------------------
def test_parser_inspect_node():
    args = simulation.build_parser().parse_args(
        ["inspect", "node", "--mph", "m.mph", "--path", "component(comp1)", "--methods"]
    )
    assert args.func is simulation.cmd_inspect_node
    assert args.path == "component(comp1)" and args.methods


def test_parser_export_image_extent_clean_sidecar():
    args = simulation.build_parser().parse_args(
        ["export", "image", "--mph", "m", "--plotgroup", "pg", "--out", "o.png",
         "--extent", "0", "1", "-1", "1", "--clean"]
    )
    assert args.func is simulation.cmd_export_image
    assert args.extent == [0.0, 1.0, -1.0, 1.0]
    assert args.clean and not args.no_sidecar


def test_parser_post_framebox():
    args = simulation.build_parser().parse_args(
        ["post", "framebox", "--image", "i.png", "--sidecar", "s.json"]
    )
    assert args.func is simulation.cmd_post_framebox
    assert args.sidecar.name == "s.json"


def test_parser_node_set():
    args = simulation.build_parser().parse_args(
        ["node", "set", "--mph", "m", "--path", "result(pg).feature(surf1)",
         "--set", "rangecoloractive=on", "--set", "rangecolormin=-160", "--save", "o.mph"]
    )
    assert args.func is simulation.cmd_node_set
    assert args.path == "result(pg).feature(surf1)"
    assert args.set == ["rangecoloractive=on", "rangecolormin=-160"]
    assert args.save.name == "o.mph"


def test_parser_node_set_requires_set():
    with pytest.raises(SystemExit):
        simulation.build_parser().parse_args(["node", "set", "--mph", "m", "--path", "result(pg)"])


def test_parser_export_image_scale_flags():
    args = simulation.build_parser().parse_args(
        ["export", "image", "--mph", "m", "--plotgroup", "pg", "--out", "o.png",
         "--color-range", "-160", "160", "--polar-rmax", "60",
         "--geom-bbox", "-0.4", "0.4", "-0.2", "0.4"]
    )
    assert args.func is simulation.cmd_export_image
    assert args.color_range == [-160.0, 160.0]
    assert args.polar_rmax == 60.0
    assert args.geom_bbox == [-0.4, 0.4, -0.2, 0.4]


def test_parser_diagnose():
    args = simulation.build_parser().parse_args(["diagnose", "--mph", "m.mph"])
    assert args.func is simulation.cmd_diagnose
    assert args.mph.name == "m.mph"


def test_parser_diagnose_requires_mph():
    with pytest.raises(SystemExit):
        simulation.build_parser().parse_args(["diagnose"])


# ---------------------------------------------------------------------------
# server 子命令（跨进程常驻会话）——parser + 无状态文件行为（不启 JVM）
# ---------------------------------------------------------------------------
def test_parser_server_start_stop_status():
    args = simulation.build_parser().parse_args(["server", "start", "--port", "2100", "--cores", "2"])
    assert args.func is simulation.cmd_server_start
    assert args.port == 2100 and args.cores == 2
    args = simulation.build_parser().parse_args(["server", "stop"])
    assert args.func is simulation.cmd_server_stop
    args = simulation.build_parser().parse_args(["server", "status", "--port", "2100"])
    assert args.func is simulation.cmd_server_status and args.port == 2100


def test_parser_connect_port_global_flag():
    args = simulation.build_parser().parse_args(
        ["--connect-port", "2036", "inspect", "tree", "--mph", "m.mph"]
    )
    assert args.connect_port == 2036
    assert args.func is simulation.cmd_inspect_tree
    # 不传时默认 None
    args = simulation.build_parser().parse_args(["doctor"])
    assert args.connect_port is None


def test_server_status_no_state_not_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(simulation._session, "settings", SimpleNamespace(runs_dir=tmp_path))
    assert simulation.main(["server", "status"]) == 0
    out = capsys.readouterr().out
    assert "running    : False" in out


def test_server_stop_no_state_is_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(simulation._session, "settings", SimpleNamespace(runs_dir=tmp_path))
    assert simulation.main(["server", "stop"]) == 0
    assert "未运行" in capsys.readouterr().out
