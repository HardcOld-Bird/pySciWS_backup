"""imagine CLI 门面单测：子命令接线（args.func）+ 关键 dest + 离线 dry-run 冒烟。

只测纯离线路径（parser 结构、gen/i2i/run --dry-run、palette 抽色）；真实出图属 hardware，
不在此触发。
"""

from __future__ import annotations

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import imagine


def test_parser_wires_commands():
    parser = imagine.build_parser()
    cases = {
        "doctor": imagine.cmd_doctor,
        "ingest": imagine.cmd_ingest,
        "adjust": imagine.cmd_adjust,
        "sheet": imagine.cmd_sheet,
        "gallery": imagine.cmd_gallery,
        "ledger": imagine.cmd_ledger,
        "list": imagine.cmd_list,
        "gen": imagine.cmd_gen,
        "i2i": imagine.cmd_i2i,
        "run": imagine.cmd_run,
        "workflows": imagine.cmd_workflows,
        "palette": imagine.cmd_palette,
        "bridge": imagine.cmd_bridge,
    }
    for cmd, fn in cases.items():
        argv = [cmd]
        # 补齐各命令的 required 参数，仅为解析出 func
        if cmd in ("ingest",):
            argv += ["--src", "x.png"]
        elif cmd == "adjust":
            argv += ["x.png", "--out", "o.png"]
        elif cmd == "sheet":
            argv += ["a.png"]
        elif cmd in ("gen", "i2i"):
            argv += ["--prompt", "p"]
            if cmd == "i2i":
                argv += ["--image", "a.png"]
        elif cmd == "run":
            argv += ["--workflow", "w"]
        elif cmd == "palette":
            argv += ["--src", "x.png"]
        elif cmd == "bridge":
            argv += ["--ref", "x.png", "--research", "r", "--slug", "s"]
        args = parser.parse_args(argv)
        assert args.func is fn, cmd


def test_comfy_nested_subcommands():
    parser = imagine.build_parser()
    assert parser.parse_args(["comfy", "doctor"]).func is imagine.cmd_comfy_doctor
    assert parser.parse_args(["comfy", "nodes"]).func is imagine.cmd_comfy_nodes
    assert parser.parse_args(["comfy", "server", "start"]).func is imagine.cmd_comfy_server_start
    assert parser.parse_args(["comfy", "server", "stop"]).func is imagine.cmd_comfy_server_stop
    assert parser.parse_args(["comfy", "server", "status"]).func is imagine.cmd_comfy_server_status


def test_gen_dest_defaults():
    args = imagine.build_parser().parse_args(["gen", "--prompt", "p"])
    assert args.seed == 0
    assert args.n == 1
    assert args.max_images == 1
    assert args.dry_run is False
    assert args.no_thinking is False
    assert args.group is False


def test_comfy_server_start_cpu_default_true():
    args = imagine.build_parser().parse_args(["comfy", "server", "start"])
    assert args.cpu is True
    args = imagine.build_parser().parse_args(["comfy", "server", "start", "--no-cpu"])
    assert args.cpu is False


def test_i2i_image_is_append():
    args = imagine.build_parser().parse_args(
        ["i2i", "--image", "a.png", "--image", "b.png", "--prompt", "p"]
    )
    assert args.image == ["a.png", "b.png"]


def test_bridge_dest_names():
    args = imagine.build_parser().parse_args(
        ["bridge", "--ref", "r.png", "--research", "demo", "--slug", "fig1",
         "--palette-name", "my-pal", "--no-copy-ref"]
    )
    assert args.palette_name == "my-pal"
    assert args.no_copy_ref is True
    assert args.research == "demo"


# ---------------------------------------------------------------------------
# 离线 main() 冒烟（dry-run / 抽色 / 错误处理）
# ---------------------------------------------------------------------------
def test_main_gen_dry_run(capsys):
    rc = imagine.main(["gen", "--prompt", "a cover", "--seed", "3", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "JimengSeedream4" in out
    assert "--dry-run" in out


def test_main_i2i_dry_run_no_file_check(capsys):
    # dry-run 不检查源图存在性（用本地文件名占位）
    rc = imagine.main(
        ["i2i", "--image", "nope_a.png", "--image", "nope_b.png", "--prompt", "p", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "LoadImage" in out


def test_main_gen_bad_provider_returns_1(capsys):
    rc = imagine.main(["gen", "--provider", "does-not-exist", "--prompt", "p", "--dry-run"])
    assert rc == 1
    assert "gen 失败" in capsys.readouterr().err


def test_main_palette_on_real_image(tmp_path, capsys):
    img = tmp_path / "ref.png"
    Image.new("RGB", (30, 30), (12, 130, 200)).save(img)
    rc = imagine.main(["palette", "--src", str(img), "--n", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "主色" in out and "#" in out


def test_main_palette_missing_returns_1(capsys):
    rc = imagine.main(["palette", "--src", "definitely_missing.png"])
    assert rc == 1
    assert "palette 失败" in capsys.readouterr().err


def test_main_run_bad_args_json_returns_1(tmp_path, monkeypatch, capsys):
    # 造一个已存工作流，喂非法 --args JSON
    import dataclasses

    from pysci.skills.ai_drawing.tools import workflows as wf

    monkeypatch.setattr(wf, "settings", dataclasses.replace(wf.settings, workflows_dir=tmp_path))
    monkeypatch.setattr(imagine, "settings", dataclasses.replace(imagine.settings, workflows_dir=tmp_path))
    wf.save_workflow(wf.txt2img_seedream("x"), "w")
    rc = imagine.main(["run", "--workflow", "w", "--args", "{bad json"])
    assert rc == 1
    assert "不是合法 JSON" in capsys.readouterr().err


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        imagine.main([])
