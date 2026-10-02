"""imagine CLI 门面单测：子命令接线（args.func）+ 关键 dest + 离线执行冒烟。

覆盖三件事，**全部离线**：
1. parser 结构：每个子命令是否接到正确的 ``cmd_*``，required 参数是否到位；
2. 云端出图的参数护栏是否被 CLI 正确转述（dry-run 打印脱敏请求体）；
3. "出图 → 落盘 → 内嵌元数据 → 记账 → 渲染账本"的接线（用假的 ArkResponse 顶替网络）。

真实出图属 hardware（会花钱），不在此触发。
"""

from __future__ import annotations

import base64
import dataclasses
import io
from pathlib import Path

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import ark_client as _ark
from pysci.skills.ai_drawing.tools import imagine
from pysci.skills.ai_drawing.tools import ledger as _ledger

# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
_DATA_DIRS = ("assets", "gallery", "recipes", "prompts", "cache", "runs")


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """把技能数据区整体重定向到 tmp_path，并放开 data/ 落点护栏。

    settings 的字段名带 ``_dir`` 后缀，但**目录名用自然名**（``assets/`` 等）——这样用例里
    ``tmp_env / "assets"`` 读起来与真实数据区一致，不会因为两边名字差一个后缀而误判。

    护栏本身由 ``tests/test_paths.py`` 覆盖；这里测的是接线，不得往真实 assets/
    或真实账本里写东西。
    """
    dirs = {f"{name}_dir": tmp_path / name for name in _DATA_DIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for mod in (imagine, _ledger):
        monkeypatch.setattr(
            mod,
            "settings",
            dataclasses.replace(mod.settings, module_dir=tmp_path, **dirs),
        )
    monkeypatch.setattr(imagine, "assert_within_data", lambda p, **kw: Path(p))
    return tmp_path


def _png_bytes(color=(200, 30, 30), size=(4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


#: 组图是**按模型的契约能力**：默认档（5.0-flash）不支持组图，故组图接线测试显式指定
#: 一个契约上支持组图的模型，避免测试意图与账号默认值耦合。
GROUP_CAPABLE_MODEL = "doubao-seedream-4-5-251128"


def _fake_response(
    n: int = 1, *, model: str = "doubao-seedream-4-0-250828"
) -> _ark.ArkResponse:
    """造一个 n 张 PNG（b64_json）的响应，顶替真实云端返回。"""
    b64 = base64.b64encode(_png_bytes()).decode("ascii")
    data = [{"b64_json": b64, "size": "4x4"} for _ in range(n)]
    return _ark.ArkResponse(
        model=model,
        created=1700000000,
        images=[_ark.ArkImage(index=i, b64_json=b64, size="4x4") for i in range(n)],
        usage={"generated_images": n, "output_tokens": 10 * n, "total_tokens": 10 * n},
        request={"model": model, "prompt": "p", "watermark": False},
        raw={
            "model": model,
            "created": 1700000000,
            "data": data,
            "usage": {"generated_images": n},
        },
    )


# ---------------------------------------------------------------------------
# parser 接线
# ---------------------------------------------------------------------------
def test_parser_wires_commands():
    parser = imagine.build_parser()
    cases = {
        "doctor": imagine.cmd_doctor,
        "models": imagine.cmd_models,
        "list": imagine.cmd_list,
        "ledger": imagine.cmd_ledger,
        "gallery": imagine.cmd_gallery,
        "gen": imagine.cmd_gen,
        "i2i": imagine.cmd_i2i,
        "edit": imagine.cmd_edit,
        "layers": imagine.cmd_layers,
        "mark": imagine.cmd_mark,
        "adjust": imagine.cmd_adjust,
        "sheet": imagine.cmd_sheet,
        "img": imagine.cmd_img,
        "ingest": imagine.cmd_ingest,
        "palette": imagine.cmd_palette,
        "bridge": imagine.cmd_bridge,
    }
    for cmd, fn in cases.items():
        argv = [cmd]
        # 补齐各命令的 required 参数，仅为解析出 func
        if cmd in ("ingest", "mark", "palette"):
            argv += ["--src", "x.png"]
        elif cmd == "adjust":
            argv += ["x.png", "--out", "o.png"]
        elif cmd == "sheet":
            argv += ["a.png"]
        elif cmd == "img":
            argv += ["split", "x.png"]
        elif cmd in ("gen",):
            argv += ["--prompt", "p"]
        elif cmd in ("i2i", "edit", "layers"):
            argv += ["--image", "a.png", "--prompt", "p"]
        elif cmd == "bridge":
            argv += ["--ref", "x.png", "--research", "r", "--slug", "s"]
        args = parser.parse_args(argv)
        assert args.func is fn, cmd


def test_comfy_subcommands_are_gone():
    """重构后不应再有 comfy / run / workflows 子命令（死代码已整层移除）。"""
    parser = imagine.build_parser()
    for gone in ("comfy", "run", "workflows"):
        with pytest.raises(SystemExit):
            parser.parse_args([gone])
        assert not hasattr(imagine, f"cmd_{gone}")


def test_img_nested_subcommands():
    parser = imagine.build_parser()
    for op in (
        "split",
        "composite",
        "fuse",
        "inpaint",
        "mask",
        "morph",
        "warp",
        "measure",
        "align",
    ):
        argv = ["img", op]
        if op == "composite":
            argv += ["--base", "b.png", "--layer", "l.png"]
        elif op == "fuse":
            argv += ["s.png", "--base", "b.png"]
        elif op == "inpaint":
            argv += ["s.png", "--mask", "m.png"]
        elif op == "align":
            argv += ["s.png", "--ref", "r.png"]
        else:
            argv += ["s.png"]
        args = parser.parse_args(argv)
        assert args.func is imagine.cmd_img, op
        assert args.img_cmd == op


def test_gen_dest_defaults():
    """默认值即护栏：不出组图、不加水印、不要 png、走 b64_json。"""
    args = imagine.build_parser().parse_args(["gen", "--prompt", "p"])
    assert args.seed is None  # 不伪造"可复现"假象
    assert args.n == 1
    assert args.max_images is None
    assert args.group is False
    assert args.watermark is False
    assert args.output_format is None
    assert args.response_format == "b64_json"
    assert args.web_search is False
    assert args.dry_run is False


def test_i2i_image_is_append():
    args = imagine.build_parser().parse_args(
        ["i2i", "--image", "a.png", "--image", "b.png", "--prompt", "p"]
    )
    assert args.image == ["a.png", "b.png"]


def test_ingest_uses_recipe_not_workflow():
    args = imagine.build_parser().parse_args(
        ["ingest", "--src", "x.png", "--recipe", "cover_v1"]
    )
    assert args.recipe == "cover_v1"
    assert not hasattr(args, "workflow")


def test_bridge_dest_names():
    args = imagine.build_parser().parse_args(
        [
            "bridge",
            "--ref",
            "r.png",
            "--research",
            "demo",
            "--slug",
            "fig1",
            "--palette-name",
            "my-pal",
            "--no-copy-ref",
        ]
    )
    assert args.palette_name == "my-pal"
    assert args.no_copy_ref is True
    assert args.research == "demo"


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        imagine.main([])


# ---------------------------------------------------------------------------
# 自检 / 清单（离线、无 key 也要能用）
# ---------------------------------------------------------------------------
def test_main_doctor_without_key(capsys, monkeypatch):
    """未配 ARK_API_KEY 时 doctor 不崩，且给出可执行的下一步。"""
    s = dataclasses.replace(imagine.settings, ark_api_key=None)
    monkeypatch.setattr(imagine, "settings", s)
    assert imagine.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "images/generations" in out
    assert "ARK_API_KEY" in out
    assert "***" not in out  # 无 key 时不该出现脱敏占位


def test_main_models_prints_matrix(capsys):
    assert imagine.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "doubao-seedream-4-0-250828" in out
    assert imagine.PRO_MODEL_HINT in out
    assert "不保证完全一致" in out  # 诚实标注 seed 的弱语义
    assert "models --live" in out  # 矩阵顶部必须指一条核对真值的路


def test_main_models_live_reads_ark(monkeypatch, capsys):
    """--live 走 GET /models；未收录的 ID 要显式标出，不假装本地表是全的。"""
    unseen = {"id": "doubao-seedream-9-9-999999"}
    monkeypatch.setattr(imagine._ark, "list_models", lambda **kw: [unseen])
    assert imagine.main(["models", "--live"]) == 0
    out = capsys.readouterr().out
    assert unseen["id"] in out and "本地矩阵未收录" in out


def test_main_models_live_error_returns_1(monkeypatch, capsys):
    def boom(**kw):
        raise imagine._ark.ArkError("未配置 ARK_API_KEY", status_code=401)

    monkeypatch.setattr(imagine._ark, "list_models", boom)
    assert imagine.main(["models", "--live"]) == 1
    assert "models --live 失败" in capsys.readouterr().err


def test_main_list_offline(tmp_env, capsys):
    assert imagine.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "assets" in out and "gallery" in out and "recipes" in out


def test_main_ledger_render_and_stats(tmp_env, capsys):
    _ledger.record(prompt="p1", backend="ark", model="m1")
    assert imagine.main(["ledger", "--render"]) == 0
    assert (tmp_env / "LEDGER.md").is_file()
    capsys.readouterr()
    assert imagine.main(["ledger", "--stats"]) == 0
    assert "总记录数：1" in capsys.readouterr().out
    # 默认路径（查询）也会先渲染，保证 Read LEDGER.md 总是新鲜的
    assert imagine.main(["ledger"]) == 0
    assert "p1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dry-run：请求体护栏（不联网、不计费）
# ---------------------------------------------------------------------------
def test_main_gen_dry_run(capsys):
    rc = imagine.main(["gen", "--prompt", "a cover", "--size", "1K", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "images/generations" in out
    assert "--dry-run" in out
    # 方舟 watermark 默认 true → 请求体里必须显式 false
    assert '"watermark": false' in out
    assert '"response_format": "b64_json"' in out
    assert '"size": "1K"' in out


def test_main_gen_dry_run_omits_unset_seed(capsys):
    """seed 未给就不进请求体（给了反而制造"可复现"的错觉）。"""
    imagine.main(["gen", "--prompt", "p", "--dry-run"])
    out = capsys.readouterr().out
    assert '"seed"' not in out  # 不能直接查 "seed"：模型名 seedream 也包含它


def test_main_i2i_dry_run_redacts_data_uri(tmp_path, capsys):
    """本地参考图自动转 data URI，但打印时截断（否则刷屏 + 泄露进日志）。"""
    src = tmp_path / "ref.png"
    src.write_bytes(_png_bytes())
    rc = imagine.main(["i2i", "--image", str(src), "--prompt", "p", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "data:image/png;base64," in out
    assert "…<data-uri " in out
    assert len(out) < 4000  # 未把整段 base64 打出来


def test_main_i2i_dry_run_passes_url_through(capsys):
    imagine.main(["i2i", "--image", "https://x/a.png", "--prompt", "p", "--dry-run"])
    assert "https://x/a.png" in capsys.readouterr().out


def test_main_gen_dry_run_group(capsys):
    imagine.main(
        [
            "gen",
            "--prompt",
            "p",
            "--model",
            GROUP_CAPABLE_MODEL,
            "--group",
            "--max-images",
            "4",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert '"sequential_image_generation": "auto"' in out
    assert '"max_images": 4' in out


def test_main_gen_dry_run_web_search(capsys):
    imagine.main(["gen", "--prompt", "p", "--web-search", "--dry-run"])
    assert '"type": "web_search"' in capsys.readouterr().out


def test_main_edit_dry_run_defaults_to_pro(capsys):
    """edit/layers 需要 5.0-pro；未显式给 --model 时自动切。"""
    imagine.main(["edit", "--image", "https://x/m.png", "--prompt", "p", "--dry-run"])
    assert imagine.PRO_MODEL_HINT in capsys.readouterr().out


def test_main_layers_dry_run_defaults_to_pro(capsys):
    imagine.main(["layers", "--image", "https://x/m.png", "--prompt", "p", "--dry-run"])
    assert imagine.PRO_MODEL_HINT in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 参数护栏：错误必须返回 1 + 人话，而不是 traceback
# ---------------------------------------------------------------------------
def test_main_gen_seed_out_of_range_returns_1(capsys):
    rc = imagine.main(["gen", "--prompt", "p", "--seed", "70000", "--dry-run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "参数校验失败" in err and "seed" in err


def test_main_gen_png_on_non_main_50_returns_1(capsys):
    """output_format=png 仅 5.0 主档支持；提前拦下并给出 --live / --extra 两条出路。"""
    rc = imagine.main(["gen", "--prompt", "p", "--output-format", "png", "--dry-run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "5.0 主档" in err and "models --live" in err


def test_main_gen_png_on_main_50_dry_run_ok(capsys):
    """真实 5.0 主档 ID（**无** lite 后缀）必须被放行。

    回归防护：早前用昵称 "5-0-lite" 做子串匹配，把合法 ID 误拒在本地护栏上。
    """
    rc = imagine.main(
        [
            "gen",
            "--prompt",
            "p",
            "--model",
            "doubao-seedream-5-0-260128",
            "--output-format",
            "png",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert '"output_format": "png"' in capsys.readouterr().out


def test_main_gen_group_on_main_50_dry_run_ok(capsys):
    """组图同理：真实 5.0 主档 ID 不得被本地护栏误拒。"""
    rc = imagine.main(
        [
            "gen",
            "--prompt",
            "p",
            "--model",
            "doubao-seedream-5-0-260128",
            "--group",
            "--max-images",
            "2",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert '"sequential_image_generation": "auto"' in capsys.readouterr().out


def test_main_gen_group_on_flash_returns_1(capsys):
    """默认档 flash 按契约不支持组图：本地拦下（rc=1），不把请求发给方舟。"""
    rc = imagine.main(
        ["gen", "--prompt", "p", "--group", "--max-images", "2", "--dry-run"]
    )
    assert rc == 1
    assert "不支持组图" in capsys.readouterr().err


def test_main_gen_bad_extra_json_returns_1(capsys):
    rc = imagine.main(["gen", "--prompt", "p", "--extra", "{bad json", "--dry-run"])
    assert rc == 1
    assert "合法 JSON" in capsys.readouterr().err


def test_main_gen_extra_must_be_object(capsys):
    rc = imagine.main(["gen", "--prompt", "p", "--extra", "[1,2]", "--dry-run"])
    assert rc == 1
    assert "JSON 对象" in capsys.readouterr().err


def test_main_i2i_missing_ref_returns_1(capsys):
    """dry-run 也要读本地参考图（要转 base64）→ 文件不存在给一句人话。"""
    rc = imagine.main(
        ["i2i", "--image", "definitely_missing.png", "--prompt", "p", "--dry-run"]
    )
    assert rc == 1
    assert "参数校验失败" in capsys.readouterr().err


def test_main_gen_empty_prompt_rejected_by_parser():
    with pytest.raises(SystemExit):
        imagine.main(["gen", "--dry-run"])  # --prompt 是 required


# ---------------------------------------------------------------------------
# mark：交互编辑的前置工具（纯 PIL，离线可跑）
# ---------------------------------------------------------------------------
def test_main_mark_rect_and_arrow(tmp_env, capsys):
    src = tmp_env / "base.png"
    Image.new("RGB", (200, 160), (250, 250, 250)).save(src)
    out = tmp_env / "marked.png"
    rc = imagine.main(
        [
            "mark",
            "--src",
            str(src),
            "--out",
            str(out),
            "--rect",
            "20,20,80,60",
            "--arrow",
            "180,10,120,140",
        ]
    )
    assert rc == 0
    assert out.is_file()
    assert out.stat().st_size > src.stat().st_size  # 确实画了东西
    txt = capsys.readouterr().out
    # 标签按 rect → arrow 顺序自动编号，与 prompt 写法对应
    assert "A: 框选区域" in txt
    assert "B: 箭头所指位置" in txt
    assert "imagine edit --image" in txt


def test_main_mark_labels_many(tmp_env, capsys):
    src = tmp_env / "b.png"
    Image.new("RGB", (300, 300), "white").save(src)
    argv = ["mark", "--src", str(src), "--out", str(tmp_env / "m.png")]
    for i in range(3):
        argv += ["--rect", f"{i * 10},{i * 10},20,20"]
    assert imagine.main(argv) == 0
    txt = capsys.readouterr().out
    assert "A:" in txt and "B:" in txt and "C:" in txt


def test_main_mark_bad_rect_returns_1(tmp_env, capsys):
    src = tmp_env / "b.png"
    Image.new("RGB", (50, 50), "white").save(src)
    rc = imagine.main(["mark", "--src", str(src), "--rect", "1,2,3"])
    assert rc == 1
    assert "--rect 格式应为 x,y,w,h" in capsys.readouterr().err


def test_main_mark_missing_src_returns_1(capsys):
    rc = imagine.main(["mark", "--src", "nope.png", "--rect", "1,2,3,4"])
    assert rc == 1
    assert "mark 失败" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 本地路径：ingest / gallery / palette / adjust / sheet
# ---------------------------------------------------------------------------
def test_main_ingest_records_ledger(tmp_env, capsys):
    src = tmp_env / "from_imagegen.png"
    src.write_bytes(_png_bytes())
    rc = imagine.main(
        [
            "ingest",
            "--src",
            str(src),
            "--slug",
            "hero",
            "--prompt",
            "a hero image",
            "--backend",
            "imagegen",
            "--recipe",
            "cover_v1",
        ]
    )
    assert rc == 0
    dest = tmp_env / "assets" / "hero.png"
    assert dest.is_file()
    entries = _ledger.query()
    assert len(entries) == 1
    assert entries[0].backend == "imagegen"
    assert entries[0].recipe == "cover_v1"
    # 入库图自带 provenance（与云端出图路径一致）——账本丢了也能溯源
    meta = _ledger.read_metadata(dest)
    assert meta["prompt"] == "a hero image"
    assert meta["backend"] == "imagegen"
    assert (tmp_env / "LEDGER.md").is_file()
    assert "已入库" in capsys.readouterr().out


def test_main_ingest_missing_returns_1(capsys):
    rc = imagine.main(["ingest", "--src", "definitely_missing.png"])
    assert rc == 1
    assert "ingest 失败" in capsys.readouterr().err


def test_main_gallery_add_and_list(tmp_env, capsys):
    src = tmp_env / "pick.png"
    src.write_bytes(_png_bytes())
    assert imagine.main(["gallery", "--add", str(src), "--slug", "ref1"]) == 0
    assert (tmp_env / "gallery" / "ref1.png").is_file()
    capsys.readouterr()
    assert imagine.main(["gallery"]) == 0
    assert "ref1.png" in capsys.readouterr().out


def test_main_palette_on_real_image(tmp_env, capsys):
    img = tmp_env / "ref.png"
    Image.new("RGB", (30, 30), (12, 130, 200)).save(img)
    assert imagine.main(["palette", "--src", str(img), "--n", "3"]) == 0
    out = capsys.readouterr().out
    assert "主色" in out and "#" in out
    assert "imagine bridge" in out


def test_main_palette_missing_returns_1(capsys):
    rc = imagine.main(["palette", "--src", "definitely_missing.png"])
    assert rc == 1
    assert "palette 失败" in capsys.readouterr().err


def test_main_adjust_resize(tmp_env, capsys):
    src = tmp_env / "a.png"
    Image.new("RGB", (80, 60), "navy").save(src)
    out = tmp_env / "small.png"
    assert (
        imagine.main(["adjust", str(src), "--out", str(out), "--resize", "40", "0"])
        == 0
    )
    with Image.open(out) as im:
        assert im.size == (40, 30)  # 0 → 按长宽比自动
    assert "视觉校验" in capsys.readouterr().out


def test_main_adjust_missing_returns_1(capsys):
    assert imagine.main(["adjust", "nope.png", "--out", "o.png"]) == 1
    assert "adjust 失败" in capsys.readouterr().err


def test_main_sheet(tmp_env, capsys):
    paths = []
    for i in range(3):
        p = tmp_env / f"s{i}.png"
        Image.new("RGB", (40, 40), (i * 60, 90, 120)).save(p)
        paths.append(str(p))
    out = tmp_env / "sheet.png"
    assert imagine.main(["sheet", *paths, "--out", str(out), "--cols", "3"]) == 0
    assert out.is_file()
    assert "3 张" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 云端出图的完整接线（用假响应顶替网络）
# ---------------------------------------------------------------------------
def test_gen_end_to_end_offline(tmp_env, monkeypatch, capsys):
    """出图 → 落盘（后缀按真实字节判定）→ 内嵌元数据 → 记账 → 渲染账本 → 存快照。"""
    monkeypatch.setattr(imagine._ark, "generate", lambda *a, **kw: _fake_response())
    rc = imagine.main(
        [
            "gen",
            "--prompt",
            "a red square",
            "--slug",
            "unit",
            "--recipe",
            "cover_v1",
            "--notes",
            "unit test",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "已出图 1 张" in out
    assert "gallery --add" in out  # 结尾必提示归档（方舟不保证复现）

    files = sorted((tmp_env / "assets").glob("unit*"))
    assert [f.suffix for f in files] == [".png"]  # 不是按 output_format 猜的 .jpg

    entries = _ledger.query()
    assert len(entries) == 1
    e = entries[0]
    assert e.backend == "ark" and e.prompt == "a red square"
    assert e.recipe == "cover_v1" and e.notes == "unit test"
    assert e.extra["kind"] == "gen"
    assert e.extra["generated_images"] == 1

    assert _ledger.read_metadata(files[0])["prompt"] == "a red square"
    assert (tmp_env / "runs" / "unit.json").is_file()
    assert (tmp_env / "LEDGER.md").is_file()


def test_gen_passes_watermark_false_and_single_image(tmp_env, monkeypatch):
    seen: dict = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return _fake_response()

    monkeypatch.setattr(imagine._ark, "generate", fake)
    imagine.main(["gen", "--prompt", "p"])
    assert seen["watermark"] is False  # 方舟默认 true，必须显式关
    assert seen["max_images"] == 1
    assert "sequential" not in seen


def test_gen_group_passes_sequential(tmp_env, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        imagine._ark,
        "generate",
        lambda prompt, **kw: (seen.update(kw), _fake_response(2))[1],
    )
    imagine.main(
        [
            "gen",
            "--prompt",
            "p",
            "--model",
            GROUP_CAPABLE_MODEL,
            "--group",
            "--max-images",
            "3",
        ]
    )
    assert seen["sequential"] is True
    assert seen["max_images"] == 3


def test_i2i_passes_reference_paths(tmp_env, monkeypatch):
    src = tmp_env / "ref.png"
    src.write_bytes(_png_bytes())
    seen: dict = {}
    monkeypatch.setattr(
        imagine._ark,
        "generate",
        lambda prompt, **kw: (seen.update(kw), _fake_response())[1],
    )
    assert imagine.main(["i2i", "--image", str(src), "--prompt", "p"]) == 0
    assert seen["images"] == [str(src)]


def test_gen_records_every_image_of_a_group(tmp_env, monkeypatch):
    """组图 3 张 → 3 条账本记录（逐张可溯源），但只渲染一次 Markdown。"""
    monkeypatch.setattr(imagine._ark, "generate", lambda *a, **kw: _fake_response(3))
    assert (
        imagine.main(
            [
                "gen",
                "--prompt",
                "p",
                "--model",
                GROUP_CAPABLE_MODEL,
                "--group",
                "--max-images",
                "3",
                "--slug",
                "grp",
            ]
        )
        == 0
    )
    assert len(_ledger.query()) == 3
    assert len(sorted((tmp_env / "assets").glob("grp*"))) == 3


def test_gen_n_loops_independent_calls(tmp_env, monkeypatch):
    """--n>1 在非组图路径下要真的发多次请求（方舟单请求只出一张）。"""
    hits: list[dict] = []

    def fake(prompt, **kw):
        hits.append(kw)
        return _fake_response()

    monkeypatch.setattr(imagine._ark, "generate", fake)
    assert imagine.main(["gen", "--prompt", "p", "--n", "3", "--slug", "pick"]) == 0
    assert len(hits) == 3
    assert all(kw["max_images"] == 1 for kw in hits)
    assert len(_ledger.query()) == 3
    names = sorted(f.name for f in (tmp_env / "assets").glob("pick*"))
    assert names == ["pick_0.png", "pick_1.png", "pick_2.png"]


def test_gen_n_clamped_by_cost_guard(tmp_env, monkeypatch, capsys):
    """--n 超过成本护栏 max_images 时被夹取（不默默烧额度）。"""
    # tmp_env 已把数据区重定向完，这里只需改护栏本身
    monkeypatch.setattr(
        imagine, "settings", dataclasses.replace(imagine.settings, max_images=2)
    )
    hits: list[dict] = []
    monkeypatch.setattr(
        imagine._ark,
        "generate",
        lambda prompt, **kw: (hits.append(kw), _fake_response())[1],
    )
    assert imagine.main(["gen", "--prompt", "p", "--n", "99", "--dry-run"]) == 0
    assert "夹取" in capsys.readouterr().out
    assert imagine.main(["gen", "--prompt", "p", "--n", "99"]) == 0
    assert len(hits) == 2


def test_gen_later_call_failure_keeps_earlier_output(tmp_env, monkeypatch, capsys):
    """多次请求中后续失败 → 保住已得产出并返回 0，不把整批判死。"""
    state = {"i": 0}

    def fake(prompt, **kw):
        state["i"] += 1
        if state["i"] > 1:
            raise _ark.ArkError("ServiceUnavailable", status_code=503)
        return _fake_response()

    monkeypatch.setattr(imagine._ark, "generate", fake)
    assert imagine.main(["gen", "--prompt", "p", "--n", "3", "--slug", "half"]) == 0
    err = capsys.readouterr().err
    assert "第 2/3 次失败" in err
    assert len(_ledger.query()) == 1


def test_gen_all_failed_returns_1(tmp_env, monkeypatch, capsys):
    """整批产出都失败（单张级 error）→ 返回 1，且不写账本。"""
    resp = _fake_response(1)
    resp.images = [
        _ark.ArkImage(index=0, error={"code": "InternalError", "message": "boom"})
    ]
    resp.usage = {"generated_images": 0}
    monkeypatch.setattr(imagine._ark, "generate", lambda *a, **kw: resp)
    assert imagine.main(["gen", "--prompt", "p"]) == 1
    assert "无任何成功产出" in capsys.readouterr().err
    assert _ledger.query() == []


def test_gen_ark_error_returns_1(tmp_env, monkeypatch, capsys):
    """云端抛 ArkError → 打印 "{kind} 失败：<msg>" 并返回 1。

    真实的 ``generate()`` 会把 ``hint()`` 折进 ``args``（那一行为由
    ``test_ark_client.py`` 覆盖）；这里照它的**产物形态**造错，验证 CLI 原样透出
    而不吞掉排障提示。
    """

    def boom(*a, **kw):
        raise _ark.ArkError(
            "InvalidApiKey | 提示：检查 .env 里的 ARK_API_KEY",
            status_code=401,
            code="InvalidApiKey",
        )

    monkeypatch.setattr(imagine._ark, "generate", boom)
    assert imagine.main(["gen", "--prompt", "p"]) == 1
    err = capsys.readouterr().err
    assert "gen 失败" in err
    assert "ARK_API_KEY" in err  # 提示被原样透出


def test_gen_partial_failure_still_saves_ok_ones(tmp_env, monkeypatch, capsys):
    resp = _fake_response(2)
    resp.images[1] = _ark.ArkImage(index=1, error={"code": "X", "message": "bad"})
    monkeypatch.setattr(imagine._ark, "generate", lambda *a, **kw: resp)
    assert imagine.main(["gen", "--prompt", "p", "--slug", "part"]) == 0
    assert "第 1 张失败" in capsys.readouterr().err
    assert len(_ledger.query()) == 1
