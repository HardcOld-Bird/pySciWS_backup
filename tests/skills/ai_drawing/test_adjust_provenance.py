"""`adjust` 的 PNG 溯源内嵌（backlog 20261016-031600-drawing）。

场景：本账号可用的 5.0-pro/flash 都**不支持 `output_format=png`**（仅已下架型号支持），
故「jpeg 出图 → `adjust` 转 png」是唯一路径；而 `gen` 的 jpeg 不带 `pysci:*`、`adjust` 原先
也不写入 → 交付图**静默缺失内嵌溯源**（页4 有、经同路径的页却无），只剩 `runs/*.json` 台账
兜底（且其 b64 已剥离，图像不可回放）。

修法：`adjust` 在 `--out *.png` 时**自动**从 manifest 记账（或同名 `runs/<slug>.json` 快照）
继承 `prompt/model/backend(/recipe)` 写入 PNG tEXt；亦可 `--from-ledger <runs/x.json>` 显式
指定，`--no-provenance` 关闭。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import imagine, ledger

_DIRS = ("assets", "gallery", "recipes", "prompts", "cache", "runs", "trash")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """技能数据区整体落 tmp；同时重定向 ``imagine.settings`` 与 ``ledger.settings``
    （二者在生产里是同一对象，``dataclasses.replace`` 会拆成两份，须都指过去）。"""
    dirs = {f"{n}_dir": tmp_path / n for n in _DIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    s = dataclasses.replace(imagine.settings, module_dir=tmp_path, **dirs)
    monkeypatch.setattr(imagine, "settings", s)
    monkeypatch.setattr(ledger, "settings", s)
    monkeypatch.setattr(imagine, "assert_within_data", lambda p, **kw: Path(p))
    return tmp_path


def _jpeg(path: Path, color=(20, 90, 160), size=(80, 60)) -> Path:
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# ① 自动：manifest 记账 → PNG 内嵌
# ---------------------------------------------------------------------------
def test_adjust_png_inherits_provenance_from_manifest(env, capsys):
    src = _jpeg(env / "gen.jpeg")
    ledger.record(
        prompt="a red nebula cover",
        model="doubao-seedream-5-0-pro-260628",
        backend="ark",
        recipe="cover_v1",
        out=src,
    )

    out = env / "out.png"
    rc = imagine.main(["adjust", str(src), "--out", str(out)])

    assert rc == 0
    meta = ledger.read_metadata(out)
    assert meta["prompt"] == "a red nebula cover"
    assert meta["model"] == "doubao-seedream-5-0-pro-260628"
    assert meta["backend"] == "ark"
    assert meta["recipe"] == "cover_v1"
    assert "已内嵌溯源" in capsys.readouterr().out


def test_adjust_png_inherits_from_same_stem_snapshot(env):
    """manifest 无记录时，回落到同名 ``runs/<slug>.json`` 快照。"""
    src = _jpeg(env / "gen.jpeg")
    (env / "runs" / "gen.json").write_text(
        '{"request": {"prompt": "from snapshot", "model": "m9"}}', encoding="utf-8"
    )

    out = env / "out.png"
    assert imagine.main(["adjust", str(src), "--out", str(out)]) == 0

    meta = ledger.read_metadata(out)
    assert meta["prompt"] == "from snapshot"
    assert meta["model"] == "m9"
    assert meta["backend"] == "ark"  # 快照只由云端出图写
    assert "snapshot" in meta


# ---------------------------------------------------------------------------
# ② 显式 --from-ledger / --no-provenance
# ---------------------------------------------------------------------------
def test_adjust_from_ledger_flag_wins(env):
    src = _jpeg(env / "gen.jpeg")
    snap = env / "runs" / "other.json"
    snap.write_text(
        '{"request": {"prompt": "explicit", "model": "mx"}}', encoding="utf-8"
    )

    out = env / "out.png"
    rc = imagine.main(
        ["adjust", str(src), "--out", str(out), "--from-ledger", str(snap)]
    )

    assert rc == 0
    assert ledger.read_metadata(out)["prompt"] == "explicit"


def test_adjust_no_provenance_flag_skips(env):
    src = _jpeg(env / "gen.jpeg")
    ledger.record(prompt="p", model="m", backend="ark", out=src)

    out = env / "out.png"
    assert imagine.main(["adjust", str(src), "--out", str(out), "--no-provenance"]) == 0
    assert ledger.read_metadata(out) == {}


def test_adjust_explicit_missing_snapshot_warns(env, capsys):
    src = _jpeg(env / "gen.jpeg")
    out = env / "out.png"
    missing = env / "runs" / "nope.json"

    rc = imagine.main(
        ["adjust", str(src), "--out", str(out), "--from-ledger", str(missing)]
    )

    assert rc == 0  # 溯源是兜底，取不到不该让转换失败
    assert ledger.read_metadata(out) == {}
    assert "未取到 provenance" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ③ 边界：非 PNG 不写；无来源时静默；像素不被写坏
# ---------------------------------------------------------------------------
def test_adjust_non_png_out_has_no_provenance(env):
    src = _jpeg(env / "gen.jpeg")
    ledger.record(prompt="p", model="m", backend="ark", out=src)

    out = env / "out.jpeg"
    assert imagine.main(["adjust", str(src), "--out", str(out)]) == 0
    assert ledger.read_metadata(out) == {}  # JPEG 不写（重存会二次压缩）


def test_adjust_without_source_is_silent(env, capsys):
    src = _jpeg(env / "plain.jpeg")  # 既无记账也无同名快照
    out = env / "out.png"

    rc = imagine.main(["adjust", str(src), "--out", str(out)])

    assert rc == 0
    assert ledger.read_metadata(out) == {}
    captured = capsys.readouterr()
    assert "未取到 provenance" not in captured.err  # 未显式要求 → 不噪声
    assert "已内嵌溯源" not in captured.out


def test_adjust_provenance_write_keeps_pixels(env):
    """内嵌 tEXt 不得写坏像素（用**无损** PNG 源，避开 JPEG 有损带来的假阴性）。"""
    src = env / "gen.png"
    Image.new("RGB", (80, 60), (11, 22, 33)).save(src)
    ledger.record(prompt="p", model="m", backend="ark", out=src)

    out = env / "out.png"
    assert imagine.main(["adjust", str(src), "--out", str(out)]) == 0

    with Image.open(out) as im:
        assert im.mode == "RGB" and im.size == (80, 60)
        assert im.getpixel((0, 0)) == (11, 22, 33)


# ---------------------------------------------------------------------------
# ④ 文档面：-h 明示两个开关
# ---------------------------------------------------------------------------
def test_help_documents_provenance_flags(capsys):
    with pytest.raises(SystemExit):
        imagine.main(["adjust", "-h"])
    out = capsys.readouterr().out
    assert "--from-ledger" in out
    assert "--no-provenance" in out


def test_skill_doc_documents_adjust_provenance():
    """技能文档面明示 adjust 的 PNG 溯源继承（只打真本；部署副本在 worktree 里不存在）。"""
    from pysci.paths import ORCHESTRATION_ROOT

    md = (
        ORCHESTRATION_ROOT / "skills" / "ai-drawing" / "references" / "cli.md"
    ).read_text(encoding="utf-8")
    assert "--from-ledger" in md
    assert "--no-provenance" in md
    assert "auto-inherits" in md
