"""`adjust` 的破坏性写护栏：**误用不销毁、可完整找回**（backlog 20261015-210400）。

实案（tb-3 P.p2 交付披露）：一次 `adjust` 用法错误 + 调用侧管道掩盖退出码 → 两张**已计费
生成图**被永久销毁；`runs/*.json` 快照已剥离 b64，**无从恢复**。

修法四道护栏（见 `postprocess.adjust_image` docstring）：①源先完整解码（动 dst 之前就失败）；
②就地覆盖默认**拒绝**（须 `--confirm`）；③任何会销毁既有文件的写，先把该文件**复制进回收区**
（`trash/`，带时间戳）——未给回收区就拒绝破坏性写；④原子落盘（`.tmp` + `os.replace`）。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import imagine
from pysci.skills.ai_drawing.tools import postprocess as pp

_DIRS = ("assets", "gallery", "recipes", "prompts", "cache", "runs", "trash")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """技能数据区（含 trash）整体落 tmp，并放开 data/ 落点护栏——绝不碰真实回收区。"""
    dirs = {f"{n}_dir": tmp_path / n for n in _DIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        imagine,
        "settings",
        dataclasses.replace(imagine.settings, module_dir=tmp_path, **dirs),
    )
    monkeypatch.setattr(imagine, "assert_within_data", lambda p, **kw: Path(p))
    return tmp_path


def _png(path: Path, color=(20, 90, 160), size=(80, 60)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# ① 就地覆盖：默认拒绝，且**字节不动**
# ---------------------------------------------------------------------------
def test_in_place_without_confirm_refuses_and_keeps_bytes(env, capsys):
    src = _png(env / "gen.png")
    before = src.read_bytes()

    rc = imagine.main(["adjust", str(src), "--out", str(src), "--resize", "40", "0"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "就地覆盖" in err and "--confirm" in err
    assert src.read_bytes() == before, "误用（就地覆盖未确认）竟改动了原图"


def test_in_place_raw_api_also_refuses(env):
    src = _png(env / "g.png")
    before = src.read_bytes()
    with pytest.raises(ValueError, match="就地覆盖"):
        pp.adjust_image(src, src, resize=(40, 0), trash_dir=env / "trash")
    assert src.read_bytes() == before


def test_in_place_with_confirm_succeeds_and_original_is_recoverable(env, capsys):
    src = _png(env / "g.png")
    before = src.read_bytes()

    rc = imagine.main(
        [
            "adjust",
            str(src),
            "--out",
            str(src),
            "--confirm",
            "--resize",
            "40",
            "0",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert src.read_bytes() != before  # 确实改写了
    trashed = list((env / "trash").glob("g.*.png"))
    assert trashed, "确认就地覆盖后未留回收副本"
    assert trashed[0].read_bytes() == before, "回收副本与原件不一致——找回会失真"
    assert "回收区" in out  # 回报可见


# ---------------------------------------------------------------------------
# ③ 覆盖既有文件（非就地）：先把被覆盖者救进回收区
# ---------------------------------------------------------------------------
def test_overwriting_existing_dst_backs_it_up(env):
    src = _png(env / "a.png", color=(10, 10, 10))
    dst = _png(env / "b.png", color=(250, 250, 250))
    sentinel = dst.read_bytes()

    assert imagine.main(["adjust", str(src), "--out", str(dst)]) == 0

    assert dst.read_bytes() != sentinel
    trashed = list((env / "trash").glob("b.*.png"))
    assert trashed and trashed[0].read_bytes() == sentinel, "被覆盖的原件没能完整找回"


def test_no_backup_when_dst_is_new(env):
    src = _png(env / "a.png")
    dst = env / "new.png"
    # 显式把调整关掉也合法（纯格式转换路径）
    assert imagine.main(["adjust", str(src), "--out", str(dst)]) == 0
    assert dst.is_file()
    assert list((env / "trash").iterdir()) == [], "非破坏性写不该产生回收副本"


# ---------------------------------------------------------------------------
# ④ 错误时**不动文件**
# ---------------------------------------------------------------------------
def test_bad_args_leave_dst_untouched(env, capsys):
    src = _png(env / "a.png")
    dst = _png(env / "keep.png", color=(7, 7, 7))
    sentinel = dst.read_bytes()

    rc = imagine.main(["adjust", str(src), "--out", str(dst), "--resize", "0", "0"])

    assert rc == 1
    assert "adjust 失败" in capsys.readouterr().err
    assert dst.read_bytes() == sentinel, "参数非法却改动了目标文件"


def test_missing_src_creates_nothing(env):
    dst = env / "never.png"
    assert imagine.main(["adjust", str(env / "nope.png"), "--out", str(dst)]) == 1
    assert not dst.exists()


def test_no_backup_requirement_without_trash_dir_refuses(env):
    """护栏 ③ 的硬边：会销毁既有文件却没给回收区 → **拒绝**（宁可不做）。"""
    src = _png(env / "a.png")
    dst = _png(env / "b.png")
    before = dst.read_bytes()
    with pytest.raises(ValueError, match="回收区"):
        pp.adjust_image(src, dst)  # trash_dir=None
    assert dst.read_bytes() == before


# ---------------------------------------------------------------------------
# ② 文档面：-h 与技能文档各有一句明示
# ---------------------------------------------------------------------------
def test_help_documents_confirm_and_trash(capsys):
    with pytest.raises(SystemExit):
        imagine.main(["adjust", "-h"])
    out = capsys.readouterr().out
    assert "--confirm" in out
    assert "回收区" in out and "trash" in out


def test_skill_doc_documents_the_guard():
    """技能文档面有一句明示（criteria ②）——断言只打真本（部署副本在 worktree 里不存在）。"""
    from pysci.paths import ORCHESTRATION_ROOT

    doc = ORCHESTRATION_ROOT / "skills" / "ai-drawing" / "references" / "cli.md"
    md = doc.read_text(encoding="utf-8")
    assert "trash" in md and "--confirm" in md
