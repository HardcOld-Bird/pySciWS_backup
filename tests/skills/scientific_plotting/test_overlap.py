"""零遮挡判据 `figures overlap` 的回归（backlog 20261016-014160）。

钉四件事：①蒙版/距离的像素语义（合成图）；②**缓存自防护**——旧缓存+新图必须自动重生成或
rc=2 明确报错（绝不出假读数）；③判据**单实现**（两 pod 的 bench 草稿件若带同口径，须薄调用
CLI）；④CLI 端到端。

合成图口径：白底 + 一行黑线（row 30）+ 一块彩色数据区。文字用**块**表示；「压在彩色区上」
用**混合色**像素建模（文字笔画的抗锯齿边缘 = 文字色与数据区色的混合 —— 这才是真实被判据
抓到的像素：既与删文字版有差（墨迹），又仍是彩色（数据区））。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.scientific_plotting.tools import figures, occlusion

H = W = 64
LINE_ROW = 30
COLORED = (5, 25, 5, 15)  # x0,x1,y0,y1 彩色数据区块
BLEND = (100, 30, 30)  # 文字边缘压在色块上的混合色：chroma=70>40 且 |Δ| 于色块 =100>30


def _base() -> np.ndarray:
    """白底 + 黑线（row 30, col 10..50）+ 彩色数据区。"""
    a = np.full((H, W, 3), 255, dtype=np.uint8)
    a[LINE_ROW, 10:50] = (0, 0, 0)
    x0, x1, y0, y1 = COLORED
    a[y0:y1, x0:x1] = (200, 60, 60)
    return a


def _paint(
    a: np.ndarray, rows: tuple[int, int], cols: tuple[int, int], color
) -> np.ndarray:
    b = a.copy()
    b[rows[0] : rows[1], cols[0] : cols[1]] = color
    return b


def _write(tmp: Path, audited: np.ndarray, *, name: str = "fig") -> tuple[Path, Path]:
    pa, pw = tmp / f"{name}.png", tmp / f"{name}_white.png"
    Image.fromarray(audited).save(pa)
    Image.fromarray(_base()).save(pw)  # 删文字版 = 无文字的同一版式
    return pa, pw


def _mk_pair(tmp: Path, audited: np.ndarray) -> tuple[Path, Path]:
    return _write(tmp, audited)


def _age(path: Path, seconds: float) -> None:
    """把文件 mtime 设为「当前 − seconds」（模拟旧缓存）。"""
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime - seconds))


# ---------------------------------------------------------------------------
# ① 蒙版与距离
# ---------------------------------------------------------------------------
def test_ink_mask_equals_text_glyphs():
    aud = _paint(_base(), (40, 44), (28, 36), (0, 0, 0))  # 白底上的黑字
    ink = occlusion.ink_mask(aud, _base())
    assert int(ink.sum()) == 4 * 8  # 文字块像素数


def test_ink_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        occlusion.ink_mask(np.zeros((4, 4, 3), int), np.zeros((5, 4, 3), int))


def test_colored_and_line_masks():
    base = _base()
    assert int(occlusion.colored_mask(base).sum()) == 20 * 10  # 色块
    line = occlusion.line_mask(base)
    assert int(line.sum()) == 40 and bool(line[LINE_ROW, 20])  # 仅黑线，白底/色块不在内


def test_overlap_detects_glyph_over_colored_area():
    """文字笔画的混合边缘落在彩色数据区 → 墨迹∩彩色 > 0（FAIL）。"""
    aud = _paint(_base(), (8, 12), (8, 12), BLEND)
    rep = occlusion.render_ink_verdict(aud, _base())
    assert rep["ink_colored_px"] == 4 * 4
    assert rep["pass"] is False


def test_min_distance_reading_for_text_near_line():
    """文字紧贴图线（未压）→ 覆盖 0、最短距=1px（「间距读数」）。"""
    aud = _paint(_base(), (29, 30), (20, 24), (0, 0, 0))  # 恰在线上方一行
    ink, line = occlusion.ink_mask(aud, _base()), occlusion.line_mask(_base())
    assert occlusion.overlap_count(ink, line) == 0
    assert occlusion.min_distance_px(ink, line) == pytest.approx(1.0)


def test_clean_figure_passes():
    aud = _paint(_base(), (40, 44), (28, 36), (0, 0, 0))  # 文字离线与色块
    assert occlusion.render_ink_verdict(aud, _base())["pass"] is True


# ---------------------------------------------------------------------------
# ② 缓存自防护（本单核心：旧缓存 + 新图 → 重生成或明确报错，不出假读数）
# ---------------------------------------------------------------------------
def test_fresh_cache_ok(tmp_path):
    img, white = _mk_pair(tmp_path, _paint(_base(), (40, 44), (28, 36), (0, 0, 0)))
    c = occlusion.check_caches(img, [white])
    assert c.ok and c.stale == () and c.missing == ()


def test_missing_cache_errors(tmp_path):
    img, white = _mk_pair(tmp_path, _base())
    white.unlink()
    c = occlusion.check_caches(img, [white])
    assert not c.ok and "缺失" in c.message


def test_stale_cache_without_rebuild_errors(tmp_path):
    """旧缓存 + 新图（无 --rebuild）→ 明确报错，绝不出读数。"""
    img, white = _mk_pair(tmp_path, _base())
    _age(white, 3600)
    c = occlusion.check_caches(img, [white])
    assert not c.ok and "过期" in c.message and "rebuild" in c.message


def test_stale_cache_with_rebuild_regenerates(tmp_path):
    """旧缓存 + 新图 + --rebuild → 先重生成再复检（自动重生成路径）。"""
    img, white = _mk_pair(tmp_path, _base())
    _age(white, 3600)
    calls: list[str] = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)
        white.write_bytes(white.read_bytes())  # 重生成 → mtime 刷新
        return None

    c = occlusion.check_caches(
        img, [white], rebuild="regenerate-please", runner=fake_runner
    )
    assert calls == ["regenerate-please"]
    assert c.ok and c.message == "缓存新鲜"


# ---------------------------------------------------------------------------
# ③ 判据单实现：bench 草稿件若带同口径，必须是薄调用 CLI
# ---------------------------------------------------------------------------
def test_chroma_judgment_single_implementation():
    """两 pod 的 bench 草稿件里不得再出现同口径实现（|Δ|>30 ∩ chroma>40）。"""
    bench = sorted(
        p for p in (ORCHESTRATION_ROOT / "pods").glob("*/bench/*.py") if p.is_file()
    )
    if not bench:
        pytest.skip("本机无 bench 草稿件（gitignored 运行期面，干净检出即空）")
    offenders = []
    for p in bench:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        reimplements = "chroma" in txt and "> 40" in txt and "> 30" in txt
        delegates = "figures overlap" in txt or "occlusion" in txt
        if reimplements and not delegates:
            offenders.append(str(p.relative_to(ORCHESTRATION_ROOT)))
    assert offenders == [], f"bench 侧仍有同口径实现（应薄调用 CLI）：{offenders}"


# ---------------------------------------------------------------------------
# ④ CLI 端到端
# ---------------------------------------------------------------------------
def test_cli_overlap_pass(tmp_path, capsys):
    img, white = _mk_pair(tmp_path, _paint(_base(), (40, 44), (28, 36), (0, 0, 0)))
    rc = figures.main(["overlap", str(img), "--white", str(white)])
    out = capsys.readouterr().out
    assert rc == 0 and "总判：PASS" in out and "缓存新鲜" in out


def test_cli_overlap_fail_on_ink_over_colored(tmp_path, capsys):
    img, white = _mk_pair(tmp_path, _paint(_base(), (8, 12), (8, 12), BLEND))
    rc = figures.main(["overlap", str(img), "--white", str(white)])
    out = capsys.readouterr().out
    assert rc == 1 and "总判：FAIL" in out and "墨迹∩彩色数据区" in out


def test_cli_overlap_stale_cache_rc2(tmp_path, capsys):
    img, white = _mk_pair(tmp_path, _base())
    _age(white, 3600)
    rc = figures.main(["overlap", str(img), "--white", str(white)])
    out = capsys.readouterr().out
    assert rc == 2 and "过期" in out and "总判" not in out  # 不出读数


def test_cli_overlap_rebuild_command_runs(tmp_path, capsys):
    """真子进程路径：--rebuild 指向一个把缓存刷新到最新 mtime 的脚本。"""
    img, white = _mk_pair(tmp_path, _paint(_base(), (40, 44), (28, 36), (0, 0, 0)))
    _age(white, 3600)
    script = tmp_path / "regen.py"
    script.write_text(
        "from pathlib import Path\n"
        f"p = Path(r'{white}')\n"
        "p.write_bytes(p.read_bytes())  # 重写同内容 → mtime 刷新为「新」\n",
        encoding="utf-8",
    )
    rc = figures.main(
        [
            "overlap",
            str(img),
            "--white",
            str(white),
            "--rebuild",
            f'"{sys.executable}" "{script}"',
        ]
    )
    out = capsys.readouterr().out
    assert "缓存新鲜" in out and rc == 0


def test_cli_overlap_box_scopes_reading(tmp_path, capsys):
    """--box 把判读限制在裁切窗内（多面板图逐面板量）。"""
    img, white = _mk_pair(tmp_path, _paint(_base(), (8, 12), (8, 12), BLEND))
    # 窗口避开色块（x 30..60, y 30..60）→ 无交叠 → PASS
    rc = figures.main(
        ["overlap", str(img), "--white", str(white), "--box", "30", "60", "30", "60"]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "总判：PASS" in out
