"""``figures overlap`` 的进度可观察性与降采样口径（backlog 20261017-101924-figure）。

缺陷：4k 级成品单次判据 8–10 分钟且**无进度输出**（stdout 重定向块缓冲 → 进程像挂死，
成员起了 3 个重复后台任务）。修法：逐阶段进度（flush）＋ ``--downsample`` 粗判＋报告头写明尺度。

用**合成图对**（同一版式、白版删去文字块）保证可复跑；真图对实测见交付正文。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from pysci.skills.scientific_plotting.tools import figures as fig  # noqa: E402
from pysci.skills.scientific_plotting.tools import occlusion as occ  # noqa: E402


def _synth_arrays(w: int = 800, h: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """合成「受审图 / 删文字版」：白底 + 一条横线（两版共有）+ 一块文字（仅受审图）。"""
    base = np.full((h, w, 3), 255, dtype=np.uint8)
    base[h - 120 : h - 118, 20 : w - 20] = 0  # 图线
    white = base.copy()
    img = base.copy()
    img[20 : 20 + min(300, h - 200), 20 : 20 + min(300, w - 100)] = 0  # 文字块
    return img, white


def _write_pair(tmp_path, *, w=800, h=800, dpi=(600, 600)):
    img, white = _synth_arrays(w, h)
    a = tmp_path / "text.png"
    b = tmp_path / "notext.png"
    Image.fromarray(img).save(a, dpi=dpi)
    time.sleep(0.02)
    Image.fromarray(white).save(b, dpi=dpi)
    return a, b


# ---------------------------------------------------------------------------
# ① 降采样：同判 + 耗时 ≤1/3
# ---------------------------------------------------------------------------
def test_downsample_same_verdict_and_faster():
    img, white = _synth_arrays()
    t = time.perf_counter()
    full = occ.render_ink_verdict(img, white)
    t_full = time.perf_counter() - t

    t = time.perf_counter()
    half = occ.render_ink_verdict(
        occ.downsample_rgb(img, 2), occ.downsample_rgb(white, 2)
    )
    t_half = time.perf_counter() - t

    assert half["pass"] is full["pass"], (full, half)
    assert t_half <= t_full / 3, (
        f"降采样未达 ≤1/3：full={t_full:.1f}s half={t_half:.1f}s"
    )


def test_downsample_rgb_shapes():
    arr = np.zeros((100, 80, 3), dtype=np.uint8)
    assert occ.downsample_rgb(arr, 1).shape == (100, 80, 3)
    assert occ.downsample_rgb(arr, 2).shape == (50, 40, 3)
    assert occ.downsample_rgb(arr, 4).shape == (25, 20, 3)


# ---------------------------------------------------------------------------
# ② 逐阶段进度 + 周期心跳（flush，可被重定向日志 tail 看到）
# ---------------------------------------------------------------------------
def test_stage_progress_is_reported_in_order():
    seen: list[str] = []
    img, white = _synth_arrays(400, 400)
    occ.render_ink_verdict(img, white, progress=seen.append)
    assert seen == ["蒙版", "交集", "最近距", "汇总"]


def test_ticker_prints_with_flush(capsys):
    ticker = fig._ProgressTicker(interval=0.02)
    with ticker:
        ticker.stage = "最近距"
        time.sleep(0.08)
    out = capsys.readouterr().out
    assert "最近距" in out and "已" in out


# ---------------------------------------------------------------------------
# ③ 报告头写出尺度（px + dpi）＋ -h 耗时提示
# ---------------------------------------------------------------------------
def test_cli_header_reports_scale_and_dpi(tmp_path, capsys):
    a, b = _write_pair(tmp_path, w=160, h=120)
    rc = fig.main(["overlap", str(a), "--white", str(b), "--downsample", "2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "downsample=1/2" in out and "判据尺度" in out and "600 dpi" in out
    assert "@尺度 1/2" in out


def test_help_mentions_cost_and_downsample(capsys):
    with pytest.raises(SystemExit):
        fig.main(["overlap", "-h"])
    out = capsys.readouterr().out
    assert "--downsample" in out
    assert "4k" in out and "分钟" in out
