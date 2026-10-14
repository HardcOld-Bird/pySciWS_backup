"""防写穿哨兵双向（PSEP_3 收尾单）：写真实状态面 → **失败且报用例名**；豁免 → 放行。

哨兵本体在 ``conftest.py`` 的 autouse 夹具 ``_state_write_guard``（对真实 state/data 根的
写操作即刻 raise）。本文件证明它**双向**成立 + 不误杀 tmp 写入。
"""

from __future__ import annotations

import pathlib

import pytest

from pysci import paths as _paths

#: 真实根（导入期读取；哨兵只在**写**路径上判，不改这些常量）。
_REAL_STATE = _paths.ORCH_STATE_ROOT
_REAL_DATA = _paths.DATA_ROOT


def test_sentinel_blocks_real_state_write_and_names_case():
    """写真实 state → AssertionError，消息含**本用例名**与具体路径（漏隔离用例名可读）。"""
    probe = _REAL_STATE / "_sentinel_probe.txt"
    with pytest.raises(AssertionError) as exc:
        probe.write_text("x", encoding="utf-8")
    msg = str(exc.value)
    assert "防写穿哨兵" in msg
    assert "test_sentinel_blocks_real_state_write_and_names_case" in msg
    assert "_sentinel_probe.txt" in msg


def test_sentinel_blocks_real_data_write():
    """真实 ``data/`` 面同样拦（零例外面）。"""
    with pytest.raises(AssertionError, match="防写穿哨兵"):
        (_REAL_DATA / "_sentinel_probe.txt").write_text("x", encoding="utf-8")


def test_sentinel_blocks_mkdir_and_open_append():
    """不止 write_text：mkdir / open(w|a) 等写入口同样拦。"""
    with pytest.raises(AssertionError, match="防写穿哨兵"):
        (_REAL_STATE / "_sentinel_probe_dir").mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssertionError, match="防写穿哨兵"):
        with (_REAL_STATE / "_sentinel_probe_append.txt").open("a", encoding="utf-8"):
            pass
    with pytest.raises(AssertionError, match="防写穿哨兵"):
        (_REAL_STATE / "_sentinel_probe_bytes.bin").write_bytes(b"x")


def test_sentinel_allows_tmp_writes(tmp_path):
    """哨兵**不误杀**：tmp 面照常写（含 mkdir/open）。"""
    (tmp_path / "ok.txt").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    with (sub / "a.txt").open("a", encoding="utf-8") as fh:
        fh.write("y")
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "x"
    assert pathlib.Path(sub / "a.txt").read_text(encoding="utf-8") == "y"


@pytest.mark.allow_real_state
def test_sentinel_exempt_allows_real_write():
    """豁免标记 → 放行真实写入；用例**就地清理**（不留痕污染，豁免面另由收尾汇总留痕）。"""
    probe = _REAL_STATE / "_sentinel_exempt_probe.txt"
    try:
        probe.write_text("x", encoding="utf-8")
        assert probe.is_file()
    finally:
        probe.unlink(missing_ok=True)
