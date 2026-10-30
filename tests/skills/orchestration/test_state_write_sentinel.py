"""防写穿哨兵双向（PSEP_3 收尾单）：写真实状态面 → **失败且报用例名**；豁免 → 放行。

哨兵本体在 ``conftest.py`` 的 autouse 夹具 ``_state_write_guard``（对真实 state/data 根的
写操作即刻 raise）。本文件证明它**双向**成立 + 不误杀 tmp 写入。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

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


# ---------------------------------------------------------------------------
# 留痕不占据 tail 末尾（backlog 20261015-132939-fix3；实案 A.a5 误杀的前提核验）
# ---------------------------------------------------------------------------
def test_exemption_note_is_not_the_last_stdout_line():
    """**实测**：留痕块不在 stdout 末尾——pytest 的计数行（``N passed in Xs``）在收尾 hook
    **之后**才写，故 ``checks.run_checks`` 取的 ``output_tail`` 末行永远是判据摘要。

    这条把「留痕不占据 tail」钉成事实（挪动 hook 到更晚阶段、或让留痕成为最后输出，立即红）；
    留痕本身仍走 stdout（改 stderr 会被 pytest 捕获吞掉 → 成功路径上审计线消失）。
    """
    target = pathlib.Path(__file__).resolve()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{target}::test_sentinel_exempt_allows_real_write",
            "-q",
        ],
        cwd=str(_paths.PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stdout[-400:]
    assert "豁免留痕" in proc.stdout, "该用例须真触发豁免留痕（否则本判据空转）"
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    assert "passed" in last, f"末行应为判据摘要，实为：{last!r}"
