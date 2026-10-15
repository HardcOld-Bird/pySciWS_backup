"""audit_figure_dir 的适用边界：非 scaffold 图目录 → N/A（而非泛化 build FAIL）。

背景（G4）：beam_scatter 一类**散落脚本**产出的图目录没有 scaffold 管线（无 ``<slug>.py``
且 src 侧无同名脚本），``audit`` 无法重建 Figure。旧行为把``discover_pipeline`` 的
``FileNotFoundError`` 一律吞成 ``ERROR build``，把一个**工具不适用**渲染成**产物失败**——
审查端据此判 FAIL 是误伤。新行为：先探边界，无任何管线模块时返回 ``applicable=False`` 的
N/A 副值（``ok`` 仍为 True，rc=0），提示退回视觉判读 + 数据追溯。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.scientific_plotting.tools import audit

CODES_NON_SCAFFOLD = "not-scaffold"
CODES_BUILD = "build"


def _codes(rep: audit.AuditReport) -> list[str]:
    return [i.code for i in rep.issues]


def test_non_scaffold_dir_is_na_not_fail(tmp_path: Path):
    """无任何 .py 的图目录（仅散落产物）→ N/A，且**不**报 build ERROR。"""
    fd = tmp_path / "figures" / "beam_scatter_like"
    fd.mkdir(parents=True)
    (fd / "combined.png").write_bytes(b"\x89PNG\r\n")  # 有产物但无管线

    rep = audit.audit_figure_dir(fd)

    assert rep.applicable is False
    assert CODES_NON_SCAFFOLD in _codes(rep)
    assert CODES_BUILD not in _codes(rep)
    assert rep.ok is True  # N/A 不是失败
    assert "N/A" in rep.report()
    assert "FAIL" not in rep.report()


def test_ambiguous_multiple_py_is_still_build_error(tmp_path: Path):
    """目录内有多个 .py（入口歧义）→ 仍是 build ERROR（区别于「工具不适用」）。"""
    fd = tmp_path / "figures" / "ambiguous"
    fd.mkdir(parents=True)
    (fd / "a.py").write_text("# a\n", encoding="utf-8")
    (fd / "b.py").write_text("# b\n", encoding="utf-8")

    rep = audit.audit_figure_dir(fd)

    assert rep.applicable is True
    assert CODES_BUILD in _codes(rep)
    assert rep.ok is False


def test_missing_dir_is_build_error(tmp_path: Path):
    """图目录不存在 → build ERROR（非 N/A）。"""
    rep = audit.audit_figure_dir(tmp_path / "figures" / "nope")

    assert rep.applicable is True
    assert CODES_BUILD in _codes(rep)


def test_in_memory_report_defaults_applicable():
    """默认构造的报表 applicable=True（内存内 audit 恒适用）。"""
    rep = audit.AuditReport()
    assert rep.applicable is True
    assert "verdict: PASS" in rep.report()
