"""scienceplots 审美层集成（返工1 M5）：契约层优先、可选依赖 fail-soft、CLI 可见。

背景：批语 5 要求「积极使用 scienceplots 预设（Nature/Science/APS 系审美优先）」。本文件
把「契约层覆盖 scienceplots 冲突键」这一优先级铁律钉成机械判据，并核对开关默认关、非法
风格名可见失败、CLI `styles` 列出可用风格。
"""

from __future__ import annotations

import pytest

from pysci.skills.scientific_plotting.tools import style

pytest.importorskip("scienceplots")


def test_scienceplots_available_lists_styles():
    names = style.scienceplots_available()
    assert "science" in names and "no-latex" in names


def test_scienceplots_rc_merges_requested_styles():
    rc = style.scienceplots_rc(("science", "no-latex"))
    assert rc, "scienceplots 已安装却取不到 rcParams"
    assert rc.get("text.usetex") is False, (
        "no-latex 收尾应关闭 usetex（未装 LaTeX 也能渲染）"
    )


def test_unknown_style_raises_visible_failure():
    """静默失效升级为可见失败：请求未注册风格 → KeyError。"""
    with pytest.raises(KeyError):
        style.scienceplots_rc(("definitely-not-a-style",))


def test_contract_layer_wins_over_scienceplots():
    """铁律：契约层（设计宽度/字体/可编辑导出）覆盖 scienceplots 的同名键。"""
    rc, info = style.build_rcparams("aps", width="double", scienceplots=True)
    # scienceplots 的 'science' 设 figure.figsize=[3.5, 2.625]；契约层须覆盖为 aps 双栏 170mm。
    assert rc["figure.figsize"][0] == pytest.approx(style.mm_to_inch(170.0))
    assert rc["pdf.fonttype"] == 42 and rc["svg.fonttype"] == "none"
    assert rc["mathtext.fontset"] == "dejavusans"
    assert info.scienceplots == style.SCIENCEPLOTS_DEFAULT
    # scienceplots 的**非冲突**键确实生效（审美基底在位）
    assert "grid.linewidth" in rc


def test_off_by_default():
    rc, info = style.build_rcparams("aps")
    assert info.scienceplots == ()
    assert "grid.linewidth" not in rc


def test_style_context_accepts_switch_and_restores():
    import matplotlib.pyplot as plt

    before = plt.rcParams["pdf.fonttype"]
    with style.style_context("nature", scienceplots=True) as info:
        assert plt.rcParams["pdf.fonttype"] == 42
        assert info.scienceplots == style.SCIENCEPLOTS_DEFAULT
    assert plt.rcParams["pdf.fonttype"] == before, "退出 style_context 应还原 rcParams"


def test_resolve_scienceplots_shapes():
    assert style.resolve_scienceplots(True) == style.SCIENCEPLOTS_DEFAULT
    assert style.resolve_scienceplots(("nature",)) == ("nature",)
    assert style.resolve_scienceplots(None) == ()
    assert style.resolve_scienceplots(False) == ()


def test_styles_cli_lists_scienceplots(capsys):
    from pysci.skills.scientific_plotting.tools import figures

    rc = figures.cmd_styles(type("A", (), {})())
    out = capsys.readouterr().out
    assert rc == 0
    assert "scienceplots" in out and "science" in out


def test_runner_threads_scienceplots_from_style_yaml(tmp_path, monkeypatch):
    """STYLE.yaml 的 `scienceplots: true` 由 runner 透传到 style_context（端到端接线）。"""
    import contextlib
    import types

    import matplotlib.pyplot as plt

    from pysci.skills.scientific_plotting.tools import runner

    data_figures = tmp_path / "data" / "research" / "1_gain_ep" / "article" / "figures"
    src_figures = (
        tmp_path / "src" / "pysci" / "research" / "gain_ep" / "article" / "figures"
    )
    fd = data_figures / "fig_sp"
    (fd / "out").mkdir(parents=True, exist_ok=True)
    src_figures.mkdir(parents=True)
    (src_figures / "fig_sp.py").write_text("x = 1\n", encoding="utf-8")
    (fd / "STYLE.yaml").write_text("scienceplots: true\n", encoding="utf-8")
    monkeypatch.setattr(runner, "figures_code_root", lambda research: src_figures)
    monkeypatch.setattr(runner, "assert_within_data", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "_import_pipeline",
        lambda p: types.SimpleNamespace(build_figure=lambda **k: plt.figure()),
    )
    seen: dict = {}
    real = runner.style_context

    @contextlib.contextmanager
    def spy(preset, **kw):
        seen["preset"] = preset
        seen.update(kw)
        with real(preset, **kw) as info:
            yield info

    monkeypatch.setattr(runner, "style_context", spy)
    res = runner.build_figure_dir(fd, formats=("png",))
    assert res.error is None, res.error
    assert seen.get("scienceplots") is True
