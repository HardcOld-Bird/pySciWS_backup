"""样式契约机械断言：尺寸契约解析 / 样式资产键值（纯文本解析）/ 叠加应用。

纯逻辑不依赖绘图栈；rcParams 应用用例在 matplotlib 就位后跑（skipif 保底，
依赖代理装好即自动生效）。
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from pysci.skills.scientific_plotting import styles

_HAS_MPL = find_spec("matplotlib") is not None


def test_journal_width_contract():
    assert styles.resolve_width_mm("aps", "single") == 85.0
    assert styles.resolve_width_mm("aps", "double") == 170.0
    assert styles.resolve_width_mm("nature", "single") == 89.0
    assert styles.resolve_width_mm("nature", "double") == 183.0
    assert styles.resolve_width_mm("APS", "Double") == 170.0  # 大小写不敏感


def test_unknown_contract_raises():
    with pytest.raises(ValueError, match="未知尺寸契约"):
        styles.resolve_width_mm("science", "single")
    with pytest.raises(ValueError, match="未知尺寸契约"):
        styles.resolve_width_mm("aps", "triple")


def test_mm_to_inch():
    assert styles.mm_to_inch(25.4) == pytest.approx(1.0)
    assert styles.mm_to_inch(170.0) == pytest.approx(170.0 / 25.4)


def test_style_assets_present():
    assert set(styles.list_styles()) == {"house", "journal-aps", "journal-nature"}
    for name in styles.list_styles():
        assert styles.style_path(name).is_file()


def test_style_missing_lists_available():
    with pytest.raises(FileNotFoundError, match="journal-aps"):
        styles.style_path("no-such-style")


def test_house_cycle_matches_house_mplstyle():
    keys = styles.read_style_keys("house")
    cycle_hex = styles.HOUSE_CYCLE
    for hex_color in cycle_hex:
        assert hex_color.lstrip("#").lower() in keys["axes.prop_cycle"].lower()
    # 顺序：prop_cycle 里 hex 的出现顺序与 HOUSE_CYCLE 一致
    positions = [
        keys["axes.prop_cycle"].lower().index(h.lstrip("#").lower()) for h in cycle_hex
    ]
    assert positions == sorted(positions)


def test_house_style_is_colorblind_safe_subset():
    # 家风循环 ⊂ Okabe-Ito，且恰好去掉黄色 #F0E442
    assert set(styles.HOUSE_CYCLE) <= set(styles.OKABE_ITO)
    assert "#F0E442" not in styles.HOUSE_CYCLE
    assert len(styles.HOUSE_CYCLE) == len(styles.OKABE_ITO) - 1


def test_journal_styles_carry_fonttype_contract():
    for name, first_font, size in (
        ("journal-aps", "arial", "8"),
        ("journal-nature", "helvetica", "7"),
    ):
        keys = styles.read_style_keys(name)
        assert keys["pdf.fonttype"] == "42"
        assert keys["ps.fonttype"] == "42"
        assert keys["svg.fonttype"] == "none"
        assert keys["font.size"] == size
        assert keys["font.sans-serif"].split(",")[0].strip().lower() == first_font


def test_read_style_keys_skips_comments():
    keys = styles.read_style_keys("house")
    assert all(not k.startswith("#") for k in keys)
    assert keys["axes.spines.top"] == "False"


def test_ensure_plotting_env_redirects_home_and_mplconfig(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "C:/old/home")
    monkeypatch.setenv("USERPROFILE", "C:/old/home")
    monkeypatch.setenv("MPLCONFIGDIR", "C:/old/mpl")
    target = styles.ensure_plotting_env(tmp_path / "home")
    assert target == tmp_path / "home"
    assert target.is_dir()
    import os

    assert os.environ["HOME"] == str(target)
    assert os.environ["USERPROFILE"] == str(target)
    assert os.environ["MPLCONFIGDIR"] == str(target / "matplotlib")
    # 默认目标在 PLOTTING_ROOT 缓存区（monkeypatch 收尾还原 env，不留全局副作用）
    assert styles.ensure_plotting_env().is_relative_to(styles.PLOTTING_ROOT)


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装（并行依赖代理安装中）")
def test_style_context_applies_and_restores():
    import matplotlib

    before = matplotlib.rcParams["pdf.fonttype"]
    with styles.style_context("house", "journal-aps"):
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42
        assert matplotlib.rcParams["svg.fonttype"] == "none"
        assert matplotlib.rcParams["font.size"] == 8
    assert matplotlib.rcParams["pdf.fonttype"] == before


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装（并行依赖代理安装中）")
def test_load_rc_params_layering():
    merged = styles.load_rc_params("house", "journal-nature")
    # 契约层最后应用：字号取 journal-nature 的 7，色循环取 house 的家风循环
    assert merged["font.size"] == 7
    assert "0072b2" in str(merged["axes.prop_cycle"]).lower()
