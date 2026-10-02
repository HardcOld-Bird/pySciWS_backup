"""paths 锚点单测：AI_DRAWING_ROOT / research_artwork_dir / assert_within_data。

只验证路径解析的确定性行为（不写盘、不联网）。
"""

from __future__ import annotations

import pytest

from pysci import paths


def test_ai_drawing_root_anchor():
    assert (
        paths.AI_DRAWING_ROOT == paths.PROJECT_ROOT / "data" / "skills" / "ai_drawing"
    )
    # 与其它技能数据区同构（同在 data/skills/ 下）
    assert paths.AI_DRAWING_ROOT.parent == paths.PLOTTING_ROOT.parent


def test_research_artwork_dir_mirrors_fig_dir():
    """artwork 目录与 figures 目录同在 article/ 下，仅末级名不同。"""
    fig = paths.research_fig_dir("gain_ep")
    art = paths.research_artwork_dir("gain_ep")
    assert art == fig.parent / "artwork"
    assert art.name == "artwork"
    # 带 slug → 追加一级
    assert (
        paths.research_artwork_dir("gain_ep", slug="fig1_cover") == art / "fig1_cover"
    )


def test_research_artwork_dir_resolves_numbered_asset(tmp_path, monkeypatch):
    """存在带数字前缀的资产目录时，artwork 落在该目录下。"""
    fake_asset = tmp_path / "3_demo"
    (fake_asset / "article").mkdir(parents=True)
    monkeypatch.setattr(paths, "ASSET_ROOT", tmp_path)
    art = paths.research_artwork_dir("demo")
    assert art == fake_asset / "article" / "artwork"


def test_assert_within_data_accepts_ai_drawing_root():
    # AI 绘图数据区在 data/ 根内 → 护栏放行，返回解析后的绝对路径
    resolved = paths.assert_within_data(paths.AI_DRAWING_ROOT, what="AI 绘图数据区")
    assert resolved == paths.AI_DRAWING_ROOT.resolve()


def test_assert_within_data_rejects_stray(tmp_path):
    # data/ 根外的路径 → 抛 ValueError（防 stray）
    with pytest.raises(ValueError, match="data/ 根内"):
        paths.assert_within_data(tmp_path / "stray.png", what="测试产物")
