"""pysci.paths 规范产物解析器 + 防 stray 护栏单测（P2-a）。

覆盖：research_fig_dir / research_theory_dir 的规范位置解析，以及
assert_within_data 对 data/ 根内放行、根外（stray）写入前即报错。
"""

from __future__ import annotations

import pytest

from pysci import paths


def test_research_fig_dir_canonical_location():
    p = paths.research_fig_dir("gain_ep")
    assert p == paths.research_asset_dir("gain_ep") / "article" / "figures"
    assert paths.DATA_ROOT in p.parents


def test_research_fig_dir_with_slug():
    p = paths.research_fig_dir("gain_ep", slug="fig1_ep_band")
    assert p.name == "fig1_ep_band"
    assert p.parent == paths.research_fig_dir("gain_ep")


def test_research_theory_dir_canonical_location():
    p = paths.research_theory_dir("gain_ep")
    assert p == paths.research_asset_dir("gain_ep") / "theory"
    assert paths.research_theory_dir("gain_ep", slug="cmt").name == "cmt"


def test_assert_within_data_accepts_inside():
    inside = paths.DATA_ROOT / "skills" / "scientific_plotting" / "x.png"
    assert paths.assert_within_data(inside) == inside.resolve()
    # data 根本身也放行
    assert paths.assert_within_data(paths.DATA_ROOT) == paths.DATA_ROOT.resolve()


def test_assert_within_data_rejects_stray(tmp_path):
    stray = tmp_path / "scripts" / "out.png"
    with pytest.raises(ValueError, match="不在 data/ 根内"):
        paths.assert_within_data(stray, what="图产物")
