"""`export image` 的品牌（COMSOL logo）清除（backlog 20261015-204911）。

现场（I.i3 校准审查 P2）：生成脚本显式传 `clean=True`，导出图右上角仍带 COMSOL 蓝黄方形
logo（`raw/far_full_L.png` 黄像素 bbox=(881,13)-(884,16)），6 张远场 PNG 全带入最终综合图
→ reviewer 判 H3 FAIL。根因：`clean` 只关了 colorbar/图例/标题，**没碰品牌**。

修法两手（原生优先、像素兜底）：
1. 原生：Image 导出节点 `options*d=on` + `logo*d=off`（手册「Image (Export)」把 logo 列为
   optional component）；
2. 兜底：导出后像素核验——角区仍有品牌像素就地抹除并附 warning（原生开关在部分版本/绘图组
   类型上不生效，故不能只靠它）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pysci.paths import PROJECT_ROOT
from pysci.skills.comsol_simulation.tools import export as _export
from pysci.skills.comsol_simulation.tools import postprocess as pp

#: 真实带 logo 的导出物（I.i3 证据链的原始素材）；缺失则相关用例 skip。
REAL_BRANDED = (
    PROJECT_ROOT
    / "data/research/1_gain_ep/article/figures/beam_scatter/raw/far_full_L.png"
)

BRAND_BLUE = (100, 144, 200)
BRAND_YELLOW = (240, 240, 96)


def _canvas_with_logo(w: int = 400, h: int = 300) -> np.ndarray:
    """白底 + 右上角蓝黄品牌标记（仿真 COMSOL logo 的形状与配色）。"""
    a = np.full((h, w, 3), 255, dtype=np.uint8)
    a[5:20, w - 60 : w - 45] = BRAND_BLUE
    a[6:18, w - 43 : w - 27] = BRAND_YELLOW
    return a


# ---------------------------------------------------------------------------
# 探测（纯函数）
# ---------------------------------------------------------------------------
def test_detects_blue_yellow_pair_as_brand():
    box = pp.find_brand_box(_canvas_with_logo())
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 > 300 and y0 < 25 and x1 > x0 and y1 > y0


def test_yellow_only_corner_is_not_brand():
    """只黄不蓝 → 不是品牌（色标/等高线常带黄，防误报）。"""
    a = np.full((300, 400, 3), 255, dtype=np.uint8)
    a[5:20, 350:365] = BRAND_YELLOW
    assert pp.find_brand_box(a) is None


def test_blue_only_corner_is_not_brand():
    a = np.full((300, 400, 3), 255, dtype=np.uint8)
    a[5:20, 350:365] = BRAND_BLUE
    assert pp.find_brand_box(a) is None


def test_brand_colors_outside_corner_window_are_not_brand():
    """蓝黄成对但**在图中部**（= 数据面）→ 窗口判据把它排除。"""
    a = np.full((600, 800, 3), 255, dtype=np.uint8)
    a[300:320, 400:420] = BRAND_BLUE
    a[300:320, 425:445] = BRAND_YELLOW
    assert pp.find_brand_box(a) is None


def test_clean_png_has_no_brand():
    assert pp.find_brand_box(np.full((300, 400, 3), 255, dtype=np.uint8)) is None


# ---------------------------------------------------------------------------
# 抹除（文件级）
# ---------------------------------------------------------------------------
def test_strip_brand_clears_synthetic_logo(tmp_path):
    p = tmp_path / "syn.png"
    Image.fromarray(_canvas_with_logo()).save(p)
    assert pp.count_brand_pixels(p) > 0

    out = pp.strip_brand(p)

    assert out["found"] is True
    assert pp.count_brand_pixels(p) == 0
    assert out["fill"] == [255, 255, 255]  # 取自周边白底


def test_strip_brand_noop_on_clean_png(tmp_path):
    p = tmp_path / "clean.png"
    Image.fromarray(np.full((300, 400, 3), 255, dtype=np.uint8)).save(p)
    before = p.read_bytes()
    assert pp.strip_brand(p)["found"] is False
    assert p.read_bytes() == before  # 未动文件


@pytest.mark.skipif(not REAL_BRANDED.exists(), reason="真实带 logo 素材缺失")
def test_strip_brand_clears_the_real_comsol_logo(tmp_path):
    """复用 I.i3 的口径核验：真实导出物抹除后，角区品牌像素 **0 命中**。"""
    p = tmp_path / "copy.png"
    p.write_bytes(REAL_BRANDED.read_bytes())
    assert pp.count_brand_pixels(p) > 0  # 前提：原图确有品牌像素

    out = pp.strip_brand(p)

    assert out["found"] is True
    assert pp.count_brand_pixels(p) == 0


# ---------------------------------------------------------------------------
# export_image 集成（假 COMSOL 节点）
# ---------------------------------------------------------------------------
class _FakeNode:
    def __init__(self, tag, kind, png_writer=None):
        self.tag, self.kind = tag, kind
        self.store: dict[str, object] = {}
        self._png_writer = png_writer

    def set(self, key, value):
        self.store[key] = value

    def run(self):
        if self._png_writer is not None:
            out = Path(str(self.store["pngfilename"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self._png_writer()).save(out)


class _FakeSeq:
    def __init__(self, created_hook):
        self._nodes: dict[str, _FakeNode] = {}
        self._hook = created_hook

    def tags(self):
        return list(self._nodes)

    def create(self, tag, kind):
        n = self._hook(tag, kind)
        self._nodes[tag] = n
        return n

    def get(self, tag):
        return self._nodes.get(tag)


class _FakeResultHolder:
    def __init__(self, exports, dsets):
        self._exports, self._dsets = exports, dsets

    def export(self):
        return self._exports

    def dataset(self):
        return self._dsets


class _FakePG:
    def __init__(self):
        self.store: dict[str, object] = {}

    def set(self, key, value):
        self.store[key] = value


class _FakeModel:
    """最小假 model：导出节点 run() 时写出「带品牌」的 PNG，供像素兜底链验。"""

    def __init__(self, png_writer):
        self.exports = _FakeSeq(lambda t, k: _FakeNode(t, k, png_writer))
        self.dsets = _FakeSeq(lambda t, k: _FakeNode(t, k))
        self._holder = _FakeResultHolder(self.exports, self.dsets)
        self.pg = _FakePG()

    def result(self, tag=None):
        return self._holder if tag is None else self.pg


def test_clean_sets_native_logo_off_and_pixels_are_clean(tmp_path):
    """clean=True → 原生 logo*d=off + options*d=on；且导出物角区品牌像素 0 命中。"""
    m = _FakeModel(lambda: _canvas_with_logo())
    out = tmp_path / "branded.png"

    r = _export.export_image(m, "pg_far", out, clean=True)

    assert r.ok, r.report()
    node = m.exports.tags() and m.exports.get(next(iter(m.exports.tags())))
    for prop in ("options1d", "options2d", "options3d"):
        assert node.store.get(prop) == "on", f"{prop} 未置 on：{node.store}"
    for prop in ("logo1d", "logo2d", "logo3d"):
        assert node.store.get(prop) == "off", f"{prop} 未置 off：{node.store}"
    # 像素兜底生效：原图带品牌 → 抹除 → 0 命中
    assert pp.count_brand_pixels(out) == 0
    assert any("已就地抹除" in w for w in r.warnings), r.warnings


def test_sidecar_records_brand_readout(tmp_path):
    m = _FakeModel(lambda: _canvas_with_logo())
    out = tmp_path / "b.png"

    _export.export_image(m, "pg_far", out, clean=True)

    import json

    sc = json.loads((tmp_path / "b.sidecar.json").read_text(encoding="utf-8"))
    assert sc["brand"]["found"] is True
    assert sc["brand"]["bbox"] and "brand_pixels_stripped" in sc["clean_applied"]


def test_no_strip_by_default_keeps_pixels(tmp_path):
    """不带 clean/strip_brand → **不动像素**（默认行为不变，品牌仍在）。"""
    m = _FakeModel(lambda: _canvas_with_logo())
    out = tmp_path / "keep.png"

    r = _export.export_image(m, "pg_far", out)

    assert r.ok
    assert pp.count_brand_pixels(out) > 0
    assert not any("已就地抹除" in w for w in r.warnings)


def test_strip_brand_alone_without_clean(tmp_path):
    """显式 --strip-brand（clean=False）也清品牌，但**不**关 colorbar/标题。"""
    m = _FakeModel(lambda: _canvas_with_logo())
    out = tmp_path / "sb.png"

    _export.export_image(m, "pg_far", out, strip_brand=True)

    assert pp.count_brand_pixels(out) == 0
    node = m.exports.get(next(iter(m.exports.tags())))
    assert "options2d" not in node.store  # 原生 logo 开关只随 clean 走
