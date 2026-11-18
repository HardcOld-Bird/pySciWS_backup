"""postprocess 的机械断言：rembg mock / alpha 合规验证 / Pillow 裁剪缩放护栏。"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from pysci.skills.image_generation import postprocess


class FakeRembg(types.ModuleType):
    """rembg 的最小替身：new_session + remove（合成 alpha 输出）。"""

    def __init__(self):
        super().__init__("rembg")
        self.sessions: list[str] = []

    def new_session(self, model: str = "u2net"):
        self.sessions.append(model)
        return f"session:{model}"

    def remove(self, im, session=None, alpha_matting: bool = False):
        arr = np.asarray(im.convert("RGBA")).copy()
        arr[:, :, 3] = 0  # 全透明背景
        arr[2:6, 2:6, 3] = 255  # 中心一块不透明主体
        return Image.fromarray(arr)


def _rgb_png(tmp_path, name="src.png", size=(8, 8)):
    p = tmp_path / name
    Image.new("RGB", size, (10, 120, 200)).save(p, format="PNG")
    return p


def test_remove_background_with_mock_rembg(tmp_path, monkeypatch):
    """rembg mock：抠图 → RGBA PNG 落盘，session 按 model 建立。"""
    fake = FakeRembg()
    monkeypatch.setitem(sys.modules, "rembg", fake)
    src = _rgb_png(tmp_path)
    dest = tmp_path / "cutout.png"
    out = postprocess.remove_background(src, dest, model="u2netp")
    assert out == dest
    assert fake.sessions == ["u2netp"]
    with Image.open(dest) as im:
        assert im.mode == "RGBA"
        alpha = np.asarray(im)[:, :, 3]
    assert alpha.max() == 255 and alpha.min() == 0


def test_remove_background_missing_rembg_graceful(tmp_path, monkeypatch):
    """rembg 未安装：抛带安装提示的 PostprocessError，不裸崩。"""
    monkeypatch.setitem(sys.modules, "rembg", None)  # 触发 ImportError
    with pytest.raises(postprocess.PostprocessError) as excinfo:
        postprocess.remove_background(_rgb_png(tmp_path), tmp_path / "x.png")
    assert "rembg" in str(excinfo.value)


def test_verify_alpha_ok_and_fake_transparent(tmp_path):
    """合规透明底：RGBA、alpha max==255 且有透明像素；251 假透明判 FAIL。"""
    ok_p = tmp_path / "ok.png"
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[:, :, :3] = 200
    arr[:, :, 3] = 255
    arr[0:2, 0:2, 3] = 0  # 有透明区
    Image.fromarray(arr).save(ok_p)
    report = postprocess.verify_alpha(ok_p)
    assert report["ok"] is True
    assert report["alpha_max"] == 255
    assert report["transparent_px"] > 0

    fake_p = tmp_path / "fake.png"
    arr2 = arr.copy()
    arr2[:, :, 3] = np.where(arr2[:, :, 3] == 0, 0, 251)  # "差一点不透明"
    Image.fromarray(arr2).save(fake_p)
    report2 = postprocess.verify_alpha(fake_p)
    assert report2["ok"] is False
    assert "251" in report2["reason"]

    rgb_p = _rgb_png(tmp_path, "rgb.png")
    report3 = postprocess.verify_alpha(rgb_p)
    assert report3["ok"] is False
    assert "alpha" in report3["reason"]


def test_crop_resize_order_and_guard(tmp_path):
    """crop → resize 固定顺序；默认拒绝覆盖既有目标（防误毁已计费产物）。"""
    src = _rgb_png(tmp_path, "big.png", size=(10, 10))
    dest = tmp_path / "out.png"
    out = postprocess.crop_resize(src, dest, crop=(2, 2, 6, 6), size=(8, 8))
    with Image.open(out) as im:
        assert im.size == (8, 8)
    # 覆盖护栏
    with pytest.raises(postprocess.PostprocessError, match="拒绝覆盖"):
        postprocess.crop_resize(src, dest, size=(4, 4))
    postprocess.crop_resize(src, dest, size=(4, 4), overwrite=True)
    with Image.open(dest) as im:
        assert im.size == (4, 4)


def test_crop_resize_bounds_and_missing(tmp_path):
    src = _rgb_png(tmp_path, "s.png", size=(5, 5))
    with pytest.raises(postprocess.PostprocessError, match="越界"):
        postprocess.crop_resize(src, tmp_path / "x.png", crop=(0, 0, 9, 9))
    with pytest.raises(postprocess.PostprocessError, match="不存在"):
        postprocess.crop_resize(tmp_path / "nope.png", tmp_path / "y.png")
