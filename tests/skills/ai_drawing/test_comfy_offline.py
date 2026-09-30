"""comfy_client / comfy_session 的**离线**单测（不需活体服务器）。

只覆盖确定性、不联网重活的路径：URL 解析、探活对死端口返回 False、上传/提交对不可达或
缺文件抛错、状态字典结构。活体内省 / 真实出图见 test_comfy_live.py（hardware+slow）。
"""

from __future__ import annotations

import dataclasses

import pytest

from pysci.skills.ai_drawing.tools import comfy_session
from pysci.skills.ai_drawing.tools.comfy_client import (
    ComfyClient,
    ComfyError,
    save_images,
)
from pysci.skills.ai_drawing.tools.config import ComfyInstall

# 一个几乎不可能被占用的端口（连接立即被拒）。
_DEAD_URL = "http://127.0.0.1:1"


@pytest.fixture
def dead_session(monkeypatch):
    """把 comfy_session.settings 指向死端口，使探活确定性返回 False。"""
    monkeypatch.setattr(
        comfy_session,
        "settings",
        dataclasses.replace(
            comfy_session.settings, comfy=ComfyInstall(server_url=_DEAD_URL)
        ),
    )


def test_url_host_port_parsing():
    assert comfy_session._url_host_port("http://127.0.0.1:8188") == ("127.0.0.1", 8188)
    assert comfy_session._url_host_port("https://comfy.example.com") == ("comfy.example.com", 443)
    assert comfy_session._url_host_port("http://h") == ("h", 8188)  # 无端口 → 默认


def test_is_available_false_for_dead_port(dead_session):
    assert comfy_session.is_available() is False


def test_check_available_shape(dead_session):
    info = comfy_session.check_available()
    assert info["port_alive"] is False
    assert info["host"] == "127.0.0.1"
    assert info["port"] == 1
    assert "server_url" in info and "comfy_cli" in info


def test_require_available_raises_when_dead(dead_session):
    with pytest.raises(comfy_session.SessionError, match="不可达"):
        comfy_session.require_available()


def test_status_shape(dead_session):
    st = comfy_session.status()
    assert st["running"] is False
    assert st["host"] == "127.0.0.1" and st["port"] == 1
    assert "state_file" in st


def test_client_is_reachable_false_for_dead_port():
    c = ComfyClient(base_url=_DEAD_URL, timeout=1.0)
    assert c.is_reachable(timeout=1.0) is False


def test_client_base_url_strips_slash():
    c = ComfyClient(base_url="http://x:8188/")
    assert c.base_url == "http://x:8188"


def test_client_upload_missing_file_raises(tmp_path):
    c = ComfyClient(base_url=_DEAD_URL, timeout=1.0)
    with pytest.raises(ComfyError, match="不存在"):
        c.upload_image(tmp_path / "nope.png")


def test_client_queue_prompt_unreachable_raises():
    c = ComfyClient(base_url=_DEAD_URL, timeout=1.0)
    with pytest.raises(ComfyError):
        c.queue_prompt({"1": {"class_type": "SaveImage", "inputs": {}}})


def test_client_system_stats_unreachable_raises():
    c = ComfyClient(base_url=_DEAD_URL, timeout=1.0)
    with pytest.raises(ComfyError):
        c.system_stats()


def test_save_images_writes_bytes(tmp_path):
    imgs = [
        {"filename": "a.png", "subfolder": "", "type": "output", "data": b"\x89PNG-a"},
        {"filename": "b.png", "subfolder": "", "type": "output", "data": b"\x89PNG-b"},
    ]
    written = save_images(imgs, tmp_path / "out", stem="gen")
    assert len(written) == 2
    assert all(p.is_file() for p in written)
    # 多张 + stem → 名字含序号，且不覆盖
    assert written[0].read_bytes() == b"\x89PNG-a"
    assert written[0].name != written[1].name


def test_save_images_sanitizes_separators(tmp_path):
    imgs = [{"filename": "sub/dir\\x.png", "data": b"x"}]
    written = save_images(imgs, tmp_path / "out")
    assert "/" not in written[0].name and "\\" not in written[0].name
