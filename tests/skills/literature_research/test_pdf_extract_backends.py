"""pdf_extract MinerU 云端后端离线快照测试（阶段 4：mineru-open-sdk 替换手写 HTTP 层）。

锁定以下契约，全部通过 monkeypatch 拦截 mineru SDK / settings / fitz，绝不触网、绝不
消耗 MinerU 配额：
- ``_mineru_extract_pages`` 以正确 kwargs 调用 ``MinerU.extract``（vlm/公式/表格/OCR/
  英文/latex/超时），并在用后 ``close()``；
- 超 600 页时 ``_extract_with_mineru_cloud`` 用 SDK 的 ``pages=`` 分段（而非 fitz 物理切分）；
- state 非 done / markdown 为空 → 抛 ExtractionFailed；
- ``_mineru_cloud_ready`` 依赖 MINERU_TOKEN + mineru 模块；
- 后端优先级与 ``_BACKEND_IMPL`` 注册表不变；手写 HTTP/分页旧符号已彻底移除。
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import pdf_extract as pe


# ---------------------------------------------------------------------------
# 假 mineru SDK（记录构造 token 与 extract() 调用参数）
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, markdown, state):
        self.markdown = markdown
        self.state = state


class _FakeMinerU:
    """返回内容由类属性 result_md / result_state 控制，便于逐例改写。"""

    instances: list = []
    result_md = "# Title\n\nSome extracted markdown body with enough content."
    result_state = "done"

    def __init__(self, token=None, **kwargs):
        self.token = token
        self.init_kwargs = kwargs
        self.calls: list[tuple] = []
        self.closed = False
        type(self).instances.append(self)

    def extract(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return _FakeResult(type(self).result_md, type(self).result_state)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_mineru(monkeypatch):
    _FakeMinerU.instances = []
    _FakeMinerU.result_md = (
        "# Title\n\nSome extracted markdown body with enough content."
    )
    _FakeMinerU.result_state = "done"
    mod = types.ModuleType("mineru")
    mod.MinerU = _FakeMinerU
    monkeypatch.setitem(sys.modules, "mineru", mod)
    return _FakeMinerU


def _with_token(monkeypatch, token="tok-123"):
    monkeypatch.setattr(
        pe, "settings", dataclasses.replace(pe.settings, mineru_token=token)
    )


# ---------------------------------------------------------------------------
# _mineru_extract_pages —— SDK 调用契约
# ---------------------------------------------------------------------------
def test_extract_pages_sdk_kwargs(fake_mineru, monkeypatch):
    _with_token(monkeypatch, "tok-123")
    md = pe._mineru_extract_pages(Path("paper.pdf"))
    inst = fake_mineru.instances[0]
    assert inst.token == "tok-123"  # token 从 settings 注入 SDK
    src, kw = inst.calls[0]
    assert src == "paper.pdf"  # str(path) 作为 source 位置参数
    assert kw["model"] == "vlm"
    assert kw["formula"] is True  # 公式→LaTeX
    assert kw["table"] is True
    assert kw["ocr"] is True
    assert kw["language"] == "en"  # 覆盖 SDK 默认 "ch"
    assert kw["extra_formats"] == ["latex"]
    assert kw["pages"] is None  # 整份
    assert kw["timeout"] == pe.MINERU_TIMEOUT
    assert inst.closed is True  # 用后 close()
    assert md == fake_mineru.result_md


def test_extract_pages_passes_page_range(fake_mineru, monkeypatch):
    _with_token(monkeypatch)
    pe._mineru_extract_pages(Path("big.pdf"), pages="601-1200")
    _, kw = fake_mineru.instances[0].calls[0]
    assert kw["pages"] == "601-1200"


def test_extract_pages_empty_token_passes_none(fake_mineru, monkeypatch):
    """token 为空 → 传 None（SDK 转 flash-only 模式，auth 方法将抛 NoAuthClientError）。"""
    _with_token(monkeypatch, "")
    pe._mineru_extract_pages(Path("p.pdf"))
    assert fake_mineru.instances[0].token is None


def test_extract_pages_raises_on_failed_state(fake_mineru, monkeypatch):
    _with_token(monkeypatch)
    fake_mineru.result_state = "failed"
    with pytest.raises(pe.ExtractionFailed):
        pe._mineru_extract_pages(Path("p.pdf"))


def test_extract_pages_raises_on_empty_markdown(fake_mineru, monkeypatch):
    _with_token(monkeypatch)
    fake_mineru.result_md = "   \n  "
    with pytest.raises(pe.ExtractionFailed):
        pe._mineru_extract_pages(Path("p.pdf"))


# ---------------------------------------------------------------------------
# _extract_with_mineru_cloud —— 分段策略（SDK pages=，非 fitz 物理切分）
# ---------------------------------------------------------------------------
def _capture_pages(monkeypatch, n_pages):
    monkeypatch.setattr(pe, "_pdf_page_count", lambda p: n_pages)
    calls: list = []

    def fake(p, pages=None):
        calls.append(pages)
        return f"[{pages}]"

    monkeypatch.setattr(pe, "_mineru_extract_pages", fake)
    return calls


def test_cloud_single_call_under_limit(monkeypatch):
    calls = _capture_pages(monkeypatch, 120)
    out = pe._extract_with_mineru_cloud(Path("p.pdf"))
    assert calls == [None]  # 整份一次
    assert out == "[None]"


def test_cloud_exactly_at_limit_is_single(monkeypatch):
    calls = _capture_pages(monkeypatch, pe.MINERU_MAX_PAGES)  # 600
    pe._extract_with_mineru_cloud(Path("p.pdf"))
    assert calls == [None]  # 600 <= 600 → 不分段


def test_cloud_unknown_page_count_is_single(monkeypatch):
    calls = _capture_pages(monkeypatch, 0)  # fitz 不可用/未知 → 保守整份尝试
    pe._extract_with_mineru_cloud(Path("p.pdf"))
    assert calls == [None]


def test_cloud_chunks_over_limit(monkeypatch):
    calls = _capture_pages(monkeypatch, 601)
    out = pe._extract_with_mineru_cloud(Path("p.pdf"))
    assert calls == ["1-600", "601-601"]  # 用 pages= 分段，不切临时文件
    assert "MinerU pages 1-600" in out
    assert "MinerU pages 601-601" in out
    assert "[1-600]" in out and "[601-601]" in out


def test_cloud_chunks_multi_segment(monkeypatch):
    calls = _capture_pages(monkeypatch, 1500)
    pe._extract_with_mineru_cloud(Path("p.pdf"))
    assert calls == ["1-600", "601-1200", "1201-1500"]


# ---------------------------------------------------------------------------
# 后端可用性 / 注册表
# ---------------------------------------------------------------------------
def test_cloud_ready_needs_token_and_module(monkeypatch):
    monkeypatch.setattr(pe, "_has_module", lambda name: name == "mineru")
    _with_token(monkeypatch, "t")
    assert pe._mineru_cloud_ready() is True
    _with_token(monkeypatch, "")
    assert pe._mineru_cloud_ready() is False  # 无 token
    _with_token(monkeypatch, "t")
    monkeypatch.setattr(pe, "_has_module", lambda name: False)
    assert pe._mineru_cloud_ready() is False  # 无 mineru 模块


def test_backend_registry_and_priority():
    assert pe.PREFERRED_ORDER == ("mineru-cloud", "pymupdf4llm")
    assert set(pe._BACKEND_IMPL) == {"mineru-cloud", "pymupdf4llm"}
    assert pe._BACKEND_IMPL["mineru-cloud"] is pe._extract_with_mineru_cloud
    assert pe._BACKEND_IMPL["pymupdf4llm"] is pe._extract_with_pymupdf4llm


def test_mineru_constants():
    assert pe.MINERU_MAX_PAGES == 600  # SDK Precision 上限（旧手写为 200）
    assert pe.MINERU_CHUNK_PAGES == 600
    assert pe.MINERU_MODEL == "vlm"
    assert pe.MINERU_LANGUAGE == "en"


def test_legacy_handwritten_symbols_removed():
    """手写 HTTP 上传/轮询/解压与 fitz 物理分页已彻底移除。"""
    for gone in (
        "_mineru_extract_one",
        "_split_pdf_into_chunks",
        "MINERU_API_BASE",
        "MINERU_MODEL_VERSION",
        "MINERU_IS_OCR",
        "MINERU_POLL_INTERVAL",
        "MINERU_POLL_TIMEOUT",
    ):
        assert not hasattr(pe, gone), gone


def test_pdf_page_count_missing_file_returns_zero(tmp_path):
    assert pe._pdf_page_count(tmp_path / "nope.pdf") == 0
