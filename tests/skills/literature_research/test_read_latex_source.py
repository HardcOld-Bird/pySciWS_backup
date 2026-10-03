"""``cmd_read`` 的 arXiv LaTeX-source 接线（方案 WP-G G5）及其降级加固。

SKILL.md 一直声称 ``read`` 依赖 arXiv LaTeX-source 路径做**公式忠实抽取**，但
``cmd_read`` 调 :func:`pdf_extract.extract_pdf` 时从不传 ``prefer_latex_source``——那条
路径只在 ``pdf_extract`` 自己的 ``__main__`` 后门里可达。声称与实现脱节了这么久而无人
发现，正因为它没有任何测试盯着。本文件就是那根钉子。

对物理声学用户这不是锦上添花：PDF 解析出的公式是碎片（``MinerU`` 云端能救，但要 token、
要排队、且对复杂 align 环境仍会错），而 LaTeX 源码里的 ``$$...$$`` 是**作者亲手写的**原文。

后半部分测的是接线带来的新风险及其加固：``_extract_from_arxiv_source`` 的 docstring 承诺
「失败返回 None」，但实现只把 ``import`` 包进了 try。接线之后它的异常会让整个
``extract_pdf`` 在第 1 步就死掉——连 PDF 后端都不试。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pysci.skills.literature_research.tools import arxiv_client, pdf_extract, research

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
_ARXIV_ID = "1803.04110"
_DOI = "10.1103/PhysRevLett.121.124501"


def _bundle(pdf_path: Path | None, html_path: Path | None = None) -> SimpleNamespace:
    """``browser_fetch.fetch_all`` 的返回形状（``cmd_read`` 只读这些属性）。"""
    return SimpleNamespace(
        pdf_path=pdf_path,
        html_path=html_path,
        ok=True,
        adapter="generic",
        browser="chromium",
        seconds=0.1,
        cloudflare=False,
        institutional_access=False,
        supp_paths=[],
        url="https://example.org/landing",
        title="Topological Acoustic States",
    )


def _args(target: str, **kw: Any) -> argparse.Namespace:
    """``cmd_read`` 读的 args 字段。``note=False`` 免得去碰真实的 ``papers/``。"""
    base: dict[str, Any] = {
        "target": target,
        "backend": None,
        "force": False,
        "note": False,
        "overwrite": False,
        "headed": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _fm(arxiv_id: str = "", **kw: Any) -> dict[str, Any]:
    """``_build_frontmatter`` 的返回形状（只列本文件关心的键）。"""
    base: dict[str, Any] = {
        "title": "Topological acoustic states in a non-Hermitian lattice",
        "short_title": "topological acoustic states",
        "arxiv_id": arxiv_id,
        "doi": _DOI,
    }
    base.update(kw)
    return base


@pytest.fixture
def extract_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """挡掉 ``cmd_read`` 的抓取与抽取，只记录 ``extract_pdf`` 收到的 kwargs。

    刻意把 ``_safe_fetch`` 与 ``arxiv_client.download_pdf`` **两条**取 PDF 的路都指向同
    一个实存的假文件：``cmd_read`` 只在 ``pdf_path.exists()`` 时才抽取，若文件不存在，
    测试会因为「压根没调 extract_pdf」而通过——那是假绿，不是接线正确。
    """
    calls: list[dict] = []
    # ``Settings`` 是 ``frozen=True`` 的 dataclass，不能逐个 setattr；造一份替身整体换掉
    # ``research.settings`` 就行——``cmd_read`` 里的每一处 ``settings.*`` 都经由那个名字。
    # 其余模块（``pdf_extract`` / ``arxiv_client``）仍指向真 settings，但它们的落盘动作
    # 在本夹具里全部被 mock 掉了，碰不到真实 ``data/``。
    monkeypatch.setattr(
        research,
        "settings",
        replace(
            research.settings,
            cache_dir=tmp_path / "cache",
            cache_api_responses=tmp_path / "cache" / "api_responses",
            cache_pdfs=tmp_path / "cache" / "pdfs",
            cache_extracted=tmp_path / "cache" / "extracted",
            cache_html_fulltext=tmp_path / "cache" / "html_fulltext",
        ),
    )

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")

    def _fake_extract(path: Any, **kwargs: Any) -> str:
        calls.append({"path": Path(path), **kwargs})
        return "# 抽取出来的正文\n"

    monkeypatch.setattr(research.pdf_extract, "extract_pdf", _fake_extract)
    monkeypatch.setattr(
        research, "_safe_fetch", lambda url, out_dir, args: _bundle(pdf)
    )
    monkeypatch.setattr(arxiv_client, "download_pdf", lambda *a, **k: pdf)
    # WP-H H5 之后，``cmd_read`` 在起浏览器之前会先试一次 OA 直链的**真 HTTP GET**。
    # 本文件只关心 LaTeX 接线，故把这第一档直接短路：否则 ``kind == "url"`` 的用例
    # 会在每次跑测试时联外网（慢、且无网环境下结果不同）。
    monkeypatch.setattr(research, "_try_oa_pdf", lambda *a, **k: None)
    return calls


def _mock_metadata(monkeypatch: pytest.MonkeyPatch, fm: dict | None) -> None:
    """把元数据链路指向给定的 frontmatter（``None`` = 解析失败，``fm`` 保持 None）。"""
    work = None if fm is None else {"id": "W1"}
    monkeypatch.setattr(
        research,
        "_resolve_work",
        lambda raw: ("openalex", work) if work else (None, None),
    )
    monkeypatch.setattr(research, "_enrich_work", lambda w, **k: w)
    monkeypatch.setattr(research, "_build_frontmatter", lambda src, w: dict(fm or {}))


# ===========================================================================
#  G5 —— cmd_read 真的把 arXiv id 传下去了
# ===========================================================================
def test_read_passes_the_arxiv_id_when_the_target_is_one(
    extract_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户直接给的 id 是最权威的来源，原样传下去。

    这里 frontmatter 的 ``arxiv_id`` 故意留空：``kind == "arxiv"`` 分支必须取 ``val``
    而不是 ``fm["arxiv_id"]``，否则「元数据没挖到 arXiv 预印本」会让用户明明手打的
    arXiv id 也失效——而那正是 ``read 1803.04110`` 最常见的用法。
    """
    _mock_metadata(monkeypatch, _fm())

    assert research.cmd_read(_args(_ARXIV_ID)) == 0

    assert len(extract_calls) == 1
    assert extract_calls[0]["prefer_latex_source"] == _ARXIV_ID


def test_read_passes_the_arxiv_id_from_metadata_for_a_doi_target(
    extract_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DOI 入口的论文若有 arXiv 预印本，同样该走 LaTeX 路径。

    那个 id 是 ``_resolve_work`` 从 OpenAlex 元数据里挖出来的，用户根本不知道它存在，
    也就无从在命令行里给出。少了这条回落，``read <doi>`` 永远只能拿 PDF 碎片公式。
    """
    _mock_metadata(monkeypatch, _fm(arxiv_id=_ARXIV_ID))

    assert research.cmd_read(_args(_DOI)) == 0

    assert extract_calls[0]["prefer_latex_source"] == _ARXIV_ID


def test_read_does_not_request_latex_for_a_paper_without_an_arxiv_id(
    extract_calls: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """纯期刊论文必须传 ``None`` 而不是空串。

    ``extract_pdf`` 的判据是 ``if prefer_latex_source:``，空串同样为假，所以传 ``""``
    在功能上无害；但传 ``None`` 才是签名的本意（``str | None``），且 ``aid or None``
    让「无 id」这件事在调用点上显式可见。更重要的是：``download_source`` 会真发一次
    HTTP 请求，对每篇无预印本的论文白发一次是纯浪费，而 arXiv 对高频请求返 429。
    """
    _mock_metadata(monkeypatch, _fm())

    assert research.cmd_read(_args(_DOI)) == 0

    assert extract_calls[0]["prefer_latex_source"] is None


def test_read_survives_a_url_target_with_no_metadata_at_all(
    extract_calls: list[dict],
) -> None:
    """``fm`` 到第 4 步才被兜底成 ``_minimal_fm``，而抽取发生在第 3 步——此刻它仍是
    ``None``。直接写 ``fm.get("arxiv_id")`` 会 AttributeError，让一次本来能成功的抓取
    整个失败。``kind == "url"`` 根本不查元数据，所以这是 ``fm is None`` 的必经形态。
    """
    assert research.cmd_read(_args("https://example.org/some/paper.pdf")) == 0

    assert len(extract_calls) == 1
    assert extract_calls[0]["prefer_latex_source"] is None


def test_read_reports_the_latex_path_it_is_about_to_try(
    extract_calls: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """进度提示要说出走了哪条路。

    LaTeX 抽取比 PDF 慢（要下载 tar.gz 再解包），没有这行提示时用户会以为卡住了；
    而它同时是排查「公式怎么还是碎片」的第一现场——一看就知道 id 有没有被认出来。
    """
    _mock_metadata(monkeypatch, _fm(arxiv_id=_ARXIV_ID))

    research.cmd_read(_args(_DOI))

    out = capsys.readouterr().out
    assert "arXiv LaTeX 源码" in out and _ARXIV_ID in out


# ===========================================================================
#  加固 —— LaTeX 路径不得有能力否决 PDF 路径
# ===========================================================================
def test_arxiv_source_returns_none_when_the_download_blows_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """兑现 docstring 那句「失败返回 None」。

    ``download_source`` 的 ``dest_dir.mkdir`` 在它自己的 try **之外**，磁盘满 / 无写权限
    / 路径非法都会抛 OSError。接线之前这只影响 ``pdf_extract`` 的 ``__main__`` 后门；
    接线之后它落在 ``read`` 主干上。
    """

    def _boom(*a: Any, **k: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(arxiv_client, "download_source", _boom)

    assert pdf_extract._extract_from_arxiv_source(_ARXIV_ID) is None


def test_arxiv_source_returns_none_when_the_tex_conversion_blows_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """下载与解包都成功、转换阶段才炸，同样要归为「这条路不通」而不是致命错误。"""
    tar = tmp_path / "src.tar.gz"
    tar.write_bytes(b"fake")
    monkeypatch.setattr(arxiv_client, "download_source", lambda *a, **k: tar)
    monkeypatch.setattr(
        arxiv_client, "extract_tex_from_source", lambda p: r"\section{Intro}"
    )

    def _boom(tex: str) -> str:
        raise ValueError("malformed LaTeX")

    monkeypatch.setattr(pdf_extract, "_tex_to_markdown", _boom)

    assert pdf_extract._extract_from_arxiv_source(_ARXIV_ID) is None


def test_extract_pdf_still_extracts_the_pdf_when_the_latex_path_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """加固的**目的**，端到端：LaTeX 只是更准的一条路，不该能否决那条能用的路。

    若异常外泄，``extract_pdf`` 在第 1 步就死掉、连后端都不试，于是一篇本来能抽出正文
    的论文变成「什么都没抽到」。这比公式不够准坏得多——前者是质量下降，后者是功能丧失。
    """
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")

    def _boom(*a: Any, **k: Any) -> None:
        raise OSError(28, "No space left on device")

    tried: list[str] = []

    def _fake_backend(path: Path) -> str:
        tried.append("pymupdf4llm")
        return "# PDF 正文"

    monkeypatch.setattr(arxiv_client, "download_source", _boom)
    monkeypatch.setattr(pdf_extract, "_BACKEND_IMPL", {"pymupdf4llm": _fake_backend})
    monkeypatch.setattr(pdf_extract, "_pick_backend", lambda name: "pymupdf4llm")

    md = pdf_extract.extract_pdf(
        pdf,
        backend="pymupdf4llm",
        use_cache=False,
        write_cache=False,
        prefer_latex_source=_ARXIV_ID,
    )

    assert md == "# PDF 正文"
    assert tried == ["pymupdf4llm"]  # 后端确实被试过了
    err = capsys.readouterr().err
    assert "arXiv LaTeX source failed" in err and _ARXIV_ID in err


def test_extract_pdf_prefers_the_latex_source_when_it_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对照面：LaTeX 可用时**不得**再去碰 PDF 后端。

    没有这条，上面那个「回落」测试会与「LaTeX 根本没被优先」的实现同样通过——两条一起
    才钉住优先级本身。这也正是 G5 的全部价值所在：公式来自作者亲手写的 ``$$...$$``。
    """
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")

    def _should_not_run(path: Path) -> str:
        raise AssertionError("LaTeX 源码可用时不该走 PDF 后端")

    monkeypatch.setattr(
        pdf_extract, "_extract_from_arxiv_source", lambda aid: "# 来自 LaTeX 源码\n"
    )
    monkeypatch.setattr(pdf_extract, "_BACKEND_IMPL", {"pymupdf4llm": _should_not_run})
    monkeypatch.setattr(pdf_extract, "_pick_backend", lambda name: "pymupdf4llm")

    md = pdf_extract.extract_pdf(
        pdf,
        backend="pymupdf4llm",
        use_cache=False,
        write_cache=False,
        prefer_latex_source=_ARXIV_ID,
    )

    assert md == "# 来自 LaTeX 源码\n"
