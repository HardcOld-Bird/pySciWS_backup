"""``cmd_read`` 的抓取顺序与网页正文兜底（方案 WP-H H5）。

两部分，各自钉住一类失败：

**H5-1 先试 OA 直链**。原实现对 DOI 一律拼 ``https://doi.org/{doi}`` 直接起 Playwright，
而 ``oa_url`` 只在 ``kind == "openalex"`` 分支被用到。对 OpenAlex 已标明开放获取的论文，
这等于每次都为一份能普通 GET 到的 PDF 白付一次浏览器启动 + 页面渲染 + 反爬博弈，
而浏览器那条路还会因为 Cloudflare / 选择器失配而失败——**明明有免费的直链**。

**H5-2 网页正文兜底不再冒充 PDF 抽取**。原实现把 ``bundle.html_path`` 整篇读进 ``md``，
于是它被写进 ``cache/extracted/<stem>_fulltext.md`` 并记为 ``extracted_md_path``：
RAG 语料里混进一段来源与质量都不同的网页正文（公式通常丢失），同一份内容在两个
Tier A 目录里各存一遍，而且从 frontmatter 上**看不出**这篇的全文其实是网页抓的。

注：方案原文说这里塞的是「原始 HTML」，实测**不是**——``browser_fetch.to_markdown``
写的是 trafilatura 抽出的结构化 markdown（``favor_precision``、不含评论），头部还带
browser_fetch 自己的 frontmatter。所以污染是「异源 frontmatter + 公式缺失 + 重复副本 +
provenance 丢失」，不是「HTML 标记灌进语料」。修正后的实现也因此更简单：那份文件
本来就在 ``cache/html_fulltext/<slug>/<slug>.md``，只需把路径记进新字段，不必再复制。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pysci.skills.literature_research.tools import research

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
_DOI = "10.1103/PhysRevLett.121.124501"
_OA_URL = "https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.124501"
_PDF_BYTES = b"%PDF-1.7\n%some pdf\n%%EOF\n"


def _args(target: str, **kw: Any) -> argparse.Namespace:
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


def _bundle(pdf_path: Path | None, html_path: Path | None = None) -> SimpleNamespace:
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
        char_count=1234,
    )


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """把 ``cmd_read`` 看到的全部路径指到 tmp_path，并挡掉真网络与真抽取。

    返回一个可写的盒子：``box["work"]`` 决定元数据（``None`` = 解析失败），
    ``box["fetch_calls"]`` / ``box["extract_calls"]`` 记录两条下游路径**是否被走到**。
    """
    monkeypatch.setattr(
        research,
        "settings",
        replace(
            research.settings,
            cache_dir=tmp_path / "cache",
            cache_pdfs=tmp_path / "cache" / "pdfs",
            cache_extracted=tmp_path / "cache" / "extracted",
            cache_html_fulltext=tmp_path / "cache" / "html_fulltext",
        ),
    )

    box: dict[str, Any] = {
        "work": None,
        "fm": None,
        "fetch_calls": [],
        "extract_calls": [],
        "bundle": _bundle(None),
        "merged_fm": [],
    }

    monkeypatch.setattr(
        research,
        "_resolve_work",
        lambda raw: ("openalex", box["work"]) if box["work"] else (None, None),
    )
    monkeypatch.setattr(research, "_enrich_work", lambda w, **k: w)
    monkeypatch.setattr(
        research, "_build_frontmatter", lambda src, w: dict(box["fm"] or {})
    )

    def _fake_fetch(url: str, out_dir: Path, args: Any) -> Any:
        box["fetch_calls"].append(url)
        return box["bundle"]

    def _fake_extract(path: Any, **kwargs: Any) -> str:
        box["extract_calls"].append(Path(path))
        return "# 抽取出来的正文\n"

    def _fake_merge(fm: dict, **kwargs: Any) -> tuple[Path, str, list[str]]:
        box["merged_fm"].append(dict(fm))
        return tmp_path / "note.md", research.NOTE_CREATED, []

    monkeypatch.setattr(research, "_safe_fetch", _fake_fetch)
    monkeypatch.setattr(research.pdf_extract, "extract_pdf", _fake_extract)
    monkeypatch.setattr(research, "_merge_note", _fake_merge)
    return SimpleNamespace(root=tmp_path, box=box)


class _FakeResponse:
    """``requests.Response`` 里 ``_try_oa_pdf`` 真正用到的那一小块。"""

    def __init__(self, status_code: int = 200, body: bytes = b"") -> None:
        self.status_code = status_code
        self._body = body

    def iter_content(self, chunk_size: int = 1) -> Any:
        # 刻意切成 1 字节一片：验魔数的循环必须自己累够 5 字节，而不能假定
        # 第一个 chunk 就有 ``%PDF-``（``iter_content`` 只保证**至多** chunk_size）。
        for i in range(0, len(self._body), 1):
            yield self._body[i : i + 1]


class _FakeSession:
    def __init__(self, resp: _FakeResponse | Exception, log: list) -> None:
        self._resp = resp
        self._log = log

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *a: Any) -> bool:
        return False

    def get(self, url: str, **kw: Any) -> _FakeResponse:
        self._log.append({"url": url, **kw})
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _fake_http(
    monkeypatch: pytest.MonkeyPatch, resp: _FakeResponse | Exception
) -> list[dict]:
    """把 ``research.http_session`` 换成假实现，返回它收到的请求日志。"""
    log: list[dict] = []
    monkeypatch.setattr(
        research, "http_session", lambda *a, **k: _FakeSession(resp, log)
    )
    return log


def _seed_oa_work(box: dict[str, Any]) -> None:
    """造一个「OpenAlex 已标明开放获取」的现场：``oa_url`` 必须挂在 **work** 上。

    挂在 ``fm`` 上是无效的——``cmd_read`` 在抓取阶段读的是 ``(work or {}).get("oa_url")``；
    frontmatter 要到第 4 步才参与决策，那时 PDF 早就抓完了。这里同时填 ``fm["oa_url"]``
    只为让夹具形状贴近真实产物（``work_to_note_frontmatter`` 确实会把它写进笔记），
    被断言依赖的是 work 上那一份。``work["doi"]`` 供 ``_oa_pdf_filename`` 派生稳定文件名。
    """
    box["work"] = {"id": "W1", "doi": _DOI, "oa_url": _OA_URL}
    box["fm"] = {"title": "T", "short_title": "t", "doi": _DOI, "oa_url": _OA_URL}


# ===========================================================================
#  H5-1 抓取顺序：OA 直链在浏览器之前
# ===========================================================================
def test_a_doi_with_an_oa_url_never_launches_a_browser(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OA 直链拿到 PDF 时，``_safe_fetch`` 一次都不该被调用。

    这是 H5-1 的全部价值：不是「多一条路」，而是**把浏览器那条路彻底省掉**。
    断言用 ``fetch_calls == []`` 而不是「最终拿到了 PDF」——后者在旧实现下同样成立
    （浏览器也能拿到），钉不住顺序。
    """
    box = sandbox.box
    _seed_oa_work(box)
    got = _fake_http(monkeypatch, _FakeResponse(200, _PDF_BYTES))

    assert research.cmd_read(_args(_DOI)) == 0

    assert box["fetch_calls"] == []
    assert [c["url"] for c in got] == [_OA_URL]
    # 下载到的 PDF 确实进了抽取链路（而不是拿到就扔）
    assert len(box["extract_calls"]) == 1
    assert box["extract_calls"][0].parent == sandbox.root / "cache" / "pdfs"


def test_the_browser_is_the_fallback_when_the_direct_link_fails(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """直链拿不到（403 / 落地页 / 断网）时仍要回落到浏览器——不能因为多了一条
    免费路就把原来那条能用的路弄丢。回落目标是 ``https://doi.org/{doi}``（原行为）。
    """
    box = sandbox.box
    _seed_oa_work(box)
    _fake_http(monkeypatch, _FakeResponse(403, b""))
    box["bundle"] = _bundle(sandbox.root / "from_browser.pdf")

    research.cmd_read(_args(_DOI))

    assert box["fetch_calls"] == [f"https://doi.org/{_DOI}"]


def test_a_200_html_page_is_not_mistaken_for_a_pdf(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """出版商的「无权访问」页面常返 **200 + text/html**。

    判据只看魔数而不看 ``Content-Type``：机构库常返 ``application/octet-stream``
    （按 Content-Type 判会漏掉真 PDF），而这里按 Content-Type 判会**收下**一个 HTML
    页面并把它当 PDF 送去抽取。两种误判方向相反，只有魔数同时挡住。
    """
    box = sandbox.box
    _seed_oa_work(box)
    _fake_http(
        monkeypatch,
        _FakeResponse(200, b"<html><body>Sign in to access this article</body></html>"),
    )
    box["bundle"] = _bundle(None)

    research.cmd_read(_args(_DOI))

    assert box["extract_calls"] == []
    assert box["fetch_calls"] == [f"https://doi.org/{_DOI}"]


def test_a_url_target_tries_a_plain_get_before_the_browser(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户直接粘的 URL 也走第一档：粘的是 PDF 直链时省掉整个浏览器。

    粘的是文章落地页时这一档必然失败（返回 HTML），代价是一次普通 GET，然后照常
    回落浏览器——行为与改动前一致，只是多花一个请求。
    """
    box = sandbox.box
    url = "https://example.org/some/paper.pdf"
    _fake_http(monkeypatch, _FakeResponse(200, _PDF_BYTES))

    assert research.cmd_read(_args(url)) == 0

    assert box["fetch_calls"] == []
    assert len(box["extract_calls"]) == 1


def test_an_openalex_id_without_any_url_still_exits_2(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """既无 ``oa_url`` 又无 ``doi`` 时：明确报错退出，且不发任何请求、不起浏览器。

    保留原来的前置条件检查（退出码 2），别让它被新增的第一档悄悄绕过。
    """
    box = sandbox.box
    box["work"] = {"id": "W1"}
    box["fm"] = {"title": "T", "short_title": "t"}
    log = _fake_http(monkeypatch, _FakeResponse(200, _PDF_BYTES))

    assert research.cmd_read(_args("W2748115035")) == 2

    assert log == [] and box["fetch_calls"] == []


def test_the_download_overrides_the_json_accept_header(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``http_session`` 的默认头是 ``Accept: application/json``（给元数据 API 用的）。

    拿它去要 PDF 会被部分出版商的内容协商直接拒掉，且症状是「403/406 而 OA 明明可用」
    ——极难联想到是请求头的问题，故把它钉死。
    """
    box = sandbox.box
    _seed_oa_work(box)
    log = _fake_http(monkeypatch, _FakeResponse(200, _PDF_BYTES))

    research.cmd_read(_args(_DOI))

    assert "pdf" in log[0]["headers"]["Accept"]
    assert log[0]["stream"] is True


# ===========================================================================
#  _try_oa_pdf 自身的降级与缓存语义
# ===========================================================================
def test_try_oa_pdf_reuses_the_cache_without_making_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "x.pdf"
    dest.write_bytes(_PDF_BYTES)
    log = _fake_http(monkeypatch, _FakeResponse(200, _PDF_BYTES))

    assert research._try_oa_pdf("https://e.org/x", dest) == dest
    assert log == []


def test_try_oa_pdf_refetches_under_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--refresh`` 的语义是「绕过缓存重抓」，对这一档同样成立。"""
    dest = tmp_path / "x.pdf"
    dest.write_bytes(b"%PDF-1.4 stale")
    log = _fake_http(monkeypatch, _FakeResponse(200, _PDF_BYTES))

    assert research._try_oa_pdf("https://e.org/x", dest, force=True) == dest
    assert len(log) == 1
    assert dest.read_bytes() == _PDF_BYTES


def test_try_oa_pdf_leaves_no_part_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写盘途中炸掉时不得留下半截文件。

    先写 ``.part`` 再原子改名正是为此：直接写 ``dest`` 的话，一次中断会留下一个截断的
    ``.pdf``，而下次进来会被当成「已在缓存」直接复用——那比下载失败难查得多。
    """
    dest = tmp_path / "x.pdf"

    class _Boom(_FakeResponse):
        def iter_content(self, chunk_size: int = 1) -> Any:
            yield b"%PDF-"
            raise OSError(28, "No space left on device")

    _fake_http(monkeypatch, _Boom(200, b""))

    assert research._try_oa_pdf("https://e.org/x", dest) is None
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_try_oa_pdf_reports_a_network_error_and_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """网络不可达 → ``None`` + 一行 stderr，绝不抛到 ``cmd_read`` 主干（不变量 1）。"""
    _fake_http(monkeypatch, ConnectionError("DNS 解析失败"))

    assert research._try_oa_pdf("https://e.org/x", tmp_path / "x.pdf") is None
    assert "OA 直链下载失败" in capsys.readouterr().err


def test_oa_pdf_filename_prefers_identifiers_over_the_url() -> None:
    """文件名由 DOI / OpenAlex id 派生，不由 URL 派生。

    同一篇论文换个镜像（机构库 vs 出版商 vs PMC）时仍复用同一个文件；按 URL（或其
    哈希）命名会把「同一篇」拆成「多个不同文件」，``prune --keep-referenced`` 也就
    认不出笔记引用的那一份。
    """
    a = research._oa_pdf_filename({"doi": _DOI}, "https://mirror-a.org/p/1")
    b = research._oa_pdf_filename({"doi": _DOI}, "https://mirror-b.org/p/2")
    assert a == b and a.endswith(".pdf") and "10-1103" in a

    # 无 DOI 时退到 OpenAlex id；两者都没有才用 URL（总比取不到名字好）
    assert research._oa_pdf_filename({"openalex_id": "W1"}, "u") == "w1.pdf"
    assert research._oa_pdf_filename(None, "https://e.org/A_B").endswith(".pdf")


# ===========================================================================
#  H5-2 网页正文兜底：记 extracted_html_path，不进 cache/extracted/
# ===========================================================================
def _seed_html_fallback(sandbox: SimpleNamespace) -> Path:
    """造一个「浏览器只拿到网页正文、没有 PDF」的现场。"""
    box = sandbox.box
    box["work"] = {"id": "W1", "doi": _DOI}
    box["fm"] = {"title": "T", "short_title": "t", "doi": _DOI}
    html = sandbox.root / "cache" / "html_fulltext" / "slug" / "slug.md"
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text("---\nsource_url: u\n---\n\n# T\n\n网页正文\n", encoding="utf-8")
    box["bundle"] = _bundle(None, html)
    return html


@pytest.fixture
def no_direct_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """把第一档短路。

    本组的夹具本来就不给 ``work`` 配 ``oa_url``，所以 ``cmd_read`` 压根不会试直链；
    这个夹具是道保险：将来若把 ``https://doi.org/...`` 也纳入第一档，本组用例不会
    默默变成联外网的测试。
    """
    monkeypatch.setattr(research, "_try_oa_pdf", lambda *a, **k: None)


def test_the_html_fallback_sets_extracted_html_path_and_not_extracted_md_path(
    sandbox: SimpleNamespace, no_direct_link: None
) -> None:
    """核心契约：兜底路径记进 ``extracted_html_path``，**不设** ``extracted_md_path``。

    两个字段的差别不是命名洁癖：``rag`` 只索引 ``cache/extracted/**/*.md``，所以
    「设哪个」直接决定这段网页正文进不进 RAG 语料。而它的公式通常已丢失（出版商把
    公式渲染成图片/SVG），进了语料就会让 ``rag ask`` 拿一段缺公式的正文当权威来源。
    """
    html = _seed_html_fallback(sandbox)
    box = sandbox.box

    assert research.cmd_read(_args(_DOI, note=True)) == 0

    fm = box["merged_fm"][0]
    assert fm["extracted_html_path"] == str(html)
    assert "extracted_md_path" not in fm


def test_the_html_fallback_is_not_copied_into_cache_extracted(
    sandbox: SimpleNamespace, no_direct_link: None
) -> None:
    """不复制：那份文件 browser_fetch 已经写好了，再抄一遍就是同一份内容两个副本。

    ``cache/extracted/`` 是 git 跟踪的刻意例外（MinerU 配额换来的全文要备份），把网页
    兜底混进去还会稀释那个例外的理由。
    """
    _seed_html_fallback(sandbox)

    research.cmd_read(_args(_DOI))

    extracted = sandbox.root / "cache" / "extracted"
    assert not extracted.exists() or list(extracted.iterdir()) == []


def test_the_html_fallback_says_what_it_is_and_what_is_lost(
    sandbox: SimpleNamespace,
    no_direct_link: None,
    capsys: pytest.CaptureFixture,
) -> None:
    """报告必须点破「这不是 PDF 抽取」以及公式可能已丢失。

    不说出口的后果是静默的质量下降：用户以为拿到的是抽取全文，据此写进综述的公式
    其实来自一个把公式渲染成图片的网页。
    """
    html = _seed_html_fallback(sandbox)

    research.cmd_read(_args(_DOI))

    cap = capsys.readouterr()
    assert str(html) in cap.out and "全文 HTML" in cap.out
    assert "公式" in cap.err and "trafilatura" in cap.err
    assert "rag 语料" in cap.err


def test_a_failed_extraction_no_longer_reports_a_nonexistent_file(
    sandbox: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """顺带修掉的既有谎话：抽取失败时不得再报一行指向**不存在**文件的「全文 MD」。

    ``fulltext_path`` 在第 1.5 步就按标题算出来了，与「是否真拿到全文」无关；旧报告只判
    ``if fulltext_path``，于是会打出「全文 MD : <路径>（0 字符）——精读请 Read 此文件」。
    调用方照着 Read 会得到一个文件不存在的错误，而它本以为全文已经在手。
    """
    box = sandbox.box
    box["work"] = {"id": "W1"}
    box["fm"] = {"title": "T", "short_title": "t", "doi": _DOI}
    monkeypatch.setattr(research, "_try_oa_pdf", lambda *a, **k: None)
    box["bundle"] = _bundle(None, None)  # 浏览器也没拿到任何东西

    assert research.cmd_read(_args(_DOI)) == 1

    cap = capsys.readouterr()
    assert "全文 MD" not in cap.out
    assert "未获得全文" in cap.err


def test_the_html_fallback_still_counts_as_success(
    sandbox: SimpleNamespace, no_direct_link: None
) -> None:
    """退出码 0：确实拿到了一份可读全文，只是来源较差。

    报 1 会让脚本化调用方把「有网页正文」与「什么都没拿到」当成同一类失败处理。
    """
    _seed_html_fallback(sandbox)

    assert research.cmd_read(_args(_DOI)) == 0


def test_a_real_extraction_still_sets_extracted_md_path(
    sandbox: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对照面：拿到 PDF 并抽出正文时，走的仍是原来的 ``extracted_md_path``。

    没有这条，上面两条会与「``extracted_md_path`` 被彻底废掉」的实现同样通过。
    """
    box = sandbox.box
    box["work"] = {"id": "W1"}
    box["fm"] = {"title": "T", "short_title": "t", "doi": _DOI}
    monkeypatch.setattr(research, "_try_oa_pdf", lambda *a, **k: None)
    pdf = sandbox.root / "cache" / "pdfs" / "p.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(_PDF_BYTES)
    box["bundle"] = _bundle(pdf)

    assert research.cmd_read(_args(_DOI, note=True)) == 0

    fm = box["merged_fm"][0]
    assert fm["extracted_md_path"].endswith("_fulltext.md")
    assert "extracted_html_path" not in fm
    assert box["extract_calls"] == [pdf]
