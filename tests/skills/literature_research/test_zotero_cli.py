"""zotero_cli 规范化契约离线快照测试（不打真实 zotero-cli / 不触网）。

锁定阶段 3「Zotero 层 → 委托社区 zotero-mcp 的 ``zotero-cli --json``」替换后的
差异化契约，全部通过 monkeypatch 拦截 ``subprocess.run`` / ``_run_json`` /
``shutil.which`` / ``settings``，绝不启动真实子进程：

- **frontmatter → BibTeX 规范化**（本项目保留的差异化层）：期刊走 ``@article``、
  预印本走 ``@misc``+eprint；自定义增强字段折进 ``note``（多行 ``key: value``，
  顺序即 ``_EXTRA_FIELDS``）；tags → ``keywords``；citekey 与
  ``document_writing.refs_bridge.make_citekey`` **逐字一致**（防 .bib 刷新后 \\cite 漂移）。
- **稳定信封解析**：``_run_json`` 校验 ``ok`` 后返回 ``data``；未安装 / ok:false /
  非 JSON / 空输出 / 超时 / 结构异常各抛对应异常；``_items_from_data`` /
  ``_text_from_data`` / ``_extract_key_from_add`` 对多种 data 形状尽力解析。
- **委托方法**：ping / list_items / get_item / search_items / get_bibtex /
  get_fulltext / library_info / create_item_from_metadata / add_doi / add_note
  组装正确的 ``zotero-cli`` 子命令参数并透传形状。
- **凭据翻译**：``_cli_env`` 把本项目 ``ZOTERO_USER_ID`` → ``ZOTERO_LIBRARY_ID``，
  ``ZOTERO_API_KEY`` 直传，且已存在的 shell 环境变量优先。
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import types

import pytest

from pysci.skills.literature_research.tools import notes, zotero_cli

# ---------------------------------------------------------------------------
#  fixtures：贴近 openalex/arxiv 转换器产出的 paper-note frontmatter
# ---------------------------------------------------------------------------
FM_JOURNAL = {
    "title": "Simultaneous observation of topological exceptional points",
    "authors": ["Y. Zhu", "X. Li"],
    "first_author_last_name": "Zhu",
    "year": 2018,
    "journal": "Physical Review Letters",
    "volume": "121",
    "issue": "12",
    "pages": "124501",
    "publisher": "APS",
    "doi": "10.1103/PhysRevLett.121.124501",
    "oa_url": "https://doi.org/10.1103/PhysRevLett.121.124501",
    # 自定义增强字段 → 应折进 BibTeX note
    "openalex_id": "W123",
    "wos_id": "WOS:0001",
    "jif": 8.6,
    "jcr_quartile": "Q1",
    "cited_by_count": 42,
    "oa_status": "green",
}

FM_PREPRINT = {
    "title": "Non-Hermitian metagratings with degenerate states",
    "authors": ["F. Fang"],
    "first_author_last_name": "Fang",
    "year": 2023,
    "arxiv_id": "2301.09876",
    "oa_url": "https://arxiv.org/abs/2301.09876",
}

# Zotero API 原形条目（pyzotero 透传形状）——用于跨 skill citekey 一致性核验
ZOTERO_API_ITEM_DATA = {
    "itemType": "journalArticle",
    "title": "Simultaneous observation of topological exceptional points",
    "creators": [
        {"creatorType": "author", "firstName": "Y.", "lastName": "Zhu"},
        {"creatorType": "author", "firstName": "X.", "lastName": "Li"},
    ],
    "date": "2018-06-15",
    "publicationTitle": "Physical Review Letters",
}


def _fake_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    """构造一个 subprocess.run 的替身返回值。"""
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _envelope(ok: bool, data=None, error=None, command: str = "x") -> str:
    body: dict = {"ok": ok, "command": command, "schema": 1}
    if ok:
        body["data"] = data
    else:
        body["error"] = error or {"message": "boom", "code": "E"}
    return json.dumps(body)


# ===========================================================================
#  available() / zotero_cli_bin()
# ===========================================================================
def test_available_true(monkeypatch):
    monkeypatch.setattr(zotero_cli.shutil, "which", lambda name: "/usr/bin/zotero-cli")
    assert zotero_cli.zotero_cli_bin() == "/usr/bin/zotero-cli"
    assert zotero_cli.available() is True


def test_available_false(monkeypatch):
    monkeypatch.setattr(zotero_cli.shutil, "which", lambda name: None)
    assert zotero_cli.zotero_cli_bin() is None
    assert zotero_cli.available() is False


# ===========================================================================
#  _cli_env()：凭据翻译（ZOTERO_USER_ID → ZOTERO_LIBRARY_ID）
# ===========================================================================
def test_cli_env_translates_credentials(monkeypatch):
    monkeypatch.setattr(
        zotero_cli,
        "settings",
        dataclasses.replace(
            zotero_cli.settings, zotero_api_key="K", zotero_user_id="12345"
        ),
    )
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    env = zotero_cli._cli_env()
    assert env["ZOTERO_API_KEY"] == "K"
    assert env["ZOTERO_LIBRARY_ID"] == "12345"  # 由 zotero_user_id 翻译而来
    assert env["ZOTERO_LIBRARY_TYPE"] == "user"


def test_cli_env_existing_shell_wins(monkeypatch):
    monkeypatch.setattr(
        zotero_cli,
        "settings",
        dataclasses.replace(
            zotero_cli.settings, zotero_api_key="K", zotero_user_id="12345"
        ),
    )
    monkeypatch.setenv("ZOTERO_API_KEY", "FROM_SHELL")
    env = zotero_cli._cli_env()
    assert env["ZOTERO_API_KEY"] == "FROM_SHELL"  # 已存在的显式设置优先


def test_cli_env_no_credentials_is_passthrough(monkeypatch):
    monkeypatch.setattr(
        zotero_cli,
        "settings",
        dataclasses.replace(
            zotero_cli.settings, zotero_api_key=None, zotero_user_id=None
        ),
    )
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    env = zotero_cli._cli_env()
    assert "ZOTERO_LIBRARY_ID" not in env


# ===========================================================================
#  BibTeX 辅助纯函数
# ===========================================================================
def test_bib_escape():
    assert zotero_cli._bib_escape("Gain & Loss") == r"Gain \& Loss"
    assert zotero_cli._bib_escape("50% off") == r"50\% off"
    assert zotero_cli._bib_escape("Issue #3") == r"Issue \#3"
    assert zotero_cli._bib_escape("plain") == "plain"


def test_bib_author_name():
    assert zotero_cli._bib_author_name("Y. Zhu") == "Zhu, Y."
    assert zotero_cli._bib_author_name("Zhu, Y.") == "Zhu, Y."  # 已含逗号 → 原样
    assert zotero_cli._bib_author_name("Aristotle") == "Aristotle"  # 单词名
    assert (
        zotero_cli._bib_author_name("Ludwig van Beethoven") == "Beethoven, Ludwig van"
    )
    assert zotero_cli._bib_author_name("") == ""
    assert zotero_cli._bib_author_name(None) == ""


def test_bib_authors_string_and_dict_forms():
    assert zotero_cli._bib_authors(["Y. Zhu", "X. Li"]) == "Zhu, Y. and Li, X."
    assert zotero_cli._bib_authors([{"name": "Y. Zhu"}]) == "Zhu, Y."
    assert (
        zotero_cli._bib_authors([{"firstName": "Y.", "lastName": "Zhu"}]) == "Zhu, Y."
    )
    assert zotero_cli._bib_authors([]) == ""
    assert zotero_cli._bib_authors(None) == ""


def test_bib_year():
    assert zotero_cli._bib_year({"year": 2018}) == "2018"
    assert zotero_cli._bib_year({"publication_date": "2018-06-15"}) == "2018"
    assert zotero_cli._bib_year({"year": "", "publication_date": "2020-01"}) == "2020"
    assert zotero_cli._bib_year({}) == ""


def test_is_preprint():
    assert zotero_cli._is_preprint({"journal": "arXiv preprint"}) is True
    assert zotero_cli._is_preprint({"publisher": "arXiv"}) is True
    assert zotero_cli._is_preprint({"arxiv_id": "2301.09876", "journal": ""}) is True
    assert zotero_cli._is_preprint({"journal": "Physical Review Letters"}) is False
    # 有 arxiv_id 但已见刊（journal 非空）→ 视作期刊文章
    assert (
        zotero_cli._is_preprint({"arxiv_id": "2301.09876", "journal": "Nature"})
        is False
    )


# ===========================================================================
#  make_citekey
# ===========================================================================
def test_make_citekey_journal():
    assert zotero_cli.make_citekey(FM_JOURNAL) == "zhu2018simultaneous"


def test_make_citekey_skips_stopwords():
    fm = {
        "first_author_last_name": "Doe",
        "year": 2020,
        "title": "The role of gain in acoustic systems",
    }
    assert zotero_cli.make_citekey(fm) == "doe2020role"  # "the" 是停用词 → 取 "role"


def test_make_citekey_falls_back_to_authors():
    fm = {"authors": ["X. Li"], "year": 2021, "title": "Metagratings"}
    assert zotero_cli.make_citekey(fm) == "li2021metagratings"


def test_make_citekey_all_fallbacks():
    assert zotero_cli.make_citekey({}) == "anonnduntitled"


def test_citekey_matches_refs_bridge_contract():
    """跨 skill 护栏：调研入库与写作导出的 citekey 必须逐字一致，避免 \\cite 漂移。"""
    from pysci.skills.document_writing.tools import refs_bridge

    assert zotero_cli.make_citekey(FM_JOURNAL) == refs_bridge.make_citekey(
        ZOTERO_API_ITEM_DATA
    )


# ===========================================================================
#  frontmatter_to_bibtex —— 规范化契约
# ===========================================================================
def test_frontmatter_to_bibtex_journal_fields():
    bib = zotero_cli.frontmatter_to_bibtex(FM_JOURNAL)
    assert bib.startswith("@article{zhu2018simultaneous,")
    assert "author = {Zhu, Y. and Li, X.}" in bib
    assert "title = {Simultaneous observation of topological exceptional points}" in bib
    assert "year = {2018}" in bib
    assert "journal = {Physical Review Letters}" in bib
    assert "volume = {121}" in bib
    assert "number = {12}" in bib  # issue → BibTeX number
    assert "pages = {124501}" in bib
    assert "publisher = {APS}" in bib
    assert "doi = {10.1103/PhysRevLett.121.124501}" in bib
    assert "url = {https://doi.org/10.1103/PhysRevLett.121.124501}" in bib
    assert bib.rstrip().endswith("}")


def test_frontmatter_extras_folded_into_note_in_order():
    bib = zotero_cli.frontmatter_to_bibtex(FM_JOURNAL)
    note = (
        "openalex_id: W123\n"
        "wos_id: WOS:0001\n"
        "jif: 8.6\n"
        "jcr_quartile: Q1\n"
        "cited_by_count: 42\n"
        "oa_status: green"
    )
    assert f"note = {{{note}}}" in bib
    # FM_JOURNAL 没有 scimago_quartile → 该行必须整个缺席，而不是写成空值
    assert "scimago_quartile" not in bib


def test_frontmatter_folds_scimago_quartile_next_to_jcr():
    """两套分区必须成对抵达 Zotero Extra。

    回归：``scimago_quartile`` 曾不在 ``_EXTRA_FIELDS`` 里（而它的 WoS 对应物
    ``jcr_quartile`` 在），后果是 SCImago 分区永远到不了 Zotero——而它对物理声学
    领域恰恰是当前唯一可用的分区源（WoS Journals API 尚在申请中）。
    """
    fm = dict(FM_JOURNAL)
    fm["scimago_quartile"] = "Q1"
    bib = zotero_cli.frontmatter_to_bibtex(fm)
    note = (
        "openalex_id: W123\n"
        "wos_id: WOS:0001\n"
        "jif: 8.6\n"
        "jcr_quartile: Q1\n"
        "scimago_quartile: Q1\n"
        "cited_by_count: 42\n"
        "oa_status: green"
    )
    assert f"note = {{{note}}}" in bib


def test_extra_fields_are_known_frontmatter_keys():
    """``_EXTRA_FIELDS`` 必须是 ``notes.FIELD_ORDER`` 的子集。

    这些键只从 frontmatter 取值，写错一个字母不会报错，只会静默变成「永远取不到」。
    本断言把那个静默失败变成一条红测试。
    """
    unknown = [k for k in zotero_cli._EXTRA_FIELDS if k not in notes.FIELD_ORDER]
    assert unknown == []


def test_extra_fields_pair_both_quartile_sources():
    """两个分区字段同在或同不在；只收录其一是无意义的不对称。"""
    assert {"jcr_quartile", "scimago_quartile"} <= set(zotero_cli._EXTRA_FIELDS)


def test_frontmatter_tags_become_keywords():
    bib = zotero_cli.frontmatter_to_bibtex(FM_JOURNAL, tags=["ep", "acoustic", ""])
    assert "keywords = {ep, acoustic}" in bib  # 空 tag 被剔除


def test_frontmatter_no_tags_no_keywords():
    bib = zotero_cli.frontmatter_to_bibtex(FM_JOURNAL)
    assert "keywords = {" not in bib


def test_frontmatter_skips_zero_and_empty_extras():
    fm = {
        "title": "T",
        "first_author_last_name": "Doe",
        "year": 2020,
        "cited_by_count": 0,  # 0 视作缺失
        "openalex_id": "",  # 空串视作缺失
        "jif": 0.0,  # 0.0 == 0 → 视作缺失
    }
    bib = zotero_cli.frontmatter_to_bibtex(fm)
    assert "note = {" not in bib


def test_frontmatter_to_bibtex_preprint():
    bib = zotero_cli.frontmatter_to_bibtex(FM_PREPRINT)
    assert bib.startswith("@misc{fang2023non,")
    assert "author = {Fang, F.}" in bib
    assert "eprint = {2301.09876}" in bib
    assert "archivePrefix = {arXiv}" in bib
    assert "year = {2023}" in bib
    assert "journal = {" not in bib  # 预印本无 journal 字段


def test_frontmatter_preprint_strips_arxiv_prefix_and_sets_howpublished():
    fm = {
        "title": "X",
        "first_author_last_name": "Doe",
        "year": 2022,
        "arxiv_id": "arXiv:2201.00001",
        "journal": "arXiv",
    }
    bib = zotero_cli.frontmatter_to_bibtex(fm)
    assert bib.startswith("@misc{")
    assert "eprint = {2201.00001}" in bib  # 前缀已剥离
    assert "howpublished = {arXiv}" in bib  # journal 存在 → howpublished


def test_frontmatter_custom_citekey():
    bib = zotero_cli.frontmatter_to_bibtex(FM_JOURNAL, citekey="mykey2024")
    assert bib.startswith("@article{mykey2024,")


def test_frontmatter_escapes_title():
    fm = {
        "title": "Gain & loss in acoustic #1 systems",
        "first_author_last_name": "Doe",
        "year": 2020,
    }
    bib = zotero_cli.frontmatter_to_bibtex(fm)
    assert r"title = {Gain \& loss in acoustic \#1 systems}" in bib


def test_frontmatter_abstract_truncated():
    fm = {
        "title": "T",
        "first_author_last_name": "Doe",
        "year": 2020,
        "abstract": "x" * 2000,
    }
    bib = zotero_cli.frontmatter_to_bibtex(fm)
    assert "abstract = {" in bib
    assert "..." in bib
    assert ("x" * 1300) not in bib  # 截断到 1200 + "..."


# ===========================================================================
#  信封 data 解析辅助
# ===========================================================================
def test_items_from_data_variants():
    assert zotero_cli._items_from_data([{"key": "A"}, {"key": "B"}]) == [
        {"key": "A"},
        {"key": "B"},
    ]
    assert zotero_cli._items_from_data({"items": [{"key": "A"}]}) == [{"key": "A"}]
    assert zotero_cli._items_from_data({"results": [{"key": "A"}]}) == [{"key": "A"}]
    assert zotero_cli._items_from_data({"entries": [{"key": "A"}]}) == [{"key": "A"}]
    assert zotero_cli._items_from_data({"data": [{"key": "A"}]}) == [{"key": "A"}]
    assert zotero_cli._items_from_data([{"key": "A"}, "junk", 5]) == [{"key": "A"}]
    assert zotero_cli._items_from_data({"foo": 1}) == []
    assert zotero_cli._items_from_data(None) == []


def test_text_from_data_variants():
    assert zotero_cli._text_from_data("raw") == "raw"
    assert zotero_cli._text_from_data({"bibtex": "@article{x}"}) == "@article{x}"
    assert zotero_cli._text_from_data({"text": "T"}) == "T"
    assert zotero_cli._text_from_data({"fulltext": "Body"}) == "Body"
    assert zotero_cli._text_from_data({"text": ""}) == ""  # 空串跳过
    assert zotero_cli._text_from_data({}) == ""
    assert zotero_cli._text_from_data(None) == ""


def test_extract_key_from_add_variants():
    assert zotero_cli._extract_key_from_add({"key": "ABCD1234"}) == "ABCD1234"
    assert zotero_cli._extract_key_from_add({"item_key": "ABCD1234"}) == "ABCD1234"
    assert zotero_cli._extract_key_from_add({"itemKey": "ABCD1234"}) == "ABCD1234"
    assert zotero_cli._extract_key_from_add({"item": {"key": "ABCD1234"}}) == "ABCD1234"
    assert (
        zotero_cli._extract_key_from_add({"item": {"data": {"key": "ABCD1234"}}})
        == "ABCD1234"
    )
    assert (
        zotero_cli._extract_key_from_add({"items": [{"key": "ABCD1234"}]}) == "ABCD1234"
    )
    assert zotero_cli._extract_key_from_add({"items": ["ABCD1234"]}) == "ABCD1234"
    # 兜底：从自由文本里抓 8 位 key
    assert (
        zotero_cli._extract_key_from_add({"text": "Created item ABCD1234 ok"})
        == "ABCD1234"
    )
    assert zotero_cli._extract_key_from_add({}) == ""
    assert zotero_cli._extract_key_from_add(None) == ""


# ===========================================================================
#  _run_json —— 稳定信封校验（monkeypatch subprocess.run）
# ===========================================================================
def _patch_bin(monkeypatch, path="zotero-cli"):
    monkeypatch.setattr(zotero_cli, "zotero_cli_bin", lambda: path)


def test_run_json_success_returns_data(monkeypatch):
    _patch_bin(monkeypatch)
    monkeypatch.setattr(
        zotero_cli.subprocess,
        "run",
        lambda *a, **k: _fake_proc(stdout=_envelope(True, data={"x": 1})),
    )
    assert zotero_cli._run_json(["config"]) == {"x": 1}


def test_run_json_not_installed(monkeypatch):
    _patch_bin(monkeypatch, None)
    with pytest.raises(zotero_cli.ZoteroNotConfigured):
        zotero_cli._run_json(["config"])


def test_run_json_error_envelope(monkeypatch):
    _patch_bin(monkeypatch)
    monkeypatch.setattr(
        zotero_cli.subprocess,
        "run",
        lambda *a, **k: _fake_proc(
            stdout=_envelope(False, error={"message": "boom", "code": "E42"})
        ),
    )
    with pytest.raises(zotero_cli.ZoteroCliError) as ei:
        zotero_cli._run_json(["search", "x"])
    assert "boom" in str(ei.value)
    assert "E42" in str(ei.value)


def test_run_json_empty_stdout(monkeypatch):
    _patch_bin(monkeypatch)
    monkeypatch.setattr(
        zotero_cli.subprocess,
        "run",
        lambda *a, **k: _fake_proc(stdout="", stderr="diag", returncode=1),
    )
    with pytest.raises(zotero_cli.ZoteroCliError):
        zotero_cli._run_json(["config"])


def test_run_json_non_json_stdout(monkeypatch):
    _patch_bin(monkeypatch)
    monkeypatch.setattr(
        zotero_cli.subprocess,
        "run",
        lambda *a, **k: _fake_proc(stdout="not json at all"),
    )
    with pytest.raises(zotero_cli.ZoteroCliError):
        zotero_cli._run_json(["config"])


def test_run_json_non_dict_envelope(monkeypatch):
    _patch_bin(monkeypatch)
    monkeypatch.setattr(
        zotero_cli.subprocess,
        "run",
        lambda *a, **k: _fake_proc(stdout="[1, 2, 3]"),
    )
    with pytest.raises(zotero_cli.ZoteroCliError):
        zotero_cli._run_json(["config"])


def test_run_json_timeout(monkeypatch):
    _patch_bin(monkeypatch)

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="zotero-cli", timeout=1)

    monkeypatch.setattr(zotero_cli.subprocess, "run", _raise)
    with pytest.raises(zotero_cli.ZoteroCliError):
        zotero_cli._run_json(["config"])


def test_run_json_filenotfound_maps_to_notconfigured(monkeypatch):
    _patch_bin(monkeypatch)

    def _raise(*a, **k):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(zotero_cli.subprocess, "run", _raise)
    with pytest.raises(zotero_cli.ZoteroNotConfigured):
        zotero_cli._run_json(["config"])


# ===========================================================================
#  ZoteroCli.ping
# ===========================================================================
def test_ping_unavailable(monkeypatch):
    monkeypatch.setattr(zotero_cli, "available", lambda: False)
    zb = zotero_cli.ZoteroCli()
    assert zb.backend == "zotero-cli(unavailable)"
    res = zb.ping()
    assert res["ok"] is False
    assert "zotero-cli" in res["error"]


def test_ping_available(monkeypatch):
    monkeypatch.setattr(zotero_cli, "available", lambda: True)
    monkeypatch.setattr(
        zotero_cli, "_run_json", lambda args, *, timeout=None: {"local": True}
    )
    zb = zotero_cli.ZoteroCli()
    assert zb.backend == "zotero-cli"
    res = zb.ping()
    assert res["ok"] is True
    assert res["config"] == {"local": True}


def test_ping_swallows_cli_error(monkeypatch):
    monkeypatch.setattr(zotero_cli, "available", lambda: True)

    def _boom(args, *, timeout=None):
        raise zotero_cli.ZoteroCliError("config failed")

    monkeypatch.setattr(zotero_cli, "_run_json", _boom)
    res = zotero_cli.ZoteroCli().ping()
    assert res["ok"] is False
    assert "config failed" in res["error"]


# ===========================================================================
#  ZoteroCli 读操作（捕获组装的子命令参数）
# ===========================================================================
def _capture_run_json(monkeypatch, canned):
    captured: dict = {}

    def _fake(args, *, timeout=None):
        captured["args"] = args
        return canned

    monkeypatch.setattr(zotero_cli, "_run_json", _fake)
    return captured


def test_list_items_default_search(monkeypatch):
    captured = _capture_run_json(monkeypatch, {"items": [{"key": "A"}]})
    out = zotero_cli.ZoteroCli().list_items(limit=10)
    assert captured["args"] == ["search", "", "--limit", "10"]
    assert out == [{"key": "A"}]


def test_list_items_collection(monkeypatch):
    captured = _capture_run_json(monkeypatch, [])
    zotero_cli.ZoteroCli().list_items(collection="COLL")
    assert captured["args"][:3] == ["get", "collection-items", "COLL"]


def test_list_items_tag(monkeypatch):
    captured = _capture_run_json(monkeypatch, [])
    zotero_cli.ZoteroCli().list_items(tag="ep")
    assert captured["args"][:4] == ["search", "--mode", "tag", "ep"]


def test_list_items_clamps_limit(monkeypatch):
    captured = _capture_run_json(monkeypatch, [])
    zotero_cli.ZoteroCli().list_items(limit=999)
    assert captured["args"][-1] == "100"  # 上限 100


def test_get_item_returns_data(monkeypatch):
    monkeypatch.setattr(
        zotero_cli,
        "_run_json",
        lambda args, *, timeout=None: {"key": "A", "data": {"title": "T"}},
    )
    assert zotero_cli.ZoteroCli().get_item("A") == {"key": "A", "data": {"title": "T"}}


def test_get_item_raises_instead_of_printing_to_stdout(monkeypatch, capsys):
    """失败必须抛，且**绝不**往 stdout 写东西。

    ``library get`` 的 stdout 就是 JSON 数据通道；此前这里独自吞异常并把中文错误打到
    stdout，调用方拿到的既不是合法 JSON 也不是可判读的诊断。抛上去之后由
    ``cmd_library`` 用 ``ping`` 区分「桥断了」与「库里没这条」（见 test_cli_consistency）。
    """

    def _boom(args, *, timeout=None):
        raise zotero_cli.ZoteroCliError("nope")

    monkeypatch.setattr(zotero_cli, "_run_json", _boom)

    with pytest.raises(zotero_cli.ZoteroCliError, match="nope"):
        zotero_cli.ZoteroCli().get_item("A")

    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == "", "诊断属于调用方，不在这一层自己 print"


def test_get_item_returns_none_for_a_non_dict_payload(monkeypatch):
    """``ok:true`` 但 data 不是 dict：查询成功了，答案就是「没有」。

    这是 ``cmd_library`` 唯一能说「（未找到）」并返 0 的分支，与上面那条抛异常的
    失败路径必须分得开。
    """
    monkeypatch.setattr(zotero_cli, "_run_json", lambda args, *, timeout=None: [])

    assert zotero_cli.ZoteroCli().get_item("A") is None


def test_search_items_default_mode(monkeypatch):
    captured = _capture_run_json(monkeypatch, {"items": [{"key": "A"}]})
    out = zotero_cli.ZoteroCli().search_items("ep acoustic", limit=5)
    assert captured["args"] == ["search", "ep acoustic", "--limit", "5"]
    assert out == [{"key": "A"}]


def test_search_items_explicit_mode(monkeypatch):
    captured = _capture_run_json(monkeypatch, [])
    zotero_cli.ZoteroCli().search_items("q", mode="semantic")
    assert captured["args"][:3] == ["search", "--mode", "semantic"]


def test_get_bibtex(monkeypatch):
    monkeypatch.setattr(
        zotero_cli, "_run_json", lambda args, *, timeout=None: {"bibtex": "@article{x}"}
    )
    assert zotero_cli.ZoteroCli().get_bibtex("K") == "@article{x}"


def test_get_fulltext(monkeypatch):
    monkeypatch.setattr(
        zotero_cli, "_run_json", lambda args, *, timeout=None: {"text": "body"}
    )
    assert zotero_cli.ZoteroCli().get_fulltext("K") == "body"


def test_library_info(monkeypatch):
    monkeypatch.setattr(
        zotero_cli, "_run_json", lambda args, *, timeout=None: {"count": 10}
    )
    assert zotero_cli.ZoteroCli().library_info() == {"count": 10}


# ===========================================================================
#  ZoteroCli 写操作
# ===========================================================================
def test_create_item_from_metadata_builds_bibtex(monkeypatch):
    captured = _capture_run_json(monkeypatch, {"key": "NEWKEY12"})
    out = zotero_cli.ZoteroCli().create_item_from_metadata(
        FM_JOURNAL, tags=["ep", "acoustic"]
    )
    assert captured["args"][:3] == ["add", "bibtex", "--bibtex"]
    bib = captured["args"][3]
    assert bib.startswith("@article{zhu2018simultaneous,")
    assert "keywords = {ep, acoustic}" in bib
    assert out == {"key": "NEWKEY12", "raw": {"key": "NEWKEY12"}}


def test_create_item_collections_skip_blank(monkeypatch):
    captured = _capture_run_json(monkeypatch, {"key": "K1"})
    zotero_cli.ZoteroCli().create_item_from_metadata(
        FM_JOURNAL, collections=["MyColl", "  "]
    )
    assert captured["args"][-2:] == ["-c", "MyColl"]  # 空白 collection 被剔除


def test_add_doi(monkeypatch):
    captured = _capture_run_json(monkeypatch, {"key": "NEW1"})
    out = zotero_cli.ZoteroCli().add_doi("10.1/x", collections=["C"])
    assert captured["args"] == ["add", "doi", "10.1/x", "-c", "C"]
    assert out == {"key": "NEW1", "raw": {"key": "NEW1"}}


def test_add_note(monkeypatch):
    captured = _capture_run_json(monkeypatch, {"key": "NOTE1"})
    out = zotero_cli.ZoteroCli().add_note("PARENT12", "hello", tags=["t1"])
    assert captured["args"] == [
        "notes",
        "create",
        "--item-key",
        "PARENT12",
        "--text",
        "hello",
        "--tags",
        "t1",
    ]
    assert out == {"key": "NOTE1"}
