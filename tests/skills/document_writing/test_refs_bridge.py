"""refs_bridge 测试：BibTeX 转换器（离线，用 mock Zotero 条目 JSON）。"""

from __future__ import annotations

from pysci.skills.document_writing.tools import refs_bridge

JOURNAL_ITEM = {
    "key": "ABCD1234",
    "data": {
        "itemType": "journalArticle",
        "title": "Simultaneous observation of topological exceptional points",
        "creators": [
            {"creatorType": "author", "firstName": "Y.", "lastName": "Zhu"},
            {"creatorType": "author", "firstName": "X.", "lastName": "Li"},
        ],
        "date": "2018-06-15",
        "DOI": "10.1103/PhysRevLett.121.124501",
        "publicationTitle": "Physical Review Letters",
        "volume": "121",
        "issue": "12",
        "pages": "124501",
        "url": "https://doi.org/10.1103/PhysRevLett.121.124501",
        "extra": "openalex_id: W123\njif: 8.6",
    },
}

PREPRINT_ITEM = {
    "key": "EFGH5678",
    "data": {
        "itemType": "preprint",
        "title": "Non-Hermitian metagratings with degenerate states",
        "creators": [{"creatorType": "author", "firstName": "F.", "lastName": "Fang"}],
        "date": "2023",
        "repository": "arXiv",
        "archiveID": "arXiv:2301.09876",
        "extra": "arxiv_id: 2301.09876",
        "url": "https://arxiv.org/abs/2301.09876",
    },
}


def test_make_citekey_journal():
    key = refs_bridge.make_citekey(JOURNAL_ITEM["data"])
    assert key == "zhu2018simultaneous"


def test_make_citekey_preprint():
    key = refs_bridge.make_citekey(PREPRINT_ITEM["data"])
    assert key.startswith("fang2023")


def test_item_to_bibtex_journal():
    bib = refs_bridge.item_to_bibtex(JOURNAL_ITEM)
    assert bib.startswith("@article{zhu2018simultaneous,")
    assert "author = {Zhu, Y. and Li, X.}" in bib
    assert "journal = {Physical Review Letters}" in bib
    assert "year = {2018}" in bib
    assert "volume = {121}" in bib
    assert "number = {12}" in bib
    assert "pages = {124501}" in bib
    assert "doi = {10.1103/PhysRevLett.121.124501}" in bib
    assert bib.rstrip().endswith("}")


def test_item_to_bibtex_preprint_arxiv():
    bib = refs_bridge.item_to_bibtex(PREPRINT_ITEM)
    assert bib.startswith("@misc{")
    assert "eprint = {2301.09876}" in bib
    assert "archivePrefix = {arXiv}" in bib
    assert "howpublished = {arXiv}" in bib


def test_item_to_bibtex_custom_citekey():
    bib = refs_bridge.item_to_bibtex(JOURNAL_ITEM, citekey="mykey2024")
    assert bib.startswith("@article{mykey2024,")


def test_flat_item_without_data_wrapper():
    flat = JOURNAL_ITEM["data"]
    bib = refs_bridge.item_to_bibtex(flat)
    assert "@article{zhu2018simultaneous," in bib


def test_escapes_ampersand():
    item = {
        "data": {
            "itemType": "journalArticle",
            "title": "Gain & loss in acoustic systems",
            "creators": [{"lastName": "Doe", "firstName": "J."}],
            "date": "2021",
        }
    }
    bib = refs_bridge.item_to_bibtex(item)
    assert r"Gain \& loss" in bib
