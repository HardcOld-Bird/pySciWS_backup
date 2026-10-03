"""WP-E 的离线回归测试：期刊质量指标层（listed_in → journal_tier + SCImago SJR 索引）。

护栏对象是三类问题：

1. **``listed_in`` 被整个丢掉**（E1）：``_extract_source`` 旧实现只取 ``summary_stats``，
   于是期刊档次只剩引用类指标可用。对声学这类**低引用密度**领域严重失真——
   ``J. Acoust. Soc. Am.`` 的 OpenAlex ``2yr_mean_citedness`` 只有 0.82，但 JUFO 的专家
   小组把它判为 3 级（最高档），与 Nature / PRL 同级。本文件用**实测数据**做夹具。
2. **``jcr_quartile`` 恒为空导致的死代码**（E1）：``_print_rows`` / ``_render_candidates``
   里 ``if r["jcr_quartile"]`` 那段分支自写成之日起从未为真（官方 JCR 分区需 WoS
   Journals API）。回退到 ``journal_tier`` 后尾注才真的携带信息。
3. **SCImago 索引的一切降级路径**（E2）：索引未建 / 损坏 / ISSN 无命中，都必须静默
   返回空值而不抛异常（不变量 1）；但**构建**路径相反——CSV 缺列必须报错，否则静默
   产出空索引会让所有笔记的 ``scimago_quartile`` 悄悄留空。

全部测试离线：OpenAlex 的 ``get_source`` / ``get_work`` 一律 monkeypatch；SCImago 索引用
夹具 CSV 现建。``conftest.isolated_scimago_index`` 保证测试结果与用户本机是否建过真实
索引无关。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pysci.skills.literature_research.tools import (
    arxiv_client,
    journal_metrics,
    notes,
    openalex_client,
    research,
)
from pysci.skills.literature_research.tools.config import settings

# ---------------------------------------------------------------------------
# 夹具：OpenAlex ``sources.listed_in`` 的实测值（方案 E1 的表格）
# ---------------------------------------------------------------------------
#: 每本期刊的实测 ``listed_in`` 与**期望档次**。数据来自 OpenAlex 生产端点，不是编的。
#:
#: 关键对照是 JASA：``2yr_mean_citedness`` 只有 0.82（引用类指标会把它判成低档），
#: 但 JUFO 专家评议给 3 级、Norway 给 2 级 → ``top``，与 Nature / PRL 同档。
MEASURED: dict[str, dict[str, Any]] = {
    "Nature": {
        "2yr_mean_citedness": 19.27,
        "listed_in": [
            "cwts-core",
            "doyens",
            "erih-plus",
            "jufo-3",
            "ki-jl-3",
            "medline",
            "norway-2",
        ],
        "tier": "top",
        "basis": ["jufo-3", "norway-2", "ki-jl-3"],
    },
    "Physical Review Letters": {
        "2yr_mean_citedness": 8.97,
        "listed_in": ["cwts-core", "jufo-3", "ki-jl-2", "medline", "norway-2"],
        "tier": "top",
        "basis": ["jufo-3", "norway-2"],
    },
    "Physical Review B": {
        "2yr_mean_citedness": 3.98,
        "listed_in": ["cwts-core", "jufo-2", "norway-2"],
        "tier": "top",
        "basis": ["norway-2"],
    },
    "Journal of the Acoustical Society of America": {
        "2yr_mean_citedness": 0.82,
        "listed_in": [
            "cwts-core",
            "erih-plus",
            "jufo-3",
            "ki-jl-1",
            "medline",
            "norway-2",
        ],
        "tier": "top",
        "basis": ["jufo-3", "norway-2"],
    },
    "Ultrasonics": {
        "2yr_mean_citedness": 4.44,
        "listed_in": ["cwts-core", "doyens", "jufo-1", "medline", "norway-1"],
        "tier": "basic",
        "basis": ["jufo-1", "norway-1"],
    },
}

#: 官方 SCImago CSV 的表头（截取前 10 列，含一个带括号年份的列用于版本年推断）。
CSV_HEADER = (
    "Rank;Sourceid;Type;Title;Issn;SJR;SJR Best Quartile;H index;"
    "Total Docs. (2024);Country"
)

#: 5 行夹具，逐行覆盖一种真实会遇到的形态：
#:
#: - Nature：单值 ISSN + **欧洲小数逗号**（``24,000`` = 24.0）
#: - PRL：**引号包住**的多值 ISSN（csv reader 正常解析为一个字段）
#: - PRB：**未加引号**的多值 ISSN —— ``;`` 既是多值分隔符又是字段分隔符，该行会比表头
#:   多出一个字段，构建器必须做错位补偿，否则 SJR / Quartile / H index 全部读错列
#: - JASA：低 SJR（真实世界它就是 0.81 量级）
#: - Ultrasonics：校验位是 **X**（``0041-624X``）
CSV_ROWS = [
    "1;17609;journal;Nature;0028-0836;24,000;Q1;982;869;United Kingdom",
    '3;2120;journal;Physical Review Letters;"0031-9007;1079-7114";8.970;Q1;1005;2478;United States',
    "7;2136;journal;Physical Review B;2469-9950;2469-9969;3.200;Q1;672;4811;United States",
    "318;12657;journal;J. Acoust. Soc. Am.;0001-4966;0.810;Q2;190;412;United States",
    "611;2100264;journal;Ultrasonics;0041-624X;1.290;Q1;213;271;Netherlands",
]

FIXTURE_CSV = CSV_HEADER + "\n" + "\n".join(CSV_ROWS) + "\n"

#: 夹具索引里应当出现的全部归一化 ISSN（5 行 → 7 个 key，因两行是多值）。
EXPECTED_KEYS = {
    "00280836",
    "00319007",
    "10797114",
    "24699950",
    "24699969",
    "00014966",
    "0041624X",
}


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """把夹具 CSV 落盘。文件名里带年份，用于测试「表头推断失败时退回文件名」。"""
    p = tmp_path / "scimagojr 2024.csv"
    p.write_text(FIXTURE_CSV, encoding="utf-8")
    return p


@pytest.fixture
def built_index(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """构建夹具索引并让 ``index_path()`` 指向它（覆盖 conftest 的「不存在」默认态）。"""
    dest = tmp_path / "scimago_index.json"
    journal_metrics.build_scimago_index(csv_file, out_path=dest)
    monkeypatch.setattr(journal_metrics, "index_path", lambda: dest)
    journal_metrics._CACHE.clear()
    return dest


def _source(name: str, **extra: Any) -> dict[str, Any]:
    """按实测数据造一个 OpenAlex source dict。"""
    m = MEASURED[name]
    return {
        "openalex_id": f"S{name[:3].upper()}",
        "display_name": name,
        "issn": ["0000-0000"],
        "h_index": 500,
        "2yr_mean_citedness": m["2yr_mean_citedness"],
        "listed_in": m["listed_in"],
        **extra,
    }


# ===========================================================================
#  E1 —— derive_journal_tier（实测数据夹具）
# ===========================================================================
@pytest.mark.parametrize("name", sorted(MEASURED))
def test_derives_tier_from_measured_listed_in(name):
    """方案 E1 表格里的 5 本期刊，逐本核对派生结果。"""
    tier, basis = notes.derive_journal_tier(MEASURED[name]["listed_in"])
    assert tier == MEASURED[name]["tier"], name
    assert basis == MEASURED[name]["basis"], name


def test_nature_basis_contains_jufo3():
    """方案点名的断言：Nature → ``top``，且 basis 含 ``jufo-3``。"""
    tier, basis = notes.derive_journal_tier(MEASURED["Nature"]["listed_in"])
    assert tier == "top"
    assert "jufo-3" in basis


def test_ultrasonics_is_basic_not_top():
    """Ultrasonics 只有 jufo-1 / norway-1 → ``basic``（最高档取自两套名单的共同结论）。"""
    tier, basis = notes.derive_journal_tier(MEASURED["Ultrasonics"]["listed_in"])
    assert tier == "basic"
    assert basis == ["jufo-1", "norway-1"]


def test_jasa_is_top_despite_low_citedness():
    """**本层存在的理由**：JASA 的 2yr_mean_citedness 只有 0.82，档次却是 top。

    若哪天有人把 ``derive_journal_tier`` 改成按引用指标算，这条会立刻失败。
    """
    assert (
        MEASURED["Journal of the Acoustical Society of America"]["2yr_mean_citedness"]
        < 1.0
    )
    tier, _ = notes.derive_journal_tier(
        MEASURED["Journal of the Acoustical Society of America"]["listed_in"]
    )
    assert tier == "top"


def test_physrevb_takes_max_across_lists():
    """PRB 的 jufo-2（leading）与 norway-2（top）冲突时取**最高档**。"""
    tier, basis = notes.derive_journal_tier(MEASURED["Physical Review B"]["listed_in"])
    assert tier == "top"
    assert basis == ["norway-2"]  # jufo-2 只到 leading，不进 basis


@pytest.mark.parametrize("listed_in", [[], None, (), [""]])
def test_empty_listed_in_yields_empty_tier(listed_in):
    """空输入 → ``("", [])``，而不是 ``"basic"``（「无从判断」≠「判定为低档」）。"""
    assert notes.derive_journal_tier(listed_in) == ("", [])


@pytest.mark.parametrize(
    "listed_in",
    [
        ["cwts-core"],
        ["medline"],
        ["cwts-core", "erih-plus", "medline", "doaj", "doyens"],
    ],
)
def test_binary_membership_lists_do_not_derive_a_tier(listed_in):
    """``cwts-core`` / ``medline`` 等是**二元**收录标记，不含档次，不参与派生。"""
    assert notes.derive_journal_tier(listed_in) == ("", [])


def test_unknown_list_name_is_ignored():
    """OpenAlex 将来新增名单时不得崩，也不得凭空造档次。"""
    assert notes.derive_journal_tier(["some-future-list-3"]) == ("", [])


def test_out_of_range_level_is_ignored():
    """Norway 只有 1/2 两级；``norway-3`` 属越界，忽略而非当成 top。"""
    assert notes.derive_journal_tier(["norway-3"]) == ("", [])


@pytest.mark.parametrize("listed_in", ["jufo-3", 42, object(), {"jufo-3"}])
def test_non_iterable_or_non_string_input_degrades(listed_in):
    """不变量 1：畸形输入一律降级为 ``("", [])``，绝不抛异常。

    ``{"jufo-3"}`` 是 set——可迭代且元素是字符串，因此**能**正常派生；这里只断言不抛。
    """
    tier, basis = notes.derive_journal_tier(listed_in)
    assert isinstance(tier, str) and isinstance(basis, list)


def test_set_input_still_derives():
    assert notes.derive_journal_tier({"jufo-3"}) == ("top", ["jufo-3"])


def test_mixed_junk_elements_are_skipped():
    got = notes.derive_journal_tier([None, 3, "jufo-2", "medline", "norway-1"])
    assert got == ("leading", ["jufo-2"])


def test_case_and_whitespace_insensitive():
    assert notes.derive_journal_tier(["  JUFO-3 "]) == ("top", ["jufo-3"])


def test_basis_order_is_stable_regardless_of_input_order():
    """``basis`` 按名单声明序（jufo → norway → ki-jl）排，与 ``listed_in`` 原序无关——
    否则同一本刊的 frontmatter 会因 API 返回顺序抖动而产生无意义的 diff。"""
    a = notes.derive_journal_tier(["norway-2", "jufo-3"])[1]
    b = notes.derive_journal_tier(["jufo-3", "norway-2"])[1]
    assert a == b == ["jufo-3", "norway-2"]


def test_ki_jl_hyphenated_list_name_is_split_correctly():
    """``ki-jl-2`` 必须拆成 ``ki-jl`` + ``2``，惰性量词若写成贪婪会拆成 ``ki`` + ``jl-2``。"""
    assert notes.derive_journal_tier(["ki-jl-3"]) == ("top", ["ki-jl-3"])
    assert notes.derive_journal_tier(["ki-jl-2"]) == ("leading", ["ki-jl-2"])
    assert notes.derive_journal_tier(["ki-jl-1"]) == ("basic", ["ki-jl-1"])


# ===========================================================================
#  E1 —— 两个转换器都产出三个新字段
#
#  （``_extract_source`` 保留 ``listed_in`` 的两个回归用例住在
#   ``test_openalex_normalize.py``——那是归一化契约的自然归属。）
# ===========================================================================
def test_openalex_converter_derives_tier(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        openalex_client, "get_source", lambda **kw: _source("Ultrasonics")
    )
    fm = openalex_client.work_to_note_frontmatter(
        {"title": "T", "journal_openalex_id": "S123"}
    )
    assert fm["journal_tier"] == "basic"
    assert fm["journal_tier_basis"] == ["jufo-1", "norway-1"]
    assert fm["listed_in"] == MEASURED["Ultrasonics"]["listed_in"]


def test_openalex_converter_without_source_leaves_tier_blank(
    monkeypatch: pytest.MonkeyPatch,
):
    """无 ``journal_openalex_id`` 时不查 source，三字段一律留空而不编造。"""
    monkeypatch.setattr(openalex_client, "get_source", lambda **kw: None)
    fm = openalex_client.work_to_note_frontmatter({"title": "T"})
    assert fm["journal_tier"] == ""
    assert fm["journal_tier_basis"] == []
    assert fm["listed_in"] == []


def test_arxiv_converter_leaves_tier_blank():
    """arXiv 的 Atom 响应里没有期刊记录，三字段必须留空待 ``_attach_journal_metrics`` 回填。"""
    fm = arxiv_client.arxiv_to_note_frontmatter(
        {"title": "T", "published": "2018-03-12T04:08:19Z"}
    )
    assert fm["journal_tier"] == ""
    assert fm["journal_tier_basis"] == []
    assert fm["listed_in"] == []


def test_new_fields_are_in_field_order():
    for k in ("journal_tier", "journal_tier_basis", "listed_in"):
        assert k in notes.FIELD_ORDER, k


def test_new_fields_come_after_journal_h_index():
    """字段序与模板一致，``index --fix`` 的规范化 diff 才不含字段搬家噪声。"""
    i = notes.FIELD_ORDER.index("journal_h_index")
    assert notes.FIELD_ORDER[i + 1 : i + 4] == (
        "journal_tier",
        "journal_tier_basis",
        "listed_in",
    )


# ===========================================================================
#  E1 —— _attach_journal_metrics 回填档次
# ===========================================================================
def _mock_lookup(
    monkeypatch: pytest.MonkeyPatch, source: dict[str, Any] | None
) -> None:
    monkeypatch.setattr(research.openalex_client, "get_work", lambda **kw: None)
    monkeypatch.setattr(research.openalex_client, "get_source", lambda **kw: source)


def test_attach_fills_tier_and_listed_in(monkeypatch: pytest.MonkeyPatch, capsys):
    """D2/E1 的交汇点：arXiv 笔记原本三个字段全空，补齐后应带上专家评议档次。"""
    _mock_lookup(monkeypatch, _source("Journal of the Acoustical Society of America"))
    fm = arxiv_client.arxiv_to_note_frontmatter(
        {
            "title": "T",
            "published": "2018-01-01T00:00:00Z",
            "journal_ref": "J. Acoust. Soc. Am. 143, 1245 (2018)",
        }
    )
    got = research._attach_journal_metrics(fm)
    assert got["journal_tier"] == "top"
    assert got["journal_tier_basis"] == ["jufo-3", "norway-2"]
    assert (
        got["listed_in"]
        == MEASURED["Journal of the Acoustical Society of America"]["listed_in"]
    )
    assert got["jif"] == 0.82
    out = capsys.readouterr().out
    assert "journal_tier" in out


def test_attach_does_not_overwrite_existing_tier(monkeypatch: pytest.MonkeyPatch):
    """用户可能手工把档次改成自己的判断；机器值不得顶掉它。"""
    _mock_lookup(monkeypatch, _source("Nature"))
    fm = {
        "journal": "Nature",
        "doi": "",
        "jif": None,
        "journal_tier": "leading",
        "journal_tier_basis": ["手工"],
        "listed_in": [],
    }
    got = research._attach_journal_metrics(fm)
    assert got["journal_tier"] == "leading"
    assert got["journal_tier_basis"] == ["手工"]
    assert got["listed_in"] == MEASURED["Nature"]["listed_in"]  # 空的照补


def test_attach_tier_lookup_failure_is_silent(monkeypatch: pytest.MonkeyPatch, capsys):
    """不变量 1：``get_source`` 抛异常时三字段保持空，只留一行 stderr 提示。"""

    def _boom(**kw: Any) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr(research.openalex_client, "get_work", _boom)
    monkeypatch.setattr(research.openalex_client, "get_source", _boom)
    fm = {"journal": "Nature", "doi": "", "jif": None, "journal_tier": ""}
    got = research._attach_journal_metrics(fm)
    assert got["journal_tier"] == ""
    assert "未取到期刊指标" in capsys.readouterr().err


def test_attach_fills_scimago_quartile(
    built_index: Path, monkeypatch: pytest.MonkeyPatch
):
    """E1 与 E2 共用同一条链路：一次 ``get_source`` 同时喂饱档次与分区。"""
    _mock_lookup(
        monkeypatch, _source("Physical Review Letters", issn=["0031-9007", "1079-7114"])
    )
    fm = {
        "journal": "Phys. Rev. Lett.",
        "doi": "",
        "jif": None,
        "scimago_quartile": "",
        "journal_tier": "",
    }
    got = research._attach_journal_metrics(fm)
    assert got["scimago_quartile"] == "Q1"
    assert got["journal_tier"] == "top"


# ===========================================================================
#  E1 —— 展示层：_row / _print_rows / _render_index
# ===========================================================================
def test_row_carries_journal_tier():
    r = research._row("openalex", {"title": "T", "journal_tier": "top"})
    assert r["journal_tier"] == "top"


def test_row_derives_tier_from_listed_in():
    """work dict 只带 ``listed_in`` 时就地派生，展示层不比 frontmatter 少一维信息。"""
    r = research._row(
        "openalex",
        {"title": "T", "listed_in": MEASURED["Ultrasonics"]["listed_in"]},
    )
    assert r["journal_tier"] == "basic"


def test_row_tier_defaults_to_empty():
    assert research._row("arxiv", {"title": "T"})["journal_tier"] == ""


def test_print_rows_falls_back_to_tier(capsys):
    """``jcr_quartile`` 恒为空 → 过去这个尾注分支是死代码；现在应显 ``tier:top``。"""
    research._print_rows(
        [("openalex", {"title": "T", "jif": 8.97, "journal_tier": "top"})]
    )
    out = capsys.readouterr().out
    assert "[JIF 8.97 tier:top]" in out


def test_print_rows_prefers_jcr_quartile_over_tier(capsys):
    """WoS Journals API 接入后 ``jcr_quartile`` 会有真值，届时它优先于派生档次。"""
    research._print_rows(
        [("openalex", {"title": "T", "jcr_quartile": "Q1", "journal_tier": "top"})]
    )
    out = capsys.readouterr().out
    assert "Q1" in out and "tier:top" not in out


def test_print_rows_tier_only_without_jif(capsys):
    """无 JIF 时尾注不应带前导空格（旧实现用字符串拼接会产生 `` tier:top``）。"""
    research._print_rows([("openalex", {"title": "T", "journal_tier": "top"})])
    assert "[tier:top]" in capsys.readouterr().out


def test_render_candidates_falls_back_to_tier():
    """shortlist 是 AI 初筛时读的东西，同一处死代码同样要回退。"""
    md = research._render_candidates(
        [("openalex", {"title": "T", "journal": "Nature", "journal_tier": "top"})]
    )
    assert "tier:top" in md


def test_render_index_has_a_tier_column():
    md = research._render_index([("2018_zhu_x", {"title": "T", "year": 2018})])
    header = next(line for line in md.splitlines() if line.startswith("| # |"))
    assert "Tier" in header
    # Tier 必须是**独立**列，不能污染 Journal 列
    assert header.split("|").index(" Tier ") > header.split("|").index(" Journal ")


def test_render_index_tier_column_shows_dash_when_unknown():
    md = research._render_index([("2018_zhu_x", {"title": "T", "year": 2018})])
    row = next(line for line in md.splitlines() if line.startswith("| 1 |"))
    assert row.count("|") == 10  # 9 列 → 10 个分隔符
    assert "| — |" in row


def test_render_index_tier_column_shows_value():
    md = research._render_index(
        [("2018_zhu_x", {"title": "T", "year": 2018, "journal_tier": "top"})]
    )
    row = next(line for line in md.splitlines() if line.startswith("| 1 |"))
    assert "| top |" in row


# ===========================================================================
#  E2 —— normalize_issn
# ===========================================================================
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0031-9007", "00319007"),
        ("00319007", "00319007"),
        ("1079-711x", "1079711X"),
        ("  0041-624X ", "0041624X"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_issn(raw, expected):
    assert journal_metrics.normalize_issn(raw) == expected


# ===========================================================================
#  E2 —— build_scimago_index
# ===========================================================================
def test_build_index_from_fixture_csv(csv_file: Path, tmp_path: Path):
    dest = tmp_path / "idx.json"
    payload = journal_metrics.build_scimago_index(csv_file, out_path=dest)
    assert dest.exists()
    assert set(payload["by_issn"]) == EXPECTED_KEYS
    assert payload["_meta"]["sjr_year"] == 2024  # 从表头 Total Docs. (2024) 推断
    assert payload["_meta"]["n_entries"] == len(EXPECTED_KEYS)
    assert payload["_meta"]["source_csv"] == csv_file.name
    assert "Scopus" in payload["_meta"]["attribution"]


def test_build_index_writes_compact_json(csv_file: Path, tmp_path: Path):
    """紧凑序列化：约 4 万条时这决定了索引是 1.4 MB 还是 4 MB，git 能否长期承受。"""
    dest = tmp_path / "idx.json"
    journal_metrics.build_scimago_index(csv_file, out_path=dest)
    text = dest.read_text(encoding="utf-8")
    assert "\n" not in text.strip()
    assert '": [' not in text and '", "' not in text


def test_multi_value_issn_builds_both_keys(csv_file: Path, tmp_path: Path):
    """方案点名的用例：``"0031-9007;1079-7114"`` 两个 key 都要建立，指向同一条记录。"""
    payload = journal_metrics.build_scimago_index(
        csv_file, out_path=tmp_path / "idx.json"
    )
    by = payload["by_issn"]
    assert by["00319007"] == by["10797114"]
    assert by["00319007"][0] == "Q1"
    assert by["00319007"][2] == 1005


def test_unquoted_multi_value_issn_is_realigned(csv_file: Path, tmp_path: Path):
    """未加引号的多值 ISSN 会让该行多出一个字段；不做错位补偿就会读错列。

    这是 SCImago CSV 的真实形态（``;`` 既是字段分隔符又是多值分隔符），补偿逻辑一旦
    退化，PRB 的 SJR 会读成第二个 ISSN、分区会读成 SJR，而测试数值全是 plausible 的
    字符串——不会报错，只会静默产出垃圾索引。
    """
    payload = journal_metrics.build_scimago_index(
        csv_file, out_path=tmp_path / "idx.json"
    )
    by = payload["by_issn"]
    assert by["24699950"] == by["24699969"] == ["Q1", 3.2, 672]


def test_european_decimal_comma_is_parsed(csv_file: Path, tmp_path: Path):
    payload = journal_metrics.build_scimago_index(
        csv_file, out_path=tmp_path / "idx.json"
    )
    assert payload["by_issn"]["00280836"] == ["Q1", 24.0, 982]


def test_x_check_digit_issn_is_kept(csv_file: Path, tmp_path: Path):
    payload = journal_metrics.build_scimago_index(
        csv_file, out_path=tmp_path / "idx.json"
    )
    assert "0041624X" in payload["by_issn"]


def test_comma_delimited_csv_also_works(tmp_path: Path):
    """分隔符靠表头判定而非写死——写死分号会让逗号版静默产出空索引。"""
    p = tmp_path / "comma.csv"
    p.write_text(
        "Rank,Sourceid,Type,Title,Issn,SJR,SJR Best Quartile,H index,Country\n"
        '1,17609,journal,Nature,"0028-0836,1476-4687",24.000,Q1,982,UK\n',
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["by_issn"]["00280836"][0] == "Q1"
    assert "14764687" in payload["by_issn"]


def test_year_override_wins(csv_file: Path, tmp_path: Path):
    payload = journal_metrics.build_scimago_index(
        csv_file, year=2023, out_path=tmp_path / "idx.json"
    )
    assert payload["_meta"]["sjr_year"] == 2023


def test_year_inferred_from_filename_when_header_has_none(tmp_path: Path):
    p = tmp_path / "scimagojr-2025-export.csv"
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n1;0028-0836;24.0;Q1;982\n",
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["_meta"]["sjr_year"] == 2025


def test_year_is_none_when_uninferable(tmp_path: Path):
    """推断不出就留 ``None``，不编一个年份出来。"""
    p = tmp_path / "export.csv"
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n1;0028-0836;24.0;Q1;982\n",
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["_meta"]["sjr_year"] is None


def test_unknown_quartile_value_becomes_empty(tmp_path: Path):
    """SCImago 对新刊给 ``"-"``；按「无分区」处理，但 SJR / H index 仍保留。"""
    p = tmp_path / "new.csv"
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n1;0028-0836;0.100;-;3\n",
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["by_issn"]["00280836"] == ["", 0.1, 3]


def test_duplicate_issn_keeps_first_row(tmp_path: Path):
    """CSV 按 Rank 升序，首个即最优记录；后来的重复项不得覆盖它。"""
    p = tmp_path / "dup.csv"
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n"
        "1;0028-0836;24.0;Q1;982\n"
        "9999;0028-0836;0.1;Q4;1\n",
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["by_issn"]["00280836"] == ["Q1", 24.0, 982]


def test_utf8_bom_is_tolerated(tmp_path: Path):
    p = tmp_path / "bom.csv"
    p.write_bytes(
        b"\xef\xbb\xbfRank;Issn;SJR;SJR Best Quartile;H index\n"
        b"1;0028-0836;24.0;Q1;982\n"
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["by_issn"]["00280836"][0] == "Q1"


@pytest.mark.parametrize(
    "header",
    [
        "Rank;Title;SJR;SJR Best Quartile;H index",  # 缺 Issn
        "Rank;Issn;SJR;H index",  # 缺 SJR Best Quartile
    ],
)
def test_missing_required_column_raises(header: str, tmp_path: Path):
    """构建路径**必须**报错：静默产出空索引比构建失败难查得多。"""
    p = tmp_path / "bad.csv"
    p.write_text(header + "\n1;0028-0836;24.0;Q1;982\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")


def test_no_issn_parsed_raises(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n1;;24.0;Q1;982\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="没解析出任何 ISSN"):
        journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")


def test_empty_csv_raises(tmp_path: Path):
    p = tmp_path / "zero.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV 为空"):
        journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")


def test_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        journal_metrics.build_scimago_index(
            tmp_path / "nope.csv", out_path=tmp_path / "idx.json"
        )


def test_sourceid_is_not_mistaken_for_an_issn(tmp_path: Path):
    """``Sourceid`` 是 11 位纯数字，与无连字符 ISSN 形态接近；只扫 Issn 列才不会误收。"""
    p = tmp_path / "ids.csv"
    p.write_text(
        "Rank;Sourceid;Issn;SJR;SJR Best Quartile;H index\n"
        "1;21100200807;0028-0836;24.0;Q1;982\n",
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert set(payload["by_issn"]) == {"00280836"}


def test_short_row_does_not_crash(tmp_path: Path):
    """SCImago 导出偶有尾列缺失；越界取值返回空串而不是 IndexError。"""
    p = tmp_path / "short.csv"
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n1;0028-0836;24.0;Q1\n",
        encoding="utf-8",
    )
    payload = journal_metrics.build_scimago_index(p, out_path=tmp_path / "idx.json")
    assert payload["by_issn"]["00280836"] == ["Q1", 24.0, None]


# ===========================================================================
#  E2 —— lookup / quartile_for / status
# ===========================================================================
def test_lookup_exact_hit(built_index: Path):
    got = journal_metrics.lookup("0031-9007")
    assert got is not None
    assert got["quartile"] == "Q1"
    assert got["sjr"] == 8.97
    assert got["h_index"] == 1005
    assert got["issn"] == "00319007"
    assert got["sjr_year"] == 2024


def test_lookup_normalized_hit(built_index: Path):
    """去掉连字符后同样命中（索引 key 与查询 key 走同一个归一化函数）。"""
    assert journal_metrics.lookup("00319007") is not None
    assert journal_metrics.lookup("0041624x")["quartile"] == "Q1"


def test_lookup_accepts_candidate_list(built_index: Path):
    """OpenAlex source 的 ``issn`` 是数组；传数组时取首个命中。"""
    got = journal_metrics.lookup(["9999-9999", "0001-4966"])
    assert got is not None and got["issn"] == "00014966"
    assert got["quartile"] == "Q2"


def test_lookup_miss_returns_none(built_index: Path):
    assert journal_metrics.lookup("9999-9999") is None


@pytest.mark.parametrize("bad", [None, "", "   ", 42, [], ["", None]])
def test_lookup_garbage_input_returns_none(built_index: Path, bad):
    """不变量 1：垃圾输入静默 miss，不抛异常。"""
    assert journal_metrics.lookup(bad) is None


def test_quartile_for(built_index: Path):
    assert journal_metrics.quartile_for("0001-4966") == "Q2"
    assert journal_metrics.quartile_for(["0031-9007"]) == "Q1"


def test_quartile_for_miss_is_empty_string(built_index: Path):
    """返回 ``""`` 而非 ``None``：``scimago_quartile`` 在模板里是字符串字段，
    空串才能被 ``merge_frontmatter`` 当作「待填」。"""
    assert journal_metrics.quartile_for("9999-9999") == ""


def test_status_when_built(built_index: Path):
    st = journal_metrics.status()
    assert st["exists"] is True
    assert st["readable"] is True
    assert st["n_entries"] == len(EXPECTED_KEYS)
    assert st["sjr_year"] == 2024
    assert st["built_at"]
    assert st["source_csv"] == "scimagojr 2024.csv"
    assert "Scopus" in st["attribution"]
    assert st["download_url"] == journal_metrics.DOWNLOAD_URL


def test_status_when_missing(isolated_scimago_index: Path):
    """索引不存在 → 不抛异常，``exists`` 为假，其余计数字段为 0/None。"""
    st = journal_metrics.status()
    assert st["exists"] is False
    assert st["readable"] is False
    assert st["n_entries"] == 0
    assert st["sjr_year"] is None
    assert st["path"] == str(isolated_scimago_index)


def test_status_when_corrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """损坏的索引比没有索引更糟：``exists`` 为真但 ``readable`` 为假，让 doctor 能提示重建。"""
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(journal_metrics, "index_path", lambda: bad)
    journal_metrics._CACHE.clear()
    st = journal_metrics.status()
    assert st["exists"] is True and st["readable"] is False
    assert journal_metrics.quartile_for("0031-9007") == ""


def test_status_when_schema_is_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad = tmp_path / "wrong.json"
    bad.write_text(json.dumps({"unexpected": 1}), encoding="utf-8")
    monkeypatch.setattr(journal_metrics, "index_path", lambda: bad)
    journal_metrics._CACHE.clear()
    assert journal_metrics.status()["readable"] is False
    assert journal_metrics.lookup("0031-9007") is None


def test_degradation_without_index(isolated_scimago_index: Path):
    """索引未建时的三件套：不抛异常、``quartile_for`` 空串、``status`` 报未建。"""
    assert not isolated_scimago_index.exists()
    assert journal_metrics.quartile_for("0031-9007") == ""
    assert journal_metrics.lookup("0031-9007") is None
    assert journal_metrics.status()["exists"] is False


def _rewrite(dest: Path) -> Path:
    """造一份只含 Nature 的 CSV，放在索引旁边，用于测试缓存失效。"""
    p = dest.with_name("small.csv")
    p.write_text(
        "Rank;Issn;SJR;SJR Best Quartile;H index\n1;0028-0836;24.0;Q1;982\n",
        encoding="utf-8",
    )
    return p


def test_index_is_cached_within_a_process(built_index: Path):
    """约 1.4 MB 的 JSON 不该每篇笔记重读一遍；缓存键含 mtime/size，重建后自动失效。"""
    assert journal_metrics.quartile_for("0031-9007") == "Q1"
    assert journal_metrics._CACHE  # 已载入
    # 把内容不同的索引写到**同一路径**，缓存必须失效（PRL 消失、只剩 Nature）
    journal_metrics.build_scimago_index(_rewrite(built_index), out_path=built_index)
    assert journal_metrics.lookup("0031-9007") is None
    assert journal_metrics.quartile_for("0028-0836") == "Q1"


def test_default_index_path_follows_module_dir():
    """路径由 ``settings.module_dir`` 锚定；写成 ``data/data/`` 嵌套会让用户找不到文件。"""
    assert settings.scimago_index_path == (
        settings.module_dir / "data" / "scimago_index.json"
    )
    assert settings.scimago_index_path.name == journal_metrics.INDEX_FILENAME


def test_scimago_ready_reflects_file_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import dataclasses

    fake_root = tmp_path / "module"
    st = dataclasses.replace(settings, module_dir=fake_root)
    assert st.scimago_ready is False
    st.scimago_index_path.parent.mkdir(parents=True, exist_ok=True)
    st.scimago_index_path.write_text("{}", encoding="utf-8")
    assert st.scimago_ready is True


def test_summary_mentions_scimago():
    """``config --help`` 之外的常规排障入口是 ``settings.summary()``，新指标层要在里面。"""
    s = settings.summary()
    assert "scimago_index" in s and "scimago_ready" in s


# ===========================================================================
#  E3 —— journal 子命令
# ===========================================================================
def test_parser_accepts_journal_subcommand():
    args = research.build_parser().parse_args(["journal", "status"])
    assert args.cmd == "journal" and args.action == "status"
    assert args.func is research.cmd_journal


def test_parser_journal_lookup_takes_issn_positionally():
    args = research.build_parser().parse_args(["journal", "lookup", "0031-9007"])
    assert args.action == "lookup" and args.issn == "0031-9007"


def test_parser_journal_build_scimago_flags():
    args = research.build_parser().parse_args(
        ["journal", "build-scimago", "--csv", "x.csv", "--year", "2025"]
    )
    assert args.csv == "x.csv" and args.year == 2025


def test_journal_status_reports_unbuilt(isolated_scimago_index: Path, capsys):
    """未建索引时 ``journal status`` 必须给出**可执行的下一步**，而不是只说「没有」。"""
    assert research.main(["journal", "status"]) == 0
    out = capsys.readouterr().out
    assert "未建" in out
    assert "build-scimago" in out
    assert journal_metrics.DOWNLOAD_URL in out


def test_journal_status_reports_built(built_index: Path, capsys):
    assert research.main(["journal", "status"]) == 0
    out = capsys.readouterr().out
    assert "已建" in out and str(len(EXPECTED_KEYS)) in out and "2024" in out


def test_journal_status_json(isolated_scimago_index: Path, capsys):
    assert research.main(["journal", "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exists"] is False


def test_journal_status_corrupt_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(journal_metrics, "index_path", lambda: bad)
    journal_metrics._CACHE.clear()
    assert research.main(["journal", "status"]) == 1
    assert "无法解析" in capsys.readouterr().out


def test_journal_build_scimago_requires_csv(capsys):
    assert research.main(["journal", "build-scimago"]) == 2
    assert "--csv" in capsys.readouterr().err


def test_journal_build_scimago_end_to_end(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """CLI 全链路：建索引 → 落到 ``index_path()`` → 立刻可查。"""
    dest = tmp_path / "sub" / "scimago_index.json"
    monkeypatch.setattr(journal_metrics, "index_path", lambda: dest)
    journal_metrics._CACHE.clear()

    assert research.main(["journal", "build-scimago", "--csv", str(csv_file)]) == 0
    out = capsys.readouterr().out
    assert dest.exists()
    assert str(len(EXPECTED_KEYS)) in out
    assert "SOURCE.md" in out  # 提示用户补数据来源台账
    assert journal_metrics.quartile_for("0031-9007") == "Q1"


def test_journal_build_scimago_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setattr(journal_metrics, "index_path", lambda: tmp_path / "idx.json")
    bad = tmp_path / "bad.csv"
    bad.write_text("Rank;Issn\n1;0028-0836\n", encoding="utf-8")
    assert research.main(["journal", "build-scimago", "--csv", str(bad)]) == 1
    assert "构建失败" in capsys.readouterr().err


def test_journal_lookup_requires_issn(capsys):
    assert research.main(["journal", "lookup"]) == 2
    assert "ISSN" in capsys.readouterr().err


def test_journal_lookup_shows_all_layers(
    built_index: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """``lookup`` 的价值在于**并排**：SCImago 分区、专家评议分级、引用类指标同时可见。"""
    monkeypatch.setattr(
        research.openalex_client,
        "get_source",
        lambda **kw: _source("Physical Review Letters", issn=["0031-9007"]),
    )
    assert research.main(["journal", "lookup", "0031-9007"]) == 0
    out = capsys.readouterr().out
    assert "SCImago" in out and "Q1" in out
    assert "journal_tier" in out and "top" in out
    assert "jufo-3" in out  # basis 可见 → 结论可审计
    assert "8.97" in out
    # 第四层：诚实说明官方 JCR 未接入，并指名唯一能提供它的 API
    assert "未接入" in out and "Web of Science Journals API" in out


def test_journal_lookup_rounds_jif_like_the_note(
    built_index: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """实测：OpenAlex 给 PRL ``8.966303149123881``，而笔记的 ``jif`` 是 ``round(x, 2)``。

    人类可读输出若直接打原始值，同一本刊就有两个数、无从判断哪个对；``--json`` 反过来
    必须保留原始精度（给程序用，不该丢信息）。
    """
    raw = 8.966303149123881
    monkeypatch.setattr(
        research.openalex_client,
        "get_source",
        lambda **kw: _source("Physical Review Letters", **{"2yr_mean_citedness": raw}),
    )
    assert research.main(["journal", "lookup", "0031-9007"]) == 0
    out = capsys.readouterr().out
    assert "8.97" in out
    assert str(raw) not in out

    assert research.main(["journal", "lookup", "0031-9007", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["jif"] == raw


def test_journal_lookup_without_openalex(
    built_index: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """OpenAlex 不可达时仍应显示 SCImago 那一层（静默降级，不整体失败）。"""

    def _boom(**kw: Any) -> Any:
        raise RuntimeError("down")

    monkeypatch.setattr(research.openalex_client, "get_source", _boom)
    assert research.main(["journal", "lookup", "0031-9007"]) == 0
    out = capsys.readouterr().out
    assert "Q1" in out
    assert "无数据" in out
    assert "RuntimeError" not in out


def test_journal_lookup_json(
    built_index: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setattr(
        research.openalex_client, "get_source", lambda **kw: _source("Nature")
    )
    assert research.main(["journal", "lookup", "0028-0836", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["journal_tier"] == "top"
    assert payload["scimago"]["quartile"] == "Q1"
    assert payload["openalex"]["display_name"] == "Nature"


def test_journal_lookup_unknown_issn_is_honest(
    isolated_scimago_index: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """两层都没数据时要说清「索引未建」而不是含糊的「无数据」。"""
    monkeypatch.setattr(research.openalex_client, "get_source", lambda **kw: None)
    assert research.main(["journal", "lookup", "9999-9999"]) == 0
    assert "索引未建" in capsys.readouterr().out


def test_doctor_reports_journal_metrics(isolated_scimago_index: Path, capsys):
    """``doctor`` 是排障第一站，新指标层必须在里面。"""
    assert research.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "【期刊质量指标】" in out
    assert "listed_in" in out
    assert "SCImago" in out and "未建" in out
    assert "WoS Journals API" in out


def test_doctor_reports_built_index(built_index: Path, capsys):
    assert research.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "已建" in out and str(len(EXPECTED_KEYS)) in out
