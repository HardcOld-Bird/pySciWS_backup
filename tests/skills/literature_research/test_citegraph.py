"""``research citegraph``（引文图谱 / 滚雪球）的回归测试。

覆盖方案 WP-F F3 明列的两项——``works_by_ids`` 在 50 条边界正确分批、``_row()`` 对被引
工作的投影正确——外加把整条链路钉住：``_norm_openalex_id`` → ``_extract_work_summary``
→ ``_resolve_openalex_work`` → ``_sort_rows`` → ``cmd_citegraph``。

**桩打在 :func:`openalex_client._get` 上**，那是本模块唯一的 HTTP 出口（``get_work`` /
``works_by_ids`` / ``works_citing`` / ``get_source`` 全经它）。打在更高层（直接替换
``works_by_ids``）会让分批逻辑本身逃过测试——而分批恰恰是本文件要验的东西。同理，
``cmd_citegraph`` 的端到端测试也让真实的 ``works_by_ids`` / ``works_citing`` 跑起来。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from pysci.skills.literature_research.tools import openalex_client as oa
from pysci.skills.literature_research.tools import research

# ---------------------------------------------------------------------------
# 夹具：伪造 OpenAlex 原始响应
# ---------------------------------------------------------------------------
#: 内联 ``primary_location.source`` 的形态取自**真实缓存**的 work 响应
#: （``openalex_https_api_openalex_org_works_doi_10_1038_2Fnphys1914_*.json``）。
#: 关键点：work 响应里的 source 是**内联对象**且带 ``listed_in``，所以期刊档次可以
#: 零额外请求地派生——这正是本文件最后一组测试要钉住的行为。
_REAL_INLINE_LISTED_IN = ["cwts-core", "jufo-3", "ki-jl-2", "norway-2"]


def _raw_work(
    *,
    oid: str = "W1000001",
    title: str = "A Study of Something",
    year: int = 2020,
    doi: str = "10.1103/physrevlett.121.124501",
    cited: int = 5,
    refs: tuple[str, ...] = (),
    src_name: str = "Physical Review Letters",
    listed_in: list[str] | None = None,
    issn: tuple[str, ...] = ("0031-9007", "1079-7114"),
    author: str = "Zhu, Wenjie",
    inline_source: bool = True,
) -> dict[str, Any]:
    """造一份 OpenAlex work 的**原始** JSON（``_extract_work_summary`` 的输入形态）。"""
    source: dict[str, Any] | None = None
    if inline_source:
        source = {
            "id": "https://openalex.org/S31939",
            "display_name": src_name,
            "issn_l": issn[0] if issn else "",
            "issn": list(issn),
            "listed_in": list(
                _REAL_INLINE_LISTED_IN if listed_in is None else listed_in
            ),
            "host_organization_name": "American Physical Society",
            "is_oa": False,
            "is_in_doaj": False,
            "is_core": True,
            "type": "journal",
        }
    return {
        "id": f"https://openalex.org/{oid}" if oid else "",
        "display_name": title,
        "publication_year": year,
        "publication_date": f"{year}-03-12",
        "type": "article",
        "doi": f"https://doi.org/{doi}" if doi else "",
        "cited_by_count": cited,
        "cited_by_percentile_year": {"min": 0.9},
        "open_access": {"is_oa": True, "oa_status": "green"},
        "primary_location": {"source": source, "landing_page_url": "", "pdf_url": ""},
        "best_oa_location": {
            "pdf_url": "",
            "landing_page_url": "https://arxiv.org/abs/x",
        },
        "authorships": [
            {
                "author": {"display_name": author, "orcid": "", "id": ""},
                "institutions": [],
                "is_corresponding": False,
                "raw_affiliation_string": "",
            }
        ],
        "biblio": {
            "volume": "121",
            "issue": "3",
            "first_page": "124501",
            "last_page": "",
        },
        "abstract_inverted_index": None,
        "concepts": [],
        "topics": [],
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "related_works": [],
        "counts_by_year": [],
    }


def _ids_from(filter_str: str) -> list[str]:
    """从 ``openalex_id:W1|W2|...`` 里取出 id 列表。"""
    if "openalex_id:" not in filter_str:
        return []
    return filter_str.split("openalex_id:", 1)[1].split("|")


def _echo_batch(params: dict[str, Any]) -> dict[str, Any]:
    """``/works?filter=openalex_id:...`` 的通用应答：每个 id 回一个最小 work。

    这样「返回条数 == 请求条数」，便于对分批做算术断言。DOI 必须**逐条唯一**：
    ``_row`` 的去重键以 DOI 为先，若沿用 ``_raw_work`` 的默认 DOI，一批里的所有条目
    会被 :func:`research._dedupe` 当成同一篇而只剩一条，把分批测试变成去重测试。
    """
    ids = _ids_from(str(params.get("filter", "")))
    return {
        "results": [
            _raw_work(oid=i, title=f"Ref {i}", doi=f"10.1103/ref.{i.lower()}")
            for i in ids
        ],
        "meta": {
            "count": len(ids),
            "db_response_time_ms": 1,
            "page": None,
            "per_page": len(ids),
        },
    }


def _is_single(url: str, p: dict[str, Any]) -> bool:
    """单篇 ``GET /works/...``（``get_work``）：URL 落在 ``/works/`` 之后且**没有** filter。"""
    return "/works/" in url and "filter" not in p


def _is_list(url: str, p: dict[str, Any]) -> bool:
    """批量查询（``works_by_ids`` / ``works_citing``）：``GET /works?filter=...``。"""
    return url.endswith("/works") and "filter" in p


def _empty_page(params: dict[str, Any]) -> dict[str, Any]:
    """「查到 0 条」的应答。"""
    return {"results": [], "meta": {"count": 0}}


class _Recorder:
    """替换 :func:`openalex_client._get`，按谓词分派应答并记录每一次调用。

    未命中任何 handler 时**抛 AssertionError** 而不是返回空 dict：静默返回会把「测试
    忘了配桩」伪装成「OpenAlex 没数据」，于是降级路径被误判为通过。
    """

    def __init__(
        self, handlers: Iterable[tuple[Callable[[str, dict], bool], Callable]] = ()
    ):
        self.handlers = list(handlers)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, params: dict[str, Any] | None = None, use_cache: bool = True
    ) -> dict[str, Any]:
        p = dict(params or {})
        self.calls.append({"url": url, "params": p})
        for match, fn in self.handlers:
            if match(url, p):
                return fn(p)
        raise AssertionError(f"未预期的 OpenAlex 调用：{url} params={p}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Recorder:
        monkeypatch.setattr(oa, "_get", self)
        return self

    # -- 便捷查询 --
    @property
    def filters(self) -> list[str]:
        return [c["params"].get("filter", "") for c in self.calls]

    def batch_sizes(self) -> list[int]:
        return [len(_ids_from(f)) for f in self.filters]

    def single_work_calls(self) -> list[str]:
        """单篇 ``GET /works/...``（不带 filter）的 URL 列表。"""
        return [c["url"] for c in self.calls if "filter" not in c["params"]]


#: 单篇 work 查询（``get_work``）的谓词：URL 落在 /works/ 之后且**没有** filter。
_is_single = lambda url, p: "/works/" in url and "filter" not in p  # noqa: E731
#: 批量查询（``works_by_ids`` / ``works_citing``）的谓词。
_is_list = lambda url, p: url.endswith("/works") and "filter" in p  # noqa: E731


def _single(payload: dict[str, Any]) -> Callable[[dict], dict[str, Any]]:
    return lambda p: payload


def _ns(**kw: Any) -> argparse.Namespace:
    """``cmd_citegraph`` 的 args，字段齐备（``_write_shortlist`` 会 getattr 一堆）。"""
    base = {
        "id": "10.1103/physrevlett.121.124501",
        "direction": "both",
        "limit": 25,
        "year": None,
        "sort": "citations",
        "save": False,
        "purpose": None,
        "json": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


# ===========================================================================
# _norm_openalex_id
# ===========================================================================
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("W2789790776", "W2789790776"),
        ("w2789790776", "W2789790776"),  # 小写归一
        ("2789790776", "W2789790776"),  # 纯数字补 W
        ("https://openalex.org/W2789790776", "W2789790776"),
        ("http://www.openalex.org/w2789790776/", "W2789790776"),  # www + 尾斜杠
        ("  W2789790776  ", "W2789790776"),  # 首尾空白
    ],
)
def test_norm_openalex_id_accepts_the_usual_spellings(raw: str, expected: str) -> None:
    assert oa._norm_openalex_id(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "   ",
        "10.1103/physrevlett.121.124501",  # DOI 不是 OpenAlex id
        "2401.12345",  # arXiv id 也不是
        "W123",  # 位数不足（正则要求 \d{5,}）
        "S31939",  # source id：补 W 后变成 WS31939，仍不合法
        "nonsense",
    ],
)
def test_norm_openalex_id_rejects_everything_else(raw: Any) -> None:
    """非法 id 一律归一为空串由调用方跳过：``works_by_ids`` 把 id 用 ``|`` 拼进 filter，
    一个畸形 id 会让**整批**请求返回 400，而不只是丢掉它自己。"""
    assert oa._norm_openalex_id(raw) == ""


# ===========================================================================
# works_by_ids：分批（方案明列项）
# ===========================================================================
@pytest.mark.parametrize(
    "n,expected_batches,expected_sizes",
    [
        (1, 1, [1]),
        (49, 1, [49]),
        (50, 1, [50]),  # 恰好在上限，不分批
        (51, 2, [50, 1]),  # 越界一条即分两批
        (100, 2, [50, 50]),
        (120, 3, [50, 50, 20]),
    ],
)
def test_works_by_ids_batches_at_the_50_boundary(
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    expected_batches: int,
    expected_sizes: list[int],
) -> None:
    rec = _Recorder([(_is_list, _echo_batch)]).install(monkeypatch)
    ids = [f"W{100000 + i}" for i in range(n)]

    got = oa.works_by_ids(ids)

    assert len(rec.calls) == expected_batches
    assert rec.batch_sizes() == expected_sizes
    assert len(got) == n


def test_works_by_ids_returns_empty_without_any_http_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无可用的合法 id 时**一次请求都不发**。无 key 降级态下 OpenAlex 每天只给
    ~100 credits，一次空 filter 请求就是白烧一份配额（而且会返回 400）。"""
    rec = _Recorder([(_is_list, _echo_batch)]).install(monkeypatch)

    assert oa.works_by_ids([]) == []
    assert oa.works_by_ids(["", None, "not-an-id"]) == []
    assert rec.calls == []


def test_works_by_ids_dedupes_and_skips_illegal_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder([(_is_list, _echo_batch)]).install(monkeypatch)

    got = oa.works_by_ids(
        ["W100001", "https://openalex.org/W100001", "w100001", "10.1/junk", "W100002"]
    )

    # 三种写法的同一个 id + 一个非法 DOI + 另一个 id → 净 2 个
    assert _ids_from(rec.filters[0]) == ["W100001", "W100002"]
    assert len(got) == 2


@pytest.mark.parametrize(
    "batch_size,expected_sizes",
    [
        (10, [10, 10, 5]),  # 显式小批
        (999, [50, 50, 50, 50, 25]),  # 夹到硬上限 50
        (0, [1] * 25),  # 夹到 1
        (-5, [1] * 25),  # 负数同样夹到 1（否则 range step 为负，一条也不发）
    ],
)
def test_works_by_ids_clamps_batch_size(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, expected_sizes: list[int]
) -> None:
    rec = _Recorder([(_is_list, _echo_batch)]).install(monkeypatch)
    ids = [f"W{100000 + i}" for i in range(25) if batch_size != 999] or [
        f"W{100000 + i}" for i in range(225)
    ]

    oa.works_by_ids(ids, batch_size=batch_size)

    assert rec.batch_sizes() == expected_sizes


def test_works_by_ids_sets_per_page_to_the_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``per-page`` 必须等于该批 id 数：OpenAlex 默认 per-page=25，而一批最多 50 个 id，
    不显式抬上去会**静默丢掉后 25 条**——引文网络凭空缺一半，且没有任何报错。"""
    rec = _Recorder([(_is_list, _echo_batch)]).install(monkeypatch)

    oa.works_by_ids([f"W{100000 + i}" for i in range(50)])

    assert rec.calls[0]["params"]["per-page"] == 50


def test_works_by_ids_survives_a_single_batch_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """一批网络失败只跳过该批（留一行 stderr），其余批照常返回。

    不变量 1：降级路径不抛、不阻塞。滚雪球取 120 条参考文献要发 3 批，任一批撞上
    配额耗尽都不该把另外两批的成果一起扔掉。
    """
    state = {"n": 0}

    def _flaky(params: dict[str, Any]) -> dict[str, Any]:
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("quota exhausted")
        return _echo_batch(params)

    rec = _Recorder([(_is_list, _flaky)]).install(monkeypatch)

    got = oa.works_by_ids([f"W{100000 + i}" for i in range(120)])

    assert len(rec.calls) == 3  # 三批都试过了，没有中途放弃
    assert len(got) == 70  # 120 = 50+50+20；第 2 批的 50 条丢了，另两批的 70 条保住
    err = capsys.readouterr().err
    assert "works_by_ids" in err and "第 2 批" in err


def test_works_by_ids_result_length_may_be_shorter_than_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """契约：调用方**不得**假定返回与输入一一对应。

    OpenAlex 查不到的 id（已合并/删除的记录）不会出现在结果里，因此返回列表通常更短。
    ``cmd_citegraph`` 据此打印「缺 K 条」而不是报错。
    """

    def _partial(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [_raw_work(oid="W100000", title="Only Survivor")],
            "meta": {"count": 1},
        }

    _Recorder([(_is_list, _partial)]).install(monkeypatch)

    got = oa.works_by_ids(["W100000", "W100001", "W100002"])

    assert len(got) == 1
    assert got[0]["title"] == "Only Survivor"


# ===========================================================================
# works_citing（forward 方向）
# ===========================================================================
def test_works_citing_uses_the_cites_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder([(_is_list, _empty_page)]).install(monkeypatch)

    oa.works_citing("W2789790776")

    assert rec.filters == ["cites:W2789790776"]


def test_works_citing_year_window_wraps_the_bounds_outward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--year 2020-2026`` 必须译成 ``>2019`` 与 ``<2027``：OpenAlex 的 ``>`` / ``<``
    是**严格**不等，直接写 ``>2020,<2026`` 会把 2020 与 2026 两年的文献全部漏掉——
    而用户给的区间是闭区间。"""
    rec = _Recorder([(_is_list, _empty_page)]).install(monkeypatch)

    oa.works_citing("W2789790776", year_from=2020, year_to=2026)

    assert rec.filters == [
        "cites:W2789790776,publication_year:>2019,publication_year:<2027"
    ]


def test_works_citing_without_a_year_window_omits_the_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder([(_is_list, _empty_page)]).install(monkeypatch)

    oa.works_citing("W2789790776")

    assert "publication_year" not in rec.filters[0]


@pytest.mark.parametrize(
    "per_page,expected", [(0, 1), (-3, 1), (25, 25), (200, 200), (999, 200)]
)
def test_works_citing_clamps_per_page(
    monkeypatch: pytest.MonkeyPatch, per_page: int, expected: int
) -> None:
    rec = _Recorder([(_is_list, _empty_page)]).install(monkeypatch)

    oa.works_citing("W2789790776", per_page=per_page)

    assert rec.calls[0]["params"]["per-page"] == expected


def test_works_citing_passes_sort_through(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder([(_is_list, _empty_page)]).install(monkeypatch)

    oa.works_citing("W2789790776", sort="publication_date:desc")

    assert rec.calls[0]["params"]["sort"] == "publication_date:desc"


@pytest.mark.parametrize("bad", ["", "10.1103/x", "2401.12345", "W123"])
def test_works_citing_rejects_a_non_openalex_id(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """id 非法时**抛 ValueError** 而不是静默返回空集：这是调用方的输入错误而非环境
    故障，返回空集会被读成「无人引用这篇论文」——一个完全相反的结论。

    ``cites:`` 过滤只能用 OpenAlex id；``cmd_citegraph`` 因此对 ``--direction forward``
    在解析不出 id 时返回退出码 2。
    """
    rec = _Recorder([(_is_list, _empty_page)]).install(monkeypatch)

    with pytest.raises(ValueError, match="OpenAlex work id"):
        oa.works_citing(bad)
    assert rec.calls == []


def test_works_citing_degrades_to_an_empty_result_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """网络故障与「id 非法」相反：那是环境问题，必须静默降级（不变量 1），
    返回与 :func:`search_works` 同形状的空结果，让上层照常走完。"""

    def _boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("connection reset")

    _Recorder([(_is_list, _boom)]).install(monkeypatch)

    res = oa.works_citing("W2789790776")

    assert res == {"results": [], "meta": {"count": 0}, "next_cursor": None}
    assert "works_citing" in capsys.readouterr().err


def test_works_citing_returns_meta_count_and_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``meta.count`` 是**全部**前向引用的总数，而不是本次取回的条数——``cmd_citegraph``
    靠两者的差值提示「还有 N 条未取，调高 --limit」。丢了它，用户无从知道自己漏了多少。"""
    payload = {
        "results": [_raw_work(oid="W9000001", title="Citing Paper")],
        "meta": {
            "count": 37,
            "db_response_time_ms": 12,
            "page": None,
            "per_page": 1,
            "next_cursor": "abc123",
        },
    }
    _Recorder([(_is_list, lambda p: payload)]).install(monkeypatch)

    res = oa.works_citing("W2789790776", per_page=1)

    assert res["meta"]["count"] == 37
    assert res["next_cursor"] == "abc123"
    assert len(res["results"]) == 1
    assert res["results"][0]["title"] == "Citing Paper"


# ===========================================================================
# _extract_work_summary：max_refs
# ===========================================================================
@pytest.mark.parametrize(
    "max_refs,expected_kept",
    [
        (None, 100),  # 显式 None 回落默认
        (100, 100),
        (20, 20),
        (0, 0),
        (-5, 0),  # 负数必须归一为 0
    ],
)
def test_extract_work_summary_caps_referenced_works(
    max_refs: int | None, expected_kept: int
) -> None:
    """默认 100 而非旧值 20：一篇 PRL 级论文的参考文献表常有 40-60 条，20 条上限会
    静默丢掉一半以上的引文网络，而 ``citegraph --direction backward`` 正是要拿它们
    去批量取回被引工作。

    负数归一为 0 是必须的：Python 的 ``lst[:-5]`` 是「去掉末尾 5 条」而不是「不保留」，
    于是 ``max_refs=-5`` 会留下 115 条——一个比默认值还大的数，完全违背调用方意图。
    """
    refs = tuple(f"W{200000 + i}" for i in range(120))
    w = oa._extract_work_summary(_raw_work(refs=refs), max_refs=max_refs)

    assert len(w["referenced_works"]) == expected_kept
    # 总数永远是**未截断**的真实值，否则「缺 K 条」的提示会算错
    assert w["referenced_works_count"] == 120


def test_extract_work_summary_default_max_refs_is_100() -> None:
    refs = tuple(f"W{200000 + i}" for i in range(120))
    w = oa._extract_work_summary(_raw_work(refs=refs))

    assert len(w["referenced_works"]) == 100
    assert w["referenced_works_count"] == 120


def test_extract_work_summary_strips_the_openalex_url_prefix() -> None:
    w = oa._extract_work_summary(_raw_work(refs=("W200001",)))

    assert w["openalex_id"] == "W1000001"
    assert w["referenced_works"] == ["W200001"]
    assert w["doi"] == "10.1103/physrevlett.121.124501"


# ===========================================================================
# _extract_work_summary：内联 listed_in → journal_tier（零额外请求）
# ===========================================================================
def test_extract_work_summary_derives_tier_from_the_inline_source() -> None:
    """期刊档次来自 work 响应**内联**的 ``primary_location.source.listed_in``，
    不需要额外调 ``get_source``。于是 search / citegraph / get 三条不查 source 的路径
    也带档次；否则展示层的 tier 尾注恒空，WP-E 的 E1 等于没落地。"""
    w = oa._extract_work_summary(_raw_work())

    assert w["listed_in"] == _REAL_INLINE_LISTED_IN
    assert w["journal_tier"] == "top"  # jufo-3 与 norway-2 都判顶级
    assert "jufo-3" in w["journal_tier_basis"]
    assert "norway-2" in w["journal_tier_basis"]


@pytest.mark.parametrize(
    "listed_in,expected_tier",
    [
        (["cwts-core", "jufo-1", "norway-1"], "basic"),  # 只有基础档
        (["jufo-2"], "leading"),
        (["cwts-core", "medline", "erih-plus"], ""),  # 都不在 TIER_LISTS 里
        ([], ""),
    ],
)
def test_extract_work_summary_tier_tracks_the_inline_lists(
    listed_in: list[str], expected_tier: str
) -> None:
    w = oa._extract_work_summary(_raw_work(listed_in=listed_in))

    assert w["journal_tier"] == expected_tier


def test_extract_work_summary_survives_a_missing_primary_location() -> None:
    """预印本常没有 ``primary_location.source``（或整个 ``primary_location`` 为 null）。
    那只是「档次未知」，不是错误——三个字段全部空着即可，不得抛。"""
    for raw in (_raw_work(inline_source=False), {"display_name": "Bare", "id": ""}):
        w = oa._extract_work_summary(raw)

        assert w["listed_in"] == []
        assert w["journal_tier"] == ""
        assert w["journal_tier_basis"] == []
        assert w["journal_issn"] == []


def test_extract_work_summary_exposes_the_inline_issn_list() -> None:
    """``journal_issn`` 是内联 source 的 ISSN **列表**（PRL 为 ``0031-9007;1079-7114``）。
    ``work_to_note_frontmatter`` 用它作为 SCImago 查询的回落源，于是 ``get_source``
    失败时分区仍能填上。"""
    w = oa._extract_work_summary(_raw_work())

    assert w["journal_issn"] == ["0031-9007", "1079-7114"]
    assert w["journal_issn_l"] == "0031-9007"


# ===========================================================================
# work_to_note_frontmatter：回落到内联字段
# ===========================================================================
def test_note_frontmatter_falls_back_to_the_inline_listed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_source`` 返回 None 时（无 key 降级、配额耗尽、期刊未被收录），档次与
    SCImago 分区仍从 work dict 的内联字段派生。

    降级态下这两个字段静默全空是最坏的结果：那恰恰是最需要免费指标层的时刻，
    而用户看不到任何提示，只会以为「这本期刊没有档次信息」。
    """
    monkeypatch.setattr(oa, "get_source", lambda **kw: None)
    seen: list[Any] = []
    monkeypatch.setattr(
        oa.journal_metrics, "quartile_for", lambda issn: seen.append(issn) or "Q1"
    )

    fm = oa.work_to_note_frontmatter(oa._extract_work_summary(_raw_work()))

    assert fm["journal_tier"] == "top"
    assert "jufo-3" in fm["journal_tier_basis"]
    assert fm["listed_in"] == _REAL_INLINE_LISTED_IN
    assert fm["scimago_quartile"] == "Q1"
    assert seen == [["0031-9007", "1079-7114"]]  # 用内联 ISSN 查的
    # jif 与 h_index 只能来自 get_source，降级态下如实留空
    assert fm["jif"] is None
    assert fm["journal_h_index"] is None


def test_note_frontmatter_prefers_get_source_over_the_inline_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_source`` 的 ``listed_in`` 优先：它同一次调用还给了 ``h_index`` 与
    ``2yr_mean_citedness``，是更权威也更完整的一份。"""
    monkeypatch.setattr(
        oa,
        "get_source",
        lambda **kw: {
            "listed_in": ["jufo-1"],  # 与内联的 jufo-3 冲突，应取这一份
            "issn": ["0031-9007"],
            "h_index": 982,
            "2yr_mean_citedness": 8.97,
        },
    )

    fm = oa.work_to_note_frontmatter(oa._extract_work_summary(_raw_work()))

    assert fm["journal_tier"] == "basic"
    assert fm["journal_tier_basis"] == ["jufo-1"]
    assert fm["listed_in"] == ["jufo-1"]
    assert fm["jif"] == 8.97
    assert fm["journal_h_index"] == 982


def test_note_frontmatter_tier_and_basis_stay_self_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tier / basis / listed_in 三者必须来自**同一份**名单：``work_to_note_frontmatter``
    因此重算 :func:`derive_journal_tier` 而不是直接取 ``w["journal_tier"]``。若一个来自
    ``get_source``、另一个来自内联，笔记里就会出现「tier=basic 但 basis=jufo-3」这种
    自相矛盾、无法审计的记录。"""
    monkeypatch.setattr(
        oa, "get_source", lambda **kw: {"listed_in": ["jufo-2"], "issn": []}
    )

    fm = oa.work_to_note_frontmatter(oa._extract_work_summary(_raw_work()))

    from pysci.skills.literature_research.tools.notes import derive_journal_tier

    assert (fm["journal_tier"], fm["journal_tier_basis"]) == derive_journal_tier(
        fm["listed_in"]
    )


# ===========================================================================
# _row()：投影（方案明列项）
# ===========================================================================
def test_row_projects_a_cited_openalex_work() -> None:
    """被引工作经 ``_extract_work_summary`` → ``_row`` 的完整投影。"""
    summary = oa._extract_work_summary(_raw_work(title="Cited Work", cited=42))

    r = research._row("openalex", summary)

    assert r == {
        "title": "Cited Work",
        "first_author_last_name": "zhu",
        "year": 2020,
        "journal": "Physical Review Letters",
        "doi": "10.1103/physrevlett.121.124501",
        "arxiv_id": "",
        "openalex_id": "W1000001",
        "cited_by_count": 42,
        "oa_status": "green",
        "jif": None,  # jif 只来自 get_source，检索结果里没有
        "jcr_quartile": "",  # 需 WoS Journals API（未接入）
        "journal_tier": "top",  # ← 内联 listed_in 派生，这一维过去恒空
        "source": "openalex",
    }


def test_row_does_not_override_an_explicit_journal_tier() -> None:
    """work dict 已带 ``journal_tier`` 时**不再**从 ``listed_in`` 重算：调用方可能刚用
    ``get_source`` 的权威名单覆盖过它。"""
    w = {"title": "T", "journal_tier": "leading", "listed_in": ["jufo-3"]}

    assert research._row("openalex", w)["journal_tier"] == "leading"


def test_row_derives_tier_from_listed_in_when_tier_is_absent() -> None:
    """回退分支：给的是**原始** source 数据（未经 ``_extract_work_summary``）时，
    展示层就地派生，使 tier 这一维不比 frontmatter 少。"""
    w = {"title": "T", "listed_in": ["jufo-1", "norway-1"]}

    assert research._row("openalex", w)["journal_tier"] == "basic"


def test_row_tolerates_an_empty_work_dict() -> None:
    """防御式取值：任一源给出畸形/缺字段的记录都不该让整份清单崩掉。"""
    r = research._row("openalex", {})

    assert r["title"] == ""
    assert r["year"] is None
    assert r["journal"] == ""
    assert r["cited_by_count"] is None
    assert r["journal_tier"] == ""


def test_row_parses_an_arxiv_journal_ref_into_a_journal_name() -> None:
    """arXiv 的 work dict 只有自由文本 ``journal_ref``（一整串引文）。直接当刊名展示会
    把 Journal 列变成 ``Phys. Rev. Lett. 121, 124501 (2018)``，与 frontmatter 里已修好的
    ``journal`` 字段不一致，所以这里走同一个解析器。"""
    w = {"title": "T", "journal_ref": "Phys. Rev. Lett. 121, 124501 (2018)"}

    r = research._row("arxiv", w)

    assert r["journal"] == "Phys. Rev. Lett."


def test_row_falls_back_to_the_literal_arxiv_label() -> None:
    """连 ``journal_ref`` 都没有的纯预印本，Journal 列显 ``arXiv`` 而不是空——空会被
    读成「元数据缺失」，而这其实是一条完整、正确的信息。"""
    assert research._row("arxiv", {"title": "T"})["journal"] == "arXiv"
    assert research._row("openalex", {"title": "T"})["journal"] == ""


def test_row_defaults_oa_status_to_green_for_arxiv() -> None:
    """arXiv 上的一切都是绿色 OA；Atom 响应里没有 ``oa_status`` 字段，故就地补默认。"""
    assert research._row("arxiv", {"title": "T"})["oa_status"] == "green"
    assert research._row("openalex", {"title": "T"})["oa_status"] == ""


def test_row_reads_year_from_whichever_source_spelled_it() -> None:
    """三个源用三种键名给年份（OpenAlex ``publication_year``、arXiv ``published`` 的
    前四位、WoS ``pub_year``），投影必须都认。"""
    assert research._row("openalex", {"publication_year": 2021})["year"] == 2021
    assert research._row("arxiv", {"year": 2022})["year"] == 2022
    assert research._row("wos", {"pub_year": "2023"})["year"] == "2023"
    assert (
        research._row("arxiv", {"published": "2024-05-06T07:08:09Z"})["year"] == "2024"
    )


def test_print_rows_shows_the_tier_tail_note(capsys: pytest.CaptureFixture) -> None:
    """端到端证明 WP-E E1 落地：``jcr_quartile`` 恒空（需 WoS Journals API），尾注回退
    到 ``journal_tier`` 后才**真的**打印出东西。改之前这个分支永远不触发，等于死代码。"""
    summary = oa._extract_work_summary(_raw_work())

    research._print_rows([("openalex", summary)])

    out = capsys.readouterr().out
    assert "tier:top" in out
    assert "Physical Review Letters" in out


def test_print_rows_prefers_jcr_quartile_over_tier(
    capsys: pytest.CaptureFixture,
) -> None:
    """将来 WoS Journals API 接入后 ``jcr_quartile`` 会有值，那时它优先（官方分区比
    专家评议名单更贴合「期刊档次」的通俗含义）。"""
    research._print_rows(
        [("openalex", {"title": "T", "jcr_quartile": "Q1", "journal_tier": "top"})]
    )

    out = capsys.readouterr().out
    assert "Q1" in out
    assert "tier:top" not in out


def test_render_candidates_carries_the_tier_too() -> None:
    """shortlist 快照走的是 ``_render_candidates`` 而不是 ``_print_rows``，两处必须一致：
    快照是**可复现的落盘产物**，AI 后续初筛分级全靠它，比一次性的终端输出更重要。"""
    text = research._render_candidates(
        [("openalex", oa._extract_work_summary(_raw_work()))]
    )

    assert "tier:top" in text
    assert "Physical Review Letters" in text


# ===========================================================================
# _sort_rows
# ===========================================================================
def _rows(*pairs: tuple[str, int | None]) -> list[tuple[str, dict]]:
    return [("openalex", {"title": t, "cited_by_count": c}) for t, c in pairs]


def test_sort_rows_by_citations_descending() -> None:
    got = research._sort_rows(_rows(("a", 1), ("b", 99), ("c", 50)), "citations")

    assert [w["title"] for _, w in got] == ["b", "c", "a"]


def test_sort_rows_by_date_descending() -> None:
    rows = [
        ("openalex", {"title": "old", "publication_year": 2010}),
        ("openalex", {"title": "new", "publication_year": 2024}),
        ("arxiv", {"title": "mid", "year": 2018}),
    ]

    got = research._sort_rows(rows, "date")

    assert [w["title"] for _, w in got] == ["new", "mid", "old"]


def test_sort_rows_leaves_relevance_untouched() -> None:
    """``relevance`` 是服务端排的（OpenAlex 的 ``relevance_score:desc``），客户端无从
    重算，故原样返回——返回的甚至是**同一个 list 对象**，不做无谓拷贝。"""
    rows = _rows(("a", 1), ("b", 99))

    assert research._sort_rows(rows, "relevance") is rows


def test_sort_rows_treats_a_missing_citation_count_as_zero() -> None:
    """``cited_by_count`` 为 None（源没给）时当 0 排到末尾，而不是让 ``sorted`` 撞上
    ``None < int`` 的 TypeError 把整份清单崩掉。"""
    got = research._sort_rows(_rows(("a", None), ("b", 5), ("c", None)), "citations")

    assert [w["title"] for _, w in got][0] == "b"


def test_sort_rows_tolerates_a_garbage_year() -> None:
    rows = [
        ("openalex", {"title": "bad", "publication_year": "n.d."}),
        ("openalex", {"title": "good", "publication_year": 2020}),
    ]

    got = research._sort_rows(rows, "date")

    assert [w["title"] for _, w in got] == ["good", "bad"]


# ===========================================================================
# _parse_year
# ===========================================================================
@pytest.mark.parametrize(
    "spec,expected",
    [
        ("2023-2026", (2023, 2026)),
        ("2023-", (2023, None)),
        ("2023", (2023, 2023)),
        (" 2023 - 2026 ", (2023, 2026)),  # 容忍空白
        ("", (None, None)),
        (None, (None, None)),
        ("last five years", (None, None)),  # 解析不出即不限，不报错
        ("23-26", (None, None)),
    ],
)
def test_parse_year(spec: str | None, expected: tuple[int | None, int | None]) -> None:
    assert research._parse_year(spec) == expected


# ===========================================================================
# _resolve_openalex_work
# ===========================================================================
def test_resolve_openalex_work_maps_an_arxiv_id_to_its_datacite_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """arXiv 的 Atom 响应里**没有**引文网络，所以 citegraph 一律走 OpenAlex；而 arXiv id
    在 OpenAlex 里要以 DataCite 为其自动登记的 DOI 形态 ``10.48550/arXiv.<id>`` 去查。"""
    rec = _Recorder([(_is_single, _single(_raw_work(oid="W7777777")))]).install(
        monkeypatch
    )

    kind, work = research._resolve_openalex_work("2401.12345")

    assert kind == "arxiv"
    assert work is not None and work["openalex_id"] == "W7777777"
    assert "10.48550" in rec.single_work_calls()[0]
    assert "2401.12345" in rec.single_work_calls()[0]


def test_resolve_openalex_work_uses_the_id_endpoint_for_an_openalex_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder([(_is_single, _single(_raw_work(oid="W2789790776")))]).install(
        monkeypatch
    )

    kind, work = research._resolve_openalex_work("W2789790776")

    assert kind == "openalex"
    assert work is not None
    assert rec.single_work_calls()[0].endswith("/works/W2789790776")


def test_resolve_openalex_work_returns_none_for_a_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一个裸 URL 既不是 DOI 也不是 id，OpenAlex 无从查起。**零请求**返回 None，
    由 ``cmd_citegraph`` 给出可操作的错误信息（退出码 2）。"""
    rec = _Recorder([]).install(monkeypatch)

    kind, work = research._resolve_openalex_work("https://example.com/paper.pdf")

    assert (kind, work) == ("url", None)
    assert rec.calls == []


def test_resolve_openalex_work_reports_an_http_failure_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """HTTP 故障在 :func:`openalex_client.get_work` 内部就被吞掉（返回 None），于是
    本函数走的是**正常返回** ``(kind, None)`` 而不是自己的 except 分支。

    提示必须落在 **stderr**：``citegraph --json`` 把 JSON 写在 stdout，一行文字混进去
    会让调用方的 ``json.loads(stdout)`` 直接失败——而那正是 ``--json`` 唯一的用途。
    """

    def _boom(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("503 Service Unavailable")

    _Recorder([(_is_single, _boom)]).install(monkeypatch)

    kind, work = research._resolve_openalex_work("10.1103/x")

    assert (kind, work) == ("doi", None)
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout 保持干净，供 --json 使用
    assert "get_work failed" in captured.err and "503" in captured.err


def test_resolve_openalex_work_swallows_errors_from_the_extractor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``get_work`` **自己**抛异常时（例如畸形响应让 ``_extract_work_summary`` 撞上
    AttributeError），本函数的 except 分支接手：留一行提示、返回 ``(kind, None)``，不抛。

    那条崩溃路径并非想象：``_extract_work_summary`` 曾在 ``primary_location.source``
    为 null 的预印本上抛 AttributeError（见其 publisher 行的注释）。解析失败不该把
    整个命令带下去，而应落到 ``cmd_citegraph`` 那个可操作的退出码 2 分支。
    """

    def _raise(**kw: Any) -> None:
        raise AttributeError("'NoneType' object has no attribute 'get'")

    monkeypatch.setattr(oa, "get_work", _raise)

    kind, work = research._resolve_openalex_work("2401.12345")

    assert (kind, work) == ("arxiv", None)
    err = capsys.readouterr().err
    assert "OpenAlex 解析失败" in err and "AttributeError" in err


# ===========================================================================
# cmd_citegraph：端到端
# ===========================================================================
def _install_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed_refs: tuple[str, ...] = ("W200001", "W200002"),
    citing: tuple[dict, ...] = (),
    citing_total: int | None = None,
    seed: dict[str, Any] | None = None,
) -> _Recorder:
    """装一套完整的引文图谱桩：一篇种子论文 + 它引的 + 引它的。"""
    seed_work = _raw_work(
        oid="W1000001",
        title="The Seed Paper",
        refs=seed_refs,
        **(seed or {}),
    )
    citing_raws = list(citing) or []
    total = len(citing_raws) if citing_total is None else citing_total

    def _list_handler(params: dict[str, Any]) -> dict[str, Any]:
        f = str(params.get("filter", ""))
        if f.startswith("cites:"):
            return {
                "results": citing_raws,
                "meta": {"count": total, "per_page": params.get("per-page")},
            }
        return _echo_batch(params)

    return _Recorder(
        [(_is_single, _single(seed_work)), (_is_list, _list_handler)]
    ).install(monkeypatch)


def test_citegraph_requires_an_id() -> None:
    """缺参数是**用法错**（退出码 2），不是「检查后发现问题」（那是 1）。"""
    assert research.cmd_citegraph(_ns(id="")) == 2
    assert research.cmd_citegraph(_ns(id="   ")) == 2


def test_citegraph_exits_2_when_openalex_cannot_resolve_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """OpenAlex 查不到该记录 → 2，且错误信息要给出可操作的下一步。"""
    monkeypatch.setattr(oa, "get_work", lambda **kw: None)

    assert research.cmd_citegraph(_ns(id="10.9999/nope")) == 2
    err = capsys.readouterr().err
    assert "无法从 OpenAlex 解析" in err
    assert "research get" in err  # 指路：先确认该记录是否被收录


def test_citegraph_backward_only_lists_the_references(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rec = _install_graph(monkeypatch, seed_refs=("W200001", "W200002"))

    assert research.cmd_citegraph(_ns(direction="backward")) == 0

    out = capsys.readouterr().out
    assert "Ref W200001" in out and "Ref W200002" in out
    assert len(rec.calls) == 2  # 1 次种子解析（/works/doi:…）+ 1 次批量取回
    assert len(rec.single_work_calls()) == 1
    # 最后那次是批量取回：filter 里就是种子的两个 referenced_works，且顺序未被打乱
    assert _ids_from(rec.filters[-1]) == ["W200001", "W200002"]


def test_citegraph_reports_an_empty_backward_branch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """OpenAlex 未记录参考文献时留一行提示、退出码仍是 0：这是**正常的空结果**
    （很多新论文与部分期刊确实没有解析出的引文表），不是失败。"""
    _install_graph(monkeypatch, seed_refs=())

    assert research.cmd_citegraph(_ns(direction="backward")) == 0

    captured = capsys.readouterr()
    assert "未记录它的参考文献" in captured.err


def test_citegraph_backward_reports_how_many_ids_were_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``referenced_works_count`` 是未截断的真实总数，而 ``works_by_ids`` 通常取回更少
    （已合并/未收录的记录不会返回）。差值必须报出来，否则「引文网络缺一块」是静默的。"""

    def _list_handler(params: dict[str, Any]) -> dict[str, Any]:
        ids = _ids_from(str(params.get("filter", "")))
        return {
            "results": [_raw_work(oid=ids[0], title="Only One")],
            "meta": {"count": 1},
        }

    monkeypatch.setattr(
        oa,
        "get_work",
        lambda **kw: oa._extract_work_summary(
            _raw_work(refs=("W200001", "W200002", "W200003"))
        ),
    )
    monkeypatch.setattr(oa, "_get", _Recorder([(_is_list, _list_handler)]))

    assert research.cmd_citegraph(_ns(direction="backward")) == 0
    err = capsys.readouterr().err
    assert "参考文献 3 条" in err and "取回 1 条" in err and "缺 2 条" in err


def test_citegraph_forward_exits_2_without_an_openalex_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``cites:`` 过滤**只能**用 OpenAlex id。解析不出 id 时 forward 无法进行 → 2
    （用法/前提不成立），而不是静默给出空清单让用户以为「无人引用」。"""
    monkeypatch.setattr(
        oa,
        "get_work",
        lambda **kw: {"title": "No Id", "openalex_id": "", "referenced_works": []},
    )

    assert research.cmd_citegraph(_ns(direction="forward")) == 2
    assert "forward 方向需要 OpenAlex id" in capsys.readouterr().err


def test_citegraph_both_tolerates_a_missing_openalex_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """但 ``both`` 时 forward 只是**一个分支**：跳过它、照常给 backward，退出码 0。
    与 forward-only 的区别是有意的——用户要的是「尽量多拿」，不是「全有或全无」。"""
    monkeypatch.setattr(
        oa,
        "get_work",
        lambda **kw: {
            "title": "No Id",
            "openalex_id": "",
            "referenced_works": ["W200001"],
            "referenced_works_count": 1,
        },
    )
    _Recorder([(_is_list, _echo_batch)]).install(monkeypatch)

    assert research.cmd_citegraph(_ns(direction="both")) == 0
    captured = capsys.readouterr()
    assert "forward 方向需要 OpenAlex id" in captured.err
    assert "Ref W200001" in captured.out


def test_citegraph_forward_requests_per_page_equal_to_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forward 由**服务端**按排序取 top-N：拉回 200 条只为了显示 25 条既慢又白烧配额。"""
    rec = _install_graph(monkeypatch, seed_refs=(), citing=(_raw_work(oid="W9000001"),))

    assert research.cmd_citegraph(_ns(direction="forward", limit=7)) == 0

    fwd_call = next(
        c for c in rec.calls if c["params"].get("filter", "").startswith("cites:")
    )
    assert fwd_call["params"]["per-page"] == 7


def test_citegraph_forward_uses_a_generous_page_when_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--limit 0`` = 不限：此时给 OpenAlex 的单页上限 200，而不是退化成默认 25。"""
    rec = _install_graph(monkeypatch, seed_refs=(), citing=(_raw_work(oid="W9000001"),))

    assert research.cmd_citegraph(_ns(direction="forward", limit=0)) == 0

    fwd_call = next(
        c for c in rec.calls if c["params"].get("filter", "").startswith("cites:")
    )
    assert fwd_call["params"]["per-page"] == 200


def test_citegraph_passes_the_year_window_and_sort_to_openalex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _install_graph(monkeypatch, seed_refs=(), citing=(_raw_work(oid="W9000001"),))

    research.cmd_citegraph(_ns(direction="forward", year="2020-2026", sort="date"))

    fwd_call = next(
        c for c in rec.calls if c["params"].get("filter", "").startswith("cites:")
    )
    assert "publication_year:>2019" in fwd_call["params"]["filter"]
    assert "publication_year:<2027" in fwd_call["params"]["filter"]
    assert fwd_call["params"]["sort"] == "publication_date:desc"


def test_citegraph_both_dedupes_with_backward_winning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``both`` 时去掉两个方向的重叠项（OpenAlex 的引文数据有少量双向脏记录），
    且**保留 backward 那一条**：backward 先入，重叠意味着同一篇，谁先谁留。

    不去重的后果是同一篇在两个分区里各出现一次，而两个分区在滚雪球里的下一步
    动作完全不同（「它引的」要往回追源头，「引它的」要往前追跟进），一条重复会把
    判断带偏。
    """
    dup_doi = "10.1103/duplicated"
    seed = _raw_work(oid="W1000001", title="The Seed Paper", refs=("W200001",))

    def _list(params: dict[str, Any]) -> dict[str, Any]:
        if str(params.get("filter", "")).startswith("cites:"):
            # forward 里既有与 backward 重叠的那一篇，也有一篇它独有的
            return {
                "results": [
                    _raw_work(oid="W9000001", title="The Duplicate", doi=dup_doi),
                    _raw_work(
                        oid="W9000002", title="Forward Only", doi="10.1103/fwd-only"
                    ),
                ],
                "meta": {"count": 2},
            }
        return {
            "results": [_raw_work(oid="W200001", title="The Duplicate", doi=dup_doi)],
            "meta": {"count": 1},
        }

    _Recorder([(_is_single, _single(seed)), (_is_list, _list)]).install(monkeypatch)

    assert research.cmd_citegraph(_ns(direction="both")) == 0

    out = capsys.readouterr().out
    assert out.count("The Duplicate") == 1  # 两个分区加起来只出现一次
    assert "Forward Only" in out
    # 且保留的是 backward 那一条：它属于 backward 分区
    _, fwd_part = out.split("── forward")
    back_part = out.split("── forward")[0]
    assert "The Duplicate" in back_part
    assert "The Duplicate" not in fwd_part


def test_citegraph_sorts_before_truncating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``--sort citations --limit 1`` 必须给出「被引最高的那 1 条」而不是「随便 50 条里
    挑最高的」：所以 backward 先取全量、客户端排序、**最后**才截断。"""
    _install_graph(
        monkeypatch,
        seed_refs=("W200001", "W200002", "W200003"),
    )
    monkeypatch.setattr(
        oa,
        "_get",
        _Recorder(
            [
                (
                    _is_single,
                    _single(
                        _raw_work(
                            oid="W1000001",
                            title="Seed",
                            refs=("W200001", "W200002", "W200003"),
                        )
                    ),
                ),
                (
                    _is_list,
                    lambda p: {
                        "results": [
                            _raw_work(oid="W200001", title="Low", cited=1),
                            _raw_work(oid="W200002", title="High", cited=999),
                            _raw_work(oid="W200003", title="Mid", cited=50),
                        ],
                        "meta": {"count": 3},
                    },
                ),
            ]
        ),
    )

    assert research.cmd_citegraph(_ns(direction="backward", limit=1)) == 0

    out = capsys.readouterr().out
    assert "High" in out
    assert "Low" not in out and "Mid" not in out


def test_citegraph_returns_0_when_both_directions_are_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """两个方向都空 → 0 + 一行提示。空结果是正常状态，不该在 CI 里被当成失败。"""
    _install_graph(monkeypatch, seed_refs=(), citing=(), citing_total=0)

    assert research.cmd_citegraph(_ns(direction="both")) == 0
    assert "两个方向都无结果" in capsys.readouterr().err


def test_citegraph_json_keeps_the_two_directions_apart(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``--json`` 把两个方向分开而不是混成一个大数组：一篇文献是「它引的」还是「引它的」，
    对滚雪球的下一步决策完全不同，混起来就丢了最有用的那一维。"""
    _install_graph(
        monkeypatch,
        seed_refs=("W200001",),
        citing=(_raw_work(oid="W9000001", title="Citing It", doi="10.1103/citing"),),
        citing_total=13,
    )

    assert research.cmd_citegraph(_ns(direction="both", json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "10.1103/physrevlett.121.124501"
    assert payload["openalex_id"] == "W1000001"
    assert payload["title"] == "The Seed Paper"
    assert payload["backward"]["referenced_works_count"] == 1
    assert payload["backward"]["resolved_count"] == 1
    assert [r["title"] for r in payload["backward"]["rows"]] == ["Ref W200001"]
    assert payload["forward"]["total_citing"] == 13
    assert payload["forward"]["returned_count"] == 1
    assert [r["title"] for r in payload["forward"]["rows"]] == ["Citing It"]
    # rows 里是 _row() 的投影，因此带 journal_tier 这一维
    assert "journal_tier" in payload["backward"]["rows"][0]


def test_citegraph_json_reports_the_untruncated_forward_total(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """``total_citing`` 取 ``meta.count``（全部前向引用数），``returned_count`` 取本次
    实际条数。两者的差就是用户还没看到的部分。"""
    _install_graph(
        monkeypatch,
        seed_refs=(),
        citing=(_raw_work(oid="W9000001", title="One Of Many"),),
        citing_total=250,
    )

    research.cmd_citegraph(_ns(direction="forward", json=True, limit=1))

    payload = json.loads(capsys.readouterr().out)
    assert payload["forward"] == {
        "total_citing": 250,
        "returned_count": 1,
        "rows": payload["forward"]["rows"],
    }


def test_citegraph_save_writes_a_snapshot_named_by_the_openalex_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """``--save`` 让滚雪球结果与 ``search`` 快照同样可复现。文件名用 **openalex_id**
    而不是原始标识：后者可能是长 DOI，而 ``slugify`` 截到 40 字，两篇同期刊同卷的论文
    会撞出同一个快照名（于是第二次滚雪球静默覆盖第一次的成果）。"""
    outdir = tmp_path / "shortlists"
    monkeypatch.setattr(research, "SHORTLISTS_DIR", outdir)
    _install_graph(monkeypatch, seed_refs=("W200001",), citing=())

    assert research.cmd_citegraph(_ns(direction="backward", save=True)) == 0

    files = list(outdir.glob("*.md"))
    assert len(files) == 1
    assert "citegraph-backward-w1000001" in files[0].name
    text = files[0].read_text(encoding="utf-8")
    assert "Ref W200001" in text  # 候选清单落了盘
    assert "机器检索结果" in text
    assert "query_string: citegraph-backward-W1000001" in text
    assert "已保存快照" in capsys.readouterr().out


def test_citegraph_lets_a_failing_save_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--save`` 落盘失败时**让异常冒泡**，不吞。

    与 ``search --save``（走同一个 :func:`research._write_shortlist`）的行为一致。磁盘写
    不进去（只读挂载 / 满盘 / 权限）是严重的环境故障：吞掉它并返回 0 会让用户以为
    快照已落盘，而那份「可复现」的证据其实并不存在——比当场报错更坏。注意图谱本身
    已经打在 stdout 上了，所以冒泡不会让用户丢掉数据。
    """
    monkeypatch.setattr(research, "SHORTLISTS_DIR", tmp_path / "nope")
    _install_graph(monkeypatch, seed_refs=("W200001",), citing=())

    def _boom(query: str, args: Any, rows: Any) -> Path:
        raise OSError("read-only file system")

    monkeypatch.setattr(research, "_write_shortlist", _boom)

    with pytest.raises(OSError, match="read-only"):
        research.cmd_citegraph(_ns(direction="backward", save=True))
