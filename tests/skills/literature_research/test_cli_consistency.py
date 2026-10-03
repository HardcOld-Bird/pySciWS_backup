"""``research`` CLI 的一致性钉子（方案 WP-H）。

这个 CLI 是 skill 文档承诺给 LLM 的**唯一**入口，而 LLM 读不到实现——它只能读 ``-h``
与文档。因此 CLI 的自洽性本身就是功能：一个词的四种含义、一个能力只在某几个子命令上
存在、一句「返回空」分不清「真没有」和「配额耗尽」，都会让调用方做出错误的下一步决策，
且**无从察觉**。

本文件按 WP-H 的分项组织：

- **H1** ``--force`` 的四重语义拆分（``read``/``ingest`` → ``--refresh``，``add`` →
  ``--allow-fail``，``index`` 保留）。拆名的全部意义在于 ``add --force`` 开的是**引用
  核验安全门**，与「绕过缓存重抓」是完全不同的两件事，却曾共用一个词。
- **H2** ``search --json``：``get`` / ``citecheck`` / ``rag`` / ``citegraph`` 都有 ``--json``，
  唯独 ``search`` 没有——而它恰恰是 LLM 最需要程序化筛选的那一步。
- **H3** 参数形态收敛为两种（动词+目标 / 名词+action），并把判据写进 ``-h`` 的 epilog。
  ``ingest`` 是唯一不合形的（用 ``--status`` 布尔充当 action），补上位置参数。
- **H4** ``library list`` 的诚实性：它不是真枚举（上游无「列全部」命令，退化为空关键词
  检索且 limit 被夹到 ≤100），不说的话「只回了 25 条」会被读成「库里就 25 条」。
  连带修掉一个真 bug：``list`` / ``search`` 失败时抛未处理 traceback（``zotero_cli``
  只在 ``ping`` / ``get_item`` 里自己吞异常），而 Zotero 没开是常态。
- **H6** OpenAlex 降级态下的空结果告警：无 key 时限 ~100 credits/天，配额耗尽与「真没
  文献」在返回值上完全同形，静默返回空集会让调用方把前者当后者写进综述。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

import pytest

from pysci.skills.literature_research.tools import research

# ---------------------------------------------------------------------------
# 夹具与工具
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    return research.build_parser().parse_args(argv)


def _help_text(cmd: str, capsys: pytest.CaptureFixture) -> str:
    """取某个子命令的 ``-h`` 输出（``-h`` 走 ``SystemExit(0)``，帮助打在 stdout）。"""
    with pytest.raises(SystemExit) as exc:
        _parse([cmd, "-h"])
    assert exc.value.code == 0
    return capsys.readouterr().out


# ===========================================================================
#  H1 —— --force 的四重语义拆分
# ===========================================================================
#: 新主名 → 它所属的子命令与调用形态（``target`` 只是让 argparse 有位置参数可吃）。
_NEW_NAMES = {
    "read": ("--refresh", ["read", "10.1/x"]),
    "ingest": ("--refresh", ["ingest"]),
    "add": ("--allow-fail", ["add", "10.1/x"]),
}


@pytest.mark.parametrize("cmd", sorted(_NEW_NAMES))
def test_the_new_primary_flag_sets_the_internal_dest(cmd: str) -> None:
    """新主名必须落到 ``dest="force"``。

    这是拆分能做到「零波及」的关键：内部所有读取点（``cmd_read`` 的缓存判据、
    ``_citation_gate(force=...)``、``run_ingest(force=...)``）都读 ``args.force``，
    dest 不变意味着改的只是 CLI 表面，实现一行未动。若哪天有人把 dest 也改了，
    这条测试会先炸，而不是让某个分支静默失效。
    """
    flag, base = _NEW_NAMES[cmd]

    args = _parse([*base, flag])

    assert args.force is True
    # 没给旧名时不该留下「用户敲过 --force」的痕迹，否则迁移提示会凭空冒出来
    assert args.force_deprecated is False


@pytest.mark.parametrize("cmd", sorted(_NEW_NAMES))
def test_the_new_primary_flag_defaults_to_false(cmd: str) -> None:
    _, base = _NEW_NAMES[cmd]

    assert _parse(base).force is False


@pytest.mark.parametrize("cmd", sorted(_NEW_NAMES))
def test_the_deprecated_force_still_works_and_warns(
    cmd: str, capsys: pytest.CaptureFixture
) -> None:
    """旧 ``--force`` 照常工作，但必须向 stderr 打一行迁移提示。

    「不做破坏性变更」与「不再诱导旧用法」要同时成立：功能保留，提示走 stderr
    （不污染可能被判读的 stdout），且 ``-h`` 里查不到旧名（见下条测试）。
    """
    _, base = _NEW_NAMES[cmd]

    args = _parse([*base, "--force"])
    assert args.force is False, "归一发生在 main()，parse 阶段只记录旧名被敲过"
    assert args.force_deprecated is True

    research._migrate_force_alias(args)

    assert args.force is True
    err = capsys.readouterr().err
    assert "--force" in err and f"{cmd} {_NEW_NAMES[cmd][0]}" in err


def test_the_migration_hint_tells_the_two_meanings_apart(
    capsys: pytest.CaptureFixture,
) -> None:
    """``add`` 与 ``read`` 的提示措辞必须不同——否则拆名的意义就丢了。

    拆名的动机不是好看，是 ``add --force`` 把一个**安全门**开关伪装成了普通的缓存刷新。
    如果迁移提示只说「改名了」，用户/LLM 学到的是「同一个东西换了个拼写」，
    误用的风险原封不动。
    """
    add_args = _parse(["add", "10.1/x", "--force"])
    research._migrate_force_alias(add_args)
    add_err = capsys.readouterr().err

    read_args = _parse(["read", "10.1/x", "--force"])
    research._migrate_force_alias(read_args)
    read_err = capsys.readouterr().err

    assert "安全门" in add_err
    assert "缓存" in read_err
    assert add_err != read_err


def test_index_keeps_force_as_its_primary_name(
    capsys: pytest.CaptureFixture,
) -> None:
    """``index --force`` 原样保留：那里的语义确实是「papers/ 为空时也强制写」，最贴近原义。

    连带断言它**没有** ``force_deprecated`` 这个 dest——``_migrate_force_alias`` 对它是
    彻底的空操作，不会把一条不该出现的迁移提示打到用户脸上。
    """
    args = _parse(["index", "--force"])

    assert args.force is True
    assert not hasattr(args, "force_deprecated")

    research._migrate_force_alias(args)
    assert capsys.readouterr().err == ""


def test_migration_is_a_noop_for_a_bare_namespace() -> None:
    """直接构造 ``Namespace`` 调 ``cmd_*`` 的路径（既有测试全走这条）不得被炸。

    ``_migrate_force_alias`` 只在 ``main()`` 里调用，但它是公开形状的内部函数，
    将来也可能被别处调到。缺 ``force_deprecated`` 与缺 ``cmd`` 都必须安全。
    """
    research._migrate_force_alias(argparse.Namespace(force=False))
    research._migrate_force_alias(argparse.Namespace())


@pytest.mark.parametrize("cmd", sorted(_NEW_NAMES))
def test_help_hides_the_old_name_and_shows_the_new_one(
    cmd: str, capsys: pytest.CaptureFixture
) -> None:
    """``-h`` 是 LLM 唯一读得到的接口：旧名隐藏（不诱导），新名可见（可发现）。"""
    flag, _ = _NEW_NAMES[cmd]

    text = _help_text(cmd, capsys)

    assert flag in text
    assert "--force" not in text


def test_help_still_advertises_force_for_index(
    capsys: pytest.CaptureFixture,
) -> None:
    assert "--force" in _help_text("index", capsys)


def test_main_runs_the_migration_before_dispatching(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """端到端：``main()`` 必须在派发前归一，否则 ``cmd_*`` 看到的还是 ``force=False``。

    这一步是拆分能真正生效的唯一保证——parser 层改得再对，只要 ``main()`` 忘了调
    ``_migrate_force_alias``，旧名就会静默退化成「什么都不做」（因为 dest 不同），
    而那正是最坏的失败形态：用户以为加了 ``--force``。
    """
    seen: dict[str, Any] = {}

    def _fake_cmd_read(args: argparse.Namespace) -> int:
        seen["force"] = args.force
        return 0

    monkeypatch.setattr(research, "cmd_read", _fake_cmd_read)
    # autoclean 会去碰真实 cache/ 目录；本测试只关心参数归一
    monkeypatch.setattr(research.cache_manager, "maybe_autoclean", lambda: None)

    assert research.main(["read", "10.1/x", "--force"]) == 0

    assert seen["force"] is True
    assert "--refresh" in capsys.readouterr().err


# ===========================================================================
#  H2 / H6 —— search --json 与降级态空结果告警
# ===========================================================================
_WORK: dict[str, Any] = {
    "openalex_id": "W2748115035",
    "doi": "10.1103/x",
    "title": "Topological acoustic states in a non-Hermitian lattice",
    "publication_year": 2021,
    "cited_by_count": 12,
    "first_author_last_name": "Zhu",
    "journal": "Nature Physics",
    "journal_tier": "top",
    "oa_status": "gold",
}

#: ``_row()`` 的完整投影。这里把键集**写死**而不是反过来问 ``_row``：它是
#: ``references/search.md`` 里向 LLM 承诺的 schema，少一个键就意味着文档失实。
_ROW_SCHEMA = {
    "title",
    "first_author_last_name",
    "year",
    "journal",
    "doi",
    "arxiv_id",
    "openalex_id",
    "cited_by_count",
    "oa_status",
    "jif",
    "jcr_quartile",
    "journal_tier",
    "source",
}


def _search_args(**kw: Any) -> argparse.Namespace:
    """``cmd_search`` 读的 args 字段（多数是直接属性访问，故必须齐备）。"""
    base: dict[str, Any] = {
        "query": "non-Hermitian topological acoustics",
        "source": "auto",
        "year": None,
        "limit": 15,
        "sort": "relevance",
        "min_citations": None,
        "oa_only": False,
        "enrich": False,
        "save": False,
        "purpose": None,
        "json": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """把两个检索后端换成可控假实现，返回一个可写的容器。

    ``cmd_search`` 只读 ``results`` / ``meta.count``（OpenAlex）与 ``entries`` /
    ``total_results``（arXiv），故假实现只需这四个键。
    """
    box: dict[str, list] = {"openalex": [], "arxiv": []}
    monkeypatch.setattr(
        research.openalex_client,
        "search_works",
        lambda *a, **k: {
            "results": box["openalex"],
            "meta": {"count": len(box["openalex"])},
        },
    )
    monkeypatch.setattr(
        research.arxiv_client,
        "search_arxiv",
        lambda *a, **k: {
            "entries": box["arxiv"],
            "total_results": len(box["arxiv"]),
        },
    )
    return box


def _set_openalex_key(monkeypatch: pytest.MonkeyPatch, key: str | None) -> None:
    """``Settings`` 是 ``frozen=True`` 的 dataclass，只能整体造一份替身换掉。"""
    monkeypatch.setattr(
        research, "settings", replace(research.settings, openalex_api_key=key)
    )


def test_search_json_emits_a_row_array_on_a_clean_stdout(
    sources: dict[str, list], capsys: pytest.CaptureFixture
) -> None:
    """``--json`` 的 stdout 必须是可直接 ``json.loads`` 的 ``_row()`` 数组。

    这里真正的断言不是「能解析」（那只是手段），而是「**只有** JSON」：
    各源的取回条数一旦泄露到 stdout，``| ConvertFrom-Json`` 就在第一行失败，
    而错误信息会指向 JSON 语法而不是真正的病因。
    """
    sources["openalex"].append(_WORK)

    assert research.cmd_search(_search_args(json=True)) == 0

    cap = capsys.readouterr()
    payload = json.loads(cap.out)
    assert isinstance(payload, list) and len(payload) == 1
    assert set(payload[0]) == _ROW_SCHEMA
    assert payload[0]["source"] == "openalex"
    assert payload[0]["journal_tier"] == "top"
    assert "[search]" not in cap.out, "进度提示必须全部走 stderr"
    assert "[search] OpenAlex" in cap.err


def test_search_json_keeps_working_with_zero_hits(
    sources: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """空结果是 ``[]`` 而不是「（无结果）」。

    后者对人类友好，对机器是灾难：调用方分不出「这个方向真没文献」与「输出格式
    又变了」。同时 ``[]`` 仍是合法 JSON，上游管道不会断。
    """
    _set_openalex_key(monkeypatch, "OA-KEY")  # 排除 H6 告警的干扰，只测 JSON 形态

    assert research.cmd_search(_search_args(json=True)) == 0

    assert json.loads(capsys.readouterr().out) == []


def test_search_without_json_still_prints_the_human_table(
    sources: dict[str, list], capsys: pytest.CaptureFixture
) -> None:
    """回归：不加 ``--json`` 时行为逐字不变（进度提示仍在 stdout）。"""
    sources["openalex"].append(_WORK)

    assert research.cmd_search(_search_args()) == 0

    out = capsys.readouterr().out
    assert "[search] OpenAlex: 取回 1 条" in out
    assert "Zhu—— Topological acoustic states" in out


def test_search_warns_when_an_empty_result_may_just_be_an_exhausted_quota(
    sources: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """降级态 + 空结果 → 必须告警（方案 H6，实测确认的运行状态）。"""
    _set_openalex_key(monkeypatch, None)

    assert research.cmd_search(_search_args()) == 0

    err = capsys.readouterr().err
    assert "降级态" in err and "--source arxiv" in err


def test_search_stays_quiet_about_the_quota_when_a_key_is_configured(
    sources: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """有 key 时空结果就是空结果，不该拿配额去给用户一个假借口。"""
    _set_openalex_key(monkeypatch, "OA-KEY")

    assert research.cmd_search(_search_args()) == 0

    assert capsys.readouterr().err == ""


def test_search_does_not_blame_openalex_when_it_was_never_consulted(
    sources: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """``--source arxiv`` 压根没查 OpenAlex，对它喊配额耗尽是误导。"""
    _set_openalex_key(monkeypatch, None)

    assert research.cmd_search(_search_args(source="arxiv")) == 0

    assert "降级态" not in capsys.readouterr().err


def test_search_only_warns_on_an_empty_result_set(
    sources: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """拿到了结果就不存在「不可判读」的问题，告警会变成噪声。"""
    _set_openalex_key(monkeypatch, None)
    sources["openalex"].append(_WORK)

    assert research.cmd_search(_search_args()) == 0

    assert capsys.readouterr().err == ""


def test_the_quota_warning_never_pollutes_the_json_stdout(
    sources: dict[str, list],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """H2 与 H6 的交互：告警走 stderr，JSON 走 stdout，两者同时生效时仍各自纯净。"""
    _set_openalex_key(monkeypatch, None)

    assert research.cmd_search(_search_args(json=True)) == 0

    cap = capsys.readouterr()
    assert json.loads(cap.out) == []
    assert "降级态" in cap.err


# ===========================================================================
#  H3 —— 参数形态收敛：ingest 的 action 位置参数 + 可推导的 epilog
# ===========================================================================
def _registered_subcommands() -> list[str]:
    """枚举已注册的子命令名（不点 argparse 的私有类名，只鸭子判形状）。"""
    for action in research.build_parser()._actions:
        choices = getattr(action, "choices", None)
        if (
            isinstance(choices, dict)
            and choices
            and all(isinstance(v, argparse.ArgumentParser) for v in choices.values())
        ):
            return sorted(choices)
    raise AssertionError("找不到子命令注册表——argparse 内部结构变了？")


def test_ingest_defaults_to_the_run_action() -> None:
    """``research ingest`` 不带 action 时必须是 ``run``。

    这是向后兼容的全部：旧写法 ``ingest --dry-run --theme cpa_ep`` 里根本没有
    action，默认值一变就会让既有命令静默改行为。
    """
    assert _parse(["ingest"]).action == "run"


def test_ingest_accepts_status_as_a_positional_action() -> None:
    assert _parse(["ingest", "status"]).action == "status"


def test_ingest_rejects_an_unknown_action() -> None:
    """``choices`` 必须真的生效：拼错的 action 该当场报用法错，而不是默默跑入库。"""
    with pytest.raises(SystemExit) as exc:
        _parse(["ingest", "statsu"])
    assert exc.value.code == 2


def test_the_legacy_status_flag_is_still_accepted() -> None:
    """``ingest --status`` 保留：它已进了文档与肌肉记忆，不做破坏性变更。"""
    args = _parse(["ingest", "--status"])

    assert args.action == "run"
    assert args.status is True


def _ingest_args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "action": "run",
        "manifest": None,
        "status": False,
        "priority": None,
        "theme": None,
        "backend": None,
        "limit_pages": None,
        "limit_files": None,
        "dry_run": False,
        "force": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_ingest_summarises_for_either_spelling_of_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """两种写法都只能汇总，且**绝不**碰 ``run_ingest``。

    后者是本测试的真正重点：``ingest status`` 是只读查询，若因为参数归一写错而
    跑起了真入库（复制文件 + 调 MinerU 云端扣额度），那是代价高昂的静默失败。
    """
    ran: list[Any] = []
    monkeypatch.setattr(
        research.local_ingest, "load_manifest", lambda path: {"entries": []}
    )
    monkeypatch.setattr(research.local_ingest, "summarize", lambda data: "汇总文本")
    monkeypatch.setattr(
        research.local_ingest,
        "run_ingest",
        lambda *a, **k: ran.append((a, k)) or 0,
    )

    assert research.cmd_ingest(_ingest_args(action="status")) == 0
    assert research.cmd_ingest(_ingest_args(status=True)) == 0

    assert capsys.readouterr().out.count("汇总文本") == 2
    assert ran == []


def test_cmd_ingest_runs_the_pipeline_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    seen: dict[str, Any] = {}

    def _fake_run(manifest: str, **kw: Any) -> int:
        seen.update({"manifest": manifest, **kw})
        return 0

    monkeypatch.setattr(research.local_ingest, "run_ingest", _fake_run)

    assert research.cmd_ingest(_ingest_args(theme="cpa_ep", force=True)) == 0

    assert seen["theme"] == "cpa_ep"
    # H1 的 dest 归一：``--refresh`` / 旧 ``--force`` 都落到 ``args.force``，再原样传下去
    assert seen["force"] is True
    assert capsys.readouterr().out == ""


def test_the_epilog_accounts_for_every_registered_subcommand() -> None:
    """epilog 声称「14 个全部落在这两类里」，那就得真能对上号。

    这条测试把 epilog 从「写给人看的散文」变成受约束的契约：新增第 15 个子命令而
    忘了归类，测试会直接报出漏的是哪个，而不是让文档静默失实（那正是 WP-G G6
    清了一整轮的那类矛盾）。
    """
    names = _registered_subcommands()

    assert len(names) == 14, f"子命令数变了：{names}"
    missing = [n for n in names if n not in research._CLI_SHAPE_EPILOG]
    assert not missing, f"epilog 漏了这些子命令：{missing}"


def test_the_epilog_is_actually_rendered_with_its_manual_alignment(
    capsys: pytest.CaptureFixture,
) -> None:
    """epilog 的对齐靠手排空格，必须用 ``RawDescriptionHelpFormatter`` 才不会被重折。

    断言里带上行首的两个空格：默认 formatter 会把这一段重新折行成一团，而那种输出
    比没有 epilog 更难读。
    """
    with pytest.raises(SystemExit):
        research.build_parser().parse_args(["-h"])

    out = capsys.readouterr().out
    assert "子命令只有两种参数形态" in out
    assert "\n  A  动词 + 位置目标" in out
    assert "\n  B  名词 + action 位置参数" in out


# ===========================================================================
#  H4 —— library list 的诚实性（与它的失败路径）
# ===========================================================================
@pytest.fixture
def zotero(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """把 Zotero 桥换成可控替身；返回一个可写的行为容器。"""
    box: dict[str, Any] = {"items": [], "exc": None, "calls": []}

    class _Fake:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def list_items(self, **kw: Any) -> list:
            box["calls"].append(("list", kw))
            if box["exc"]:
                raise box["exc"]
            return box["items"]

        def search_items(self, query: str, **kw: Any) -> list:
            box["calls"].append(("search", query, kw))
            if box["exc"]:
                raise box["exc"]
            return box["items"]

    monkeypatch.setattr(research.zotero_cli, "available", lambda: True)
    monkeypatch.setattr(research.zotero_cli, "ZoteroCli", _Fake)
    return box


def _library_args(**kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "action": "list",
        "query": None,
        "key": None,
        "type": None,
        "limit": 25,
    }
    base.update(kw)
    return argparse.Namespace(**base)


_ITEM = {
    "key": "ABCD1234",
    "data": {
        "title": "Topological acoustic states",
        "itemType": "journalArticle",
        "date": "2021-03-12",
        "creators": [{"lastName": "Zhu"}],
    },
}

#: ``list`` 空结果时那句点破的字面核心。抽成常量而不内联：它同时被正向
#: （list 必须有）与反向（search 必须无）两条断言引用，两边写歪了就变成两条
#: 同时为真的空断言。
_EMPTY_LIST_DISCLAIMER = "并不代表库是空的"


def test_library_list_says_out_loud_that_it_is_best_effort(
    zotero: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """告知必须走 **stderr**，且必须说出替代方案。

    走 stderr：它是关于输出可信度的元信息，不是一条结果；混在 stdout 的条目列表里
    会让调用方把它当成第 0 条文献。说出替代方案：只说「不全」而不说「那该怎么办」
    等于把问题抛回给用户。
    """
    zotero["items"] = [_ITEM]

    assert research.cmd_library(_library_args()) == 0

    cap = capsys.readouterr()
    assert "尽力而为" in cap.err
    assert "library search --query" in cap.err
    assert "100" in cap.err, "上游把 limit 夹到 ≤100，这个硬上限必须说出口"
    # 结果仍在 stdout，且没被告知行污染
    assert "ABCD1234" in cap.out
    assert "尽力而为" not in cap.out


def test_library_search_does_not_carry_the_list_caveat(
    zotero: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """``search`` 是真检索，不该被贴上「结果不全属正常」的标签。

    把告知挂在两个 action 上看似更保险，实际是把一个准确的声明稀释成一句噪声。
    """
    zotero["items"] = [_ITEM]

    assert research.cmd_library(_library_args(action="search", query="acoustic")) == 0

    cap = capsys.readouterr()
    assert "尽力而为" not in cap.err
    assert "ABCD1234" in cap.out


def test_library_list_passes_limit_and_type_through(
    zotero: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """告知不得偷走参数：``--limit`` / ``--type`` 必须原样交给上游。"""
    research.cmd_library(_library_args(limit=80, type="journalArticle"))

    assert zotero["calls"] == [("list", {"limit": 80, "item_type": "journalArticle"})]


def test_an_empty_library_list_refuses_to_claim_the_library_is_empty(
    zotero: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """``list`` 返回 0 条时必须点破「这并不代表库是空的」。

    这是实测而非假设：本机 Zotero 确实有藏书，``library list`` 返回 0 条，而
    ``library search --query acoustic`` 能命中。沿用通用的「（无条目）」会被读成
    「你的库是空的」，而那正是本项要消除的那类不可判读输出。退出码仍为 0：
    查询本身成功了，只是结果不可信。

    否定词必须是**字面连续**的：这串走终端 stderr，markdown 的 ``**不**`` 不渲染，
    只会显示成字面星号并把句子割裂（本项第一版就是这么写的，因而断言失配）。
    """
    zotero["items"] = []

    assert research.cmd_library(_library_args()) == 0

    cap = capsys.readouterr()
    assert _EMPTY_LIST_DISCLAIMER in cap.err
    assert "**" not in cap.err, "终端 stderr 里不得出现 markdown 强调标记"
    assert "（无条目）" not in cap.out


def test_an_empty_library_search_is_reported_plainly(
    zotero: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """对比项：``search`` 的 0 条是真没匹配上，不需那层注解。"""
    zotero["items"] = []

    assert research.cmd_library(_library_args(action="search", query="zzz")) == 0

    cap = capsys.readouterr()
    assert "（无条目）" in cap.out
    assert _EMPTY_LIST_DISCLAIMER not in cap.err


@pytest.mark.parametrize("action", ["list", "search"])
def test_a_failing_zotero_bridge_degrades_instead_of_crashing(
    action: str, zotero: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """Zotero 没开 / 本地 API 未授权时：一行提示 + 退出码 1，而不是 traceback。

    ``zotero_cli`` 只在 ``ping`` / ``get_item`` 里自己吞异常，``list_items`` /
    ``search_items`` 会把 ``ZoteroCliError`` 一路抛到 ``main()``——而后者只接
    ``KeyboardInterrupt``。那是降级路径上的未处理崩溃（违反不变量 1）。
    """
    zotero["exc"] = research.zotero_cli.ZoteroCliError("Zotero 本地 API 未开启")

    rc = research.cmd_library(_library_args(action=action, query="acoustic"))

    assert rc == 1
    err = capsys.readouterr().err
    assert f"[library] {action} 失败" in err and "本地 API" in err
