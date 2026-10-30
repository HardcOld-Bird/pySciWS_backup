"""``research review`` 子命令的离线回归测试（全部在 tmp_path 下，不碰真实 reviews/）。

护栏对象是 WP-F F1：``REVIEWS_DIR`` 此前**定义了但全文件无人使用**，而设计良好的
``templates/review_note.md``（132 行）是孤儿模板。本子命令把它接上，只做三件机械活：

* ``new``    —— 实例化模板 + 从检索快照聚合检索式（脚手架）；
* ``status`` —— 校验 ``papers_reviewed`` 的 ``[[wiki-link]]`` 与两个计数（完整性门）；
* ``sync``   —— 重算链接与计数并**只重写 frontmatter**（同步）。

锁定的四条不变量：

1. ``new`` 产出的 frontmatter **键集与模板一致**——模板是综述的字段契约，漏一个键就会
   让下游按字段取值的代码静默拿不到东西；
2. ``sync`` 的**正文逐字节保留**（含 CRLF 文件的换行风格），且**不追加 Changelog**——
   Changelog 记的是综述的认知进展，机械的计数刷新写进去会淹掉真正值得看的条目；
3. ``sync`` **不静默删除断链**——那会把需要修的问题藏起来；断链只从计数里排除；
4. 解析不出 frontmatter 的综述（人工手写）**绝不被自动改写**，且 ``status`` 对它退出码 1
   ——一份没能被检查的综述默默通过一道门，比报错危险得多。

**明确不在覆盖范围**：综述正文生成。``review_note.md`` 的 §1-§5 是 LLM 的核心判断工作，
做成模板填充器只会把活的判断变成僵的套话，因此 ``new`` 只填 frontmatter 与
``--purpose`` 指定的那一行「综述目标」。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import notes, research

# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
TOPIC = "Exceptional points in acoustic metamaterials"
TOPIC_SLUG = "exceptional-points-in-acoustic-metamaterials"

#: 综述正文的最小夹具。**故意带内容**：sync 的「正文逐字节不变」只有在正文非空、
#: 且含会被 YAML/markdown 处理误伤的形状（列表、表格、代码块）时才是个有意义的断言。
REVIEW_BODY = (
    "\n"
    "# EP — 综述笔记\n"
    "\n"
    "> **综述目标**：厘清声学 EP 的拓扑保护边界\n"
    "\n"
    "## 2. 关键脉络（Key Threads）\n"
    "\n"
    "- 代表论文：[[2018_zhu_topological-edge-state]]\n"
    "- 核心思想：\n"
    "\n"
    "| 年份 | 里程碑工作 |\n"
    "|---|---|\n"
    "| 2018 | zhu |\n"
    "\n"
    "```\n"
    "OpenAlex:\n"
    '  search = "..."\n'
    "```\n"
    "\n"
    "## 8. Changelog\n"
    "\n"
    "- 2026-09-01: 初稿由 AI 生成\n"
)


def _ns(**kw) -> argparse.Namespace:
    """构造 ``cmd_review`` 需要的最小 Namespace（默认走 ``new``）。"""
    base = dict(
        action="new",
        target=None,
        purpose=None,
        from_shortlist=None,
        time_window=None,
        dry_run=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """把 ``research`` 的三个目录常量指到 tmp_path，测试绝不触碰真实数据区。

    ``TEMPLATES_DIR`` **不重定向**：``review new`` 应当套用的是仓库里真实的
    ``templates/review_note.md``，用假模板会让「键集与模板一致」这条断言失去意义。
    """
    papers = tmp_path / "papers"
    reviews = tmp_path / "reviews"
    shortlists = tmp_path / "shortlists"
    for d in (papers, reviews, shortlists):
        d.mkdir()
    monkeypatch.setattr(research, "PAPERS_DIR", papers)
    monkeypatch.setattr(research, "REVIEWS_DIR", reviews)
    monkeypatch.setattr(research, "SHORTLISTS_DIR", shortlists)
    return papers, reviews, shortlists


def _seed_paper(papers: Path, stem: str, status: str = "unread") -> Path:
    """在 papers/ 下造一篇笔记（``status`` 决定它算不算「已处理」）。"""
    fm = {
        "title": stem.replace("-", " ").replace("_", " ").title(),
        "first_author_last_name": stem.split("_")[1] if "_" in stem else stem,
        "year": int(stem[:4]) if stem[:4].isdigit() else 2020,
        "status": status,
    }
    p = papers / f"{stem}.md"
    p.write_text(
        notes.render_note(fm, f"# {stem}\n\n## Changelog\n\n- created\n"),
        encoding="utf-8",
        newline="\n",
    )
    return p


def _seed_review(
    reviews: Path,
    name: str = "2026-09_ep_survey.md",
    *,
    links: list | None = None,
    total: int = 0,
    read: int = 0,
    body: str = REVIEW_BODY,
    newline: str = "\n",
    topic: str = TOPIC,
) -> Path:
    """造一份综述笔记。``total`` / ``read`` 是**声明值**，故意可与实况不符以测漂移。"""
    fm = {
        "topic": topic,
        "topic_slug": notes.slugify(topic, 48),
        "created_date": "2026-09-01",
        "last_updated": "2026-09-01",
        "author": "AI Agent + User",
        "status": "draft",
        "time_window": "2018-01 至 2026-09",
        "sources_used": ["openalex"],
        "query_strings": ["exceptional point acoustics"],
        "inclusion_criteria": "",
        "exclusion_criteria": "",
        "papers_reviewed": list(links or []),
        "papers_total_count": total,
        "papers_read_count": read,
    }
    p = reviews / name
    p.write_text(notes.render_note(fm, body), encoding="utf-8", newline=newline)
    return p


def _seed_shortlist(
    shortlists: Path,
    name: str,
    *,
    sources: list[str],
    query_string: str,
) -> Path:
    """造一份检索快照。键名照抄 ``templates/shortlist.md``：``sources`` 是列表，
    而 ``query_string`` 是**单数**字符串——review 模板要的是复数列表，聚合时得合并。"""
    fm = {
        "query_date": "2026-09-01",
        "query_slug": notes.slugify(query_string, 40),
        "queried_by": "AI Agent (research.py)",
        "purpose": query_string,
        "sources": sources,
        "query_string": query_string,
        "filters": {"year_range": "2018-2026", "min_citations": None},
        "sort_by": "relevance_score:desc",
        "results_returned": 15,
        "results_after_screening": 0,
        "screening_status": "pending",
    }
    p = shortlists / name
    p.write_text(
        notes.render_note(fm, f"\n# {query_string}\n\n## 机器检索结果\n\n（略）\n"),
        encoding="utf-8",
        newline="\n",
    )
    return p


def _body_bytes(path: Path) -> bytes:
    """取文件里 frontmatter 之后的**原始字节**（不做任何换行归一化）。

    「正文逐字节不变」这条不变量只有在字节层面断言才算数：比对 str 会让
    ``\\r\\n`` ↔ ``\\n`` 的静默翻译逃过检查，而那正是 Windows 上最容易发生的事。
    结束行本身也可能是 LF 或 CRLF，两种都要认。
    """
    raw = path.read_bytes()
    assert raw.startswith(b"---")
    for sep in (b"\n---\r\n", b"\n---\n"):
        i = raw.find(sep, 3)
        if i >= 0:
            return raw[i + len(sep) :]
    raise AssertionError(f"{path.name} 里找不到 frontmatter 结束行")


# ===========================================================================
#  review new —— 脚手架
# ===========================================================================
def test_review_new_creates_file_with_expected_name(dirs, capsys):
    _papers, reviews, _sl = dirs
    rc = research.cmd_review(_ns(action="new", target=TOPIC))
    assert rc == 0
    expected = reviews / f"{research._today()[:7]}_{TOPIC_SLUG}_survey.md"
    assert expected.is_file()
    assert f"{expected}" in capsys.readouterr().out


def test_review_new_filename_shape(dirs):
    """文件名形状钉死为 ``{YYYY-MM}_{topic_slug}_survey.md``（方案 F1 的约定）。"""
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    names = [p.name for p in reviews.glob("*.md")]
    assert len(names) == 1
    assert re.fullmatch(r"\d{4}-\d{2}_.+_survey\.md", names[0])


def test_review_new_frontmatter_key_set_matches_template(dirs):
    """**键集与真模板完全一致**——模板是综述的字段契约。

    多一个键意味着工具在往契约外塞东西；少一个键意味着下游按字段取值会静默拿不到。
    两种漂移都不会报错，只能靠这条断言拦住。
    """
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert set(fm) == set(notes.template_defaults("review_note.md"))


def test_review_new_fills_machine_owned_fields(dirs):
    _papers, reviews, _sl = dirs
    today = research._today()
    research.cmd_review(
        _ns(action="new", target=TOPIC, time_window="2018-01 至 2026-09")
    )
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert fm["topic"] == TOPIC
    assert fm["topic_slug"] == TOPIC_SLUG
    assert fm["created_date"] == today
    assert fm["last_updated"] == today
    assert fm["status"] == "draft"
    assert fm["time_window"] == "2018-01 至 2026-09"
    assert fm["papers_reviewed"] == []
    assert fm["papers_total_count"] == 0
    assert fm["papers_read_count"] == 0


def test_review_new_leaves_criteria_for_the_ai(dirs):
    """纳入/排除标准是**方法学判断**，只能由人/AI 填——工具留空而不猜。"""
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert fm["inclusion_criteria"] == ""
    assert fm["exclusion_criteria"] == ""


def test_review_new_body_comes_from_real_template(dirs):
    """正文是模板骨架（§1-§8 全在），而**不是**工具生成的综述内容。"""
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    _fm, body = notes.split_note(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    for heading in (
        "## 1. 领域概览",
        "## 2. 关键脉络",
        "## 3. 时间线",
        "## 4. 争议与未解问题",
        "## 5. 与我的研究的接口",
        "## 6. 投稿建议",
        "## 7. 检索式记录",
        "## 8. Changelog",
    ):
        assert heading in body


def test_review_new_fills_topic_placeholder_and_keeps_unknown_ones(dirs):
    """``{{topic}}`` 被填掉，而 ``{{...}}`` 这类**给 AI 的**占位符原样保留。

    ``_fill_placeholders`` 对未知键返回原文，这正是「不做综述生成器」的执行点：
    模板里那些 ``路线 A：{{...}}`` 是留给 AI 判断的槽位，工具不猜。
    """
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    _fm, body = notes.split_note(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert f"# {TOPIC} — 综述笔记" in body
    assert "{{topic}}" not in body
    assert "{{...}}" in body


def test_review_new_purpose_lands_in_body(dirs):
    """``--purpose`` 只替换「综述目标」那一行的括号提示，不动其余正文。"""
    _papers, reviews, _sl = dirs
    research.cmd_review(
        _ns(action="new", target=TOPIC, purpose="声学 EP 的拓扑保护边界")
    )
    _fm, body = notes.split_note(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert "> **综述目标**：声学 EP 的拓扑保护边界" in body
    assert "（一句话说明这份综述要回答什么问题）" not in body
    # 其余段落没被牵连
    assert "## 8. Changelog" in body


def test_review_new_without_purpose_leaves_the_prompt(dirs):
    """不给 ``--purpose`` 时模板提示行原样保留（工具不编造综述目标）。"""
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    _fm, body = notes.split_note(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert "（一句话说明这份综述要回答什么问题）" in body


def test_review_new_warns_when_purpose_has_nowhere_to_go(dirs, monkeypatch, capsys):
    """模板改了导致提示行找不到时，``--purpose`` **不得静默丢失**。"""
    monkeypatch.setattr(
        research, "_REVIEW_PURPOSE_RE", re.compile(r"@@never-matches@@")
    )
    _papers, reviews, _sl = dirs
    rc = research.cmd_review(_ns(action="new", target=TOPIC, purpose="目标 X"))
    assert rc == 0  # 综述照样建成
    err = capsys.readouterr().err
    assert "--purpose 未写入" in err
    assert "目标 X" in err


def test_review_new_aggregates_shortlists(dirs, capsys):
    """从多份快照聚合 ``sources_used`` 与 ``query_strings``：并集、去重、保序。

    这是 ``--from-shortlist`` 存在的全部理由——快照里存着**实际提交**的检索式，
    让用户再手抄一遍必然漂移，而综述的 §7「检索式记录（Reproducibility）」要的
    恰恰是可复现的那一条。
    """
    _papers, reviews, shortlists = dirs
    _seed_shortlist(
        shortlists,
        "2026-09-01_ep.md",
        sources=["openalex", "arxiv"],
        query_string="exceptional point",
    )
    _seed_shortlist(
        shortlists,
        "2026-09-02_bic.md",
        sources=["arxiv", "wos"],
        query_string="bound state in continuum",
    )
    rc = research.cmd_review(
        _ns(
            action="new",
            target=TOPIC,
            from_shortlist=["2026-09-01_ep.md", "2026-09-02_bic.md"],
        )
    )
    assert rc == 0
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    # arxiv 在两份快照里都有 → 只出现一次；顺序按首次出现
    assert fm["sources_used"] == ["openalex", "arxiv", "wos"]
    assert fm["query_strings"] == ["exceptional point", "bound state in continuum"]


def test_review_new_from_shortlist_accepts_stem_and_full_path(dirs):
    """``--from-shortlist`` 接受不带 ``.md`` 的文件名与完整路径两种写法。"""
    _papers, reviews, shortlists = dirs
    a = _seed_shortlist(
        shortlists, "2026-09-01_ep.md", sources=["openalex"], query_string="q1"
    )
    _seed_shortlist(
        shortlists, "2026-09-02_bic.md", sources=["arxiv"], query_string="q2"
    )
    research.cmd_review(
        _ns(action="new", target=TOPIC, from_shortlist=["2026-09-01_ep", str(a)])
    )
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    # 同一份快照给了两次 → 去重后只贡献一条检索式；第二份未被指定，不出现
    assert fm["sources_used"] == ["openalex"]
    assert fm["query_strings"] == ["q1"]


def test_review_new_missing_shortlist_degrades_with_notice(dirs, capsys):
    """快照找不到时只留一行提示（不变量 1）：综述照样建成，检索式列表留空。"""
    _papers, reviews, _sl = dirs
    rc = research.cmd_review(
        _ns(action="new", target=TOPIC, from_shortlist=["no-such-snapshot"])
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "找不到检索快照 no-such-snapshot" in err
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert fm["sources_used"] == []
    assert fm["query_strings"] == []


def test_review_new_ambiguous_shortlist_is_not_guessed(dirs, capsys):
    """片段匹配到多份快照时**不猜**：猜错会把别的检索式写进综述的复现记录里。"""
    _papers, reviews, shortlists = dirs
    _seed_shortlist(
        shortlists, "2026-09-01_ep.md", sources=["openalex"], query_string="q1"
    )
    _seed_shortlist(
        shortlists, "2026-09-02_ep.md", sources=["arxiv"], query_string="q2"
    )
    rc = research.cmd_review(_ns(action="new", target=TOPIC, from_shortlist=["ep"]))
    assert rc == 0
    assert "找不到检索快照 ep" in capsys.readouterr().err
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert fm["query_strings"] == []


def test_review_new_shortlist_with_empty_query_string_is_reported(dirs, capsys):
    """快照的 ``query_string`` 为空时明确报出来，而不是让检索式悄悄少一条。"""
    _papers, reviews, shortlists = dirs
    _seed_shortlist(
        shortlists, "2026-09-01_x.md", sources=["openalex"], query_string=""
    )
    research.cmd_review(
        _ns(action="new", target=TOPIC, from_shortlist=["2026-09-01_x.md"])
    )
    assert "query_string 为空" in capsys.readouterr().err


def test_review_new_refuses_to_overwrite(dirs, capsys):
    """同名综述已存在时拒绝覆盖（退出码 2），原文件一个字节都不动。"""
    _papers, reviews, _sl = dirs
    research.cmd_review(_ns(action="new", target=TOPIC))
    out = next(reviews.glob("*.md"))
    before = out.read_bytes()
    rc = research.cmd_review(_ns(action="new", target=TOPIC))
    assert rc == 2
    assert out.read_bytes() == before
    assert "已存在，不覆盖" in capsys.readouterr().err


def test_review_new_empty_topic_returns_two(dirs, capsys):
    assert research.cmd_review(_ns(action="new", target="")) == 2
    assert research.cmd_review(_ns(action="new", target=None)) == 2
    assert "需要一个主题" in capsys.readouterr().err


def test_review_new_degenerate_topic_falls_back_to_untitled(dirs):
    """全标点的主题落到 ``untitled`` slug——这是 :func:`notes.slugify` 的既定契约。

    故意**不**在 ``cmd_review`` 里多加一道拦截：``slugify`` 永不返回空串，那样一个
    ``if not slug`` 守卫会是永远走不到的死代码。退化主题的后果（一份名为
    ``*_untitled_survey.md`` 的综述，而 frontmatter 的 ``topic`` 如实记着用户敲的东西）
    是可恢复且不具破坏性的，不值得为它牺牲一个清晰的实现。
    """
    _papers, reviews, _sl = dirs
    assert research.cmd_review(_ns(action="new", target="///")) == 0
    p = next(reviews.glob("*.md"))
    assert p.name == f"{research._today()[:7]}_untitled_survey.md"
    fm = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    assert fm["topic"] == "///"  # 如实记录，不编造
    assert fm["topic_slug"] == "untitled"


def test_review_new_missing_template_returns_two(dirs, monkeypatch, capsys):
    monkeypatch.setattr(research, "TEMPLATES_DIR", dirs[0] / "no-such-templates")
    assert research.cmd_review(_ns(action="new", target=TOPIC)) == 2
    assert "review_note.md 缺失" in capsys.readouterr().err
    # 模板都没有 ⇒ 不该留下一个 reviews/ 里的半成品
    assert list(dirs[1].glob("*.md")) == []


def test_review_unknown_action_returns_two(dirs, capsys):
    assert research.cmd_review(_ns(action="frobnicate")) == 2
    assert "未知 action" in capsys.readouterr().err


# ===========================================================================
#  wiki 链接解析
# ===========================================================================
@pytest.mark.parametrize(
    "value,expected",
    [
        (["[[2018_zhu_x]]"], ["2018_zhu_x"]),
        # 别名与锚点都要截掉，否则解析不到实存文件
        (["[[2018_zhu_x|Zhang 2018]]"], ["2018_zhu_x"]),
        (["[[2018_zhu_x#§2]]"], ["2018_zhu_x"]),
        # 手写综述常见的裸 stem（不带括号）
        (["2018_zhu_x"], ["2018_zhu_x"]),
        # 带 .md 后缀与路径前缀
        (["[[2018_zhu_x.md]]"], ["2018_zhu_x"]),
        (["[[papers/2018_zhu_x]]"], ["papers/2018_zhu_x"]),
        # 空项、None 与空壳 ``[[]]`` 都被跳过（后者若回落成裸字符串，
        # 会被 status 渲染成一条名为 ``[[[[]]]]`` 的假断链）
        (["", None, "[[]]"], []),
        ("[[2018_zhu_x]]", ["2018_zhu_x"]),  # 标量而非列表
        (None, []),
    ],
)
def test_wiki_targets(value, expected):
    assert research._wiki_targets(value) == expected


def test_resolve_paper_link_accepts_path_prefix(dirs):
    """``[[papers/2018_zhu_x]]`` 这种带目录前缀的写法也要能解析到实存笔记。"""
    papers, _reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    assert research._resolve_paper_link("2018_zhu_x") == papers / "2018_zhu_x.md"
    assert research._resolve_paper_link("papers/2018_zhu_x") == papers / "2018_zhu_x.md"
    assert research._resolve_paper_link("2099_nobody_x") is None
    assert research._resolve_paper_link("") is None


def test_resolve_in_dir_refuses_ambiguous_glob(dirs):
    _papers, reviews, _sl = dirs
    _seed_review(reviews, "2026-09_ep_survey.md")
    _seed_review(reviews, "2026-10_ep_survey.md")
    assert research._resolve_in_dir(reviews, "ep") is None  # 两个候选 → 不猜
    assert research._resolve_in_dir(reviews, "2026-09_ep_survey.md") is not None
    assert research._resolve_in_dir(reviews, "2026-09_ep_survey") is not None  # 补 .md
    assert research._resolve_in_dir(reviews, "") is None


# ===========================================================================
#  review status —— 链接完整性门
# ===========================================================================
def test_review_status_healthy_returns_zero(dirs, capsys):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    _seed_paper(papers, "2021_gu_y", status="unread")
    _seed_review(reviews, links=["[[2018_zhu_x]]", "[[2021_gu_y]]"], total=2, read=1)
    assert research.cmd_review(_ns(action="status")) == 0
    out = capsys.readouterr().out
    assert "解析成功 2，断链 0" in out
    assert "计数偏差 : 无" in out


def test_review_status_broken_link_returns_one(dirs, capsys):
    """断链 → 退出码 1（与 ``index --check`` / ``citecheck`` 的 CI 友好约定一致）。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(reviews, links=["[[2018_zhu_x]]", "[[2099_ghost_z]]"], total=1)
    assert research.cmd_review(_ns(action="status")) == 1
    out = capsys.readouterr().out
    assert "断链 1" in out
    assert "[[2099_ghost_z]]" in out


def test_review_status_count_drift_alone_returns_zero(dirs, capsys):
    """计数漂移**不是**失败——那正是 ``sync`` 要修的东西。

    若漂移也返回 1，``status`` 就没法当「sync 之前先看看要改什么」的预检用了。
    """
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    _seed_review(reviews, links=["[[2018_zhu_x]]"], total=99, read=0)
    assert research.cmd_review(_ns(action="status")) == 0
    out = capsys.readouterr().out
    assert "papers_total_count 99 → 应为 1" in out
    assert "papers_read_count 0 → 应为 1" in out
    assert "research review sync" in out


@pytest.mark.parametrize(
    "status,counts_as_read",
    [
        ("unread", False),
        ("reading", True),
        ("read", True),
        ("archived", True),
        ("rejected", True),
        ("", False),  # 缺 status 的笔记不算已处理
    ],
)
def test_review_status_read_criterion(dirs, status, counts_as_read):
    """``papers_read_count`` 的判据是 ``status != "unread"``（方案 F1 的定义）。

    这个口径偏宽（reading / archived / rejected 都算），但它是**可复算**的：任何更细的
    判据都得先约定 ``archived`` 到底算不算读过，而那个约定不在数据里。
    """
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status=status)
    _seed_review(reviews, links=["[[2018_zhu_x]]"], total=1, read=0)
    aud = research._review_audit(next(reviews.glob("*.md")))
    assert aud["read_actual"] == (1 if counts_as_read else 0)


def test_review_status_bare_stem_link_resolves(dirs):
    """手写综述里常见的裸 stem（不带 ``[[ ]]``）不该被报成断链。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(reviews, links=["2018_zhu_x"], total=1)
    assert research.cmd_review(_ns(action="status")) == 0


def test_review_status_duplicate_links_counted_once(dirs):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(
        reviews, links=["[[2018_zhu_x]]", "[[2018_zhu_x]]", "2018_zhu_x"], total=1
    )
    aud = research._review_audit(next(reviews.glob("*.md")))
    assert aud["total_actual"] == 1
    assert len(aud["links"]) == 1


def test_review_status_scans_all_by_default(dirs, capsys):
    """缺省扫 ``reviews/*.md`` 全部：一份有断链就把整个门拉红。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(reviews, "2026-09_ok_survey.md", links=["[[2018_zhu_x]]"], total=1)
    _seed_review(reviews, "2026-10_bad_survey.md", links=["[[2099_ghost]]"], total=0)
    assert research.cmd_review(_ns(action="status")) == 1
    out = capsys.readouterr().out
    assert "2026-09_ok_survey.md" in out and "2026-10_bad_survey.md" in out


def test_review_status_explicit_target(dirs, capsys):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(reviews, "2026-09_ok_survey.md", links=["[[2018_zhu_x]]"], total=1)
    _seed_review(reviews, "2026-10_bad_survey.md", links=["[[2099_ghost]]"], total=0)
    # 只查完好的那份 → 0，且不打印另一份
    assert research.cmd_review(_ns(action="status", target="2026-09_ok_survey.md")) == 0
    assert "2026-10_bad_survey.md" not in capsys.readouterr().out


def test_review_status_missing_explicit_target_returns_two(dirs, capsys):
    """显式指定却找不到 → 2（用法错），与「目录下本来就没综述」的 0 区分开。"""
    assert research.cmd_review(_ns(action="status", target="nope.md")) == 2
    assert "找不到综述" in capsys.readouterr().err


def test_review_status_empty_reviews_dir_returns_zero(dirs, capsys):
    """空目录是正常降级（不变量 1）：一行提示 + 退出码 0。"""
    assert research.cmd_review(_ns(action="status")) == 0
    assert "没有综述笔记" in capsys.readouterr().out


def test_review_status_missing_reviews_dir_returns_zero(dirs, monkeypatch, capsys):
    monkeypatch.setattr(research, "REVIEWS_DIR", dirs[0] / "no-such-reviews")
    assert research.cmd_review(_ns(action="status")) == 0
    assert "reviews/ 目录不存在" in capsys.readouterr().out


def test_review_status_unparseable_file_returns_one_and_leaves_it_alone(dirs):
    """无 frontmatter 的人工综述：退出码 1，且**绝不被改写**。"""
    _papers, reviews, _sl = dirs
    p = reviews / "2026-09_handwritten_survey.md"
    original = "# 我的手写综述\n\n没有 frontmatter。\n"
    p.write_text(original, encoding="utf-8")
    assert research.cmd_review(_ns(action="status")) == 1
    assert p.read_text(encoding="utf-8") == original


# ===========================================================================
#  review sync —— 只重写 frontmatter
# ===========================================================================
def test_review_sync_body_is_byte_identical(dirs):
    """**核心不变量**：sync 之后正文的原始字节与之前完全相同。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    p = _seed_review(
        reviews, links=["[[2018_zhu_x]]", "[[2099_ghost]]"], total=0, read=0
    )
    before = _body_bytes(p)
    assert research.cmd_review(_ns(action="sync")) == 0
    assert _body_bytes(p) == before
    assert before  # 断言不是空对空


def test_review_sync_body_byte_identical_under_crlf(dirs):
    """CRLF 综述：sync 后正文的 ``\\r\\n`` 不得被翻译成 ``\\n``。

    ``_read_note_text`` 探测换行风格、``_write_note_text`` 按同一风格写回，这条链一旦
    断了，Windows 上每次 sync 都会把整份综述的正文重写成另一种换行——git diff 里
    看不出原因，而「正文逐字节不变」这条不变量就静默失效了。
    """
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    p = _seed_review(reviews, links=["[[2018_zhu_x]]"], total=0, newline="\r\n")
    assert b"\r\n" in p.read_bytes()
    before = _body_bytes(p)
    assert b"\r\n" in before
    research.cmd_review(_ns(action="sync"))
    after = _body_bytes(p)
    assert after == before
    assert b"\r\n" in after


def test_review_sync_updates_counts(dirs, capsys):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    _seed_paper(papers, "2021_gu_y", status="unread")
    _seed_review(reviews, links=["[[2018_zhu_x]]", "[[2021_gu_y]]"], total=0, read=0)
    assert research.cmd_review(_ns(action="sync")) == 0
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert fm["papers_total_count"] == 2
    assert fm["papers_read_count"] == 1
    assert "已重写 frontmatter" in capsys.readouterr().out


def test_review_sync_excludes_broken_links_from_total_but_keeps_them(dirs, capsys):
    """断链从计数里排除，但**不从 ``papers_reviewed`` 里删掉**。

    静默删掉会把「链接写错了」这个需要人修的问题藏起来——下次 sync 之后连痕迹都没有，
    而综述里那段引用它的正文还留在原地。
    """
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(reviews, links=["[[2018_zhu_x]]", "[[2099_ghost_z]]"], total=0)
    assert research.cmd_review(_ns(action="sync")) == 0
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    assert fm["papers_reviewed"] == ["[[2018_zhu_x]]", "[[2099_ghost_z]]"]
    assert fm["papers_total_count"] == 1  # 只算解析成功的那一篇
    assert "断链 [[2099_ghost_z]]" in capsys.readouterr().err


def test_review_sync_normalizes_and_dedupes_links(dirs):
    """裸 stem 统一成 ``[[...]]``，重复项去掉——计数才有确定的分母。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    _seed_review(
        reviews,
        links=["2018_zhu_x", "[[2018_zhu_x]]", "[[2018_ZHU_X]]"],
        total=0,
    )
    research.cmd_review(_ns(action="sync"))
    fm = notes.load_frontmatter(next(reviews.glob("*.md")).read_text(encoding="utf-8"))
    # 大小写不同但指向同一篇 → 只留首次出现的那条
    assert fm["papers_reviewed"] == ["[[2018_zhu_x]]"]
    assert fm["papers_total_count"] == 1


def test_review_sync_preserves_human_fields(dirs):
    """sync 只碰机器字段：主题、状态、时间窗、检索式、纳入标准一律不动。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    p = _seed_review(reviews, links=["[[2018_zhu_x]]"], total=0)
    before = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    research.cmd_review(_ns(action="sync"))
    after = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    for key in (
        "topic",
        "topic_slug",
        "created_date",
        "author",
        "status",
        "time_window",
        "sources_used",
        "query_strings",
        "inclusion_criteria",
        "exclusion_criteria",
    ):
        assert after[key] == before[key], key


def test_review_sync_refreshes_last_updated(dirs, monkeypatch):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    p = _seed_review(reviews, links=["[[2018_zhu_x]]"], total=0)
    monkeypatch.setattr(research, "_today", lambda: "2027-01-15")
    research.cmd_review(_ns(action="sync"))
    fm = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    assert fm["last_updated"] == "2027-01-15"
    assert fm["created_date"] == "2026-09-01"  # 创建日期不变


def test_review_sync_does_not_append_changelog(dirs):
    """sync **不追加 Changelog**（方案 F1：body 不动）。

    Changelog 记的是综述的认知进展（起草 / 审校 / 补证），而计数刷新是机械的、可能
    频繁发生，写进去会把真正值得看的条目淹掉。
    """
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    p = _seed_review(reviews, links=["[[2018_zhu_x]]"], total=0)
    research.cmd_review(_ns(action="sync"))
    _fm, body = notes.split_note(p.read_text(encoding="utf-8"))
    # 正文完全没动：第 8 节里还是模板那一条，没多出机器追加的行
    assert body == REVIEW_BODY
    assert "## 8. Changelog\n\n- 2026-09-01: 初稿由 AI 生成\n" in body


def test_review_sync_is_idempotent(dirs, capsys):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    _seed_review(reviews, links=["[[2018_zhu_x]]"], total=0)
    assert research.cmd_review(_ns(action="sync")) == 0
    capsys.readouterr()
    assert research.cmd_review(_ns(action="sync")) == 0
    assert "已是最新" in capsys.readouterr().out


def test_review_sync_unchanged_preserves_mtime(dirs):
    """无变更时**完全不触碰文件**——保留 mtime，与 ``_merge_note`` 的 ``unchanged`` 一致。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x")
    today = research._today()
    p = _seed_review(reviews, links=["[[2018_zhu_x]]"], total=1, read=0)
    # 让 last_updated 也已经是今天，否则它总会被算成一处变更
    fm = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    fm["last_updated"] = today
    p.write_text(notes.render_note(fm, REVIEW_BODY), encoding="utf-8", newline="\n")
    before = os.stat(p).st_mtime_ns
    assert research._review_sync(p) == ("unchanged", [])
    assert os.stat(p).st_mtime_ns == before


def test_review_sync_dry_run_writes_nothing(dirs, capsys):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    p = _seed_review(reviews, links=["[[2018_zhu_x]]"], total=0)
    before = p.read_bytes()
    assert research.cmd_review(_ns(action="sync", dry_run=True)) == 0
    assert p.read_bytes() == before
    out = capsys.readouterr().out
    assert "将重写 frontmatter" in out
    assert "--dry-run：未写入任何文件" in out


def test_review_sync_dry_run_reports_which_keys(dirs, capsys):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    _seed_review(reviews, links=["2018_zhu_x"], total=0)
    research.cmd_review(_ns(action="sync", dry_run=True))
    out = capsys.readouterr().out
    for key in ("papers_reviewed", "papers_total_count", "papers_read_count"):
        assert key in out


def test_review_sync_skips_unparseable_and_returns_one(dirs, capsys):
    """无 frontmatter 的人工综述：跳过、不改写、退出码 1。"""
    _papers, reviews, _sl = dirs
    p = reviews / "2026-09_handwritten_survey.md"
    original = "# 手写综述\n\n没有 frontmatter。\n"
    p.write_text(original, encoding="utf-8")
    assert research.cmd_review(_ns(action="sync")) == 1
    assert p.read_text(encoding="utf-8") == original
    assert "跳过" in capsys.readouterr().err


def test_review_sync_explicit_target(dirs):
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    good = _seed_review(
        reviews, "2026-09_a_survey.md", links=["[[2018_zhu_x]]"], total=0
    )
    other = _seed_review(reviews, "2026-10_b_survey.md", links=[], total=7)
    research.cmd_review(_ns(action="sync", target="2026-09_a_survey.md"))
    assert (
        notes.load_frontmatter(good.read_text(encoding="utf-8"))["papers_total_count"]
        == 1
    )
    # 未被指定的那份一个字节都不动
    assert (
        notes.load_frontmatter(other.read_text(encoding="utf-8"))["papers_total_count"]
        == 7
    )


def test_review_sync_missing_target_returns_two(dirs, capsys):
    assert research.cmd_review(_ns(action="sync", target="nope.md")) == 2
    assert "找不到综述" in capsys.readouterr().err


def test_review_sync_empty_dir_returns_zero(dirs, capsys):
    assert research.cmd_review(_ns(action="sync")) == 0
    assert "无可同步对象" in capsys.readouterr().out


def test_review_sync_then_status_is_clean(dirs):
    """端到端：``sync`` 之后 ``status`` 应当完全干净（退出码 0、无偏差）。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    _seed_paper(papers, "2021_gu_y", status="reading")
    _seed_review(reviews, links=["2018_zhu_x", "[[2021_gu_y]]"], total=0, read=0)
    assert research.cmd_review(_ns(action="sync")) == 0
    assert research.cmd_review(_ns(action="status")) == 0


def test_review_new_then_sync_then_status_roundtrip(dirs):
    """``new`` 产出的综述能被 ``sync`` / ``status`` 直接消费（键名不是两套）。"""
    papers, reviews, _sl = dirs
    _seed_paper(papers, "2018_zhu_x", status="read")
    research.cmd_review(_ns(action="new", target=TOPIC))
    p = next(reviews.glob("*.md"))
    fm = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    fm["papers_reviewed"] = ["[[2018_zhu_x]]"]
    _fm, body = notes.split_note(p.read_text(encoding="utf-8"))
    p.write_text(notes.render_note(fm, body), encoding="utf-8", newline="\n")
    before = _body_bytes(p)
    assert research.cmd_review(_ns(action="sync")) == 0
    assert _body_bytes(p) == before
    assert research.cmd_review(_ns(action="status")) == 0
    fm2 = notes.load_frontmatter(p.read_text(encoding="utf-8"))
    assert fm2["papers_total_count"] == 1 and fm2["papers_read_count"] == 1
