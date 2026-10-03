"""``cache_manager.referenced_paths()`` 的离线回归测试（全部在 tmp_path 下）。

护栏对象是 WP-C 修掉的第二段后果链：``referenced_paths()`` 原先用手写的逐行正则解析
frontmatter，它**跳过所有以 ``-`` 开头的行**（把 YAML block-style 的列表项误当噪声），
并且把带引号的值连同引号一起返回。两个缺陷叠加的结果是：对仓库里现存的三篇笔记，
本函数一律返回空集 → ``cache prune --keep-referenced`` 的保护**完全失效** →
一次 prune 就能把 MinerU 配额换来的 ``cache/extracted/*.md`` 全删掉（而这些文件是
git 跟踪的刻意例外，重获要再花一次配额）。

因此本文件锁定的核心契约是：

1. block-style 与 flow-style 两种真实写法的 ``local_pdf_path`` / ``extracted_md_path``
   都能被收集；
2. 相对路径按 ``settings.project_root`` 解析，且 raw 与 resolve 两种形态**都**收录
   （``_is_referenced`` 两种都会比）；
3. ``prune --keep-referenced`` 真的会跳过被引用的文件、删掉没被引用的。

WP-H（方案 H5）之后追加第 4 条：``read`` 拿不到 PDF 时只把网页正文兜底记为
``extracted_html_path``（**不**设 ``extracted_md_path``，以免污染 rag 语料），因此该字段
必须同样受保护——否则「只有 HTML 全文」的笔记会在一次 prune 里丢掉它唯一的全文。

第 5 条是同一个缺陷的**更大一处实例**：``referenced_paths()`` 只读笔记，而
``research ingest`` 的产物（实测 73 个文件，含数本教材）**根本不挂在任何笔记上**——
manifest 才是它们的唯一真源。于是整个入库语料对 ``--keep-referenced`` 不可见，而
LRU 按 mtime 从最老删起时，最老的恰恰就是它们。故保护集改由
:func:`cache_manager.protected_paths` 给出（笔记 ∪ manifest）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.skills.literature_research.tools import cache_manager, notes

BODY = "\n# body\n\n## Changelog\n\n- 2026-09-21: Created\n"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """把 cache_manager 看到的全部路径指到 tmp_path（``Settings`` 是 frozen dataclass）。"""
    papers = tmp_path / "papers"
    pdfs = tmp_path / "cache" / "pdfs"
    extracted = tmp_path / "cache" / "extracted"
    html = tmp_path / "cache" / "html_fulltext"
    for d in (papers, pdfs, extracted, html):
        d.mkdir(parents=True)
    monkeypatch.setattr(
        cache_manager,
        "settings",
        dataclasses.replace(
            cache_manager.settings,
            project_root=tmp_path,
            module_dir=tmp_path,
            cache_pdfs=pdfs,
            cache_extracted=extracted,
            cache_html_fulltext=html,
        ),
    )
    return SimpleNamespace(
        root=tmp_path, papers=papers, pdfs=pdfs, extracted=extracted, html=html
    )


def _write_note(env: SimpleNamespace, name: str, text: str) -> Path:
    path = env.papers / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _flow_note(fm: dict[str, object], body: str = BODY) -> str:
    """手写一份 flow-style 笔记（照 ``papers/2023_fang_*.md`` 的形状），不走 dump。"""
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            items = ", ".join(f'"{x}"' for x in v)
            lines.append(f"{k}: [{items}]")
        else:
            lines.append(f'{k}: "{v}"')
    lines += ["---", body.lstrip("\n")]
    return "\n".join(lines)


def _as_posix(p: Path) -> str:
    """绝对路径统一成正斜杠形态：Windows 上 ``Path`` 认它，YAML 里也无需转义反斜杠。"""
    return p.as_posix()


# ===========================================================================
#  收集契约
# ===========================================================================
def test_referenced_fields_constant():
    """``REFERENCED_FIELDS`` 是笔记引用产物的契约；加字段时须同步本测试。

    三个字段对应 ``read`` 的三种产出：下载的 PDF、PDF 抽取的全文（进 rag 语料）、
    网页正文兜底（不进 rag 语料，但仍是该篇唯一的全文）。漏掉任一个都不会报错，
    只会让那类笔记的产物在一次 ``prune`` 里静默消失。
    """
    assert cache_manager.REFERENCED_FIELDS == (
        "local_pdf_path",
        "extracted_md_path",
        "extracted_html_path",
    )


def test_collects_block_style_note(env):
    pdf = env.pdfs / "1803.04110.pdf"
    md = env.extracted / "topological_fulltext.md"
    fm = {
        "title": "Topological Edge State",
        "year": 2018,
        "authors": ["Weiwei Zhu", "Yong Li"],
        "local_pdf_path": _as_posix(pdf),
        "extracted_md_path": _as_posix(md),
    }
    _write_note(env, "2018_zhu_topological.md", notes.render_note(fm, BODY))

    refs = cache_manager.referenced_paths()

    assert Path(_as_posix(pdf)) in refs
    assert Path(_as_posix(md)) in refs
    assert pdf.resolve() in refs and md.resolve() in refs


def test_collects_flow_style_note(env):
    """回归：旧的手写正则对 flow-style 笔记返回空集，保护形同虚设。"""
    pdf = env.pdfs / "2301.00001.pdf"
    md = env.extracted / "fang_fulltext.md"
    fm = {
        "title": "Non-Hermitian Metagratings",
        "year": 2023,
        "authors": ["Xinsheng Fang", "Yong Li"],
        "local_pdf_path": _as_posix(pdf),
        "extracted_md_path": _as_posix(md),
    }
    _write_note(env, "2023_fang_metagratings.md", _flow_note(fm))

    refs = cache_manager.referenced_paths()

    # 先确认夹具确实是 flow-style（否则本测试什么也没回归到）
    text = (env.papers / "2023_fang_metagratings.md").read_text(encoding="utf-8")
    assert 'authors: ["Xinsheng Fang", "Yong Li"]' in text
    assert Path(_as_posix(pdf)) in refs
    assert Path(_as_posix(md)) in refs


def test_quotes_are_not_part_of_the_path(env):
    """旧实现把 YAML 的引号当成路径的一部分返回，于是永远匹配不上真实文件。"""
    md = env.extracted / "quoted_fulltext.md"
    fm = {"title": "T", "extracted_md_path": _as_posix(md)}
    _write_note(env, "2020_a_t.md", _flow_note(fm))

    refs = cache_manager.referenced_paths()

    assert Path(_as_posix(md)) in refs
    assert not any(str(p).startswith('"') for p in refs)


def test_relative_paths_resolve_against_project_root(env):
    """相对路径按 ``settings.project_root`` 解析，raw 与 resolve 两种形态都收录。"""
    (env.extracted / "rel_fulltext.md").write_text("x", encoding="utf-8")
    fm = {"title": "T", "extracted_md_path": "cache/extracted/rel_fulltext.md"}
    _write_note(env, "2020_a_rel.md", notes.render_note(fm, BODY))

    refs = cache_manager.referenced_paths()

    joined = env.root / "cache" / "extracted" / "rel_fulltext.md"
    assert joined in refs
    assert joined.resolve() in refs


def test_skips_blank_and_non_string_values(env):
    """空串 / null / 列表都不该被当成路径（旧实现会把 ``[]`` 拼进 Path 里炸掉）。"""
    fm = {
        "title": "T",
        "local_pdf_path": "",
        "extracted_md_path": None,
        "authors": ["A"],
    }
    _write_note(env, "2020_a_blank.md", notes.render_note(fm, BODY))
    assert cache_manager.referenced_paths() == set()


def test_skips_whitespace_only_value(env):
    fm = {"title": "T", "local_pdf_path": "   "}
    _write_note(env, "2020_a_ws.md", notes.render_note(fm, BODY))
    assert cache_manager.referenced_paths() == set()


def test_missing_papers_dir_returns_empty(tmp_path, monkeypatch):
    """``papers/`` 不存在时返回空集且不抛异常（不变量 1：降级路径静默）。"""
    monkeypatch.setattr(
        cache_manager,
        "settings",
        dataclasses.replace(
            cache_manager.settings, module_dir=tmp_path / "nope", project_root=tmp_path
        ),
    )
    assert cache_manager.referenced_paths() == set()


def test_unparsable_note_is_skipped_not_fatal(env):
    """坏文件只跳过，不让整个 prune 崩掉。"""
    _write_note(env, "broken.md", "---\ntitle: [unclosed\n---\nbody\n")
    good = env.extracted / "good_fulltext.md"
    _write_note(
        env,
        "2020_a_good.md",
        notes.render_note({"title": "T", "extracted_md_path": _as_posix(good)}, BODY),
    )

    refs = cache_manager.referenced_paths()

    assert Path(_as_posix(good)) in refs


def test_is_referenced_matches_both_raw_and_resolved_forms(env):
    md = env.extracted / "x_fulltext.md"
    _write_note(
        env,
        "2020_a_x.md",
        notes.render_note({"title": "T", "extracted_md_path": _as_posix(md)}, BODY),
    )
    refs = cache_manager.referenced_paths()

    assert cache_manager._is_referenced(Path(_as_posix(md)), refs)
    assert cache_manager._is_referenced(md, refs)
    assert cache_manager._is_referenced(md.resolve(), refs)
    assert not cache_manager._is_referenced(env.extracted / "other.md", refs)
    assert not cache_manager._is_referenced(md, set())  # 空集 → 一律不保护


# ===========================================================================
#  端到端：prune --keep-referenced 的保护真的生效
# ===========================================================================
def _seed_cache(env: SimpleNamespace) -> tuple[Path, Path]:
    """在 cache/extracted/ 下放一个被引用的文件与一个孤儿文件。"""
    kept = env.extracted / "kept_fulltext.md"
    orphan = env.extracted / "orphan_fulltext.md"
    kept.write_text("K" * 4096, encoding="utf-8")
    orphan.write_text("O" * 4096, encoding="utf-8")
    _write_note(
        env,
        "2020_a_kept.md",
        notes.render_note(
            {"title": "T", "extracted_md_path": _as_posix(kept)},
            BODY,
        ),
    )
    return kept, orphan


def test_prune_keeps_referenced_file(env):
    kept, orphan = _seed_cache(env)

    n, freed, removed, remain = cache_manager.prune_tier_a(
        max_mb=0, dry_run=False, keep_referenced=True
    )

    assert n == 1 and freed == 4096
    assert orphan in removed
    assert not orphan.exists()
    assert kept.exists()  # ← 保护生效；旧实现下这一行会失败
    assert remain == 4096


def test_prune_dry_run_deletes_nothing(env):
    kept, orphan = _seed_cache(env)

    n, freed, removed, _ = cache_manager.prune_tier_a(
        max_mb=0, dry_run=True, keep_referenced=True
    )

    assert n == 1 and freed == 4096 and removed == [orphan]
    assert kept.exists() and orphan.exists()


def test_prune_without_keep_referenced_deletes_both(env):
    """显式关掉保护时两个都删——保护是 opt-out 而非无条件。"""
    kept, orphan = _seed_cache(env)

    n, freed, _, _ = cache_manager.prune_tier_a(
        max_mb=0, dry_run=False, keep_referenced=False
    )

    assert n == 2 and freed == 8192
    assert not kept.exists() and not orphan.exists()


# ===========================================================================
#  WP-H H5：网页正文兜底（extracted_html_path）同样受保护
# ===========================================================================
def test_collects_extracted_html_path(env):
    """只有 ``extracted_html_path`` 的笔记（无 PDF 可抽）也必须被收集。

    这正是 H5 引入 ``extracted_html_path`` 时的连带风险：新字段若不进
    ``REFERENCED_FIELDS``，``cmd_read`` 的兜底产物（``cache/html_fulltext/`` 下）就成了
    无人认领的孤儿，而它是那篇论文**唯一**的全文。
    """
    html = env.html / "10-1000-x" / "10-1000-x.md"
    html.parent.mkdir(parents=True)
    html.write_text("# body", encoding="utf-8")
    _write_note(
        env,
        "2020_a_htmlonly.md",
        notes.render_note(
            {
                "title": "T",
                "extracted_md_path": "",
                "extracted_html_path": _as_posix(html),
            },
            BODY,
        ),
    )

    refs = cache_manager.referenced_paths()

    assert Path(_as_posix(html)) in refs
    assert html.resolve() in refs


def test_prune_keeps_referenced_html_fallback(env):
    """端到端：``cache/html_fulltext/`` 里被笔记引用的 bundle 不被 prune 删掉。

    注意淘汰单元的粒度差异：``pdfs/`` 与 ``extracted/`` 以**单文件**为单元，而
    ``html_fulltext/`` 以**整个 ``<slug>/`` 目录**为单元（见 ``_collect_tier_a_units``，
    引用判定对该目录下所有文件取或），故 ``removed`` 里是目录而不是那份 ``.md``。
    这对 H5 反而正好：``extracted_html_path`` 指向 bundle 里的正文，就顺带保住了同目录
    的补充材料——它们是同一次浏览器抓取的产物，分开淘汰没有意义。
    """
    kept = env.html / "10-1000-y" / "10-1000-y.md"
    kept.parent.mkdir(parents=True)
    kept.write_text("K" * 4096, encoding="utf-8")
    orphan = env.html / "10-1000-z" / "10-1000-z.md"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("O" * 4096, encoding="utf-8")
    _write_note(
        env,
        "2020_a_htmlkept.md",
        notes.render_note({"title": "T", "extracted_html_path": _as_posix(kept)}, BODY),
    )

    n, freed, removed, _ = cache_manager.prune_tier_a(
        max_mb=0, dry_run=False, keep_referenced=True
    )

    assert n == 1 and freed == 4096
    assert orphan.parent in removed and not orphan.exists()
    assert kept.exists() and kept.parent.exists()


# ===========================================================================
#  ingest 清单：prune 的第二个保护来源
# ===========================================================================
def _write_manifest(env: SimpleNamespace, entries: list[object]) -> Path:
    """照 ``ingest/manifest.json`` 的真实形状写一份最小清单。

    夹具已把 ``module_dir`` 指到 ``tmp_path``，故路径是 ``env.root/ingest/manifest.json``。
    只写 ``version`` 与 ``entries`` 两个键：那是本模块唯一读的部分。
    """
    p = env.root / "ingest" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return p


def test_manifest_products_are_collected(env):
    """``md_path`` / ``pdf_path`` 都收，raw 与 resolve 两种形态都收（与笔记那条同规则）。"""
    md = env.extracted / "group_pubs" / "fang_ep_metagrating.md"
    pdf = env.pdfs / "group_pubs" / "fang_ep_metagrating.pdf"
    for p in (md, pdf):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    _write_manifest(
        env,
        [
            {
                "id": "group_pubs/fang",
                "status": "done",
                "md_path": _as_posix(md),
                "pdf_path": _as_posix(pdf),
            }
        ],
    )

    refs = cache_manager.ingest_manifest_paths()

    assert Path(_as_posix(md)) in refs and md.resolve() in refs
    assert Path(_as_posix(pdf)) in refs and pdf.resolve() in refs


def test_prune_keeps_an_ingest_product_that_no_note_references(env):
    """**核心回归**：ingest 产物不挂在任何笔记上，只认笔记就会把它删掉。

    这就是仓库里的真实形态：``cache/extracted/`` 下几十个由 ``research ingest`` 产出的
    文件（含数本 >150 页的教材），而 ``papers/`` 只有三篇笔记、``extracted_md_path``
    全空。它们全是花 MinerU 配额换的，且是 git 跟踪的刻意例外——被 LRU 按 mtime
    删掉时，最老的入库批次恰恰排在最前面。
    """
    ingested = env.extracted / "group_pubs" / "fang_ep_metagrating.md"
    ingested.parent.mkdir(parents=True, exist_ok=True)
    ingested.write_text("I" * 4096, encoding="utf-8")
    orphan = env.extracted / "orphan_fulltext.md"
    orphan.write_text("O" * 4096, encoding="utf-8")
    _write_manifest(env, [{"id": "group_pubs/fang", "md_path": _as_posix(ingested)}])

    # 刻意不写任何笔记：ingest 产物本来就不挂在笔记上。这一行同时钉住「只靠
    # referenced_paths 确实保护不到」，否则下面的存活断言可能被别的机制偶然满足。
    assert cache_manager.referenced_paths() == set()

    n, freed, removed, _ = cache_manager.prune_tier_a(
        max_mb=0, dry_run=False, keep_referenced=True
    )

    assert n == 1 and freed == 4096
    assert orphan in removed and not orphan.exists()
    assert ingested.exists(), "manifest 登记过的产物必须受保护"


def test_protected_paths_is_the_union_of_both_sources(env):
    """``protected_paths`` = 笔记 ∪ manifest；而 ``referenced_paths`` 仍**只**是笔记。

    不把 manifest 合进 ``referenced_paths`` 是为了保住它的语义（「笔记引用了什么」）：
    本文件有三处 ``== set()`` 的断言靠它成立，混进另一个来源会让那些断言失去意义。
    """
    md = env.extracted / "from_manifest.md"
    md.write_text("x", encoding="utf-8")
    noted = env.extracted / "from_note.md"
    noted.write_text("x", encoding="utf-8")
    _write_manifest(env, [{"md_path": _as_posix(md)}])
    _write_note(
        env,
        "2020_a_n.md",
        notes.render_note({"title": "T", "extracted_md_path": _as_posix(noted)}, BODY),
    )

    refs = cache_manager.referenced_paths()
    prot = cache_manager.protected_paths()

    assert Path(_as_posix(noted)) in refs
    assert Path(_as_posix(md)) not in refs, "manifest 不得混进笔记那一层的语义"
    assert Path(_as_posix(md)) in prot and Path(_as_posix(noted)) in prot


def test_manifest_entries_without_paths_are_ignored(env):
    """``pending`` / ``skipped`` 条目没有 ``md_path``，不得产出幽灵路径或炸掉。

    真实 manifest 里另有一份 skipped 列表，元素只有 ``source`` / ``kind`` / ``status``；
    ``pending`` 条目则连 ``pdf_path`` 都还没回填。照单全收会把 ``None`` 拼进 ``Path``。
    """
    _write_manifest(
        env,
        [
            {"id": "a", "status": "pending"},
            {"id": "b", "md_path": "", "pdf_path": None},
            {"source": "x.nb", "kind": "nb", "status": "skipped"},
            "not-a-dict",
        ],
    )

    assert cache_manager.ingest_manifest_paths() == set()


def test_manifest_relative_paths_resolve_against_project_root(env):
    """相对路径按 ``settings.project_root`` 解析，与笔记那条来源同一套规则。"""
    (env.extracted / "rel.md").write_text("x", encoding="utf-8")
    _write_manifest(env, [{"md_path": "cache/extracted/rel.md"}])

    refs = cache_manager.ingest_manifest_paths()

    joined = env.root / "cache" / "extracted" / "rel.md"
    assert joined in refs and joined.resolve() in refs


@pytest.mark.parametrize(
    "raw",
    ["", "not json at all", "[]", '{"entries": "nope"}', '{"entries": [1, null]}'],
)
def test_a_missing_or_broken_manifest_degrades_to_an_empty_set(env, raw: str) -> None:
    """manifest 缺失 / 损坏 / 形状不符 ⇒ 空集，**绝不抛**。

    本函数在保护路径上：抛异常会让「保护」变成「拒绝清理」，那比不保护更糟（不变量 1）。
    这也是为何不复用 ``local_ingest.load_manifest``——那个「缺失就报错」的语义在这里是
    错的（除了 import 成环之外另一个理由）。

    参数里的空串代表「根本没这个文件」。
    """
    if raw:
        p = env.root / "ingest" / "manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw, encoding="utf-8")

    assert cache_manager.ingest_manifest_paths() == set()
    # 退化后保护集必须恰好回到「只有笔记」，而不是整体失效或整体报错
    assert cache_manager.protected_paths() == cache_manager.referenced_paths()
