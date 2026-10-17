"""非学术网页的取证规范文档面（backlog 20261016-025806-lit）。

lit 摩擦：为 PPT 截取 OpenAI / Anthropic 官方发布页时，`openai.com` 反爬一律 403、
`docs.anthropic.com` 地区不可用、basic.md §4 的 fallback 代理当时也不通（curl exit 7）；
官方一手页取不到只能降级到二手互证，且「哪些是直取、哪些是二手互证」没有统一口径，
每单都要自行摸索。

本文件钉住 `references/fetch-evidence.md` 把口径写成机械可查的两件事：
- **两个标签**：``[直取]`` 与 ``[二手互证]`` 的定义与「不得把二手当直取」；
- **出路必须写死**：403 / 地区限制两类失败、代理启用前置（引用 basic.md §4，不复制）、
  可接受的替代来源清单与不可接受清单。

文在、且从 `sources.md` 可一步指到（SKILL.md 顶部预算已满，索引入口走 sources.md）。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.devops.tools import budget as budget_mod

REF = ORCHESTRATION_ROOT / "skills" / "literature-research" / "references"
SPEC = REF / "fetch-evidence.md"
SOURCES = REF / "sources.md"


def _text(p) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ① 两个取证标签的口径
# ---------------------------------------------------------------------------
def test_spec_exists_and_states_both_labels():
    md = _text(SPEC)
    assert "[直取]" in md, "缺一手直取标签"
    assert "[二手互证]" in md, "缺二手互证标签"
    # 核心禁令：不得把二手当一手（这是「统一口径」的意义所在）
    assert "Never present a" in md and "[二手互证]" in md
    # 二手互证必须 ≥2 个独立权威来源互证
    assert "≥2" in md and "independent" in md


def test_spec_names_the_two_failure_classes():
    md = _text(SPEC)
    assert "403" in md, "未点名反爬 403"
    assert "openai.com" in md and "developers.openai.com" in md
    assert "claude.com/app-unavailable-in-region" in md, "未点名地区不可用现象"


# ---------------------------------------------------------------------------
# ② 代理前置条件：引用 basic.md §4，不另立一份、不许持久化
# ---------------------------------------------------------------------------
def test_spec_routes_proxy_rules_to_basic_md_not_a_copy():
    md = _text(SPEC)
    assert "127.0.0.1:7890" in md, "未给出 fallback 代理地址"
    assert "basic.md" in md and "§4" in md, "代理纪律必须回指 basic.md §4（单一真源）"
    assert "single-command" in md, "未写明仅单命令级注入"
    assert "never persisted" in md or "禁止持久化" in md, "未写明禁止持久化"
    # 代理不通（curl exit 7）时不得重试循环，改走替代取证
    assert "exit 7" in md


# ---------------------------------------------------------------------------
# ③ 替代取证：可接受清单 + 不可接受清单
# ---------------------------------------------------------------------------
def test_spec_gives_acceptable_and_unacceptable_sources():
    md = _text(SPEC)
    assert "web.archive.org" in md, "缺存档快照路径"
    assert "github.com" in md, "缺「官方仅存于可直达的官方面」路径"
    low = md.lower()
    assert "content farm" in low, "缺不可接受来源（SEO 农场）"
    assert "未取证" in md or "not acceptable" in low, "缺「够不上二手互证即标未取证」"


# ---------------------------------------------------------------------------
# ④ 可达性 + 预算：从 sources.md 一步可指，且分册自身不超预算
# ---------------------------------------------------------------------------
def test_sources_indexes_the_spec():
    assert "fetch-evidence.md" in _text(SOURCES), "sources.md 未索引取证规范"


def test_spec_within_reference_budget():
    assert budget_mod.disk_size(SPEC) <= budget_mod.LIMIT, (
        "按需档单文件 ≤8192B 预算被破（CRLF 等价）"
    )
