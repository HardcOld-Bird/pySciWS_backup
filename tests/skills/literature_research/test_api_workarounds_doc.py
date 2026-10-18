"""第三方学术 API 的本机已知对策（backlog 20261018-092248-lit）。

lit 摩擦：本机直连 PubMed E-utilities 报
``schannel: ... CRYPT_E_REVOCATION_OFFLINE (0x80092013)``，须加 ``--ssl-no-revoke`` 才通；
Semantic Scholar API 返回 429（共享无 key 池限流）。对策钉进 ``references/sources.md``，
后续会话不必重复试错。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.devops.tools import budget as budget_mod

REF = ORCHESTRATION_ROOT / "skills" / "literature-research" / "references"
SOURCES = REF / "sources.md"


def _md() -> str:
    """正文（空白折叠为单空格）：断言短语不受 markdown 换行位置影响。"""
    return " ".join(SOURCES.read_text(encoding="utf-8").split())


def test_pubmed_eutils_schannel_workaround_documented():
    md = _md()
    assert "eutils.ncbi.nlm.nih.gov" in md, "未点名 PubMed E-utilities"
    assert "CRYPT_E_REVOCATION_OFFLINE" in md and "0x80092013" in md, "缺失败签名"
    assert "--ssl-no-revoke" in md, "缺已知对策"
    # 对策边界：只针对单命令，不得退化为关闭 TLS 校验
    assert "do **not**" in md and "`-k`" in md, "未写明不得用 -k 关校验"


def test_semantic_scholar_rate_limit_fallback_documented():
    md = _md()
    assert "429" in md, "未点名 S2 限流"
    assert "back off" in md, "缺退避对策"
    assert "OpenAlex" in md and "Crossref" in md, "缺无 key 替代源"


def test_sources_stays_within_reference_budget():
    assert budget_mod.disk_size(SOURCES) <= budget_mod.LIMIT, (
        "按需档单文件 ≤8192B 预算被破（CRLF 等价）"
    )
