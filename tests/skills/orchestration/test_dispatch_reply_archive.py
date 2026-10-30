"""归档审批回复的 fail-soft（backlog 20261015-132939-fix3；实案 job 20261015-122406）。

缺陷链（2026-10-15 实发）：reply 源文件已被人工/上一跳移走 → 裸 ``shutil.move`` 抛
``FileNotFoundError`` **逃出** ``do_dispatch`` → 整个**批次**异常退出 → 「已派发、未结账」
的条目随后被重派 → 同一隔离面出现两个并发活跃写者（下游即 tui-ux 的
``<blocked tags="concurrent_writer">``）。

判据：**源缺失＝已归档**，幂等跳过；任何 ``OSError`` 都不逃出归档面——归档只是收拾目录，
回复正文早已随任务书附送、也已在交付正文与台账内务面。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.orchestration.tools import dispatch


def _replies(tmp_path: Path) -> tuple[Path, list[Path]]:
    d = tmp_path / "replies" / "devops"
    d.mkdir(parents=True)
    p = d / "20261015-085823-devops.md"
    p.write_text("审批回复正文\n", encoding="utf-8")
    return d, [p]


def test_archives_existing_reply_into_sent(tmp_path):
    d, files = _replies(tmp_path)
    dispatch._archive_replies(files)
    assert not files[0].exists(), "源应被移走"
    assert (d / "sent" / files[0].name).read_text(encoding="utf-8") == "审批回复正文\n"


def test_missing_source_is_an_idempotent_skip(tmp_path):
    """实案同款：源已在 sent/ 侧、源文件已被移走 → **不再抛**，只留痕一行。"""
    d, files = _replies(tmp_path)
    (d / "sent").mkdir()
    (d / "sent" / files[0].name).write_text("审批回复正文\n", encoding="utf-8")
    files[0].unlink()  # 人工/上一跳已移走
    traced: list[tuple[str, dict]] = []

    dispatch._archive_replies(
        files, trace=lambda stage, **kw: traced.append((stage, kw))
    )

    assert [t[0] for t in traced] == ["reply_already_archived"]
    assert (d / "sent" / files[0].name).read_text(encoding="utf-8") == "审批回复正文\n"


def test_any_oserror_is_swallowed_with_trace(tmp_path, monkeypatch):
    """exists() 之后被抢走（race）等其余 OSError 同样不逃出——归档面绝不反噬派发。"""
    _d, files = _replies(tmp_path)

    def boom(*_a, **_k):
        raise FileNotFoundError("源在 exists() 之后被抢走")

    monkeypatch.setattr(dispatch.shutil, "move", boom)
    traced: list[str] = []
    dispatch._archive_replies(files, trace=lambda stage, **kw: traced.append(stage))

    assert traced == ["reply_archive_skipped"]


# ---------------------------------------------------------------------------
# 派发级阳性用例（backlog 20261015-131302-devops 判据②：驱动一次 dispatch 收尾）
# ---------------------------------------------------------------------------
MEMBER = "quotamember"
OK = "<result>完工，commit abc1234 已合并。</result>"
BUILTIN = "Qwen3.8-Flash"


def test_dispatch_settle_survives_a_missing_reply_source(
    iso_dispatch, monkeypatch, tmp_path
):
    """**实案同款**：待归档的源已被移走（``sent/`` 侧已在）→ `do_dispatch` 照常收尾、**不抛**。

    旧代码在此从归档循环里抛出 ``FileNotFoundError`` → 逃出 ``do_dispatch`` → 批次异常退出
    （job 20261015-122406 无 ``.done``）→ 条目坠落重派 → 双写者。本用例把判据钉在**派发级**
    （不只是归档 helper）：驱动一次完整的 dispatch 收尾，断言返回正常。
    """
    from .conftest import envelope, fake_headless, set_models

    set_models(iso_dispatch, flash=BUILTIN)
    fake_headless(monkeypatch, lambda n, cmd: envelope(OK, is_error=False, rc=0))
    replies = iso_dispatch / "replies" / MEMBER
    (replies / "sent").mkdir(parents=True, exist_ok=True)
    stale = replies / "already-archived.md"
    (replies / "sent" / stale.name).write_text("回复正文\n", encoding="utf-8")
    assert not stale.exists(), "前提：源已被移走、sent/ 侧已在（实案形态）"
    monkeypatch.setattr(dispatch, "collect_replies", lambda _mid: ("回复正文", [stale]))

    o = dispatch.do_dispatch(MEMBER, text="干活", quiet=True, no_checks=True)

    assert o.code == 0 and o.kind == "result", (
        f"归档面不得反噬派发（实案：FileNotFoundError 逃出 → 批次崩 + 无 .done）——实得 {o}"
    )
