"""派发面标签纠正（Tier-A 20261018-212801-tier-a）。

实案：devops 的 `whitelist_conflict` 在 14 天/20 跳窗内 5 次达阈值；其中 3 次实为
merge 交集 / 文档面失败——词表 `whitelist_conflict` 定义明确「只归写权限面冲突，merge
交集用 merge_dirty_face」。本文件钉住机械执行面：

- 交付正文含 `[✗] 交集` 且标签为 `whitelist_conflict` → 台账/结果改为
  `merge_dirty_face`，并把原件写进 `tag-corrections.jsonl`（审计可回溯）；
- 无该证据的真实写权限冲突**原样保留**（不猜）。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch

from .conftest import envelope, fake_headless


def _blocked(tags: str, body: str) -> str:
    return f'<blocked tags="{tags}">\n{body}\n</blocked>'


def _run(iso_dispatch, monkeypatch, result: str):
    fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(result, is_error=False, stop="end_turn", rc=0),
    )
    outcome = dispatch.do_dispatch("quotamember", text="t", quiet=True, no_checks=True)
    rows = [
        json.loads(line)
        for line in (iso_dispatch / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return outcome, rows[-1]


def test_merge_intersection_is_reclassified(iso_dispatch, monkeypatch):
    body = "卡点（merge rc=2）\n[✗] 交集 1 项——合并会覆盖 main 工作区的未提交改动\n"
    outcome, row = _run(iso_dispatch, monkeypatch, _blocked("whitelist_conflict", body))

    assert outcome.tags == ["merge_dirty_face"], "结果面用纠正后的标签"
    assert row["tags"] == ["merge_dirty_face"], "台账（Tier A 计数源）用纠正后的标签"
    corrections = iso_dispatch / "tag-corrections.jsonl"
    rec = json.loads(corrections.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["tags_original"] == ["whitelist_conflict"], "原件留痕"
    assert rec["tags_corrected"] == ["merge_dirty_face"]


def test_genuine_write_conflict_is_untouched(iso_dispatch, monkeypatch):
    body = "pod-guard 拦下：目标路径不在本 pod 允许写面内"
    outcome, row = _run(iso_dispatch, monkeypatch, _blocked("whitelist_conflict", body))

    assert outcome.tags == ["whitelist_conflict"]
    assert row["tags"] == ["whitelist_conflict"]
    assert not (iso_dispatch / "tag-corrections.jsonl").exists(), "不得无证据改判"
