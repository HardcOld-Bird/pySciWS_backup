# /// script
# requires-python = ">=3.13"
# ///
"""一次性回灌：README §13 裁决记录表（现 `references/decisions.md`）→ ``state/preference-journal.jsonl``（README §6 判据纪律/E0 判据 3）。

决策日志的存量原料层。**属一次性搬运工具，放 ``orchestration/bench/`` 不落 src**（设计纪律：
src 只定义可复用行为，此处是把文档表格里已存在的裁决誊进 JSONL，非体系运行时逻辑）。

每行 README §13 表格 → 一条 journal 记录：

    {"ts": 日期, "situation": "README §13 裁决记录（<日期>）", "options": "",
     "choice": <裁决原文摘要>, "reason": "", "source": "backfill-readme13",
     "kind": "ruling", "decider": <裁决人>}

幂等：默认跳过文件里已存在的 ``source=backfill-readme13`` 行（按 (ts, choice) 去重），
只补新增；``--force`` 则重生成整个 backfill 段（保留其他 source 的行）。

用法（worktree 内借主 venv）::

    PYTHONPATH="<wt>/src" python orchestration/bench/backfill_preference_journal.py [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import journal

#: §13 裁决记录表已迁 ``orchestration/references/decisions.md``（README v3.0 精简重写，2026-10-15）。
README_PATH = PROJECT_ROOT / "orchestration" / "references" / "decisions.md"
SOURCE = "backfill-readme13"


def parse_readme13(md_text: str) -> list[dict]:
    """抽取 §13 表格数据行（``| 2026…`` 开头），逐行 → journal 记录骨架。"""
    records: list[dict] = []
    in_section = False
    for line in md_text.splitlines():
        if line.startswith("## 13."):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not (in_section and line.startswith("| 2026")):
            continue
        # 固定三列表格："| 日期 | 裁决 | 裁决人 |" → 去首尾竖线后按竖线切三格。
        inner = line.strip().strip("|")
        cols = [c.strip() for c in inner.split("|")]
        date, ruling, decider = (cols + ["", "", ""])[:3]
        records.append(
            {
                "ts": date,
                "situation": f"README §13 裁决记录（{date}）",
                "options": "",
                "choice": ruling[:400],
                "reason": "",
                "source": SOURCE,
                "kind": "ruling",
                "decider": decider,
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="回灌 README §13 裁决记录到决策日志")
    ap.add_argument(
        "--force", action="store_true", help="重生成 backfill 段（先去旧再写新）"
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将写入的行数，不落盘")
    args = ap.parse_args(argv)

    if not README_PATH.exists():
        print(f"[!] README 缺失：{README_PATH}")
        return 2
    records = parse_readme13(README_PATH.read_text(encoding="utf-8"))
    print(f"README §13 解析出 {len(records)} 条裁决记录")

    target = journal.JOURNAL_PATH
    if args.dry_run:
        for r in records[:3]:
            print("  e.g. " + r["ts"] + " → " + r["choice"][:60])
        print(f"[dry-run] 目标 {target}，不写入")
        return 0

    # --force：保留非 backfill 行，重写 backfill 段（去旧防重复）。
    if args.force and target.exists():
        others = [r for r in journal.read_records(target) if r.get("source") != SOURCE]
        target.write_text("", encoding="utf-8")
        for r in others:
            journal.append(r, path=target)
        print(
            f"[i] --force：清 {len(records)} 条旧 backfill，保留 {len(others)} 条其他来源"
        )

    existing = {(r.get("ts"), r.get("choice")) for r in journal.read_records(target)}
    written = 0
    for r in records:
        if (r["ts"], r["choice"]) in existing:
            continue
        journal.append(r, path=target)
        written += 1
    total = journal.line_count(target)
    print(f"[√] 新写入 {written} 条；{target} 现共 {total} 行")
    if total < len(records):
        print(f"[!] 行数 {total} < §13 条目 {len(records)}——判据 3 未满足")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
