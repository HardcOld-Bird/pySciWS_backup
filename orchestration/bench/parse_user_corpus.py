# /// script
# requires-python = ">=3.13"
# ///
"""一次性入库：用户 prompt 语料（倒序文本）→ ``state/preference-journal.jsonl``。

E6b（README §6 决策日志来源②）twin 训练原料的最大增量源。语料在
``data/others/prompts_storage/``（含个人表述，故该目录 ``.gitignore`` 排除——只把
**截断引用**誊进跟踪的 journal，原文永不出本机）。**属一次性搬运工具，放
``orchestration/bench/`` 不落 src**（同 ``backfill_preference_journal.py`` 的纪律）。

语料格式规律（用户口述 2026-10-12，解析规则的唯一权威）：

- 文件**倒序**：开头较新、末尾较旧；
- **连续 ≥2 空行**（≥3 个换行符）= 消息分隔符（上=new 下=old）；
- 单空行/单换行**不是**分隔，仅同一条消息内的排版续写；
- 实测三文件均为 CRLF 行尾 → 先归一成 LF 再切（否则 ``\\n{3,}`` 判据永不命中）；
- 解析后输出**正序**（旧→新）时间线。

每条消息 → 一条 journal 记录：

    {"ts": "corpus:<全局正序号>-F<文件号>#<文件内正序号>", "situation": 首行摘要,
     "options": "", "choice": 原文截断引用, "reason": "",
     "source": "user-corpus", "kind": "prompt",
     "corpus_file": 文件名, "corpus_rank": 文件内正序号, "full_chars": 原文字数,
     "truncated": 是否截断}

``corpus_rank`` **从文件末尾（最旧端）起算**：用户在文件顶部追加新消息时，既有记录的
编号不变——故幂等去重按 ``(corpus_file, corpus_rank)``，增量入库只补新消息。
文件号=文件名前导数字（1=近期…3=早期），时段旧→新即文件号**降序**。

用法（worktree 内借主 venv，语料在 gitignored 主树，以绝对路径只读引用）::

    PYTHONPATH="<wt>/src" python orchestration/bench/parse_user_corpus.py \\
        [--corpus-dir <主树>/data/others/prompts_storage] [--max-chars 400] \\
        [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import journal

#: 语料目录（默认为主树绝对路径——worktree 不含 gitignored 的语料）。
CORPUS_DIR = PROJECT_ROOT / "data" / "others" / "prompts_storage"

SOURCE = "user-corpus"
KIND = "prompt"

#: ``choice`` 截断上限（与 backfill 同口径）：journal 是审计与训练原料层，
#: 不是原文镜像；个人表述全文留在 gitignored 语料里。
DEFAULT_MAX_CHARS = 400

#: 首行摘要上限（``situation``）。
SITUATION_MAX = 90

#: 消息分隔符=连续 ≥2 空行（≥3 个换行符）；空行允许只含空格/制表符。
SEP_RE = re.compile(r"\n(?:[ \t]*\n){2,}")

#: 文件前导数字=时段号（无数字前缀的文件排到最后，由人工改名归队）。
FILE_NO_RE = re.compile(r"^\s*(\d+)")


def normalize_eol(text: str) -> str:
    """CRLF/CR → LF（语料实测全为 CRLF，不归一则 ``\\n{3,}`` 判据永不命中）。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_messages(text: str) -> list[str]:
    """按 ≥2 空行切分，返回**文内顺序**（新→旧）的消息列表（空白块丢弃）。"""
    return [p.strip() for p in SEP_RE.split(normalize_eol(text)) if p.strip()]


def file_no(name: str) -> int:
    """文件名前导数字（时段号）；缺失返回 0（排在所有编号文件之后）。"""
    m = FILE_NO_RE.match(name)
    return int(m.group(1)) if m else 0


def parse_file(path: Path) -> list[dict]:
    """单文件 → **正序**（旧→新）消息骨架列表。

    ``rank`` 从文件末尾（最旧端）起算，故用户在顶部追加新消息不影响既有编号。
    """
    msgs = split_messages(path.read_text(encoding="utf-8-sig"))
    total = len(msgs)
    return [
        {
            "file": path.name,
            "fileno": file_no(path.name),
            "rank": rank,
            "total": total,
            "text": text,
        }
        for rank, text in enumerate(reversed(msgs))
    ]


def restore_timeline(paths: list[Path]) -> list[dict]:
    """多文件 → 全语料**正序**时间线：时段号降序拼接（F3 早期 → F1 近期），补全局 ``seq``。"""
    ordered = sorted(paths, key=lambda p: (-file_no(p.name), p.name))
    msgs = [m for p in ordered for m in parse_file(p)]
    for seq, m in enumerate(msgs, 1):
        m["seq"] = seq
    return msgs


def first_line(text: str) -> str:
    """首个非空行（消息通常以称呼/问题开头，即该条的 situation 摘要源）。"""
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def to_record(m: dict, *, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """消息骨架 → journal 记录（ts 自带「相对时序标注 + 文件锚」）。"""
    text = m["text"]
    truncated = len(text) > max_chars
    return {
        "ts": f"corpus:{m['seq']:04d}-F{m['fileno']}#{m['rank']:03d}",
        "situation": first_line(text)[:SITUATION_MAX],
        "options": "",
        "choice": text[:max_chars] + ("…" if truncated else ""),
        "reason": "",
        "source": SOURCE,
        "kind": KIND,
        "corpus_file": m["file"],
        "corpus_rank": m["rank"],
        "full_chars": len(text),
        "truncated": truncated,
    }


def _key(rec: dict) -> tuple:
    """幂等去重键（文件锚 + 末尾起算的正序号；对顶部追加稳定）。"""
    return (rec.get("corpus_file"), rec.get("corpus_rank"))


def discover(corpus_dir: Path) -> list[Path]:
    """语料文件（``*.txt``，按时段号降序→升序名）；目录缺失或空返回空表。"""
    if not corpus_dir.is_dir():
        return []
    return sorted(corpus_dir.glob("*.txt"), key=lambda p: (-file_no(p.name), p.name))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="用户 prompt 语料倒序解析入库（E6b）")
    ap.add_argument(
        "--corpus-dir", default=str(CORPUS_DIR), help="语料目录（默认主树）"
    )
    ap.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="choice 截断"
    )
    ap.add_argument(
        "--force", action="store_true", help="重生成 user-corpus 段（先去旧再写新）"
    )
    ap.add_argument("--dry-run", action="store_true", help="只解析与对账，不落盘")
    args = ap.parse_args(argv)

    corpus_dir = Path(args.corpus_dir)
    files = discover(corpus_dir)
    if not files:
        print(f"[!] 语料目录无 *.txt：{corpus_dir}")
        return 2

    msgs = restore_timeline(files)
    per_file = {}
    for m in msgs:
        per_file[m["file"]] = m["total"]
    print(f"解析 {len(files)} 文件 → {len(msgs)} 条消息（正序 旧→新）")
    for f in sorted(files, key=lambda p: file_no(p.name)):
        print(f"  F{file_no(f.name)} {f.name}：{per_file.get(f.name, 0)} 条")

    records = [to_record(m, max_chars=args.max_chars) for m in msgs]
    target = journal.JOURNAL_PATH
    before = journal.line_count(target)

    if args.dry_run:
        for r in records[:2] + records[-1:]:
            print(f"  e.g. {r['ts']} → {r['situation'][:40]}")
        print(f"[dry-run] 目标 {target}（现 {before} 行），不写入")
        return 0

    existing = {
        _key(r) for r in journal.read_records(target) if r.get("source") == SOURCE
    }
    if args.force and existing:
        others = [r for r in journal.read_records(target) if r.get("source") != SOURCE]
        target.write_text("", encoding="utf-8")
        for r in others:
            journal.append(r, path=target)
        existing = set()
        print(f"[i] --force：清 {len(others)} 条其他来源后重写（保留非语料段）")

    written = 0
    for r in records:
        if _key(r) in existing:
            continue
        journal.append(r, path=target)
        written += 1

    after = journal.line_count(target)
    delta = after - before
    print(f"[√] 新写入 {written} 条；{target.name} 现共 {after} 行（原 {before} 行）")
    if delta != written:
        print(f"[!] 对账失败：journal 新增 {delta} 行 ≠ 解析写入 {written} 条")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
