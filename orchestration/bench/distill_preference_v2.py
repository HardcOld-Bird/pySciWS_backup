# /// script
# requires-python = ">=3.13"
# ///
"""一次性蒸馏跳：用户 prompt 语料 → ``PREFERENCE.md`` v2 **diff 草案**（E6b）。

批准单（``state/suggestions/approved/20261012-164601-leader.md``）规格 2：twin 判断通道
未生效（``taste_enabled=false``，``orch taste`` 拒绝派发），故由本脚本**复用 taste 的派发
机械**（宪章注入 + 冷启动单跳 + ``eval_hop`` 不入生产台账）跑一次蒸馏，但输出**只落草案
文件**——生效宪章 ``pods/twin/PREFERENCE.md`` 一字不动，等用户确认。

与 ``taste.judge`` 的差异只在两处，其余同法：

- 案卷不是「二选一」而是「读语料写 diff」，故输出格式由提示词硬覆盖（twin 规程里的
  JSON 表态格式在这条支路不适用）；
- 留痕不落 ``state/evals/taste/``（那是一案一判的盲审料），只落 ``--out`` 指定的草案。

冷启动纪律照旧：``session="new"`` 且 twin 的 registry ``cold_start`` 会在 dispatch 层
拒绝任何非 new 会话。twin 无写权限（不配 ``extra_dirs``），产出物只能经 ``<result>``
正文回收，由**本脚本**落盘——草案是 devops 写的，不是 twin 改自己的宪章。

用法（在 main 树上跑，pod 的 gitignored 部署面齐全；语料默认即主树绝对路径）::

    PYTHONPATH="<wt>/src" python orchestration/bench/distill_preference_v2.py --dry
    PYTHONPATH="<wt>/src" python orchestration/bench/distill_preference_v2.py \\
        --out orchestration/pods/twin/PREFERENCE-v2-draft.md
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools.dispatch import do_dispatch
from pysci.skills.orchestration.tools.presets import normalize_role, resolve_role
from pysci.skills.orchestration.tools.registry import Registry
from pysci.skills.orchestration.tools.taste import PREFERENCE_NAME, preference_text

#: 生效宪章（只读基线）与草案落点默认值。
EFFECTIVE = PROJECT_ROOT / "orchestration" / "pods" / "twin" / PREFERENCE_NAME
DEFAULT_DRAFT = (
    PROJECT_ROOT / "orchestration" / "pods" / "twin" / "PREFERENCE-v2-draft.md"
)
CORPUS_DIR = PROJECT_ROOT / "data" / "others" / "prompts_storage"

#: 批准单规格 2 点名「flash 冷启动会话」（旧档名 flash 归一即 ``main``）。默认必须显式给
#: main：成员档位会随预设的 pro 名单漂走（2026-10-13 实测不传时 twin 落 pro），而成本
#: 敏感是护栏级偏好（宪章 P-03）——不靠运气继承。
DEFAULT_TIER = "main"

#: 交付协议标签（``<artifact>`` 等）不属于宪章正文，落盘前剥离——否则用户把草案并进
#: 生效面时要把协议噪声一起搬过去。
PROTOCOL_TAG_RE = re.compile(r"\n?<artifact\b[^>]*>[\s\S]*?</artifact>\s*$")

#: 草案首行硬标记（delivery 里也要求 twin 原样带上，双保险：肉眼与 grep 都能判非生效）。
DRAFT_MARK = "<!-- PREFERENCE v2 草案 · 待用户确认 · 不生效（生效面=pods/twin/PREFERENCE.md） -->"


def _load_parser():
    """加载同目录的语料解析脚本——分隔规则只有一个真源，蒸馏侧不另写一套切分。"""
    src = Path(__file__).with_name("parse_user_corpus.py")
    spec = importlib.util.spec_from_file_location("parse_user_corpus", src)
    assert spec and spec.loader, f"解析脚本缺失：{src}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_slice(puc, corpus_dir: Path, fileno: int, newest: int) -> list[dict]:
    """取某时段文件的**最新 N 条**（默认前半）：``parse_file`` 已还原成正序，切片取尾部。"""
    hits = [p for p in puc.discover(corpus_dir) if puc.file_no(p.name) == fileno]
    if not hits:
        raise SystemExit(f"[!] 语料目录里没有时段号 {fileno} 的文件：{corpus_dir}")
    msgs = puc.parse_file(hits[0])
    take = newest if newest > 0 else math.ceil(len(msgs) / 2)
    return msgs[-take:]


def build_prompt(pref_v1: str, slice_msgs: list[dict], *, corpus_file: str) -> str:
    """组装注入正文：v1 基线 → 语料原文 → 蒸馏任务 → 硬输出格式（同 taste.build_prompt 的次序理由）。"""
    body = "\n\n".join(
        f"### 语料 #{m['rank'] + 1:03d}"
        f"（{corpus_file} 正序 {m['rank'] + 1}/{m['total']}，旧→新）\n"
        f"{m['text']}"
        for m in slice_msgs
    )
    return (
        "# 本次任务：从用户语料蒸馏判断宪章 v2（twin 蒸馏跳）\n\n"
        "这一跳由批准单 20261012-164601-leader 派发。**冷启动、无记忆**照旧；但本跳不要求"
        "二选一表态，**下面「输出要求（硬格式）」覆盖你规程里的 JSON 判断格式**——产出是"
        "草案文本本身，不是判决。\n\n"
        "## 现行宪章 v1（diff 基线，用户已审阅确认，逐字引用条款号）\n\n"
        f"{pref_v1.strip()}\n\n"
        f"## 用户 prompt 语料（{corpus_file}，已按倒序还原为**正序** 旧→新，"
        f"共 {len(slice_msgs)} 条=该文件最新的一半）\n\n"
        "语料原文含大量一次性事务指令（「帮我改这个文件」类），那些**不构成偏好**；"
        "要找的是反复出现的取舍、标准、语气与禁令。\n\n"
        f"{body}\n\n"
        "## 任务\n\n"
        "1. 通读语料，找出 v1 未覆盖或覆盖不充分的**稳定偏好**；\n"
        "2. 产出 v1 的 **v2 增量草案**——只写 diff，不重写 v1：\n"
        "   - 【新增】续编条款号（P-22 起），沿用 v1 的分区（A 价值序 / B 决策形式 /\n"
        "     C 工程文档纪律 / D 观测验收安全 / E 判例 / F 已知未决）与写法密度；\n"
        "   - 【修订】对既有条款的收紧或补充，写明「P-xx 原文 → 新表述」；\n"
        "   - 【判例】值得进 E 节判例表的「局面 / 选项 / 选择 / 条款」四列行；\n"
        "3. 每条都必须附**语料证据**：引用原文片段（≤40 字）+ 语料条目号；无证据的猜测"
        "一律不写；\n"
        "4. 与 v1 冲突时**以语料为准但标出分歧**（写明「与 P-xx 张力」），不要静默改写"
        "用户已确认的条款；\n"
        "5. 结尾列「未采纳的候选与理由」（≤5 条），让用户看见你丢掉了什么。\n\n"
        "## 输出要求（硬格式）\n\n"
        f"- 交付走 `<result>` 块，正文即草案 markdown；正文**首行必须是**这一行：\n"
        f"  `{DRAFT_MARK}`\n"
        "- 然后依次是 `## 新增条款` / `## 修订条款` / `## 判例增补` / `## 未采纳候选`；\n"
        "- 你没有写权限，**不要尝试写文件**；草案文本就是你唯一产出；\n"
        "- 总长 ≤ 6000 字，宁可少而硬，不要多而虚。"
    )


def header(meta: dict) -> str:
    """草案抬头（谁派的、读了什么、生效与否）——正文由 twin 交付回收。"""
    return (
        f"{DRAFT_MARK}\n\n"
        "# PREFERENCE.md v2 增量草案（diff 形态，非生效文件）\n\n"
        f"> **状态：待用户确认。** 生效面仍是 `orchestration/pods/twin/{PREFERENCE_NAME}`"
        f"（v1，用户 2026-10-12 审阅同意），本文件不被任何裁决引用。\n"
        f"> **来源**：语料 `{meta['corpus_file']}` 最新 {meta['n']} 条"
        f"（正序 {meta['first_rank']}–{meta['last_rank']}，倒序还原）+ v1 基线；"
        f"蒸馏跳= twin 冷启动单跳（sid `{meta['sid'][:8]}`，档位 `{meta['tier']}`，"
        f"{meta['turns']} 轮，eval_hop 不入生产台账）。\n"
        f"> **生成**：`orchestration/bench/distill_preference_v2.py`"
        f"（批准单 20261012-164601-leader）。\n\n---\n\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="twin 宪章 v2 蒸馏跳（E6b，只落草案）")
    ap.add_argument(
        "--corpus-dir", default=str(CORPUS_DIR), help="语料目录（默认主树）"
    )
    ap.add_argument(
        "--fileno", type=int, default=1, help="时段号=文件名前导数字（默认 1 近期）"
    )
    ap.add_argument("--newest", type=int, default=0, help="取最新 N 条；0=该文件前半")
    ap.add_argument("--member", default="twin", help="蒸馏成员（默认 twin）")
    ap.add_argument(
        "--model-tier",
        type=normalize_role,
        choices=["main", "pro", "minimum"],
        default=DEFAULT_TIER,
        help=f"蒸馏跳档位（旧名 flash→main、max→pro）；默认 {DEFAULT_TIER}=批准单点名的 flash",
    )
    ap.add_argument("--out", default=str(DEFAULT_DRAFT), help="草案落点")
    ap.add_argument("--dry", action="store_true", help="只打印注入内容与哈希，不派发")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.name == PREFERENCE_NAME or out.resolve() == EFFECTIVE.resolve():
        print(f"[!] 拒写生效面：--out 不能是 {PREFERENCE_NAME}（本单只产草案）")
        return 2

    reg = Registry.load()
    pref_v1 = preference_text(reg, args.member)
    if not pref_v1.strip():
        print(
            f"[!] v1 基线缺失：{args.member} pod 内无 {PREFERENCE_NAME}，无基线不蒸馏"
        )
        return 2

    puc = _load_parser()
    slice_msgs = select_slice(puc, Path(args.corpus_dir), args.fileno, args.newest)
    corpus_file = slice_msgs[0]["file"]
    prompt = build_prompt(pref_v1, slice_msgs, corpus_file=corpus_file)

    if args.dry:
        print(prompt)
        print(
            f"\n— 注入 {len(prompt)} B（v1 宪章 {len(pref_v1)} B + 语料 "
            f"{len(prompt) - len(pref_v1)} B / {len(slice_msgs)} 条），"
            f"taste_enabled={reg.taste_enabled}（蒸馏跳不受该开关拦，产出不生效）"
        )
        return 0

    tier = resolve_role(args.member, args.model_tier)
    print(
        f"→ 蒸馏跳：{args.member} 冷启动（v1 {len(pref_v1)} B + 语料 {len(slice_msgs)} 条，"
        f"档位 {tier}）"
    )
    outcome = do_dispatch(
        args.member,
        text=prompt,
        slug=f"distill-pref-v2-{corpus_file[:8]}",
        session="new",
        model_tier=args.model_tier,
        no_checks=True,
        attach_replies=False,
        eval_hop=True,
    )
    body = PROTOCOL_TAG_RE.sub("", (outcome.body or "").strip()).strip()
    if outcome.code != 0 or len(body) < 200:
        print(
            f"[✗] 蒸馏跳未产出（code={outcome.code} kind={outcome.kind} "
            f"正文 {len(body)} 字），草案未落盘"
        )
        print(f"    交付正文前 400 字：\n{body[:400]}")
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        header(
            {
                "corpus_file": corpus_file,
                "n": len(slice_msgs),
                "first_rank": slice_msgs[0]["rank"] + 1,
                "last_rank": slice_msgs[-1]["rank"] + 1,
                "sid": outcome.sid,
                "tier": tier,
                "turns": outcome.envelope.num_turns if outcome.envelope else 0,
            }
        )
        + body
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[√] 草案落 {out}（{len(body)} B，会话 {outcome.sid[:8]}）"
        f"\n[NEXT] 生效面未动；用户确认后再把 v2 增量并进 {PREFERENCE_NAME}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
