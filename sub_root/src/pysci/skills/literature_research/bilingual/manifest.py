"""work.md / manifest / bilingual.md 三契约 + 防覆写闸 + check-work 结构核验。

契约平移自旧 bilingual.py（语义不变）：

- **work.md 行契约**：每行 ``ID<TAB>英文原文``（译者只换 TAB 后部分）；
- **manifest 切片契约**：单元 + 透传块按 ``order`` 还原文档序（:mod:`.units` 生产、
  本模块序列化）；
- **bilingual.md 渲染契约**：**英文侧从 manifest 程序化重嵌入**（逐字一致，不取
  work.md——误改英文行不伤交付物），句对 = 中文行 + 斜体英文行、段界 ``---``、
  图片原位图注紧随、透传块原样、摘要/正文中文独行小标题（2026-11-03 用户裁决：
  管线不得发明源文没有的英文标注）。

护栏 ADR（2026-11-17）：旧 pod-guard.mjs（DSH PreToolUse 钩子）依赖旧编排体系，不平移；
等价语义由 :func:`check_work_structure` 承载（work.md 行首 ID 序列 × manifest 机械核验），
submit 前**强制执行**，显式 ``--skip-check-work`` 才可 override（见 :mod:`.cli`）。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci import paths

from .units import ParsedDoc

#: 翻译规则（work.md 头部注释行与翻译指引共用一份文本）。
RULES_TEXT = (
    "把每行 TAB 后的英文整行替换为中文翻译；保留行首 ID；不增删行；"
    "行内 $公式$、引文标注照抄；术语先读源文档统一；完成后跑 `bilingual submit`"
)

#: 行首 ID 前缀（对账/核验共用）：类型字母 + 4 位序号 + TAB/行尾。
ID_PREFIX_RE = re.compile(r"^([A-Z]\d{4})(?:\t|$)")

#: 用户侧副本文件名最大长度（超长截断；扩展名另加不受影响）。
_NAME_MAX = 60


def workspace_root() -> Path:
    """工作区根 ``data/skills/literature_research/bilingual/``。"""
    return paths.LITERATURE_ROOT / "bilingual"


def inbox_root() -> Path:
    """用户 inbox 副本根 ``用户的inbox/bilingual/``（submit 末步只增不删）。

    该目录属用户全权管理区（AGENTS.md「内容目录与用户 inbox 政策」）：仅写
    最终交付物、严禁删改既有内容、自动化管线不得依赖其内容。
    """
    return paths.INBOX_ROOT / "bilingual"


def workdir_for(stem: str) -> Path:
    """某篇的工作区目录（写入前过 assert_within_data 防 stray 护栏）。"""
    return paths.assert_within_data(workspace_root() / stem, what="bilingual 工作区")


def render_work_md(doc: ParsedDoc, translations: dict[str, str] | None = None) -> str:
    """生成 work.md 内容：头部 ``#`` 注释规则 + 每单元 ``ID<TAB>英文原文``。

    段落之间以空行分隔（仅便读；对账按 ID 不按行号）。摘要段（role=abstract）前后
    以 ``#`` 注释行标注（对账/核验都跳过注释行）；``translations`` 提供时行尾用译文
    （TM 回填通道）。
    """
    lines = [
        f"# 翻译规则：{RULES_TEXT}",
        "# 行首 ID 与行数不许改动（submit 前 check-work 硬校验）。",
        "",
    ]
    prev_para: int | None = None
    abs_seen = False
    for u in sorted(doc.units, key=lambda x: x.order):
        if prev_para is not None and u.para != prev_para:
            lines.append("")
        if u.role == "abstract" and not abs_seen:
            lines.append(
                "# ↓摘要段（role=abstract；识别有误改 manifest.json 的 role 字段后重新 submit）"
            )
            abs_seen = True
        if u.role != "abstract" and abs_seen:
            lines.append("# ↑摘要段结束（以下为正文）")
            abs_seen = False
        tail = (translations or {}).get(u.id, u.en)
        lines.append(f"{u.id}\t{tail}")
        prev_para = u.para
    return "\n".join(lines) + "\n"


def build_manifest(doc: ParsedDoc, source_md: Path, no_images: bool) -> dict[str, Any]:
    """构造 manifest 字典：单元切片 + 透传块 + 文档元信息。

    Args:
        doc: 解析结果。
        source_md: 源 markdown 路径（元信息记录）。
        no_images: 本次是否跳过图片回补（submit 据此拒绝出稿）。

    Returns:
        可直接 ``json.dumps`` 的 manifest 字典。
    """
    return {
        "version": 1,
        "source_md": str(source_md),
        "no_images": bool(no_images),
        "images": list(doc.images),
        "units": [asdict(u) for u in sorted(doc.units, key=lambda x: x.order)],
        "blocks": [asdict(b) for b in sorted(doc.blocks, key=lambda x: x.order)],
    }


def read_work_translations(work_md: Path) -> dict[str, str]:
    """读 work.md 为 ``{ID: 译文}``；``#`` 注释行与空行忽略，行首 ID 为键。

    Args:
        work_md: work.md 路径。

    Returns:
        ID → 该行 TAB 后的全文（译文或未改的英文原文）。
    """
    out: dict[str, str] = {}
    for line in work_md.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ID_PREFIX_RE.match(line)
        if not m:
            continue
        _, _, tail = line.partition("\t")
        out[m.group(1)] = tail
    return out


def load_workspace(stem: str) -> tuple[Path, dict[str, Any]]:
    """载入工作区的 manifest，返回 ``(workdir, manifest)``。

    Raises:
        FileNotFoundError: 工作区或 manifest 不存在（未跑 prepare）。
    """
    workdir = workdir_for(stem)
    manifest_path = workdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"工作区未就绪（先跑 bilingual prepare）：{manifest_path}"
        )
    return workdir, json.loads(manifest_path.read_text(encoding="utf-8"))


def check_work_structure(workdir: Path, manifest: dict[str, Any]) -> list[str]:
    """check-work 护栏：work.md 行结构 × manifest 机械核验（pod-guard 等价物）。

    核验项：work.md 存在；非注释非空行必须带行首 ID；ID 序列与 manifest 单元序列
    **逐位一致**（不多、不少、不重排）。返回错误清单（空 = 通过）——submit 前强制
    执行（:mod:`.cli`），``--skip-check-work`` 显式 override。

    Args:
        workdir: 工作区目录。
        manifest: :func:`load_workspace` 载入的 manifest。

    Returns:
        人类可读错误行清单；通过时为空表。
    """
    work_md = workdir / "work.md"
    if not work_md.is_file():
        return [f"work.md 不存在（先跑 bilingual prepare）：{work_md}"]
    errors: list[str] = []
    ids: list[str] = []
    for lineno, line in enumerate(
        work_md.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ID_PREFIX_RE.match(line)
        if m:
            ids.append(m.group(1))
        else:
            errors.append(f"第 {lineno} 行缺行首 ID（或格式错误）：{line[:40]!r}")
    expected = [str(u["id"]) for u in manifest.get("units", [])]
    if len(ids) != len(expected):
        errors.append(
            f"行数不等：work.md ID 行 {len(ids)} ≠ manifest 单元 {len(expected)}"
        )
    if ids != expected:
        errors.append("ID 前缀序列与 manifest 不一致（不增删行、不改行首 ID、不重排）")
    return errors


def _translated_rows(work_md: Path) -> list[str]:
    """work.md 中已含译文的行 ID（行尾含 CJK 即视为译文——防覆写闸的判据）。"""
    from .reconcile import has_cjk

    return [
        k for k, tail in read_work_translations(work_md).items() if has_cjk(tail)
    ]


def guard_translated_workspace(workdir: Path, *, force: bool) -> tuple[int, str]:
    """防覆写闸（backlog 20261023-163700-devops）：含译文工作区再 prepare 的保护。

    工作区即生产资料——旧 work.md 的译文被静默覆盖＝数据丢失口（2026-10-23 实测事故）。
    默认**拒绝**（rc=2）；``--force`` 放行并**自动备份**旧 work.md/manifest 为
    ``*.bak-<ts>``。返回 ``(rc, 输出行)``：rc=0 继续、rc=2 调用方直接返回。
    """
    old_work = workdir / "work.md"
    if not old_work.is_file():
        return 0, ""
    translated = _translated_rows(old_work)
    if not translated:
        return 0, ""  # 无译文：幂等重跑不设门
    if not force:
        return 2, (
            f"[bilingual] 拒绝覆写：工作区已有译文 {len(translated)} 行"
            f"（如 {translated[0]}）——防译文丢失默认拒绝。确认覆写加 --force"
            "（自动备份旧 work.md/manifest）；要保留译文改用 --reuse TM 回填"
        )
    ts = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    saved: list[str] = []
    for name in ("work.md", "manifest.json"):
        src = workdir / name
        if src.is_file():
            shutil.copy2(src, workdir / f"{name}.bak-{ts}")
            saved.append(name)
    return 0, f"[bilingual] --force：旧 {'/'.join(saved)} 已备份为 *.bak-{ts}"


def render_bilingual_md(manifest: dict[str, Any], translations: dict[str, str]) -> str:
    """生成 bilingual.md：英文侧从 manifest 重嵌入（逐字一致），中文侧取 work.md。

    输出格式契约（:mod:`renderer` 的输入契约，勿手改形态）：标题 ``# 中文``（保留源
    ``#`` 层级）紧跟一行 ``*English heading*``；句对为中文行 + 斜体英文行，句对之间
    空一行；段落边界 ``---`` 单独成行；图片原位后紧跟图注句对（同 para，不隔 ``---``）；
    公式/表格/参考文献原样原位。摘要段（role=abstract）前插 ``## 摘要``、其后插
    ``---`` + ``## 正文``（中文独行小标题、**无英文行**——2026-11-03 用户裁决）。
    """
    items: list[dict[str, Any]] = sorted(
        [dict(x, _kind="unit") for x in manifest["units"]]
        + [dict(x, _kind="block") for x in manifest["blocks"]],
        key=lambda x: x["order"],
    )
    lines: list[str] = []
    prev_para: int | None = None
    abs_on = False  # 已进入摘要段
    body_marked = False
    for idx, it in enumerate(items):
        nxt = items[idx + 1] if idx + 1 < len(items) else None
        if prev_para is not None and it["para"] != prev_para:
            lines.append("---")
        is_abs = it["_kind"] == "unit" and it.get("role") == "abstract"
        if is_abs and not abs_on:
            lines.append("## 摘要")
            abs_on = True
        if not is_abs and abs_on and not body_marked:
            lines.append("---")
            lines.append("## 正文")
            body_marked = True
        if it["_kind"] == "block":
            lines.extend(str(it["content"]).split("\n"))
        else:
            zh = translations[it["id"]].strip()
            if it["type"] == "H":
                level = int(it.get("level") or 2)
                lines.append(f"{'#' * level} {zh}")
            else:
                lines.append(zh)
            lines.append(f"*{it['en']}*")
            # 句对之间空一行（同 para 的后继句对才隔空行；图片后紧跟图注不隔）。
            if nxt is not None and nxt["para"] == it["para"] and nxt["_kind"] == "unit":
                lines.append("")
        prev_para = it["para"]
    return "\n".join(lines).rstrip() + "\n"


def output_basename(manifest: dict[str, Any], translations: dict[str, str]) -> str:
    """用户侧副本文件名（不含扩展名）= 文档中文标题（2026-10-23 用户裁决：语义命名）。

    取首个 ``H`` 单元的中文译文；无 H 单元回退首个可译单元中文译文（都缺则英文原文）；
    剔除非法文件名字符、压缩空白、截断至 :data:`_NAME_MAX`；空结果回退 ``bilingual``。
    """
    units = sorted(manifest["units"], key=lambda x: x["order"])
    title = ""
    for u in units:
        if u["type"] == "H":
            title = (translations.get(u["id"]) or "").strip() or u["en"].strip()
            break
    if not title:
        for u in units:
            title = (translations.get(u["id"]) or "").strip()
            if title:
                break
    base = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", title)
    base = re.sub(r"\s+", " ", base).strip().strip(".")
    return base[:_NAME_MAX].strip() or "bilingual"
