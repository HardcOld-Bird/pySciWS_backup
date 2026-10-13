"""REFACTOR 收官并入 README 的快照断言（backlog 20261013-072000-leader，v1.0 §12 收尾）。

三件事锁死：
① 退役事实——orchestration/ 下的原设计文档（REFACTOR 文档，下称 REF）已删除，代码/文档
   面零 REF 字面量悬空引用（口径=**旧路径** orchestration/REFACTOR.md；根目录另有 2026-10-14
   的在役新蓝图 REFACTOR.md，引用它正当，2026-10-14 m1 收窄口径；state 台账与用户语料属留档面，不计）；
② 融合映射——README 各节含收官并入的新面（头部状态行、§1 三角色、§3.1 无记忆、
   §4.1 改进机构面、§4.5 硬警示、§5.3–5.5、§6 五要素、§9 三闸、§12 收官段、§13 三行）；
③ 指针清零——README 内不得再有 `REFACTOR §N` 式节号指针（历史名提及不算指针）。
"""

from __future__ import annotations

import re
from pathlib import Path

from pysci.paths import PROJECT_ROOT

#: 判据「代码/文档」面：这些根下任何文件都不得再出现 REF 字面量。
#: 本测试文件自身豁免（判据字面量所在处），字面量经拼接避免自我命中。
#: **口径收窄（2026-10-14，REFACTOR m1）**：只禁**被删除的旧文档路径** `orchestration/REFACTOR.md`
#: ——v1.0 收官把它并入了 README，那条路径必须彻底清零。根目录另有**新的在役蓝图** `REFACTOR.md`
#: （2026-10-14 用户批准，M 系列施工依据），引用它是正当的，不该被这条旧快照判据误伤。
REF = "orchestration/REFACTOR" + ".md"
CODE_DOC_ROOTS = (
    "src",
    "tests",
    "orchestration/skills",
    "orchestration/guards",
    "orchestration/bench",
    "orchestration/pods",
)
CODE_DOC_FILES = (
    "orchestration/README.md",
    "pyproject.toml",
    ".gitignore",
)
#: 留档面（判据豁免）：状态台账/收件箱/用户原始语料——记录事实本身，不是引用指针。
SKIP_PARTS = ("state/", "inbox/", "outbox/", "__pycache__", ".venv", ".ruff_cache")


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def test_refactor_md_is_deleted():
    """判据：原设计文档已删除（git log 可查；工作树不再存在）。"""
    assert not (PROJECT_ROOT / "orchestration" / REF).exists()


def test_no_dangling_refactor_md_reference_in_code_or_docs():
    """判据：代码/文档面 REF 字面量清零（本单留档与 state 台账除外）。"""
    offenders: list[str] = []
    here = Path(__file__).resolve()
    for root in CODE_DOC_ROOTS:
        base = PROJECT_ROOT / root
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.resolve() == here:
                continue
            if p.suffix not in {".py", ".md", ".mjs", ".toml", ".yaml", ".json"}:
                continue
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            if any(s in rel for s in SKIP_PARTS):
                continue
            if REF in p.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(rel)
    for rel in CODE_DOC_FILES:
        p = PROJECT_ROOT / rel
        if p.exists() and REF in p.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert offenders == [], f"REF 字面量悬空引用残留：{offenders}"


def test_readme_has_no_section_number_pointers_to_refactor():
    """指针清零：README 内 `REFACTOR §N` 式指针为零（历史名提及、无节号者合法）。"""

    readme = _read("orchestration/README.md")
    assert not re.search(r"REFACTOR\s*(v1\.0\s*)?§", readme), (
        "README 仍指向已删除文档的节号"
    )


def test_header_declares_closeout():
    """映射 1：头部状态行 = 收官 + 机制已并入本文。"""
    readme = _read("orchestration/README.md")
    head = readme[: readme.index("## 1. 角色与进程模型")]
    assert "已收官" in head and "机制已并入本文" in head, head[:600]


def test_role_table_has_three_closed_loop_members():
    """映射 2：§1 角色表含 inspector / bencher / twin 三行摘要。"""
    readme = _read("orchestration/README.md")
    table = readme[readme.index("## 1. 角色与进程模型") : readme.index("### 1.1")]
    assert "| **inspector** |" in table
    assert "| **bencher** |" in table
    assert "| **twin** |" in table
    assert "无记忆" in table and "不给平局" in table, "twin 行须含冷启动/终裁要点"


def test_memory_section_has_twin_discipline_and_preference_pointer():
    """映射 3：§3.1 twin 无记忆纪律 + PREFERENCE.md（pod 内只读）指针。"""
    sec = _read("orchestration/README.md")
    assert "无记忆纪律" in sec and "pods/twin/" in sec and "PREFERENCE.md" in sec


def test_channel_section_is_provider_preset_mode():
    """映射 4：§3.5 = 提供商预设模式（真源 toml、三预设、回退粘滞、回退告警、minimum 独立）。"""
    readme = _read("orchestration/README.md")
    sec = readme[readme.index("### 3.5") : readme.index("## 4. orch CLI")]
    for marker in (
        "data/pysci_settings.toml",
        "qwen",
        "mimo",
        "deepseek",
        "降级粘滞窗",
        "回退必告警",
        "独立、不随预设走",
    ):
        assert marker in sec, f"§3.5 缺预设模式要素：{marker}"


def test_command_surface_lists_improvement_institutions():
    """映射 5：§4.1 改进机构面含 bench/jury/taste/eval/inspector 子命令 + mirror 摘要。"""
    readme = _read("orchestration/README.md")
    sec = readme[readme.index("### 4.1") : readme.index("### 4.2")]
    for marker in (
        "orch jury",
        "orch taste",
        "orch eval",
        "orch bench-add",
        "orch inspector-up",
        "pysci-mirror",
    ):
        assert marker in sec, f"§4.1 缺命令摘要：{marker}"


def test_wait_section_carries_mount_lines_warning():
    """映射 5b：§4.5 含 mount_lines 硬警示（watch 须唯一顶层命令）。"""
    readme = _read("orchestration/README.md")
    sec = readme[readme.index("### 4.5") : readme.index("## 5. 质量保证")]
    assert "唯一顶层命令" in sec and "mount_lines" in sec


def test_quality_section_has_verdict_tier_a_and_harness():
    """映射 6：§5 含判决包小节（四态/例外/修复形状分流）+ Tier A + harness 套件三面。"""
    readme = _read("orchestration/README.md")
    sec = readme[
        readme.index("## 5. 质量保证") : readme.index("## 6. 基础设施改进循环")
    ]
    assert "### 5.3 判决包" in sec
    for marker in (
        "**J**",
        "**M**",
        "**J+M**",
        "override",
        "例外通道",
        "修复形状分流",
        "restore",
    ):
        assert marker in sec, f"§5.3 缺要素：{marker}"
    assert "### 5.4 Tier A" in sec and "阈值自动立案" in sec
    assert "### 5.5 Tier B" in sec and "bencher 所有" in sec and "副本 A/B" in sec


def test_improvement_loop_has_five_closed_loop_faces():
    """映射 7：§6 含 inspector 晋升/绿色通道/override 率/双队列迟滞/decisions 审计面。"""
    readme = _read("orchestration/README.md")
    sec = readme[
        readme.index("## 6. 基础设施改进循环") : readme.index("## 7. devops 作业模式")
    ]
    for marker in (
        "自动晋升",
        "绿色通道",
        "override 率",
        "双队列迟滞",
        "pysci-mirror decisions",
        "免判决+change 置顶",
    ):
        assert marker in sec, f"§6 缺要素：{marker}"


def test_enforcement_layer_has_three_recovery_gates():
    """映射 8：§9 强制层含 WIP-salvage / sid 在途闸 / CHANNEL-STOP 各一行。"""
    readme = _read("orchestration/README.md")
    sec = readme[readme.index("## 9. 强制层") : readme.index("## 10. 端到端工作流")]
    for marker in ("WIP-salvage", "sid 在途闸", "CHANNEL-STOP"):
        assert marker in sec, f"§9 缺恢复面闸：{marker}"


def test_phase_section_has_closeout_and_night_log():
    """映射 10：§12 收官段在册（E0–E7 状态 + 夜航要点），删除前已并入。"""
    readme = _read("orchestration/README.md")
    sec = readme[readme.index("### 自改进体系 E0–E7") : readme.index("## 13. 裁决记录")]
    for marker in (
        "**E0 纪律与原料层**",
        "**E7 twin 上线**",
        "融合与退役",
        "夜航",
        "WIP-salvage 先例",
    ):
        assert marker in sec, f"§12 收官段缺要素：{marker}"


def test_verdict_register_has_closeout_rows():
    """映射 9：§13 收官三行登记——收官 / PREFERENCE v2+twin 全量 / 修复形状分流+审计面。

    行数**不锁死**：§13 是追加式裁决台账，同日后到的裁决照常追加（路由 v3 那行先于本单
    合并进 main，就是反例——按「恰有 3 行」钉死会红一次合法登记）。判据是三行收官登记逐条在场。
    """
    readme = _read("orchestration/README.md")
    sec = readme[readme.index("## 13. 裁决记录") :]
    sec = sec[: sec.index("## 14.")]
    rows = [ln for ln in sec.splitlines() if ln.startswith("| 2026-10-13 |")]
    assert rows, "§13 至少要有 2026-10-13 的收官登记"
    joined = "\n".join(rows)
    assert (
        "实施收官" in joined and "PREFERENCE v2" in joined and "修复形状分流" in joined
    )
