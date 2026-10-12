"""CLI 调用范式的快照断言（backlog 20261012-125455-devops）。

**为什么要机械执法**：`uv run <目标>` 不带 `--no-sync` 会**先隐式 sync 项目环境**再执行。
Windows 上这次 sync 撞运行中 exe 的锁（典型=组长的 ``orch watch``）时，uv 的行为是
**先删光其余入口**再报错——2026-10-12 main 实测：rc=2、`.venv/Scripts` 少 6 个入口、
裸命令集体失联，工具本体还没跑起来。公约数 §3 已把范式改成 `uv run --no-sync`，但
「靠文档提醒」的失效方式是静默的：某个提示语、某条案卷示例、某个 check 命令漏改，
就重新埋下同根火柴，且没人知道何时被点着。本测试把「哪些写法会点火」变成机械事实。

口径（写成判据而不是感觉）：**凡是会被照抄执行的调用写法**——代码里的提示串与
docstring 示例、技能真本、根 rules、编排 README、注册表的 check 命令——`uv run` 后紧跟
可执行目标（``pysci-*`` / ``python`` / ``pytest``）时必须带 `--no-sync`。纯散文（「以
`uv run` 为准」「退化成 `uv run` 就地新建空 venv」）后面没有目标词，不在判据内。

只读层（各 pod 的 charter）组员写不了——pod-guard 拦（本会话 ``PYSCI_READONLY`` 首项
就是 ``.qoder/rules/charter.md``），须组长转呈用户授权后代落。这些文件登记在
``READONLY_PENDING``，且**豁免会过期**：文件一旦合规，哨兵反而变红，提醒删掉那一行。
否则豁免清单会悄悄长成永久的假绿——那正是它要防的东西。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pysci.paths import PROJECT_ROOT

#: `uv run` 后面紧跟的可执行目标——出现即视为「一条会被照抄的命令」。
_TARGET = r"(?:pysci-[\w.-]+|python\b|pytest\b)"
#: 违规写法：uv run → 目标词，中间没有 --no-sync。
_BAD = re.compile(rf"uv run\s+(?!--no-sync\s*){_TARGET}")
#: 折行的命令：本行只到 `uv run` 就换行，目标词落在下一行（Markdown 里很常见）。
_HANGING = re.compile(r"uv run\s*$")

#: 组员无写权限、须组长代落的只读面（pod id）。落定后从这里删掉。
READONLY_PENDING = {
    "devops",
    "drawing",
    "figure",
    "lit",
    "model3d",
    "sim",
    "theory",
    "writing",
}


def logical_lines(text: str) -> list[tuple[int, str]]:
    """把物理行拼成「逻辑行」，返回 ``(起始行号, 文本)``。

    折行的命令必须当成一行看——``uv run\\n  --no-sync pytest`` 是一条**合规**命令，
    按物理行扫会把反斜杠-free 的排版当成漏改，测试就会因为格式而红。
    """
    out: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start = i + 1
        buf = lines[i]
        while _HANGING.search(buf.replace("`", " ")) and i + 1 < len(lines):
            i += 1
            buf = f"{buf} {lines[i].strip()}"
        out.append((start, buf))
        i += 1
    return out


def bad_calls(text: str) -> list[tuple[int, str]]:
    """返回文本里所有「会点火的 uv run 写法」：``(行号, 该段上下文)``。"""
    hits: list[tuple[int, str]] = []
    for lineno, line in logical_lines(text):
        flat = line.replace("`", " ")
        for m in _BAD.finditer(flat):
            hits.append((lineno, flat[m.start() : m.start() + 60].strip()))
    return hits


def scan(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for p in paths:
        for lineno, snippet in bad_calls(p.read_text(encoding="utf-8")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            problems.append(f"{rel}:{lineno}  {snippet}")
    return problems


def corpus() -> list[Path]:
    """会被照抄的调用面。

    边界（写清，免得日后被当成「全仓都管」）：代码与提示串、技能真本、根 rules、
    组员 AGENTS.md、项目/编排 README、pyproject 注释、registry 的 check 命令。
    **不含** ``scripts/`` 与 ``data/`` 下的研究资产脚本（另属组员口径，见交付观察项）、
    不含 ``tests/``（本文件的反例素材本来就写着违规形）。
    """
    files: list[Path] = []
    files += sorted((PROJECT_ROOT / "src").rglob("*.py"))
    files += sorted((PROJECT_ROOT / "orchestration" / "skills").rglob("*.md"))
    files += sorted((PROJECT_ROOT / ".qoder" / "rules").glob("*.md"))
    files += sorted((PROJECT_ROOT / "orchestration" / "pods").glob("*/AGENTS.md"))
    files.append(PROJECT_ROOT / "orchestration" / "README.md")
    files.append(PROJECT_ROOT / "README.md")
    files.append(PROJECT_ROOT / "pyproject.toml")
    return [f for f in files if f.is_file()]


# ---------------------------------------------------------------------------
# ① 判据本身：扫描器不能是「永远绿」的摆设
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expect",
    [
        ("命令不存在时 fallback `uv run pysci-X`", 1),
        ("运行统一用 `uv run python ...` / `uv run pytest`", 2),
        ('print("  再试：uv run pysci-orch status")', 1),
        ("`PYTHONPATH=<wt>/src uv run pysci-dev doctor`", 1),
        # 合规写法（含折行）与纯散文：都不该报。
        ("fallback `uv run --no-sync pysci-X`", 0),
        ("`PYTHONPATH=<wt>/src uv run\n      --no-sync pytest <wt>/tests`", 0),
        ("迁移期以 `uv run` 为准", 0),
        ("退化成 `uv run` 就地新建空 venv（已知坑）", 0),
        ("裸 `uv sync`/`uv run` 遇运行中 exe 会先删光入口", 0),
        ("裸命令首选/uv run fallback/禁重试循环", 0),
    ],
)
def test_scanner_judges_by_shape_not_by_hope(text: str, expect: int):
    """阳性/阴性各钉一组：扫描器失灵=整个防漂移面失灵，比漏改一处更贵。"""
    assert len(bad_calls(text)) == expect, (text, bad_calls(text))


# ---------------------------------------------------------------------------
# ② 调用面全清：一处漏改即失败，报告带 文件:行号 与片段
# ---------------------------------------------------------------------------
def test_copyable_command_faces_all_carry_no_sync():
    problems = scan(corpus())
    assert not problems, "以下会被照抄的 `uv run` 写法缺 --no-sync：\n" + "\n".join(
        problems
    )


def test_registered_checks_do_not_sync_on_every_hop():
    """registry 的 check 命令由派发进程逐跳执行——隐式 sync 在这里是**每跳**的风险。"""
    data = json.loads(
        (PROJECT_ROOT / "orchestration" / "state" / "registry.json").read_text(
            encoding="utf-8"
        )
    )
    for name, spec in (data.get("checks") or {}).items():
        argv = spec["cmd"]
        for i, tok in enumerate(argv[:-2]):
            if tok == "uv" and argv[i + 1] == "run":
                assert argv[i + 2] == "--no-sync", f"{name}: {argv}"


def test_basic_md_states_the_paradigm_in_one_voice():
    """§3 与 §5 是同一条纪律的两个说法：一处 --no-sync、一处裸 uv run = 口径分裂。"""
    text = (PROJECT_ROOT / ".qoder" / "rules" / "basic.md").read_text(encoding="utf-8")
    assert "uv run --no-sync" in text
    assert bad_calls(text) == []


# ---------------------------------------------------------------------------
# ③ 只读层豁免表：会过期的哨兵
# ---------------------------------------------------------------------------
def charter_file(pod: str) -> Path:
    return (
        PROJECT_ROOT
        / "orchestration"
        / "pods"
        / pod
        / ".qoder"
        / "rules"
        / "charter.md"
    )


def test_readonly_exemptions_are_still_needed():
    """豁免不许腐化成永久沉默：已合规的 charter 必须从表里删掉。"""
    still_stale = {
        pod
        for pod in READONLY_PENDING
        if bad_calls(charter_file(pod).read_text(encoding="utf-8"))
    }
    assert still_stale == READONLY_PENDING, (
        f"已合规、请从 READONLY_PENDING 删掉：{sorted(READONLY_PENDING - still_stale)}"
    )


def test_no_unregistered_charter_carries_the_stale_paradigm():
    """新 pod 若照旧写 `uv run pysci-X`，这里红——要么组长代落，要么登记进豁免表。"""
    unregistered: list[str] = []
    for f in sorted(
        (PROJECT_ROOT / "orchestration" / "pods").glob("*/.qoder/rules/charter.md")
    ):
        if (
            bad_calls(f.read_text(encoding="utf-8"))
            and f.parts[-4] not in READONLY_PENDING
        ):
            unregistered.append(f.relative_to(PROJECT_ROOT).as_posix())
    assert not unregistered, f"未登记的只读面仍会点火：{unregistered}"


def test_charter_fix_text_is_known_to_leader():
    """豁免不是甩手：每条待落改动都要有一行「改成什么」的可核对描述。

    组长代落 charter 须逐字判读，若清单里漏了某个 pod，本测试先红——防「表还在、
    内容描述已过期」这种第二层漂移。
    """
    for pod in sorted(READONLY_PENDING):
        text = charter_file(pod).read_text(encoding="utf-8")
        hits = bad_calls(text)
        assert hits, pod
        lineno, snippet = hits[0]
        assert "pysci-" in snippet or "python" in snippet, (pod, snippet)
