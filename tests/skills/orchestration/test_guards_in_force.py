"""guards-in-force 元哨兵（backlog 20261010-193601-devops）。

**为什么需要元哨兵**：本会话把 delivery-gate 与 pod-guard 的 fail-open 加了 `[<name>
fail-open]` stderr 留痕；写了 test_delivery_gate_budget.py 29 例、test_pod_guard_blocks.py
8 例；guard_probe.py 也接进 doctor。但**这些约束只对已有的两个 guard 生效**——未来
新增第三个 guard（例如某类 Bash 前置检查、tool-input schema 校验器），若没有同规格的
阳性/阴性 pytest + fail-open 留痕，它一出生就是一个静默空壳，观测面看不到，与首版
delivery-gate 的 TDZ 事故同构。

本文件是**结构性哨兵**：扫描 `orchestration/guards/*.mjs`，对每个 guard 断言三件事——

1. **有活体测试**：`tests/skills/orchestration/` 下存在某份 pytest 文件直接以
   `node <该 guard>` 的姿势调用（判据：文件正文含 `"guards"` + `"<name>.mjs"` 关键
   字面量，或直接引用 `GUARD = ... <name>.mjs` 常量）；
2. **有阳性用例**：该测试文件至少一个函数以 `rc == 2` / `exit 2` 断言违规被拦；
3. **guard 有 fail-open 留痕**：guard 源文件里出现 `[<name> fail-open]` 字面量。

任一不满足即红，红名直接指出「缺什么、加哪儿」，无需人脑记规矩。**新增 guard 时**，
写 `orchestration/guards/<new>.mjs` 会立刻撞本哨兵——把「约定」变成「机制」。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT

GUARDS_DIR = ORCHESTRATION_ROOT / "guards"
TESTS_DIR = Path(__file__).resolve().parent  # tests/skills/orchestration/


def _enumerate_guard_sources() -> list[Path]:
    if not GUARDS_DIR.exists():
        return []
    return sorted(p for p in GUARDS_DIR.glob("*.mjs") if p.is_file())


GUARD_NAMES = [p.stem for p in _enumerate_guard_sources()]


def _test_files_that_invoke(name: str) -> list[Path]:
    """返回**真跑该 guard** 的测试文件：正文同时出现 `<name>.mjs` 与 `guards` 目录引用。

    约定：guard 源在 `orchestration/guards/<name>.mjs`；测试文件通过
    `from pysci.paths import ORCHESTRATION_ROOT` + `ORCHESTRATION_ROOT / "guards" / "<name>.mjs"`
    或直接 `"guards" / "<name>.mjs"` 引用；两者共现视为「真跑」。
    """
    hits: list[Path] = []
    needle_file = f"{name}.mjs"
    for f in TESTS_DIR.glob("test_*.py"):
        text = f.read_text(encoding="utf-8", errors="replace")
        if needle_file in text and re.search(r'["\'/]guards["\']', text):
            hits.append(f)
    return hits


def _file_asserts_block_verdict(text: str) -> bool:
    """文本里出现「exit 2 / rc==2 / returncode==2」任一断言形态即视为阳性用例存在。"""
    patterns = (
        r"assert\s+rc\s*==\s*2\b",
        r"assert\s+\w+_rc\s*==\s*2\b",
        r"assert\s+proc\.returncode\s*==\s*2\b",
        r"==\s*\(2\b",  # (rc, err) == (2, ...) 元组断言
        r"\[\s*2\s*,",  # 断言 rc 落在 [2, ...] 集合
    )
    return any(re.search(p, text) for p in patterns)


@pytest.mark.parametrize("name", GUARD_NAMES, ids=GUARD_NAMES)
def test_guard_has_fail_open_trace(name: str):
    """guard 源必须含 `[<name> fail-open]` stderr 留痕（backlog 20261010-193601-devops）。"""
    src = GUARDS_DIR / f"{name}.mjs"
    text = src.read_text(encoding="utf-8")
    needle = f"[{name} fail-open]"
    assert needle in text, (
        f"{src.name} 缺 fail-open stderr 留痕：应在吞异常处向 stderr 打 {needle!r}。"
        "静默 catch 会让 guard 有 bug 时伪装成一切正常（首版 TDZ 事故），"
        "guard_probe 与 doctor 观测面均无法识别。"
    )


@pytest.mark.parametrize("name", GUARD_NAMES, ids=GUARD_NAMES)
def test_guard_has_in_force_tests(name: str):
    """至少一份 test_*.py 真跑该 guard，且至少含一个 exit 2 阳性断言。"""
    hits = _test_files_that_invoke(name)
    assert hits, (
        f"没有 pytest 直跑 {name}.mjs 的用例文件（约定：tests/skills/orchestration/"
        "test_*.py 用 subprocess `node <GUARD>` + stdin JSON 断言违规 exit 2）。"
        "「手动测一下 exit code」不足以对抗 fail-open 静默退化。"
    )
    text_blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in hits)
    assert _file_asserts_block_verdict(text_blob), (
        f"{name} 的测试文件里没有断言违规 exit 2 的用例：只测「正常时放行」不能证明"
        "guard 在执法（fail-open 语义下，rc=0 有两种原因）。加一条 `assert rc == 2` "
        "阳性用例，喂一个明确违规的 fixture。"
    )


def test_enumerate_guards_is_not_empty():
    """若扫描返回空目录（比如 pysci.paths 漂了）——直接让本文件红，而不是静默 passing。"""
    assert GUARD_NAMES, (
        f"orchestration/guards/*.mjs 为空——元哨兵失去意义。实际扫描根：{GUARDS_DIR}"
    )


def test_enumerated_guards_cover_known_pair():
    """至少锁死 delivery-gate + pod-guard 两个现存 guard 被枚举到——防路径漂移。"""
    assert {"delivery-gate", "pod-guard"}.issubset(set(GUARD_NAMES))
