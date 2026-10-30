"""TUI 改版 s3（task tui-overhaul-20261016-s3-modules）：廉价指纹 + 提示清理 + 图标空格。

钉五件事：

1. **性能**：``_fingerprint_module(..., digest=…)`` 的 ``cursor()`` **不构建全详情**；
   ``_ttl_memo`` 2 秒内同参只算一次（越窗才重算）；plans/system 的真接线各走
   ``plans.plan_digest`` / ``_system_stable`` 指纹；
2. **禁语**：七模块 detail 输出不含用户点名的旧提示字样；
3. **图标空格**：tools 源码无「状态图标紧贴文字」（成对闭合符/字符串字面量末位豁免）；
4. **系统详情首行=倒计时**；``_COUNTDOWN_MARK`` 排除语义不变（stable 指纹不含倒计时行）；
5. plans 概览 mark 新取值（带空格）。
"""

from __future__ import annotations

import re
from pathlib import Path

from pysci.skills.orchestration.tools import plans
from pysci.skills.orchestration.tools.tui import modules as _mods
from pysci.skills.orchestration.tools.tui.modules import modules as all_modules
from pysci.skills.orchestration.tools.tui.watermark import fingerprint


# ---------------------------------------------------------------------------
# ① 廉价指纹 + TTL memo（性能钉）
# ---------------------------------------------------------------------------
def test_fingerprint_cursor_uses_digest_without_detail():
    calls = {"detail": 0}

    def detail() -> list[str]:
        calls["detail"] += 1
        return ["很贵的一行"]

    mod = _mods._fingerprint_module("x", "X", lambda: [], detail, digest=lambda: "D1")
    assert mod.cursor() == "D1"
    assert calls["detail"] == 0, "传 digest 的模块 cursor 不得构建全详情"
    # 指纹族语义（同原实现）：空水位=未看过→0；变了→1；一致→0
    assert mod.updates_since("") == 0
    assert mod.updates_since("旧指纹") == 1
    assert mod.updates_since("D1") == 0
    # 缺省（无 digest）仍回落 fingerprint(detail())
    mod2 = _mods._fingerprint_module("y", "Y", lambda: [], lambda: ["a"])
    assert mod2.cursor() == fingerprint(["a"])


def test_ttl_memo_reuses_within_ttl(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(_mods.time, "time", lambda: clock["t"])
    calls = {"n": 0}

    def fn() -> int:
        calls["n"] += 1
        return calls["n"]

    memo = _mods._ttl_memo(fn, ttl=2.0)
    assert memo() == 1 and memo() == 1
    assert calls["n"] == 1, "2 秒内同参只算一次"
    clock["t"] += 1.9
    assert memo() == 1 and calls["n"] == 1
    clock["t"] += 0.2  # 越窗 → 重算
    assert memo() == 2 and calls["n"] == 2


def test_plans_module_wired_to_plan_digest(monkeypatch):
    calls = {"detail": 0}

    def spy_detail() -> list[str]:
        calls["detail"] += 1
        return []

    monkeypatch.setattr(plans, "plan_digest", lambda: "DIGEST")
    monkeypatch.setattr(_mods, "_plans_detail", spy_detail)
    mod = next(m for m in all_modules() if m.key == "plans")
    assert mod.cursor() == "DIGEST"
    assert calls["detail"] == 0, "plans 徽标指纹不得构建计划详情"


def test_system_module_wired_to_stable_fingerprint(monkeypatch):
    monkeypatch.setattr(_mods, "_system_stable", lambda: ["固定行"])
    mod = next(m for m in all_modules() if m.key == "system")
    assert mod.cursor() == fingerprint(["固定行"])


# ---------------------------------------------------------------------------
# ② 禁语钉（用户点名清理的提示字样）
# ---------------------------------------------------------------------------
_REMOVED_PROMPTS = (
    "推进：",
    "多 plan 并行",
    "reviewer 附属队列",
    "审查面（inspector 队列）超时项",
    "审查入口",
    "批复入口",
    "口述动作",
    "Enter / `t` 进面板块",
)


def test_module_details_drop_named_prompts():
    for m in all_modules():
        joined = "\n".join(m.detail())
        for bad in _REMOVED_PROMPTS:
            assert bad not in joined, f"{m.key} 详情仍含旧提示字样：{bad}"


def test_pending_detail_has_deviation_block():
    mod = next(m for m in all_modules() if m.key == "pending")
    joined = "\n".join(mod.detail())
    assert "偏差报告（强制处理队列）" in joined  # s3：新增区块（同队列同待遇）


# ---------------------------------------------------------------------------
# ③ 图标空格（tools 全目录机械判据）
# ---------------------------------------------------------------------------
_ICON_CHARS = "✔⊘▶◆●★"
#: 豁免：图标后紧跟成对闭合符或引导/双引号（= 图标在括号内或字符串字面量末位，
#: 不构成「图标紧贴文字」；见任务书 §3 的「成对符号/句末除外」）。
_ICON_EXEMPT = frozenset(" ]）)\"'`")


def test_no_icon_glued_to_text_in_tools_source():
    tools_dir = Path(_mods.__file__).resolve().parents[1]
    hits: list[str] = []
    for p in sorted(tools_dir.rglob("*.py")):
        for no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(f"[{_ICON_CHARS}](.)", line):
                if m.group(1) in _ICON_EXEMPT:
                    continue
                hits.append(f"{p.name}:{no}: {line.strip()}")
    assert not hits, "图标后须隔一个空格再跟文字：\n" + "\n".join(hits)


def test_overview_marks_are_spaced():
    assert plans._plan_mark({}) == "▶ 进行中"
    assert plans._plan_mark({"completed_at": "2026-10-16T10:00:00+08:00"}) == "✔ 完成"
    assert plans._plan_mark({"abandoned_at": "2026-10-16T10:00:00+08:00"}) == "⊘ 中止"


# ---------------------------------------------------------------------------
# ④ 系统详情重排：首行=倒计时；stable 指纹排除倒计时行
# ---------------------------------------------------------------------------
def test_system_detail_first_line_is_countdown(monkeypatch):
    monkeypatch.setattr(
        _mods,
        "_countdown_line",
        lambda *a, **k: "  倒计时：距下次切换还有 1 分 1 秒（x → 入谷）",
    )
    lines = _mods._system_detail()
    assert lines[0].startswith("  倒计时"), "系统详情首行必须是倒计时行"
    assert all(_mods._COUNTDOWN_MARK not in ln for ln in _mods._system_stable()), (
        "stable 指纹必须排除倒计时行（徽标不因每秒跳动而常亮）"
    )
