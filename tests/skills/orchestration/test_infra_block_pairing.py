"""infra_suggestion 块配对不得跨段落（backlog 20261017-053820-devops）。

缺陷：交付正文**引用/讨论** `<infra_suggestion>` 标签时，gate 的 ``INFRA_RE``（全局非贪婪）
把先出现的**裸开口标签**与后文完整块的闭标签**跨段落配对**成一块——属性面取自裸标签（空）
→ 误判「缺 kind」退回（成员在正文里明明给了带 kind 的完整块）。

修法：块体**不得越过另一个开口标签**（遇下一个 ``<infra_suggestion`` 即止）。于是裸引用不
参与配对；而**整块自身**（裸开＋同处闭）仍照常执法（缺 kind 依旧 exit 2）。

判据（success_criteria）：
① 裸开口 + 后文带 kind 的完整块 → gate **不退回**；
② 真正**只有**不带属性开口标签的块 → 仍 exit 2（且报错回显被配对原文）；
③ Python 侧解析同一交付取到**真正的**完整块（不跨段配对）；
④ 两个真块并存 → 仍报「合并成一个」（旧执法不被削弱）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools.delivery import parse_delivery

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"

# ① 正文先裸开一个标签引用，后文才是真正的提案块
MSG_PROSE_THEN_BLOCK = (
    "<result>完成。\n\n"
    "正文在讨论标签本身：<infra_suggestion> 的 kind 属性是必填的。\n\n"
    "以下才是本条提案：\n\n"
    '<infra_suggestion kind="usability">\n'
    "kind: usability\n"
    "success_criteria: 演示用\n"
    "场景：正文引用标签被误配。\n"
    "</infra_suggestion>\n"
    "</result>"
)

# ② 真正的提案**只有裸开口标签**（无 kind）——必须仍被退回
MSG_BARE_ONLY = (
    "<result>ok</result>\n\n"
    "<infra_suggestion>\nkind: usability\n场景：无属性开口。\n</infra_suggestion>"
)

# ④ 两个真块并存（旧执法：报「合并成一个」）——不得被本次改动洗掉
MSG_TWO_BLOCKS = (
    "<result>ok</result>\n"
    '<infra_suggestion kind="defect">甲</infra_suggestion>\n'
    '<infra_suggestion kind="usability">乙</infra_suggestion>'
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    (r / "orchestration" / "state").mkdir(parents=True)
    (r / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps({"version": 1, "tags": {}}, ensure_ascii=False), encoding="utf-8"
    )
    (r / "pod").mkdir()
    return r


def run_gate(root: Path, msg: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYSCI_PROJECT_ROOT"] = str(root)
    env["PYSCI_POD"] = str(root / "pod")
    env.pop("PYSCI_DEPLOYED_SKILLS", None)
    proc = subprocess.run(
        ["node", str(GUARD)],
        input=json.dumps({"last_assistant_message": msg, "stop_hook_active": False}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return proc.returncode, proc.stderr


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="需要 node 运行 guard"
)


# ---------------------------------------------------------------------------
# ① 裸引用 + 后文完整块 → 不退回
# ---------------------------------------------------------------------------
def test_prose_reference_then_real_block_is_accepted(root):
    rc, err = run_gate(root, MSG_PROSE_THEN_BLOCK)
    assert rc == 0, err
    assert "缺 kind" not in err, err


def test_python_parser_picks_the_real_block(root):
    """Python 侧同修：取到**完整块**而非跨段配对的巨块。"""
    d = parse_delivery(MSG_PROSE_THEN_BLOCK)
    assert d.infra_kind == "usability", d.infra_kind
    assert "场景：正文引用标签被误配" in d.infra_suggestion
    assert "kind 属性是必填的" not in d.infra_suggestion, "跨段配对未修好"


# ---------------------------------------------------------------------------
# ② 真正只有裸开口（无 kind）→ 仍 exit 2，且回显被配对原文
# ---------------------------------------------------------------------------
def test_bare_block_without_kind_still_rejected(root):
    rc, err = run_gate(root, MSG_BARE_ONLY)
    assert rc == 2, err
    assert "缺 kind" in err, err
    assert "被配对的块开头" in err, "报错应回显被配对原文，帮成员分辨是引用被误配"


def test_python_parser_bare_block_has_empty_kind():
    d = parse_delivery(MSG_BARE_ONLY)
    assert d.infra_kind == "" and "场景：无属性开口" in d.infra_suggestion


# ---------------------------------------------------------------------------
# ④ 两个真块并存 → 旧执法（合并成一个）不被削弱
# ---------------------------------------------------------------------------
def test_two_real_blocks_still_flagged(root):
    rc, err = run_gate(root, MSG_TWO_BLOCKS)
    assert rc == 2, err
    assert "合并成" in err, err
