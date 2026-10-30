"""delivery-gate 的 result/blocked 互斥执法（insp-20261022-141826）。

charter 交付协议：最终回复必须**含且仅含一个** `<result>`（成功）或 `<blocked>`（失败）
标签块；旧 gate 只用存在性判断（`TAG_RE.test`），同给两块被放行，而下游解析器只取第一块
→ 审计面判成功、正文含失败，交付状态分裂。真跑 guard 回归（承 20261010-193601 教训）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"

RESULT = "已完成。<result>交付正文</result>"
BLOCKED = "命令找不到，无法继续。<blocked>CLI 缺子命令</blocked>"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """自带词表的项目根（guard 从 ``PYSCI_PROJECT_ROOT`` 定位）。"""
    r = tmp_path / "proj"
    (r / "orchestration" / "state").mkdir(parents=True)
    (r / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tags": {"cli_missing": "CLI 缺能力"},
                "reasons": {"needs_decision": "待裁决"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
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


def test_result_and_blocked_together_rejected(root):
    """inspector 复现：两块同给 → exit 2（互斥生效）。"""
    rc, err = run_gate(root, "<result>成功</result><blocked>失败</blocked>")
    assert rc == 2
    assert "互斥" in err and "含且仅含一个" in err


def test_blocked_with_attrs_still_rejected(root):
    """带属性的 blocked 同样计入互斥。"""
    rc, err = run_gate(
        root, '<result>好</result><blocked reason="needs_decision">待裁决</blocked>'
    )
    assert rc == 2 and "互斥" in err


def test_only_result_passes(root):
    assert run_gate(root, RESULT) == (0, "")


def test_only_blocked_passes(root):
    assert run_gate(root, BLOCKED) == (0, "")


def test_no_block_still_rejected(root):
    rc, err = run_gate(root, "没有任何交付块的正文")
    assert rc == 2 and "含且仅含一个" in err


def test_prose_bare_tag_not_counted(root):
    """正文里裸引用 <blocked>（无闭合）不参与配对——不误伤讨论标签的交付。"""
    msg = "<result>好</result>（曾以 <blocked> 而停链，本次成功）"
    assert run_gate(root, msg) == (0, "")
