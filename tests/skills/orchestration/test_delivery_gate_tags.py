"""delivery-gate 的失败标签词表校验回归（E1，backlog 20261011-233001-leader）。

执法面：``<blocked tags="a,b">`` 的**属性存在时**才校验——每个标签必须在
``orchestration/state/failure-tags.json`` 的键集内，否则 exit 2 并给出词表全文；
``tags`` 缺省（不标注）合法，不因「没打标签」拒收交付。

全部用例直接 ``node`` 跑真 guard（承 20261010-193601 教训：guard 的 fail-open 会静默
吞掉 guard 自身的 bug，只有真跑才暴露）。词表用 tmp 里的定制文件（``PYSCI_PROJECT_ROOT``
注入），故「随附词表必须含这些键」的耦合被切断，同时另有一例专门钉住随附词表本身。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools.failure_tags import DEFAULT_TAGS, load_tags

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"

BLOCKED = "命令找不到，无法继续。<blocked>CLI 缺子命令</blocked>"
RESULT = "已完成。<result>交付正文</result>"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """一个自带词表的项目根（guard 从 ``PYSCI_PROJECT_ROOT`` 定位）。"""
    r = tmp_path / "proj"
    (r / "orchestration" / "state").mkdir(parents=True)
    (r / "orchestration" / "state" / "failure-tags.json").write_text(
        json.dumps(
            {
                "version": 1,
                "tags": {"cli_missing": "CLI 缺能力", "tool_bug": "工具缺陷"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (r / "pod").mkdir()
    return r


def run_gate(root: Path, msg: str) -> tuple[int, str]:
    """以真实 Stop hook 的姿势调用 guard：stdin 收事件 JSON，根路径走环境注入。"""
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


def test_blocked_without_tags_passes(root):
    """不标注是协议里的合法选择（缺省=空列表），不得因此拒收。"""
    assert run_gate(root, BLOCKED) == (0, "")


def test_blocked_with_in_vocab_tags_passes(root):
    rc, err = run_gate(root, '<blocked tags="cli_missing">CLI 缺子命令</blocked>')
    assert (rc, err) == (0, "")


def test_multi_tags_pass(root):
    """多标签与中文全角逗号混输都要认（组员手打习惯）。"""
    rc, err = run_gate(root, '<blocked tags="cli_missing，tool_bug">两个原因</blocked>')
    assert (rc, err) == (0, "")


def test_empty_tags_attribute_passes(root):
    rc, err = run_gate(root, '<blocked tags="">没查明</blocked>')
    assert (rc, err) == (0, "")


def test_unknown_tag_is_blocked(root):
    rc, err = run_gate(root, '<blocked tags="bogus_thing">瞎写的标签</blocked>')
    assert rc == 2
    assert "bogus_thing" in err
    # 整改指令要给出可用清单，否则组员只能瞎猜
    assert "cli_missing" in err and "tool_bug" in err


def test_mixed_known_and_unknown_reports_only_unknown(root):
    rc, err = run_gate(root, '<blocked tags="cli_missing,nope1,nope2">混合</blocked>')
    assert rc == 2
    listed = err.split("含词表外标签")[1].split("（当前")[0]
    assert "nope1" in listed and "nope2" in listed
    assert "cli_missing" not in listed  # 合法标签不该出现在整改指令的罪名里


def test_case_insensitive_tags_pass(root):
    """大小写不敏感：解析侧统一小写，别让组员因 Shift 被退回。"""
    rc, err = run_gate(root, '<blocked tags="CLI_MISSING">同上</blocked>')
    assert (rc, err) == (0, "")


def test_result_block_tags_not_checked(root):
    """tags 只属于失败分类面；``<result>`` 带 tags 属性不校验（解析侧也不入库）。"""
    rc, err = run_gate(root, '<result tags="bogus">做完了</result>')
    assert (rc, err) == (0, "")


def test_missing_vocab_file_fails_open(root):
    """词表文件被删 → 无法校验，只能放行（但不得静默：留 fail-open 痕迹）。"""
    (root / "orchestration" / "state" / "failure-tags.json").unlink()
    rc, err = run_gate(root, '<blocked tags="anything">正文</blocked>')
    assert rc == 0
    assert "fail-open" in err


def test_broken_vocab_json_fails_open(root):
    (root / "orchestration" / "state" / "failure-tags.json").write_text(
        "{坏 json", encoding="utf-8"
    )
    rc, err = run_gate(root, '<blocked tags="anything">正文</blocked>')
    assert rc == 0
    assert "fail-open" in err


# ===========================================================================
#  随附词表本身：文件形状与兜底集合的耦合
# ===========================================================================


def test_shipped_vocab_matches_default_keys():
    """随附词表的键集必须与代码兜底集一致——否则 guard 放行/统计另算（同一文件两侧读）。"""
    assert set(load_tags()) == set(DEFAULT_TAGS)


def test_shipped_vocab_definitions_non_empty():
    """纪律条款要求「增补须附定义」：随附文件每条都得有定义。"""
    empty = [t for t, d in load_tags().items() if not d.strip()]
    assert not empty, f"以下标签缺定义：{empty}"
