"""交付块属性面的**引号感知**扫描（backlog 20261012-025325-devops）。

威胁面：属性旧写法 `[^>]*` 用「第一个 `>`」界定标签开头，于是属性**值里**的 `>`
（比较符号、数学式、HTML 片段）会让匹配在引号内部提前收尾，剩余部分被当正文起点：

    <result reason="x > y">正文 B</result>          → body 变成 'y">正文 B'
    <artifact check="none" reason="优于 > 阈值">p/q.pdf</artifact>
                                                    → path 变成 '阈值">p/q.pdf'

比 parse_error 更隐蔽的是**它一切正常**：kind 仍判 result、``warning`` 仍为空
（严格面自认为匹配上了，兜底面不触发），台账一片绿灯，而机械验收去 check 一个
不存在的路径。同理 gate 侧 `[^>]*` 会把 ``tags=``/``kind=`` 截在属性面之外 →
gate 看不见标签、跳过词表校验，而 Python 侧读得到（或反之）——同一交付两侧读数分裂。

锁住的核心性质：
① 值内含 `>` / `</` 的交付，正文与产物 path/reason 干净，且 ``warning`` 恒空
   （这是现协议形态，不许走兜底）；
② **因果对照**：换回旧 `[^>]*` 属性面，同一输入必须复现污染（否则测试脱靶）；
③ 严格面匹配不上的畸形（未闭合引号）不静默通过：同行者由宽松面捞回并附
   ``PROTOCOL_LAG_HINT``，跨行吞段者照旧 ``parse_error``——本次改动不许把违规洗成告警，
   也不许把可救的交付洗成静默截断；
④ Node 侧 delivery-gate 与 Python 侧对同一交付**读数一致**，且两侧属性模式同源
   （单点钉住字面量，防只改一侧）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import delivery
from pysci.skills.orchestration.tools.delivery import (
    PROTOCOL_LAG_HINT,
    parse_delivery,
)

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"

#: 建议书的复现原文（逐字，不重新措辞——success_criteria 要的就是这两条的读数）。
RESULT_GT_ATTR = '<result reason="x > y">正文 B</result>'
ARTIFACT_GT_ATTR = '<result>内嵌 <artifact check="none" reason="优于 > 阈值">p/q.pdf</artifact></result>'

# --- ① 值内含 > 的交付：干净读数、不走兜底 ----------------------------------


def test_gt_in_attr_keeps_result_body_intact():
    d = parse_delivery(RESULT_GT_ATTR)
    assert d.kind == "result" and d.body == "正文 B", d
    assert d.warning == "", "现协议形态绝不走兜底（污染读数不该伪装成版本错位）"


def test_gt_in_artifact_attr_yields_clean_path_and_reason():
    d = parse_delivery(ARTIFACT_GT_ATTR)
    assert d.kind == "result" and d.body.startswith("内嵌"), d
    assert [(a.path, a.check, a.reason) for a in d.artifacts] == [
        ("p/q.pdf", "none", "优于 > 阈值")
    ], "path 被引号内的 > 截断是本单事故本体"


def test_close_tag_text_in_attr_value_is_not_a_boundary():
    """`</` 出现在值里同样不许当标签结束（组员会写「形如 </result 的串」这类说明）。"""
    d = parse_delivery('<result reason="形如 </artifact 的串">正文 C</result>')
    assert (d.kind, d.body, d.warning) == ("result", "正文 C", ""), d

    a = parse_delivery(
        '<result><artifact check="none" reason="对比 </artifact 串">p/q.pdf</artifact></result>'
    )
    assert [(x.path, x.reason) for x in a.artifacts] == [
        ("p/q.pdf", "对比 </artifact 串")
    ], a


def test_blocked_tags_after_a_gt_value_still_read():
    """tags 排在含 `>` 的属性**之后**时，旧面把属性面截断 → tags 静默丢失。"""
    d = parse_delivery(
        '<blocked reason="劣于 > 阈值" tags="cli_missing">正文</blocked>'
    )
    assert d.kind == "blocked" and d.body == "正文", d
    assert d.tags == ["cli_missing"], d


def test_apostrophe_inside_double_quoted_value_is_fine():
    """双引号段允许裸单引号（it's / 中文撇号），否则真实写作习惯会被截断。"""
    d = parse_delivery('<result reason="it\'s > 0">正文 D</result>')
    assert (d.kind, d.body, d.warning) == ("result", "正文 D", ""), d


# --- ② 因果对照：旧属性面必须复现污染 ---------------------------------------


LEGACY_ATTRS_STRICT = {
    "result": re.compile(
        r"<result\b(?P<attrs>[^>]*)>(?P<body>.*?)</result>", re.DOTALL
    ),
    "blocked": re.compile(
        r"<blocked\b(?P<attrs>[^>]*)>(?P<body>.*?)</blocked>", re.DOTALL
    ),
}
LEGACY_ATTRS_ARTIFACT = re.compile(
    r"<artifact\b(?P<attrs>[^>]*)>(?P<path>.*?)</artifact>", re.DOTALL
)


def _legacy_attrs(monkeypatch):
    """把属性面换回建议书复现时的写法（``[^>]*``），只换这一处，其余不变。"""
    monkeypatch.setattr(delivery, "_STRICT_BLOCKS", dict(LEGACY_ATTRS_STRICT))
    monkeypatch.setattr(delivery, "_ARTIFACT_RE", LEGACY_ATTRS_ARTIFACT)


def test_legacy_attr_face_reproduces_the_pollution(monkeypatch):
    """断言反过来（旧面也读得干净）说明测试没打在病灶上。"""
    _legacy_attrs(monkeypatch)
    bad = parse_delivery(RESULT_GT_ATTR)
    assert bad.kind == "result" and bad.warning == "", bad
    assert bad.body == 'y">正文 B', bad.body  # 静默截断，无告警——事故本体

    worse = parse_delivery(ARTIFACT_GT_ATTR)
    assert [(a.path, a.reason) for a in worse.artifacts] == [('阈值">p/q.pdf', "")], (
        worse
    )


# --- ③ 畸形形态仍走 parse_error / warning 面 --------------------------------


def test_unclosed_quote_on_one_line_is_rescued_with_warning():
    """严格面匹配不上（引号不成对）→ 宽松面捞回并告警，不与兜底面语义打架。"""
    d = parse_delivery('<result reason="未闭合 >正文 B</result>')
    assert d.kind == "result" and d.body == "正文 B", d
    assert PROTOCOL_LAG_HINT in d.warning, d.warning


def test_unclosed_quote_spanning_lines_does_not_swallow_the_delivery():
    """引号段禁跨行的理由：未闭合引号若跨行找配对，会把整段正文吞进属性面。

    期望落点是 ``parse_error``（既非静默截断、也非「救回来」）：正文里出现裸 ``"`` 且
    后面还跟着 ``>`` 时，任何面都不该自信地读数。
    """
    d = parse_delivery('<result reason="换行\n后续正文</result>\n')
    assert d.kind == "parse_error" and d.warning == "", d


# --- ④ Node 侧 gate 同形，两侧读数一致 --------------------------------------

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="需要 node 运行 guard"
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """自带词表的项目根 + 空 pod（gate 从 ``PYSCI_PROJECT_ROOT``/``PYSCI_POD`` 定位）。"""
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


def run_gate(root: Path, msg: str, guard: Path = GUARD) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYSCI_PROJECT_ROOT"] = str(root)
    env["PYSCI_POD"] = str(root / "pod")
    env.pop("PYSCI_DEPLOYED_SKILLS", None)
    proc = subprocess.run(
        ["node", str(guard)],
        input=json.dumps({"last_assistant_message": msg, "stop_hook_active": False}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return proc.returncode, proc.stderr


def legacy_gate_copy(tmp_path: Path) -> Path:
    """把 guard 的属性面换回旧写法 ``[^>]*`` 的一份副本（因果对照用）。"""
    src = GUARD.read_text(encoding="utf-8")
    old = re.sub(
        r"const ATTRS = String\.raw`[^`]*`;",
        "const ATTRS = String.raw`[^>]*`;",
        src,
        count=1,
    )
    assert old != src and "String.raw`[^>]*`" in old, "换面失败，对照无意义"
    dst = tmp_path / "delivery-gate-legacy.mjs"
    dst.write_text(old, encoding="utf-8")
    return dst


def test_gate_reads_gt_in_attrs_and_passes(root):
    """值里的 `>` 不得让 gate 判成「缺交付块」（旧面靠第一个 `>` 收尾仍能匹配，
    但属性面被截断——本例先把「放行」钉住，下一条钉住读数一致）。"""
    assert run_gate(root, RESULT_GT_ATTR) == (0, "")
    assert run_gate(root, ARTIFACT_GT_ATTR) == (0, "")


def test_gate_and_parser_agree_on_tags_after_gt_value(root):
    """同一交付，gate 与解析器都必须看见 ``tags``：词表外标签两侧一致地拦/记。

    旧属性面（``[^>]*``）在 ``优于 > 阈值`` 处截断属性 → gate 看不见 tags、放行，
    而 Python 侧的 ``_ATTR_RE`` 从完整属性面读到 bogus_thing → 两侧读数分裂。
    """
    msg = '<blocked reason="优于 > 阈值" tags="bogus_thing">正文</blocked>'
    rc, err = run_gate(root, msg)
    assert rc == 2 and "bogus_thing" in err, (rc, err)
    assert parse_delivery(msg).tags == ["bogus_thing"]


def test_gate_with_legacy_attrs_silently_skips_vocab(root, tmp_path):
    """因果对照：旧面的属性截断让 gate **看不见** tags → 词表外标签照样放行。

    这不是「宽松一点」，而是执法面失聪——同一交付 Python 侧记 bogus_thing 进台账、
    gate 侧零拦截。断言反过来（旧面也拦得住）说明本单的 Node 侧改造无的放矢。
    """
    legacy = legacy_gate_copy(tmp_path)
    # 前置对照：旧面对「tags 排在最前」照常拦截——失聪只发生在含 > 的属性之后，
    # 不是整条 gate 坏了（否则本例的证明力归零）。
    assert run_gate(root, '<blocked tags="bogus_thing">正文</blocked>', legacy)[0] == 2
    rc, err = run_gate(
        root,
        '<blocked reason="优于 > 阈值" tags="bogus_thing">正文</blocked>',
        legacy,
    )
    assert rc == 0, (rc, err)  # 旧面：看不见 tags，放行
    assert parse_delivery(
        '<blocked reason="优于 > 阈值" tags="bogus_thing">正文</blocked>'
    ).tags == ["bogus_thing"]  # 解析侧看得见 → 两侧分裂实锤


def test_gate_reads_infra_kind_after_gt_value(root):
    """``kind=`` 排在含 `>` 的属性之后时，旧面截断 → gate 报「缺 kind」退回改正。

    组员因此被逼着改掉一个本来合规的写法（协议执行端反过来污染协议）。
    """
    msg = (
        "<result>交付正文</result>"
        '<infra_suggestion note="优于 > 阈值" kind="defect">'
        "kind: defect\nsuccess_criteria：属性面引号感知\n</infra_suggestion>"
    )
    assert run_gate(root, msg) == (0, "")
    d = parse_delivery(msg)
    assert d.infra_kind == "defect", d.infra_kind  # 两侧同读 kind


def test_both_sides_declare_the_same_attr_pattern():
    """属性面只有一份语义：两侧字面量必须一致（只改一侧 = 读数漂移的源头）。"""
    src = GUARD.read_text(encoding="utf-8")
    node_attrs = re.search(r"const ATTRS = String.raw`([^`]*)`;", src).group(1)
    assert node_attrs == delivery._ATTRS, (node_attrs, delivery._ATTRS)
