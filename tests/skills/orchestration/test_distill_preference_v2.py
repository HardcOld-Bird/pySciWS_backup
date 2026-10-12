"""宪章 v2 蒸馏跳脚本的非派发面（E6b，backlog 20261012-164601-leader）。

真派发要烧一跳，本文件只锁**不需要派发就能错**的三件事：

1. **拒写生效面**：``--out`` 指到 ``pods/twin/PREFERENCE.md`` 必须 rc=2 且不启动会话
   （批准单「不动生效面」是这条支路唯一的安全边界，草案错了用户能看出，生效面错了
   就是把没确认的偏好当用户裁决用）；
2. **切片语义**：默认取该时段文件的**最新一半**（用户指令 2026-10-12 17:1x：优先吸收
   靠上部=最新内容，语料量大时至少覆盖前半）；
3. **协议标签不入草案**：``<artifact>`` 之类交付协议噪声不得留在宪章正文里。

夹具为构造样本，不碰 ``data/others/`` 真语料。
"""

from __future__ import annotations

import importlib.util

import pytest

from pysci.paths import PROJECT_ROOT

BENCH_SCRIPT = PROJECT_ROOT / "orchestration" / "bench" / "distill_preference_v2.py"
EFFECTIVE = PROJECT_ROOT / "orchestration" / "pods" / "twin" / "PREFERENCE.md"


@pytest.fixture(scope="module")
def dv():
    spec = importlib.util.spec_from_file_location("distill_preference_v2", BENCH_SCRIPT)
    assert spec and spec.loader, f"蒸馏脚本缺失：{BENCH_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    body = "\n\n\n".join(f"第{i}条消息" for i in range(1, 6))  # 文内序=新→旧
    (root / "1近期.txt").write_bytes(body.encode("utf-8"))
    return root


def test_refuses_to_write_effective_charter(dv, capsys):
    """--out 命中生效宪章（含别名路径）即拒，且拒在派发之前（零成本、零会话）。"""
    assert dv.main(["--out", str(EFFECTIVE)]) == 2
    assert dv.main(["--out", str(EFFECTIVE.parent / "./PREFERENCE.md")]) == 2
    assert "拒写生效面" in capsys.readouterr().out


def test_default_slice_is_newest_half_of_the_period(dv, fake_corpus):
    """默认前半=最新 ceil(N/2) 条（rank 2/3/4），倒序文件**顶部**那几条必须在内。"""
    puc = dv._load_parser()
    picked = dv.select_slice(puc, fake_corpus, fileno=1, newest=0)
    assert [m["text"] for m in picked] == ["第3条消息", "第2条消息", "第1条消息"]
    assert [m["rank"] for m in picked] == [
        2,
        3,
        4,
    ]  # rank 从最旧端起算，4=文件顶部=最新

    assert [m["text"] for m in dv.select_slice(puc, fake_corpus, 1, newest=1)] == [
        "第1条消息"
    ]


def test_unknown_period_exits_without_dispatch(dv, fake_corpus):
    """时段号没有对应文件=硬退出（不静默给一个空注入去烧一跳）。"""
    puc = dv._load_parser()
    with pytest.raises(SystemExit):
        dv.select_slice(puc, fake_corpus, fileno=7, newest=0)


def test_protocol_tags_stripped_from_draft_body(dv):
    """交付尾部的 <artifact> 标签不入草案正文（草案要能被用户直接并进宪章）。"""
    body = "## 新增条款\n- **P-22 …**\n\n" + (
        '<artifact check="none" reason="无落盘产物">无落盘产物</artifact>'
    )
    assert (
        dv.PROTOCOL_TAG_RE.sub("", body.strip()).strip() == "## 新增条款\n- **P-22 …**"
    )


def test_prompt_carries_baseline_corpus_and_hard_format(dv):
    """注入三件套齐备：v1 基线全文、语料原文（标条目号）、覆盖判断格式的输出要求。"""
    prompt = dv.build_prompt(
        "# PREFERENCE.md — twin 判断宪章\n- **P-01 …**",
        [{"text": "我偏好先草案后实施", "rank": 33, "total": 66, "file": "1近期.txt"}],
        corpus_file="1近期.txt",
    )
    assert "P-01" in prompt and "我偏好先草案后实施" in prompt
    assert "### 语料 #034" in prompt  # rank 从 0 起算 → 人读条目号从 1 起算
    assert dv.DRAFT_MARK in prompt
    assert "覆盖你规程里的 JSON 判断格式" in prompt
