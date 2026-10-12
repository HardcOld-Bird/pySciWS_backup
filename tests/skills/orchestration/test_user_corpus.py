"""用户 prompt 语料倒序解析入库（E6b，backlog 20261012-164601-leader）。

锁定的判据（批准单 success_criteria，夹具全为构造样本，**不碰 ``data/others/`` 真语料**）：

1. **≥3 换行才切**：连续 ≥2 空行=消息分隔符；
2. **单空行不误切**：同一条消息内的排版续写必须留在一条里；
3. **倒序还原**：文件顶部=最新，解析后须输出旧→新正序，抽查 3 条与原文逐字对应；
4. **对账**：journal 新增行数=解析消息数，``source=user-corpus`` 可查，重跑不重复入库。

被解析脚本是 ``orchestration/bench/`` 下的一次性搬运工具（不落 src），故按文件路径
加载——测的就是入库时真正执行的那份代码。
"""

from __future__ import annotations

import importlib.util

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import journal

BENCH_SCRIPT = PROJECT_ROOT / "orchestration" / "bench" / "parse_user_corpus.py"


@pytest.fixture(scope="module")
def puc():
    """按路径加载解析脚本（bench 不在包内，同 test_budget_precommit 的加载法）。"""
    spec = importlib.util.spec_from_file_location("parse_user_corpus", BENCH_SCRIPT)
    assert spec and spec.loader, f"解析脚本缺失或不可加载：{BENCH_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def corpus(tmp_path):
    """语料目录脚手架：``write("<数字>名.txt", ...).load()`` → 正序时间线。"""
    root = tmp_path / "corpus"
    root.mkdir()

    class _Corpus:
        path = root

        def write(self, name: str, text: str):
            # write_bytes：跳过 write_text 的换行转换，夹具里的 \r\n 就是盘上的 \r\n
            # （否则 Windows 上 \n 再被转一次，分隔判据凭空多出一档 → 假阳性/假阴性都可能）。
            (root / name).write_bytes(text.encode("utf-8"))
            return root / name

    return _Corpus()


# ── 判据 1/2：分隔符=连续 ≥2 空行，单空行仅排版 ──────────────────────────────


def test_two_or_more_blank_lines_split_messages(puc):
    """阳性例①：≥3 个换行（连续 ≥2 空行）切成两条。"""
    msgs = puc.split_messages("上面更新的一条\n\n\n下面更早的一条")
    assert msgs == ["上面更新的一条", "下面更早的一条"]

    # 空行里混了空格/制表符也算空行（用户手写语料的常态）。
    assert len(puc.split_messages("甲\n \n\t\n乙")) == 2
    # 分隔符超过两个空行（三个、四个……）仍只算一个分隔，不多切出空条目。
    assert len(puc.split_messages("甲\n\n\n\n\n乙")) == 2


def test_single_blank_line_is_continuation_not_split(puc):
    """阳性例②：单空行（及段内单换行）不得把一条消息切开。"""
    one = "第一段\n续写行\n\n第二段排版\n尾行"
    msgs = puc.split_messages(one)
    assert len(msgs) == 1
    assert msgs[0] == one  # 原文逐字保留（含内部空行与换行）


def test_crlf_corpus_splits_after_eol_normalization(puc):
    """语料实测全为 CRLF：不归一 ``\\r\\n`` 则 ``\\n{3,}`` 判据永不命中（整文件成一条）。"""
    assert puc.split_messages("甲\r\n\r\n\r\n乙") == ["甲", "乙"]
    assert len(puc.split_messages("甲\r\n\r\n乙\r\n\r\n丙")) == 1


# ── 判据 3：倒序还原 ─────────────────────────────────────────────────────────


def test_reverse_ordered_file_is_restored_old_to_new(puc, corpus):
    """阳性例③：抽查 3 条——文件自上而下（新→旧）须还原成旧→新，正文逐字对应。"""
    newest = "致谢与下一步\n请再帮我核对一遍上节的推导"
    middle = "关于档位选型\n我建议 minimum 走火山免费档"
    oldest = "初始提问\n这个项目的编排该怎么搭"
    body = "\n\n\n".join([newest, middle, oldest])
    path = corpus.write("1近期.txt", body)

    msgs = puc.parse_file(path)
    assert [m["text"] for m in msgs] == [oldest, middle, newest]
    assert [m["rank"] for m in msgs] == [0, 1, 2]
    assert [m["total"] for m in msgs] == [3, 3, 3]
    assert all(m["fileno"] == 1 for m in msgs)


def test_multi_file_timeline_spans_old_periods_first(puc, corpus):
    """时段号=文件名前导数字；旧时段（号大）整段排在新时段（号小）之前。"""
    corpus.write("3早期.txt", "早期新\n\n\n早期老")
    corpus.write("2稍早.txt", "稍早的一条")
    corpus.write("1近期.txt", "近期的一条")

    timeline = puc.restore_timeline(puc.discover(corpus.path))
    assert [m["text"] for m in timeline] == [
        "早期老",
        "早期新",
        "稍早的一条",
        "近期的一条",
    ]
    assert [m["seq"] for m in timeline] == [1, 2, 3, 4]
    assert timeline[0]["fileno"] == 3 and timeline[-1]["fileno"] == 1


# ── 判据 4：入库对账、截断与幂等 ─────────────────────────────────────────────


def test_main_appends_one_line_per_message(puc, corpus, iso_state):
    """journal 新增行数=解析消息数；每条带 source=user-corpus 与「时序+文件锚」ts。"""
    corpus.write("3早期.txt", "早期第二条\n\n\n早期第一条")
    corpus.write(
        "1近期.txt",
        "最新的一条\r\n\r\n\r\n次新的一条\r\n\r\n这一行与上一行同属一条消息",
    )
    before = journal.line_count()

    assert puc.main(["--corpus-dir", str(corpus.path)]) == 0

    recs = journal.read_records()
    assert len(recs) == before + 4 == 4
    assert all(r["source"] == "user-corpus" and r["kind"] == "prompt" for r in recs)
    assert [r["ts"] for r in recs] == [
        "corpus:0001-F3#000",
        "corpus:0002-F3#001",
        "corpus:0003-F1#000",
        "corpus:0004-F1#001",
    ]
    assert [r["situation"] for r in recs] == [
        "早期第一条",
        "早期第二条",
        "次新的一条",
        "最新的一条",
    ]
    # 单空行续写留在同一条里（choice 含换行，未被误切成两条）。
    assert "这一行与上一行同属一条消息" in recs[2]["choice"]
    assert {r["corpus_file"] for r in recs} == {"3早期.txt", "1近期.txt"}


def test_choice_truncated_but_full_length_recorded(puc):
    """个人表述只入截断引用（原文留 gitignored 语料），长度与截断标记可查。"""
    text = "问" + "字" * 500
    rec = puc.to_record(
        {"text": text, "seq": 7, "fileno": 1, "rank": 3, "file": "1x.txt", "total": 10},
        max_chars=400,
    )
    assert rec["choice"].startswith("问") and len(rec["choice"]) == 401  # 400 + 省略号
    assert rec["truncated"] is True and rec["full_chars"] == 501
    assert rec["situation"] == "问" + "字" * 89  # 首行摘要另有上限


def test_rerun_is_idempotent_and_prepend_adds_only_new(puc, corpus, iso_state):
    """去重键=(文件, 末尾起算序号)：重跑零新增；用户在顶部追加只补最新那一条。"""
    path = corpus.write("1近期.txt", "第二条\r\n\r\n\r\n第一条")
    assert puc.main(["--corpus-dir", str(corpus.path)]) == 0
    first_two = journal.read_records()
    assert len(first_two) == 2

    assert puc.main(["--corpus-dir", str(corpus.path)]) == 0
    assert len(journal.read_records()) == 2  # 重跑不重复入库

    path.write_bytes("第三条\r\n\r\n\r\n第二条\r\n\r\n\r\n第一条".encode())
    assert puc.main(["--corpus-dir", str(corpus.path)]) == 0
    after = journal.read_records()
    assert len(after) == 3
    assert after[:2] == first_two  # 既有记录的锚与序号不因顶部追加而漂移
    assert after[-1]["situation"] == "第三条"


def test_dry_run_writes_nothing(puc, corpus, iso_state):
    corpus.write("1近期.txt", "甲\n\n\n乙")
    assert puc.main(["--corpus-dir", str(corpus.path), "--dry-run"]) == 0
    assert journal.read_records() == []


def test_missing_corpus_dir_fails_loudly(puc, tmp_path, capsys):
    """语料目录不在=rc 2 并报出路径（worktree 里默认指主树，漏 --corpus-dir 要看得见）。"""
    absent = tmp_path / "nope"
    assert puc.main(["--corpus-dir", str(absent)]) == 2
    assert "无 *.txt" in capsys.readouterr().out
