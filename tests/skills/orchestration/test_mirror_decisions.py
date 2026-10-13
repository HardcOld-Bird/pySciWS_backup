"""inspector 决策审计面（用户 2026-10-13 06:2x 裁决，backlog 20261013-063000-leader）。

ctx 水位闸教训：inspector 免判决自动晋升的重要决策对用户隐蔽。两面执法：
``mirror status`` 增「决策」水位段（§10 注册纪律）；``pysci-mirror decisions`` 按决策链
呈现全貌——发现 → class → remedy_shape → 判决记录 → 实施 commit → 核验状态，
免判决+change 置顶。只读拼装（findings × backlog × run 摘要 × jury 案卷 × 决策日志），
不新增状态面。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import inspectd, mirror

from .conftest import seed_backlog


def _seed_finding(stamp: str, *, shape: str = "change", cls: str = "mechanical"):
    """登记一条发现并返回 rec（id 由 append_finding 生成，别自己猜）。"""
    if cls == "mechanical":
        return inspectd.append_finding(
            cls="mechanical",
            title=f"发现 {stamp}",
            repro="cmd",
            evidence="out",
            kind="defect",
            criteria="- 判据",
            remedy_shape=shape,
            stamp=stamp,
        )
    return inspectd.append_finding(
        cls="judgment", title=f"发现 {stamp}", criteria="- 判据", stamp=stamp
    )


def _backlog_entry(fid: str, *, channel: str, status: str = "done", note: str = ""):
    return {
        "id": fid,
        "member": "inspector",
        "source": "inspector",
        "summary": f"发现 {fid}",
        "channel": channel,
        "remedy_shape": "change",
        "status": status,
        "note": note,
    }


# ===========================================================================
#  1. status 概览段（§10 注册纪律）
# ===========================================================================
def test_status_items_registers_decision_section():
    names = [name for name, _fn in mirror.STATUS_ITEMS]
    assert "决策" in names, "新增「需用户关注事项」必须进登记表"


def test_status_decision_line_reports_watermark(iso_state):
    fid = _seed_finding("t1")["id"]
    seed_backlog(
        iso_state,
        [_backlog_entry(fid, channel="mechanical", status="pending")],
    )
    line = mirror._line_inspector_decisions()
    assert "自动晋升=1" in line and "待审计=1" in line
    assert "免判决+change=1" in line and "〔重点〕" in line, "免判决+change 必须高亮"
    assert "decisions" in line, "概览行要给全貌入口"


def test_status_decision_line_quiet_when_clean(iso_state):
    line = mirror._line_inspector_decisions()
    assert "免判决+change=0" in line and "〔重点〕" not in line


# ===========================================================================
#  2. decisions 全貌命令：决策链逐段可读
# ===========================================================================
def test_decisions_renders_full_chain(iso_state, capsys):
    fid = _seed_finding("chain")["id"]
    seed_backlog(
        iso_state,
        [
            _backlog_entry(
                fid,
                channel="mechanical",
                status="done",
                note="drain 批量销账（commit abc1234）",
            )
        ],
    )
    rc = mirror.main(["decisions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert fid in out, "发现段"
    assert "remedy_shape=change" in out, "修复形状段"
    assert "免判决（mechanical 通道）" in out, "判决段"
    assert "实施：abc1234" in out, "commit 段（note 提取）"
    assert "backlog=done" in out, "核验段"


def test_decisions_jury_case_shows_case_path(iso_state, capsys):
    fid = _seed_finding("jurycase")["id"]
    seed_backlog(iso_state, [_backlog_entry(fid, channel="jury")])
    case = iso_state / "evals" / "jury-cases" / f"{fid}.md"
    case.parent.mkdir(parents=True, exist_ok=True)
    case.write_text("# 案卷", encoding="utf-8")
    mirror.main(["decisions"])
    out = capsys.readouterr().out
    assert f"{fid}.md" in out, "J 案卷路径要出现在判决段"
    assert "J 案卷" in out


def test_decisions_free_change_sorts_first(iso_state, capsys):
    """免判决+change 置顶（最需审计者优先）——即便 restore 那条登记更晚。"""
    restore_id = _seed_finding("sorta", shape="restore")["id"]
    change_id = _seed_finding("sortz", shape="change")["id"]
    seed_backlog(
        iso_state,
        [
            _backlog_entry(restore_id, channel="mechanical"),
            _backlog_entry(change_id, channel="mechanical"),
        ],
    )
    mirror.main(["decisions"])
    out = capsys.readouterr().out
    assert out.index(change_id) < out.index(restore_id), (
        "免判决+change 必须排在 restore 之前"
    )
    assert "[!]" in out.split(change_id)[0][-60:], "置顶项带重点标记"


def test_decisions_filters_by_shape_and_class(iso_state, capsys):
    fid = _seed_finding("filt")["id"]
    seed_backlog(iso_state, [_backlog_entry(fid, channel="mechanical")])
    mirror.main(["decisions", "--shape", "restore"])
    assert "无匹配" in capsys.readouterr().out
    mirror.main(["decisions", "--shape", "change"])
    assert fid in capsys.readouterr().out
    mirror.main(["decisions", "--days", "0", "--class", "mechanical"])
    assert fid in capsys.readouterr().out
    mirror.main(["decisions", "--class", "judgment"])
    assert "无匹配" in capsys.readouterr().out


def test_decisions_is_read_only(iso_state, capsys):
    """只读拼装零新状态面：跑一遍命令前后，发现台账与 backlog 字节等。"""
    fid = _seed_finding("ro")["id"]
    seed_backlog(iso_state, [_backlog_entry(fid, channel="mechanical")])
    before = (
        inspectd.FINDINGS_PATH.read_bytes(),
        (iso_state / "backlog.json").read_bytes(),
    )
    mirror.main(["decisions"])
    capsys.readouterr()
    after = (
        inspectd.FINDINGS_PATH.read_bytes(),
        (iso_state / "backlog.json").read_bytes(),
    )
    assert before == after, "决策面是观察面，不许写状态"


def test_decisions_empty_state_is_graceful(iso_state, capsys):
    assert mirror.main(["decisions"]) == 0
    assert "无匹配" in capsys.readouterr().out


def test_decision_counts_match_rows(iso_state):
    n1 = _seed_finding("n1")["id"]
    n2 = _seed_finding("n2")["id"]
    seed_backlog(
        iso_state,
        [
            _backlog_entry(n1, channel="mechanical", status="done"),
            _backlog_entry(n2, channel="jury", status="pending"),
        ],
    )
    c = inspectd.decision_counts()
    assert c["promoted"] == 2
    assert c["open"] == 1, "n2 pending"
    assert c["free_change"] == 1, "n1 免判决+change；n2 有判决不计"
    assert c["judged"] == 1


def test_decisions_commit_from_run_done(iso_state, capsys):
    """实施 commit 的另一读路：run 摘要 .done 的 items[].id 命中（note 缺失时兜底）。"""
    fid = _seed_finding("rundone")["id"]
    seed_backlog(iso_state, [_backlog_entry(fid, channel="mechanical")])
    runs = iso_state / "devops-runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "20261013-010000.done").write_text(
        json.dumps(
            {"items": [{"id": fid, "status": "done", "commit": "def5678"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mirror.main(["decisions"])
    out = capsys.readouterr().out
    assert "实施：def5678" in out
