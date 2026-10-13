"""会话记录审计工具链（种子 backlog 20261013-130000-leader，用户设计 2026-10-13）

inspector v2 的 transcript 工具链：`list` 优先级选靶 → `scan` 简版定位 → `fetch` 按锚
取原文。**纯 agent 导向**：返回值即阅读面，落盘只是缓存。

success_criteria 逐条对上：

- **三态判定矩阵** → ① 组（首达 5 可检／检查后已检／新增 5 重可检／静止 2h 重可检／
  短会不可检／持续活跃不误判）；
- **优先级实测**（leader 先、新先）→ ② 组（构造 leader+组员混合样本，验排序与预算）；
- **噪声过滤** → ③ 组（`<synthetic>` 合成回合与 attachment 等簿记行不进简版）；
- **超长截断 + 大小标注** → ③ 组；
- **行锚可复现**（finding 的 `--repro` 引锚）→ ④ 组（scan 出的锚拿去 fetch 能取回同一段）；
- **缓存目录清理有效** → ⑤ 组。

样本一律**自造**（临时 projects 根 + 自写 jsonl）：真会话档在 ``~/.qoder-cn`` 下，
本仓测试不得读它（体量上百 MB、且随开发者机器漂）。
"""

from __future__ import annotations

import json
from pathlib import Path

from pysci.skills.orchestration.tools import transcript as tr

NOW = 1_800_000_000.0  # 固定「现在」，静止判据才可断言
HOUR = 3600.0


def _user(text: str, *, ts: str = "2026-10-13T10:00:00.000Z") -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }


def _ts_at(hour: int, minute: int = 0) -> str:
    """固定日期的某时刻（会话内各记录共用，last_active 判据才可断言）。"""
    return f"2026-10-13T{hour:02d}:{minute:02d}:00.000Z"


def _tool_result(ts: str = "2026-10-13T10:00:01.000Z") -> dict:
    """工具结果回灌——也走 ``type=user``，但**不是**新回合（回合数不许虚高）。"""
    return {
        "type": "user",
        "timestamp": ts,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
    }


def _assistant(
    blocks: list[dict],
    *,
    model: str = "seatchannel",
    ts: str = "2026-10-13T10:00:02.000Z",
) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {"role": "assistant", "model": model, "content": blocks},
    }


def _bookkeeping(kind: str = "attachment") -> dict:
    return {"type": kind, "timestamp": "2026-10-13T10:00:00.500Z", "attachment": {}}


def _session(
    path: Path, rounds: int, *, with_noise: bool = False, hour: int = 10
) -> Path:
    """造一个 ``rounds`` 回合的会话档（每回合 = user + assistant text + 一条工具调用）。

    ``hour``：该会话内所有记录的时间戳小时——``last_active`` 取自**记录时间戳**
    （不是文件 mtime），故「新 > 旧」的用例靠它区分。
    """
    stamp = _ts_at(hour)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        json.dumps({"type": "workspace-directories", "directories": ["/x"]}),
        json.dumps(
            {"type": "runtime-config", "model": "m", "timestamp": 1_791_600_000_000}
        ),
    ]
    for i in range(1, rounds + 1):
        lines.append(json.dumps(_user(f"第 {i} 轮任务书正文", ts=stamp)))
        if with_noise:
            lines.append(json.dumps(_bookkeeping("attachment")))
            lines.append(
                json.dumps(
                    _assistant(
                        [
                            {
                                "type": "text",
                                "text": "Custom model service error, retry later",
                            }
                        ],
                        model=tr.SYNTHETIC_MODEL,
                    )
                )
            )
        lines.append(json.dumps(_tool_result()))
        lines.append(
            json.dumps(
                _assistant(
                    [
                        {"type": "thinking", "thinking": "内部推理" * 40},
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                        {"type": "text", "text": f"第 {i} 轮正式回复"},
                    ]
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _project(root: Path, key: str) -> Path:
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cursor(tmp_path: Path) -> Path:
    return tmp_path / "cursor.json"


# ---------------------------------------------------------------------------
# ① 三态判定矩阵（纯函数）
# ---------------------------------------------------------------------------
def test_first_reach_five_is_checkable():
    """首达 5 回合且未检过 → 可检（4 回合还不行）。"""
    assert tr.classify(None, rounds=5, last_active=NOW, now=NOW) == tr.CHECKABLE
    assert tr.classify(None, rounds=4, last_active=NOW, now=NOW) == tr.SHORT


def test_short_active_session_not_checkable():
    """短于 5 回合且未静止 → 不检查（持续活跃不误判：刚到 2 回合不算）。"""
    assert tr.classify(None, rounds=2, last_active=NOW - 60, now=NOW) == tr.SHORT


def test_checked_stays_checked_without_new_rounds():
    prev = {"rounds_seen": 6, "checked_at": "2026-10-13T10:00:00+08:00"}
    assert tr.classify(prev, rounds=6, last_active=NOW - 60, now=NOW) == tr.CHECKED


def test_new_five_rounds_recheckable():
    """已检后新增 ≥5 回合 → 无条件重可检（不管静止与否）。"""
    prev = {"rounds_seen": 6, "checked_at": "2026-10-13T10:00:00+08:00"}
    assert tr.classify(prev, rounds=11, last_active=NOW, now=NOW) == tr.CHECKABLE
    assert tr.classify(prev, rounds=10, last_active=NOW, now=NOW) == tr.CHECKED


def test_static_two_hours_with_new_round_recheckable():
    """静止 2h 且期间新增 ≥1 回合 → 重可检（只多一轮也值得复看）。"""
    prev = {"rounds_seen": 6, "checked_at": "2026-10-13T10:00:00+08:00"}
    frozen = NOW - 3 * HOUR  # 最后活跃在 3 小时前
    assert tr.classify(prev, rounds=7, last_active=frozen, now=NOW) == tr.CHECKABLE


def test_active_session_with_one_new_round_not_recheckable():
    """活跃会话只多一轮、又没静止 → 仍是已检查（持续活跃不误判）。"""
    prev = {"rounds_seen": 6, "checked_at": "2026-10-13T10:00:00+08:00"}
    assert tr.classify(prev, rounds=7, last_active=NOW - 60, now=NOW) == tr.CHECKED


def test_static_but_no_new_round_not_recheckable():
    """静止到点但期间零新增回合 → 不重可检（没有新内容可看）。"""
    prev = {"rounds_seen": 6, "checked_at": "2026-10-13T10:00:00+08:00"}
    assert (
        tr.classify(prev, rounds=6, last_active=NOW - 5 * HOUR, now=NOW) == tr.CHECKED
    )


# ---------------------------------------------------------------------------
# ② 优先级与预算（list）
# ---------------------------------------------------------------------------
def test_list_orders_leader_first_then_newest(tmp_path):
    """优先级：leader 会话 > 组员，同组内新 > 旧（新=最后活跃晚）。"""
    root = tmp_path / "projects"
    leader_key = "D--repo-root"
    _session(_project(root, leader_key) / "l-old.jsonl", 5, hour=8)
    _session(_project(root, leader_key) / "l-new.jsonl", 5, hour=11)
    _session(
        _project(root, "D--x-orchestration-pods-devops") / "m-old.jsonl", 5, hour=9
    )
    _session(
        _project(root, "D--x-orchestration-pods-devops") / "m-new.jsonl", 5, hour=10
    )

    refs = tr.list_sessions(
        root=root, cursor_path=_cursor(tmp_path), leader_key=leader_key, budget=0
    )
    kinds = [r.kind for r in refs]
    assert kinds[0] == "leader" and kinds[1] == "leader", "leader 会话必须排在组员前面"
    assert kinds[2:] == ["member", "member"]

    lsids = [r.sid for r in refs if r.kind == "leader"]
    msids = [r.sid for r in refs if r.kind == "member"]
    assert lsids == ["l-new", "l-old"], "同组内新 > 旧（组长那组）"
    assert msids == ["m-new", "m-old"], "同组内新 > 旧（组员那组）"


def test_list_budget_keeps_top_three(tmp_path):
    """每轮 ≤3 会话（用户确认）：预算在排序**之后**截断，拿到的是最该看的那几个。"""
    root = tmp_path / "projects"
    d = _project(root, "D--x-orchestration-pods-devops")
    for i in range(6):
        _session(d / f"s{i}.jsonl", 5)
    refs = tr.list_sessions(
        root=root, cursor_path=_cursor(tmp_path), leader_key="", budget=tr.ROUND_BUDGET
    )
    assert len(refs) == tr.ROUND_BUDGET == 3
    assert (
        tr.list_sessions(
            root=root, cursor_path=_cursor(tmp_path), leader_key="", budget=0
        )
        and len(
            tr.list_sessions(
                root=root, cursor_path=_cursor(tmp_path), leader_key="", budget=0
            )
        )
        == 6
    )


def test_list_only_checkable_by_default(tmp_path):
    """缺省只列可检；``only_checkable=False`` 才把 short/checked 一并列出。"""
    root = tmp_path / "projects"
    d = _project(root, "D--x-orchestration-pods-figure")
    _session(d / "big.jsonl", 5)
    _session(d / "tiny.jsonl", 1)
    cur = _cursor(tmp_path)

    only = tr.list_sessions(root=root, cursor_path=cur, leader_key="", budget=0)
    assert [r.sid for r in only] == ["big"]
    everything = tr.list_sessions(
        root=root, cursor_path=cur, leader_key="", budget=0, only_checkable=False
    )
    assert {r.sid for r in everything} == {"big", "tiny"}
    assert {r.sid: r.state for r in everything}["tiny"] == tr.SHORT, (
        "1 回合且从未检过 = 太短，不是 checked"
    )


def test_mark_checked_moves_session_out_of_checkable(tmp_path):
    """检查后记游标：同一会话不再出现在可检列表（否则每轮都顶同一个）。"""
    root = tmp_path / "projects"
    d = _project(root, "D--x-orchestration-pods-devops")
    _session(d / "s1.jsonl", 6)
    cur = _cursor(tmp_path)
    refs = tr.list_sessions(root=root, cursor_path=cur, leader_key="", budget=0)
    assert [r.sid for r in refs] == ["s1"]

    cu = tr.load_cursor(cur)
    tr.mark_checked(cu, refs[0], now_iso="2026-10-13T18:00:00+08:00")
    tr.save_cursor(cu, cur)

    assert tr.list_sessions(root=root, cursor_path=cur, leader_key="", budget=0) == []


# ---------------------------------------------------------------------------
# ③ 简版：噪声过滤 + 折叠 + 截断标注
# ---------------------------------------------------------------------------
def test_scan_filters_synthetic_and_bookkeeping(tmp_path):
    """`<synthetic>` 合成回合与 attachment 等簿记行不进简版（注入噪声）。"""
    p = _session(tmp_path / "s.jsonl", 2, with_noise=True)
    text = tr.scan_session(p, verbose=True)
    assert "Custom model service error" not in text, (
        "合成回合是渠道故障注入，不是组员说的话"
    )
    assert "attachment" not in text
    assert "runtime-config" not in text
    assert "[thinking]" in text, "思考折叠为单行"
    assert "[tool: Bash," in text, "工具调用折叠为单行（名称+尺寸）"
    assert "第 1 轮正式回复" in text and "第 2 轮正式回复" in text


def test_tool_line_marks_size(tmp_path):
    """工具调用单行带尺寸——「这步搬了多少字节」是判读疑点的原料。"""
    p = _session(tmp_path / "s.jsonl", 1)
    text = tr.scan_session(p, verbose=True)
    assert "[tool: Bash, " in text and "B]" in text


def test_long_text_truncated_with_actual_size(tmp_path):
    """超长正文截断并标注**实际**大小（「截断了」必须可见，否则会被当全文读）。"""
    long_text = "长" * 500
    p = tmp_path / "long.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_user(long_text)),
                json.dumps(_assistant([{"type": "text", "text": long_text}])),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text = tr.scan_session(p)
    assert "…（共 " in text and "字）" in text, "截断处必须标注实际字数"
    assert len(text) < len(long_text) * 2


def test_tool_result_not_counted_as_round(tmp_path):
    """工具结果回灌不算新回合（否则回合数虚高一倍，三态判定全错）。"""
    p = _session(tmp_path / "s.jsonl", 3)
    rounds = tr.build_rounds(p)
    assert len(rounds) == 3
    assert all("工具结果" not in r.user for r in rounds)


def test_verbose_body_is_wider_than_index_slice(tmp_path):
    """``--verbose`` 正文比索引行宽——同宽就成了索引的复读机，读正文也没多得信息。"""
    p = tmp_path / "wide.jsonl"
    body = "甲" * 300
    p.write_text(
        "\n".join(
            [
                json.dumps(_user(body)),
                json.dumps(_assistant([{"type": "text", "text": body}])),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text = tr.scan_session(p, verbose=True)
    body_line = next(
        ln for ln in text.splitlines() if ln.strip().startswith("user: 甲")
    )
    assert len(body_line) > tr.SLICE_CHARS + 50, (
        "正文行必须比索引宽（索引 80、正文 400），否则 verbose 无增益"
    )


# ---------------------------------------------------------------------------
# ④ 行锚与 fetch（可复现）
# ---------------------------------------------------------------------------
def test_scan_index_carries_anchor_and_fetch_returns_same_span(tmp_path):
    """finding 的 ``--repro`` 引行锚：scan 出的锚拿去 fetch，取回的是**同一段**。"""
    root = _project(tmp_path / "projects", "proj")
    p = _session(root / "anchor.jsonl", 3)

    text = tr.scan_session(p)
    anchor = (
        next(ln for ln in text.splitlines() if "[2]" in ln).split("anchor=")[1].strip()
    )
    sid, a, b = tr.parse_anchor(anchor)
    assert sid == "anchor" and a <= b

    fetched = tr.fetch_anchor(anchor, root=tmp_path / "projects")
    assert "第 2 轮任务书正文" in fetched, "按锚取回的原段必须含该回合的 user 正文"
    assert "第 1 轮" not in fetched and "第 3 轮" not in fetched, (
        "锚是**有界**取段：不许把相邻回合也端出来（否则锚形同虚设）"
    )


def test_fetch_anchor_reports_missing_session(tmp_path):
    """锚指向不存在的会话 → 明确报错（不许静默取错文件）。"""
    out = tr.fetch_anchor("ghost.jsonl:L1-L5", root=tmp_path / "projects")
    assert "找不到会话" in out


def test_fetch_truncates_huge_record_with_size(tmp_path):
    """有界返回：超长记录截断并标注原大小（零盲读、有界返回）。"""
    root = _project(tmp_path / "projects", "proj")
    p = root / "big.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_user("开始")),
                json.dumps(
                    _assistant(
                        [{"type": "text", "text": "巨" * (tr.FETCH_SLICE_CHARS * 3)}]
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tr.fetch_anchor("big.jsonl:L1-L2", root=tmp_path / "projects")
    assert "截断，原 " in out and "字）" in out


def test_parse_anchor_rejects_bad_shape():
    for bad in ("nope", "sid.jsonl:L5-L1", "sid.jsonl:1-2", "abc.txt:L1-L2"):
        try:
            tr.parse_anchor(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} 应被拒")


def test_fetch_grep_returns_line_and_round_anchor(tmp_path):
    root = _project(tmp_path / "projects", "proj")
    p = _session(root / "g.jsonl", 3)
    out = tr.fetch_grep("g", "第 2 轮任务书正文", root=tmp_path / "projects")
    assert "g.jsonl" in out and "L" in out
    assert "无匹配" not in out
    assert "无匹配" in tr.fetch_grep("g", "绝不存在的内容", root=tmp_path / "projects")


# ---------------------------------------------------------------------------
# ⑤ 缓存清理
# ---------------------------------------------------------------------------
def test_prune_cache_keeps_latest(tmp_path, monkeypatch):
    """缓存区保留最近 N 期自动清理（缓存不是资产，但也不能无限长）。"""
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path / "cache")
    tr.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(7):
        p = tr.CACHE_DIR / f"d{i}.txt"
        p.write_text("x", encoding="utf-8")
        import os

        os.utime(p, (1_800_000_000 + i, 1_800_000_000 + i))

    dropped = tr.prune_cache(keep=3)
    assert dropped == 4
    left = sorted(p.name for p in tr.CACHE_DIR.glob("*.txt"))
    assert left == ["d4.txt", "d5.txt", "d6.txt"], "保留的是最新三期"


# ---------------------------------------------------------------------------
# ⑥ CLI 全链：list → scan → fetch（success_criteria「一次真实巡检全链」）
# ---------------------------------------------------------------------------
def _cli_env(tmp_path, monkeypatch):
    """把工具链的三处落盘面都指到 tmp：projects 根 / 游标 / 缓存。"""
    from pysci.skills.inspector.tools import inspect as cli

    root = tmp_path / "projects"
    cursor = tmp_path / "cursor.json"
    monkeypatch.setattr(tr, "PROJECTS_ROOT", root)
    monkeypatch.setattr(tr, "CURSOR_PATH", cursor)
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path / "cache")
    return cli, root, cursor


def test_cli_chain_list_scan_fetch(tmp_path, monkeypatch, capsys):
    """一次真实巡检全链：list 选靶 → scan 拿锚 → fetch 按锚取回原文。"""
    cli, root, cursor = _cli_env(tmp_path, monkeypatch)
    _session(
        _project(root, "D--x-orchestration-pods-devops") / "chain.jsonl", 6, hour=9
    )

    rc = cli.main(["transcript", "list"])
    out = capsys.readouterr().out
    assert rc == 0 and "chain" in out, "list 要列出可检会话"

    rc = cli.main(["transcript", "scan", "chain", "--verbose"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "anchor=" in out, "scan 必须给出行锚（下一步 fetch 的入口）"
    anchor = (
        next(ln for ln in out.splitlines() if "anchor=" in ln)
        .split("anchor=")[1]
        .strip()
    )

    rc = cli.main(["transcript", "fetch", "--anchor", anchor])
    out = capsys.readouterr().out
    assert rc == 0 and "任务书正文" in out, "按 scan 给的锚能取回原文=可复现"

    # scan 已把该会话记成已检查 → 下轮不再顶出来（三态游标真的接上了）
    assert tr.load_cursor(cursor).get("chain"), "scan 必须写游标，否则每轮重复检同一个"
    rc = cli.main(["transcript", "list"])
    assert "chain" not in capsys.readouterr().out


def test_cli_fetch_requires_a_selector(tmp_path, monkeypatch, capsys):
    """fetch 不给选择器 → rc=2 且点名两种合法用法（不猜、不静默取空）。"""
    cli, _root, _cursor = _cli_env(tmp_path, monkeypatch)
    rc = cli.main(["transcript", "fetch"])
    out = capsys.readouterr().out
    assert rc == 2 and "--anchor" in out and "--grep" in out


def test_cli_scan_unknown_sid_reports_target_hint(tmp_path, monkeypatch, capsys):
    """找不到会话 → rc=2 并指回去 list（比静默空输出可行动）。"""
    cli, _root, _cursor = _cli_env(tmp_path, monkeypatch)
    rc = cli.main(["transcript", "scan", "nosuch"])
    out = capsys.readouterr().out
    assert rc == 2 and "transcript list" in out
