"""BYOK 渠道级故障识别与全局停止（backlog 20261012-194500-leader）。

2026-10-12 实测：渠道服务面故障时 CLI 不报 credit 文案，而是合成一条 assistant 回合说
"Custom model service error, please retry later or switch to another model."
（``model="<synthetic>"``、``stop_reason=stop_sequence`` 而 ``stop_sequence`` 为空）。
现行 quota 模式表零命中 → 同渠道自动重试再烧一跳 → 条目带伤推进 → 18:35-19:38 连烧四单。
本组用例钉住修复面的识别与处置两段：**三特征合取识别 → 不换档时不重试、换档时只一次**；
全局停止与不连坐见 ``test_drain_channel_error_stops_globally``。额度路径的既有用例在
``test_quota_awareness.py`` / ``test_flash_fallback.py``，watch 面在 ``test_watch.py``。
"""

from __future__ import annotations

import json
from argparse import Namespace

from pysci.skills.orchestration.tools import dispatch, drain, orch, sessions
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.runner import Envelope

from .conftest import (
    CHANNEL_TEXT,
    QUOTA_TEXT,
    REAL_MODEL_TURN,
    append_turn,
    cmd_model,
    envelope,
    fake_headless,
    read_backlog,
    seed_backlog,
    set_models,
    sid_from_argv,
)

BUILTIN = "Qwen3.8-Flash"  # 主档渠道
FB = "byok-flash-id"  # 备用渠道（<tier>_fallback 的占位 modelID）
OK_BODY = "<result>完工，commit fff9999 已合并。</result>"


def _broken_channel(state):
    """响应器：每次派生都往该会话档尾追加一条**合成故障轮**，返回渠道故障 envelope。

    形状支读的是会话档，而 sid 由 do_dispatch 自己生成，故从 argv 把它接回来再落盘——
    用例量的是真实读路，不是把探针 patch 掉的自证。
    """

    def responder(n, cmd):
        append_turn(
            state,
            sid_from_argv(cmd),
            model=sessions.SYNTHETIC_MODEL,
            text=CHANNEL_TEXT,
            stop_reason="stop_sequence",
            stop_sequence="",
        )
        return envelope(CHANNEL_TEXT)

    return responder


def _member():
    """夹具成员（会话档落在 iso_dispatch/sessions，路径由 patch 过的 pod_sessions_dir 定）。"""
    from pysci.skills.orchestration.tools.registry import Registry

    return Registry.load().member("quotamember")


def _done(state, name: str, **over) -> None:
    """落一份 ``devops-runs/<name>.done``（status/watch 的收尾读数面）。"""
    d = state / "devops-runs"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"count": 0, "elapsed_s": 1.0, "items": [], **over}
    (d / f"{name}.done").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# ① 形状探针：sessions.last_assistant_record / channel_failure_turn
# ---------------------------------------------------------------------------
def test_shape_probe_reads_synthetic_failure_turn(iso_dispatch):
    append_turn(
        iso_dispatch,
        "s1",
        model=sessions.SYNTHETIC_MODEL,
        text=CHANNEL_TEXT,
    )
    assert (
        sessions.channel_failure_turn(
            _member(), "s1", text_re=dispatch.CHANNEL_ERROR_RE
        )
        is True
    )


def test_shape_probe_false_for_real_model_turn(iso_dispatch):
    """真实模型回合说了那句固定句 → **不是**故障（model 是渠道名，不是 <synthetic>）。

    文案单特征误伤的代价是整条队列停下，故合取的另一支必须由代码量出来。
    """
    append_turn(iso_dispatch, "s2", model=REAL_MODEL_TURN, text=CHANNEL_TEXT)
    assert (
        sessions.channel_failure_turn(
            _member(), "s2", text_re=dispatch.CHANNEL_ERROR_RE
        )
        is False
    )


def test_shape_probe_false_when_stop_sequence_not_empty(iso_dispatch):
    """``stop_sequence`` 有值 = 自称停在真实序列上，与故障形状（空序列）不同。"""
    append_turn(
        iso_dispatch,
        "s3",
        model=sessions.SYNTHETIC_MODEL,
        text=CHANNEL_TEXT,
        stop_sequence="\n</result>",
    )
    assert (
        sessions.channel_failure_turn(
            _member(), "s3", text_re=dispatch.CHANNEL_ERROR_RE
        )
        is False
    )


def test_shape_probe_none_when_session_absent(iso_dispatch):
    """会话档读不出 → None（判据缺席），调用方自行决定采信文案与否，探针不猜。"""
    assert (
        sessions.channel_failure_turn(
            _member(), "ghost", text_re=dispatch.CHANNEL_ERROR_RE
        )
        is None
    )


def test_shape_probe_skips_bookkeeping_lines_after_turn(iso_dispatch):
    """故障轮之后还有簿记行（type 非 assistant）时仍命中末轮 assistant 记录。"""
    sid = "s4"
    append_turn(iso_dispatch, sid, model=sessions.SYNTHETIC_MODEL, text=CHANNEL_TEXT)
    p = iso_dispatch / "sessions" / f"{sid}.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "active-leaf", "sid": sid}) + "\n")
    assert (
        sessions.channel_failure_turn(_member(), sid, text_re=dispatch.CHANNEL_ERROR_RE)
        is True
    )


def test_last_assistant_record_survives_window_truncation(iso_dispatch):
    """档大于读取窗口时首行是半行——必须跳过它，而不是当「档里没有 assistant 记录」。

    真实会话档可达数 MB（本 pod 实测 6.3 M 字符），探针只读尾部窗口。窗口起点必然落在
    某条完整记录的中间，那条残片含 ``"assistant"`` 字样却不是合法 JSON；若它被当成记录
    解析（抛错）或被当成末轮（误判），判据就在长档上失效——故障轮总在档尾，故必须照常
    读出来。
    """
    sid = "s5"
    for i in range(400):  # 每条 ~2 KiB 的真实回合，把档推过窗口边界
        append_turn(
            iso_dispatch,
            sid,
            model=REAL_MODEL_TURN,
            text=f"第 {i} 轮正常输出：" + "汉" * 700,
            stop_reason="end_turn",
        )
    p = iso_dispatch / "sessions" / f"{sid}.jsonl"
    assert p.stat().st_size > sessions.TAIL_BYTES, "夹具没造出超窗档，测不到截断路径"
    append_turn(iso_dispatch, sid, model=sessions.SYNTHETIC_MODEL, text=CHANNEL_TEXT)
    msg = sessions.last_assistant_record(_member(), sid)
    assert msg is not None and msg.get("model") == sessions.SYNTHETIC_MODEL
    assert (
        sessions.channel_failure_turn(_member(), sid, text_re=dispatch.CHANNEL_ERROR_RE)
        is True
    )


# ---------------------------------------------------------------------------
# ② 检测面：三特征合取 + 与额度面的次序
# ---------------------------------------------------------------------------
def test_detect_channel_needs_text_and_shape(iso_dispatch):
    sid = "d1"
    append_turn(iso_dispatch, sid, model=sessions.SYNTHETIC_MODEL, text=CHANNEL_TEXT)
    env = Envelope(result=CHANNEL_TEXT, is_error=True, stop_reason="stop_sequence")
    assert (
        dispatch._detect_channel_exhaustion(env, "", "", member=_member(), sid=sid)
        is True
    )


def test_detect_channel_not_fooled_by_body_mention(iso_dispatch):
    """组员正文复述那句故障原文，而末轮是真实回合 → 不升级为系统性故障。"""
    sid = "d2"
    append_turn(iso_dispatch, sid, model=REAL_MODEL_TURN, text="任务完成，收尾报告")
    env = Envelope(result=f"复盘材料引用原文：{CHANNEL_TEXT}", is_error=True)
    assert (
        dispatch._detect_channel_exhaustion(env, "", "", member=_member(), sid=sid)
        is False
    )


def test_detect_channel_falls_back_to_text_when_unreadable(iso_dispatch):
    """会话档读不出时采信文案单特征：漏判的代价是队列再逐单空烧一轮（本单要修的缺陷）。"""
    env = Envelope(result=CHANNEL_TEXT, is_error=True)
    assert (
        dispatch._detect_channel_exhaustion(
            env, "", "", member=_member(), sid="no-such-session"
        )
        is True
    )


def test_detect_systemic_prefers_quota(iso_dispatch):
    """两类同现时取额度：处置相同，但「该充值」是更硬的成因读数。"""
    sid = "d3"
    append_turn(iso_dispatch, sid, model=sessions.SYNTHETIC_MODEL, text=CHANNEL_TEXT)
    env = Envelope(result=f"{CHANNEL_TEXT}｜{QUOTA_TEXT}", is_error=True)
    assert dispatch._detect_systemic(env, "", "", member=_member(), sid=sid) == (
        "quota_exhausted"
    )


def test_detect_systemic_empty_for_ordinary_failure(iso_dispatch):
    env = Envelope(result="internal boom", is_error=True)
    assert dispatch._detect_systemic(env, "", "", member=_member(), sid="d4") == ""


def test_channel_error_re_matches_measured_wording():
    """模式表吃实测原文（大小写与措辞逐字），改坏了就退回「识别不到」的老缺陷。"""
    assert dispatch.CHANNEL_ERROR_RE.search(CHANNEL_TEXT)
    assert dispatch.CHANNEL_ERROR_RE.search(
        "custom model service error, please retry later or switch to another model"
    )


def test_systemic_kinds_and_text_are_one_source():
    """kind 清单与文案同源：三处消费面（drain/bench/inspect 停止、watch、status）共读。"""
    assert dispatch.SYSTEMIC_KINDS == ("quota_exhausted", "channel_error")
    assert set(dispatch.SYSTEMIC_TEXT) == set(dispatch.SYSTEMIC_KINDS)
    assert "渠道" in dispatch.SYSTEMIC_TEXT["channel_error"][0]
    assert "channel_error" in dispatch.systemic_error_text("channel_error")


# ---------------------------------------------------------------------------
# ③ do_dispatch 集成：同席重试先自救 / 换席是后手 / 备用渠道仍坏 → channel_error
# ---------------------------------------------------------------------------
# 本组三条的语义在 2026-10-13 反转过一次：旧裁决（backlog 20261012-194500-leader）判
# channel_error 为首败即换席，实测发现它是**瞬态**的（Mimo 平台侧算力不足，一日 4 起），
# 同会话 resume 大概率即愈；而换席落到 DeepSeek 付费席——付费席降为后手（用户裁决
# 2026-10-13，backlog 20261013-143000-leader）。重试细则的单测在 test_same_seat_retry.py，
# 这里只钉「先自救、后换席」的次序骨架，且一律把次数闸压到 1 次以免用例真睡 45s×N。
def _one_same_seat_retry(monkeypatch, state):
    """把退避换成 no-op + 次数闸压到 1：用例测**次序**，不测退避时长（那是同席重试组的活）。"""
    monkeypatch.setattr(dispatch, "_sleep", lambda s: None)
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data.setdefault("model_fallback", {})["same_seat_retry"] = {
        "channel_error": 1,
        "backoff_s": 0,
        "budget_s": 0,
    }
    reg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_do_dispatch_channel_error_retries_same_seat_first(iso_dispatch, monkeypatch):
    """渠道故障**先**同席 resume 重试一次（席位不动），仍败才轮到换席通路。"""
    _one_same_seat_retry(monkeypatch, iso_dispatch)
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback="")
    calls = fake_headless(monkeypatch, _broken_channel(iso_dispatch))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "channel_error" and o.code == 2
    assert len(calls) == 2, "首跳 + 1 次同席重试；无下一席可换故止于此"
    assert o.same_seat_retries == 1, "重试次数必须在读数面上可见"
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert '"channel_error"' in line
    assert "渠道级故障" in line, "台账 error 面须分得开 quota/channel 两类（要求 3）"


def test_do_dispatch_channel_switches_to_fallback_once(iso_dispatch, monkeypatch):
    """同席重试没救回来才换档：第二席走 <tier>_fallback，成功即正常交付。"""
    _one_same_seat_retry(monkeypatch, iso_dispatch)
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)

    def responder(n, cmd):
        if n <= 2:  # 首跳 + 同席重试都坏 → 才轮到 fallback 席
            return _broken_channel(iso_dispatch)(n, cmd)
        return envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)

    calls = fake_headless(monkeypatch, responder)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)
    assert o.kind == "result" and o.code == 0
    assert len(calls) == 3
    assert cmd_model(calls[0]) == BUILTIN and cmd_model(calls[1]) == BUILTIN, (
        "自救的那两跳必须停在原席——偷换席位就等于没测到降为后手"
    )
    assert cmd_model(calls[2]) == FB, "第三跳才是真实换席"


def test_do_dispatch_channel_fallback_also_broken(iso_dispatch, monkeypatch):
    """自救 + 换席都救不回 → channel_error（drain 全局停止语义），不再三跳。"""
    _one_same_seat_retry(monkeypatch, iso_dispatch)
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)
    calls = fake_headless(monkeypatch, _broken_channel(iso_dispatch))
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)
    assert o.kind == "channel_error"
    assert len(calls) == 3, "首跳 + 1 次自救 + 1 次换席；换席重试仍只给一次"
    assert cmd_model(calls[2]) == FB


def test_do_dispatch_channel_report_distinguishes_kind(
    iso_dispatch, monkeypatch, capsys
):
    _one_same_seat_retry(monkeypatch, iso_dispatch)  # 不 patch 就真睡 45s 退避
    set_models(iso_dispatch, flash=BUILTIN, flash_fallback=FB)
    fake_headless(monkeypatch, _broken_channel(iso_dispatch))
    dispatch.do_dispatch("quotamember", text="出图", quiet=False)
    out = capsys.readouterr().out
    assert "渠道级失败（channel_error）" in out
    assert "同席" in out, "报告面必须点名已经自救过，否则看着像直接跳了付费席"
    assert "额度" not in out.split("[NEXT]")[0], "报告面别把渠道故障说成额度耗尽"
    assert "[NEXT]" in out and "registry models" in out


def test_channel_success_mentioning_text_not_reclassified(iso_dispatch, monkeypatch):
    """成功交付正文里出现该文案（复盘类任务）不得重分类——检测只在失败分支。"""
    body = f"<result>故障复盘完成，原文为 {CHANNEL_TEXT}，已钉住判据。</result>"
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(body, is_error=False, stop="end_turn", rc=0),
    )
    o = dispatch.do_dispatch(
        "quotamember", text="复盘渠道故障", quiet=True, no_checks=True
    )
    assert o.kind == "result" and o.code == 0
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# ④ drain 全局停止 + 不连坐；status 读数分列
# ---------------------------------------------------------------------------
def _channel_outcome():
    return DispatchOutcome(
        code=2,
        kind="channel_error",
        body="",
        error="渠道级故障：custom model service error",
    )


def test_drain_channel_error_stops_globally(iso_state, monkeypatch):
    seed_backlog(
        iso_state,
        [
            {"id": "c1", "status": "pending", "summary": "s1", "member": "devops"},
            {"id": "c2", "status": "pending", "summary": "s2", "member": "devops"},
        ],
    )
    calls = []

    def fake_dispatch(member, **kw):
        calls.append(kw.get("slug"))
        return _channel_outcome()

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)
    assert summary["stopped"] == "channel_error"
    assert "渠道级故障" in summary["error"] and "quota" not in summary["error"]
    assert calls == ["c1"], "c2 不得再被派发（渠道坏了不是任务书坏了）"
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["c1"]["status"] == "pending", "种子无罪复位，不标 needs_leader"
    assert "needs_leader_at" not in by_id["c1"]
    assert by_id["c2"]["status"] == "pending", "组包/剩余条目不连坐"
    assert not drain.LOCK_PATH.exists(), "锁已释放"
    done = json.loads(run_log.with_suffix(".done").read_text(encoding="utf-8"))
    assert done["stopped"] == "channel_error"
    text = run_log.read_text(encoding="utf-8")
    assert "[C] 全局停止" in text, "run 日志的行首标记要能一眼分清 [Q] 额度 / [C] 渠道"


def test_channel_error_not_counted_as_item_failure(iso_state):
    """系统性故障不进 FAILED_ITEM_STATUSES：否则「渠道坏了」会污染重复失败阈值立案。"""
    assert "channel_error" not in drain.FAILED_ITEM_STATUSES
    _done(
        iso_state,
        "20261012-040000.done",
        items=[{"id": "c", "status": "channel_error"}],
    )
    assert drain.failure_counts() == {}


def test_status_names_channel_stop(iso_state, monkeypatch, capsys):
    """orch status 的停止方式读数分列两类（要求 3）：处置动作本就不同。"""
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    _done(iso_state, "20261012-050000.done", stopped="channel_error", count=0)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "渠道级故障全局停止" in out
    assert "待充值" not in out, "渠道故障不该提示充值"


def test_status_names_quota_stop(iso_state, monkeypatch, capsys):
    monkeypatch.setattr(drain, "_pid_alive", lambda pid: False)
    _done(iso_state, "20261012-060000.done", stopped="quota_exhausted", count=0)
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "额度耗尽全局停止" in out and "待充值或换渠道" in out
