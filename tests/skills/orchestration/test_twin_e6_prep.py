"""E6-prep 阳性回归：twin 无记忆纪律、宪章只读、taste 开关、mirror 分流（backlog 20261012-005900-leader）。

任务书第 6 条点名的四组阳性各对应一段：

1. **冷启动强制**：成员 ``cold_start`` 使 ``do_dispatch`` 在任何外呼之前拒绝非 new 会话；
2. **readonly_extra 拦截 PREFERENCE.md 写**：registry → ``build_env`` → ``PYSCI_READONLY``
   → **真 pod-guard（node 直跑）** exit 2，同 pod 内普通文件放行（对照组）；
3. **``taste_enabled=false`` 拒绝派发**：开关未置真时 ``orch taste`` 不调 do_dispatch；
   置真后走通且**每跳新 sid、session 恒为 new**（无记忆的可测面）；
4. **mirror journal 分流落点对**：①只写决策日志、②只写强制队列、复选两边都写，
   且队列读数进 ``orch status``（组长可见）。

另钉住脚手架在位与预算（pod 五件套 + 两 hook 接线 + agentsMdExcludes；PREFERENCE.md
草案标记 + ≤8192 B）——这些是「机械面先行、生效确认留用户」的实物凭据。
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.paths import PODS_ROOT, PROJECT_ROOT
from pysci.skills.orchestration.tools import (
    dispatch,
    mirror,
    orch,
    registry,
    runner,
    taste,
)
from pysci.skills.orchestration.tools import journal as journal_tool
from pysci.skills.orchestration.tools import ledger as ledger_tool
from pysci.skills.orchestration.tools import sessions as dispatch_sessions
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.registry import Registry

from .test_pod_guard_blocks import NODE_REQUIRED, run_guard

#: 技能部署 manifest（唯一真本 → 部署目标映射）
MANIFEST_PATH = PROJECT_ROOT / "orchestration" / "skills" / "manifest.toml"

PREF_TEXT = "# 判断宪章（测试夹具）\n\n状态：**草案 v1，待用户确认生效**\n\n- P-01 客观优质 > 审美\n"

CASE_TEXT = """## 问题
文档超预算时该怎么办？

## 选项 A
把细节移到同目录 references/，SKILL.md 只留决策与调用面。

## 选项 B
放宽预算到 16 KB，一次装下全部内容。
"""

JUDGE_JSON = '{"choice": "A", "reason": "P-10 不新增抽象层级", "confidence": 0.7}'


@pytest.fixture
def twin_face(tmp_path, monkeypatch):
    """隔离 twin 判断面：tmp registry（开关可翻）+ tmp pod（宪章）+ 留痕/队列/台账重定向。

    真实 state 的台账/建议/回复/会话池一并 patch 到 tmp：本夹具允许跑到 ``run_headless``
    之前的全部链路，绝不能往 ``orchestration/state/`` 落测试痕迹。
    """
    pod = tmp_path / "pods" / "twin"
    (pod / "inbox").mkdir(parents=True)
    (pod / "PREFERENCE.md").write_text(PREF_TEXT, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    reg_file = state / "registry.json"

    def write_reg(enabled: object) -> None:
        reg_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "exe": None,
                    "models": {"max": "", "flash": "", "flash_fallback": ""},
                    "taste_enabled": enabled,
                    "members": {
                        "twin": {
                            "pod": str(pod),
                            "model_tier": "flash",
                            "max_turns": 8,
                            "timeout_s": 900,
                            "mcp_config": ".qoder/mcp.json",
                            "cold_start": True,
                            "readonly_extra": ["PREFERENCE.md", "AGENTS.md"],
                        }
                    },
                    "checks": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    write_reg(False)
    # 假 exe：让 session="new" 那一支真的走到 run_headless（否则在 exe 解析处就返回，
    # 「闸没拦」就无从证明）
    exe = tmp_path / "fake-exe.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("PYSCI_ORCH_EXE", str(exe))
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    # Registry.save 的落点绑在真实 registry.json——no-op 防测试写配置面
    monkeypatch.setattr(registry.Registry, "save", lambda self: None)
    case = tmp_path / "cases" / "budget.md"
    case.parent.mkdir()
    case.write_text(CASE_TEXT, encoding="utf-8")
    monkeypatch.setattr(taste, "TASTE_DIR", tmp_path / "evals" / "taste")
    monkeypatch.setattr(mirror, "MIRROR_QUEUE_PATH", state / "mirror-queue.jsonl")
    monkeypatch.setattr(
        journal_tool, "JOURNAL_PATH", state / "preference-journal.jsonl"
    )
    monkeypatch.setattr(ledger_tool, "LEDGER_PATH", state / "ledger.jsonl")
    monkeypatch.setattr(dispatch, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", state / "taskbook-lint.json")
    monkeypatch.setattr(
        dispatch_sessions, "pod_sessions_dir", lambda pod_dir: state / "sessions"
    )
    return SimpleNamespace(
        tmp=tmp_path,
        pod=pod,
        state=state,
        reg_file=reg_file,
        case=case,
        write_reg=write_reg,
    )


def _fake_judge_hop(calls: list[dict]):
    """假派发：记录 kwargs，每跳给不同 sid，交付正文为判断 JSON。"""

    def fake(member_id: str, **kw):
        calls.append({"member": member_id, **kw})
        return DispatchOutcome(
            code=0,
            kind="result",
            member=member_id,
            sid=f"sid-{len(calls)}",
            body=JUDGE_JSON,
        )

    return fake


# ===========================================================================
#  1. 冷启动强制（任务书 6-1）
# ===========================================================================


def test_cold_start_rejects_resume_before_any_launch(twin_face, monkeypatch, capsys):
    """cold_start 成员传 ``session="latest"`` → code=2 且一次都不外呼（无记忆落代码）。"""
    launched: list = []
    monkeypatch.setattr(
        dispatch,
        "run_headless",
        lambda *a, **k: launched.append(a) or (0, "{}", ""),
    )
    out = dispatch.do_dispatch("twin", text="判断一案", session="latest", quiet=False)
    assert out.code == 2
    assert "cold_start" in out.error
    assert not launched, "硬闸必须在外呼之前"
    assert "冷启动纪律" in capsys.readouterr().out


def test_cold_start_still_allows_new_session(twin_face, monkeypatch):
    """同一成员 ``session="new"`` 不被本闸拦（闸只禁复用，不禁新会话）。"""
    launched: list = []

    def fake(*a, **k):
        launched.append(a)
        return 1, '{"result": "", "is_error": true}', "boom"

    monkeypatch.setattr(dispatch, "run_headless", fake)
    out = dispatch.do_dispatch("twin", text="判断一案", session="new")
    assert "cold_start" not in out.error
    assert launched, "session=new 应当放行到外呼层"


# ===========================================================================
#  2. readonly_extra → 真 pod-guard 拦截宪章写（任务书 6-2）
# ===========================================================================


def test_build_env_injects_readonly_extra(twin_face):
    """registry 的 readonly_extra 经 build_env 并入 PYSCI_READONLY（链路起点）。"""
    env = runner.build_env(Registry.load().member("twin"), [])
    readonly = env["PYSCI_READONLY"].split(";")
    assert "PREFERENCE.md" in readonly
    assert "AGENTS.md" in readonly


@NODE_REQUIRED
def test_pod_guard_blocks_preference_write(twin_face):
    """真 guard：pod 内写 PREFERENCE.md → exit 2「命中只读层」；同 pod 写普通文件放行。"""
    readonly = runner.build_env(Registry.load().member("twin"), [])["PYSCI_READONLY"]
    rc, err = run_guard(
        file_path=str(twin_face.pod / "PREFERENCE.md"),
        pod=str(twin_face.pod),
        project_root=str(PROJECT_ROOT),
        readonly=readonly.split(";"),
    )
    assert rc == 2, err
    assert "只读层" in err and "PREFERENCE.md" in err
    rc2, err2 = run_guard(
        file_path=str(twin_face.pod / "notes.md"),
        pod=str(twin_face.pod),
        project_root=str(PROJECT_ROOT),
        readonly=readonly.split(";"),
    )
    assert rc2 == 0, err2


# ===========================================================================
#  3. taste 开关与判断通道（任务书 6-3）
# ===========================================================================


def test_taste_enabled_only_accepts_true(twin_face):
    """开关严格判真：非布尔/假值一律 False（宁可不开，不猜用户意思）。"""
    for raw in (False, None, "true", 1, {}):
        twin_face.write_reg(raw)
        assert Registry.load().taste_enabled is False, f"值 {raw!r} 被判真"
    twin_face.write_reg(True)
    assert Registry.load().taste_enabled is True


def test_taste_disabled_refuses_dispatch(twin_face, monkeypatch, capsys):
    """未生效：judge 返回 2、打印拒绝语与 [NEXT]，一次都不派发。"""
    calls: list[dict] = []
    monkeypatch.setattr(taste, "do_dispatch", _fake_judge_hop(calls))
    code, rec = taste.judge(twin_face.case)
    assert code == 2 and rec == {}
    assert calls == []
    out = capsys.readouterr().out
    assert taste.DISABLED_HINT in out
    assert "[NEXT]" in out


def test_cmd_taste_dry_runs_while_disabled(twin_face, capsys):
    """``--dry`` 是查看注入内容的安全面：未生效也能跑（零外呼、零成本）。"""
    rc = taste.cmd_taste(
        argparse.Namespace(
            case_file=str(twin_face.case), dry=True, member="twin", model_tier=None
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "判断宪章" in out and "taste_enabled=False" in out
    assert "## 选项 B" in out, "案卷两选项都要注入"


def test_taste_enabled_dispatches_cold_start_each_hop(twin_face, monkeypatch):
    """生效后：session 恒 new、每跳 sid 不同、结构化选择落 ``evals/taste/<case>.json``。"""
    twin_face.write_reg(True)
    calls: list[dict] = []
    monkeypatch.setattr(taste, "do_dispatch", _fake_judge_hop(calls))
    code1, rec1 = taste.judge(twin_face.case, quiet=True)
    code2, rec2 = taste.judge(twin_face.case, quiet=True)
    assert (code1, code2) == (0, 0)
    assert [c["session"] for c in calls] == ["new", "new"]
    assert all(c["eval_hop"] is True for c in calls), "判断跳不该进生产台账"
    assert rec1["sid"] != rec2["sid"], "同一会话被复用＝记忆泄漏"
    assert rec1["choice"] == "A" and rec1["confidence"] == 0.7
    p = Path(taste.TASTE_DIR) / "budget.json"
    assert p.is_file()
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["case_id"] == "budget" and on_disk["member"] == "twin"
    assert on_disk["dossier_hash"]


def test_taste_missing_charter_refuses(twin_face, monkeypatch):
    """宪章正文缺失 → 不派发（无依据不判断，不凭空以用户名义表态）。"""
    twin_face.write_reg(True)
    (twin_face.pod / "PREFERENCE.md").unlink()
    calls: list[dict] = []
    monkeypatch.setattr(taste, "do_dispatch", _fake_judge_hop(calls))
    code, _ = taste.judge(twin_face.case, quiet=True)
    assert code == 2 and calls == []


def test_parse_judgement_clamps_and_rejects():
    """选择钳制：非 A/B 置空、confidence 越界钳到 0–1、读不出即 parsed=False。"""
    j = taste.parse_judgement('{"choice": "视情况", "confidence": 9}')
    assert j["parsed"] and j["choice"] == "" and j["confidence"] == 1.0
    assert taste.parse_judgement("没有 JSON")["parsed"] is False


# ===========================================================================
#  4. mirror 分流与组长可见性（任务书 6-4/6-5）
# ===========================================================================


def test_journal_pref_only_writes_log(twin_face):
    """①偏好陈述 → 决策日志 kind=preference，队列不动（不要求行动）。"""
    mirror.route("我讨厌为了格式再建一层抽象", [mirror.CHOICE_PREFERENCE])
    recs = journal_tool.read_records()
    assert len(recs) == 1 and recs[0]["kind"] == "preference"
    assert recs[0]["source"] == "mirror"
    assert not (twin_face.state / "mirror-queue.jsonl").exists()


def test_journal_directive_only_writes_queue(twin_face):
    """②行动指令 → 强制队列 status=pending，日志不动。"""
    mirror.route("把 pod-guard 的白名单提示改成人话", [mirror.CHOICE_DIRECTIVE])
    assert journal_tool.read_records() == []
    pend = mirror.pending_directives()
    assert len(pend) == 1 and pend[0]["status"] == "pending"
    assert pend[0]["source"] == "mirror"


def test_journal_done_entries_leave_pending_view(twin_face):
    """组长处理过（status=done）的指令不再算待派发——否则队列永不见底。"""
    mirror.append_directive("第一条")
    mirror.append_directive("第二条")
    q = twin_face.state / "mirror-queue.jsonl"
    lines = q.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(
        {**json.loads(lines[0]), "status": "done"}, ensure_ascii=False
    )
    q.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pend = mirror.pending_directives()
    assert len(pend) == 1 and pend[0]["text"] == "第二条"


def test_journal_both_selects_writes_both_sides(twin_face):
    """复选 1,2 → 同一句话两侧都落（偏好即指令的口述常见形态）。"""
    mirror.route("以后所有案卷都要带成本读数", [1, 2])
    assert len(journal_tool.read_records()) == 1
    assert len(mirror.read_queue()) == 1


def test_journal_cli_noninteractive_requires_select(twin_face, capsys):
    """非 tty 且没给 --select：不猜、不阻塞、不写任何文件。"""
    rc = mirror.main(["journal", "随手一句口述"])
    assert rc == 2
    assert "--select" in capsys.readouterr().out
    assert mirror.read_queue() == [] and journal_tool.read_records() == []


def test_journal_menu_eof_without_traceback(twin_face, monkeypatch, capsys):
    """tty 但 stdin 无内容（agent 会话常态）：干净拒绝，不抛 EOFError traceback。

    上面那条非交互用例走不到这一支——它的 stdin 不是 tty，在 isatty 处就返回了。
    真跑 ``pysci-mirror journal`` 时 stdin 常是「伪 tty + 空管道」，两条路都得封。
    """
    monkeypatch.setattr(mirror.sys.stdin, "isatty", lambda: True)

    def _eof(*_a):
        raise EOFError

    monkeypatch.setattr(mirror, "input", _eof, raising=False)
    rc = mirror.main(["journal", "随手一句口述"])
    assert rc == 2
    assert "--select" in capsys.readouterr().out
    assert mirror.read_queue() == [] and journal_tool.read_records() == []


def test_cmd_status_panorama_covers_spec_items(twin_face, capsys):
    """status 全景覆盖任务书四项：待盲审/宪章/开关/队列水位（§10 注册纪律）。"""
    mirror.route("指令：给 mirror 加 review 面", [2])
    rc = mirror.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0
    for key in ("待盲审", "宪章", "taste_enabled", "行动指令队列待处理"):
        assert key in out, f"概览项缺席：{key}"
    assert "草案 v1" in out
    assert len(mirror.STATUS_ITEMS) >= 4


def test_orch_status_shows_mirror_queue(iso_state, iso_registry, monkeypatch, capsys):
    """组长侧可见性：mirror 队列里的未办行动指令出现在 ``orch status``，已办的不再刷屏。"""
    reg_file, _runtime, _backup = iso_registry
    reg_file.write_text(
        json.dumps(
            {"version": 1, "exe": None, "models": {}, "members": {}, "checks": {}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    q = iso_state / "mirror-queue.jsonl"
    monkeypatch.setattr(mirror, "MIRROR_QUEUE_PATH", q)
    mirror.append_directive("给 twin 加盲审面")
    mirror.append_directive("已办的那条")
    lines = q.read_text(encoding="utf-8").splitlines()
    lines[1] = json.dumps(
        {**json.loads(lines[1]), "status": "done"}, ensure_ascii=False
    )
    q.write_text("\n".join(lines) + "\n", encoding="utf-8")
    orch.cmd_status(argparse.Namespace())
    out = capsys.readouterr().out
    assert "mirror 行动指令待派发=1" in out
    assert "给 twin 加盲审面" in out
    assert "已办的那条" not in out


# ===========================================================================
#  5. 脚手架在位与预算（任务书 1/2/3 的实物凭据）
# ===========================================================================


def test_twin_pod_scaffold_present():
    """pod 五件套 + 宪章在位；settings 两 hook 接线 + agentsMdExcludes（十 pod 先例）。"""
    pod = PODS_ROOT / "twin"
    for rel in (
        ".gitignore",
        ".qoder/settings.json",
        ".qoder/mcp.json",
        ".qoder/rules/charter.md",
        "AGENTS.md",
        "PREFERENCE.md",
    ):
        assert (pod / rel).is_file(), f"缺 {rel}"
    s = json.loads((pod / ".qoder/settings.json").read_text(encoding="utf-8"))
    assert s["agentsMdExcludes"] == ["**/leader-only.md"]
    names = {
        h["name"]
        for section in ("Stop", "PreToolUse")
        for grp in s["hooks"][section]
        for h in grp["hooks"]
    }
    assert names == {"delivery-gate", "pod-guard"}


def test_preference_effective_marked_and_within_budget():
    """PREFERENCE.md：v2 生效态标注（用户 2026-10-13 确认）+ 每文件 ≤8192 B。

    不断言 LF：本机 `core.autocrlf=true`，checkout 出来必然是 CRLF——预算口径本来就是
    **落盘字节（CRLF 计税）**，所以量 `len(raw)` 即正确判据，行尾形态不是。
    （原草案态断言「草案 v1/待用户确认生效」随用户确认转正而退役。）
    """
    p = PODS_ROOT / "twin" / "PREFERENCE.md"
    raw = p.read_bytes()
    text = raw.decode("utf-8")
    assert "v2 生效" in text and "taste_enabled=true" in text
    assert len(raw) <= 8192, f"宪章超预算：{len(raw)} B"


def test_twin_charter_and_agents_discipline():
    """charter 含无记忆/不投客观票/终裁不给平局；AGENTS.md 是固定最小占位。"""
    charter = (PODS_ROOT / "twin" / ".qoder" / "rules" / "charter.md").read_text(
        encoding="utf-8"
    )
    for key in ("冷启动", "--resume", "平局", "客观质量", "置信度", "不得积累笔记"):
        assert key in charter, f"charter 缺纪律条款：{key}"
    agents = (PODS_ROOT / "twin" / "AGENTS.md").read_text(encoding="utf-8")
    assert len(agents.encode("utf-8")) < 2048, "AGENTS.md 开始积累了"
    assert "禁止填写" in agents


def test_registry_ships_twin_disabled_by_default():
    """出厂默认关（代码而非配置当前值）+ 开着必须有用户批准单 + twin 条目无 extra_dirs。

    原来这里断言 ``registry.json`` 的当前值为 False，把「运行面读数」当成「出厂默认」
    锁死——于是用户 2026-10-12 批准上线（批准单 20261012-164601-leader 补充条款原话
    「先上线了再说」）之后，每一次合法运行都打破回归。真正要守的两件事分开钉：

    - **默认**：:data:`registry.DEFAULT_TASTE_ENABLED` 必须 False（代码不替用户按开关）；
    - **授权留痕**：配置里若为 true，批准单必须存在且点名该开关——悄悄开启仍然被拦。
    """
    data = json.loads(
        (PROJECT_ROOT / "orchestration" / "state" / "registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry.DEFAULT_TASTE_ENABLED is False
    if data.get("taste_enabled") is True:
        approval = (
            PROJECT_ROOT
            / "orchestration"
            / "state"
            / "suggestions"
            / "approved"
            / "20261012-164601-leader.md"
        )
        assert approval.is_file(), "taste_enabled=true 却找不到授权批准单"
        assert "taste_enabled" in approval.read_text(encoding="utf-8")
    twin = data["members"]["twin"]
    assert twin["cold_start"] is True
    assert "PREFERENCE.md" in twin["readonly_extra"]
    assert not twin.get("extra_dirs"), "extra_dirs 即写白名单，给了就不再只读"


def test_taste_on_without_approval_is_caught(monkeypatch, tmp_path):
    """上一条的留痕分支不能空转：造一棵「taste_enabled=true 但没有批准单」的假树，必须被咬。

    只改本模块的 ``PROJECT_ROOT`` 全局——被测函数在函数体内读它，不碰真仓库任何面。
    """
    state = tmp_path / "orchestration" / "state"
    state.mkdir(parents=True)
    (state / "registry.json").write_text(
        json.dumps(
            {
                "taste_enabled": True,
                "members": {
                    "twin": {"cold_start": True, "readonly_extra": ["PREFERENCE.md"]}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "PROJECT_ROOT", tmp_path)

    with pytest.raises(AssertionError, match="授权批准单"):
        test_registry_ships_twin_disabled_by_default()


def test_twin_intentionally_has_no_manifest_target():
    """原单规格第 2 条要求「manifest 追加 twin 部署目标」——本用例把**不适用**的裁定钉住。

    twin 无领域技能真本：判断输入只有「宪章 + 案卷」，由 ``orch taste`` 直接注入；给它挂
    任何领域技能都会把绘图/写作方法论混进审美判断。而 manifest 的每个 target 必须指向存在
    的 ``source`` 目录，凭空造一个 ``orchestration/skills/twin/`` 就是未被要求的文件。
    ``reviewer`` 是同形先例（同样不在 targets 里）。若将来真要给 twin 挂技能，改这条测试
    并附理由，别让「规格里写过」变成默认动作。
    """
    data = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    targets = [t for s in data["skills"] for t in s.get("targets", [])]
    assert not any("pods/twin" in t.replace("\\", "/") for t in targets)
    assert (PODS_ROOT / "twin" / ".qoder" / "skills").exists() is False
    # 反向护栏：manifest 里每条 target 的父目录都必须是已存在 pod 的 .qoder/skills 或项目根
    for t in targets:
        rel = t.replace("\\", "/")
        if rel == ".qoder/skills":
            continue
        pod_name = rel.split("/")[2]
        assert (PODS_ROOT / pod_name).is_dir(), f"manifest 指向不存在的 pod：{t}"
