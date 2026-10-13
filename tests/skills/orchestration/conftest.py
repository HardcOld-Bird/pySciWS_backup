"""编排测试的共享夹具：把编排状态路径重定向到 tmp，隔离真实项目 state。

被测模块（drain/workflow/dispatch）在函数调用时读取各自的模块级 Path 常量，故
``monkeypatch.setattr`` 这些常量即可把 backlog/lock/runs/suggestions/replies 全部
落到 tmp_path，测试互不污染、不触碰 ``orchestration/state/``。
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from pysci.skills.orchestration.tools import (
    benchd,
    channels,
    dispatch,
    drain,
    failure_tags,
    inspectd,
    journal,
    jury,
    ledger,
    orch,
    registry,
    routing,
    settings,
    verdict,
    watch,
    workflow,
)
from pysci.skills.orchestration.tools import sessions as dispatch_sessions

#: 运行态文件名（与 registry.RUNTIME_NAME 同步；夹具直接落盘用，不 import 常量以免耦合）。
RUNTIME_FILE = "sessions-runtime.json"

#: 真实 ``Registry.save``——模块导入时捕获。默认夹具为防污染把它 no-op 掉，
#: 需要断言落盘（读-改-写合并语义）的用例用本引用还原。
REAL_REGISTRY_SAVE = registry.Registry.save


#: 路由时钟的测试基准：2026-10-14 是**普通周三的峰段内**（chinesecalendar 实测
#: ``get_holiday_detail(2026-10-14) == (False, None)``）→ 谷时隐性态**不生效**，
#: 全部既有 jury/preset 用例看到的席链与面板都还是 toml 那一行的预设。
#: 为什么必须 autouse 冻结（而不是各用例自己传 ``moment=``）：席链与仲裁面板现在
#: 由时钟驱动，夜里跑套件会看到 ``main`` 链变成 DeepSeek 席、面板第三席换族——
#: 两千多个用例的断言随墙上时钟漂，红得一地却一行代码都没坏。
#: 要测谷态就用 :func:`set_routing_clock` 或 :func:`valley_clock` 显式改时刻。
PEAK_MOMENT = datetime(2026, 10, 14, 10, 0, tzinfo=ZoneInfo(routing.CN_TZ_NAME))
#: 谷态基准：同一周的周六（``2026-10-17``）——周末全天谷，与节假日库无关，最稳的谷样本。
VALLEY_MOMENT = datetime(2026, 10, 17, 20, 0, tzinfo=ZoneInfo(routing.CN_TZ_NAME))


def set_routing_clock(monkeypatch, moment: datetime) -> datetime:
    """把路由时钟钉到给定时刻（返回该时刻，用例可直接断言）。"""
    monkeypatch.setattr(routing, "now_cn", lambda: moment)
    return moment


@pytest.fixture(autouse=True)
def no_real_backoff_sleep(monkeypatch):
    """退避/避让的真实等待在测试里一律换 no-op（同席重试 45s×10、避让 ≤300s）。

    抽象落点是 ``dispatch._sleep``（调度层唯一等待出口），patch 一处即覆盖两条新通路：
    channel_error 同席重试（backlog 20261013-143000-leader）与峰谷边界避让
    （backlog 20261013-144500-leader）。留痕面不随之消失——重试次数/耗时走台账字段
    ``same_seat_retries``/``same_seat_retry_ms``，避让走 ``hold_s``，两者都仍被断言。

    需要验**等待时长本身**的用例（test_same_seat_retry 的退避/预算组）自行再 patch 一次
    记录秒数——monkeypatch 后写覆盖前写，且都在用例作用域内回收，互不串味。不这么做
    每个碰到渠道故障的用例都得自己记着补 patch，漏一个就拖 45s（实测 timeout）。
    """
    monkeypatch.setattr(dispatch, "_sleep", lambda seconds: None)
    # 带宽延期的等待与「睡到最近到期」同理：真睡 600s/次没人跑得动，而睡被垫成
    # no-op 后时钟不动 → 「全队皆延期」那一支会**原地空转**（实测挂住整套）。
    # 故测试里默认让到期查询恒返 0（等价「已到期，无活可等」）；要验那一支的用例
    # （test_bandwidth_defer）自行覆盖本桩返回值的序列。
    from pysci.skills.orchestration.tools import workflow as _wf

    monkeypatch.setattr(drain, "_sleep", lambda seconds: None)
    monkeypatch.setattr(_wf, "next_defer_s", lambda **_kw: 0.0)


@pytest.fixture(autouse=True)
def freeze_routing_clock(monkeypatch, tmp_path):
    """默认冻结在峰段——隐性 DeepSeek 态是**时间驱动**的配置面覆盖，测试必须自己选时刻。

    顺带把切换留痕的两个落盘点进 ``tmp_path``（兜底隔离）：``iso_state``/``iso_dispatch``
    各自也会 patch，但两者都没挂的用例只要走到 ``do_dispatch``（→ ``note_switch``）就会真写
    仓库状态面——实测泄漏形态：``routing-last.json`` 留下一条冻结时点的假基线，之后第一个
    真实进程拿它当「上次读数」，往决策日志写一场**没发生过的峰谷跨越**。autouse 先于显式
    fixture 实例化，挂了 iso 的用例会被 iso 的路径再覆盖回自己的 state 目录，互不抢道。
    """
    set_routing_clock(monkeypatch, PEAK_MOMENT)
    monkeypatch.setattr(routing, "ROUTING_STATE_PATH", tmp_path / "routing-last.json")
    monkeypatch.setattr(journal, "JOURNAL_PATH", tmp_path / "preference-journal.jsonl")


@pytest.fixture
def valley_clock(monkeypatch):
    """把路由时钟搬到周六夜间（隐性态生效）的用例专用夹具。"""
    return set_routing_clock(monkeypatch, VALLEY_MOMENT)


# 2026-10-10 额度事故原文（会话 jsonl 实测）
QUOTA_TEXT = (
    "You've reached your credit usage limit. Please upgrade your "
    "subscription plan to get more resources. Report Issue (input /feedback)"
)

# 2026-10-12 渠道级故障固定句（BYOK 渠道服务面故障，会话 6f0cc8e5 档尾实测）：
# CLI 合成的那条 assistant 回合正文，逐字照抄实测形状。
CHANNEL_TEXT = (
    "Custom model service error, please retry later or switch to another model."
)

#: 真实模型回合的 model 读数（健康档实测渠道名）——形状判据靠它与 ``<synthetic>`` 区分。
REAL_MODEL_TURN = "qfmodel"

#: :func:`fake_headless` 派生瞬间回报给 ``on_spawn`` 的假 pid（第四面回收的测试锚）。
FAKE_HOP_PID = 424242


def seed_settings(
    base, preset: str = "mimo", *, inspector: bool = True
) -> pathlib.Path:
    """在 ``base`` 落一份最小 ``pysci_settings.toml`` 并返回其路径（预设面隔离用）。"""
    path = base / "pysci_settings.toml"
    flag = "true" if inspector else "false"
    path.write_text(
        f'[model]\npreset = "{preset}"\n\n[inspector]\nenabled = {flag}\n',
        encoding="utf-8",
    )
    return path


def write_preset(preset: str) -> pathlib.Path:
    """改**当前生效的隔离 toml**（``settings.SETTINGS_PATH`` 指向的那份）里的 preset。

    必须写指针指向的文件：多层夹具会把真源挪到不同 tmp 目录（iso_registry 再指一次），
    写错目录=读的还是旧值，「改一行换预设」的用例就测了个寂寞。
    """
    path = settings.SETTINGS_PATH
    path.write_text(f'[model]\npreset = "{preset}"\n', encoding="utf-8")
    return path


def write_routing(**kv) -> pathlib.Path:
    """在**当前生效的隔离 toml** 上落 ``[routing]`` 块，保留 ``[model]``/``[inspector]``。

    整份重建而不是往文件尾追加：追加写法二次调用就出现两个 ``[routing]`` 表，
    ``tomllib`` 直接抛错 → ``read_settings`` fail-open 返回 ``{}`` → 用例以为在测「配置无效」
    其实在测「上一行把文件写坏了」。值按 TOML 字面量给（``true`` / ``3600`` / ``["glm"]``）。
    """
    path = settings.SETTINGS_PATH
    data = settings.read_settings()
    inspector = data.get("inspector") or {"enabled": True}
    # 路由键已并入 `[model]`（用户 2026-10-13「模型类设置集中一处」）→ 既有用例写新位置；
    # 旧位置 `[routing]` 的兼容读由专门用例覆盖，别把兼容路径混进常规用例（那会让
    # 「回落告警」在几十个用例里反复出现，掩盖真正该看的那条）。
    model = {k: v for k, v in (data.get("model") or {}).items()}
    model.setdefault("preset", "mimo")
    model.update(kv)
    body = ["[model]"]
    for key, value in model.items():
        body.append(f"{key} = {_toml_value(value)}")
    body.append("")
    body.append("[inspector]")
    for key, value in inspector.items():
        body.append(f"{key} = {_toml_value(value)}")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _toml_value(value) -> str:
    """Python 值 → TOML 字面量（bool 必须小写，str 用双引号，list 内元素同规则）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return '"' + str(value).replace('"', '\\"') + '"'


def set_presets(state, **tables):
    """改写隔离 registry.json 的 ``presets`` 块（``qwen={...}`` / ``mimo={...}`` 形参）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    face = data.get("presets")
    if not isinstance(face, dict):
        face = {}
        data["presets"] = face
    for name, table in tables.items():
        face[name] = table
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def iso_registry(tmp_path, monkeypatch):
    """把 ``REGISTRY_PATH`` 指向 tmp，返回 (reg_file, runtime_file, backup_file)。

    配置面/运行态/迁移备份三份同目录，随 patch 一起隔离；``Registry.load`` 显式读模块级
    ``REGISTRY_PATH``，``save`` 写 ``self.path`` 及其父目录，故 patch 一处即两面生效。
    """
    reg_file = tmp_path / "registry.json"
    runtime_file = tmp_path / registry.RUNTIME_NAME
    backup_file = tmp_path / registry.BACKUP_NAME
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    # 预设真源隔离（backlog 20261012-225500-leader）：不 patch 则任何读预设的用例都读到
    # 开发者 data/pysci_settings.toml 的真实行——组长改一行 preset，测试跟着漂。
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "pysci_settings.toml")
    seed_settings(tmp_path)
    return reg_file, runtime_file, backup_file


@pytest.fixture
def iso_state(tmp_path, monkeypatch):
    """重定向编排状态路径到 ``tmp_path/state``，返回该 state 目录。

    ``exist_ok`` 而非裸 ``mkdir()``：与 :func:`iso_dispatch` 同锚 ``tmp_path/state``，
    而 dispatch 入口内嵌巡检拉起，用例须同时挂两者才把面隔离干净（漏挂 iso_state 的
    dispatch 用例真起过 ``_inspect-worker``——backlog 20261012-191512-devops）。
    """
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    # 隔离面根目录（backlog 20261012-234000-leader）：抢救判据要读 ``<root>/<name>``
    # 在不在、进不进去写 git——不 patch 就是拿开发者机器上真实的 worktree 台账判脏，
    # 测试结论随开发者当时开了几棵树而变（更坏：对真树打出抢救提交）。
    wt_root = tmp_path / "worktrees"
    wt_root.mkdir(exist_ok=True)
    monkeypatch.setattr(workflow, "WORKTREES_ROOT", wt_root)
    monkeypatch.setattr(workflow, "BACKLOG_PATH", state / "backlog.json")
    monkeypatch.setattr(workflow, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(workflow, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(workflow, "TIER_A_FILED_PATH", state / "tier-a-filed.json")
    monkeypatch.setattr(ledger, "LEDGER_PATH", state / "ledger" / "deliveries.jsonl")
    monkeypatch.setattr(failure_tags, "TAGS_PATH", state / "failure-tags.json")
    # E5b：每条判决（approve/reject/核验落地）都写决策日志，不 patch 就真写仓库里那份
    # 长期资产（git 跟踪——测试污染会直接变成脏跟踪面）。
    monkeypatch.setattr(journal, "JOURNAL_PATH", state / "preference-journal.jsonl")
    monkeypatch.setattr(verdict, "JURY_CASE_DIR", state / "evals" / "jury-cases")
    monkeypatch.setattr(dispatch, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", state / "taskbook-lint.json")
    monkeypatch.setattr(drain, "LOCK_PATH", state / "devops.lock")
    monkeypatch.setattr(drain, "DEVOPS_RUNS_DIR", state / "devops-runs")
    monkeypatch.setattr(drain, "CANCEL_PATH", state / "devops.cancel")
    monkeypatch.setattr(watch, "WATCH_PATH", state / "watch.json")
    # 预设真源（toml）隔离：默认 mimo；切预设的用例用 write_preset(state, ...) 改一行。
    monkeypatch.setattr(settings, "SETTINGS_PATH", state / "pysci_settings.toml")
    seed_settings(state)
    # 渠道回退事件流（backlog 20261012-211500-leader 规格 4b）：降级一跳就追加一行，
    # 不 patch 则任何跑到换席通路的用例都真写仓库 state/channel-events.jsonl。
    monkeypatch.setattr(channels, "EVENTS_PATH", state / "channel-events.jsonl")
    # E3-prep（仲裁团）：裁决留痕/降级队列/冻结简报三份状态同归 tmp——不 patch 会让
    # mock 测试真往 orchestration/state/evals/ 写案卷（与真实提案混在一起）。
    monkeypatch.setattr(jury, "JURY_DIR", state / "evals" / "jury")
    monkeypatch.setattr(jury, "JURY_QUEUE_DIR", state / "evals" / "jury-queue")
    monkeypatch.setattr(jury, "JURY_BRIEF_PATH", state / "evals" / "jury-brief.md")
    # 路由 v3：切换留痕的运行态（跨进程比对上次读数）——不 patch 就真写仓库那份，
    # 而且上一用例留下的读数会把本用例的「首次只立基线」分支吃掉（误报一次切换）。
    monkeypatch.setattr(routing, "ROUTING_STATE_PATH", state / "routing-last.json")
    # drain 启动的模型映射自愈默认 no-op（防测试真起 exe/写真实 registry；
    # 测刷新行为的用例自行 monkeypatch 覆盖）
    monkeypatch.setattr(
        drain,
        "refresh_models",
        lambda: {"changed": {}, "missing": [], "catalog_size": 0},
    )
    # 脏跟踪面哨兵默认「干净」：不 patch 则 orch status 用例会真跑 ``git status`` 读
    # 开发者工作区，输出随仓库脏净而变（backlog 20261012-080410-devops 方案 C）。
    monkeypatch.setattr(orch, "dirty_tracked_state", lambda **kw: [])
    # E5a：巡检引擎**内嵌**在 status/approve/dispatch 入口（§7「生命周期由代码保证」），
    # 所以凡是跑到这三个入口的用例都会试着拉起 worker。默认 no-op 派生 + 六面状态重定向：
    # 不 patch 就是每个此类用例真起一个后台进程，并把发现台账/迟滞态/开关写进仓库 state。
    # 这是**用例面**的 patch；系统性的最后一道在 drain._spawn_detached（认测试进程即停），
    # 漏挂本夹具的用例由此兜住，但要跑真 Popen 面的用例得自己清 TEST_PROCESS_ENV。
    monkeypatch.setattr(inspectd, "_spawn_detached", lambda argv, **kw: None)
    insp = state / "inspector"
    monkeypatch.setattr(inspectd, "STATE_DIR", insp)
    monkeypatch.setattr(inspectd, "FINDINGS_PATH", insp / "findings.jsonl")
    monkeypatch.setattr(inspectd, "THROTTLE_PATH", insp / "throttle.json")
    monkeypatch.setattr(inspectd, "QUEUE_PATH", insp / "queue.json")
    monkeypatch.setattr(inspectd, "EVIDENCE_DIR", insp / "evidence")
    monkeypatch.setattr(inspectd, "ENABLE_PATH", state / "inspector.enabled")
    monkeypatch.setattr(
        inspectd,
        "INSPECT_PATHS",
        drain.WorkerPaths(
            lock=state / "inspector.lock",
            runs=state / "inspector-runs",
            cancel=state / "inspector.cancel",
        ),
    )
    return state


@pytest.fixture
def iso_bench(tmp_path, monkeypatch, iso_state):
    """重定向 bencher 测量队列四面（队列/锁/run/取消）到 ``iso_state`` 目录下。

    E4 的 worker 生命周期落点与 devops drain **同构但不同名**（复用同一套锁原语，
    落点各一套）：不 patch 就会真写 ``orchestration/state/bencher*``。默认把
    ``_spawn_detached`` 与 ``run_case`` 都桩掉——前者防测试真起后台进程，后者防外呼。
    """
    state = iso_state
    monkeypatch.setattr(benchd, "QUEUE_PATH", state / "bencher" / "queue.json")
    monkeypatch.setattr(
        benchd,
        "BENCH_PATHS",
        drain.WorkerPaths(
            lock=state / "bencher.lock",
            runs=state / "bencher-runs",
            cancel=state / "bencher.cancel",
        ),
    )
    # 派生默认 no-op（测 argv 的用例自行覆盖）；eval 真派发一律禁起
    monkeypatch.setattr(benchd, "_spawn_detached", lambda argv, **kw: None)
    monkeypatch.setattr(benchd._eval, "run_case", _boom_run_case)
    return state


def _boom_run_case(*a, **k):
    raise AssertionError("测试不该真跑 eval.run_case（请 patch benchd._eval.run_case）")


@pytest.fixture
def iso_inspect(iso_state, monkeypatch):
    """巡检引擎用例的基座：在 :func:`iso_state` 的隔离面之上把**让路判定**钉成确定值。

    ``busy_surfaces`` 默认读真实 devops/bencher 锁与 ``plans/*.state.json``——开发者机器
    上恰好有一个 drain 在跑，测试就会看到 ``stopped=yield`` 而不是预期的 ``budget_hops``。
    闸门语义本身有专测（patch 到指定 busy 值），其余用例只关心「全闲时引擎该干什么」，
    故这里默认清空 busy，并让 ``plan_run_active`` 恒假。
    """
    monkeypatch.setattr(inspectd, "busy_surfaces", lambda **kw: [])
    return iso_state


def set_inspect_busy(monkeypatch, busy):
    """把让路面钉成给定清单（测「生产侧忙 → 只读巡检」的闸门语义）。"""
    monkeypatch.setattr(inspectd, "busy_surfaces", lambda **kw: list(busy))


def seed_backlog(state, items):
    """写入 backlog.json（items 为条目字典列表）。"""
    (state / "backlog.json").write_text(
        json.dumps({"version": 1, "items": items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_backlog(state):
    """读回 backlog.json 的 items 列表。"""
    return json.loads((state / "backlog.json").read_text(encoding="utf-8"))["items"]


# ---------------------------------------------------------------------------
# git 桩：按「隔离面台账」自洽应答（抢救判据的测试替身）
# ---------------------------------------------------------------------------
class GitFaces:
    """替 ``drain._run_git`` 造一个**内部自洽**的假 git——按面台账答题，不是全绿。

    为什么不用 ``lambda *a: (0, "")``：抢救判据读的是「目录在不在、登记不登记、有没有
    未提交/未合并内容」这一组**互相约束**的事实，而全绿桩把它们全钉成「未知」——于是
    只能测出保守方向（永不删），测不出「确认干净照旧销毁」那一支，也测不出脏面到底
    有没有真被打成提交。台账式应答让每个用例只声明它关心的那几个事实，其余由本类推
    导出一致读数（remove 之后目录就没了、commit 之后 head 就变了）。

    用法::

        git = GitFaces(wt_root).patch(monkeypatch)
        git.face("a1", dirty=True)                 # 在册活面 + 未提交改动
        git.face("a2", unmerged=2, head="c0ffee1")  # 干净但有未合并提交
        git.face("a3", unmerged=1, registered=False, lingering=True)  # remove 摘不掉的残骸
    """

    def __init__(self, wt_root):
        self.wt_root = pathlib.Path(wt_root)
        self.specs: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.commits: list[tuple[str, str]] = []  # (面名, 提交消息)

    def face(
        self,
        name: str,
        *,
        dirty: bool = False,
        unmerged: int = 0,
        head: str = "c0ffee1",
        branch: bool = True,
        registered: bool = True,
        lingering: bool = False,
    ) -> GitFaces:
        """声明一个隔离面的事实（``registered``=在册；``lingering``=remove 后目录仍趴着）。"""
        path = self.wt_root / name
        spec = {
            "name": name,
            "path": path,
            "branch": branch,
            "branch_name": f"wt-{name}",
            "head": head,
            "dirty": dirty,
            "unmerged": unmerged,
            "registered": registered,
            "lingering": lingering,
        }
        self.specs[name] = spec
        if registered or lingering:
            path.mkdir(parents=True, exist_ok=True)
        return self

    # -- 应答面 --------------------------------------------------------------
    def _by_path(self, raw: str) -> dict | None:
        want = pathlib.Path(raw).as_posix().lower()
        for spec in self.specs.values():
            if spec["path"].as_posix().lower() == want:
                return spec
        return None

    def _by_branch(self, bname: str) -> dict | None:
        for spec in self.specs.values():
            if spec["branch"] and spec["branch_name"] == bname:
                return spec
        return None

    def run(self, *args: str) -> tuple[int, str]:
        self.calls.append(args)
        head, *rest = args
        if head == "-C":
            path, *tail = rest
            spec = self._by_path(path)
            if spec is None or not spec["registered"]:
                # 未登记/未知目录：git 在里面根本读不出仓库（残骸的真实下场）
                return 1, "fatal: not a git repository"
            if tail[:2] == ["status", "--short"]:
                return 0, "M src/x.py" if spec["dirty"] else ""
            if tail[:1] == ["add"]:
                return 0, ""
            if tail[:2] == ["commit", "--no-verify"]:
                msg = tail[-1]
                self.commits.append((spec["name"], msg))
                spec["head"] = "wipsalv1"
                spec["unmerged"] = max(spec["unmerged"], 1)
                spec["dirty"] = False
                return 0, "[wt-… 新建 WIP]"
            return 0, ""
        if args[:3] == ("rev-parse", "--verify", "--quiet"):
            bname = str(args[3]).removeprefix("refs/heads/")
            spec = self._by_branch(bname)
            return (0, spec["head"]) if spec else (1, "fatal: not a valid object name")
        if args[:2] == ("rev-parse", "--short"):
            spec = self._by_branch(str(args[2]))
            return (0, spec["head"]) if spec else (1, "")
        if args[:2] == ("rev-list", "--count"):
            bname = str(args[2]).split("..")[-1]
            spec = self._by_branch(bname)
            if spec is None:
                return 1, "fatal: bad revision"
            return 0, str(spec["unmerged"])
        if args[:2] == ("worktree", "list"):
            lines = [f"worktree {pathlib.Path('.').resolve().as_posix()}"]
            lines += [
                f"worktree {s['path'].as_posix()}"
                for s in self.specs.values()
                if s["registered"]
            ]
            return 0, "\n".join(lines)
        if args[:3] == ("worktree", "remove", "--force"):
            spec = self._by_path(str(args[3]))
            if spec is None or not spec["registered"]:
                return 1, "fatal: is not a working tree"
            spec["registered"] = False
            if not spec["lingering"]:
                shutil.rmtree(spec["path"], ignore_errors=True)
            return 0, ""
        if args[:2] == ("branch", "-D"):
            spec = self._by_branch(str(args[2]))
            if spec is None:
                return 1, "error: branch not found"
            spec["branch"] = False
            spec["registered"] = False
            return 0, f"Deleted branch {spec['branch_name']}"
        if args[:2] == ("worktree", "prune"):
            return 0, ""
        return 0, ""

    def patch(self, monkeypatch, module=None) -> GitFaces:
        """把本台账接到 ``drain._run_git``（module 可换别的被测模块）。"""
        target = module if module is not None else drain
        monkeypatch.setattr(target, "_run_git", self.run)
        return self

    def removed(self, name: str) -> bool:
        """该面的目录是否已随 remove 消失（残骸面答 False，供断言）。"""
        spec = self.specs.get(name)
        return bool(spec) and not spec["path"].exists()


def seed_suggestion(state, member="devops", sid=None):
    """在 suggestions/ 落一条**合规**待审批建议文件，返回其 id（文件名 stem）。

    E0-B 后 cmd_approve 强制条目带非空 kind 与 success_criteria，故默认正文含这两枚
    标记行（约定见 delivery.parse_suggestion_fields）；要测「缺字段拒绝入队」的用例
    自行覆写文件正文。
    """
    d = state / "suggestions"
    d.mkdir(parents=True, exist_ok=True)
    sid = sid or f"20261010-120000-{member}"
    (d / f"{sid}.md").write_text(
        f"# 改进建议（{member}，待审批）\n\nkind: usability\n建议正文首行。\n\n"
        "success_criteria：\n- 默认判据（可证伪）\n",
        encoding="utf-8",
    )
    return sid


def seed_raw_suggestion(state, body: str, member="devops", sid=None):
    """落一条**任意正文**的待审批建议（测 kind/success_criteria 缺失拒绝路径用）。"""
    d = state / "suggestions"
    d.mkdir(parents=True, exist_ok=True)
    sid = sid or f"20261010-120000-{member}"
    (d / f"{sid}.md").write_text(body, encoding="utf-8")
    return sid


# ---------------------------------------------------------------------------
# do_dispatch 全隔离夹具 + 假 headless（quota/fallback 系列共用）
# ---------------------------------------------------------------------------
@pytest.fixture
def iso_dispatch(tmp_path, monkeypatch):
    """隔离 do_dispatch：tmp registry/ledger + 假 pod + 假 exe。

    默认 registry models 走**旧档形态**（``flash``/``flash_fallback`` → main 角色兼容读）；
    用 :func:`set_models` 改链、:func:`set_presets` 写新预设块。返回 state 目录。
    """
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)  # 与 iso_state 同锚，组合挂载时不互撞
    pod = tmp_path / "pods" / "quotamember"
    (pod / "inbox").mkdir(parents=True)
    reg_file = state / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "version": 1,
                "exe": None,
                "models": {"max": "", "flash": "", "flash_fallback": ""},
                "members": {
                    "quotamember": {
                        "pod": str(pod),
                        "max_turns": 3,
                        "timeout_s": 5,
                    }
                },
                "checks": {},
                # 同席重试（backlog 20261013-143000-leader）测试口径：出厂缺省 10 次，
                # 但既有用例量的是**换席次序**，重试次数随缺省走会让每个渠道故障用例
                # 多出 10 跳、且漏算一次就红。故压到 1：「先自救再换席」的骨架不变，
                # 用例只需在首跳后多认一跳。验次数闸/壁钟闸本身的用例自行覆盖写
                # （Registry 每次调用重读盘，后写胜出）。
                "model_fallback": {
                    "same_seat_retry": {
                        "channel_error": 1,
                        "backoff_s": 0,
                        "budget_s": 0,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # 运行态会话池预置为空：配置已无内嵌 sessions，load 不会触发迁移（夹具确定化）
    (state / RUNTIME_FILE).write_text(
        json.dumps({"version": 1, "members": {"quotamember": {"sessions": []}}}),
        encoding="utf-8",
    )
    exe = tmp_path / "fake-exe.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("PYSCI_ORCH_EXE", str(exe))
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    # Registry.save 的 path 默认值在类定义时绑定，patch 类属性改不了 __init__ 默认——
    # 直接 no-op save，防止测试写真实 registry.json
    monkeypatch.setattr(registry.Registry, "save", lambda self: None)
    monkeypatch.setattr(ledger, "LEDGER_PATH", state / "ledger.jsonl")
    monkeypatch.setattr(settings, "SETTINGS_PATH", state / "pysci_settings.toml")
    seed_settings(state)
    monkeypatch.setattr(channels, "EVENTS_PATH", state / "channel-events.jsonl")
    # 路由面：do_dispatch 每跳都读一次峰谷并可能落「上次读数」+ 决策日志。不 patch
    # 就成了测试往真实 worktree 状态面写 routing-last.json / preference-journal.jsonl。
    monkeypatch.setattr(routing, "ROUTING_STATE_PATH", state / "routing-last.json")
    monkeypatch.setattr(journal, "JOURNAL_PATH", state / "preference-journal.jsonl")
    monkeypatch.setattr(dispatch, "REPLIES_DIR", state / "replies")
    monkeypatch.setattr(dispatch, "SUGGESTIONS_DIR", state / "suggestions")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", state / "taskbook-lint.json")
    # 会话 jsonl 存储面（est_tokens 的测量对象）改到 ``<state>/sessions``：既隔离 est
    # 增量断言，也防测试真去读开发者机器上 ~10 MB 级的真实会话文件（sessions 直接
    # import 该函数，故 patch sessions 命名空间而非 registry）
    monkeypatch.setattr(
        dispatch_sessions, "pod_sessions_dir", lambda pod: state / "sessions"
    )
    return state


def grow_session(state, sid: str, text: str) -> int:
    """往隔离会话 jsonl 追加文本（模拟该跳说了话），返回追加后的字符总数。

    真实会话文件是 append-only 的 JSON 行，而 est_tokens 只看字符增量，故测试只需追加。
    """
    p = state / "sessions" / f"{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(text)
    return len(p.read_text(encoding="utf-8"))


def sid_from_argv(cmd) -> str:
    """从派发 argv 取会话 id（``--session-id``/``--resume`` 的取值）。

    渠道故障的形状判据读的是**会话档**，而 sid 由 do_dispatch 自己生成（新会话=随机
    uuid），用例只能从 argv 里把它接回来，再往对应文件写末轮形状。
    """
    for flag in ("--session-id", "--resume"):
        if flag in cmd:
            return cmd[cmd.index(flag) + 1]
    return ""


def append_turn(
    state,
    sid: str,
    *,
    model: str,
    text: str,
    stop_reason: str = "stop_sequence",
    stop_sequence: str = "",
) -> None:
    """往隔离会话档追加一条 assistant 记录（钉「这一跳是怎么收尾的」的形状）。

    只写 :func:`sessions.channel_failure_turn` 读的那几个字段：真实模型回合与上游故障
    合成回合的差别全在 ``model`` / ``stop_reason`` / ``stop_sequence`` 三者上。
    """
    rec = {
        "type": "assistant",
        "message": {
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "content": [{"type": "text", "text": text}],
        },
    }
    p = state / "sessions" / f"{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def enable_registry_save(monkeypatch):
    """还原真实 ``Registry.save``（默认夹具为防污染把它 no-op 掉了）。

    真实 save 是**读-改-写按键合并**：配置面磁盘权威（未声明改动则一字不写），运行态
    按成员/按 sid 合并。落点随 load 绑定的 ``self.path`` 移动，本夹具已把 REGISTRY_PATH
    指到 state 目录，故只写隔离副本、不碰真实 registry.json。
    """
    monkeypatch.setattr(registry.Registry, "save", REAL_REGISTRY_SAVE)


def read_registry(state):
    """读回隔离 registry.json（配置面）的 data dict。"""
    return json.loads((state / "registry.json").read_text(encoding="utf-8"))


def read_runtime(state):
    """读回隔离 sessions-runtime.json（运行态会话池）的 data dict。"""
    return json.loads((state / RUNTIME_FILE).read_text(encoding="utf-8"))


def write_runtime(state, data):
    """写入隔离 sessions-runtime.json（预置会话池用）。"""
    (state / RUNTIME_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def set_models(state, **models):
    """改写隔离 registry.json 的 models 块（do_dispatch 调用时 Registry.load 重读）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["models"] = models
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def fake_headless(monkeypatch, responder):
    """monkeypatch dispatch.run_headless；responder(call_index, cmd) -> (rc, out, err)。

    返回 calls 列表（每项为该次 argv），供断言所用模型（``-m`` 后的值）。
    ``on_spawn`` 若被调用方传入，就以 :data:`FAKE_HOP_PID` 回调一次——孤儿会话回收
    （backlog 20261012-184000-leader）依赖派生瞬间的 pid 登记，桩不给 pid 就等于
    把那条通路测成了哑巴。
    """
    calls = []

    def run(cmd, cwd=None, env=None, timeout_s=None, on_spawn=None):
        calls.append(cmd)
        if on_spawn is not None:
            on_spawn(FAKE_HOP_PID)
        return responder(len(calls), cmd)

    monkeypatch.setattr(dispatch, "run_headless", run)
    return calls


def envelope(result, *, is_error=True, stop="stop_sequence", rc=1, ctx=0.0):
    """构造 run_headless 的 (rc, stdout, err) 三元组（stdout 为 envelope JSON）。

    ``ctx``：本跳结束时的上下文占用比（envelope 的 ``context_usage_ratio``）。渠道故障的
    形状判据读它，``touch_session`` 也把它落进会话池（纯观测——fe68a40 的水位闸已回退，
    不参与会话选择）；不给就默认 0。
    """
    return (
        rc,
        json.dumps(
            {
                "result": result,
                "is_error": is_error,
                "stop_reason": stop,
                "num_turns": 1,
                "duration_ms": 13000,
                "session_id": "aaaa1111",
                "usage": {"context_usage_ratio": ctx},
            }
        ),
        "",
    )


def cmd_model(cmd):
    """从 argv 提取 ``-m`` 后的模型串（无 -m 时返回 ''，代表走用户级默认）。"""
    return cmd[cmd.index("-m") + 1] if "-m" in cmd else ""


# ---------------------------------------------------------------------------
# E3-prep 仲裁团夹具（jury 全链路 mock：不起网络、不碰真实 state/evals）
# ---------------------------------------------------------------------------
CASE_BODY = """targets: guards/target.txt

## 问题
是否采纳该基建改动？

## 选项 A
实施改动：{a}

## 选项 B
维持现状：{b}

## 证据
台账与 diff 见自动附着材料段。
"""


#: 逐席 provider 下每席各有 key_env；测试夹具让所有桩席位共用这一个变量名，
#: 需要「一席缺 Key」的场景时再单独给某席换名（见 test_jury 的回退阳性用例）。
SEAT_KEY_ENV = "PYSCI_JURY_TEST_KEY"


@pytest.fixture
def iso_jury(tmp_path, monkeypatch, iso_state, iso_registry):
    """隔离仲裁团三面（留痕/队列/简报）+ 假项目根 + 桩 git，返回 state 目录。

    ``PROJECT_ROOT`` 改到 tmp：``file_reading`` 的涉事文件读数与 ``read_key`` 的
    ``.env`` 回落都在 tmp 里发生，测试自造文件即可，不受真实仓库工作区状态影响。
    """
    monkeypatch.setattr(jury, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(jury, "run_git", lambda *a: "stub-git-output")
    monkeypatch.delenv(SEAT_KEY_ENV, raising=False)
    (tmp_path / "state" / "evals").mkdir(parents=True, exist_ok=True)
    (tmp_path / "guards").mkdir(parents=True, exist_ok=True)
    # write_bytes 而非 write_text：Windows 下文本写会 smudge 成 CRLF，
    # 「涉事文件字节读数」断言的字节数就随平台漂移
    (tmp_path / "guards" / "target.txt").write_bytes(b"line1\nline2\n")
    return iso_state


def seed_case(state, *, case_id="case-1", a="实施改动，收益是机械可验收", b="维持现状"):
    """落一份合规范案卷（两选项内容默认可区分，供双序映射断言），返回其路径。"""
    d = state / "evals" / "cases"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{case_id}.md"
    p.write_text(CASE_BODY.format(a=a, b=b), encoding="utf-8")
    return p


def set_jury_panel(reg_file, panel, **extra):
    """写隔离 registry.json 的 ``jury`` 块（E3-live 逐席 provider 结构）。

    ``panel`` 为席位清单，每项可以是型号串（自动补 name/base_url/key_env=SEAT_KEY_ENV）
    或完整席位 dict（原样并入，缺 key_env 回落 SEAT_KEY_ENV）。``extra`` 逐键覆盖 jury 块
    （如 ``deadline_s`` / ``pricing`` / ``fallback`` / ``seats``）。名称沿用 ``panel``
    是测试侧的历史包袱，落到配置面已是 ``seats``。
    """
    data = {"version": 1, "exe": None, "models": {}, "members": {}, "checks": {}}
    seats: list[dict] = []
    for item in panel:
        if isinstance(item, dict):
            seat = {"key_env": SEAT_KEY_ENV, **item}
            seat.setdefault("name", seat.get("model", "seat"))
            seat.setdefault("base_url", "https://stub.invalid/v1")
        else:
            seat = {
                "name": str(item),
                "model": str(item),
                "base_url": f"https://stub.invalid/{item}/v1",
                "key_env": SEAT_KEY_ENV,
            }
        seats.append(seat)
    block: dict = {"seats": seats, "fallback": {"enabled": False}, **extra}
    data["jury"] = block
    reg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def ai_reply(choice, *, reason="理由", confidence=0.8, tokens=(100, 50)):
    """OpenAI 兼容响应体（choices/usage），供 transport 桩返回。"""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"choice": choice, "reason": reason, "confidence": confidence},
                        ensure_ascii=False,
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    }


def display_label_for_liked(messages, liked_text):
    """案卷里 ``liked_text`` 落在哪个显示标签下（双序对局里 A/B 会互换）。

    裁判「按内容判决」的正确模拟：不是记住 A 或 B，而是认出自己偏好的那段文字
    当前挂在哪个标签上——这样 BA 局的原始答案必然翻转，映射回真实选项后才可比。
    """
    user = messages[-1]["content"]
    block_a = re.search(r"## 选项 A\n(.*?)\n\n## 选项 B", user, re.S).group(1)
    return "A" if liked_text in block_a else "B"


def prefer_juror(liked_text, **kw):
    """内容驱动型裁判桩：永远投给它偏好的那段内容（双序两局都判同一真实选项）。"""

    def transport(model, messages, *, base_url, key, timeout_s, **ignored):
        del model, base_url, key, timeout_s, ignored
        return ai_reply(display_label_for_liked(messages, liked_text), **kw)

    return transport


def biased_first_juror(**kw):
    """位置偏好型裁判桩：永远投标签 A —— BA 局映射后与 AB 局矛盾 → inconclusive。"""

    def transport(model, messages, *, base_url, key, timeout_s, **ignored):
        del model, messages, base_url, key, timeout_s, ignored
        return ai_reply("A", **kw)

    return transport
