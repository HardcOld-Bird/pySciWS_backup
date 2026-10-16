"""E5a inspector 上线的**在位性**阳性回归（backlog 20261012-120003-leader）。

与 ``test_bencher_pod.py`` 同理：巡检引擎的行为面另有 ``test_inspect_engine.py`` 覆盖，
本文件只量实物——pod 脚手架、注册表条目（含写白名单与只读追加）、技能真本与部署目标、
CLI 入口、gitignore 面、文档回灌。这些错了不会让任何逻辑测试变红，只会让**下一次自主
激活静默失败**（技能没部署、写面给错导致 finding 落不了盘、``pysci-inspect`` 装不上）。

inspector 的在位性还多一条别处没有的断言：**它没有 override 面**（§5.2）。这是代码事实
而非纪律条款，因此必须以「参数表里取不到这个参数」的形式量化，而不是以「文档写了不许」
的形式——后者正是 ``consult`` 软选项从不被使用的老路。
"""

from __future__ import annotations

import json
import tomllib

import pytest

from pysci.paths import PODS_ROOT, PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / "orchestration" / "skills" / "manifest.toml"
REGISTRY_PATH = PROJECT_ROOT / "orchestration" / "state" / "registry.json"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
SKILL_DIR = PROJECT_ROOT / "orchestration" / "skills" / "inspector"
POD = PODS_ROOT / "inspector"

#: inspector 会话**不许**使用的 CLI（组长/设施专属；charter 明写）
FORBIDDEN_CLIS = ("pysci-orch", "pysci-dev")


def test_inspector_pod_scaffold_present():
    """pod 五件套 + rubric 在位，settings 两 hook 接线与组员模板逐字节相同。"""
    for rel in (
        ".gitignore",
        ".qoder/settings.json",
        ".qoder/mcp.json",
        ".qoder/rules/charter.md",
        "AGENTS.md",
        "rubrics/design-review.md",
    ):
        assert (POD / rel).is_file(), f"缺 {rel}"
    s = json.loads((POD / ".qoder/settings.json").read_text(encoding="utf-8"))
    assert s["agentsMdExcludes"] == ["**/leader-only.md"], (
        "写错排除面会把 basic.md 公约数一起吞掉（charter 明写的坑）"
    )
    names = {
        h["name"]
        for section in ("Stop", "PreToolUse")
        for grp in s["hooks"][section]
        for h in grp["hooks"]
    }
    assert names == {"delivery-gate", "pod-guard"}
    # 组员模板跨 pod 逐字节一致（bencher E4 先例）：漂移意味着脚手架手抄而非同源
    template = (PODS_ROOT / "bencher" / ".qoder" / "settings.json").read_bytes()
    assert (POD / ".qoder" / "settings.json").read_bytes() == template


def test_inspector_pod_gitignore_covers_runtime_faces():
    """pod 内 gitignored 运行面（outbox/部署副本/inbox 正文）齐备。"""
    text = (POD / ".gitignore").read_text(encoding="utf-8")
    for entry in ("outbox/", ".qoder/skills/", "inbox/task-*.md"):
        assert entry in text, f"pod .gitignore 缺 {entry}"


def test_inspector_docs_within_harness_budget():
    """charter / AGENTS.md / SKILL.md / references / rubric 各 ≤8192 **落盘字节**。"""
    faces = (
        POD / ".qoder" / "rules" / "charter.md",
        POD / "AGENTS.md",
        POD / "rubrics" / "design-review.md",
        SKILL_DIR / "SKILL.md",
        SKILL_DIR / "references" / "patrol-craft.md",
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "orchestration"
        / "references"
        / "inspector-engine.md",
    )
    for f in faces:
        assert f.is_file(), f"缺文件：{f}"
        n = len(f.read_bytes())
        assert n <= 8192, f"{f.name} 超全量注入档预算：{n} B"


def test_inspector_charter_declares_readonly_permission_face():
    """charter 把「全项目只读 + 产出只落两面 + 豁口」写成条款。

    写面清单必须与 registry ``extra_dirs`` **逐字一致**：不一致时 inspector 自己的
    ``intent-drift`` 巡检项就会把这条当成发现（文档说一面、闸门放行另一面）。
    """
    charter = (POD / ".qoder" / "rules" / "charter.md").read_text(encoding="utf-8")
    for key in (
        "全项目只读",
        "orchestration/state/inspector",
        "orchestration/state/evals/jury-cases",
        "没有 override 面",
        "pysci-inspect",
    ):
        assert key in charter, f"charter 缺权限/纪律条款：{key}"
    for cli in FORBIDDEN_CLIS:
        assert cli in charter, f"charter 未声明不使用 {cli}"
    # §0.2 豁口必须在 charter 里点名（否则组员会以为自己该巡自己的锤子）
    assert "§0.2" in charter and "豁口" in charter

    dirs = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["members"][
        "inspector"
    ]["extra_dirs"]
    for d in dirs:
        assert d in charter, f"registry 写面 {d} 未写进 charter（两面漂移）"


def test_registry_registers_inspector_with_produce_writeface():
    """registry：默认档 flash、写白名单只有产出两面、rubric 面只读。"""
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    m = data["members"]["inspector"]
    assert "model_tier" not in m, (
        "档位退役（2026-10-12 22:4x）：角色由预设 pro 名单解析，不再落成员条目"
    )
    assert m["extra_dirs"] == [
        "orchestration/state/inspector",
        "orchestration/state/evals/jury-cases",
    ], "extra_dirs 是写白名单：多给一项就越权（巡检员动手即失独立性）"
    assert m["readonly_extra"] == ["rubrics"], (
        "rubrics 是护栏级只读层（修订权归用户），必须经 pod-guard 拦写"
    )
    assert m["pod"].replace("\\", "/").endswith("orchestration/pods/inspector")
    assert 0 < int(m["max_turns"]) <= 200 and int(m["timeout_s"]) > 0


def test_registry_inspector_params_are_hysteresis_legal():
    """顶层 ``inspector`` 块（若显式写）必须迟滞合法，且单跳超时必须短于墙钟预算。"""
    from pysci.skills.orchestration.tools.registry import DEFAULT_INSPECTOR

    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    block = {k: v for k, v in data.get("inspector", {}).items() if k != "_note"}
    par = dict(DEFAULT_INSPECTOR, **block)
    assert par["resume_backlog"] < par["pause_backlog"], "恢复阈值必须严格低于暂停阈值"
    assert par["resume_bench"] < par["pause_bench"]
    assert par["max_hops"] >= 1 and par["wallclock_s"] > 0 and par["cooldown_s"] >= 0
    timeout = data["members"]["inspector"]["timeout_s"]
    assert timeout < par["wallclock_s"], (
        "单跳超时 ≥ 激活墙钟预算时，「预算到顶自停」退化成「跳完才发现超预算」"
    )


def test_manifest_ships_inspector_skill_targeting_only_inspector_pod():
    """技能真本在位 + 部署目标只有 inspector pod（一成员一技能一 CLI，§0.9）。"""
    data = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    hit = [s for s in data["skills"] if s.get("name") == "inspector"]
    assert len(hit) == 1
    src = hit[0]["source"].replace("\\", "/")
    assert (PROJECT_ROOT / src / "SKILL.md").is_file(), f"真本缺失：{src}"
    targets = [t.replace("\\", "/") for t in hit[0]["targets"]]
    assert targets == ["orchestration/pods/inspector/.qoder/skills"]


def test_pyproject_registers_pysci_inspect_entry_point():
    """console script 指向**可导入且带 main** 的模块（venv-sync 后 pysci-inspect 才可用）。"""
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert 'pysci-inspect = "pysci.skills.inspector.tools.inspect:main"' in text
    import importlib

    mod = importlib.import_module("pysci.skills.inspector.tools.inspect")
    assert callable(mod.main) and callable(mod.build_parser)


def test_state_faces_gitignored():
    """巡检运行态（队列/台账/证据/开关/锁/run/取消）整面不入库。"""
    text = GITIGNORE_PATH.read_text(encoding="utf-8")
    for entry in (
        "/orchestration/state/inspector/",
        "/orchestration/state/inspector.enabled",
        "/orchestration/state/inspector.lock",
        "/orchestration/state/inspector-runs/",
        "/orchestration/state/inspector.cancel",
    ):
        assert entry in text, f".gitignore 缺 {entry}"


def test_readme_declares_inspector_role_and_readonly_limits():
    """README §1 角色行 + 成员 id 清单 + 一成员一技能一 CLI 段。"""
    readme = _doc_corpus()
    row = next(ln for ln in readme.splitlines() if ln.startswith("| **inspector** |"))
    for token in ("state/inspector", "pysci-inspect", "无 override", "只读"):
        assert token in row, f"README inspector 行缺 {token}"
    assert "`twin` `inspector`" in readme or " `inspector`" in readme


def test_leader_skill_declares_inspector_lifecycle_faces():
    """组长技能给出全部生命周期命令面（缺席=组长不知道该敲什么）。"""
    text = (
        PROJECT_ROOT / "orchestration" / "skills" / "orchestration" / "SKILL.md"
    ).read_text(encoding="utf-8")
    for face in ("inspector {start,status,patrol,assign,enable,disable}",):
        assert face in text, f"组长技能缺 CLI 面：{face}"
    assert "references/inspector-engine.md" in text, "口径文档必须给出一行索引"


def test_inspect_cli_parses_documented_surface():
    """parser 真能构造且 charter/SKILL 写的四条命令面可用。

    ``--class`` 曾因 ``add_argument("--class", "cls")`` 的第二位置串被 argparse 当成
    option string 而**整个 CLI 构造即抛 ValueError**——组员一侧表现为 pysci-inspect 全灭、
    巡检一跳都落不了盘。这类错误只在真构造 parser 时才暴露，故必须在此量一次。
    """
    from pysci.skills.inspector.tools.inspect import build_parser

    ns = build_parser().parse_args(
        [
            "finding",
            "add",
            "--class",
            "mechanical",
            "--title",
            "现象",
            "--repro",
            "cmd",
            "--evidence",
            "out",
            "--criteria",
            "- 判据",
        ]
    )
    assert (ns.cls, ns.title, ns.criteria) == ("mechanical", "现象", "- 判据")
    assert build_parser().parse_args(["finding", "list"]).status == ""
    assert build_parser().parse_args(["promote", "insp-1"]).finding == "insp-1"
    assert build_parser().parse_args(["submit", "insp-1", "--jury"]).jury is True
    assert build_parser().parse_args(["gate"]).cmd == "gate"


def test_inspector_has_no_override_surface():
    """**代码事实**核验：inspector 通道里根本不存在 override 参数（§5.2）。

    文档写「不许自裁」是软条款（consult 教训）；这里断言的是取不到这个参数：
    ① 引擎晋升函数签名里没有；② CLI 收到 ``--override`` 直接 argparse 报错；
    ③ 判断类唯一出口是 ``submit --jury``（``--jury`` 是必填项）。
    """
    import inspect as pyinspect

    from pysci.skills.inspector.tools import inspect as insp_cli
    from pysci.skills.orchestration.tools import inspectd

    for fn in (inspectd.submit_judgment, inspectd.promote_finding):
        assert "override" not in pyinspect.signature(fn).parameters, fn.__name__
    with pytest.raises(SystemExit):
        insp_cli.build_parser().parse_args(["submit", "insp-x", "--override"])
    with pytest.raises(SystemExit):
        insp_cli.build_parser().parse_args(["submit", "insp-x"])  # 缺 --jury

    # 组长侧 approve 才有 override 面（对照：不是全局没有这个词，是 inspector 通道没有）
    orch_text = (
        PROJECT_ROOT
        / "src"
        / "pysci"
        / "skills"
        / "orchestration"
        / "tools"
        / "orch.py"
    ).read_text(encoding="utf-8")
    assert "--override" in orch_text


def test_patrol_menu_is_engine_owned_and_complete():
    """巡检菜单九项在**引擎侧**固定（组员不自选），且 id 索引与自覆盖标注自洽。"""
    from pysci.skills.orchestration.tools import inspectd

    assert len(inspectd.PATROL_MENU) == 9
    ids = [e["id"] for e in inspectd.PATROL_MENU]
    assert len(set(ids)) == len(ids), "菜单 id 重复会让 PATROL_BY_ID 静默吞项"
    assert set(inspectd.PATROL_BY_ID) == set(ids)
    for e in inspectd.PATROL_MENU:
        assert e["brief"].strip(), f"{e['id']} 缺 brief（组员拿到空任务书）"
    self_cover = [e["id"] for e in inspectd.PATROL_MENU if e.get("self_cover")]
    assert self_cover == ["pipeline-self"], "§0.2 自覆盖项应唯一且明确豁口"
    rubric_items = [e["id"] for e in inspectd.PATROL_MENU if e.get("rubric")]
    assert rubric_items == ["design-rubric"]
    assert (POD / "rubrics" / "design-review.md").is_file()


def test_inspect_queue_and_finding_vocabularies_match_spec():
    """队列三型与发现两类别名漂移即静默死信（CLI choices 与引擎常量同源）。"""
    from pysci.skills.orchestration.tools import inspectd

    assert inspectd.QUEUE_KINDS == ("patrol", "assignment", "tier-a-draft")
    assert inspectd.FINDING_CLASSES == ("mechanical", "judgment")
    assert inspectd.FINDING_STATUSES == (
        "pending",
        "promoted",
        "rejected",
        "queued",
        "escalated",
        # 动作在组长侧 → 不派 devops，挂起等组长（backlog 20261013-184213-devops）
        "leader_todo",
    )
    assert inspectd.QUEUE_PATH.name == "queue.json"
    assert inspectd.FINDINGS_PATH.name == "findings.jsonl"
    assert inspectd.INSPECT_PATHS.lock.name == "inspector.lock"
    assert inspectd.INSPECT_PATHS.runs.name == "inspector-runs"
    assert inspectd.INSPECT_PATHS.cancel.name == "inspector.cancel"


def test_workq_is_single_source_for_both_queues():
    """benchd 与 inspectd 的**队列形状**都委派 :mod:`workq`（§0.9：共享逻辑做库）。

    只量形状（读写/取项/终态/计数），不量各业务的判据：benchd 的 A/B 污染闸与
    inspectd 的迟滞闸留在各自模块——那两份不是同一件事的两个入口，是一件事的两个
    必要层次（rubric ② 的判据）。
    """
    import inspect as pyinspect

    from pysci.skills.orchestration.tools import benchd, inspectd, workq

    for mod in (benchd, inspectd):
        src = pyinspect.getsource(mod)
        for call in ("workq.append", "workq.take_next", "workq.finish", "workq.counts"):
            assert call in src, (
                f"{mod.__name__} 的队列面未委派 {call}（形状又写了一份）"
            )
    for name in (
        "load",
        "save",
        "append",
        "take_next",
        "finish",
        "pending_count",
        "counts",
    ):
        assert callable(getattr(workq, name))
    assert workq.TERMINAL_STATUSES == ("done", "failed")
    assert benchd.TERMINAL_STATUSES is workq.TERMINAL_STATUSES


def test_engine_reference_declares_gate_stop_reasons():
    """``references/inspector-engine.md`` 写明自停原因与「无 CLI 旋钮」口径。"""
    text = (
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "orchestration"
        / "references"
        / "inspector-engine.md"
    ).read_text(encoding="utf-8")
    for token in (
        "budget_hops",
        "budget_wallclock",
        "switch_off",
        "cooperative",
        "yield",
        "throttled",
        "quota_exhausted",
    ):
        assert token in text, f"引擎文档缺自停原因 {token}（组长读不到就没有共识）"
    assert "队列空**不自毁**" in text or "队列空不自毁" in text.replace("**", "")


def test_inspector_deployed_copy_matches_source_when_present():
    """副本一旦存在必须与真本同源（禁止手改 pod 内部署副本）。"""
    deployed = POD / ".qoder" / "skills" / "inspector" / "SKILL.md"
    if not deployed.exists():
        pytest.skip("副本未部署（合并后 pysci-dev sync 生成）")
    assert deployed.read_bytes() == (SKILL_DIR / "SKILL.md").read_bytes()


def _doc_corpus() -> str:
    """文档语料（README v3.0 拆分后口径）：references/legacy-readme-v2.2.md（v2.2 原文留档）
    + README.md（v3.0 正文）+ references/*.md（细节块）——doc-face 断言据此判「文档面声明 X」。
    """
    root = PROJECT_ROOT / "orchestration"
    parts: list[str] = []
    legacy = root / "references" / "legacy-readme-v2.2.md"
    if legacy.exists():
        parts.append(legacy.read_text(encoding="utf-8"))
    parts.append((root / "README.md").read_text(encoding="utf-8"))
    refs = root / "references"
    if refs.is_dir():
        for f in sorted(refs.glob("*.md")):
            if f.name != "legacy-readme-v2.2.md":
                parts.append(f.read_text(encoding="utf-8"))
    return (chr(10) * 2).join(parts)
