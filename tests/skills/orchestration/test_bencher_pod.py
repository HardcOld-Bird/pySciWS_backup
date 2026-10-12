"""E4 bencher 上线的**在位性**阳性回归（backlog 20261012-120002-leader）。

任务书面要的是一张「设施图」：pod 脚手架、注册表条目、技能真本与部署目标、CLI 入口、
gitignore 面、文档所有权移交。这些错了不会让任何逻辑测试变红——它只会让下一次派发
静默失败（技能没部署、写白名单给错、CLI 装不上）。故本文件**只量实物**，不测行为：
行为面在 ``test_bench_queue.py``（队列/worker）与 ``test_bench_ab.py``（副本/污染）。

写面断言取的是 charter 的**权限面**（bencher 唯一常设写面 = ``state/evals``）：这条
给错就等于把「测量员不许改被测量的东西」这一职责边界写成空话。
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

#: bencher 会话**不许**使用的 CLI（组长/设施专属；charter 明写）
FORBIDDEN_CLIS = ("pysci-orch", "pysci-dev")


def test_bencher_pod_scaffold_present():
    """pod 五件套在位 + settings 两 hook 接线 + agentsMdExcludes（十 pod 先例同形）。"""
    pod = PODS_ROOT / "bencher"
    for rel in (
        ".gitignore",
        ".qoder/settings.json",
        ".qoder/mcp.json",
        ".qoder/rules/charter.md",
        "AGENTS.md",
    ):
        assert (pod / rel).is_file(), f"缺 {rel}"
    s = json.loads((pod / ".qoder/settings.json").read_text(encoding="utf-8"))
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


def test_bencher_pod_gitignore_covers_runtime_faces():
    """pod 内 gitignored 运行面（bench/outbox/inbox 正文/部署副本）齐备。

    这四者是**基线归档的对象**（git 保不住的组员面）：漏一个，副本 A/B 就少还原一面，
    测出来的读数不再是该 ref 的真实表现。
    """
    text = (PODS_ROOT / "bencher" / ".gitignore").read_text(encoding="utf-8")
    for entry in ("bench/", "outbox/", ".qoder/skills/", "inbox/task-*.md"):
        assert entry in text, f"pod .gitignore 缺 {entry}"


def test_bencher_docs_within_harness_budget():
    """charter / AGENTS.md / SKILL.md / references 各 ≤8192 **落盘字节**（CRLF 计税）。"""
    faces = (
        PODS_ROOT / "bencher" / ".qoder" / "rules" / "charter.md",
        PODS_ROOT / "bencher" / "AGENTS.md",
        PROJECT_ROOT / "orchestration" / "skills" / "bencher" / "SKILL.md",
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "bencher"
        / "references"
        / "bench-ops.md",
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "orchestration"
        / "references"
        / "drain-watch.md",
    )
    for f in faces:
        assert f.is_file(), f"缺文件：{f}"
        n = len(f.read_bytes())
        assert n <= 8192, f"{f.name} 超全量注入档预算：{n} B"


def test_bencher_charter_declares_permission_face():
    """charter 把「写面=state/evals、代码树零写、不自修设施」写成条款（职责边界）。"""
    charter = (PODS_ROOT / "bencher" / ".qoder" / "rules" / "charter.md").read_text(
        encoding="utf-8"
    )
    for key in ("orchestration/state/evals", "零写权", "infra_suggestion", "二元判决"):
        assert key in charter, f"charter 缺权限/纪律条款：{key}"
    for cli in FORBIDDEN_CLIS:
        assert cli in charter, f"charter 未声明不使用 {cli}"


def test_registry_registers_bencher_with_eval_writeface():
    """registry：默认档 flash、写白名单只有 state/evals（extra_dirs 即 pod-guard 白名单）。"""
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    m = data["members"]["bencher"]
    assert "model_tier" not in m, (
        "档位退役：被测档位由案卷 tier: 决定，成员身份不再自带档位"
    )
    assert m["extra_dirs"] == ["orchestration/state/evals"], (
        "extra_dirs 是写白名单：多给一项就越权，少给一项则案卷/基线写不进"
    )
    assert m["pod"].replace("\\", "/").endswith("orchestration/pods/bencher")
    assert 0 < int(m["max_turns"]) <= 200 and int(m["timeout_s"]) > 0


def test_manifest_ships_bencher_skill_targeting_only_bencher_pod():
    """技能真本在位 + 部署目标只有 bencher pod（一成员一技能一 CLI，§0.9）。"""
    data = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    hit = [s for s in data["skills"] if s.get("name") == "bencher"]
    assert len(hit) == 1
    src = hit[0]["source"].replace("\\", "/")
    assert (PROJECT_ROOT / src / "SKILL.md").is_file(), f"真本缺失：{src}"
    targets = [t.replace("\\", "/") for t in hit[0]["targets"]]
    assert targets == ["orchestration/pods/bencher/.qoder/skills"]


def test_pyproject_registers_pysci_bench_entry_point():
    """console script 指向**可导入且带 main** 的模块（uv sync 后 pysci-bench 才可用）。"""
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert 'pysci-bench = "pysci.skills.bencher.tools.bench:main"' in text
    import importlib

    mod = importlib.import_module("pysci.skills.bencher.tools.bench")
    assert callable(mod.main) and callable(mod.build_parser)


def test_state_faces_gitignored():
    """队列/锁/run/取消 + 基线物理归档都不入库（永久审计在 git + ledger + 判决文档）。"""
    text = GITIGNORE_PATH.read_text(encoding="utf-8")
    for entry in (
        "/orchestration/state/bencher/",
        "/orchestration/state/bencher.lock",
        "/orchestration/state/bencher-runs/",
        "/orchestration/state/bencher.cancel",
        "/orchestration/state/evals/baselines/*/pods/",
    ):
        assert entry in text, f".gitignore 缺 {entry}"


def test_benchd_item_vocabulary_matches_spec():
    """队列项四型 = run/capture/ab/check（§8 任务面）；终态两型才进待查看面。"""
    from pysci.skills.orchestration.tools import benchd

    assert benchd.ITEM_KINDS == ("run", "capture", "ab", "check")
    assert benchd.TERMINAL_STATUSES == ("done", "failed")
    assert benchd.QUEUE_PATH.name == "queue.json"
    # 与 devops drain **同构但不复制**（§0.9）：落点不同、原语同一套
    assert benchd.BENCH_PATHS.lock.name == "bencher.lock"
    assert benchd.BENCH_PATHS.runs.name == "bencher-runs"
    assert benchd.BENCH_PATHS.cancel.name == "bencher.cancel"


def test_eval_runner_channel_split_is_auditable():
    """runner 取值面：台账行分得开「谁发起的测量」（E4 所有权移交的审计依据）。"""
    from pysci.skills.orchestration.tools import eval as ev

    assert (ev.RUNNER_LEADER, ev.RUNNER_BENCHER) == ("leader", "bencher")
    assert ev.RUNNER_DEFAULT == ev.RUNNER_LEADER, "旧行缺席=移交前组长侧发起"
    import inspect

    assert "runner" in inspect.signature(ev.run_case).parameters
    assert "runner" in inspect.signature(ev.latest_rows).parameters
    assert "runner" in inspect.signature(ev.write_baseline).parameters


def test_docs_handoff_records_bencher_ownership():
    """套件所有权移交在权威文档留痕：README 角色行 + devops 例外退役 + 引用面。"""
    readme = (PROJECT_ROOT / "orchestration" / "README.md").read_text(encoding="utf-8")
    assert "| **bencher** |" in readme
    bencher_row = next(
        ln for ln in readme.splitlines() if ln.startswith("| **bencher** |")
    )
    assert "state/evals" in bencher_row and "pysci-bench" in bencher_row
    # devops 的 E2 过渡例外必须写明「已退役」，否则两个 CLI 都声称拥有套件
    devops_row = next(
        ln for ln in readme.splitlines() if ln.startswith("| **devops** |")
    )
    assert "退役" in devops_row and "bencher" in devops_row

    devops_skill = (
        PROJECT_ROOT / "orchestration" / "skills" / "devops" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "已移交 bencher" in devops_skill or "移交 bencher" in devops_skill

    leader_skill = (
        PROJECT_ROOT / "orchestration" / "skills" / "orchestration" / "SKILL.md"
    ).read_text(encoding="utf-8")
    for face in ("bench-add", "bench-up", "bench-queue", "bench-seen", "bench-stop"):
        assert face in leader_skill, f"组长技能缺 CLI 面：{face}"


def test_harness_eval_reference_declares_two_channels():
    """``references/harness-eval.md`` 写明两条通道的归属与 ``--runner`` 读法。"""
    text = (
        PROJECT_ROOT
        / "orchestration"
        / "skills"
        / "orchestration"
        / "references"
        / "harness-eval.md"
    ).read_text(encoding="utf-8")
    assert "通道归属" in text, "移交后必读段落缺席：读者无从知道套件归谁"
    section = text.split("## 通道归属", 1)[1].split("\n## ", 1)[0]
    for token in ("orch eval run", "pysci-bench", "leader", "bencher"):
        assert token in section, f"通道归属段缺 {token}（两条通道必须同段对照）"
    assert "runner" in section, "台账字段名要在归属段点名（分开读数的依据）"
    assert "--runner" in text, "set-baseline/report 的通道过滤读法必须写明"


def test_bencher_deployed_copy_matches_source_when_present():
    """副本一旦存在必须与真本同源（charter：禁止手改 pod 内部署副本）。

    副本目录是 gitignored 运行态，合并后须在 main 跑 ``pysci-dev sync`` 才生成
    （devops AGENTS.md 已记录），故缺席时 skip 而非 fail。
    """
    deployed = PODS_ROOT / "bencher" / ".qoder" / "skills" / "bencher" / "SKILL.md"
    if not deployed.exists():
        pytest.skip("副本未部署（合并后 pysci-dev sync 生成）")
    source = (
        PROJECT_ROOT / "orchestration" / "skills" / "bencher" / "SKILL.md"
    ).read_bytes()
    assert deployed.read_bytes() == source, "部署副本与真本不一致（禁止手改副本）"
