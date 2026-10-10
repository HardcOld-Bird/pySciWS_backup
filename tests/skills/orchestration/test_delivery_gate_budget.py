"""delivery-gate 的 harness 预算硬闸回归（backlog 20261010-budget-delivery-gate）。

被校验的是**组员自维护层**，四条：① pod `AGENTS.md` ≤8192B；② `.qoder/rules/` 下非
charter 的 md 不得 `trigger: always_on`；③ 自建 rules 与自建 skills 的 `description` 行
合计**各** ≤8192B；④ 每个自建 `SKILL.md` ≤8192B。违规 ⇒ exit 2 + stderr 给出文件路径、
实测字节、上限与整改动作。

必须同时锁住的两条纪律：既有 `stop_hook_active` 循环防护与 fail-open——预算段自身出异常
（坏 frontmatter、pod 目录不存在、stdin 不是 JSON）**不得**卡死交付；以及「部署副本的尺寸
不该由组员的 Stop 买单」：`PYSCI_DEPLOYED_SKILLS` 缺失或命中清单时，自建技能那一支不执法。

全部用例直接 `node` 跑真 guard，不走模拟实现——测的就是线上接线的那份代码。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT, PODS_ROOT

GUARD = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"
LIMIT = 8192
OK = "结论如上。<result>已交付</result>"


def run_gate(
    pod: Path,
    *,
    msg: str = OK,
    deployed: list[str] | None = None,
    stop_hook_active: bool = False,
    stdin: str | None = None,
) -> tuple[int, str]:
    """以真实 hook 的姿势调用 guard：stdin 收 Stop 事件 JSON，PYSCI_* 走环境注入。"""
    env = dict(os.environ)
    env["PYSCI_POD"] = str(pod)
    env.pop("PYSCI_DEPLOYED_SKILLS", None)
    if deployed is not None:
        env["PYSCI_DEPLOYED_SKILLS"] = ";".join(deployed)
    payload = json.dumps(
        {"last_assistant_message": msg, "stop_hook_active": stop_hook_active}
    )
    proc = subprocess.run(
        ["node", str(GUARD)],
        input=stdin if stdin is not None else payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return proc.returncode, proc.stderr


@pytest.fixture
def pod(tmp_path: Path) -> Path:
    """一个干净的 pod：AGENTS.md 与 rules 都在预算内，无自建技能。"""
    p = tmp_path / "probe"
    (p / ".qoder" / "rules").mkdir(parents=True)
    (p / "AGENTS.md").write_text("# 记忆\n\n稳定事实若干。\n", encoding="utf-8")
    (p / ".qoder" / "rules" / "charter.md").write_text(
        "---\ntrigger: always_on\n---\n\n# charter\n", encoding="utf-8"
    )
    return p


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_clean_pod_with_valid_delivery_passes(pod):
    assert run_gate(pod) == (0, "")


# ===========================================================================
#  既有职责：格式校验与两条放行纪律
# ===========================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_missing_tags_still_rejected(pod):
    rc, err = run_gate(pod, msg="我做完了，没有标签块。")
    assert rc == 2
    assert "交付格式不合规" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_loop_protection_bypasses_everything(pod):
    """stop_hook_active ⇒ 一律放行，**含预算违规**（循环防护优先，防死循环）。"""
    (pod / "AGENTS.md").write_text("x" * (LIMIT + 500), encoding="utf-8")
    assert run_gate(pod, msg="无标签", stop_hook_active=True) == (0, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_unparsable_stdin_fails_open(pod):
    assert run_gate(pod, stdin="{not json")[0] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_missing_pod_directory_fails_open(tmp_path):
    """PYSCI_POD 指向不存在目录 ⇒ 不抛不拦（门禁自身故障不阻塞生产）。"""
    assert run_gate(tmp_path / "nope") == (0, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_format_and_budget_problems_are_reported_in_one_round(pod):
    """两类违规合并成一轮整改：否则格式退回会吃掉唯一一次拦截，预算违规永远漏网。"""
    (pod / "AGENTS.md").write_text("x" * (LIMIT + 1), encoding="utf-8")
    rc, err = run_gate(pod, msg="没有标签块")
    assert rc == 2
    assert "交付格式不合规" in err and "AGENTS.md" in err


# ===========================================================================
#  ① 全量注入档：AGENTS.md
# ===========================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_oversized_agents_md_rejected_with_actionable_detail(pod):
    body = "x" * (LIMIT + 1000)
    (pod / "AGENTS.md").write_text(body, encoding="utf-8")

    rc, err = run_gate(pod)

    assert rc == 2
    text = (pod / "AGENTS.md").stat().st_size
    assert str(text) in err  # 实测字节
    assert "8192" in err  # 上限
    assert "动作：精简" in err  # 整改动作
    assert "AGENTS.md" in err  # 文件路径


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_agents_md_at_the_limit_passes(pod):
    """边界：落盘恰 8192B 不算超（判据是 `> LIMIT`，与 find -size +8192c 同口径）。"""
    (pod / "AGENTS.md").write_text("x" * LIMIT, encoding="utf-8")
    assert run_gate(pod)[0] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_agents_md_size_is_measured_on_disk(pod):
    """CRLF 也要计税：注入的是落盘字节，不是逻辑字符数。"""
    (pod / "AGENTS.md").write_bytes("# 记忆\r\n".encode() + b"y" * LIMIT)
    assert run_gate(pod)[0] == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_missing_agents_md_is_not_a_violation(pod):
    (pod / "AGENTS.md").unlink()
    assert run_gate(pod)[0] == 0


# ===========================================================================
#  ② 自建 rules 禁 always_on（槽位只留 charter）
# ===========================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
@pytest.mark.parametrize(
    "front",
    [
        "---\ntrigger: always_on\n---\n",
        '---\ntrigger: "always_on"\n---\n',
        "---\ntrigger: ALWAYS_ON\n---\n",
        "---\ntrigger: always-on\n---\n",
        "---\ntype: rule\ntrigger: always_on\ndescription: 随手记。\n---\n",
    ],
)
def test_self_authored_always_on_rule_rejected(pod, front: str):
    (pod / ".qoder" / "rules" / "my-notes.md").write_text(
        front + "\n# 笔记\n", encoding="utf-8"
    )

    rc, err = run_gate(pod)

    assert rc == 2
    assert "always_on" in err and "charter" in err and "条件式" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_charter_always_on_is_exempt(pod):
    """charter 是只读层、always_on 槽位的合法持有者——fixture 里就带着它。"""
    assert run_gate(pod)[0] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_conditional_self_rules_pass(pod):
    for name, front in (
        ("note-md.md", "---\ntrigger: glob\nglob: **/*.md\n---\n"),
        ("tip.md", "---\ntrigger: model_decision\ndescription: 做 X 时读。\n---\n"),
        ("manual.md", "---\ntrigger: manual\n---\n"),
    ):
        (pod / ".qoder" / "rules" / name).write_text(
            front + "\n# n\n", encoding="utf-8"
        )
    assert run_gate(pod)[0] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_broken_frontmatter_fails_open(pod):
    """未闭合 / 无 frontmatter 的自建规则不得让 guard 崩掉或误报 always_on。"""
    (pod / ".qoder" / "rules" / "unclosed.md").write_text(
        "---\ntrigger: always_on\n# 忘了闭合\n", encoding="utf-8"
    )
    (pod / ".qoder" / "rules" / "plain.md").write_text(
        "# 没有 frontmatter\n", encoding="utf-8"
    )
    assert run_gate(pod) == (0, "")


# ===========================================================================
#  ③ 常驻暴露档：description 合计（rules 与 skills 各自成账）
# ===========================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_self_rules_description_total_over_budget(pod):
    for name in ("a.md", "b.md"):
        (pod / ".qoder" / "rules" / name).write_text(
            f"---\ntrigger: model_decision\ndescription: {'y' * 5000}\n---\n",
            encoding="utf-8",
        )

    rc, err = run_gate(pod)

    assert rc == 2
    assert "自建 rules 的 description 合计" in err
    assert "10026B" in err  # 2 × (5000 + len("description: "))
    assert "8192" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_self_rules_description_total_under_budget_passes(pod):
    (pod / ".qoder" / "rules" / "a.md").write_text(
        "---\ntrigger: glob\nglob: **/*.tex\ndescription: 写 tex 时读。\n---\n",
        encoding="utf-8",
    )
    assert run_gate(pod)[0] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_folded_description_is_counted(pod):
    """块标量折行的 description 也按落盘字节计入合计（漏计 = 常驻税失控而无人知）。"""
    (pod / ".qoder" / "rules" / "a.md").write_text(
        "---\ntrigger: model_decision\ndescription: >-\n"
        + "".join(f"  {'z' * 70}\n" for _ in range(120))
        + "---\n",
        encoding="utf-8",
    )
    assert run_gate(pod)[0] == 2


# ===========================================================================
#  ③④ 自建技能：每个 SKILL.md 尺寸 + description 合计；部署副本豁免
# ===========================================================================
def _skill(pod: Path, name: str, skill_bytes: int, desc_len: int) -> Path:
    d = pod / ".qoder" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    desc = "d" * desc_len
    head = f"---\nname: {name}\ndescription: {desc}\n---\n"
    (d / "SKILL.md").write_text(
        head + "x" * max(0, skill_bytes - len(head.encode())), encoding="utf-8"
    )
    return d


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_oversized_self_skill_rejected(pod):
    _skill(pod, "my-tool", LIMIT + 400, 40)

    rc, err = run_gate(pod, deployed=["devops", "orchestration"])

    assert rc == 2
    assert "自建技能 my-tool/SKILL.md" in err
    assert "references/" in err and "索引" in err


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_deployed_skill_copy_is_exempt(pod):
    """部署副本超预算是 devops 的责任（真本侧由 doctor 与 pre-commit 管），不该卡组员交付。"""
    _skill(pod, "devops", LIMIT + 400, 40)
    assert run_gate(pod, deployed=["devops"]) == (0, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_deployed_skill_list_is_matched_case_insensitively(pod):
    _skill(pod, "MyTool", LIMIT + 400, 40)
    assert run_gate(pod, deployed=["mytool"]) == (0, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_self_skill_branch_skipped_without_deployed_list(pod):
    """PYSCI_DEPLOYED_SKILLS 缺失 ⇒ 无法区分自建与部署 ⇒ 该支不执法，但 ①② 照常。"""
    _skill(pod, "whatever", LIMIT + 400, 40)
    assert run_gate(pod, deployed=None)[0] == 0

    (pod / "AGENTS.md").write_text("x" * (LIMIT + 1), encoding="utf-8")
    assert run_gate(pod, deployed=None)[0] == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_self_skills_description_total_is_separate_from_rules(pod):
    """两本账各自 ≤8192：rules 合计与 skills 合计都不超，即便两者相加已超。"""
    for name in ("a.md", "b.md"):
        (pod / ".qoder" / "rules" / name).write_text(
            f"---\ntrigger: model_decision\ndescription: {'y' * 3000}\n---\n",
            encoding="utf-8",
        )
    for name in ("s1", "s2"):
        _skill(pod, name, 2000, 3000)

    assert run_gate(pod, deployed=["devops"])[0] == 0

    _skill(pod, "s3", 2000, 3000)  # skills 侧第三笔 → 9039B > 8192
    rc, err = run_gate(pod, deployed=["devops"])
    assert rc == 2
    assert "自建 skills 的 description 合计" in err
    assert "自建 rules" not in err  # rules 那本账仍为 6018B，未被连坐


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_self_skill_references_are_not_measured(pod):
    """按需档只在被指向时进上下文：分册尺寸不由该闸执法（真本侧归 doctor）。"""
    d = _skill(pod, "my-tool", 2000, 40)
    ref = d / "references"
    ref.mkdir()
    (ref / "big.md").write_text("x" * (LIMIT * 3), encoding="utf-8")
    assert run_gate(pod, deployed=["devops"]) == (0, "")


@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
def test_skill_dir_without_skill_md_is_ignored(pod):
    d = pod / ".qoder" / "skills" / "scratch"
    d.mkdir(parents=True)
    (d / "notes.md").write_text("x" * (LIMIT * 2), encoding="utf-8")
    assert run_gate(pod, deployed=["devops"]) == (0, "")


# ===========================================================================
#  真实仓库现状：每个 pod 都应通过（哨兵，防未来漂移）
# ===========================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="需要 node 运行 guard")
@pytest.mark.skipif(not PODS_ROOT.is_dir(), reason="无 pods 目录")
def test_repo_pods_all_pass_the_gate():
    offenders = []
    for p in sorted(PODS_ROOT.iterdir()):
        if not (p / "AGENTS.md").exists():
            continue
        deployed = (
            [d.name for d in (p / ".qoder" / "skills").iterdir() if d.is_dir()]
            if (p / ".qoder" / "skills").is_dir()
            else []
        )
        rc, err = run_gate(p, deployed=deployed)
        if rc != 0:
            offenders.append((p.name, err.strip()))
    assert not offenders, offenders
