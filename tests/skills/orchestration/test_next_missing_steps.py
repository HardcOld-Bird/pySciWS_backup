"""缺 `steps` 的计划状态不得让 `next` 整体崩（backlog 20261015-154311）。

实案：`pysci-orch next --limit 10` → traceback 末行 `KeyError: 'steps'`（栈：`nextroute.cmd_next`
→ `next_lines` → `collect_actions` → `plans._worklist_panes` 的 `state["steps"]`）。
真因：盘上有计划 **YAML 却无 state 文件**（只登记未推进/旧格式）→ `_load_state` 回 `{}` →
读面直接下标。修法：`_load_state` **单点容错**（缺键补空列表，读面按「无环节」跳过）。
"""

from __future__ import annotations

from argparse import Namespace

import yaml

from pysci.skills.orchestration.tools import nextroute, plans


def test_load_state_keeps_falsy_semantics_when_file_is_absent(qface):
    """state 文件缺席 → 仍回 `{}`（**保 falsy**）：多处 `if not state:` 靠它判「计划是否存在」
    （如 `cmd_plan_drop`）；容错落在**读点**而不是这里（backlog 20261015-154311 精修版）。"""
    st = plans._load_state("no-such-plan-xyz")
    assert st == {} and not st, (
        "缺席必须保持 falsy（改成 {'steps': []} 会改掉在册判定）"
    )


def test_crash_sites_tolerate_a_state_without_steps(qface):
    """崩溃点容错：`{}`（无 steps 键）喂给两个读点都不抛。"""
    empty: dict = {}
    assert plans._inflight(empty) == []
    panes = plans._worklist_panes(
        {"id": "p", "goal": "g", "steps": [{"id": "s1", "member": "devops"}]}, empty
    )
    assert panes["inflight"] == [] and panes["need_leader"] == []


def test_load_state_without_steps_key_is_normalized(qface):
    (plans.PLANS_DIR / "odd.state.json").write_text(
        '{"plan_id": "odd", "goal": "g"}\n', encoding="utf-8"
    )
    assert plans._load_state("odd")["steps"] == []


def test_worklist_panes_tolerates_a_state_without_steps(qface):
    """崩溃点本身：`_worklist_panes(plan, {})` 曾 KeyError。"""
    plan = {"id": "p", "goal": "g", "steps": [{"id": "s1", "member": "devops"}]}
    panes = plans._worklist_panes(plan, plans._load_state("no-such-plan-xyz"))
    assert isinstance(panes, dict), "空步骤应产出空工作清单，而不是抛"


def test_next_survives_a_plan_whose_state_file_is_absent(tmp_path, monkeypatch, capsys):
    """端到端：YAML 在册、state 缺席 → `next` 照常出卡片，无 traceback。

    夹具自足（不跨模块取 ``rface``：那样导入夹具会被同名参数遮蔽、ruff F811/F401 反复打架）。
    """
    from pysci import paths as _paths
    from pysci.skills.orchestration.tools import queueworker, taskbooks

    pdir = tmp_path / "plans"
    pdir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    monkeypatch.setattr(queueworker, "wake_member", lambda member, **k: (True, "stub"))
    (pdir / "orphan.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "orphan",
                "goal": "只登记未推进的计划",
                "steps": [{"id": "s1", "member": "devops", "brief": "b"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    assert not (pdir / "orphan.state.json").exists(), "前提：state 缺席"

    nextroute.cmd_next(Namespace(limit=10))  # 曾在此抛 KeyError: 'steps'

    assert "orphan" in capsys.readouterr().out
