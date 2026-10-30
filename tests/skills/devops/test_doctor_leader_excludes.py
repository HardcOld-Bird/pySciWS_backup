"""doctor「leader 规则排除」巡检回归（backlog 20261010-pod-leader-excludes）。

锁定两侧护栏：pod settings 缺 agentsMdExcludes → 组员注入组长专属根规则（✗）；
glob 过宽连 basic.md 一起命中 → 组员丢掉全员公约数（✗）。另钉住真实仓库现状：
每个 pod 的 settings.json 都已带 **/leader-only.md 排除。
"""

from __future__ import annotations

import json

import pytest

from pysci.paths import PODS_ROOT
from pysci.skills.devops.tools import dev

HOOKS = {
    "hooks": {
        "Stop": [{"hooks": [{"name": "delivery-gate"}]}],
        "PreToolUse": [{"hooks": [{"name": "pod-guard"}]}],
    }
}


def _settings(**extra) -> str:
    return json.dumps({**extra, **HOOKS}, ensure_ascii=False)


@pytest.mark.parametrize(
    "excludes",
    [["**/leader-only.md"], ["leader-only.md"], ["**/rules/leader-only.md"]],
)
def test_leader_rule_excluded_ok(excludes):
    assert dev.exclude_problems(_settings(agentsMdExcludes=excludes)) == []


def test_missing_excludes_flagged():
    (msg,) = dev.exclude_problems(_settings())
    assert "leader 规则未排除" in msg and "leader-only.md" in msg


def test_empty_excludes_flagged():
    assert any(
        "leader 规则未排除" in m
        for m in dev.exclude_problems(_settings(agentsMdExcludes=[]))
    )


@pytest.mark.parametrize(
    "excludes", [["**/*.md"], ["**/leader-only.md", "**/basic.md"]]
)
def test_over_broad_glob_flagged(excludes):
    (msg,) = dev.exclude_problems(_settings(agentsMdExcludes=excludes))
    assert "过宽" in msg and "basic.md" in msg


def test_unrelated_glob_still_missing_leader():
    (msg,) = dev.exclude_problems(_settings(agentsMdExcludes=["**/secret-notes.md"]))
    assert "leader 规则未排除" in msg


def test_broken_json_reported():
    (msg,) = dev.exclude_problems('{"hooks": ')
    assert "无法解析" in msg


def test_doctor_pod_line_marks_missing_exclude(tmp_path):
    pod = tmp_path / "sim"
    (pod / ".qoder" / "rules").mkdir(parents=True)
    (pod / ".qoder" / "rules" / "charter.md").write_text("# c", encoding="utf-8")
    (pod / "AGENTS.md").write_text("# m", encoding="utf-8")
    (pod / ".qoder" / "settings.json").write_text(_settings(), encoding="utf-8")
    (line,) = dev._doctor_pod(pod)
    assert "[✗]" in line and "leader 规则未排除" in line

    (pod / ".qoder" / "settings.json").write_text(
        _settings(agentsMdExcludes=["**/leader-only.md"]), encoding="utf-8"
    )
    (line,) = dev._doctor_pod(pod)
    assert "[√]" in line


def test_doctor_pod_flags_leader_named_local_rule(tmp_path):
    pod = tmp_path / "sim"
    rules = pod / ".qoder" / "rules"
    rules.mkdir(parents=True)
    (rules / "charter.md").write_text("# c", encoding="utf-8")
    (rules / "leader-only-notes.md").write_text("# n", encoding="utf-8")
    (pod / "AGENTS.md").write_text("# m", encoding="utf-8")
    (pod / ".qoder" / "settings.json").write_text(
        _settings(agentsMdExcludes=["**/leader-only.md"]), encoding="utf-8"
    )
    (line,) = dev._doctor_pod(pod)
    assert "[✗]" in line and "撞名" in line


@pytest.mark.skipif(not PODS_ROOT.exists(), reason="pods 目录不存在")
def test_repo_pods_all_exclude_leader_rule():
    pods = [p for p in sorted(PODS_ROOT.iterdir()) if p.is_dir()]
    assert pods, "PODS_ROOT 下应至少有一个 pod"
    for pod in pods:
        problems = dev.exclude_problems(
            (pod / ".qoder" / "settings.json").read_text(encoding="utf-8")
        )
        assert problems == [], f"{pod.name}: {problems}"
