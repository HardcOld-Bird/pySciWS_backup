"""任务书轻量 lint（backlog 20261010-084105-devops）。

组长手写任务书易把错误 CLI 范式传染给照抄的组员；write_taskbook 落盘前按规则表扫
正文，命中仅告警不阻断。规则表 state/taskbook-lint.json（devops 可增补）。与
20261009-taskbook-cli-signature（charter/文档侧统一）互补：那里改正本，这里派发口拦截。
"""

from __future__ import annotations

import json
import re

from pysci.skills.orchestration.tools import dispatch

# 与仓库随附 taskbook-lint.json 首期规则同形（复制于此，单元测试不依赖仓库文件）
FIGURES_RULE = {
    "id": "figures-two-positional-args",
    "re": "pysci-figures\\s+(?:build|preview|audit)\\s+\\w+\\s+fig\\w+",
    "hint": "只接受单一位置参数 <figdir>",
}


def _seed(tmp_path, monkeypatch, rules):
    p = tmp_path / "taskbook-lint.json"
    p.write_text(
        json.dumps({"version": 1, "rules": rules}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", p)
    return p


def test_flags_figures_two_positional(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [FIGURES_RULE])
    warns = dispatch.lint_taskbook("请跑 pysci-figures build gain_ep fig1_ep_band 出图")
    assert len(warns) == 1
    assert "figdir" in warns[0]
    assert warns[0].startswith("[figures-two-positional-args]")


def test_flags_preview_and_audit(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [FIGURES_RULE])
    assert dispatch.lint_taskbook("pysci-figures preview gain_ep fig1_x")
    assert dispatch.lint_taskbook("pysci-figures audit theory fig2_y")


def test_clean_single_positional_not_flagged(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [FIGURES_RULE])
    # 正确形态：单一 figdir + flag，不告警
    assert (
        dispatch.lint_taskbook("pysci-figures build fig1_ep_band --width double") == []
    )


def test_clean_new_two_positional_is_legit(tmp_path, monkeypatch):
    """new 本就是两位置参数命令（research slug），规则只针对 build/preview/audit。"""
    _seed(tmp_path, monkeypatch, [FIGURES_RULE])
    assert dispatch.lint_taskbook("pysci-figures new gain_ep fig1_ep_band") == []


def test_missing_rules_file_fail_open(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", tmp_path / "nope.json")
    assert dispatch.lint_taskbook("pysci-figures build a fig1") == []


def test_corrupt_rules_file_fail_open(tmp_path, monkeypatch):
    p = tmp_path / "taskbook-lint.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", p)
    assert dispatch.lint_taskbook("pysci-figures build a fig1") == []


def test_bad_regex_skipped_others_still_run(tmp_path, monkeypatch):
    _seed(
        tmp_path,
        monkeypatch,
        [
            {"id": "bad", "re": "([unclosed", "hint": "never"},  # 非法正则→跳过
            FIGURES_RULE,  # 合法→仍生效
        ],
    )
    warns = dispatch.lint_taskbook("pysci-figures build gain_ep fig1_x")
    assert len(warns) == 1 and "figdir" in warns[0]


def test_load_rules_filters_malformed(tmp_path, monkeypatch):
    p = tmp_path / "taskbook-lint.json"
    p.write_text(
        json.dumps(
            {
                "rules": [
                    {"re": "a", "hint": "keep"},
                    {"hint": "no re field"},  # 丢：缺 re
                    "not-a-dict",  # 丢
                    42,  # 丢
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dispatch, "LINT_RULES_PATH", p)
    assert dispatch.load_lint_rules() == [{"re": "a", "hint": "keep"}]


def test_write_taskbook_warns_but_still_writes(tmp_path, monkeypatch, capsys):
    """告警不阻断：命中仍落盘任务书，且以换行结尾（不撞 end-of-file-fixer）。"""
    _seed(tmp_path, monkeypatch, [FIGURES_RULE])
    pod = tmp_path / "pod"
    dest = dispatch.write_taskbook(
        pod,
        text="跑 pysci-figures build gain_ep fig1_ep_band",
        task_file=None,
        slug="s",
    )
    out = capsys.readouterr().out
    assert "任务书 lint" in out and "figdir" in out
    assert dest.exists()
    raw = dest.read_text(encoding="utf-8")
    assert raw.endswith("\n") and not raw.endswith("\n\n")


def test_write_taskbook_clean_no_warning(tmp_path, monkeypatch, capsys):
    _seed(tmp_path, monkeypatch, [FIGURES_RULE])
    dispatch.write_taskbook(
        tmp_path / "pod",
        text="pysci-figures build fig1_ep_band",
        task_file=None,
        slug="s",
    )
    assert "任务书 lint" not in capsys.readouterr().out


def test_repo_shipped_rules_file_valid():
    """仓库随附规则表可解析、每条规则含 re+hint 且正则可编译。

    护栏：devops 增补规则若写坏 JSON 或非法正则，fail-open 会**静默失效**——本测试
    在 CI 层面把「随附规则必须真的能触发」钉住（读真实 LINT_RULES_PATH）。
    """
    rules = dispatch.load_lint_rules()
    assert rules, "随附 taskbook-lint.json 应至少含一条规则"
    for r in rules:
        assert r.get("re") and r.get("hint"), f"规则缺 re/hint：{r}"
        re.compile(r["re"])  # 非法正则会抛，测试即失败
