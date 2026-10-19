"""任务书轻量 lint（backlog 20261010-084105-devops）。

组长手写任务书易把错误 CLI 范式传染给照抄的组员；write_taskbook 落盘前按规则表扫
正文，命中仅告警不阻断。规则表 state/taskbook-lint.json（devops 可增补）。与
20261009-taskbook-cli-signature（charter/文档侧统一）互补：那里改正本，这里派发口拦截。
"""

from __future__ import annotations

import json
import re

import pytest

from pysci.skills.orchestration.tools import dispatch, plans

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
    """仓库随附规则表可解析、每条规则含 re+hint 且正则可编译（absent 可选亦须可编译）。

    护栏：devops 增补规则若写坏 JSON 或非法正则，fail-open 会**静默失效**——本测试
    在 CI 层面把「随附规则必须真的能触发」钉住（读真实 LINT_RULES_PATH）。
    """
    rules = dispatch.load_lint_rules()
    assert rules, "随附 taskbook-lint.json 应至少含一条规则"
    for r in rules:
        assert r.get("re") and r.get("hint"), f"规则缺 re/hint：{r}"
        pat = re.compile(r["re"])  # 非法正则会抛，测试即失败
        if r.get("absent"):
            re.compile(r["absent"])
        if r.get("steps"):
            assert pat.groups >= 1, f"steps 规则必须捕获环节 id：{r}"


# ---------------------------------------------------------------------------
# absent 条件（backlog 20261014-012859-devops）：re 命中且 absent 不命中才告警
# ---------------------------------------------------------------------------
RUNTIME_DIR_RULE = {
    "id": "new-runtime-dir-needs-gitignore",
    "re": (
        "(?:新建|新增|创建)[^\\n]{0,30}orchestration/state/"
        "|orchestration/state/[A-Za-z0-9_.-]+/[^\\n]{0,30}(?:新建|新增|创建|运行态)"
    ),
    "absent": "\\.gitignore",
    "hint": "白名单含运行态目录但未提 .gitignore",
}


def test_absent_rule_flags_when_gitignore_missing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [RUNTIME_DIR_RULE])
    warns = dispatch.lint_taskbook(
        "白名单：src/**、orchestration/state/queues/（运行态新建）"
    )
    assert len(warns) == 1
    assert warns[0].startswith("[new-runtime-dir-needs-gitignore]")


def test_absent_rule_silent_when_gitignore_present(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [RUNTIME_DIR_RULE])
    body = "白名单：orchestration/state/queues/（运行态新建）、.gitignore"
    assert dispatch.lint_taskbook(body) == []


def test_absent_rule_silent_on_plain_state_path_reference(tmp_path, monkeypatch):
    """**防噪声**：普通 state 路径引用（非「新建/运行态」语境）不触发——否则几乎每单都告警。"""
    _seed(tmp_path, monkeypatch, [RUNTIME_DIR_RULE])
    body = "证据：orchestration/state/suggestions/approved/20261013-071759-devops.md"
    assert dispatch.lint_taskbook(body) == []


def test_absent_rule_silent_when_no_state_dir(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [RUNTIME_DIR_RULE])
    assert dispatch.lint_taskbook("白名单：src/**、tests/**") == []


def test_bad_absent_regex_skips_rule_only(tmp_path, monkeypatch):
    _seed(
        tmp_path,
        monkeypatch,
        [
            {"id": "bad-absent", "re": "queues/", "absent": "([unclosed", "hint": "x"},
            FIGURES_RULE,
        ],
    )
    # 非法 absent 只跳过该条；其余规则照常
    warns = dispatch.lint_taskbook("queues/ 与 pysci-figures build gain_ep fig1_x")
    assert len(warns) == 1 and "figdir" in warns[0]


def test_repo_rule_flags_new_runtime_dir_without_gitignore():
    """随附规则：新运行态目录的任务书（正文无 .gitignore）应被点名告警。"""
    warns = dispatch.lint_taskbook(
        "白名单：src/**、orchestration/state/queues/（运行态新建）"
    )
    assert any("new-runtime-dir-needs-gitignore" in w for w in warns)


def test_repo_rule_silent_when_gitignore_present():
    warns = dispatch.lint_taskbook(
        "白名单：orchestration/state/queues/（运行态新建）、.gitignore"
    )
    assert not any("new-runtime-dir-needs-gitignore" in w for w in warns)


# ---------------------------------------------------------------------------
# steps 判据（tui-fixup f3；背景：m2 占位文案承诺「m3 补」×4 无人兑现）：
# 环节 id 承诺必须能在**对应计划 YAML 的 steps** 里找到；引述段不计；定位不到计划 fail-open
# ---------------------------------------------------------------------------
PLAN_RULE = {
    "id": "placeholder-step-promise",
    "re": r"(?<![A-Za-z0-9])([A-Za-z]{1,4}\d{1,2}[a-z]?)\s*(?:补|充实|建)",
    "steps": True,
    "hint": "环节承诺不可溯",
}


def _seed_plan(tmp_path, monkeypatch, plan_id, steps):
    """铺一个最小计划 YAML，并把 plans.PLANS_DIR 指到 tmp（判据经它定位计划）。"""
    d = tmp_path / "plans"
    d.mkdir(parents=True, exist_ok=True)
    body = "goal: g\nsteps:\n" + "".join(
        f"  - id: {s}\n    member: devops\n" for s in steps
    )
    (d / f"{plan_id}.plan.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    return d


def test_step_promise_flags_unregistered_step(tmp_path, monkeypatch):
    """含「mN 补」却引用了计划里不存在的环节 → 告警，并附计划 id 供核对。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1", "f2"])
    warns = dispatch.lint_taskbook(
        "后续单：待决模块细节 m9 补。", plan_id="fix-20261014"
    )
    assert len(warns) == 1
    assert warns[0].startswith("[placeholder-step-promise]")
    assert "fix-20261014" in warns[0] and "m9" in warns[0]


def test_step_promise_silent_when_step_exists(tmp_path, monkeypatch):
    """引用真实存在的环节 → 零告警（零误报的正向面）。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["m1", "m3"])
    assert dispatch.lint_taskbook("细节 m3 补。", plan_id="fix-20261014") == []


def test_step_promise_ignores_quoted_samples(tmp_path, monkeypatch):
    """引述段（「」/''）是被引用的样例或 grep 判据字面量，不算承诺 → 不告警。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    body = "缺陷：「（细节 m3 补）」×4；判据 `grep 'm3 补' src/` 零命中。"
    assert dispatch.lint_taskbook(body, plan_id="fix-20261014") == []


def test_step_promise_fail_open_without_resolvable_plan(tmp_path, monkeypatch):
    """定位不到对应计划（无注册计划）→ fail-open 不告警（无法核对就不误报）。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    monkeypatch.setattr(plans, "PLANS_DIR", tmp_path / "nope")
    assert dispatch.lint_taskbook("细节 m9 补。") == []


def test_step_promise_resolves_plan_from_body(tmp_path, monkeypatch):
    """未显式给 plan_id 时，从正文里出现的真实计划 id 定位（多命中取最长）。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    warns = dispatch.lint_taskbook("（计划 fix-20261014 环节 f1）后续 m9 补。")
    assert len(warns) == 1 and "fix-20261014" in warns[0]


def test_step_promise_zero_false_positive_on_historical_shapes(tmp_path, monkeypatch):
    """历史任务书形态回放：引述样例 / 引用真实环节 / 无承诺 → 全部零告警。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["m3", "m4", "m5"])
    bodies = [
        "占位未兑现：留着「（细节 m3 补）」×4、「（占位）…m5 充实」——均未含兑现条款。",
        "- 「计划与派发」详情放占位（m3/m5 充实，如实标注「占位」）；",
        "3. **m4 建** 已收官（环节状态 read 自 plan YAML）；",
        "验收：`grep 'm3 补\\|m5 充实\\|m5 建' src/` 零命中。",
        "细节见 references，无任何环节承诺。",
    ]
    for b in bodies:
        assert dispatch.lint_taskbook(b, plan_id="fix-20261014") == [], b


def test_plan_from_slug_strips_trailing_step(tmp_path, monkeypatch):
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    assert dispatch._plan_from_slug("fix-20261014-f1") == "fix-20261014"
    assert dispatch._plan_from_slug("unknown-slug") is None


def test_write_taskbook_uses_slug_to_resolve_plan(tmp_path, monkeypatch, capsys):
    """落盘路径：write_taskbook 由 slug 定位计划，承诺了不存在的环节即告警（不阻断落盘）。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    dest = dispatch.write_taskbook(
        tmp_path / "pod", text="后续 m9 补。", task_file=None, slug="fix-20261014-f1"
    )
    out = capsys.readouterr().out
    assert "任务书 lint" in out and "fix-20261014" in out and "m9" in out
    assert dest.exists()


# ---------------------------------------------------------------------------
# 叙述性用法不告警（2026-10-14 误报精化：「自 m4 建成后」被当成环节承诺）
# ---------------------------------------------------------------------------
def test_step_promise_ignores_completed_narrative(tmp_path, monkeypatch):
    """「自 m4 **建成**后长期空转」是讲史：m4 不在本计划 steps 也**不告警**。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    body = "事故根因（已取证）：`orchestration/state/queues/` 自 m4 建成后从未有条目。"
    assert dispatch.lint_taskbook(body, plan_id="fix-20261014") == []


@pytest.mark.parametrize(
    "body",
    [
        "面板 m4 建成后不再改动",
        "细节 m3 补完（留档）",
        "正文 m5 充实了 之后没人读",
        "自 m4 补 上线后无人接管",  # 前缀「自」叙述（动词无完成态也该跳）
    ],
)
def test_step_promise_ignores_narrative_forms(tmp_path, monkeypatch, body):
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    assert dispatch.lint_taskbook(body, plan_id="fix-20261014") == [], body


@pytest.mark.parametrize(
    "body",
    [
        "后续单：细节 m9 补。",  # 未来承诺（动词不带完成态、无叙述前缀）
        "面板 m4 建立新入口后交接",
        "正文 m7 充实读数后交付",
    ],
)
def test_step_promise_still_flags_future_promises(tmp_path, monkeypatch, body):
    """精化**不许把规则废掉**：未来承诺（mN 补/充实/建 + 宾语）照常告警。"""
    _seed(tmp_path, monkeypatch, [PLAN_RULE])
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    warns = dispatch.lint_taskbook(body, plan_id="fix-20261014")
    assert len(warns) == 1 and "placeholder-step-promise" in warns[0], body


# ---------------------------------------------------------------------------
# whitelist 判据（backlog 20261014-113347-tier-a）：白名单须覆盖正文点名的「改动目标」
# ——不覆盖的路径派发时会被 pod-guard 拦成 whitelist_conflict（白烧一跳）
# ---------------------------------------------------------------------------
WL_RULE = {
    "id": "whitelist-covers-edits",
    "re": "白名单",
    "whitelist": True,
    "hint": "白名单未覆盖正文点名的改动目标",
}


def test_whitelist_flags_edit_target_outside_list(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [WL_RULE])
    body = (
        "## 白名单（涉改文件）\n\n`src/a.py`、`tests/**`\n\n"
        "## 目标\n\n- 修改 `orchestration/guards/x.mjs` 的告警文案。\n"
    )
    warns = dispatch.lint_taskbook(body)
    assert len(warns) == 1
    assert warns[0].startswith("[whitelist-covers-edits]")
    assert "orchestration/guards/x.mjs" in warns[0]


def test_whitelist_silent_when_target_covered(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [WL_RULE])
    body = "## 白名单（涉改文件）\n\n`src/**`、`tests/**`\n\n## 目标\n\n- 修改 `src/x.py`。\n"
    assert dispatch.lint_taskbook(body) == []


def test_whitelist_ignores_plain_evidence_mentions(tmp_path, monkeypatch):
    """无改动动词的行（证据/背景引用）不是「改动目标」→ 不告警（防噪声）。"""
    _seed(tmp_path, monkeypatch, [WL_RULE])
    body = (
        "## 白名单（涉改文件）\n\n`src/a.py`\n\n"
        "证据：`orchestration/state/suggestions/approved/x.md`\n"
    )
    assert dispatch.lint_taskbook(body) == []


def test_whitelist_fail_open_without_section(tmp_path, monkeypatch):
    """未声明白名单 → 无从判断（fail-open 不告警）。"""
    _seed(tmp_path, monkeypatch, [WL_RULE])
    assert dispatch.lint_taskbook("修改 `orchestration/guards/x.mjs`。") == []


def test_whitelist_section_lines_are_not_body_targets(tmp_path, monkeypatch):
    """白名单段自身的路径不算「正文点名」——否则任何白名单都自证违规。"""
    _seed(tmp_path, monkeypatch, [WL_RULE])
    body = "## 白名单（涉改文件）\n\n新增 `src/a.py`、`tests/b.py`\n"
    assert dispatch.lint_taskbook(body) == []


def test_whitelist_ignores_negated_lines(tmp_path, monkeypatch):
    """否定语境（「是部署副本……**不**手改」）不是改动目标——m7 任务书实况（防误报）。"""
    _seed(tmp_path, monkeypatch, [WL_RULE])
    body = (
        "## 白名单（涉改文件）\n\n`orchestration/skills/**`\n\n"
        "## 说明\n\n- `.qoder/skills/orchestration/` 是部署副本（orch sync 再生），不手改。\n"
    )
    assert dispatch.lint_taskbook(body) == []


def test_whitelist_entry_with_cjk_note_still_covers(tmp_path, monkeypatch):
    """条目带中文注记（`data/taskbooks/**`（新建））仍按路径解析——m3 任务书实况。"""
    _seed(tmp_path, monkeypatch, [WL_RULE])
    body = (
        "## 白名单（涉改文件）\n\n`src/**`、`data/taskbooks/**`（新建）\n\n"
        "## 目标\n\n- 新增 `data/taskbooks/x.taskbook.md` 落盘。\n"
    )
    assert dispatch.lint_taskbook(body) == []


def test_repo_rule_whitelist_covers_edits_engages_on_shipped_table():
    """随附规则：真实 LINT_RULES_PATH 里该规则真的会触发（不是空转）。"""
    body = (
        "## 白名单（涉改文件）\n\n`src/**`\n\n"
        "## 目标\n\n- 删除 `orchestration/guards/pod-guard.mjs` 的旧分支。\n"
    )
    warns = dispatch.lint_taskbook(body)
    assert any("whitelist-covers-edits" in w for w in warns)


# ---------------------------------------------------------------------------
# reference-image-needs-semantic-alignment（返工1 M2）：任务书引用参考图路径却无
# 「参考材料语义对齐」段 → 告警；有该段 → 静默；非参考图语境的 figures/artwork → 不误报
# ---------------------------------------------------------------------------
REFIMG_RULE = {
    "id": "reference-image-needs-semantic-alignment",
    "re": (
        "绘图参考|参考图|artwork[/\\\\]|(?:figures|references)[/\\\\]"
        "[^\\s，。；、）)]*\\.(?:png|jpe?g|jpeg|svg|pdf|eps)"
    ),
    "absent": "语义对齐|逐要素解构|语义解构",
    "hint": "引用参考图却无「参考材料语义对齐」段",
}


def test_refimg_flags_reference_without_semantic_section(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [REFIMG_RULE])
    warns = dispatch.lint_taskbook("输入：`绘图参考/4ref2.png`（配色真源）")
    assert len(warns) == 1
    assert warns[0].startswith("[reference-image-needs-semantic-alignment]")


def test_refimg_silent_when_semantic_section_present(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [REFIMG_RULE])
    body = "## 参考材料语义对齐\n- 全集：`绘图参考/4ref2.png`；逐要素解构表见下"
    assert dispatch.lint_taskbook(body) == []


@pytest.mark.parametrize(
    "body",
    [
        "跑 `pysci-figures build fig1_ep_band` 出图",  # figures 作 CLI 名，非路径
        "输出目录 `article/figures/beam_scatter/`",  # 目录引用，无图像扩展名
        "grep -rn 'figures' src/pysci",  # figures 作检索词
    ],
)
def test_refimg_silent_on_non_reference_usage(tmp_path, monkeypatch, body):
    """**防噪声**：figures 作 CLI 名/目录/检索词（无图像文件）不触发。"""
    _seed(tmp_path, monkeypatch, [REFIMG_RULE])
    assert dispatch.lint_taskbook(body) == [], body


@pytest.mark.parametrize(
    "body",
    [
        "参考材料：`2ref.png` 见 `artwork/final.png`",  # artwork 图像路径
        "对照 data/research/1_gain_ep/article/figures/x/figure.png",  # figures 下图像
        "见参考图中的方框语义",  # 「参考图」字面
    ],
)
def test_refimg_flags_various_reference_forms(tmp_path, monkeypatch, body):
    _seed(tmp_path, monkeypatch, [REFIMG_RULE])
    warns = dispatch.lint_taskbook(body)
    assert len(warns) == 1 and "reference-image-needs-semantic-alignment" in warns[0], (
        body
    )


def test_repo_rule_refimg_engages_without_semantic_section():
    """随附规则：真实 LINT_RULES_PATH 里引用参考图而无语义段 → 真的告警。"""
    warns = dispatch.lint_taskbook("输入：`绘图参考/4ref2.png`（仅作配色参考）")
    assert any("reference-image-needs-semantic-alignment" in w for w in warns)


def test_repo_rule_refimg_silent_with_semantic_section():
    body = (
        "## 参考材料语义对齐\n\n- 全集：`绘图参考/4ref2.png`\n"
        "- 逐要素解构表：内框→场区位置标记→图内几何\n"
    )
    warns = dispatch.lint_taskbook(body)
    assert not any("reference-image-needs-semantic-alignment" in w for w in warns)


# ---------------------------------------------------------------------------
# title-n-measured（backlog 20261020-000717-reviewer）：标题「…」（n=K）的 K
# 须等于标题**非空白字符数**；w3/w6/w11 实测分别为 16/8/9（原写 17/7/10）
# ---------------------------------------------------------------------------
TITLE_N_RULE = {
    "id": "title-n-measured",
    "re": r"标题「([^」\n]+)」\s*[（(][^）\n]{0,80}n\s*=\s*(\d+)",
    "title_n": True,
    "hint": "标题 n 须按非空白字符实测登记",
}


@pytest.mark.parametrize(
    ("title", "declared", "measured"),
    [
        ("理论背景：模式耦合理论（CMT）", 17, 16),
        ("增益EP点的性质", 7, 8),
        ("未来计划和预期成果", 10, 9),
    ],
)
def test_title_n_flags_historical_typos(
    tmp_path, monkeypatch, title, declared, measured
):
    _seed(tmp_path, monkeypatch, [TITLE_N_RULE])
    warns = dispatch.lint_taskbook(f"标题「{title}」（n={declared}→右缘 4.469）")
    assert len(warns) == 1 and "title-n-measured" in warns[0]
    assert f"实测 n={measured}" in warns[0]


@pytest.mark.parametrize(
    ("title", "n"),
    [
        ("理论背景：模式耦合理论（CMT）", 16),
        ("增益EP点的性质", 8),
        ("未来计划和预期成果", 9),
    ],
)
def test_title_n_silent_when_measured(tmp_path, monkeypatch, title, n):
    _seed(tmp_path, monkeypatch, [TITLE_N_RULE])
    assert dispatch.lint_taskbook(f"标题「{title}」（n={n}→右缘 4.129）") == []


def test_title_n_ignores_titles_without_n(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [TITLE_N_RULE])
    assert dispatch.lint_taskbook("标题「增益EP点的性质」按预设排。") == []


def test_title_n_hint_reports_expected_right_edge(tmp_path, monkeypatch):
    """告警附实测值与应得右缘（0.34n+1.069），便于当场改对。"""
    _seed(tmp_path, monkeypatch, [TITLE_N_RULE])
    warns = dispatch.lint_taskbook("标题「增益EP点的性质」（n=7→右缘 3.449）")
    assert "实测 n=8" in warns[0] and "3.789" in warns[0]


def test_title_n_flags_declaration_on_next_line(tmp_path, monkeypatch):
    """w3 实况：标题独占一行，n 写在下一行括号里。"""
    _seed(tmp_path, monkeypatch, [TITLE_N_RULE])
    body = (
        "标题「理论背景：模式耦合理论（CMT）」\n"
        "   （装饰三层；n=17→右缘 6.849；accent 细节）。\n"
    )
    warns = dispatch.lint_taskbook(body)
    assert len(warns) == 1 and "实测 n=16" in warns[0] and "6.509" in warns[0]


def test_repo_rule_title_n_engages_on_historical_typo():
    """随附规则：真实 LINT_RULES_PATH 能拦下 w6 的 n=7（实测 8）。"""
    warns = dispatch.lint_taskbook("标题「增益EP点的性质」（n=7→右缘 3.449）")
    assert any("title-n-measured" in w for w in warns)
