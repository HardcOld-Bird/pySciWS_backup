# 任务书：组长引导面修缮（status/plan [NEXT] 文案 + command-surface 补注 + 夜航速查入技能真本）

背景：组长实测一轮大任务（tb-20261015-153336-2）暴露的上手/收尾摩擦，逐条修复。
各条独立可验，做完一条勾一条。

## ① `orch status` 增夜航读数（代码）
现状：status 显示模型预设/路由模式，**不显示夜航开关**——组长只能 grep
`data/pysci_settings.toml [night]` 才知道在不在夜航。
要求：成员/队列区之上加一行，读 `[night] enabled`（fail-open，读不出显示
「夜航：未知」），on 时附一行行为提示（决策留痕、AskUserQuestion 拦截→twin、
细则 references/patrol-watchdog.md §5）。
验收：enabled=true/false 两态下 status 各自正确显示；settings 缺失/损坏不崩。

## ② plan 收尾 [NEXT] 补 taskbooks 联动提示（代码）
现状：plan 全部环节完成的 [NEXT] 只写「向用户交付汇总」；taskbook 联动要靠
组长记得 `taskbooks sync` + `taskbooks report <id>`（实测漏了就得回头补）。
要求：plan 完成时若该 plan 被 taskbook 关联（link 过），[NEXT] 直接给出
`pysci-orch taskbooks sync` 与 `pysci-orch taskbooks report <tb-id>` 两条命令
（含正确 id）。
验收：关联/未关联两种 plan 完成时的 [NEXT] 输出各一例（未关联不加噪声）。

## ③ blocked×capacity 的 [NEXT] 提示 --session new（代码）
现状：交付 blocked 的 [NEXT] 写「续作自动复用本环节会话」——但 capacity 交回时
原会话余量已尽，SKILL.md 派发要领明文「续作必须 --session new」，两者打架，
照 [NEXT] 做就会派进死会话（实测 2026-10-15 adhoc-20261015-154311 首跳）。
要求：blocked 交付正文含「capacity/余量」类字样（或 tags 含 capacity）时，
[NEXT] 的续作命令改为 `plan run <id> --session new --task …` 并加一句原因。
验收：构造 capacity 交回样例 → [NEXT] 提示 --session new；普通 blocked 不变。

## ④ command-surface.md 补注（文档，真本
orchestration/skills/orchestration/references/command-surface.md）
- `plan run`/`plan advance` 旗标表补全：--session/--name/--model-tier/--effort/
  --max-turns/--dirs/--dry（以 -h 实测为准）；
- `plan new` 语义注：id 默认取文件名 stem（建议显式 --id）；登记时**复制** YAML 入
  `state/plans/<id>.plan.yaml`，源文件登记后可删（实测曾留冗余源文件）。

## ⑤ 夜航速查入技能真本（文档，orchestration/skills/orchestration/SKILL.md）
在命令面速查之后加一小块（≤10 行，注意 SKILL.md ≤8KB 预算——超了就压缩别处或
把细节移 references/ 并留一行索引）：
- 开关真源：`data/pysci_settings.toml [night] enabled`（热切换；`orch status` 可读）；
- 生效差异：SessionStart/PostCompact 注入政策全文＝**限时预授权**（决策权等同用户、
  事后追溯、留痕）；AskUserQuestion 被机械拦截 → 该问的转 `consult-twin`/twin 终裁；
  不得因等待用户而停滞观望；
- 终态两态：全部完成 / 严重阻碍——收尾写总结并把 enabled 改回 false；
- 细则一行索引：references/patrol-watchdog.md §5。

## ⑥ 核验 SessionStart 夜航注入链（调查 + 可能立缺陷）
组长实测：本会话开启夜航后**从未收到政策注入**，但手工跑
`printf '%s' '{"hook_event_name":"SessionStart","session_id":"t"}' | PYTHONPATH=src
.venv/Scripts/python.exe -m pysci.skills.orchestration.tools.night --settings
data/pysci_settings.toml` 输出正常（additionalContext 在）。
核验钩子触发链（.qoder/hooks/night-policy.sh × SessionStart）在「enabled=true +
新会话」下是否真把注入送进上下文；若钩子未触发/输出被吞，查明是竞态（用户先开会话
后拨开关）还是 harness 侧丢注入，属缺陷则随本单调修（改不了 harness 就
`<infra_suggestion kind="defect">` 留证上报）。

## 白名单（写权限）
- src/pysci/skills/orchestration/tools/
- tests/
- orchestration/skills/orchestration/（技能真本，改后 `pysci-dev sync` 部署）

## 验收预期
①–⑤ 逐条自证；⑥ 给出核验结论（含复现命令与输出）；全量 pytest 绿。
