# twin 判断通道与 mirror 用户控制台（E6-prep，REFACTOR §9/§10）

本册是**机械面**的操作说明。设计权威是 `orchestration/REFACTOR.md` §9（twin 三重角色）
与 §10（mirror 控制台）；本体系当前处于 E6-prep：**代码就位、通道休眠**——
`PREFERENCE.md` 仍是草案 v1，生效确认权与上线宣告权都在用户（§9.4），组长不得代置。

## 1. 一句话状态判读

```
pysci-mirror status        # 用户侧全景：待盲审/宪章状态/taste_enabled/队列水位/待决事项
```

输出里 `taste_enabled = False` 就是「twin 还没被授权对外说话」；`orch taste` 在此状态下
**拒绝派发**并打印原因，这是设计而不是故障。

## 2. `orch taste <案卷.md>`——审美孪生的单跳判断

```
uv run pysci-orch taste state/evals/cases/xxx.md --dry     # 只看注入内容，零成本
uv run pysci-orch taste state/evals/cases/xxx.md           # 真派发（须 taste_enabled=true）
uv run pysci-orch taste <案卷.md> --member twin --model-tier flash
```

- **案卷格式与 `orch jury` 同源**（`## 问题` / `## 选项 A` / `## 选项 B` [ `/ ## 证据` ]，
  见 `references/jury-panel.md`）：同一份案卷既能送客观仲裁，也能送审美终裁，
  §5.5 的「模棱两可回退链」才接得上；
- **退出码**：0=已表态并落盘；1=派发失败或读不出选择（仍留痕，看得见「跑过但没读出来」）；
  2=未生效 / 案卷不可用 / 宪章缺失；
- **留痕**：`state/evals/taste/<case-id>.json`——`{choice, reason, confidence}` 加会话 sid、
  材料哈希、轮数/耗时/ctx/est_tokens，供 `mirror review` 盲审回放与 `status` 计数；
- **成本**：一跳 flash 档、`max_turns 8` / `timeout 900 s`（registry 的 twin 条目），
  `eval_hop=True` 使判断跳**不进生产台账**、不塞建议队列（与压测跳同一隔离语义）。

### 何时该用（当前阶段的裁决边界）

| 局面 | 现在 | E6 生效后 |
|---|---|---|
| 客观通道（实测/jury）读数模棱两可 | 组长自己裁或上报用户 | 回退 twin 终裁，结果必为 A/B 之一 |
| 「好不好看/好不好用」类审美面 | 组长凭与用户的既有共识裁，或直接向用户提请二选一 | `orch taste` 代用户表态，留痕可查 |
| 最终交付验收 | **永不**由 twin 代理（§9） | 同左，代理范围不含验收 |

`PREFERENCE.md` 生效之前，**不要**为了「先跑起来」把 `taste_enabled` 置 true——误开的代价
是 twin 以用户名义对外说话，而用户从未确认过那份宪章。

## 3. 无记忆纪律的三道闸（为什么 twin 不会「记得」上次判断）

1. `orch taste` 只以 `session="new"` 调 `do_dispatch`；
2. registry 成员字段 `cold_start` 使 `do_dispatch` 在**任何外呼之前**拒绝非 new 会话
   （`code=2`，`error` 含 `cold_start`）——写在 charter 里劝一句迟早会被忘，纪律落成 exit；
3. twin 的 `AGENTS.md` 固定为最小占位且列入 `readonly_extra`，自维护层无法积累「以往判断
   的印象」。

twin pod 的写权限面：`PREFERENCE.md` 与 charter/settings/mcp/inbox 同属只读层
（`registry.readonly_extra` → `build_env` → `PYSCI_READONLY` → pod-guard exit 2）。宪章的修订
链是 **用户盲审纠正 → 偏差报告绿色通道 → devops 实施**，不由 twin 自己动笔。

**twin 是刻意的零技能 pod**（不进 `manifest.toml` targets，同 `reviewer` 先例）：它无领域技能
真本可部署，判断输入只有「宪章 + 案卷」，由 `orch taste` 直接注入。给它挂任何领域技能都会把
绘图/写作的方法论混进审美判断，且 manifest 的 target 必须指向存在的真本目录，凭空造一个
`twin/` 技能册违反「不创建未被要求的文件」。此判断由
`test_twin_intentionally_has_no_manifest_target` 钉住，后续会话不必再猜。

## 4. `pysci-mirror`——用户侧唯一关注入口

```
pysci-mirror status                      # 全景概览（§10 注册纪律：新增关注事项类型必须注册进这里）
pysci-mirror journal "我讨厌为了格式再建一层抽象"      # 交互编号菜单分流
pysci-mirror journal "<口述>" --select 1              # 非交互（脚本/测试）必须明示
```

`journal` 的**写入时机械分流**（§10）：

| 编号 | 语义 | 落点 | 是否要求行动 |
|---|---|---|---|
| ① | 偏好陈述 | `state/preference-journal.jsonl`（`source=mirror, kind=preference`） | 否——作 twin 训练语料 |
| ② | 行动指令 | `state/mirror-queue.jsonl`（`status=pending`） | **是**——组长必须写任务书派 devops，**不得 override**（用户明确要求即授权，§5.4 同一法理） |

队列水位对组长可见：`orch status` 的队列段打印
`[!] mirror 行动指令待派发=N` 并列出前若干条正文；标 `status=done` 后不再计入（否则队列
永不见底）。指令含糊或涉护栏疑虑时，组长经 mirror 回问，回问同样留痕。

**两个新留痕面的 git 归属（勿误加进 `.gitignore`）**：`state/mirror-queue.jsonl` 与
`state/evals/taste/*.json` 与 `preference-journal.jsonl` 同族——用户口述与代理裁决必留痕
（§0.4），属**跟踪面**而非运行态。它们平时不存在（首次写入才出现为 `??`），随下一次交付
一起 `git add` 即可；devops 的合并前检查只看「本次改动是否触碰」，未跟踪的留痕新文件不挡
合并。

`status` 的概览项当前覆盖：待盲审案件数（= `evals/taste/` 留痕数）、宪章状态（草案/已确认 +
字节数）、`taste_enabled` 与 inspector 开关、行动指令/jury 待重裁队列水位、一致率与 override
率趋势（占位）、待决事项（minimum 档选型、twin 上线宣告）、待用户验收的最终交付（占位）。

## 5. 本阶段未接的面（E6 后续单）

`mirror review`（盲审即训练的交互面）、`mirror overrides`/`stats`（override 率与一致率趋势）、
`mirror distill`（批量蒸馏，v0.5 已简化为「按需再做」）、偏差报告队列的生成与绿色通道接线、
§5.5 回退链的自动触发（代码就位、默认休眠）。这些在 `status` 里一律**如实标注「占位」**，
不假装存在。
