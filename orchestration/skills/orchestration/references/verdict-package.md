# 判决包流（REFACTOR §5.1/§5.6）——审批的通道、闭环与账目

实现：`src/pysci/skills/orchestration/tools/verdict.py`（共享库，组员侧 inspector
复用同库，§0.9）+ `workflow.cmd_approve/cmd_reject/backlog_complete` 接线。
本文件只写**操作面**：什么时候用哪条通道、退出码读什么、账在哪里。

## 1. 为什么审批要带判决包

基建判断（「这条提案该不该做」）如果全部由组长凭记忆与偏好当场定，两件事必然发生：
判断不可复核（只有一句 note），且组长的注意力成为全系统吞吐的瓶颈。§5.1 的解法是把
「依据」外包给三个可复核的事实源——**仲裁团（J）**、**实测（M）**、**组长自裁留痕
（override）**——并让代码在入库前检查你选了哪一个。没选就 `rc=2`，提案原地不动。

## 2. 四态与例外通道

```
J     approve <id> --jury                        组卷→双序两局→自动生效（阻塞，秒级到分钟级）
M     approve <id> --bench <案id[,案id]>          锚基线 commit，交付后自动 A/B（不阻塞）
J+M   approve <id> --jury --bench <案id>          先问裁判，核验交给实测（判决冲突时 M 覆盖）
—     approve <id> --override "<理由>"            组长定案，留痕并计入 override 率
```

- **例外通道（唯一免包）**：`kind=defect` **且**该提案是组员在 `<blocked>` 那一跳提出
  的。理由是失败已被 in-vivo 实测过，证据就是那次失败本身。判据来自建议文件的
  `delivery:` 标记行（dispatch 落盘时就固化，不是事后反查台账——台账会被归档清理）。
  所以旧版或人工撰写的建议没有标记行 ⇒ 走不进例外通道，必须给包。
  工具会在放行时提示「建议连带 `--bench` 复跑原失败案」，把一次性失败变成可比读数。
- `--override` 与 `--jury/--bench` **互斥**，同时给 → `rc=2`（参数冲突）。override 的
  语义是「我不需要第二意见」，与要第二意见的通道并存是自相矛盾。
- **经 J 或 M 判决返回的 reject/纠正不计入 override 率**。只有组长绕过可用通道自己
  定案，才是一次真实的自裁。

## 3. J 路：仲裁直接生效，组长只写回复

`--jury` 把待审批建议自动摆成仲裁案卷（`## 选项 A`=采纳，全文进卷；`## 选项 B`=维持
现状；`## 证据`=可证伪的成功判据），落 `state/evals/jury-cases/<id>.md`，然后调
`jury.adjudicate` 开庭。判决落地三态：

| 仲裁输出 | 工具行为 | rc |
| --- | --- | --- |
| `A`（采纳） | 入 backlog，journal 记 `channel=jury` | 0 |
| `B`（驳回） | **自动驳回**：写回复、移 `rejected/`、留痕，组长无需再敲 reject | 0 |
| 不可判（双序不一致） | 不入队不驳回，挂起呈组长，journal 记 `choice=pending` | 1 |
| 渠道降级（`queued`） | 案卷进 `evals/jury-queue/`，提案保持 pending | 1 |
| 案卷/接口异常 | 不入队，打印案卷路径 | 2 |

降级 rc=1 是刻意的：**「过了仲裁」这个读数必须是真开过庭**。缺 `SILICONFLOW_API_KEY`、
简报未 `approved:`、panel 未选型都走 queued，绝不静默放行。案卷格式与简报冻结规则见
`references/jury-panel.md`。

## 4. M 路：核验方式审批时一次定死，实施后只实测

§5.6 要防的是「做完之后没人判定到底有没有变好」，以及「做之前就吵个没完」。所以：

1. **审批时**（`--bench`）：取当时 HEAD 作基线锚点，入队一个 `capture` 项，并把
   `{stage, cases, baseline_commit, label}` 写进 backlog 条目的 `verify` 字段。
   基线锚点是审批时刻的 commit，与 bencher 何时真跑无关（副本按 ref 检出）。
   取不到 commit（不在仓库/git 不可用）→ `rc=2` 拒绝标 M，而不是留一个空锚点。
2. **销账时**（`backlog_complete`，由 drain 自动调用）：只认 `verify.stage == approved`，
   排一条 `ab` 项（`ref_a=基线 commit`、`ref_b=HEAD`、带 `backlog_id`），stage → `verifying`。
   判据在 stage 字段，故重复销账/孤儿重跑**不会排出第二项**。
3. **判决落地**（bencher 消化 `ab` 项后自动调 `apply_ab_verdict`，不需要人记得去做）：

| A/B verdict | 落地 | stage |
| --- | --- | --- |
| 改善 | 条目补 `verified_at` 并写「核验通过」 | `cleared` |
| 未改善 | **自动立案**：新建议进 pending，`kind=defect`、`source: post-verify`、判据=「未改善案卷须复测出改善」+ 复跑命令 | `corrected` |
| inconclusive | 挂起呈组长，标 `escalate=twin`（twin 终裁接线属 E6 全量） | `escalated` |

三态都往决策日志写一行（`source=post-verify`）。注意纠正提案**不带** `delivery:` 标记
（它背后没有失败的一跳），因此它自己也要过判决包——闭环是递归的，不是终点式的。
核验失败会污染 ledger 吗？不会：落地异常只打印 `[!] …需人工销账` 并留运行日志，
不把已完成的交付改成 blocked。

## 5. 账目：override 率与「判决包健康」

单一账本 = `state/preference-journal.jsonl` 的 `channel` 字段（每条判决都写一行）。
`orch status` 的「判决包健康」段回读它：

```
通道用量：J=n M=n J+M=n override=n 缺陷例外=n｜override 率=N%
核验在途=k（待交付后排 A/B）：<id 列表>
[!] 核验挂起（escalate=twin）=k：<id 列表> → 需组长人工终裁
```

`escalate` 非空是要立刻处理的：**没有终裁，M 通道就退化成「做完就算数」**。override 率
本身不是 KPI——它是「基建判断有多少还全靠人」的读数，抬头就说明该给某个环节补 J 或 M
通道，而不是该省掉判决。

## 6. consult / consult-twin（可选咨询，与判决包正交）

- `orch consult "<问题>"` → deputy，平级协作，随时可用。
- `orch consult-twin <案卷.md>` → twin 偏好面。**`taste_enabled=false` 时明确拒绝
  （rc=2）并提示改走 consult**，不静默降级成 deputy 咨询——「问了个不存在的声音」比
  「问不到」更坏。开关只能由用户本人置位（§5.5），组长无权代按。twin 的判决通道与
  冷启动见 `references/twin-taste-mirror.md`。

## 7. 降级与排障

- 闸挡了明明该直接办的急事：`--override "<为什么这次不等通道>"`。这是一次真实自裁，
  会进 override 率——这正是设计意图，不是惩罚。
- `--bench` 报取不到 HEAD：在 worktree 外的树里跑 orch 或 git 不可用；先 `git -C <root>
  rev-parse HEAD` 自查。
- A/B 队列项停在 pending 没落地：bencher worker 未起（`orch bench-up`）或判决未消化，
  见 `references/drain-watch.md` 的挂载纪律。
- 手工核账：`journal.read_records()` 每行含 `channel`；`verdict.health()` 是同一段的
  数据源，两者不一致即为读数面 bug（不是操作问题）。
