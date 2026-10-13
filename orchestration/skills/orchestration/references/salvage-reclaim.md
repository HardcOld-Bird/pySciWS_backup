# 抢救面（孤儿回收第五面）：进度先保住，再谈清理

回收原本只管四面（锁 / 条目 / worktree 名 / 进程）且**默认销毁**（`remove --force` + `branch -D`）：
会话被腰斩时那一半没提交的进度就这么没了，续作靠组长想起来。backlog 20261012-234000-leader
裁定把「进度」也当一面：**回收先抢救，force-remove 只对被证明干净的面开放**。

## 1. 判脏读数（`drain.worktree_salvage_state`，只读）

```
{worktree, path, present, registered, branch, branch_exists, dirty, unmerged,
 head, unknown, salvageable}
```

- `unmerged` = `git rev-list --count main..wt-<名>`；`dirty` = `git -C <面> status --short`
  非空；`salvageable` = 有面或有分支 **且**（脏 / 有未合并提交 / 证据不足）。
- **`registered` 是写入门闸**：`git worktree list --porcelain` 里有没有这个目录，是「你的
  隔离面还在」与「残骸还趴着」的唯一可靠区分。残骸目录里那个 `.git` 文件指向已删的管理项，
  进去跑 `git -C` 的读数不可信——最坏情形回落到父仓工作区，把**主仓的脏面**算成这个条目的
  进度，再 `add -A` 就把组长未提交的东西卷进抢救提交。所以未登记的一面绝不进去写。
- **读数拿不准就往「有东西要救」判**（`unknown`）：这条链路上唯一不可逆的动作是
  `remove --force`，误判干净＝销毁进度，误判脏＝只是多留一个没人删的面，代价不对称。

## 2. 抢救动作（`drain.salvage_worktree`）

`git -C <面> add -A` → `commit --no-verify -m "wip(salvage): 抢救 <id> 的未收尾进度…"`，
**分支与工作树都不删**。三点不是偷懒：

- **`--no-verify` 在这里是必需品**：pre-commit 跑 ruff 与测试，半制成品几乎必挂，钩子一失败
  抢救就静默不发生——那正是要消灭的「进度靠手动才保得住」。因此消息带
  `wip(salvage)` 标记，评审面/销账面的纪律照旧：**抢救提交不是交付，`backlog id=` 只认
  里程碑提交**；除这一处，交付 commit 仍禁 `--no-verify`。
- **`quarantined`（第四面判「旧写者没能确认离场」）不往树里写**：并发 `add`/`commit` 会把它
  正在写的半成品卷进来，只读记账 + 保面。
- **抢救失败仍返回记录**（`uncommitted=True`）：「没救成」恰恰意味着这个面更不能删。
  `commit` 走「nothing to commit」也留痕，别让人以为进度已进提交。

`cleanup_orphan_worktrees(names, run_log, salvage_sink=…, quarantined=…)` 三分支：确认干净
→ 原样销毁（**不回归的那一支**）；有未收尾 → 抢救 + 保面保分支 + 读数进 sink；隔离 → 只读。
末尾 `worktree prune` 兜底。返回值只含真清掉的名，抢救过的面不在其中（它是有意留的续作面）。

## 3. 队列自动接线（`workflow.backlog_attach_salvage` / `backlog_append_salvage`）

按条目此刻的状态分两条路，组长不必盘点：

| 条目状态 | 走法 | 为什么 |
|---|---|---|
| pending / in_progress（崩溃回收后会被自动重派） | **attach**：抢救事实就地挂到条目 `salvage` 键 + note 注记 | 下一跳回到**同一棵树**接着干，零新单 |
| needs_leader（blocked / run_failed / parse_error / 验收 FAIL） | **append**：另立 `source=salvage` 的 pending 续作单，原条目不动 | needs_leader 不再会自动重派；「这一跳失败」是组长要看到的事实，代码不替他忘掉 |
| needs_leader **且成因在 main 脏面**（`merge_dirty_face`） | **停链**：抢救照做、保面保分支，不立续作单 | 下一跳改变不了 main（实测三连跳树哈希相同）；判定见 `tier-a-tags.md` |

- 续作单插在**第一个 pending 之前**（未合并提交随 main 前进而贬值，续作不是新提案，不该再排队）。
- **幂等两道**：同一 `commit` 已挂过 → attach 返回 None；同一提交已有单在队 → append 返回 None。
- **链深上限** `SALVAGE_MAX_CHAIN=3`：到顶只留「须组长处置」注记，机器接手不能变成无限自我复制；
  needs_leader 条目不被 attach 改状态（裁决已做出，代码无权撤销）。
- `source=salvage` 与 `source: tier-a` 同字段同意图：复盘时一眼看出是机器接的手，不是组员建议。

## 4. 任务书里的续作指令（三形状）

`pysci-dev worktree add` 只会**从当前 HEAD 新建分支**，所以三种现场各有各的正确命令，写死在
任务书里（`workflow._salvage_instruction`），续作的一跳不必自己猜——猜错就是把进度丢在旧分支：

1. **面还在且登记在册**（同名续完）：`worktree add` 会报「已登记在册」，那不是故障，直接进那
   棵树续完，**不要** reset/另起，也别从 main 重开；
2. **目录已不在、分支还留着未合并抢救提交**：**别用** `pysci-dev worktree add <名>`（撞已存在
   的分支名会 128 失败），改 `git worktree add <路径> wt-<名>` 把原分支请回来；
3. **旧面没能确认干净 → 已升代 `<id>-r<N>`**：在新面里 `git cherry-pick --no-commit <sha>`
   取回进度，严禁复用旧名（那屋里可能还有第二个写者）。

抢救记录里的 `uncommitted=True` 会在指令末尾追加「开工先 `git status --short` 判读，该留的
手工移植」——自动没救到的部分，至少别让它被下一次 `add -A` 意外带走。

## 5. 观察面与读数痕迹

- **`SALVAGE` 事件**（watch 第六类，PRIORITY 末位）：`snapshot["salvage"]` =
  `workflow.backlog_salvage()` 的 id 集，差分去重只报一次；组长只收知会，不做处置动作。
- `orch status` 的抢救段列出面/分支/提交/成因（输出含「抢救续作面」）。
- `.done` 摘要：`worktrees_salvaged`（本次抢救的名）与 per-item 的 `salvage`/`continuation`；
  run 日志痕迹 `[+] 抢救孤儿 worktree …` / `[+] 未收尾进度已抢救…→ 自动续作单 <id> 已插到队首`
  / `[!] 抢救 <名>：目录在但**未登记为 worktree**…不进去写`。
- **`--dry` 一条都不执行**（不写、不删、不改运行态）。

## 6. 与升代（第四面）的共存

`stuck`（须异名重建）判据 = 热名 ∪ {目录在 **且** 未被抢救或抢救时未在册}。所以在册的续作面
不触发升代（同名续完正是要的零手动路径），残骸目录既保面又升代——两件事不互斥：残骸是上一代
`remove --force` 的半途产物，绝不与它同名并发。

## 7. 派发侧同族闸（④⑤，同一双写事故的另一半）

回收闸在**事后**清孤儿，`dispatch` 还得在**事前**别撞它还活着的会话（#69 实证两进程共写同一
sid 的历史档）。判据同第四面：池里 `pid` 判活 + `pid_is_session` 核对 argv，两闸都过才算在途；
任一不过（没记 pid、pid 已易主）→ fail-open 照续，不冤枉正常跑。命中后分流通：

- **人工派发**默认 `rc=2` 拒绝并点名占用 PID 与任务书，`--force-new-session` 才放行（放行=起
  新 sid，绝不复用占用中的）；**drain 批次**（`auto=True`）自动开新 sid 续跑，不换 sid 就会整批
  卡住。两者都在台账留 `sid_new_reason=prev_alive`，`orch ledger` 行打「换席[…]」。
- ⑤ **失败交付全文落审计副本**：`<blocked>`/run_failed 原文写 `pods/<m>/outbox/delivery-<ts>.xml`
  且 needs_leader note **不截断**——note 曾截 160 字，两份现场报告的关键证据都成了残片。
  写副本 fail-open（OSError 只告警），绝不反噬交付本身。
- **写者单一化**：一个 worktree 的自维护层（AGENTS.md）同一时刻只允许一个写者。任务书点名
  续作面即指定了唯一写者；组员发现该面有并发写者（自己的写入零成功、或目录里冒出别人的
  inbox 回执）→ **停手报 `<blocked>`**，别抢写也别 `rm -rf`（18:35 那一跳的停手是正确示范）。
