# 任务书 20261010-reclaim-worktree-cleanup

（派发时间：2026-10-10T12:23:05+08:00）

基础设施改进任务（backlog FIFO 队首，id=20261010-reclaim-worktree-cleanup）：

摘要：drain 孤儿回收顺带清理死 worker 残留的 worktree/branch；.done 作为干净完成信号 surfaced

证据：现状：backlog_reclaim_orphans 只复位 backlog 条目，不清理 worktree。若 devops 在 worktree 内中途死亡，残留 wt-<slug> worktree+branch 会使重派时 `git worktree add -b wt-<slug>` 撞名失败（2026-10-10 一度疑似发生，后证实为误报，但缺口真实存在）。要求：drain 派发时把所用 worktree 名记入 backlog 条目（或 run state），reclaim 孤儿时对该 worktree `git worktree remove --force`+`git branch -D`；兜底 `git worktree prune` 清悬挂管理项；不自动删非孤儿/活跃 worktree（防误删并发手动作业）。另：orch status 明确 surfacing「pid 死但无 .done = 崩溃」vs「.done 存在 = 干净完成」，作为用户判断 devops 状态的依据。

组长附注：（无）

要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，并注明 backlog id=20261010-reclaim-worktree-cleanup 以便销账。

合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。
