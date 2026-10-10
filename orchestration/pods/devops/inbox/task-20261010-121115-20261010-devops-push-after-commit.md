# 任务书 20261010-devops-push-after-commit

（派发时间：2026-10-10T12:11:15+08:00）

基础设施改进任务（backlog FIFO 队首，id=20261010-devops-push-after-commit）：

摘要：devops worktree 流程：cz commit 后尝试一次限时 push（best-effort，失败即忽略）

证据：用户裁决 2026-10-10：本机全程校园网，有时可直连 GitHub；不开常驻代理（带宽低/不稳，意外接管网络反有害）。要求 devops 每次 commit 后尝试一次 push，失败当预期不处理——长期总会有成功的一次，push 本不着急。remote=origin(github backup, https)，无持久代理（已核）。实现：devops 技能 SKILL.md worktree 规程第5步后加 `timeout 25 git push origin main`（或等价限时），忽略一切失败（网络/认证/超时），不重试、不设代理；charter「禁止 push」改为「限时尝试 push、失败忽略」。超时是硬要求（直连失败/凭据缺失会挂起阻塞 drain）。push 按分支增量，首次成功即带上此前所有积压提交。

组长附注：（无）

要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，并注明 backlog id=20261010-devops-push-after-commit 以便销账。

合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。
