# 任务书 20261009-taskbook-cli-signature

（派发时间：2026-10-10T08:32:51+08:00）

基础设施改进任务（backlog FIFO 队首，id=20261009-taskbook-cli-signature）：

摘要：任务书/文档中 build 命令示例与 CLI 签名不符（build 只收 figdir 位置参数，示例写成 build gain_ep fig0_orch_smoke 会 unrecognized arguments）——需统一 charter/任务书模板/SKILL.md 的命令范式

证据：组员按任务书步骤 2 照抄报 unrecognized arguments，自行改用 build '<figdir>' 成功。期望：模板统一为 build 'data/research/<n>_<线>/article/figures/<slug>'

组长附注：（无）

要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，并注明 backlog id=20261009-taskbook-cli-signature 以便销账。

合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。
