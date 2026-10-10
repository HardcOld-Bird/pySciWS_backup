# 任务书 20261009-discover-pipeline-fake-green

（派发时间：2026-10-10T08:01:47+08:00）

基础设施改进任务（backlog FIFO 队首，id=20261009-discover-pipeline-fake-green）：

摘要：runner.discover_pipeline() src 优先：图目录内定制管线被 src 脚手架占位文件静默压制，build 出占位图但流程显示成功（假绿）

证据：组员交付记录：figdir 内 fig0_orch_smoke.py 完整，但 build 渲染的是 src 占位管线；无告警。期望：build 时若 figdir 与 src 同时存在同名管线，打印实际选用者；或提供 --pipeline-in-figdir / STYLE.yaml pipeline: 覆盖项

组长附注：（无）

要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，并注明 backlog id=20261009-discover-pipeline-fake-green 以便销账。

合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。
