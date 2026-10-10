# 任务书 20261010-drain-cooperative-stop

（派发时间：2026-10-10T10:29:52+08:00）

基础设施改进任务（backlog FIFO 队首，id=20261010-drain-cooperative-stop）：

摘要：为 drain 增加协作式停止：orch drain-stop 命令写 state/devops.cancel；drain 每项间隙检查该文件→干净退出（释放锁、写部分 .done 摘要、cancel 文件自删）；orch status 展示停止方式。补足「分离进程无规范停止手段」的缺口

证据：用户问询（2026-10-10）：detached drain 目前只能手动 kill pid 或 OS 关机终止；kill 于条目中途会留下孤儿 in_progress（与 20261010-drain-orphan-reclaim 互补）

组长附注：孤儿条目回收（原 started_at=2026-10-10T10:07:42+08:00）

要求：以 worktree 隔离实施（你的 devops 技能有作业规程）；完整测试后合并、commit（不 push——push 须用户授权）；交付中报告改动清单与验证证据，并注明 backlog id=20261010-drain-cooperative-stop 以便销账。

合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。
