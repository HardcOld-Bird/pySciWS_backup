# backlog 派发并入成员队列（用户追问定案 2026-14 晚）：drain 退役为批量计划器

## 背景

dispatch-unify 留白「drain/backlog 直调不动」。现状：devops backlog 的执行仍走
`_drain-devops` → `do_dispatch` 直调（第二套 worker），与成员队列 worker 共抢
`state/devops.lock`（安全但两套实现）。用户追问「双派发为何不进 FIFO 依次处理」——
答案是应该，本单收口：**执行只留队列 worker 一份实现**。

## 目标（两件，缺一不验收）

1. **drain 退役为批量计划器**：`drain_backlog` 的组包/needs_leader/延期语义保留
   （用户裁决：devops 组包不取消），但批次**入队 `state/queues/devops.json`** 交
   队列 worker 执行——`_drain-devops` 不再自持 do_dispatch 循环（或退化为薄包装：
   计划批次→入队→wake_member）。验收含「backlog 条目 → 队列条目」的形态转换测试。
2. **唤醒语义对齐**：`wake_devops`（approve 后）改为「入队+唤醒队列 worker」；
   锁忙语义自然消失（FIFO 排队而非丢弃待拾）——PENDING-WORK 对 backlog pending
   的判据同步精化（有队列 worker 在跑即降噪，此即 144 注记场景）。
   **STALL 判活精化**（2026-14 晚实测误报：重型环节正当耗时 60min+ 被报停滞）：
   in_progress 计时须叠加**心跳判活**——backpressure LiveSession 心跳（120s）新鲜
   即「慢而不死」不告警（或阈值翻倍），心跳 stale 才报 STALL；补「慢而活/死而僵」
   双向测试。**wake-claim 自动认领补全**（wake-owner 单留白）：无 owner 时首个
   钩子会话自动认领（strict 身份可得时）；既有 owner 语义/测试不回归。

## 验收预期（机械）

- `uv run --no-sync pytest` 全仓全绿（3143+ 基线）+ 新增：approve→入队→worker
  消化→销账端到端（mock do_dispatch）、组包批次入队形态、锁busy场景不复现（排队）；
- **实测留痕**：真实 approve 一条 → 队列面板见 devops 条目 → worker 消化销账；
- devops-runs 日志语义保留（批量计划器记账）或迁移说明写 README。

## 白名单（涉改文件）

`src/pysci/skills/orchestration/tools/**`（drain/workflow/queues/queueworker/watch/
orch）、`tests/**`、`orchestration/README.md`、`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- 组包/needs_leader/延期语义不回归（既有 drain 测试全过）；
- 同一能力不留两份实现（本单就是立法者）；
- 迟到唤醒/消费账不破坏；机械验收不过不合并；
- grep 判据约定；不动 pyproject.toml；文件编辑工具；CLI 裸名；交付 XML + 合法 kind。
