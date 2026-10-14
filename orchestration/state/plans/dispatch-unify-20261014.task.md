# 派发路径统一（用户裁定 2026-10-14）：plan run/adhoc 全部走成员队列，直派退役

## 事故根因（已取证）

`orchestration/state/queues/` 自 m4 建成后**从未有条目**——m5 只交付了 `plan advance`
的入队路径，`plan run`/`plan adhoc` 仍是旧直派阻塞路径（do_dispatch 直跑）。后果：
队列面板「各成员队列皆空」照不见在途直派、active-sessions live=0、STALL/PENDING-WORK
判据全盲——用户实测「devops 施工中但队列显示空」（2026-10-14 13:2x）。这是 m5
「同一能力不留两份实现」的未竟之处，本单收口。

## 目标（三件，缺一不验收）

1. **直派退役**：`plan run`（含 blocked 续作）与 `plan adhoc` 的环节执行**一律经成员
   队列**（enqueue → worker 消化 → 完成事件驱动下一 advance），`runner.run_headless`
   的直调只保留给队列 worker 内部；对外 CLI 语义不变（plan run 仍是「推进到交接暂停」
   的同步体验，内部=入队+等待完成事件）。兼容测试：plan run/adhoc 的既有测试全过。
2. **监测全量覆盖**：
   - 面板计数含 `in_progress`（在途条目显名+起跑时刻），「皆空」只在 pending 与
     in_progress 双零时出现；
   - active-sessions 注册覆盖一切在途派发（直派退役后即全覆盖；写侧沿 m6 LiveSession
     不重写）；
   - STALL 判据作用于 in_progress（cronless-wake 已有阈值语义，确保队列化后全路径生效）。
3. **事件显示去误导**：面板事件列表区分已消费/待排空（已消费的不再整行挂「建议处置」
   文案——建议句是 emit 时写入 detail 的历史文字，显示层标注消费态即可，不改事件流）。

## 验收预期（机械）

- `uv run --no-sync pytest` 全绿（1407+ 基线）+ 新增：plan run/adhoc 经队列的端到端
  （mock do_dispatch：入队→worker→完成事件→advance 推进）；面板对在途条目显示
  in_progress；「皆空」双零判据；
- **实测留痕**：起一个 adhoc 派发 → 队列面板当帧显示 in_progress 条目与 active
  live≥1（复现命令+输出）；
- `state/queues/` 从此有真实条目历史（队列层不再是空转设施）。

## 白名单（涉改文件）

`src/pysci/skills/orchestration/tools/**`（plans/dispatch/runner/queues/queueworker/
backpressure/tui/modules/watch）、`tests/**`、`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- 同一能力不留两份实现：直调 do_dispatch 的入口收敛到队列 worker 单点；
- 事件驱动语义保留：交接暂停/机械验收回派/blocked 续作不回归；
- grep 判据约定（20261014-075230）；不动 pyproject.toml（确需改→venv-sync）；
- 编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
  交付按 pod 交付协议 XML（`<result>`/`<blocked>`）；blocked 带 tags 与 reason；
  改进建议带合法 `kind` 属性。
