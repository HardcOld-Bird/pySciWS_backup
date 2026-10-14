# 组长留档 2026-10-15（待 pd2 验收对照）

- 00:32:44 事件流出现 3 条 TASKBOOK-NEW（tb-20261015-003244-task/-2/-3，标题「返工目标大任务」，
  source=mirror-tui），但 data/taskbooks/ 目录不存在、taskbooks list 全空——**事件无真源**。
- 判断：疑为 pd2 在途期间 devops 实测 TUI 投递面（pd1 修订⑦）的残留（投后清盘、事件留痕删不掉）。
- 处置：不拆解（无真源可推进）、不惊动用户、事件留痕不删（审计特性）。
- **待办**：pd2 交付验收时核对其实测记录中此三本的来历；若属实测残留，要求补「实测后清理
  事件残留说明」纪律（事件 append-only 无法删，至少在交付正文声明）。

## 销案（00:5x，pd2 验收时）

- 孤儿 TASKBOOK-NEW 事件来历已对上：pd1/pd2 开发期 e2e 测试（「返工目标大任务」即其测试文案）写穿真实 events.jsonl 所致。
- pd2 已堵漏（rface/cface 锚 iso_state + wake stub）；实测 20 passed 后事件流零写穿。销案。

## 系统性缺陷记录（01:0x，待 pb 验收时处置）

- **同类缺陷第三次**：测试写穿真实 state——① g3（test_plan_guards 系，已由 iso_state 兜住面）；
  ② rface（test_rework_flow，pd2 已修）；③ p-adv（pb 在途测试，00:57 又写穿 sim 队列 +
  6 条孤儿 TASKBOOK-NEW 事件「高优大任务/普通大任务」）。
- **模式根因**：每个新测试文件自建夹具（rface/cface/…）就可能漏 patch 某个状态面；iso_state
  是手工清单，新增状态面/新夹具都可能漏。
- **系统性修复要求（pb 交付验收时一并派）**：conftest 加 autouse 防写穿哨兵——测试进程内
  对真实 ORCH_STATE_ROOT/data 面的写操作直接 fail（或全部 autouse 重定向到 tmp），
  让「漏隔离」从静默污染变成测试即刻失败。零星补 patch 治标不治本。
- 泄漏条目已清理（sim 队列 1 条 failed 留痕）；孤儿事件 append-only 不删。
