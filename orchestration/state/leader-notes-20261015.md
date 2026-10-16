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

## 机制缺口立案清单（2026-10-15 下午累积，待 devops 立单）

1. **shutil.move 归档 reply 源缺失崩溃**（dispatch.py:1596）：源文件已不存在时 FileNotFoundError 逃出
   → 批次异常、已派发未结账坠落重派→双写者（实案 job 20261015-122406）。修法：move 前 exists() 守卫或
   except FileNotFoundError fail-soft（归档面不反噬派发）。
2. **任务书白名单声明不注入 pod-guard**：任务书「白名单」段仅 lint 提示，PYSCI_TASK_DIRS 只来自 --dirs/
   plan step dirs——声明面与注入面脱节（b2 实案被拦）。修法：派发时把任务书白名单段自动并入 PYSCI_TASK_DIRS
   或 lint 强制 plan step dirs 同步。
3. **机械验收 orch-tests 判据被哨兵留痕行污染误杀**（A.a5 实案）：output_tail 被哨兵豁免留痕 print 占据→
   判据解析 fail。修法：判据取测试退出码/末行摘要，或哨兵留痕写 stderr/独立文件。
   （↑ 三条均已于当日晚经 backlog #148/#149/#150-151 修复销账；orch-tests 判据面经130555/132143 并单落地。）

## 旧计划账面对账裁定（2026-10-15 晚，组长；证据=交付留档/主干代码实测）

对 6 个长期占路由面的旧计划逐一对账，工作均已由其他通道落地或被取代，关闭清理（state
保留审计）。真欠账 **零**——全部工作实证落地：
- `devops-gap-20261014`（d1 QUEUE_MEMBERS 收编 devops、d2 双通道文档）——主干实证落地，关。
- `psep2-night-20261014`——n1–n4 全部落地（.plan 注册自账 4/4；n3=orch taste 盲测/盲审留痕链路，
  twin-taste-mirror.md + state/evals/taste/，组长初查 grep「盲测」漏检「盲审」一度误判未做，以 4/4 为准）。
- `psep3-20261014`（pa1/pa2/pc/pd1/pd2/pb）——superstep/自办 evidence/裁决书/返工流/路由全落地
  （plans.py:208 pd2 亦在），关。
- `rework-smooth-20261015`（f1/f2/f3）——交付留档 devops-20261015-132938/134510/140208 在盘，关。
- `tb-1-docs-20261015`（s1）——lit 交付在盘，任务书经返工流 accepted，关。
- `adhoc-20261015-093500`（tui-ux）——s1 从未派发，工作经 12:53/13:09 直派单元落地，关。

另：watchdog 喊了整晚的两个「已提交未合并」面（085823-devops / 125314-salvage）经核验内容与 main
**字节级相同**（blob 6df2df2d 三方一致，零增量），以 `pysci-dev worktree merge --no-cp-state` 合并
摘面（会话数 52→52 不减），NEEDS_LEADER 噪声源清零。
