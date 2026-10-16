# 任务书：看门狗双生/多生修复（start 幂等竞态 + run_log 同秒撞名 + 双实例不退）

## 成因证据（组长实测 2026-10-15 18:01–18:04）

1. **同参双实例**：一次 `watchdog start` 波形里出现两个 `python.exe -m …orch _watchdog-loop
   --run-log …/watchdog-runs/20261015-180151.log` 进程（ppid 链 21324→4544，相隔 7ms），
   **共存 ≥4 分钟**（未按 lock_busy 退出）。
2. **竞态窗口**：`start_watchdog` 的幂等闸是「`lock_status(...)==running` 则不动」——两次
   start 调用相隔 <100ms 时双双通过判定（17:56:26 双生、17:57:33 双生、18:02 复现）。
   触发场景：组长手工 `watchdog start` 与 Stop 钩子 watch-arm 的「看门狗自启动（v4）」
   近同时发生；每回合结束都可能撞一次。
3. **撞名放大**：`new_run_log` 按**秒**时间戳命名——同秒两次拉起得到**同一个 run_log**
   → 两个实例 argv 完全相同、共享日志、互相看不见对方。
4. **危害**：N 个实例 = 每 tick 巡检面扫描×N（风暴 git 刷屏的乘数因子）+ 锁语义失效
   （`acquire_lock` 显然没能把第二个实例挡退，为何请查明——键控 run_log 同名可能就是根因）。

## 修复要求
1. **拉起原子化**：start 的「查在跑→派生」做成互斥（文件锁/O_EXCL 创建 spawn 标记，
   持有者派生；或 spawn 前先占位）。同秒并发调用至多派生 1 个。
2. **run_log 唯一化**：命名加 pid/毫秒/序号防同秒撞名。
3. **双实例自清**：`run_watchdog` 的 `acquire_lock` 必须**真正互斥**——抢不到锁立即退出
   （留痕 `lock_busy`）；查明现状为何双实例共存（键控同名 run_log？锁文件残留？）并在
   回归里钉死：并发两个 `run_watchdog`（同参/不同参各一）→ 恰一个存活、另一个
   lock_busy 退出。
4. 顺检 `stop_watchdog` 广播纪元对**不持锁孤儿**的清退仍有效（现有语义保留）。

## 验收预期
1. 并发 start×5（同秒）→ 存活实例恰 1；
2. 同参双实例用例：第二个 run_watchdog 秒退 lock_busy 且留痕；
3. run_log 同秒两次生成 → 两个不同文件名；
4. 全量 pytest 绿；watch-arm 自启动路径与手工 start 混跑不再堆积（wmic 过滤
   `_watchdog-loop` 观察 2 分钟，实例数恒 ≤1）。

## 白名单（写权限）
- src/pysci/skills/orchestration/tools/
- tests/
