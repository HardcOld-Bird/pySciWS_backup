# 任务书：夜航差分基线缺口修补（SessionStart 顺手写基线）

## 缺陷（组长真机实测 2026-10-15 19:2x）
差分逻辑（`night.py --diff`）对「首次见到的会话」只记基线不注入（设计合理，防与 SessionStart
重复注入）。但 **SessionStart 注入时不写基线**——基线靠首个 Stop 钩子补记。于是：
会话起始时 night=on、中途拨 off、而 v7 钩子（或任一会话的首个 Stop）恰好在拨动之后才跑 →
真实跳变被当基线吞掉，刹车差分丢失（实案：本会话 `night_seen` 记成 `off`，本应是基线 `on` +
一条刹车差分）。

## 修复要求
1. `night-policy.sh`（或 `night.py` 注入路径）在 **SessionStart 注入时**把当时的
   enabled 值写入 `night_seen[session_id]`（与 --diff 同一落点/watch.json 键，单点）；
2. PostCompact 注入**不**重写基线（只补注入，不动读数——否则把跳变盖掉）；
3. 回归用例：模拟「SessionStart(on) → 拨 off → 首个 Stop」断言刹车差分**照常产出**（不再被
   吞）；首见无基线且无 SessionStart 的冷路径行为保持（记基线不注入）。

## 验收预期
1. 上述用例绿；既有 --diff 用例零回退；
2. 全量 pytest 绿。

## 白名单（写权限）
- src/pysci/skills/orchestration/tools/
- tests/
- .qoder/hooks/
