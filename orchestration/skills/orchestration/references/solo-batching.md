# 大单 solo 标记与组包过载（approve --solo）

背景（backlog 20261012-170301-leader）：drain 组包只按「同提请者 + 同 effort」聚类，
不估任务体量——多张大单并进同批 → 一跳消化 → 必触该跳 max_turns 腰斩。实锤：E4+E5a+
E5b 三单同批触顶 150（run 122624），E5a 二次腰斩、20 未提交文件险全损；E2 同形状
（run 233336）。聚类维度与风险维度错位，靠 solo 旗标补齐。

## 何时带 --solo

approve 时凡「pod 级 / 多子面」大单必带 `--solo`。典型 solo 形态：原 E0–E7 各阶段
阶段单（E2/E4/E5a…——跨多子面、动辄几十上百跳）。判据（满足其一即 solo）：

- 改动面跨 ≥2 个 pod，或 ≥2 个不相邻子系统；
- 预估明显超出一次会话预算（历史参照：E 阶段单均触过 max_turns）；
- 任务书含「多阶段实施 + 全量测试 + 合并」的完整交付链。

普通单项修复 / 巡检 / 蒸馏单**不要**标 --solo——组包省上下文的收益对小单仍然成立。

## 机制（三面执法）

- **入队**：approve --solo 写条目字段 `"solo": true`（orch status 的 [backlog] 行显示
  `[solo]`；JSON 面直接可见）；
- **组包**：drain `backlog_take_batch` 对 solo 项双向跳过——不并入组包、也不作组包
  种子（batch_size 恒 1，任务书退化为单项、无批量菜单）；
- **提醒**：`orch status` 在 pending > 组包上限（4）或**连续 ≥2 批 run_failed** 时打
  「疑似组包过载」提醒行。

## 见到过载提醒怎么办

1. 先看最近 needs_leader 条目是否是漏标的大单：`orch status` 的 needs_leader 段 +
   `state/backlog.json` 该条目；
2. 补标：approve 旗标只在入队时生效，重派前直接在 `state/backlog.json` 给该条目加
   `"solo": true`（运行态文件，人工补字段即生效，无需走审批流）；
3. 大单本就该 solo 单跳消化：补标后按正常重派流程走（补充证据后 dispatch，
   或等 drain 下一轮 FIFO 取到它）。
