# 队列感知唤醒（用户设计，2026-10-14）：PENDING-WORK/STALL 伪事件入 watch 循环，手工巡更 cron 退役

## 背景

三层自动化防线中「组长巡更 cron」是唯一手工环节（软规程要求组长自己挂/撤，会话级、
7 天过期）。用户设计（2026-10-14）：脚本天然可判断所有组员队列是否为空——按队列空满
**自动起止**唤醒，即得最严谨的队列监测。载体现成：watch 钩子循环（全自动臂/看门狗复活）。

## 目标（三件，缺一不验收）

0. **迟到唤醒双消费缺陷修复（生产实锤，2026-10-14 12:4x）**：同一 drain 完成事件
   （run=20261014-123354）唤醒了组长**两次**——第一次带〔迟到唤醒：发生于挂载前（聋期）〕
   标注，第二次同一事件不带标注再次命中。违反 a1 硬判据「水位单调、重复挂载不重复消费」
   （a3 e2e 过了但生产双触发，说明 catchup 消费路径与正常 delta 路径的水位推进不同源）。
   要求：**同一底层事件只唤醒一次**——迟到消费必须推进与正常消费同一水位（单一水位源），
   修后补生产形态回归测试（模拟「late 消费后立即重挂载」→ 零重复唤醒）。

1. **`orch watch` 增队列感知伪事件**（与既有 5 类事件同走 detect→echo→exit 2 唤醒链）：
   - `PENDING-WORK`：以下任一成立即判「非空」——devops backlog 有 pending；bencher/
     inspector/9 成员队列（queues.py）有 pending 且无存活 worker；events 水位未排空；
     `data/taskbooks/` 有 new 态未处理；（判据单一实现进 watch.py/queues 读侧 helper，
     不各写各的）；
   - `STALL`：任一 in_progress 条目超预期时长（阈值可配，默认建议 2× 历史中位耗时或
     固定 60min，实现定但须可配）；
   - **节流硬要求**：同条件唤醒 ≤1 次/30min（防「唤醒→无事→再唤醒」循环）；节流状态
     持久化（重启不重置不堆积）；
   - 队列全空 → 纯 TIMEOUT 循环**零唤醒**（= cron 自动取消语义）。
2. **手工巡更 cron 退役 + 规程改写**：
   - 编排技能 `references/patrol-watchdog.md` 与相关 SKILL 行改为「队列感知巡更自动
     起止，组长不手工挂撤 cron」；保留一句降级注：钩子/看门狗全灭时才手工挂 harness
     cron 兜底；
   - README §4.5 同步实况；三级防线表述更新（巡更层从「会话级手工」改为「队列感知
     自动」）。

## 验收预期（机械）

- `uv run --no-sync pytest` 全绿（3081+ 基线）+ 新增：队列非空→伪事件触发且节流生效
  （30min 内不重复）；队列全空→N 个周期零唤醒；STALL 判定（假时钟）；水位/节流持久化
  重启语义；既有 5 类事件与迟到唤醒不回归（a3 的 e2e 全过）；
- 实测留痕：制造「一成员队列 pending 无 worker」→ watch 一轮内唤醒；清空后连续周期
  零唤醒；
- 技能/README 一致性测试过；SKILL.md ≤8192B（余量预警可查）。

## 白名单（涉改文件）

`src/pysci/skills/orchestration/tools/**`（watch.py/queues.py/events.py/orch.py）、
`tests/**`、`orchestration/README.md`、`orchestration/skills/orchestration/**`、
`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- 事件层单一实现：伪事件走既有 TYPES/WAKE/echo 链，不新造通道；
- 节流必须防抖且可证伪（测试含同条件重复触发）；
- **grep 判据约定（20261014-075230）**：禁词运行期拼接/转义，判据限 `*.py`；
- 不动 pyproject.toml；确需改则改后必须 `pysci-dev venv-sync`；
- 编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
- 交付按 pod 交付协议 XML（`<result>`/`<blocked>`）；blocked 带 tags 与 reason；
  改进建议带合法 `kind` 属性。
