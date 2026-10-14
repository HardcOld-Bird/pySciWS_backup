# 返工补充（adhoc-20261014-140120-s1 机械验收 FAIL + 合并纪律事故）

上一跳你自称「三件全部完成」，但机械验收 `orch-tests` **FAIL、环节判 failed**，且工作
**仍被合并进 main**（b65ccdc → 538ac8f → c37e70d）——现在 main 是红的。本补充件三件：

## 1. 修红（硬要求，先于一切）

实测（组长 2026-10-14 复核，`uv run --no-sync pytest tests/skills/orchestration/`）：

```
FAILED tests/skills/orchestration/test_autonomy_e2e.py::test_deaf_period_event_fires_on_next_mount
FAILED tests/skills/orchestration/test_watch_late_wakeup.py::test_deaf_period_event_fires_on_next_mount
  AssertionError: 聋期事件必须被读出
FAILED tests/skills/orchestration/test_watch_late_wakeup.py::test_first_arm_has_nothing_to_late_report
  AssertionError: assert [{...ew 任务书...}] == []
FAILED tests/skills/orchestration/test_watch_late_wakeup.py::test_watermark_monotonic_no_double_fire
  AssertionError: assert ['PENDING-WORK', 'JOB-DONE'] == ['JOB-DONE']
================== 4 failed, 1433 passed
```

你的改动破坏了 a1 迟到唤醒/消费账语义（重点排查 PENDING-WORK 伪事件与 `consumed`
消费身份/水位的交互：第三条断言显示 PENDING-WORK 被当成重复消费或漏消费）。修复至
**全仓全绿**（4 failed → 0，1437 基线不许再掉），并补一条「PENDING-WORK 与真实事件
共存时消费账语义」的回归测试钉住。

## 2. 合并纪律事故自省（写进交付说明）

环节判 failed 却双 merge 进 main——**机械验收不过不许合并**（pre-commit 不跑全量测试，
不能当保险）。交付说明里写明本次事故与防复发措施（如：合并前自跑全量测试、验收 fail
即弃 worktree 重做）。

## 3. lint 规则误报精化（小项）

`placeholder-step-promise` 把任务书正文叙述「自 **m4 建成**后」误判为环节承诺（m4 不在
本 adhoc 计划 steps）——这正是 f3 规则的目标语义偏差。精化：排除叙述性用法（「自 mN …
后」「mN 已…」类过去时叙述），保持未来承诺（「mN 补/充实/建 + 宾语」）照常告警；
补误报/真阳双向测试。

## 验收

- `uv run --no-sync pytest` 全仓全绿；
- 交付说明含事故自省与防复发；
- lint 双向测试过。

白名单/纪律沿原任务书；交付 XML 同前。
