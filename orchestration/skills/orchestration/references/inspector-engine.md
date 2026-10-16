# 巡检引擎（inspector）生命周期与读数口径（README §6，E5a）

SKILL.md 的命令面只留一行索引；本文放**为什么这样设计**与**读数怎么解释**。
引擎住在 `orchestration/tools/inspectd.py`（组长侧），组员侧门面是 `pysci-inspect`。

## 1. 谁做什么

| 层 | 载体 | 干什么 |
|---|---|---|
| 闸门与循环 | `inspectd.py`（纯代码，零模型成本） | 开关/幂等拉起/预算/让路/迟滞/读侧检查/机械类自动晋升 |
| 巡检内容 | `pods/inspector/` 的 headless 会话 | 按引擎下发的菜单项做判断类工作，产出 finding |

分界的理由：**资源决策不许进 LLM**（成本与让路是硬约束，交给模型就是可被说服的闸门）；
**认知工作不该进代码**（「这个设计很别扭」写不出正则）。

## 2. 生命周期：代码保证，不靠组长记得

`ensure_inspector()` 内嵌在 `orch status` / `approve` / `dispatch` 三个入口——这三个是
组长日常必然敲的，巡检因此天然有激活点。它是 **fail-soft** 的：拉起环节任何异常都不许
让组长的正事命令失败，异常以消息形式留在读数里（不静默）。

`orch inspector start` 幂等三态，每种都打印**为什么没拉起**：

- 全局开关关闭 → `state/inspector.enabled` 内容为 false（文件缺席=开启）；
- 已在跑 → `state/inspector.lock` 持有者仍活；
- 冷却窗内 → 读最近一次 `.done` 的 `completed`，距今小于 `cooldown_s` 即不激活。

冷却窗专为「内嵌拉起」存在：没有它，组长每敲一次 status 就重激活一轮巡检。
`--force` 只越冷却，**不越开关也不越锁**（开关是用户的成本控制裁决，锁是幂等的定义）。

内嵌拉起意味着「凡跑到三入口就试派生」，连测试进程也算——故 `drain._spawn_detached` 有一道
咽喉闸：见 `PYTEST_CURRENT_TEST`/`PYTEST_VERSION` 即就地返回，不起进程也不写锁（backlog
20261012-191512-devops 实测两个漏挂 `iso_state` 的用例真点着过 worker）。闸在派生面不在判定
面，测试里 `spawned=True` 不作生产证据。

## 3. 单次激活的预算与停止原因

`inspect_drain()` 一轮 = 一次激活。退出条件写进 `.done` 的 `stopped` 字段：

| 值 | 含义 |
|---|---|
| `cooperative` | 收到协作停止请求（`inspector.cancel` 面；`orch inspector disable` 顺带发出） |
| `switch_off` | 跑到条目间隙发现全局开关被关 |
| `budget_hops` / `budget_wallclock` | 跳数 / 墙钟预算到顶（**成本硬上限，到顶自停**） |
| `yield` | 让路：生产侧正忙，本轮收束 |
| `throttled` | 迟滞暂停：积压超阈，停止产出 |
| `quota_exhausted` / `channel_error` | 系统性故障：额度耗尽 / 渠道级故障（与 devops drain 同口径，立即释放锁；`error` 面分列两类） |

队列空**不自毁**：抽一项菜单继续做（§7 的工作循环）。这是与 bencher worker 的关键差别
——bencher 队列空即收摊，inspector 的活是巡检本身。

## 4. 让路 ≠ 停工

`busy_surfaces()` 看三面：devops drain 锁、bencher worker 锁、计划侧派跳
（任一未完成计划的 `.state.json` mtime 距今小于 `yield_window_s`；计划没有「正在运行」
标记，故用新鲜度做判据，`0` 即关掉该判据）。

让路时：不派 LLM 跳、不晋升，但**零成本读侧检查照跑**（`check_cost_curve` 单跳工作量、
`check_context_pressure` 末跳上下文占用、`check_throttle_config` 迟滞阈值合法性）。引擎侧读侧检查发现的机械证据类
finding 同样自动晋升。

**前两支不是同一件事，别合并读数**（insp-20261012-211429-mechanical）：`est_tokens` 是
「本跳往会话档追加了多少字符」=吞吐，devops 78 跳实测与 `num_turns` 相关 **+0.81**、与
`ctx_ratio` 相关 **−0.11**——大单必然读出大数，用它诊断膨胀等于给组员下「少干活」的伪判据。
上下文压力读 `ctx_ratio`（envelope 真实读数，跨成员可比，故用绝对阈值 0.6 而非相对排名；
该阈值即 devops 巡检清单既有的「持续 >60% 提请归档」）。判**末跳**不判尾部均值：ctx_ratio
是一跳收尾时的水位，下一跳 resume 就从那儿起步，末跳 0.81 会被尾部三跳均值 0.51 稀释掉，
而那一格恰恰是要归档的信号。

## 5. 双队列迟滞（§5.2）

形状：**超过 `pause_*` 才暂停，回落到 `resume_*` 及以下才恢复**。两阈值相等就是每轮翻转
的开关，不叫节流。状态落 `state/inspector/throttle.json`（不落盘则重启即失忆，
暂停/恢复退化为抖动）。

`Registry.inspector` 校验 `resume < pause`，非法则**四个阈值整体回落出厂值**并在
`orch status` 标注「配置被忽略」——半套自定义阈值比没配更危险。

## 6. 发现分流（这是唯一的接口）

| 类别 | 出口 | 是否需要判决 |
|---|---|---|
| `mechanical`（有可重跑 repro+evidence） | `promote` → backlog `source=inspector` | **免判决**（§5.2：机械证据不需要审美裁决） |
| `judgment` | `pysci-inspect submit <id> --jury` → 采纳才晋升 | 必经 J 仲裁 |

- **没有 override 面**：`pysci-inspect` 的参数表里不存在 `--override`，
  `submit_judgment()` 的签名里也没有——§5.2 的「inspector 不允许自裁」是代码事实。
  测试 `test_inspector_has_no_override_surface` 钉住这一点。
- 仲裁判 B → 驳回且**不计 override 率**（它本来就没有 override 额度）；账只有
  `state/preference-journal.jsonl` 一本，晋升条目录 `channel` 与 `finding_id`。
- 缺 `success_criteria` 的发现判 `rejected` 而非留 `pending`：worker 每轮重扫 pending 的
  机械类，留着就是每跳重跑一次同样的拒绝。靠补判据复活，不靠重试。
- **M 路（重型 A/B）未接入自主通道**：任务书只要求判断类经 J。值得 A/B 复测的缺口走
  `bench-probe` 巡检项——inspector 读 `orch bench queue` 面并**经 bencher 队列提案**，
  不自己跑压测（§8 所有权）。
- 不可判（inconclusive）→ `escalate=twin` 挂起，`orch status` 点名待终裁。

## 7. 组长侧常用动作

```bash
pysci-orch inspector status                 # 闸门/阈值/队列/发现分布
pysci-orch inspector patrol --target code-walk --start   # 手动排一项（不给 target 即随机抽）
pysci-orch inspector assign --brief '核查 X 是否与 README §2 一致' --start
pysci-orch inspector disable                    # 成本控制：关开关 + 请求在跑 worker 协作停止
pysci-orch inspector enable
```

阈值与预算**没有 CLI 旋钮**，只在 `state/registry.json` 顶层 `inspector` 块逐键覆盖
（`max_hops` / `wallclock_s` / `cooldown_s` / `yield_window_s` / 四枚迟滞阈值）。
§7 要的是「代码保证」——可被随手关掉的成本控制等于没有。

## 8. 豁口（§0.2）

`pysci-inspect` 自身不在巡检范围（拿着锤子敲锤子敲不出结论）。巡检菜单里的
`pipeline-self` 覆盖 orch/drain/benchd/inspectd/verdict/jury/guards，但不覆盖组员手里
那个 CLI 的参数装配层。
