# Tier A 标签裁决与链路驱动（tier-a-tags 分册 02）

> 由 `tier-a-tags.md` 按 basic.md §1「超预算即拆分」移出（2026-10-18，Tier-A
> 20261018-212801-tier-a）。主册保留词表/交付面/提案分类；本册是误标裁决与
> `merge_dirty_face` 的链路行为。

## 标签裁决：按**失败动作**，不按「写不下去」的感受（Tier-A 误标复盘）

`whitelist_conflict` **只**用于「目标路径不在本 pod 允许写面内」（pod-guard 路径拦截 /
须他人或组长代写）。其余阻塞各归其类——**它是全词表里最容易被当兜底的一个**：

| 阻塞证据（正文可复核） | 正确标签 |
|---|---|
| pod-guard 拦截、白名单未声明/不可写 | `whitelist_conflict` |
| `pysci-dev worktree merge` 打印 `[✗] 交集 N 项` / rc=2 | `merge_dirty_face`（**dispatch 已机械纠正**，见下） |
| 文档面/技能真本断言失败（doc-face 测试红、README 口径过期） | `doc_stale` |
| 既有工具行为缺陷（有复现） | `tool_bug` |
| 同一隔离面/会话被并发写者占用 | `concurrent_writer` |

- **机械纠正（2026-10-18，Tier-A 20261018-212801-tier-a）**：交付正文含 `[✗] 交集` 且标签为
  `whitelist_conflict` 时，dispatch 落台账前改为 `merge_dirty_face` 并打印
  `[!] 标签纠正`；原件（`tags_original`）写入运行态
  `state/ledger/tag-corrections.jsonl`——Tier A 计数读纠正后的 `tags`，审计仍可回溯。
  无该证据的写权限面冲突**原样保留**（不猜）。
- **历史误标**可在同日志下逐条纠正（不改台账 schema——`LedgerEntry` 拒绝未知键）；2026-10-18
  已纠正 3 条（2×merge_dirty_face、1×doc_stale），使 devops 的
  `(whitelist_conflict, 14 天/20 跳)` 计数回落到 2（阈值 3）。
- **两例真白名单冲突（2026-10-12）的根因已修**：任务书「白名单（写权限）」声明未注入
  pod-guard 的链路由 backlog `20261015-132939-fix3` 修复（声明即注入 + 中文目录正则）；
  新任务书按模板写「白名单」段即可，不再需要成员在 pod 外被拦。

## 标签的第二用途：有的标签会改链路

绝大多数标签是纯统计面（by_failure 计数 → 阈值自动立案）。`merge_dirty_face` 是第一个
**驱动行为**的标签（backlog 20261013-102152-devops）：

- **认成因**：`drain.merge_intersect_cause(outcome)` 在 blocked 交付上两通道任一命中即认定——
  `tags` 含该标签，**或**正文含 `pysci-dev worktree merge` 打印的 `[✗] 交集 N 项` 行。待清文件清单
  从该行**下一行起**的缩进路径行取（同一行的尾巴不是文件，反引号脱壳）；读不到清单仍算交集类
  ——认成因比凑清单重要。
- **停链**：抢救照做（`wip(salvage)` 提交、面与分支都保住），但**不立 `source=salvage` 续作单**。
  理由：这类失败的成因在 **main 的工作区脏面**，不在被派者的树上，下一跳改变不了它——2026-10-13
  上午实测三连跳（`072000-leader` → 两跳 salvage）树哈希逐字节相同、测试结果相同，第 4 跳才撞上
  `SALVAGE_MAX_CHAIN`。续作链对它是空转。
- **挂回原条目**：清单写进条目的 `merge_intersect` 字段（结构化，不必从 note 全文再解析），note
  附处置命令，`orch status` 的 needs_leader 段渲染「合并相交待清：<文件>——清面后
  `pysci-dev worktree merge <面>`（已停链，无续作单）」，组长一跳就知道要动哪个文件。
- **组员侧教标**：任务书尾部（`workflow.backlog_task_text`）写明按该标签交付并原样贴清单；不教
  就没人标，链路只能靠正文碰运气。

**增补会改链路的标签必须同时改代码**，并在词表定义里写清它驱动哪条链路——只加词不加行为，
组员标了也没人读；只改行为不加词，统计面会把这一类算进 `unknown`。
