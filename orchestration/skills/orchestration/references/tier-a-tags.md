# Tier A 失败标签与阈值自动立案（README §5.4）

组长侧只记调用面，本页是 `orch status` / `orch stats` 出现 tier-a 条目时的处置依据。

## 词表（受控，文件形状即协议）

`orchestration/state/failure-tags.json`（git 跟踪）：

```json
{"version": 1, "discipline": "…", "tags": {"<tag>": "<定义>"},
 "infra_kinds": {"<kind>": "<定义>"}, "reasons": {"<reason>": "<定义>"}}
```

初始 10 标签 + 增补 `merge_dirty_face`（2026-10-13）共 11：`whitelist_conflict`
`cli_missing` `doc_stale` `tool_bug` `harness_gap` `model_capability` `quota` `env_broken`
`design_smell` `unknown`。提案分类两值：`defect` `usability`。成因档位四值：`needs_decision`
`capacity` `blocked_by_env` `blocked_by_upstream`。

- 增补标签/分类 = 改文件（**须附定义**）并经 devops 交付，不改代码：Python 侧唯一读取口
  `tools/failure_tags.load_tags()` / `load_infra_kinds()` / `load_reasons()`，Node 侧
  `orchestration/guards/delivery-gate.mjs` 直接读同一文件，两侧口径必须一致；
- 词表缺席/损坏 → 标签面 fail-open（不阻塞派发，只失去一次校验）；分类面与档位面回落
  `DEFAULT_INFRA_KINDS`/`REASONS_DEFAULT` 兜底清单继续执法（值少且稳定，失去执法才是事故）。

## 交付面

`<blocked tags="a,b">`：属性可选，缺省=空列表合法（不标注不阻断）；属性存在时
gate 校验词表，含词表外标签 → exit 2 + 列出可用词表。分隔符半/全角逗号、分号、
空白都收，大小写不敏感。`<result>` 上的 tags 一律忽略（只 blocked 携带失败语义）。

`<blocked reason="…">`：**单值**成因档位，同样受控、gate 词表外退回、缺省不标注合法。
`tags` 说「失败是哪一类」，`reason` 说「该谁怎么处置」——`needs_decision`（待裁决，重派无效）
vs `capacity`（余量不足，补足可重派）处置相反，故 drain 写进 `.done`（含 `by_reason` 合计），
`orch status` 分列「待裁决=N 待排期=N」。**不切分**（`reason="capacity, foo"` 整串落词表外，
切分会把「标错」伪装成「标对」）。

## 标签裁决与链路驱动

误标裁决（`whitelist_conflict` vs `merge_dirty_face`/`doc_stale`/`concurrent_writer`）、
dispatch 的机械纠正/纠错日志，以及 `merge_dirty_face` 的停链/挂回语义见
`tier-a-tags-02-mislabel-and-linkage.md`（2026-10-18 按预算拆分）。

## 提案分类（`<infra_suggestion kind="…">`）四面执法

块非空时 kind 必填且须落在 `infra_kinds` 词表内。四个面共读同一份事实——缺任一面都会被
绕过（backlog 20261012-120001-leader 的实案：非法 kind 的占位空壳直达待审批队列）：

1. **组员侧 gate**：**逐块**校验（同一条交付里的每一块，不是只看第一块），并拦「多块」
   交付——解析器只取第一块，其余内容静默丢弃 → exit 2 要求合并成一块；
2. **入库面 dispatch**：落盘前复核词表，非法/缺失 kind **不入库**并打
   `[!] 改进建议未入库`。gate 退回后那一跳仍可能以违规回复收场，队列不该因此长空壳；
3. **审批面 approve**：kind 不在词表 → 拒绝入队（rc 2）。提案来源不止交付一条路（组长
   代补、Tier A 自动立案、手写文件都直接落进 `suggestions/`，绕过任何单侧执法）；
4. **读数面 ledger**：台账 `infra_suggestion` 字段答的是**「队列里有没有这一条」**，不是
   「交付文本里有没有那个块」（backlog 20261012-132138-devops 的实案：非法 kind 那一跳
   仍打「建议✓」，组长照读数找提案却找不到）。被第 2 面拦下的跳行尾打 `建议✗未入库`
   ——分裂要可见，不是静默抹掉。

`orch suggestions` 枚举时机械标注 `[!] 空壳建议`（剥掉标题与协议标记行后实质正文
<40 字）与 `[!] kind 非法（值）`：见到空壳直接 `reject --override "空壳"`，不必细读。

## 统计面

- tags 随交付落 `state/ledger/deliveries.jsonl` 的 `tags` 字段（旧行无该字段读作空）；
  `orch ledger` 行尾渲染 `标签[a,b]`；
- `orch stats` 末尾 **by_failure 段**：标签×成员计数，按合计降序；`!` = 该
  (tag, member) 已达阈值（下次 status/approve 会立案），`[词表外]` = 台账里有但
  当前词表没有的标签（说明词表被删减，案子仍会立）；
- 窗口 = **天数窗与跳数窗取并集**（任一命中即入窗），参数读 registry：

```json
"tier_a": {"window_days": 14, "window_hops": 20, "threshold": 3}
```

缺键/非法值逐键回落默认（fail-open），改阈值即改 registry，不改代码。

## 自动立案（§2.4：统计被流程硬性依赖）

`orch status` 与 `orch approve` 入口机械调 `workflow.tier_a_sweep()`。越限的
(tag, member) → 在 `state/suggestions/` 写一枚待审批提案 `<ts>-tier-a.md`
（`kind: defect` + `source: tier-a` + `fingerprint:` + 正文含计数与台账行引用 +
`success_criteria:` 标记行），**不直接写 backlog**——是否接受这份判决是强制选择点，
代码负责把案立到人眼前，人负责点头或 override（§5.1）。

去重两机制（缺一会天天重立同一个未修好的问题）：

1. **指纹 24h TTL**：指纹 `tag|member|窗口桶`，窗口桶取窗内该组合最新一条台账行的
   日期（同一天重跑必落同桶 → 幂等），落 `state/tier-a-filed.json`（gitignore，
   运行态）；同指纹 24h 内不重复；
2. **在案抑制**：pending 面里已有同 `tag|member` 的 tier-a 提案（既没 approve 也没
   reject）→ 不再立。跨天换桶也压住，直到案子被处置。

处置后果：

- `orch approve <sid>` 正常入 backlog（`source=tier-a` 随条目带走）；审批**回执投给
  devops**（提案人不是成员，`_member_of` 会解析出 `tier-a`，不特判就永远附送不到），
  即基础设施侧收到「这一类失败已达阈值」的整改工单；
- `orch reject <sid> --override "…"` 照旧落决策日志；驳回后同 (tag, member) 若继续
  攒够且换了新窗口桶，会重新立案（驳回不等于永久豁免）；
- 秒级同名（drain 与人工 status 同时跑）不覆盖：后立的加 `.1`/`.2` 后缀。

E5 inspector 落地前，自动提案就走上面这条现行 approve 面，不新增人工检查点。
