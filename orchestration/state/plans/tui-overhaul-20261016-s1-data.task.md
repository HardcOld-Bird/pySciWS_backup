# 任务书 tui-overhaul-20261016 · s1-data：TUI 大改版数据层读数/写入扩展

## 背景 / 目标

pySci TUI（`src/pysci/skills/orchestration/tools/tui/`，Textual 8.2.8）要按用户 13 项需求全面改版：任务书按返工家族分组、任务报告 Markdown 渲染、计划与派发列表+详情页、盲审交互流程（2_REFACTOR §5.4/§10：隐去 twin 答案逐案盲审，不一致时「原因批复＋是否需恢复更改」，产出偏差报告入强制处理队列）、队列/系统/待决/决策看板卡片化、性能修复（徽标指纹不再全量解析 46 个 plan YAML）、footer 与提示文字清理。

本计划共 9 个串行环节（s1→s9，全 devops）；**本环节只做数据层**（`tools/` 下纯函数扩展与修复），**不改 `tui/`**（s2 起才动）。TUI 后续环节将直接调用本环节交付的读数/写入函数，故签名与语义须按本任务书实现（后续任务书以它们为既定接口）。

**已核实的机制事实**（照做即可，勿重推）：
- `data/taskbooks/<id>.state.json` 字段含 `rework_of`（原任务书 id）与 `rework_round`（轮次）——返工家族沿 `rework_of` 链归组即可；`<id>.taskbook.md`（frontmatter+正文）与 `<id>.report.md`（Markdown 全流程报告）同目录。
- `orchestration/state/plans/<id>.plan.yaml`（结构）+ `<id>.state.json`（状态真源，顶层含 `plan_id/created/goal/steps[]/completed_at/abandoned_at`，step 含 `id/status/review/evidence/job_id/task_file/session_id`）——**goal 与 steps 状态都在 state.json，列表与指纹无需解析 YAML**。
- twin 裁决留痕 `orchestration/state/evals/taste/<case_id>.json`：字段含 `ts, case_id, case_file, case_head, choice, confidence, reason, parsed, body, record_kind, night/authority, origin`；已有 `review: {ts, verdict, note}` 字段的 = 已复核，**无 `review` 字段 = 待盲审**（`taste.pending_review_count()` 现成）。案卷文件（`case_file`，jury 案卷 JSON）含 `question/options{A,B}` 等。
- 行动指令/强制处理队列真源 = `mirror.MIRROR_QUEUE_PATH`（`orchestration/state/mirror-queue.jsonl`），行格式 `{ts, text, status, source}`，`mirror.pending_directives()`= 未 done 行。偏差报告进**同一队列**（不新建文件）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/{plans,taskbooks,queues,workq,taste,mirror,jury}.py`：既有数据层——本环节在其上**扩展/修复**，既有能力勿重写（PSEP：同一能力不留两份实现）。
- `data/others/original_designs/2_REFACTOR.md` §5.4（偏差报告队列）与 §10（mirror review 盲审反转设计）：盲审语义依据。
- `tests/skills/orchestration/test_member_queues.py`、`test_queue_dispatch_unified.py`、`test_live_config_panel.py`、`test_twin_trace.py`：既有断言（本环节改动需同步它们，保持语义不破）。

## 交付（5 处修改，全部是纯读数/写入函数扩展）

### 1. `plans.py`——性能与结构读数

- 模块级缓存 `_YAML_CACHE` / `_STATE_CACHE`：`dict[路径, (st_mtime_ns, st_size, payload)]`；`_load_yaml`/`_load_state` 先 `stat` 比对（mtime+size 未变即命中，不解析）。key 含解析后路径，测试 monkeypatch `PLANS_DIR` 后自然隔离。
- `overview_rows() -> list[dict]`：**只读 `.state.json`**（缺 state 的 plan 回落缓存 `_load_yaml` 取 goal）。每行 `{id, goal, mark, done, total, created, completed_at, abandoned_at, current: {id, status, session_id, task_file} | None}`；`mark` ∈ `{"▶进行中","✔完成","⊘中止"}`（语义同既有 `_plan_mark`：completed_at→✔、abandoned_at→⊘、否则▶）；`done/total` 按 steps 统计；`current` = 首个非 done/skipped 步。按 `created` 降序（缺 created 用 id 降序）。
- `plan_digest() -> str`：廉价指纹 = `hashlib.sha1` of `"|".join(f"{id}:{mark}:{done}:{total}:{每步 status 向量}")` 取前 12 hex。**只含概览可见字段；不得读 YAML、不得构建 detail**。
- `plan_tree_nodes(pid) -> list[dict]`：superstep 树结构化节点（解析 plan YAML 的树 + state 对账一次）：`[{id, path, kind: "super"|"leaf", mode, status, member, brief, children: [...]}]`。既有 `plan_tree_lines` 保留不动（返工卡片仍用）。

### 2. `taskbooks.py`——家族分组与 Markdown 详情

- `families() -> list[dict]`：返工家族归组读数。规则：`rework_of` 为空或指向不存在 id 的项 = 家族根（孤儿自成一家）；每家成员 = 沿 `rework_of` 的传递闭包（按 parent map 收集，防多子分叉）；家内排序 `rework_round` 降序、同轮 `ts` 降序 → `members` = [最新…本体]。返回 `[{root, latest, members, size}]`，家族列表按 `latest.ts` 降序（最新在上）。
- `family_label(state: dict) -> str`：`rework_round >= 1` → `"返工第 n 轮"`；否则 `"本体"`。
- `md_path(tb_id: str) -> Path`：公开 `<id>.taskbook.md` 路径（后续 VSCode 打开目标）。
- `detail_markdown(tb_id: str) -> str`：详情页 **Markdown 文档单一实现**。段序保持 `detail_lines` 现语义：delivered 态「★ 批复参考（流程报告）」置顶 → 任务书正文（`read_body`）→ 批语历史（rejections）→ 流程报告（`report_sections`）→ 返工卡片（`rework_card`）。**与 `detail_lines` 共享同一段落组装私件**（改一处两面生效，不留两份实现）；`detail_lines` 保留（静态通道/测试用）。
- `detail_lines` 尾部「批复入口：…」行**删除**（用户明令清理此类提示）。

### 3. `queues.py`——`water_lines` 成员完整性修复

现状缺陷（实测）：`if not c: continue` 把计数为空的成员整行跳过（9 队只见少数几行）；在途明细行 `· 在途 <id>…` 不带成员名，看不出归属。修复要求：
- **每个 `QUEUE_MEMBERS` 成员必出一行**（含全零计数，行尾标注「（空）」或等价明示）；
- 在途明细行**带成员名**（形如 `· [m] 在途 <id>（plan·step）起于 …`）；
- 合计行/「双零皆空」行语义保持；
- 同步 `test_member_queues.py`、`test_queue_dispatch_unified.py`、`test_live_config_panel.py` 中钉旧行格式的断言（`orch status`/`orch queues` 文本面同步变化属预期）。

### 4. `taste.py`——盲审读数与复核扩展

- `blind_queue() -> list[dict]`：无 `review` 字段的记录，按 `ts` 升序。每项 `{case_id, ts, case_file, question, option_a, option_b, evidence, case_head}`——案情复用 `jury.parse_case(case_file)`（或等价既有解析）取 question/options；案卷文件缺失时回落 `case_head` + 「案卷缺失」提示（可审不崩）。**绝不含 choice/reason/body/confidence/parsed（隐去 twin 答案）**。
- `blind_case(case_id: str) -> dict | None`：完整记录（含 `choice`），供提交时一致性比对。
- `ruling_rows(limit: int = 0) -> list[dict]`：已有 `review` 的记录，新→旧；行含 `{case_id, ts, choice, confidence, reason, review, night, authority, case_head}`。
- `apply_review(case_id, *, verdict, note="", blind: dict | None = None)` 扩展（**向后兼容**：旧调用不传 blind 照常工作）：`review` 落 `{ts, verdict, note, blind}`；既有 journal 记账行为不变。`blind` 约定：`{user_choice, agree: bool, remedy: "restore"|"none"|"other"|None, requirements: str}`。

### 5. `mirror.py`——偏差报告队列（2_REFACTOR §5.4 同队列同待遇）

- `append_deviation(rec: dict) -> dict`：向 `MIRROR_QUEUE_PATH` 追加一行（`kind="deviation"`），返回落盘行。
- `pending_deviations() -> list[dict]`：`kind == "deviation" and status != "done"`。
- `deviation_lines() -> list[str]`：待决与验收页用的偏差摘要文本行（每条一行：case_id、处置、原因摘要）。
- `_line_queues` 中「偏差报告队列 = 后续单…」字样改实数（`= len(pending_deviations())`）。
- 偏差行 schema（与既有 `{ts,text,status,source}` 行兼容，`pending_directives()` 自动可见）：

```json
{"ts": "…", "kind": "deviation", "source": "blind-review", "status": "pending",
 "text": "偏差报告 <case_id>：用户选 A vs twin 选 B；处置=需恢复",
 "case_id": "…", "user_choice": "A", "twin_choice": "B", "reason": "…",
 "remedy": "restore|none|other", "requirements": "…", "taskbook_id": "…(可选)"}
```

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿（含被同步的既有测试）。
2. 性能钉（tests 内新增断言）：monkeypatch `plans._load_yaml` 令其抛异常后，`plans.overview_rows()` 与 `plans.plan_digest()` 仍正常工作（证明只读 state.json、不解析 YAML）。
3. `taste.blind_queue()` 每项 dict 的键集合不含 `choice`/`reason`/`body`/`confidence`/`parsed`。
4. `queues.water_lines()` 输出中 `queues.QUEUE_MEMBERS` 每个成员 id 都出现在对应行首（含全零成员）；在途明细行含成员 id。
5. `mirror.append_deviation({...})` 落盘行可被 `mirror.pending_deviations()` 取回；`mirror.deviation_lines()` 非空。
6. `taskbooks.families()` 对现有 `data/taskbooks/` 实测输出家族结构正确（含返工家族 `tb-20261015-194834-3` 一族：最新=返工轮次最大者）；`taskbooks.detail_markdown(tb_id)` 返回含「★」批复参考段与流程报告的 Markdown 文本（delivered 态）。
