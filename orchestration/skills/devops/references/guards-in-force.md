# guard 面的**在法活体哨兵**（backlog 20261010-193601-devops）

> devops 技能分册。约定不等于机制——首版 delivery-gate 的 TDZ 事故（13 例违规被
> 外层 catch 静默吞掉、「全绿」不代表在执法）不能靠「写进文档大家注意」防住。
> 本册记三层机械化的实现位置与判据，改 guard 或加新 guard 前照此对齐。

## 三层机械化的分工

**① guard 源自证 · fail-open 必须留痕**

`orchestration/guards/delivery-gate.mjs` 与 `pod-guard.mjs` 里，每个 catch 分支必须
调本文件的 `failOpen(where, err)` helper 向 stderr 打 `[<name> fail-open] <原因>…`；
顶层再包一层 `try { run() } catch (e) { failOpen(…); process.exit(0) }` 兜底。语义
仍是 fail-open（不阻塞生产），但**必须留痕**——观测面唯一能识别「guard 有 bug 但
看起来一切正常」的信号。

判据的**字面量**是 `[<name> fail-open]`（前缀精确）：`guards-in-force` 元哨兵与
`guard_probe.py` 都以此为关键字断言，改名即断链，故 helper 里写死。

**② doctor 活体探针 · `pysci-dev doctor` 现含**

`src/pysci/skills/devops/tools/guard_probe.py`：Temp fixture 造违规 pod（AGENTS.md
12KB / 写 charter.md 只读层），`node` 直跑真 guard，**双断言**——
（a）violation → exit 2 且 stderr 含具体理由关键字（`AGENTS.md` / `只读层`）；
（b）bad stdin → exit 0 且 stderr 含 `[<name> fail-open]`。
整段 <1s、无 headless 依赖，**不需要** agentsMdExcludes 那种无时间戳台账——
每次 doctor 现场跑即可。status 三档：`ok`（都过）/ `bad`（任一断言失败，计入
doctor rc=2）/ `inconclusive`（node 缺失或 guard 文件不存在，advisory 不硬失败——
与 fail-open 语义一致，观测层不阻塞生产，但也不掩盖）。

独立跑：`PYTHONPATH=<repo>/src python -m pysci.skills.devops.tools.guard_probe`。

**③ `guards-in-force` 元哨兵 · 结构性防漏**

`tests/skills/orchestration/test_guards_in_force.py`：扫 `orchestration/guards/*.mjs`
列表，逐 guard 断言三件事——
（a）源含 `[<name> fail-open]` 字面量；
（b）`tests/skills/orchestration/` 下至少一份 pytest 直跑该 guard（引用
`ORCHESTRATION_ROOT / "guards" / "<name>.mjs"` + `node` subprocess）；
（c）其中至少一个 `assert rc == 2` 阳性用例。

**新增 guard 时**没落实三件套会立即红——把「约定」变成机制。文件枚举用 parametrize
ids=名字，红名直接指出「缺什么、加哪儿」。

## 回归面与本会话补齐

- `tests/skills/devops/test_guard_probe.py`：探针本身端到端；含「静默 catch 退化」
  「空壳 guard」两种反例（把当前 guard 复制改写、探针必须报 bad）——保证未来把
  `[fail-open]` 留痕或 enforcement 撤回时，回归立即红。
- `tests/skills/orchestration/test_pod_guard_blocks.py`：pod-guard 侧首次补齐阳性/
  阴性/fail-open 三重覆盖（此前只有 delivery-gate 侧有 pytest 直跑用例）。
- 可路由 check 建议：`registry.json` 的 `checks` 块加
  `"guards-in-force": { "cmd": ["uv","run","pytest",
  "tests/skills/orchestration/test_pod_guard_blocks.py",
  "tests/skills/orchestration/test_guards_in_force.py",
  "tests/skills/devops/test_guard_probe.py",
  "tests/skills/orchestration/test_delivery_gate_budget.py","-q"] }`——本会话因
  main registry.json 心跳 dirty 与 branch 修改冲突不能自改，见交付 `<infra_suggestion>`。

## pod-guard 白名单顺序（本会话调整）

旧顺序下系统 tmp 分支在 pod 检查**之前**：pytest `tmp_path` 造的 fixture pod 位于
Temp 内，写入只读层会被 tmp 分支放行——既让本会话新写的活体探针与阳性回归失真，
也让生产里任何把 pod scaffold 建到 Temp 的场景（未来的 dry-run 建 pod 等）绕过
只读保护。新顺序：**pod → readonly/deployed 检查（block 或 exit 0）→ 未命中 pod 才
走 tmp/taskDirs 兜底放行**。判据字面：readonly 前缀命中即 exit 2，与是否在 Temp
无关。

## 工具面自检与两侧同形（backlog 20261012-010508-devops / 20261012-025325-devops）

**pysci-dev 跨树自检**：`pysci` 以 editable 装入，`pysci.paths.PROJECT_ROOT` 由**包文件
位置**决定、与 cwd 无关。在 worktree 目录里漏设 `PYTHONPATH=<wt>/src` 跑
`pysci-dev sync/doctor`，用的是 main 的代码面、读写的也是 main 的真本/副本/台账，
对脚下那棵树什么都不做**且照样退出 0**（`doctor` 还会据 main 状态给出「健康」的假结论）。
`dev.py::cross_tree_warning()` 在子命令派发前判定**脚下有没有自己的 `pyproject.toml`**
——worktree 就摆在 `PROJECT_ROOT/.qoder/worktrees/` 之内，所以「cwd 不在项目根下」这种
直觉判据无效。判为跨树即打印两侧根路径 + 修法并 `rc=2`；`--force` 表示知情放行；
无主之地（Temp fixture，找不到任何 pyproject）静默放行，不误伤合法用法。

**两侧属性面必须同形**：交付块属性模式在 Python（`delivery._ATTRS`）与 Node
（`delivery-gate.mjs` 的 `ATTRS`）各写一份，`test_delivery_quote_aware_attrs.py::
test_both_sides_declare_the_same_attr_pattern` 钉两者字面量相等（只改一侧=读数漂移的
源头）。裸 `[^>]*` 会在 `reason="优于 > 阈值"` 的引号内部提前收尾：解析侧正文与产物
path 被污染而 `kind` 仍 result、`warning` 仍空（严格面自认匹配上了，兜底面不触发——
比 parse_error 隐蔽，台账一片绿灯），gate 侧则把 `tags=`/`kind=` 截在属性面之外从而
看不见标签、跳过词表校验。引号段**禁跨行**：未闭合引号若跨行找配对会吞掉整段交付，
代价是跨行畸形落 `parse_error` 而非兜底救回（同行未闭合引号仍由宽松面捞回并附
`PROTOCOL_LAG_HINT`）。
