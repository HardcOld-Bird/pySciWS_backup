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
