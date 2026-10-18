# 组长改进（L 环节）与改进项三标志（用户 2026-10-17 裁决）

返工/初次派发的完整 plan **末尾**可选追加「组长改进」环节（L）：组长反思 harnessing
摩擦 → 拆解改进派单 → plan 内派 devops 实施。触发＝TUI 复选框（新建「包含组长改进」；
reject 三复选），`[taskbooks]` 可设默认值。**工具自动追加、组长零手写**。

## 1. L 结构（返工骨架与初次 plan 同构；恒居顶层末尾）

```yaml
- id: L
  mode: serial
  brief: 组长改进
  children:
    - {id: l1, kind: self, role: leader_reflection, brief: harnessing 摩擦反思,
       evidence_sections: [摩擦反思, 改进清单, 结论]}
    - {id: l2, kind: self, role: leader_improvement_plan, brief: 拆解改进派单（空清单合法）}
    - {id: l3, member: devops, brief: 改进实施, member_placeholder: true}
```

- **就绪性机械保证**：顶层 `steps[]` 是隐式 serial 组（planflow 就绪谓词）——前序全
  done 才轮到 L，「任务主体全部完成后才进组长改进」不靠自觉。
- **注入点**：返工骨架由 `build_rework_plan` 按标志组装；初次 plan 在 `taskbooks link`
  时由 `plans.ensure_leader_step` 追加（幂等键＝叶子 `role == leader_reflection`）。
- **l2 用新 role** `leader_improvement_plan`：复用 `improvement_plan` 会与 A.a4 在
  报告 role 收集处（`taskbooks._role_texts`，按 role 为键）互相覆盖。
- **正式要求文本**＝`plans.LEADER_IMPROVE_PROMPT`（l1 引导时全文打印；要点：指导是否
  充分／功能介绍是否合理／信息获取步数／orch 触发点附送会不会更好／有无误导／积极
  立单修复）。
- **空改进清单合法**：l2 写明「本轮无需改进，空清单」→ `plan advance <pid> --skip L.l3`
  收尾（`--skip` 只对 pending 叶子＝「从未需要做」，不是撤销）。这是空清单**唯一**合法
  路径。
- 报告「组长改进」段＝l1+l2 的 evidence（通用段：非返工件含 L 时也出）。

## 2. 改进项三标志（真源 `state["improve"]`）

| 标志 | 含义 | 返工骨架 |
|---|---|---|
| `infra` | A 综合项目改进（a1 成因分析→a5 改进实施） | 有则加 A |
| `leader` | L 组长改进环节 | 有则顶层末尾加 L |
| `inspector` | C 监察改进（c1 可发现性送审→c4 实施） | 有则 B∥C 并行组，否则 B 单独 |

返工执行 B 恒在；8 种组合皆合法。旧三档（`full`/`no_inspector`/`rework_only`）为
**派生兼容**字段 `mode` 的来源（leader 不参与派生）：full={infra,inspector}、
no_inspector={infra}、rework_only={}。`orch plan rework` 改进项来源优先级：
`--improve`（显式）＞ `--mode`（兼容）＞ 任务书标志（默认）。
