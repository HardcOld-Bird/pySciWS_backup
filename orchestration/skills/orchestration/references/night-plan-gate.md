# 夜航 × Plan 模式两环节（EnterPlanMode / ExitPlanMode）调研记录

> backlog `20261017-112800-devops`。**结论：按「钩子不可接管」处理，本期不实施**（符合用户
> 「不可以亦可不改进」）；夜航离机前的官方无人值守路径是 `auto` 权限档（零弹窗）。

## 层级归属结论（**官方文档核查**，非 TUI 实测）

依据 Qoder CLI 官方文档（`tools.md` / `permissions.md` / `plan-mode.md` / `hooks-reference.md`）：

1. `EnterPlanMode` / `ExitPlanMode` 是**内建工具**（`tools.md`「规划工作 — 进入或退出 Plan Mode」；
   `permissions.md` 亦点名 `general.plan.enabled: false` 时两者**不注册**）；
2. 但 `plan-mode.md` 明言 Plan 是「**一个独立的工作状态，而不是权限策略**——可与任意权限模式
   共存」，**没有任何文档说明这两个工具的批准会触发 `PreToolUse` 或 `PermissionRequest`**；
3. 因而「钩子可接管」**未经证实**；`matcher` 为 tool_name 正则（语法上可写 `ExitPlanMode`），
   `permissions.allow` 的 `ToolName` 条目同样**只是语法合法、未证生效**。

## 为什么本单不做 TUI 实测

本单由 **headless 组员**执行，无交互终端可观察「是否弹窗 / 钩子是否触发」。故只落**文档核查
结论**并给出实测配方，由组长/用户在 TUI 中实测后定层。

## TUI 实测配方（若日后要定层）

1. 配 matcher `EnterPlanMode|ExitPlanMode` 的 `PreToolUse` 钩子（把 `tool_name` 记日志）；
2. 配 `PermissionRequest` 钩子，对二者返回 allow：
   ```json
   {"hookSpecificOutput":{"hookEventName":"PermissionRequest",
     "decision":{"behavior":"allow","updatedInput":{},"updatedPermissions":[]}}}
   ```
3. TUI 里进/出 Plan：**若弹窗且钩子日志为空 → 独立 UI 层，钩子够不着**（维持本结论）；
   **若钩子被触发 → 走权限管道**，届时可按夜航开关放行（参考 `night-gate.sh` 的 ON/OFF 现读模式，
   OFF 恢复人工批准、零行为变化）；
4. 对照 `permissions.allow` 加 `"EnterPlanMode"` / `"ExitPlanMode"` 两条规则，看能否免弹窗。

## 政策修正

夜航政策文⑥ 原写「工具权限弹窗属硬阻塞——它不写钩子接管」（PSEP_2 时期定性，已过时）。现改为：
`PermissionRequest` **可**代替用户批准（夜航 ON 时对白名单工具放行）、未覆盖工具仍靠会话权限档位；
Plan 两环节属**独立工作状态、未证钩子可达**，离机前切 `auto` 档或人工确认。
