# 派发 / 成本 / 降级细则（技能 SKILL.md 的展开）

SKILL.md「派发要领 / 归档与成本 / 降级」是压缩版；本文件补回被压缩掉的细节。动派发参数、
读成本账、或 orch 降级手动派发前先读这里。

## 会话选择策略

- **默认 latest（续最近活跃）**：`dispatch <member>` 不带 `--session` 时续该成员最近活跃的
  会话——同一工作的连续环节靠它攒上下文，省重复交代。
- **无关新工作**用 `--session new --name <slug>`：另起会话，别把不相干的活塞进活跃会话
  （上下文互相污染）。
- **续更早的某会话**：先 `sessions <member> --all` 查池，再 `--session <sid>` 指定；会话池
  水位/归档见 README 与 `orch sessions --archive <sid> --distill`。

## 成本读数（BYOK）

- BYOK 模式：**max 档计量、flash 档恒 0**（flash 走包月额度，账单上不计增量）。
- `orch stats` 的覆盖率读数写作 **「覆盖 M/N 跳」**：M=有计量的跳数、N=总跳数——flash 跳
  不进 M，故 M<N 是正常形态，不是漏记。
- 耗时/成本异常增长的成员会话 → 提请组长归档轮换（见 devops 「巡检清单」）。

## 降级派发（orch 瘫痪时）

- orch 不可用 → 照 README **附录 A 手动模板派发**：**原生 exe 直调**，
  **勿用 .cmd 包装器**（`.cmd` 包装器会在 Git Bash 下把参数当路径改写、并吞掉退出码，
  令手动派发的读数与真实不符）；台账手工补记，随后尽快 `plan adhoc` 派 devops 修复。
- 手动派发同样**长任务后台运行**；不要 sleep-轮询。
