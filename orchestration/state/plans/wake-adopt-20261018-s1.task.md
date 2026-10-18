# 任务：主会话程序化识别＋自动补位（wake-adopt-20261018 s1-adopt）

## 目标

**用户 2026-10-18 直接裁决（在场指令，高于 twin；twin wake-bind-20261018-twin 的 A
裁决已 `twin review --reject` 覆核留痕）**：唤醒权归属必须**程序化硬保证**、零人工
命令，不依赖组长「记得」跑任何命令。判据（用户给定不变量：子会话存在时主会话一定
存在；主会话存在时子会话不一定存在）：

- 候选集 = **存活 ∧ cwd==项目根** 的会话；
- **主会话 = 候选集中创建更早者**；候选集仅一个时即它；
- owner 记录（上次接管的会话 id）指向者仍存活 → 保持/恢复为 owner（watchdog 被杀
  重启后按记录恢复）；
- owner 缺席/死亡 → 创建更早的存活项目根会话**自动补位**；
- 晚创建者在早者存活期间**永不**夺主；pod cwd 会话永不入候选（s2 cwd 闸已落，保留）。

## 现状（s2 已落，演进而非回退）

已落且**必须保留**：cwd 闸（watch.py `cwd_is_project_root`/`_session_cwd`/钩子
`PYSCI_HOOK_CWD` 取样）、夜航差分/注入/gate 的 owner 过滤、消费权只认 owner 侧、
「事件不丢」验收件（替身不消费→owner 迟到唤醒）。

需演进：s2 按 twin A 移除了**全部**自动转让（stale 恒 deny、SessionEnd 写释放墓碑
且不许自动升主）——按本单改回**自动**补位，但判据换成「创建更早的存活项目根会话」，
彻底废除心跳判死（旧 stale-30min 的病根：长回合心跳停摆被误判死亡）。墓碑改为纯
记录（释放事件留痕），**不得**再阻止自动补位；`wake-claim --force` 降为应急通道。

## 工作项

1. **会话登记表**（运行态，如 `state/sessions-live.json`，.gitignore 按约定附）：
   钩子自登记 `{session_id, cwd, pid, first_seen}`——SessionStart（≈会话创建）/
   首个钩子写入，SessionEnd 注销；watchdog/探活负责清理死会话登记。
2. **创建时序判据**：优先取可锚定会话进程的 OS 创建时刻（psutil `create_time()` /
   GetProcessTimes，锚定方式实测留证）；锚不到回退 `first_seen`（SessionStart 时刻，
   ≈创建时刻）。平局按 session_id 字典序（确定性）。交付写明实际取值与理由。
3. **存活性判据**：优先 **PID 探活**（Python subprocess 传 argv 调 tasklist/psutil，
   ⚠ Git Bash 下 `/FI` 会被 MSYS 转路径造成假阴性——勿经 shell 拼 Windows 参数）；
   仅当 PID 缺失才回退心跳新鲜度。**心跳过期而 PID 存活 = 存活**（长回合保护，
   测试钉死）。
4. **归属判据单点**（watch.py `hook_decision`/helper）：owner 存活→不动（含 watchdog
   重启恢复）；owner 缺席/死亡→最早创建的存活项目根会话自动升主；候选空→不动
   （事件留水位迟到唤醒）。晚创建会话恒 deny。
5. **wake-claim 面**：`status` 输出新判据读数（候选集/创建时序/owner 存活性）；
   `--force` 保留为应急通道（非常规）。
6. **保留 s2 成果零回退**：夜航 owner 过滤的「owner」改按新判据取值；pod cwd 闸、
   消费权 owner 侧、事件不丢回归全绿。
7. **文档**：orchestration/README.md §4.5、orchestration/references/wake-and-events.md、
   orchestration/skills/orchestration/references/patrol-watchdog*.md 同步新判据
   （`.qoder/rules/leader-only.md` §4 已由组长按用户裁决更新，勿动）。

## 测试与合并

- 用例钉死：①两会话并存，早者恒 owner、晚者恒 deny——**含「早者心跳过期（长回合）
  而晚者心跳新鲜」场景，早者仍保 owner**；②早者 PID 死亡→晚者自动补位；③唯一
  存活会话自动成为 owner（含冷启动）；④owner 记录跨 watchdog 重启恢复（记录指向者
  存活则不动）；⑤SessionEnd 注销后剩余存活会话自动补位（墓碑不阻止）；⑥pod cwd
  会话永不入候选/永不升主；⑦夜航面只对 owner 触发；⑧事件不丢回归（替身不消费→
  owner 迟到唤醒）。
- 双会话模拟实测留痕（不同 session_id/cwd/PID 假造或真 CLI）；
- 全量 `uv run --no-sync pytest` 绿后合并（devops 交付协议、worktree 纪律）。

## 白名单（写权限）

- .qoder/hooks
- src/pysci/skills/orchestration/tools
- tests/skills/orchestration
- orchestration/README.md
- orchestration/skills/orchestration
- orchestration/references
- .gitignore

## 验收预期

1. **零人工命令**：冷启动、两会话并存、主死亡、watchdog 被杀重启、SessionEnd 五类
   场景下 owner 归属全自动正确；
2. 两会话并存时创建更早者恒为主、晚者零自动唤醒（长回合场景不误判）；
3. 主死亡后存活项目根会话自动补位（仅一个时即它）；
4. 「事件不丢」实测验收保持通过；
5. s2 成果零回退（夜航 owner 过滤、pod cwd 闸、消费权 owner 侧回归绿）；全量 pytest
   无回归。

## 纪律

- 编辑项目文件用文件编辑工具；Git Bash 多词参数用单引号；
- 不动 pyproject.toml（确需改 → 改后必须 `pysci-dev venv-sync` 并说明）；
- 机械验收不过不许合并；交付 XML（<result>/<blocked>）；建议带合法 kind；
- 本单判据=用户 2026-10-18 直接裁决（高于 twin A）——实现中若判据不可行走
  `<blocked>` 上报，不得自行改道。
