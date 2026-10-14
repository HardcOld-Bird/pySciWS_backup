# 配置实时化 + 队列面板校核 + 空终端根治（用户实测反馈 2026-10-14）

## 事故根因（已取证，供实施参考）

用户在 `data/pysci_settings.toml` 改「入峰提前量」，改的是**被 `#` 注释的示例行**
（`# leadtime_s = 600`，工作区 diff 实证）——解析器读不到，程序仍用 `DEFAULT_LEADTIME_S
= 3600`，面板显示不变。另有三值不一致：代码默认 3600 / toml 示例 600 / prose 注释
「10 分钟」。**配置键注释示例化 = 用户改不动的陷阱**，本单根治。

## 目标（四件，缺一不验收）

1. **配置生效性（事故根治）**：
   - `leadtime_s`（入峰提前量）与新增 `boundary_hold_s`（边界延迟派发窗口，见 4）在
     toml 里以**生效键**写出（显式默认值，不注释）——用户改值即生效；非法值逐键回落
     出厂并告警（沿现校验语义）；
   - 用户目标值 **leadtime_s = 900**（用户口述；其工作区编辑实为 600/10min，实施时按
     900 落生效键并在注释注明二者差异待用户终确认）；
   - **合并纪律**：用户对 `data/pysci_settings.toml` 的未提交修改先吸收进产物再落盘，
     worktree 合并不得覆盖丢失；
   - 三处矛盾注释/默认值统一为单一真源（代码 DEFAULT、tomml 注释、README、prose）。
2. **路由面板实时化**（tui/modules「系统与路由」详情）：
   - 读数实时反映生效配置（生效值 + 真源标注：「toml 显式」或「出厂默认」）；
   - **倒计时每秒跳动**：「距下一模式切换还有 X 分 X 秒」（TUI `set_interval` 刷新），
     含下次切换时刻与目标模式（入峰/入谷）；
   - 提前量、峰段表同面板展示。
3. **队列与运行面板校核**：
   - 「各成员队列皆空」口径拆分清晰：9 成员队列 / 支撑席存量（bencher/inspector）分别
     报告，杜绝「真的为空吗」类误读；
   - 事件水位读数与 `events-seen.json` 真源一致性测试（用户曾见「已排空至 0 待排空 3」
     的滞后快照——验证面板读数即时反映 ack_seq）；
   - PENDING-WORK 判据覆盖支撑席队列核对（bencher/inspector pending 无 worker 应属
     「有活没人干」）；**inspector 开关=off 期间降噪**：注明「开关关闭不拉起」，不反复
     唤醒告警。
4. **边界延迟派发（boundary_hold_s）迁 toml**：
   - 现值在 registry（`routing.boundary_hold_s` / `reg.routing_hold_s`，dispatch
     `_boundary_hold`）——迁 `data/pysci_settings.toml` 生效键可配，registry 回退兼容
     （读一次旧位并出声迁移提示，沿 leadtime 同款语义）；
   - 同面板显示（当前窗口秒数 + 「临近边界将延迟派发」状态 + 倒计时）。
5. **空终端取证与根治**（用户报告桌面有一个空终端，身份未明）：
   - 取证：枚举 python/orch/watchdog 相关进程与其可见控制台（WMI/PowerShell
     Get-CimInstance 经 Python subprocess 传 argv，勿走 shell 的 `/` 参数坑），确认
     身份。假设候选：schtasks `pysci-watchdog` 每 5min tick 的可见 console（action=
     `python -m …orch watchdog start`，交互令牌）；某前台 `_watchdog-loop` 实例；某
     spawn 点漏 `CREATE_NO_WINDOW`（drain._spawn_detached 已有，需清点 queueworker/
     benchd/inspectd 的唤醒 spawn）；
   - 根治：全部分离进程拉起点统一隐藏窗口策略；schtasks action 换隐藏宿主（pythonw/
     wscript VBS/等价）；实测留痕：拉起 watchdog 与 worker 后**桌面无新终端窗口**、
     计划任务每 5min tick 不闪窗。

## 验收预期（机械）

- `uv run --no-sync pytest` 全绿（1407+ 基线）+ 新增：toml 生效键读取（改值即生效、
  非法回落告警）、boundary_hold_s toml/registry 兼容、面板倒计时刷新（假时钟）、
  事件水位真源一致性、PENDING-WORK 支撑席覆盖与 inspector-off 降噪；
- 实测留痕：改 toml 值 → 面板读数同帧变化（复现命令+输出）；空终端取证结论+修复后
  无窗验证；
- 代码/toml/README 数值注释一致性（grep 交叉核对，判据按 20261014-075230 约定）。

## 白名单（涉改文件）

`data/pysci_settings.toml`（**含用户未提交修改的吸收**）、
`src/pysci/skills/orchestration/tools/**`（routing/dispatch/tui/modules/queues/
watchdog/orch）、`orchestration/scripts/watchdog-scheduler.ps1`、`tests/**`、
`orchestration/README.md`、`.gitignore`（按约定已附）。

## 纪律（违约即返工）

- 单一真源：配置值只认 toml 生效键（registry 只留回退读），显示层只调用读取函数；
- **不许丢用户工作区修改**（data/pysci_settings.toml 先吸收后落盘，实施说明里注明）；
- grep 判据约定（20261014-075230）：禁词运行期拼接、判据限 `*.py`；
- 不动 pyproject.toml；确需改则改后必须 `pysci-dev venv-sync`；
- 编辑项目文件用文件编辑工具；CLI 首选裸 `pysci-X`；
- 交付按 pod 交付协议 XML（`<result>`/`<blocked>`）；blocked 带 tags 与 reason；
  改进建议带合法 `kind` 属性。
