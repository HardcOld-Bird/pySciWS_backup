# 阶段史、已知代价与风险、降级手册（README §12/§14/附录 A 全量）

> 历史沿革（Phase 0–4、E0–E7、M1–M7 收官）与已知代价/风险/降级操作；由 README §12/§14/附录 A 迁入。

## 12. 阶段计划

### Phase 0 —— 验证冲刺（**已完成**，2026-10-07/09，沙盒 `orchestration/tmp-p0` 已清理；结果如下）
- [x] 泄漏面：rules/AGENTS.md 向上注入 pod（是）；skills/项目级 MCP/settings 不泄漏
      （否）；pod 嵌套目录继承信任、自身配置正常加载（标记实测）
- [x] 裸命令 PATH 方案：~/.bashrc 追加后新登录 shell 可用（实测）
- [x] 自动压缩：本会话 jsonl 含 `isCompactSummary`×2 与数十 compaction 标记——摘要式
      延续、非截断，运行时已实际发生；envelope 另提供 `usage.context_usage_ratio`
      可做归档提示阈值
- [x] 会话池全操作：`--session-id/--name/--resume/--fork-session/--list-sessions/
      --delete-session` 全部可用；**坑**：list/delete 是跨项目聚合视图且按序号删除
      （序号漂移）→ orch 会话池索引以 registry.json 为权威，prune 须先按 sid 匹配
      序号，匹配失败仅做 registry 标记不删文件
- [x] yolo 用户级默认在 headless 生效：无权限旗标执行 Bash，`permission_denials` 为空
- [x] envelope 字段清单：`result / session_id / num_turns / is_error / stop_reason /
      duration_ms / duration_api_ms / total_credits / total_cost_usd / usage(含
      context_usage_ratio、cache 字段) / permission_denials / modelUsage`
- [x] `--mcp-config + --strict-mcp-config`：pod 会话仅见自有 server（工具可枚举=已
      连接），用户级 5 个全屏蔽；根 settings `mcp.excluded` 可精确 block 指定 server
- [x] delivery-gate（Stop hook）：无标签收尾被 exit 2 拦截、stderr 注入改正指令、
      成员补标签后放行（turns 1→2）；防循环 `stop_hook_active` 生效；yolo 下照常执行
- [x] pod-guard（PreToolUse）：只读层写拦截（stderr 原样透传给成员）、白名单区放行；
      JSON 解析失败时 fail-open（诚实条款一致）
- [x] 组员自维护层三形态：always_on 注入 / model_decision 只注描述、正文按需 /
      glob 规则在读取匹配文件后加载；自建 skill 正常发现
- [x] pod cwd 下 `uv run` 向上解析根项目、共享 `.venv`（`pysci.__file__` 指向主树）
- [x] `--add-dir` 语义：**yolo 默认下不是硬边界**（外部 cwd 无 add-dir 亦可读写项目
      文件）→ 路径纪律的唯一硬防线是 pod-guard；「副组长全项目访问」无需 add-dir
- [x] `-m` 档位传参：`-m Qwen3.8-Flash` headless 可用；`--list-models` 含
      Qwen3.8-Max/Flash（双档映射成立）
- [x] 后台长任务：130 s 与 **3600 s**（BG-OK-3600）后台 Bash 任务均存活并事件驱动
      通知，等待期零轮询；TUI 空闲下通知正常送达
- [x] 等待-恢复链路：多跳任务中途按 PID 精确击杀 → 会话 jsonl 增量落盘 →
      `--resume` 精确还原中断现场（已完成步骤定位 + 暗号记忆无损）
- [ ] 长会话成本曲线：resume 成本随跳数增长——转 Phase 1 运行期持续观察（envelope
      的 usage/cache 字段直接入台账）

### Phase 1 —— orch 骨架 + figure 纵切面（**基本完成**，2026-10-09）
- [x] `src/pysci/skills/orchestration/`：registry / dispatch / envelope 解析 / 交付解析 /
      会话池封装 / 台账（credits+耗时）/ 下一步提示（plan 与 approve 后置 Phase 2）；
      ruff 全绿；`pysci-orch` 入口注册（pyproject）
- [x] `orchestration/pods/figure/`：charter 第一版（交付协议/自检清单/CLI 范式/轻量
      笔记原则）、AGENTS.md 种子、pod-guard + delivery-gate 接线（guards 正式版单测
      8/8）、mcp.json（空集）、bench/inbox/outbox
- [x] 技能真本迁移第一份：scientific-plotting → `orchestration/skills/` + manifest +
      sync（部署/一致性校验通过；根技能列表热更新确认移出）
- [x] orchestration 技能真本第一版（组长操作知识）+ 部署到根 `.qoder/skills/`
- [x] paths.py 新增 ORCHESTRATION_ROOT / ORCH_STATE_ROOT / PODS_ROOT 锚点
- [x] 冒烟派发全环路：任务书落 inbox → 新会话（Flash 档位路由生效）→ charter 加载 →
      bench 可写 / charter 只读拦截（组员侧逐字回报）→ 交付协议解析 → 台账/会话池登记
- [x] 真实图件任务全环路（fig0_orch_smoke）：首跳 blocked（白名单与两层结构冲突，
      触发改进循环首次实战，见 §2 脚注）→ 审批+扩权续派 → 正规管线 build →
      视觉自检 → **figure-audit 机械验收 √ PASS** → 台账/审批回复附送/归档全链路自动
- [x] credits/耗时基线登记：冒烟跳 7 轮/48 s；fig0 阻塞跳 23 轮/251 s；收尾跳
      15 轮/137 s；会话 ctx 三跳后 52%；**观察**：envelope `total_credits` 恒为 0
      （疑似该账号/模型不上报计费；成本统计暂以耗时+轮数为准，持续观察）
- 验收：✅ 达成——真实任务全环路无人工兜底（blocked 分支的组长决策属设计内环节）

### Phase 2 —— 计划驱动 + 改进循环 + devops（**完成**，2026-10-09）
- [x] orch plan new/run/amend/drop/list/adhoc：状态机全生命周期实测（登记/延迟书写
      强制/对账/中止审计）；带派发的完整 run 与 dispatch 同源（dispatch 已实战）
- [x] orch approve/reject/suggestions/backlog/consult/stats：命令面实装 + 冒烟通过
      （consult 在 deputy 缺位时优雅降级）；approve 全链路待首个自然建议到来时实战
- [x] `orchestration/pods/devops/`：charter（worktree/git 纪律/巡检职责）、devops 技能
      真本+部署、`src/pysci/skills/devops/` + `pysci-dev`（doctor/sync/worktree）；
      registry 注册（max 档 + extra_dirs=项目根）
- [x] worktree venv 策略**实测定型：PYTHONPATH 覆盖法**——
      `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests`（复用主 .venv，worktree 代码优先；1452 项收集 7.5 s，
      全量 45 s）；worktree 内裸 uv run 不可用（无环境）；仅依赖变更时才 `uv sync
      --project <wt>`（重）。已写入 devops 技能
- [x] **首个完整改进循环实战**：backlog 队首（figures new 路径打印缺陷）→ orch
      dispatch devops --from-backlog（任务书自动生成）→ devops worktree 实施 +
      回归测试新增 + 全量 1416 passed → 合并 commit（cz 规范，未 push）→ 交付含
      backlog id → orch 自动销账。实测：31 轮 / 855 s / Max 档 / ctx 29%
- [x] orch archive/sessions/prune + 归档前蒸馏跳：实测通过——蒸馏跳中成员**自主整合
      升华**了此前任务经验进 AGENTS.md（STYLE.yaml 缺陷规避、数据管线约定），
      「越来越聪明」机制成立；归档后池默认正确回退工作会话
- [ ] 模型双档路由实装（registry 映射 + `-m` 传参）

### Phase 3 —— reviewer + 副组长（**完成**，2026-10-09）
- [x] `orchestration/pods/reviewer/`：charter（VERDICT 协议/独立性纪律/审查流程）、
      `rubrics/generic.md` **最小占位版 v0.1**（用户裁决：跑通流程优先，正式评审体系
      ——科学正确性/创新性/美观性 + 专属技能与程序基础设施——待框架稳定后用户与
      组长深度共议）、文献 MCP（arxiv/zotero/paper-search）、registry（max 档 +
      `readonly_extra: ["rubrics"]` 只读保护——成员级只读层扩展机制首用）
- [x] `orch review`：VERDICT 解析（单测过）+ FAIL 自动回派 + 复审一次 + 升级仲裁
      状态机；审查记录持久化 `state/reviews/`
- [x] **review PASS 链路实战**（reviewer 首战，fig0_orch_smoke）：独立复跑 audit CLI、
      手算抽查数据自洽性、Read 图件视觉判读——占位 rubric 下仍展现审查素养；
      7 轮/106 s。reviewer 并提出高质量建议（review 任务书应嵌原生产任务目标陈述，
      否则 G2 依赖生产者自述、与独立性纪律冲突）→ **approve 全链路实战通过**
      （建议→审批→backlog 第 6 位→回复入队附送）
- [ ] FAIL→回派→复审链路：代码已实现，待首个自然 FAIL 实战（占位 rubric 判定从宽，
      人为构造 FAIL 成本不划算——Phase 3 验收以 PASS 链 + 状态机代码审查为准）
- [x] `orchestration/pods/deputy/`：全 7 技能部署（**9 技能真本迁移提前全部完成**，
      根 `.qoder/skills/` 只剩 orchestration 部署副本）、平级协作条款 charter
      （拒绝权流程/异议义务/全项目视野）、extra_dirs=项目根、MCP 不配 comsol
      （license 纪律，仿真走 CLI 或改派 sim）
- [x] **consult 实战**（deputy 首战）：异议义务成立——副组长按 charter 先校正了任务书
      两处前提错误（根技能已清空、README Phase 3 未登记），再以实证给出
      「**脚手架批量建、战备逐个验收**」的 Phase 4 策略（引 figure pod 首任务即
      blocked 的先例：charter/白名单冲突只有实战暴露）。组长采纳。
- [x] 跨域双路径对比演练 → **延后至 Phase 4 末**（全员就绪后成本对比才有意义；
      用户裁决跑通流程优先）

### Phase 4 —— 全员迁移 + 组长 harness 瘦身（**基本完成**，2026-10-10）
- [x] 六专职 pod **脚手架批量建**（devops worktree 任务：sim/writing/theory/lit/
      drawing/model3d，charter 含各域自检清单，MCP 归属落位，manifest targets 追加 +
      sync；doctor 十 pod 全绿；34457f5 合并）
- [x] **lit pod 战备验收**（首个战备抽查）：MCP 隔离会话级实锤（只见 arxiv/zotero，
      无 comsol/blender）、charter 加载、arxiv 实检索成功；成员自主把 429 限流经验
      蒸馏入 AGENTS.md。**其余五 pod 战备验收随各自首个真实任务进行**（预期暴露各域
      特有冲突，参照 figure/lit 首战模式处理）
- [x] 根 `.qoder/skills/` 清空（Phase 3 提前完成）
- [x] **用户级 MCP 清零**（用户授权任何时机执行；备份存 state/backups/，gitignored）：
      根侧新会话实测零末端 MCP；lit pod strict 配置不受影响。组长本 TUI 会话的 MCP
      工具随下次重启消失。paper-search-mcp 预先存在的连接失败入 backlog
      （20261010-paper-search-mcp-down）
- [x] basic.md 重写：§3 PowerShell 细节压缩为「Git Bash 为唯一工作 shell + PS 历史
      纪律四行摘要」（完整实测结论存 git 历史）；CLI 裸命令范式入公约数；§1/§2/§6
      原文未动
- [x] orch 移交 devops 维护：自本登记起，orch/pysci-dev 代码改动一律走 backlog →
      devops worktree 实施（组长不再直接改 orch 代码；bootstrap 例外条款作废）
- [x] devops 巡检例程：pysci-dev doctor 已上线（Phase 2）；巡检节奏 = 组长按需派发
      + 每个 Phase 收尾必跑；transcript 抽查入 devops 技能清单
- [ ] 跨域双路径对比演练 → **延后至首个真实跨域科研任务**（届时 naturally 产生
      「副组长单体 vs 计划接力」对比数据，stats 记录）
- [ ] （可选加固）组长侧 permission deny：禁 Write/Edit 于生产目录——暂缓（组长
      仍需写 orchestration/state 与本文档；deny 规则需精细豁免，收益/复杂度比待评估）

### 自改进体系 E0–E7（原 REFACTOR，**收官**：2026-10-11 定稿 → 2026-10-13 全链完成并入本文）

逐阶段成功判据与实施实录见 git 历史中的原设计文档；下表是收官状态（机制落点见括注）：

- [x] **E0 纪律与原料层**：`success_criteria` 必填、`kind` 结构化、reject=override 强制
      留痕、决策日志回灌、deputy/reviewer 降档（§5.3/§6）；
- [x] **E1 Tier A 全链**：failure-tags 受控词表、`<blocked tags>` 解析、`orch stats`
      by_failure、阈值自动立案（§5.4）；
- [x] **E2 minimum 档 + harness 套件骨架**：`orch eval run`、`state/evals/harness/` 首批案、
      baseline.json（§5.5）；
- [x] **E3 仲裁团**：直连三席、双序对局多数解析、jury-brief「仅涉自身基建直通」规则、
      Key 泄漏哨兵（§5.3 J 通道）；
- [x] **E4 bencher**：pod/FIFO 非常驻、基线捕获、副本 A/B、择机查看型通知（§5.5）；
- [x] **E5 决策流硬化**：判决包四态闸、inspector 全流程（自动晋升/双队列迟滞/全局开关/
      设计审查 rubric）、E5b 交付后核验闭环与缺陷×blocked 例外通道（§5.3/§6）；
- [x] **E6 twin + mirror + 回退链**：`orch taste` 冷启动、盲审/口述分流/偏差报告绿色
      通道、`escalate=twin` 自动终裁、`pysci` 首批全景项（§3.1/§4.1/§6）；
- [x] **E7 twin 上线**：代理裁决接入原「呈用户」升级点（护栏级审批/升级仲裁/Phase 验收），
      最终交付验收仍呈用户本人；2026-10-13 PREFERENCE v2 经用户确认转正、twin 全量（§1/§13）；
- [x] **收尾：融合与退役**——生效机制并入本文（§1 角色表、§3.1、§4.1、§4.5、§5.3–5.5、
      §6、§9、§13），原设计文档删除、仅存 git 历史（即本轮收官单）。

**实施期夜航要点（审计留痕，原委记录要点）**：
- E0 首战四缺陷：registry lost-update（drain 陈旧快照回写抹配置）、watch QUEUE-COMPLETE
  脱靶吞账、锁面 released 墓碑、`.done` 收尾硬证据否决判活——全部修复并有回归（§4.5/§6
  同源纪律）；
- 2026-10-12 00:04 E1+E2 组包 `run_failed(tool_use)`：E1 半成品经抢救单续作、E2 重排队
  ——该次处置直接促成 §9 的 **WIP-salvage 先例**；
- 委托期（10-11 23:2x → 10-12 07:00）到期撤销，期内自主决定全量可追溯于决策日志
  （`kind=delegation`）；**后续愿景（用户裁决 2026-10-12）**：把「永不阻塞」做成受硬机制
  保证的项目特性——除最终交付外系统不因等待用户停摆，代理授权点全量留痕、事后可追溯；
- 2026-10-13 收官审查：ctx 水位闸绕过判决通道案 → 回退该闸 + **修复形状分流**（§5.3）+
  **后台决策审计面** `pysci decisions`（§6）——三项均经用户裁决并落码。

### 交互式控制台 + 全员 FIFO 队列化（REFACTOR-20261014，**收官**：M1–M7）

> 蓝图＝施工期根目录的 REFACTOR 蓝图（M0 用户批准五项裁决后开工；**收官后蓝图退役**，
> 裁决并入 §13、结果并入本节）。用户提案两部分：`pysci` 交互式控制台 + 全员 FIFO
> 队列化与统一事件/唤醒。

| 阶段 | 结果 |
|---|---|
| M1 | TUI spike：**Textual 8.2.8 定案**（退路 rich+prompt_toolkit 未启用）；裸 `pysci` 入口（tty→TUI / 非 tty→status） |
| M2 | TUI 主菜单七模块 + 更新水位（`state/mirror-seen.json`）+ 详情视图框架 + 模块归并迁移 |
| M3 | 用户任务书：`data/taskbooks/` 落盘 + 状态机（`new→planning→active→delivered→accepted\|rejected`）+ `TASKBOOK-NEW` 唤醒 |
| M4 | 成员 FIFO 队列（7 专职+deputy+reviewer，基于 `workq` 原语）+ worker 生命周期（非常驻、per-member 锁、孤儿回收、队空写 `.done`）+ reviewer 附属语义（宿主+review 合并=1 通知单元） |
| M5 | 统一事件/唤醒层（`state/events.jsonl`：JOB-DONE / STEP-REVIEW-DONE / QUEUE-EMPTY / TASKBOOK-NEW＋legacy 迁入）+ `plan advance` 事件驱动 + 多 plan 并行；`plan run` 保留兼容 |
| M6 | backpressure 写侧接线（活跃会话注册表 + 周期心跳）+ 全队列 defer 延期（带宽类失败回挂 `defer_until`、双闸触顶停链上报） |
| M7 | 改名清扫（旧 `pysci-*` 控制台入口退役、全仓活面零命中）+ 命令式面转 LLM 通道 + 文档并入 + 蓝图退役 |

**残余风险（如实记载）**：M1 的交互验证全在 **Pilot 无头驱动**下完成——它证明的是**交互
逻辑**，**不等于**真实终端的渲染与键盘事件可用。Git Bash/mintty、Windows Terminal、
PowerShell 5.1 下的**实机验证待用户**（devops 会话非 tty）；若实机发现 Textual 不兼容，
按**既定退路**（rich + prompt_toolkit）换栈并改写 M1 结论。

**实施要点（审计留痕）**：队列形状全部走 `workq.py` 原语、锁/停止/run 名复用
`drain.WorkerPaths` 族（§0.9 共享逻辑做库，不留第二套）；事件层单一流、`events drain`
幂等＝**合并唤醒**（防唤醒风暴）；向后兼容硬要求满足——`plan run`/`dispatch` 阻塞路径与
既有 watch 事件语义零改动。

---

## 14. 已知代价与风险

- **orch 是体系咽喉**：plan-first 使组长对 orch 依赖更深——§4.3 红线（fail-open +
  降级手册 + state 可读格式）不可妥协；orch 自身的改进走 backlog 循环；
- **组员自维护层的治理**：写坏记忆会持续污染该成员后续会话；自建内容膨胀会推高每跳
  上下文税。缓解：charter 编写纪律（AGENTS.md 控体量、笔记分类进 rules/skills 按需
  加载）+ git 可回滚（入库部分）+ devops 巡检（体量/质量抽查）+ 台账成本曲线暴露
  异常增长；
- **flash 档位质量风险**：专职组员默认 flash，复杂环节可能力不从心。缓解：计划环节
  可升档 max；blocked/返工率入台账，数据驱动调整成员默认档位；
- **声明式机械验收的滥用面**：一律 check="none" 会架空验收。缓解：台账透明 + 抽检 +
  reviewer 可见验收状态；Phase 1 观察声明行为再决定是否加软约束；
- **resume 成本随会话增长**：原生自动压缩兜底上下文窗口，但摘要有细节损失——重要
  约定及时蒸馏进自维护层，不依赖会话历史；归档阈值由台账数据定；
- **basic.md 泄漏税**：注入全员，保持精简（其「只增不减是失败信号」条款在此更重要）；
  根目录不建 AGENTS.md（建了即注入全员）；
- **reviewer 主观性**：rubric 用户审定 + VERDICT 证据义务 + 组长仲裁逃生门；
- **yolo + guard 组合的失败面**：Bash 绕过（诚实条款 §9）；transcript 审计是唯一事后
  手段；
- **PATH 持久化的边界**：仅覆盖 Git Bash 新会话；PowerShell 手动使用需另配 profile
  （可选）；旧会话快照不含新 PATH（fallback uv run 兜底）；
- **TUI 实机终端兼容待验（§12 M1 残余风险）**：`pysci` 交互面（主菜单 / 详情 / 键盘事件）
  的验证全在 **Pilot 无头驱动**下完成，证明的是交互逻辑、**不等于**真机可用；
  Git Bash/mintty、Windows Terminal、PowerShell 5.1 的实机渲染与按键**待用户实测**——
  若 Textual 不兼容，按蓝图退路换 rich + prompt_toolkit 并改写 M1 结论。

---

## 附录 A：降级手动操作手册（orch 故障时）

组长手动派发模板（Git Bash；**必须用原生 exe**，`.cmd` 包装器剥引号）：

```bash
EXE="$HOME/.qoder-cn/bin/qoderclicn/qoderclicn.exe"
POD="orchestration/pods/figure"
# 新会话（续会话：把 --session-id 换成 --resume <sid>；sid 查 --list-sessions）：
"$EXE" --cwd "$POD" -p "$(cat $POD/inbox/task-XXX.md)

--- 交付协议：最终回复须含 <result> 或 <blocked>（互斥），可附 <infra_suggestion>；
产物用 <artifact check=...> 声明。" \
  --session-id "$(uv run --no-sync python -c 'import uuid;print(uuid.uuid4())')" \
  --name "任务slug" -m <型号> --max-turns 50 -o json > /tmp/envelope.json 2>&1
uv run --no-sync python -c "import json;print(json.load(open('/tmp/envelope.json'))['result'])"
```

审批回复附送：把 `orchestration/state/replies/<member>/*.md` 内容拼进 -p 文本。
机械验收手动运行：按 check 类型对应各 CLI 审计子命令（如 `uv run --no-sync pysci-figures audit
<产物>`）。台账手工补记（credits/耗时见 envelope 字段，Phase 0 确认字段名）。
