# pysci-orch 命令全表（技能 SKILL.md 的展开）

SKILL.md「命令面（速查）」只给骨干；本文件是逐行语义与旗标。

```
pysci-orch status                        # 全景：计划/成员/队列/worker/台账/夜航
# —— 日常面 ——
pysci-orch plan new <计划.yaml> [--id X]  # 登记计划（格式见 README §4.2）
#   └ id 缺省取**文件名 stem**（建议显式 --id）；YAML **复制**进 `state/plans/<id>.plan.yaml`
#     （源任意位置、登记后可删；**勿与目标同名**——同名未定义）
pysci-orch plan advance <id> [--text/--task] [--evidence <报告.md>]  # 事件驱动：入队即返回（M5，不阻塞；成员含 devops）
#   └ --evidence：收尾 kind:self 组长环节（文件在 + 标题段 lint 过才置 done，PSEP_3 §2.1）
#     └ 落点 `data/taskbooks/<id>.<leaf>-<名>.md`；骨架=`# <标题>`+对应标题段（如 `## 拆解`）
#   └ 旗标（原同步子命令已废，旗标并入本命令）：
#     advance 旗标：--text --task --session --name --model-tier --effort --max-turns --dirs --dry --evidence
#     （--session 默认 latest，**续作余量已尽必须 new**；--evidence 见上行）
pysci-orch plan rework <tb-id> [--mode full|no_inspector|rework_only] [--plan-id X] [--force]
        # 返工任务书 → 三档骨架（A/B/C）并登记（§4.2）；[NEXT]：link → advance（不走 plan new）
pysci-orch next [--limit N]              # 跨任务路由：优先级→动作类别→序；任务卡片 + 简表（PSEP_3 §3.2）
pysci-orch taskbooks list|show <id>|link <id> --plan <pid>|sync|report <id>  # 用户大任务书（report=全流程报告）
pysci-orch queues list | queues enqueue <member> --text "..." [--review]  # 成员 FIFO 队列（含 devops）
pysci-orch events drain | events list  # 统一事件流：排空（幂等）/水位
pysci-orch plan amend <id>               # 编辑 plan YAML 后校验对账
pysci-orch plan edit <id> --step S (--brief|--append-brief|--dirs)  # 结构化改环节字段
#   └ **改环节别手编 YAML**（`plan new` 重排折行，old_string 对不上）：说「改什么」即可
pysci-orch plan adhoc <member> --text "..." [--dirs d1,d2]  # 计划外单步：**同步阻塞到完成**
#   └ 内部走阻塞推进原语（等在途收尾）⇒ **与 `plan advance` 的「入队即返回」相反**；
#     长任务**必须后台**（run_in_background），否则撞 Bash 超时（backlog 20261015-211200）
pysci-orch plan list | plan drop <id>
pysci-orch suggestions [ack <id> --as superseded|decision --by <ref> --why "<理由>"]  # 待审批（超 24h 醒目）
#     └ ack＝**第三态销账**（非否决、不进 backlog，移入 `closed/`）；立单面 `supersedes: <id>[,…]` 随 approve 联动销账
pysci-orch approve <建议id> (--jury | --bench <案id[,案id]> | --override "<理由>")
        [--note "..."] [--no-wake] [--criteria "<判据>"] [--solo]  # 四态必居其一
        # 例外=缺陷×当跳 blocked；判决包 verdict-package.md
#   └ 组长立单（**无命令**，手写即立案）：落点 `state/suggestions/<ts>-<member>.md`
#     （`<ts>`=YYYYMMDD-HHMMSS；`[member]` 列＝后缀＝**来源**成员，代补单写**目标**成员）
#     正文须含 `kind: defect|usability` 与判据标记；实质正文（去空白/标题/标记行）≥40 字，否则标 `[!] 空壳建议`。
#     流程：手写 → suggestions 自检 → approve/ack → 销账
#     └ 判据标记**接受形态**：行首／列表项 `- `／粗体 `**…**`／标题 `## `（标题形态
#       2026-10-16 起支持）；正文＝标记行后（**空行可跳过**）的连续非空行。
#       最小样例：`kind: defect` → `source: leader` → `fingerprint: <短名>`
#       → `success_criteria:` → 换行后 `- <判据>`（每行一条）。
#   └ **重复建议单**：`resolve` 不收建议单 id（作用域=backlog 条目）；走
#     `reject <id> --override "<合并至哪单>"`（历史面；以 `--override` 正文为准）
pysci-orch watch [--timeout 14400]  # 等 drain 事件（drain-watch.md §2）；漏挂由 status 补账
pysci-orch wake-claim [status] [--force [--session-id id]]
        # 唤醒权读数；归属由钩子按 session_id 自动完成（README §4.5）
pysci-orch reject <建议id> --override "<判决依据>"   # 否决：依据必填（J/M 判决的驳回不计）
pysci-orch deputy consult "<问题>"               # 咨询副组长（平级：异议义务已注入）
pysci-orch twin consult <案卷.md>  # 咨询 twin（rc=2 分型：未启用 / 冷启动须 --session new）
pysci-orch review <产物> --origin <member> [--rubric generic]
        [--goal-from <原生产任务书> | --goal "<目标陈述>"]  # --goal* 供 G2 独立核验
        # 审查链：VERDICT → FAIL 回派返工 → 复审 → 二次 FAIL 仲裁
pysci-orch jury <案卷.md> [--dry|--timeout S]  # Tier C 仲裁团（双序对局；不可达入队 rc=1）
#   └ 案卷骨架五段：`## 问题`/`## 选项 A`/`## 选项 B`/`## 证据`/`## 送审结论`（twin 同构前三段）
pysci-orch bench eval (--case <id>|--all) [--label L]  # Tier B **抽检面**（runner=leader）
pysci-orch bench baseline [--label L] [--runner leader|bencher]  # 回填 baseline.json（不指定=混合，慎）
pysci-orch bench add <run|capture|ab|check> [--cases a,b] [--ref-a R --ref-b R]
        [--label L] [--tier minimum] [--brief T] [--start]  # 测量任务入队
pysci-orch bench start [--dry] | bench queue [--all] | bench seen <id> | bench stop
        # bencher 生命周期（FIFO 非常驻）；判决进 status「待查看」不即时唤醒；要领
        # drain-watch.md、harness-eval.md
pysci-orch twin taste <案卷.md> [--dry]  # twin 判断通道（未生效即拒；twin-taste-mirror.md）
pysci-orch inspector start [--dry|--force] | inspector status | inspector patrol [--target 菜单id]
        [--start] | inspector assign --brief T [--start] | inspector enable|disable  # 巡检（§7；start=拉起，enable/disable=开关）
pysci-orch stats [--plan id|--member m|--days N]
# —— 低频顶层 ——
pysci-orch requeue <条目id> [--note "…"]   # 回排：needs_leader → pending 并唤醒 drain
pysci-orch resolve <条目id> --note "…"     # 销账：确认已达标的 needs_leader 条目 → done（--note 必填）
#   └ 两者作用域**同**：backlog 条目且 **status=needs_leader**；**不收建议单 id**（见立单）
pysci-orch watchdog start|stop|status       # 看门狗：start 幂等 / stop 广播纪元停全部
pysci-orch devops stop                       # 请求 devops worker 协作式停止（计划器 + FIFO worker）
pysci-orch refresh-models                   # 按 --list-models 刷新 registry 模型映射
pysci-orch twin review [case] (--list|--confirm|--reject) [--note "…"]  # 用户复核 twin 裁决（否决入 override 账）
# —— 底层命令面（降级/调试）——
pysci-orch dispatch <member> (--task F|--text T|--from-backlog) [--session latest|new|<sid>] ...
pysci-orch sessions <member> [--all|--archive <sid> [--distill]|--prune <sid>]
pysci-orch ledger [--member --days --stats] | sync [--check]
```

成员 id：sim figure writing theory lit drawing model3d（专职）；deputy reviewer devops
bencher twin inspector（特殊）。就绪 pod 以 `orch status` 为准。

> 内部命令（`_watchdog-loop` 等 5 个）由钩子/派生进程调用，**不进 `orch -h`**。

**队列成员**（三路径可直派）＝ 7 专职 + deputy + reviewer + **devops**；非成员三闸均拒。
**devops 双通道**（直派／组包同队列）；bencher/inspector 专队列，**不经** `queues enqueue`。细则 `dispatch-playbook.md`。

用户控制台 `pysci`：裸跑进 TUI；`pysci status|journal|decisions`＝LLM 静态通道。

**四态审批**：`approve` 必带 `--jury`/`--bench`/`--override` 之一；例外=缺陷×当跳 blocked
（判决包结构 `verdict-package.md`）。
