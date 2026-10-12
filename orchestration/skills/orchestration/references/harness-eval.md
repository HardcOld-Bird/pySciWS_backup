# Tier B harness 压测套件（orch eval）

Tier A（tier-a-tags.md）是**被动**故障统计，Tier B 是**主动**压测：拿一组固定案卷，
以最低成本档真派发，测「这套 harness 现在能不能干活、比上一轮好还是坏」。设计出处
`REFACTOR.md` §3/§1.1；A/B 副本与套件归属见下（E4）。

## 通道归属（E4 移交后必读）

| 面 | 归属 | 命令 | 台账 `runner` |
|---|---|---|---|
| 组长**抽检** | 组长 | `orch eval run` / `eval set-baseline` | `leader` |
| 套件本体（案卷/基线/A/B/判决口径） | **bencher** | `orch bench-add`/`bench-up` → `pysci-bench` | `bencher` |

- 两条通道写**同一个** `state/ledger/evals.jsonl`，靠 `runner` 字段分开；E2 前的旧行无此
  字段，读法按 `leader` 回退（不是缺陷）。对照/回填基线时**必须带 `--runner`**，否则判决
  成了两条通道的平均数。
- devops 侧的 `orch eval` 过渡授权（用户裁决 2026-10-12）已随 E4 退役：devops 只维护
  checks 注册表与本文件的机械部分，不再自跑压测。
- 基线捕获与副本 A/B 的机制、污染闸、清理纪律：bencher 技能 `references/bench-ops.md`；
  组长侧入队要领：`references/drain-watch.md` §3。

## 命令面

```
uv run --no-sync pysci-orch eval run --case <id>        # 跑一案
uv run --no-sync pysci-orch eval run --all              # 跑全套（按文件名序）
        [--model-tier max|flash|minimum]      # 覆盖案卷档位（默认案卷里的 minimum）
        [--label <轮次名>]                     # 本轮打标，供 set-baseline 精确回取
uv run --no-sync pysci-orch eval set-baseline [--label <轮次名>] [--runner leader|bencher]
        # 以最近（或指定）一轮回填基线；E4 后不带 --runner 就是混合通道（慎用）
```

**退出码**：`0`=本轮全部通过；`1`=有案失败；`2`=案卷读不到（目录缺失/指定案不存在/
形状不合法）。二元判决，无「基本通过」档。

## 案卷格式（`state/evals/harness/<case-id>.md`）

头部是 `键: 值` 标记区，`## 任务书` 小节起是**派发正文原文**（组员看到的就是它）：

| 键 | 含义 |
|---|---|
| `member:` | 派给哪个 pod（必填） |
| `tier:` | 档位，默认 `minimum` |
| `max_turns:` / `timeout_s:` | 轮数与墙钟预算（默认 12 / 900） |
| `dirs:` | 任务级白名单，逗号/分号分隔（pod-guard 用） |
| `产物: <路径> [check=<id>]` | 可多行；check 复用 `registry.json` 的 checks 表 |
| `断言:` / `断言于:` | 正则；缺省断言于交付正文，`断言于:` 改断言某项目根相对文件 |
| `禁止标签:` | 出现这些 `<blocked tags>` 即判失败（词表 `state/failure-tags.json`） |
| `权限拒绝上限:` | pod-guard 拒绝次数上限（默认 0） |
| `说明:` | 案卷意图，供人读，不参与判决 |

判据写错**当场抛错**（未知键、重复标量键、产物行不可解析、预算非整数、断言非法正则、
正文为空）——不允许「静默少测一项」。

## 判决构成（任一命中即 fail，理由全列出）

交付不是 `<result>`；出现禁止标签；权限拒绝数超上限；声明产物不存在 / 机械验收 fail /
check 类型未注册（`unrouted` 也算失败——套件要的是确定性）；断言未命中。

机械验收由套件自己跑（`do_dispatch(..., no_checks=True)`），不经派发路径，避免
「门禁故障不阻塞生产」的 fail-open 把压测也放软。

## 隔离性（为什么压测跳不进生产账）

压测派发固定 `session=new` + `attach_replies=False` + `eval_hop=True`：全新会话保证可比；
不吃审批回复也不移档 `sent/`（组长的批复留在原位，下一跳生产照常生效）；`eval_hop` 令
这一跳**不写成员生产台账、不落建议队列**。指标行落 `state/ledger/evals.jsonl`（独立台账），
故压测成本与轮数**不污染** `orch stats` 的成员排名与耗时画像。

**组员自维护层同样在隔离面内**（E2 验收实测：压测跳把「这一跳学到的东西」写进了
figure 自己的 `AGENTS.md`——git 跟踪面，于是测量噪声变成合并前检查的脏面）：
`run_case` 两头夹住这一跳。任务书之后附 `EVAL_HOP_NOTE`（本跳不得更新 `AGENTS.md`、
不得新建/修改自建 rules·skills）；派发前后各拍一次**该 pod 范围**的跟踪脏面快照，
只还原本跳新增的脏面（跳前已脏的一律不动——那可能是组长在飞的工作），还原路径进
台账行的 `reverted` 字段并在行尾点名。反复出现 `reverted` 说明纪律附注没拦住，
该在案卷 `说明:` 里收紧任务书而不是加重事后还原。

## 基线对照

`eval run` 末行输出本轮 vs `state/evals/baseline.json` 的 before/after 表。
**改善** ⟺ 通过数不减 且 均值 num_turns 不劣化，且至少一项严格变好；同分即「未改善」
（无平局档）。`_mean_turns` 丢弃 num_turns=0 的行，防「崩溃跑」靠零轮数冒充改善。
无基线时输出 `无基线` 并提示先 `set-baseline`。

样本量提醒：三案的均值轮数是一轮真派发的**单次测量**（E2 实测同案两轮 4↔10 跳），
判决读方向不读小数；要下结论看逐案 turns 与 `预算用满` 标记，而不是均值本身。

读数口径（E2 实测缺陷，backlog 20261012-075200-leader）：`-o json` 的 envelope 是
**本次进程调用**的读数，而 `do_dispatch` 在首跳崩溃/额度耗尽后会 `resume=True` 再跑一跳——
直接取末跳会把轮数、耗时、credits 和 pod-guard 拒绝次数一起做低（实测某案末跳报
turns=2、transcript 里 50+ 条 assistant 消息）。`dispatch._accumulate_hop` 已把各跳
累计成整任务读数，故台账 turns 与 transcript 可直接对账（取
`~/.qoder-cn/projects/<pod键>/<sid>.jsonl` 的 assistant 条数核）。

## 案卷设计要点（避坑）

- 判据必须**可由组员正常达成**：把历史上的 blocked 案改编成正向断言（结果合规 +
  不出现该失败标签 + 零权限拒绝），而不是造一个只能失败的案。
- 产物一律落在 gitignored 面（pod `bench/`、`data/research/eval_scratch/`），否则反复跑会把
  测量噪声变成 `git status` 脏面，挡住合并前检查。
  **但 gitignored 不等于每轮全新**：残留产物会让组员「复核现成的」而不是「生产」，
  测出来的轮数是复核成本（E2 实测 fig-basic 第二轮起 2 轮即过，因为
  `eval_scratch/fig-basic/eval_band` 还在盘上）。取数前先清 `dirs:` 指向的目录。
- `figure-audit` 只以「宽度偏差 >±2mm / 字号 < tick×0.9 / 构建失败」出 ERROR，面板
  字母与交付宽度是 WARN——`--panels` 永不致 fail，别把它当判据主力。
- 断言锚点须先实测取确定性结果（如 `pysci-research search '<精确标题>' --source arxiv
  --json` 的 `arxiv_id`），不要凭印象写 ID。
- **预算贴线＝判决不可信**（2026-10-12 E2 验收实测）：`blocked-recheck` 案内
  `max_turns: 10`，真派发跑出 turns=10/10 的 pass——判决是被轮次上限截断出来的，
  模型多做一次自证就会翻成 fail。`eval run` 会在行尾打 `turns=N/M` 并在 M 用满时
  点名 `[!] 预算用满`，对照表同样标注；见到就放宽案内 `max_turns`（预算按任务最小可
  达成量 ×1.4 留余量），别把贴线读数写进结论。
- **一轮只跑一次，别并发**：`eval run` 不锁 pod 也不锁 `dirs:` 产物，两轮同时跑会互踩
  产物并把同 label 混成多行（E2 实测：并发轮 fig-basic 14 与 16 轮两条读数，顺序轮
  9 轮）。同案同档全新会话的**两次判决可以相反**（blocked-recheck 并发轮 1 pass/1 fail
  ——断言锚在交付正文，minimum 档是否复述边界有运气成分），故单次测量不是结论，
  要回 `evals.jsonl` 看同一 label 的重复度再谈改善。
