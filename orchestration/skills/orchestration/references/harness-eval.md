# Tier B harness 压测套件（orch eval）

Tier A（tier-a-tags.md）是**被动**故障统计，Tier B 是**主动**压测：拿一组固定案卷，
以最低成本档真派发，测「这套 harness 现在能不能干活、比上一轮好还是坏」。设计出处
`REFACTOR.md` §3/§1.1（E2 面）；A/B 副本对照与 bencher 自有权属属 E4。

## 命令面

```
uv run pysci-orch eval run --case <id>        # 跑一案
uv run pysci-orch eval run --all              # 跑全套（按文件名序）
        [--model-tier max|flash|minimum]      # 覆盖案卷档位（默认案卷里的 minimum）
        [--label <轮次名>]                     # 本轮打标，供 set-baseline 精确回取
uv run pysci-orch eval set-baseline [--label <轮次名>]  # 以最近（或指定）一轮回填基线
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

## 基线对照

`eval run` 末行输出本轮 vs `state/evals/baseline.json` 的 before/after 表。
**改善** ⟺ 通过数不减 且 均值 num_turns 不劣化，且至少一项严格变好；同分即「未改善」
（无平局档）。`_mean_turns` 丢弃 num_turns=0 的行，防「崩溃跑」靠零轮数冒充改善。
无基线时输出 `无基线` 并提示先 `set-baseline`。

## 案卷设计要点（避坑）

- 判据必须**可由组员正常达成**：把历史上的 blocked 案改编成正向断言（结果合规 +
  不出现该失败标签 + 零权限拒绝），而不是造一个只能失败的案。
- 产物一律落在 gitignored 面（pod `bench/`、`data/research/eval_scratch/`——根
  `.gitignore` 已排除），否则反复跑会把测量噪声变成 `git status` 脏面，挡住合并前检查。
- `figure-audit` 只以「宽度偏差 >±2mm / 字号 < tick×0.9 / 构建失败」出 ERROR，面板
  字母与交付宽度是 WARN——`--panels` 永不致 fail，别把它当判据主力。
- 断言锚点须先实测取确定性结果（如 `pysci-research search '<精确标题>' --source arxiv
  --json` 的 `arxiv_id`），不要凭印象写 ID。
