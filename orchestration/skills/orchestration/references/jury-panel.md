# Tier C 仲裁团（`orch jury`）调用面与处置依据

> REFACTOR v1.0 §4；组卷/双序面=E3-prep（`20261011-233003-leader`），逐席 provider
> 传输面=E3-live（`20261012-164600-leader`）。本页是**组长侧**读 `orch jury` 输出后的
> 处置依据；**配置面/Key/回退告警/渠道纪律见 `references/jury-channels.md`**；权威设计
> 与用户裁决原文见 `orchestration/REFACTOR.md` §4/§5。

## 0. 定位与法院模式

评审团是**家族外、直连、阻塞**的成对比较判决面：不经 Qoder、不给裁判任何工具，
裁判**只能看见案卷**（用户裁决 v0.4：裁判自行读代码会引入 agent loop，且各裁判所见
不同会破坏双序一致性与材料哈希）。因此**材料组装责任在撰写者侧**，CLI 的补偿是自动
附着确定性材料（§2）。

硬规则：席位清单里**任何含 `qwen` 字样的型号 → 拒开庭**（排除同族自恋偏差）。选型权
归用户（账号与付费面是用户的），代码不做能力判断、不内置候选清单。

## 1. 案卷文件（撰写者交付物）

`orch jury <case-file>` 读的是一份固定形状 markdown：

```markdown
targets: src/pysci/skills/orchestration/tools/jury.py; orchestration/state/evals

# 案卷标题（可省——有下面三个小节即可）

## 问题
这次基建改动该采纳还是维持现状？（一句话，裁判据此判决）

## 选项 A
实施改动：改动内容 + 收益 + 代价

## 选项 B
维持现状：现状为什么够用 + 不改的代价

## 证据（可选）
复现命令、实测输出、失败案卷路径——写**事实**，不要写倾向
```

- 小节名认 `## 问题` / `## 选项 A` / `## 选项 B`（也认 `Option A` / `A`）；缺任一
  → `✗ 案卷不可用` + `[NEXT]` 打印格式说明，**rc=2**，不发调用。
- 可选 `targets:` 行（放在任意位置，分号/逗号/空格分隔）声明涉事路径，决定自动
  附着材料抓什么；不写则只附 git 面与台账。
- **案卷全文就是判决书正文**：写不清 = 裁决不可信。两选项须同体量、各自可独立执行。

## 2. 自动附着材料（CLI 确定性抓取，非撰写者所写）

案卷尾部自动加 `## 自动附着材料` 段：`git diff --stat`、`git diff` 全文（超 4000 B
截断并标注原文大小）、`git log --oneline -5`、`targets` 涉事文件的**字节读数**
（字节数/行数/sha256，不给正文，避免把代码塞进上下文）、相关台账行（按涉事文件名与
case id 匹配）。整段先过 `scrub()` 再进案卷。

**材料冻结**：规范 AB 序全文在 CRLF→LF 规范化后取 sha256[:16] 入 `dossier_hash`
（同 `sync.tree_hash` 的行尾教训）。裁决记录里的哈希即「裁判到底看见了什么」的
可复核指纹；事后改材料 → 哈希变 → 旧裁决可被指认为过期。

## 3. 调用面与四档退出

```bash
pysci-orch jury <case-file>            # 组卷 + 开庭 + 计票 + 落留痕（阻塞）
pysci-orch jury <case-file> --dry      # 只组卷打印（含附着材料与哈希），零 API 调用
pysci-orch jury <case-file> --timeout 600   # 覆盖整体等待上限（默认 300 s）
```

| 状态 | 打印 | 落盘 | rc | 组长处置 |
| --- | --- | --- | --- | --- |
| `recorded` | 逐席票 + 一致性 + usage/cost | `state/evals/jury/<case>.json` | 0 | 照 `[NEXT]` 给的 `approve`/`reject` 执行 |
| `dry` | 完整案卷 + 材料哈希 | 无 | 0 | 检查材料够不够，再正式开庭 |
| `queued` | 降级原因 + 队列路径 | `state/evals/jury-queue/<case>.json` | 1 | 修因（选型/Key/渠道）后重跑同命令；成功重裁自动销队列条目 |
| `declined` | 点名违约型号 | 无 | 2 | 改 `registry.jury.seats` 去掉 Qwen 系（不入队：配置面改一下就通） |

**仲裁面绝不 raise**：缺 Key、seats 未选型、registry 读坏、无一席可开庭、超时、
渠道 5xx/429、返回体不可解析、正文被截断——全部降级 `queued`。基建决策不因仲裁
不可达而停摆（§4）。

## 4. 双序、计票与 escalate

每席跑两局：A/B 与 B/A 各一次（同一材料、两种摆法；请求默认 `temperature: 0`、强制
JSON 输出 `{choice, reason, confidence}`，逐席 `params` 可覆盖，见 jury-channels.md）。

- 票的 `choice` 经 `flip()` 映射回**真实选项**后计票；
- 同席两局结论不一致 → 该席记 `inconclusive`，**两票都不计**（位置偏好被识别而非
  被 averaging 掩盖，仍留在记录里供事后复盘）；
- 有效票**少数服从多数**；
- `verdict` 为空（无有效投票）、或**平票**、或 `inconclusive` 超过有效席半数
  → `escalate: "twin"`。E6（双副本 A/B 实裁）之前这案**挂起呈组长**：由你带证据
  上报用户，不要凭直觉硬判。
- 单局瞬时失败（网络/5xx）→ 整案重跑一次（`attempts=2`）；两轮仍有失败 → `queued`。

两条 E3-live 实测渠道纪律（**同组织两局串行**、**截断算渠道错误不算无效票**）的
成因与修法见 `jury-channels.md` §5——它们改变的是「为什么入队」，不改变计票规则。

## 5. 冻结简报与「待用户确认」

`state/evals/jury-brief.md` 是护栏级统一裁决规则（项目优化面 / 既有裁决约束 /
**v0.6 直通规则**：「仅涉提议者自身专有基础设施的提案，除非有明确危害，默认 approve」
——组长主观直通豁免已取消并移入此处）。

当前是**草案 v1**：批准方式是文件里加一行 `approved: <ISO 时间>`（代码只认这枚标记）。
未批准时 `orch jury` 每次打印「简报未批准」警告但**照常可用**；批准前请勿用它做不可
撤销的裁决。

## 6. 留痕记录字段

`state/evals/jury/<case-id>.json`：`case_id` / `case_file` / `ts` / `question` /
`options` / `targets` / `dossier_hash` / `dossier_chars` / `dossier`（裁判所见全文）/
`brief{path,approved,hash}` / `seats`（逐席公开状态行，含 `role`/`defect`/`reason`/
`usable`；**无 Key 值**）/ `fallback{enabled,used,notice,missing_primary}` /
`panel`（本庭实开庭型号）/ `attempts` / `jurors`（逐席：两局票、`consistent`、计票
选择）/ `votes` / `tally` / `inconclusive` / `verdict` / `escalate` / `usage` /
`usage_by_model` / `cost{amount,unit,note,priced,unpriced}`。E3-prep 的顶层 `base_url`
已随逐席化移除。

重裁同案覆盖旧记录（以最新为准），并自动删除同案的 `jury-queue/` 条目。

## 7. 边界（本轮交付未做的事）

- **`--dry` 不校验 seats/Key**：组卷面与开庭面分离，先 dry 后开庭是省钱顺序；
- 平票/inconclusive 的 **twin 终裁**（双副本 A/B 实裁）属 E6，未实现；
- **`pricing` 出厂为空**：三家刊例须由用户按账号实价填（可能挂套餐/Agent Plan），
  缺席时留痕记 `cost.amount=null` 并附 note；
- **minimum 火山方舟档位**：调查结论=不可行，维持 flash（三道硬阻塞与替代路径见
  `state/suggestions/approved/20261012-164600-leader.md` 回填段）。
