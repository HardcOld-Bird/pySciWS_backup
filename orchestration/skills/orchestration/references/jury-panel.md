# Tier C 仲裁团（`orch jury`）调用面与处置依据

> REFACTOR v1.0 §4；E3-prep，backlog `20261011-233003-leader`。本页是**组长侧**读
> `orch jury` 输出后的处置依据；权威设计与用户裁决原文见 `orchestration/REFACTOR.md` §4/§5。

## 0. 定位与法院模式

评审团是**家族外、直连、阻塞**的成对比较判决面：不经 Qoder、不给裁判任何工具，
裁判**只能看见案卷**（用户裁决 v0.4：jury 不自行读代码——工具访问会引入 agent loop，
且各裁判所见不同会破坏双序一致性与材料哈希）。因此**材料组装责任在撰写者侧**，
CLI 的补偿是自动附着确定性材料（见 §2）。

硬性规则：`registry.jury.panel` 里**任何含 `qwen` 字样的型号 → 拒开庭**（排除同族
自恋偏差）。选型权归用户（SiliconFlow 账号与付费面是用户的），代码不做能力判断、
不内置候选清单。

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
| `recorded` | 逐裁判票 + 一致性 + usage/cost | `state/evals/jury/<case>.json` | 0 | 照 `[NEXT]` 给的 `approve`/`reject` 执行 |
| `dry` | 完整案卷 + 材料哈希 | 无 | 0 | 检查材料够不够，再正式开庭 |
| `queued` | 降级原因 + 队列路径 | `state/evals/jury-queue/<case>.json` | 1 | 修因（选型/Key/渠道）后重跑同命令；成功重裁自动销队列条目 |
| `declined` | 点名违约型号 | 无 | 2 | 改 `registry.jury.panel` 去掉 Qwen 系（不入队：配置面改一下就通） |

**仲裁面绝不 raise**：缺 Key、panel 未选型、registry 读坏、超时、渠道 5xx、返回体
不可解析——全部降级 `queued`。基建决策不因仲裁不可达而停摆（§4）。

## 4. 双序、计票与 escalate

每个型号跑两局：A/B 与 B/A 各一次（同一材料、两种摆法，`temperature=0`、强制 JSON
输出 `{choice, reason, confidence}`）。

- 票的 `choice` 经 `flip()` 映射回**真实选项**后计票；
- 同型号两局结论不一致 → 该型号记 `inconclusive`，**两票都不计**（位置偏好被识别而非
  被 averaging 掩盖，仍留在记录里供事后复盘）；
- 有效票**少数服从多数**；
- `verdict` 为空（无人有效投票）、或**平票**、或 `inconclusive` 超过有效裁判半数
  → `escalate: "twin"`。E6（双副本 A/B 实裁）之前，这案**挂起呈组长**：由你带证据
  上报用户，不要凭直觉硬判。
- 单局瞬时失败（网络/5xx）→ 整案重跑一次（`attempts=2`）；两轮仍有失败 → `queued`。

## 5. 配置面与 Key

`registry.json` 顶层 `jury` 块（低频配置面，git 跟踪）逐键覆盖，非法值回落默认，
默认 `panel: []`（语义=「尚未选型」，据此降级入队而非报错）：

```json
{"jury": {"panel": ["deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct"],
          "base_url": "https://api.siliconflow.cn/v1",
          "call_timeout_s": 120, "deadline_s": 300,
          "pricing": {"deepseek-ai/DeepSeek-V3": {"input": 2.0, "output": 8.0}}}}
```

`pricing` 单位=元/百万 token；**缺席 → 记录里 `cost.amount` 为 null 并附 note，不猜价**
（留痕数字必须能对外负责）。

Key 只从环境取：环境变量 `SILICONFLOW_API_KEY` 优先，回落项目根 `.env`（已 gitignore）。
**永不入库**：Key 只进请求头；出模块的自由文本（异常文本、裁判 reason、案卷、队列
reason）一律过 `scrub()` 掩码。`pysci-dev doctor` 的「API Key 泄漏哨兵」扫 git 跟踪
文件找疑似密钥形态（`sk-` 前缀 / `键名=长随机值` / `Bearer 长令牌`；`tests/**` 豁免
为阳性用例素材），命中报 ✗ 并给脱敏片段——把纪律变成机制。

## 6. 冻结简报与「待用户确认」

`state/evals/jury-brief.md` 是护栏级统一裁决规则（项目优化面 / 既有裁决约束 /
**v0.6 直通规则**：「仅涉提议者自身专有基础设施的提案，除非有明确危害，默认 approve」
——组长主观直通豁免已取消并移入此处）。

当前是**草案 v1**：批准方式是文件里加一行 `approved: <ISO 时间>`（代码只认这枚标记）。
未批准时 `orch jury` 每次打印「简报未批准」警告但**照常可用**；批准前请勿用它做不可
撤销的裁决。

## 7. 留痕记录字段

`state/evals/jury/<case-id>.json`：`case_id` / `case_file` / `ts` / `question` /
`options` / `targets` / `dossier_hash` / `dossier_chars` / `dossier`（裁判所见全文）/
`brief{path,approved,hash}` / `base_url` / `panel` / `attempts` / `jurors`（逐型号：
两局票、`consistent`、计票选择）/ `votes` / `tally` / `inconclusive` / `verdict` /
`escalate` / `usage` / `usage_by_model` / `cost{amount,unit,note,priced,unpriced}`。

重裁同案覆盖旧记录（以最新为准），并自动删除同案的 `jury-queue/` 条目。

## 8. 边界（本轮交付未做的事）

- **真实连通**留用户 Key：`SILICONFLOW_API_KEY` 只能由用户本人提供（REFACTOR §15），
  全套单测走 mock transport，不发任何真实请求；
- **`--dry` 不校验 panel/Key**：组卷面与开庭面分离，先 dry 后开庭是省钱顺序；
- 平票/inconclusive 的 **twin 终裁**（双副本 A/B 实裁）属 E6，未实现。
