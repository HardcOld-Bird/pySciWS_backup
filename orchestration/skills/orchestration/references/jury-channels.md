# 仲裁团渠道面（E3-live 逐席 provider、Key、回退告警）

> REFACTOR v1.0 §4 渠道修订 + §14 单点条款告警化；backlog `20261012-164600-leader`
> （用户 2026-10-12 裁决：三席官方直调减平票、SiliconFlow 降回退且回退必须告警）。
> 开庭流程与案卷格式见 `references/jury-panel.md`；本页只讲**传输面与配置面**。

## 1. 配置结构（`registry.json` 顶层 `jury` 块，git 跟踪低频面）

逐键覆盖，非法值回落出厂默认；默认 `seats: []`＝「尚未选型」，据此降级入队而非报错。

```json
{"jury": {"seats": [{"name": "deepseek", "key_env": "DEEPSEEK_API_KEY",
                     "base_url": "https://api.deepseek.com/v1",
                     "model": "deepseek-v4-pro", "params": {}}],
          "fallback": {"enabled": true, "name": "siliconflow",
                       "key_env": "SILICONFLOW_API_KEY",
                       "base_url": "https://api.siliconflow.cn/v1",
                       "model": "deepseek-ai/DeepSeek-V4-Pro"},
          "call_timeout_s": 120, "deadline_s": 300, "max_output_tokens": 8000,
          "pricing": {"deepseek-v4-pro": {"input": 2.0, "output": 8.0}}}}
```

- 每席 `base_url`/`model`/`key_env` **缺一不可**（`name` 缺则回落型号名）。缺键**不剔除**
  该席，而是标 `defect` 让它以 unavailable 身份被看见——静默剔除会让人以为庭里真有那席。
- `params`＝逐席请求参数覆盖。实测三家约束不一致：`kimi-k2.6` **只接受
  `temperature=1`**，共享写死的 0 会让那席每次 400——「主席不可达」经常其实是请求体不
  合席位胃口。请求默认 `temperature: 0` + `max_tokens: max_output_tokens` +
  `response_format: json_object`，`params` 在最后合并（可覆盖任一）。
- `fallback`＝带独立开关的**回退席**，仅当 ≥1 主席不可达时才上场，上场必告警（§4）。
- `pricing` 单位＝元/百万 token；缺席 → 记录里 `cost.amount` 为 null 并附 note，
  **不猜价**（成本数字要能对外负责）。出厂为空：刊例须由用户按账号实价填。
- 硬规则：任何席位（含回退席）型号串含 `qwen` → **拒开庭 rc=2**，点名违约型号。

## 2. Key 读取链与脱敏

**进程环境变量 → 项目根 `.env`（gitignored）→ 缺失即该席标 unavailable**。

- `.env` 走 `dotenv_values()` **读文件**，不用 `load_dotenv()`：后者会把值灌进
  `os.environ`，一次读数污染整个进程环境（Key 的可见面凭空变大）。
- **Key 只进请求头**，永不入库、永不打印。出模块的自由文本（异常文本、裁判 reason、
  案卷、队列 reason）一律过 `scrub()` 掩码；异常里的请求上下文由 `secrets=(key,)` 兜底。
- `pysci-dev doctor` 的「API Key 泄漏哨兵」扫 git 跟踪文件找疑似密钥形态（`sk-` 前缀／
  键名＝长随机值／`Bearer 长令牌`；`tests/**` 豁免为阳性用例素材），命中报 ✗ 并给脱敏
  片段——把纪律变成机制。
- 席位状态行（`Seat.public()`）**剥掉 `params` 与 Key 值**，只留 `role`/`defect`/
  `reason`/`usable`，因此可以安全落案卷并进读数面。

## 3. 席位判位与本庭组成

`seat_readiness()` 是唯一判位口：配置齐全 **且** Key 就位 → `usable`。产物
`{seats, voting, unavailable, missing_primary, fallback_enabled, fallback_used, notice}`，
其中 `voting`＝本庭实际开庭的席＝可用主席 +(若启用且可用) 回退席。

读数一行话由 `readiness_summary()` 独家产出（`orch status` 与 `pysci-mirror status`
共用同一函数，防两处文案漂移）：

```
席位 deepseek✓ glm✗(缺 GLM_API_KEY) kimi✓ siliconflow[回退]✓｜本庭 3 席
｜[!] 回退席已激活——请用户处理主席（不可达主席：glm）
```

三种不可用要被区分开：「没配好」（`defect`）／「没 Key」（`缺 <key_env>`）／「回退席
待命」（启用但未上场）——混为一谈就没法处理主席。

## 4. 回退告警三面（§14 单点条款的告警化）

回退席上场时，「回退席已激活——请用户处理主席」必须**三面可见**：

| 面 | 位置 |
| --- | --- |
| 案卷 | `state/evals/jury/<case>.json` 的 `fallback{enabled,used,notice,missing_primary}`；`[!]` 段并进 `orch jury` stdout |
| status | `orch status` 判决包健康段（`verdict.seat_alert_lines`） |
| mirror | `pysci-mirror status` 的「仲裁」行 |

告警与「是否真的用上了回退席」**解耦**：`missing_primary` 非空即报——主席掉了而回退席
也挂（本庭 0 席，入队），同样是须用户处理的事故，不能因为没用上回退席就哑掉。
阳性用例＝`state/evals/jury/e3live-disconnect-glm.json`（人为把 glm 席 `key_env` 指向不
存在的变量名，非 mock）。

## 5. 两条实测渠道纪律（mock 测不出，只有真实开庭暴露）

1. **同提供商组织内串行**。`collect_votes` 按 `(base_url, key_env)` 分组加锁，同组两局
   不得并发——官方直调的并发额度**按组织算**：实测 DeepSeek
   `max organization concurrency: 1`，同席 AB/BA 两局并发必 429。按席位分锁是错的
   （两席挂同一账号时照旧撞额度）。不同席仍并发，整案墙钟 ≈ 单局时长 × 2 而非 × 6。
2. **截断＝渠道错误，不是无效票**。三席皆推理型，正文之前先烧一大段 reasoning 且长度
   **重尾**（真案卷 prompt≈4 k 上实测 432～3986）。额度不足时 `finish_reason==
   "length"`、`content` 为空——若记成无效票，「预算不合」会被伪装成「裁判模棱两可」并
   污染 escalate 判据。现修法：报错误、点名席位与当时的 `max_output_tokens`、降级入队；
   出厂预算 8000。**改 registry 就能修因的事，不该算裁判的票。**

## 6. 真实连通现状（2026-10-12 三席直调实测）

- 三席 smoke 通过：`[√] 裁决=A（tally A:3 B:0）`，三席双序皆 `[一致]`，逐席 usage 落盘；
- 单席 Key 缺失：该席标 unavailable、全案不崩，回退席补位后本庭仍 3 席；
- 型号取自各提供商 `/models` 实测在架目录（deepseek 2 个／bigmodel 11 个／moonshot 2 个），
  `base_url` 逐家核官方 OpenAI 兼容端点；换型号＝改 registry，不动代码。
- **minimum 火山方舟档位＝不可行**，维持 flash（三道硬阻塞与替代路径见
  `state/suggestions/approved/20261012-164600-leader.md` 回填段）。
