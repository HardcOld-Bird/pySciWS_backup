# bilingual：中英对照文档生产线（①域操作分册）

句级中英对照生产线：`prepare`（切分+图片回补+TM 回填）→ 你逐句翻译 `work.md` →
`submit`（check-work 护栏 → 出稿 + 自动渲染 PDF + 交付副本）。设计原则：**除翻译本身外
一切琐事程序化**——你只把英文换成中文；切分、对账、命名、图片回补、渲染、交付全部
CLI 机械完成。实现真源 `sub_root/src/pysci/skills/literature_research/bilingual/`
（契约与状态机见包 docstring 与各模块头注）。

## 使用场景

用户点名要某篇文献的中英对照文档（「给这篇做个中英对照版」）。前置：已有①域 read
提取的 markdown（`mineru_cloud.extract_markdown` 产物 `data/skills/literature_research/
papers/<key>.md`）；图片回补优先取提取件同目录 `images/`，兜底需要源 PDF。

## 命令面（`python -m pysci.skills.literature_research.bilingual.cli …`）

| 命令 | 输入 → 输出 | 要点 |
|---|---|---|
| `prepare <源md\|stem> [--pdf <PDF>] [--no-images] [--reuse <旧工作区\|work.md\|stem>] [--force]` | 源 md → `work.md` + `manifest.json` + `images/` | stdout 打翻译指引；`--reuse` 沉淀 TM 并回填旧译文（等值优先、模糊次优）；`--no-images` 仅调试（submit 拒稿）；含译文工作区默认拒覆写，`--force` 放行并备份 `*.bak-<ts>` |
| `status <stem>` | `work.md` 对账 → 计数 + 未译/存疑/占位 ID | 对账按 ID 不按行号 |
| `check-work <stem>` | work.md 行结构 × manifest 硬校验 | 行首 ID 序列逐位一致才过（pod-guard 等价护栏） |
| `submit <stem> [--out-dir <目录>] [--skip-check-work]` | 全译 `work.md` → `bilingual.md`（+自动 PDF）→ 交付目录 | **submit 前强制 check-work**，`--skip-check-work` 显式 override；未译/占位拒出稿（rc=1）；PDF 渲染失败仅警告；`--out-dir` 须在 `data/` 或 `用户的inbox/` 内，缺省 `用户的inbox/bilingual/<stem>/`（用户全权管理区：仅最终交付物、只增不删） |
| `render <stem>` | 已有 `bilingual.md` → `bilingual.pdf` | 双栏 A4、页码奇右偶左、宽元素缩放进栏（禁跨栏）、图+首对原子；由 `renderer.py` 承载 |
| `reorder-suggest <源md\|stem> [--pdf <PDF>]` | 源 md + 源 PDF → `reorder-suggested.md` + `reorder-report.md` | 源 md 文序错乱（图注残段/公式被隔开）时先修文序；**只出建议**不改源，人工校对落盘；`reorder.py` 承载 |
| `quality <stem> [--glossary 术语表.json] [--back-check]` | 对已译行做术语一致性/回译校验 | 发现只报告（术语缺规范译名=fail 级）；回译走 deepl（`.env` 无 `DEEPL_AUTH_KEY` 时跳过并提示，不阻塞主通道） |

## 切分与契约

**待译单元** = 标题 `H` / 正文句 `S` / 图注 `C` / 表注 `T`，每行 `ID<TAB>英文原文`；
**透传块**（勿动）= 公式 `$$` 块 / 表格 HTML / 图片引用行 / 参考文献条目。

- **句切分**（wtpsplit SaT 小模型，CPU、全本地加载）三层机械保证，语义与切分器无关：
  ① 缩写哨兵（`Fig.`/`Eq.`/`et al.`/省略号换 U+E000，防误断）；② 游标逐字拼回（分隔按
  原文间隙取，防 `Fig. 3(a).` 回并漂移）；③ 句末硬契约补齐（`. ? !` + `。！？` 必切、
  省略号不作句末、**括号内不切**）。
- **切片自检**：各单元/透传块按 order 拼回源文档须无损，不过 prepare 直接拒绝。
- `bilingual.md` 形态（render 输入契约，勿手改）：标题 `# 中文`（保源 `#` 层级）紧跟
  `*English heading*`；句对 = 中文行 + 斜体英文行、句对间空行；段界 `---`；图片原位、
  图注句对紧随（不隔 `---`）；透传块原样；摘要/正文为**中文独行小标题**（`## 摘要`/
  `## 正文`，无英文行）。**英文侧从 manifest 程序化重嵌入**（逐字一致，不取 work.md）。

## work.md 工作规则

- 只把每行 **TAB 后的英文整行替换为中文**；保留行首 ID；**不增删行**——submit 前
  `check-work` 硬校验（改 ID/删行/游离行即拒出稿），确认人工裁决可 `--skip-check-work`。
- 行内 `$公式$`、引文标注照抄不译；术语先读源文档（prepare 指引给路径）统一，或配
  术语表跑 `quality --glossary`。
- 分段编辑、每批落盘；续译前先 `status` 对账。
- 状态判读：缺 ID 或内容==英文 = 未译；含 CJK = 已译；其余 = 存疑。**占位假译**
  （`假译`/`旧译-` 前缀族，或与英文原文仅差固定前缀的夹具）**不计已译**、拒绝出稿、
  不入 TM——夹具不得成交付物。

## TM 回填（`--reuse` 升级）

旧工作区（work.md + manifest.json）的真实译文沉淀进 `bilingual/tm.sqlite3`（自研薄胶水
存储：stdlib SQLite + rapidfuzz 模糊匹配 ≥75），prepare 时等值优先、模糊次优
回填；未匹配保持未译；占位假译不入 TM 不回填。**依赖注记**（2026-11-17 冒烟实测更正）：
原接缝 `translate.storage.tmdb` 自 translate-toolkit 3.19 移除后已弃用（ADR 方案 B：TM
存储层自研薄胶水），translate-toolkit 3.20.6 **无需** pin ≤3.18；rapidfuzz 缺失时
`--reuse` 报 TmDependencyError。

## 质量闸门

| 闸 | 时机 | 行为 |
|---|---|---|
| 切片自检 | prepare | 拼回源文无损，否则拒绝 |
| 防覆写闸 | prepare | 含译文工作区默认拒覆写（`--force` 备份放行） |
| 图片闭合 | prepare/submit | 引用名与落盘名不闭合拒绝出稿 |
| 对账三桶 | status/submit | 未译/占位**拒绝出稿**（rc=1） |
| check-work | submit 前强制 | 行结构破损拒绝（`--skip-check-work` 显式 override） |
| quality | 按需 | 术语一致性（fail 级）/回译校验（warn 级，deepl 无 key 跳过） |

## 与①域提取衔接

输入衔接 `papers/<key>.md`（MinerU 云提取产物）：`prepare <key>` 直接解析；图片回补
取同目录 `images/` 按名复制（MinerU 结果 zip 自带），不闭合时 PyMuPDF 按页序配对兜底
（告警）；源 PDF 按同名 `<key>.pdf`（或去 `_fulltext` 后缀）反解，再不行 `--pdf` 指定。
文序坏了先 `reorder-suggest` 人工校对再 prepare。

## 常见故障

- **缺源 PDF**：`prepare`/`reorder-suggest` 报错提示 `--pdf <PDF>` 指定。
- **缺图/图名不闭合**：同名 images/ → PyMuPDF 顺序配对（以告警为准）；`manifest.json`
  的 `images_closed` 须 true 才可出稿。
- **未译行**：submit 拒绝并列 ID → `status` 对账后补译。
- **check-work 未过**：行结构被改（改 ID/删行）→ 恢复行结构，或显式 `--skip-check-work`。
- **PDF 渲染失败**：submit 只警告（md 照常交付），随后 `render` 重试。
- **TmDependencyError**：rapidfuzz 缺失（由依赖代理安装）；TM 存储已是自研 SQLite
  薄胶水，与 translate-toolkit 版本无关（tmdb 移除史见上）。
- **wtpsplit 报模型未 vendor**：`deps_management/deps/models/wtpsplit/`（sat-3l +
  xlm-roberta-base-tokenizer）由依赖代理落盘；`WTPSAT_MODEL_DIR` 可覆盖。

**冒烟/演练纪律**：测试与冒烟一律用**临时 stem**（如 `_smoke_*`），严禁在真实 stem
工作区跑 prepare——`--force` 覆写虽有备份，误伤仍要手工回捞译文。
