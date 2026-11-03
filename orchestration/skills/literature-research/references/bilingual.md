# bilingual：中英对照文档生产线（lit 操作分册）

三环节管线的 lit 侧操作面：`prepare`（切分+图片回补）→ 你逐句翻译 `work.md` →
`submit`（出稿+自动渲染 PDF+inbox 副本）。设计原则：**除翻译本身外一切琐事程序化**——
你只把英文换成中文；排版、建文件、图片回补、移动、PDF 全部由 CLI 机械完成。
实现真源 `src/pysci/skills/literature_research/tools/bilingual.py`、`bilingual_render.py`。

## 使用场景

用户点名要某篇文献的中英对照文档（「给这篇做个中英对照版」）。前置：已有 minerU 风格
markdown（`read`/`pdf_extract` 产物，`cache/extracted/<stem>.md`）；图片回补需要源 PDF。

## 命令面（`pysci-research bilingual …`）

| 命令 | 输入 → 输出 | 要点 |
|---|---|---|
| `prepare <源md路径\|stem> [--pdf <PDF>] [--no-images] [--reuse <旧work.md\|目录\|stem>]` | 源 minerU md → `work.md` + `manifest.json` + `images/` | stdout 打翻译指引；`--pdf <PDF>` 手工指定源 PDF；`--no-images` 跳过回补（仅调试，submit 将拒绝出稿）；`--reuse` 按英文原文（归一化空白等值）回填旧译文，未匹配保持未译；含译文工作区默认拒覆写，`--force` 放行并自动备份 `*.bak-<ts>` |
| `status <stem>` | `work.md` 对账 → 计数 + 未译/存疑 ID 列表 | 对账按 ID 不按行号 |
| `submit <stem> [--out-dir <目录>]` | 全译的 `work.md` → `bilingual.md`（+自动 `bilingual.pdf`）→ 交付目录 | 未译拒绝出稿（rc=1）；PDF 渲染失败仅警告、不阻塞 md 交付；`--out-dir` 自定义交付目录（须在 `data/` 内、不存在则建；**未提供时回退默认** `data/用户的inbox/bilingual/<stem>/`） |
| `render <stem>` | 已有 `bilingual.md` → `bilingual.pdf`（同目录） | 无出稿 rc=2；`.html` 中间文件保留作调试图；PDF=双栏 A4、0.6cm 页边距、页码奇右偶左、栏间细分割线；超宽公式/表格等比缩放进栏（禁跨栏） |
| `reorder-suggest <源md路径\|stem> [--pdf <PDF>]` | 源 md + 源 PDF → 工作区 `reorder-suggested.md` + `reorder-report.md` | 文序订正**建议**（只出建议，不改源 md/cache）；缺源 PDF rc=2（`--pdf` 指定） |

**切分契约**：待译单元 = 标题 `H` / 正文句 `S` / 图注 `C` / 表注 `T`，每行
`ID<TAB>英文原文`；**透传块**（勿动）= 公式 `$$` 块 / 表格 HTML / 图片引用行 / 参考文献
条目。句切分已做缩写（`Fig.`/`Eq.`/`et al.` 等）与省略号保护，句末只认 `. ? !`；切片
自检（各单元拼回源文档须无损）不过 prepare 直接拒绝。

## work.md 工作规则

- 只把每行 **TAB 后的英文整行替换为中文**；保留行首 ID；**不增删行**——行数与行首 ID
  由 pod-guard 钩子硬校验，改动即被拦。
- 行内 `$公式$`、引文标注照抄不译；术语先读源文档（prepare 指引里给路径）统一。
- **分段编辑纪律（防流故障）**：按段落小批替换、每批落盘一次，勿整文件重写——编辑流
  中断时已译部分保留、行结构不破；续译前先 `status` 对账。
- 状态判读（`status` 口径）：缺 ID 或内容与英文相同 = 未译；含 CJK 字符 = 已译；其余 =
  存疑（自查）。**占位译文**（`假译`/`旧译-` 固定前缀起头，或与英文原文仅差固定前缀的
  测试夹具）**不计已译**——记未译并单独提示，`submit` 同口径拒绝出稿（夹具不得
  成交付物）。

## 交付物

- 工作区 `data/skills/literature_research/bilingual/<stem>/`：`work.md`、`manifest.json`、
  `images/`、`bilingual.md`、`bilingual.pdf`、`bilingual.html`（调试图）。
- 用户侧交付副本（submit 末步，只增不删）：`--out-dir <目录>` 自定义落点（须在 `data/`
  内、目录不存在则建），未提供时回退默认 `data/用户的inbox/bilingual/<stem>/`；
  **副本按文档标题命名**（首个 `H` 单元中文译文；无 H 回退首单元；非法字符清洗、
  超长截断 60 字），md/pdf 同名成对——工作区内部名（`bilingual.md/pdf/html`）不变。

## 摘要标注与译文复用（2026-10-23 裁决、2026-11-03 修订）

- **摘要识别**：prepare 把「元数据/作者行（Received/DOI/链接/作者角标）之后的首个
  段落」全组标 `role="abstract"`（manifest 各单元可辨；work.md 有 `#` 注释行标注）。
  出稿自动在摘要前插 `## 摘要`、摘要与正文之间插 `---` + `## 正文` 中文小标题（独行、
  **无英文行**——2026-11-03 裁决覆盖 10-23：原文没有的英文标注不发明）。**识别
  错误的人工修正**：改 `manifest.json` 各单元 `role` 字段（误标清空、漏标补
  `"abstract"`）后重新 `submit` 即生效。
- **译文复用**：源 md 经文序订正后重跑 prepare（单元 ID/顺序会变），`--reuse <旧
  work.md\|工作区目录\|stem>` 按**英文原文**（归一化空白等值）匹配回填旧译文——旧
  work.md 与同目录 `manifest.json` 成对读取（英文键取 manifest）；匹配不上保持未译，
  `status`/`submit` 未译拒绝逻辑不变（避免整篇重译）。**占位译文（假译类）不回填**
  （告警跳过，防夹具扩散到新工作区）。
- `bilingual.md` 形态（render 的输入契约，勿手改形态）：标题 `# 中文`（保留源 `#` 层级）
  紧跟一行 `*English heading*`；句对 = 中文行 + 斜体英文行、句对间空行；段落边界 `---`
  单独成行；图片原位、图注句对紧随其后（不隔 `---`）；公式/表格/参考文献原样原位。
  **英文侧从 manifest 程序化重嵌入**，与源文档逐字一致（不取 work.md，误改英文行不伤交付物）。
  摘要/正文标注为**中文独行标题**（`## 摘要`/`## 正文`，无英文行）。

## 文序订正（reorder-suggest，B 路径）

**使用时机**：minerU 源 md 文序错乱（图注句被拆成残段散落、公式与引导句被图块隔开、
孤立子图标签 d/c/b 散落）。工具基于**源 PDF 文本块坐标**推断真实阅读序（排版带聚类：
文本带「左栏自上而下 → 右栏自上而下」，图表带按 (y,x)），md 块与 PDF 块按文本相似度
映射；**同一 PDF 块的 md 块归组同位**（跨栏图注残段按「同带同 y」缝回图注），公式按
`\tag{N}` ↔ PDF `(N)` 尾锚映射，面板标签挂相邻图组。

**产出与人工校对口径**：`reorder-report.md` 每处「原位置 → 建议位置」+ 依据（PDF 页/
排版带/栏/纵坐标）+ 类型（移动 / 图注残段缝合 / 标签归位或剔出可译单元）；映射不上的块
单列「未映射」提示人工注意、**不硬排**。`reorder-suggested.md` 与源 md 同构（块原文
逐字，便于 diff）。**工具只出建议**——源 md 与 cache 零改动，订正由 lit 语义校对后
人工落盘（无 `--apply`）。

## 常见故障

**冒烟/演练纪律**：测试与冒烟一律用**临时 stem**（如 `_smoke_*`），严禁在真实 stem
工作区跑 prepare——含译文工作区默认拒覆写、`--force` 放行虽有自动备份（`*.bak-<ts>`），
误伤仍要手工回捞译文。

- **缺源 PDF**：prepare 报错并列候选 PDF（反解顺序：stem → 源 md 头部 DOI/arXiv id →
  标题 slug）；按报错提示 `prepare <stem> --pdf <PDF>` 指定。
- **缺图 / 图名不闭合**：先 MinerU 云端重抽按名落盘；不闭合则 PyMuPDF 按页序顺序配对
  （打告警，配对结果以告警为准）；MinerU 撞配额/断网时如实记录，用已落盘图片继续，
  `manifest.json` 的 `images_closed` 须为 true 才可出稿。
- **未译行**：submit 拒绝出稿并列 ID → `status` 对账后补译。
- **PDF 渲染失败**：submit 只警告（md 照常交付），随后 `bilingual render` 重试；缺
  pandoc/浏览器时按报错补环境。
- **未跑 prepare**：`status`/`submit` rc=2 并提示先 prepare。
