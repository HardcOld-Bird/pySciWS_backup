# drawing 组员长期记忆

> 本文件由 drawing 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：生成图配方/审美范本、各研究线视觉风格、LEDGER 入库踩过的坑。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 drawing 组员（AI 绘图专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。
- 专业知识：部署技能 ai-drawing（Skill 工具调用）；CLI `pysci-imagine`。

## 经验

### pysci-imagine 踩坑（2026-10-15 实测，代价：2 张已计费图被误删）

1. **`adjust` 的 src 是位置参数，`palette` 的 src 是 `--src`**（同一 CLI 内不一致，易错）。
   正确：`pysci-imagine adjust --out x.png src.jpg`；`pysci-imagine palette --src x.png`。
2. **禁止把 CLI 串进 `| tail` / `| head` 再用 `&&` 串联**：管道退出码取末段命令（tail=0），
   会吞掉 CLI 的失败退出码，导致后续 `rm` 等无条件执行。**要串联就裸跑、看 `echo rc=$?`**。
3. **flash 档只出 jpeg**（`--output-format png` 仅 5.0 主档支持）；任务书常要求 `.png` →
   出图后用 `adjust` 转格式即可（免费）。
4. **`runs/*.json` 快照里 `b64_json` 落盘时已被剥离**（占位 `<... chars, stripped>`）→
   **误删/丢失的生成图无法从快照恢复**。故：**Read 满意后立刻 `gallery --add` 归档**，
   不要等到整理阶段；gallery/ 才是唯一可靠留存。
5. **Tier 0 内置 ImageGen 当前不可用**（`invalid_model_error`，2026-10-15 复现 2 次）；
   方舟无 key/超额时无免费兜底可用。
6. **安全出图流水线（2026-10-15 补单验证通过）**：`gen`（flash 出 .jpg）→
   `adjust --out x.png x.jpg` → **立即** `gallery --add x.png --slug S` → `Read` 判读 →
   最后才 `rm` 临时 .jpg。串联时**绝不带 `| tail`/`| head`**，逐条裸跑看 `echo rc=$?`。
   补单说明：`runs/<slug>.json` 已存在时快照自动改名 `<slug>_1.json`，不影响。
7. **5.0 pro/flash 不支持 `output_format=png`**（`imagine models` 实测：仅已下架的
   `doubao-seedream-5-0-260128` 支持）→ **jpeg→`adjust` 转出的 png 不含 `pysci:*` 内嵌溯源**
   （`gen` 的 jpeg 本身也无）。charter 要求「PNG 内嵌溯源」时须手工补：
   `uv run --no-sync python -c` 读 `runs/<slug>.json` 的 `request.prompt/model`，用
   `PngInfo` 写 `pysci:prompt`/`pysci:model`/`pysci:backend`(=ark) 后 `im.save(..., pnginfo=...)`
   （纯本地免费，不计费）。

### PPT 整页大图配方（任务 tb-4-ppt-20261016 P.p1，一次成型）

- **16:9 整页信息图**：`--model doubao-seedream-5-0-pro-260628 --width 2816 --height 1584`
  （精确 16:9，方舟接受）。5.0-pro 中文大字渲染**很可靠**——把「画面中只允许出现下列文字」
  逐条列出（含 ①②③ 等编号），实测 3 张全部零乱码；这是「图文并茂」类图的关键手法。
- **气质配方**（浅底 keynote）：干净浅灰白底 + 深蓝 `#1F4E79` + 橙 `#D55E00` 两色系、
  扁平矢量/细线/大留白、**显式写「不要 3D 渲染、不要重渐变辉光、不要堆砌细节」**；
  阶段图由小到大 + 粗渐变箭头（**写「箭头上不要任何数字标注」**，否则会自己加 1/2 数字）。
- **代码窗口防乱码**：别让模型自由写代码（会出伪码字形）——写「窗口内只有几行浅灰占位线条
  + 光标处高亮一行中文『自动补全』，不要出现任何其他代码字符」。
- **事实锚点**：底部加一行浅灰小字（如 `Agents API · Claude Agent Teams · Dynamic Workflows`），
  只做证据点缀不展开，适配「非专业听众 + 文字极少」。
- 三候选均存 `gallery/ppt_p4_{a,b,c}.png`；交付坑：`gallery --add` 后仍要 `Read` 成品一次，
  且 `adjust` 的 `--out` **必须在 `data/` 内**（pod 的 `bench/` 会被 assert_within_data 拒绝）。
- 1_gain_ep 的 `article/artwork/` 已有**共用** `设计说明.md`（beam_layout_ref 那套）；新主题
  另起同名文件 `<slug>.设计说明.md`，勿混进旧文件。

### PPT 页5 拓扑图配方（任务 tb-4-ppt-20261016 P.p2，3 张甄选）

- **同类版式**（左指挥链 用户→TUI→组长 → 中部枢纽 → 右侧分组卡片矩阵）：prompt 写法与页4
  相同（白名单逐条列文字 + 禁数字 + 禁 3D/辉光），prompt 全文存 `runs/ppt_p5_{a,b,c}.json`。
- **成员卡片重复是主要失败模式**：A 张把「6 张特殊席位」画成 7 张且「偏好」重复 → 交付前
  **必须逐卡核对数量与文字**。对策：prompt 里写「**恰好七张**/**恰好六张**」「依次且不重复」
  「「偏好」只能出现一次，不得重复任何成员名」——B 张一次成。
- **「每席一条队列」别另加图例引线**：C 张为放大队列 + 修引线而重发，反而**每组只画出一个队列**
  （其余卡片前的队列消失）。**逐卡片重复的小元素（队列方块+小箭头）宁可靠 prompt 里
  「每张卡片左侧都紧挨三个竖直浅灰小方块 + 橙色小箭头」描述，不要靠图例引线补齐**。
- 交付首选 B（`ppt_page5_topology.png`）；A/C 仅归档 `gallery/ppt_p5_{a,c}.jpg`。
- 空壳目录坑：交付时 `artwork/` 只留最终 `.png` + 设计说明，候选 `.jpg` 归档 gallery 后即删
  （与页4 一致）。

### PPT 页7 隐喻大图配方（任务 tb-4-ppt-20261016 P.p3，火箭＝现状反思）

- **比例类隐喻的铁律：proportion 必须写成可核对的高度分数**。prompt 里写「载荷只占火箭总高约
  1/8」「箭体约占 3/4」「鼻锥小一点、别拉长」——只写「很小」会各张差异极大：A 载荷≈1/5、B≈1/8、
  C 因鼻锥被拉长又回到≈1/5。**交付前用 Read 目测分界比例**，载荷占比最小的那版隐喻最强。
- 三候选同 prompt 家族（`bench/p7_prompt_{a,b,c}.txt`，快照 `runs/ppt_p7_{a,b,c}.json`），
  交付 B（`artwork/ppt_page7_rocket.png`）。**C 的教训**：为收敛火焰而重发，却在同句里新加
  「鼻锥+舱体」描述 → 鼻锥变长、箭体上缘被画成斜切肩形（不再「规整矩形箭体」）。**改一处坏一处**：
  微调 prompt 时只改目标子句，其余子句原样保留，并复核新描述是否与旧约束冲突。
- 火焰/喷管类「重复小几何」写清**总宽度约束**（如「三个火焰总宽不超过箭体+助推器总宽」）比写
  「三角形」更有效；模型仍可能画成两层梯形叠加，可接受（仍属抽象几何块）。
- 一条浅灰**水平虚线跨越分界并向两侧各伸出火箭外一小段**是强调比例的免费手法，纯几何、不引数字，
  实测稳定复现。标题左上 + 浅灰小注左下 = 与页4/页5 成套的固定版式。

### 研究线 1_gain_ep / beam_scatter 综合图（任务 tb-3-simfig）

- 目标排版：**3 列(case) × 3 行**（行1 = 同坐标系两色雷达；行2/3 = 两个上下堆叠的
  **半圆**声场图，平直直径边）+ 右侧**只跨行 2–3** 的共享 colorbar。旧图是 3 行×4 列，勿照抄。
- 气质配方：APS/Nature、hairline、白底、色觉安全；场图对称 `RdBu_r`，曲线 Okabe–Ito
  `#0072B2`/`#D55E00`。完整可落地规格见
  `data/research/1_gain_ep/article/artwork/设计说明.md`。
- 命中目标排版的 prompt 全文存于 `runs/beam_layout_ref_2.json`/`_3.json`（原始）与
  `_2_1.json`/`_3_1.json`（补单，已实测可复现几何正确排版）。
- 最终交付：`artwork/beam_layout_ref_{2,3}.png` **几何正确**（首选）；`_1.png` 有
  「整圆场图 + colorbar 整图高」两处偏差，仅取气质。符合性表见设计说明 §0。
