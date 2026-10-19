# 视觉验收与网页截图（PPT 交付）

产出演示文稿的**最后一公里**：幻灯片是视觉产物，文字提取稿读着对、渲染出来可能排版崩。
故交付 pptx 前必须**逐页渲染 → 看渲染图 → 签自检**；本页是那条链的用法与阈值。

## 逐页渲染（视觉判读前置）

```
compose slides render '<deck.pptx>' [--out DIR] [--dpi 140]
```

固化 **pptx → pdf → 逐页 PNG** 链（LibreOffice headless + pymupdf；中间 PDF 一并落盘便于放大
核对）。产出 `<deck>_render/page-NNN.png`（`--out` 可改目录，默认 `<deck 同目录>/<stem>_render/`），
逐页 PNG 即**交付必附件**——把它们 Read 一遍，才谈得上「看过」。缺 LibreOffice 时 `doctor`
会报，渲染不可用（阅读 / 提取 / LaTeX 不依赖它）。

## 机械视觉闸（交付前必过）

```
compose slides verify '<deck.pptx>' [--render-dir DIR] [--min-ink 0.002]
```

只读**已落盘的渲染图**，三条判据任一不过即退出码非 0（FAIL）：渲染存在（目录非空）、
**页数 = 幻灯片数**、**无空白页**（与背景异色的墨迹像素占比 < `--min-ink` 判空白）。约定目录
`<stem>_render/` 与 `render` 同一约定，故渲染完直接 `verify` 即可。派发侧已把它注册为
`check="pptx-visual"`：交付里声明 `<artifact check="pptx-visual">…deck.pptx</artifact>` 即由
orch 自动验收，**缺渲染 / 页数不符即打回**。

渲染后命令会**打印可直接粘贴的 verify 调用**——受审目录只读时用 `render --out <可写目录>`，
不必记住 `--render-dir` 旗标名。

## 几何体检（版面硬伤，只读、不写盘）

```
compose slides geometry '<deck.pptx>' [--frame NAME] [--json] [--strict]
```

渲染闸只管「渲染出来了没 / 是不是空白」，命不中「渲染出来了但版面是坏的」。几何体检补后者：
纯 bbox + z 序硬判三项——**外框边被更高 z 序图片覆盖率**（压模板外框）、**图元越界**（画布 /
外框）、**图片互叠面积**（粘贴截图互相叠住）。**不渲染、不写盘**，故受审目录只读时照跑；默认
只报告（版面硬伤需人判读），`--strict` 才据违规置非零退出码。外框默认自动识别（画布面积
10%–95% 的最大矩形），识别不准时 `--frame <形状名>`。这即是 H4①/H2 的机械原料，**不必自建脚本**。

**连接符（`bentConnector*` / `straightConnector*`）不计入越界**：PowerPoint 给连接符存的是
「能容纳折线路径」的 bbox，路径常远小于它，用 bbox 判越界会报出渲染图上不存在的假阳性。故它
们的越界单列为 `提示（非违规）`（JSON 的 `connector_hints`），不进 `violations`；真正越界的
图元/图片照报。

**透明装饰图（水印）不计入互叠/描边/外框**：图片 blip 的不透明度 ≤10%（`alphaModFix/@amt`，
如学院正式预设的 8% 左侧水印）即视为装饰，不是内容图——它天然压住内容图与白卡边缘、按设计
不描边。它只在 `装饰水印（豁免互叠/描边/外框判定）` 行与 JSON 的 `decorative_pics` 中列出，
不参与图片互叠、描边、外框覆盖率/越框判定（越出**画布**照报）。

### 缺 `<a:ext>` 闸与跨页标题一致性（tb-20261017-044633）

`geometry` 另扫**全形状 XML**：`<a:xfrm>` 里**缺 `<a:ext>`**（显式尺寸未定义）即列页码＋形状
标签、**计入 violations**（一票否决）——python-pptx 读属性会返回**继承值**、渲染器也按继承值
兜底，故「渲染图正常」挡不住在 WPS 里按零/异常尺寸呈现的双盲缺陷；判据必须读 **XML 真值**。
同时做**跨页标题一致性**（各页标题尺寸/位置/字号/字体比对基准页＝出现最多的格式），差异即报。

### WPS 渲染：**wpscli 已可用**（tb-20261017-105200 更新）

旧版（12.1.0.22215）无 `wpscli`、COM 未注册、不支持 headless → **当时不可行**；用户升级至
**12.1.0.29158**（`…\12.1.0.29158\clitool\wpscli.exe`，v1.1.0 插件式）后实测
`wpscli ppt2pdf` **3 秒出 8 页 PDF**、标题/结尾页渲染正常——即用户所见口径（WPS≈MS Office）。
转换链（`compose convert` / `slides render`）现**默认优先 wpscli**，LibreOffice 为 fallback，
输出行注明所用后端；缺 `<a:ext>` 闸保留（WPS 同样会暴露该缺陷，且 wpscli 不可用时靠它兜底）。

## 焦点块动画生成（学院正式汇报预设 §5）

```
compose slides focus-anim '<deck.pptx>' --page 3 \
  --block '9:13' --block '21:11' [--fade 2:5] \
  [--focus-color 8496B0] [--blur-fill ADB9CA] [--blur-line D6DCE4] \
  [--focus-ms 250] [--blur-ms 1000] [--easing ease-out|linear] \
  [--transition-ms 350] [--no-transition] [--out copy.pptx]
```

`--block chip:shape1,shape2` 每项一块，**块序＝触发序**：首块进入页面自动聚焦，其后每次
单击「旧块失焦（1000 ms）＋新块聚焦（250 ms）」；按形状 ID 逐个打靶，不真组合。缺省值即
学院正式汇报预设 §5；`--fade 块序号:shape` 给该块加「进入·淡入」。注入后仍须
`slides render/verify` + 放映模式核对单击顺序（机械闸不验观感）。

## 排版机械判据（学院预设；实现 ③④）

```
compose slides layout '<deck.pptx>' --page 3 \
  --title '9,21' --refs '30:9' --refs '31:21' [--tol 0.05] [--json] [--strict]
```

- `--title`：焦点块标题形状 ID；≥2 个时**上缘或左缘**在容差内即 PASS（逐项读 Δtop/Δleft）。
- `--refs REF:ANCHOR`：引用框左缘与其焦点块左缘对齐；框内 `[n]` ≤3 条时要求一行一文献；
  多框上缘、下缘一致。无 `[n]` 标记的框跳过行数判据。
- 取舍：①焦点块轮廓饱满 / ②同级间距一致性**本期未实现**——需显式块分组与递归切分语义，
  当前 deck 未暴露组结构，硬判误报率高；没给 ID 的判据只打印「跳过」，不臆造读数。

## deck-spec 审计

`compose slides audit '<deck>' --spec <spec.yaml>`：装饰层指纹 / 字体全槽（latin+ea+cs+
panose）/ 图槽内容指纹 / 引用框 / 标题公式逐项 PASS/FAIL；spec 形态、切片约定与字体
API 见 `slides-audit.md`。

## 图像裁剪（置入前去刊头/图注）

```
compose crop-img '<img>' [--strip-top 182] [--strip-caption 60] \
  [--top|--bottom|--left|--right N | --top-ratio|... R] [--out '<new>'] [--dry-run]
```

四边可按像素或比例（同边二选一）；`--strip-top`＝去顶部条带（刊头）、
`--strip-caption`＝去底部图注。缺省输出 `<stem>_cropped<ext>`（不覆盖源，置入 PPT 时用
替换件）。裁后无剩余 / 同边双口径 → rc=2 且不写盘；先 `--dry-run` 核对尺寸。

## 网页截图（deck 里的官网 / 页面图）

需要把网页截进幻灯片时用本机 `playwright`（已装，复用 literature_research 的浏览器）：

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1200})
    pg.goto("https://…", wait_until="networkidle")
    pg.screenshot(path="site.png", full_page=True)  # 截整页；不带 full_page 只截视口
    b.close()
```

本机没有 Chrome/Edge 时先 `playwright install chromium` 一次；截得的 PNG 与其它素材同管理
（放研究线资产目录，勿散落）。远程站点不可达时如实记录，改用 AI 图文或已有素材替代。
