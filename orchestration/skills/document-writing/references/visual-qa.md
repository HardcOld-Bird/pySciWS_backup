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
