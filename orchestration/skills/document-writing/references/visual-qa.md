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
