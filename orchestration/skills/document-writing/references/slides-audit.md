# deck-spec 与 `slides audit`（要求真源 / 机械对账）

> 定位：学院正式预设的**要求唯一真源**落在每 deck 一份 YAML（`deck-spec.yaml`）；
> `compose slides audit <deck> --spec <spec>` 逐项对账，生产叶交付附本页 audit 输出。
> 主册/分册见 `ppt-preset-formal*.md`；本文件只讲 spec 形态与审计口径。
> 工作流（建 spec→逐页修复→audit→终检）、字体全槽坑与实证案例：
> `ppt-preset-formal-06-spec-audit.md`。

## spec 形态（YAML）

```yaml
deck: 研究背景v1.pptx
pages:
  - page: 3
    decoration:                       # 不变量：装饰层存在性+指纹
      - {name: "白卡", box_in: [0.087, 0.089, 13.25, 7.424], tol_in: 0.05}
      - {name: "Picture 3", kind: picture}   # 无 bbox 时只验存在
    fonts:                            # 每标题/胶囊/正文 run 须 latin+ea+cs（含 panose）
      - {role: 标题, shapes: [6]}
      - {role: 胶囊, shapes: [9], slots: {latin: {typeface: 汉仪大宋简},
                                         ea: {typeface: 汉仪大宋简}}}
      - {role: 正文, shapes: [13, 14]}
    slots:                            # 变量槽位：图槽内容指纹+原文图号+来源+裁切
      - shape: 8
        figure_no: "Fig. 7"
        anchor: "RPP 86 Fig.7 场分布"
        md5: "<源图/内嵌图相同 md5>"
        px: [2150, 2754]
        source: "../文献图片/第6页-.../fig7.png"
        crop: {top: 0, bottom: 0}
    titles:  [{shape: 6}]             # 公式：w = 0.34n + 0.55（algn=dist）
    capsules: [{shape: 9, cite_chars: 3}]  # c=方括号内字符数，不含 []
    refs:    [{ref: 30, anchor: 5}]   # 复用 slides layout 引用框判据
    animation: true                   # <p:timing> + onBegin + nextAc=seek
    manual: ["装饰质感（目检）"]       # 判不了 → 人工确认行，不计 FAIL
```

- 多图槽按 `slots` 顺序列出；图槽 `md5`/`px` 是**内容指纹**（与来源文件同 md5 时只写
  `source` 亦可）。spec 里不重复抄预设参数，只引用槽位/公式。
- 顶层 `manual: [...]` 亦计入「人工确认」行；审计输出必须带勾选痕迹。
- **切片**：任务书按页/工序从 spec 生成——槽位清单即工作清单；现状断言必须附机械读数
  （`figures.json` 行、`audit` 行、`crop-img --dry-run`），不得裸写。

## 命令与判据

```
compose slides audit '<deck.pptx>' --spec deck-spec.yaml [--json] [--strict]
```

- 装饰：按 `name` 或 `box_in`（±tol）匹配形状；缺失/错位 → FAIL。
- 字体：每个 run 读 `a:latin/a:ea/a:cs`（含 panose 属性）；缺槽或 typeface 不符 → FAIL。
- 图槽：目标必须为图片；内嵌 md5 与 `source` 文件 md5 一致、`px` 一致；否则 FAIL。
- 引用框：委托 `slides layout`（左缘/行数/多框上缘下缘）；标题/胶囊：框宽公式 + `algn=dist`。
- 动画：`<p:timing>` 存在、首块 `onBegin`、`p:seq nextAc="seek"`。
- 退出码：默认只报告；`--strict` 有 FAIL 即 1；spec/文件错误 2。

## 字体全槽 API

`pysci.skills.document_writing.tools.slides_fonts`：

- `set_font_slots(run, latin=..., ea=..., cs=..., panose=...)`——三槽一次写全；
- `copy_font_slots(template_run, target_run)` / `copy_slots_to_shape(template_shape, target)`——
  照模板 rPr 整元素复制（绕 python-pptx 的 latin-only）；
- `font_slots(run)` / `slots_complete(slots)`——audit 读数与判据。
