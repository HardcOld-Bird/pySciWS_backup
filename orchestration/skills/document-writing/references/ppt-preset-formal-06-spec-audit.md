# 学院正式汇报预设·分册 06：deck-spec / audit 工作流与实证

> 主册：`ppt-preset-formal.md`；spec schema 与命令细节：`slides-audit.md`；
> 拆解教义：`orchestration/skills/orchestration/references/plan-decomposition-visual.md` V5/V6。
> 本册为返工第 3 轮收口：把「要求真源 + 机械对账」定为**每页返工的标准流程**。

## 4I. 工作流（建 spec → 逐页修复 → audit 对账 → 终检）

1. **建 spec**（`data/大任务/<任务>/deck-spec.yaml`）：
   - `invariants`：装饰三层指纹（灰粗边框/白卡/院徽的 name+bbox）、字体全槽、
     版式归属、动画指针。全部**机械实测**（bbox/md5/px 读数），禁止手写估读。
   - `pages[N]`：`placeholders`（手工草稿内嵌图逐张 md5/px/bbox，唯一真源）、
     `slots`（占位指纹 + 目标 shape + `source` + `md5` + `anchor`/`figure_no`/`crop` + `order`）、
     `titles`（框宽公式）、`manual`（判不了的人工项）。
2. **逐页修复**：任务书**从 spec 切片**——槽位清单即工作清单；每叶交付附本页
   `slides audit` 输出（报告断言必须有 audit 行支撑）。
3. **audit 对账**：`compose slides audit '<deck>' --spec deck-spec.yaml [--json] [--strict]`
   → PASS / FAIL / 人工确认三态；`--strict` 有 FAIL 即非零。
4. **终检**：audit 全量 + `slides render/verify` + `slides layout` + 放映模式人工核对。
   人工确认项必须逐条勾选留痕，不得整页跳过。

## 4J. 字体全槽（中文渲染的隐性坑）

- **python-pptx 的 `run.font.name` 只写 `<a:latin>`**；中文实际走 **`<a:ea>`**。
  只设 latin 时 UI 显示字体正确、WPS/PowerPoint 渲染却回落雅黑/黑体——这是
  「字体设了但没生效」的根因（返工 3 轮实证：8 页标题全缺 `ea`）。
- 模板 rPr 为 latin+ea 双设并带 `panose/charset`；写框/重建必须用全槽 API：
  - `slides_fonts.set_font_slots(run, latin=…, ea=…, cs=…, panose=…)`
  - `slides_fonts.copy_font_slots(template_run, target_run)`（整元素复制）
  - `copy_slots_to_shape(template_shape, target_shape)`（按 run 序复制）
- audit 逐 run 校验 `latin/ea/cs`（缺 `ea` 即 FAIL）；模板无 `cs` 时 spec 的
  `require_slots` 只要求实际存在的槽（通常 `latin+ea`），不虚增要求。

## 4K. 粗外框直角（维持/回退结论）

粗外框（描边 ≥5 pt 且宽 ≥2 in）维持 **`rect` 直角**。R4 曾按用户批语改判
`roundRect adj=3000`，但用户 2026-10-19 口头改单裁决圆角太大，**回退直角并从预设
移除该要求**；封面/胶囊的圆角形不在此列。参数见
`ppt-preset-formal-07-aesthetics.md` §4O。

## 4L. V5/V6 的 PPT 语境操作化

- **V5 现状断言须附机械读数**：任务书写「现在是什么」必须附可复跑读数——
  `figures.json` 行 / `slides audit` 行 / `crop-img --dry-run` 尺寸 / 图片 md5+px。
  禁止「现上=fig3、下=fig5」这类二手断言（p10 互换反向案的根因）。
- **V6 索引去建议化**：图像索引只记事实（内容/尺寸/md5/原文图号/来源）；
  挑选规则唯一：**占位同款 > 原文图号匹配 > 其余**。建议类语句剥离；任何偏移在交付
  报告显式声明理由并经复核。

### 实证案例①：占位↔槽位按列表序错绑（p5/p8 三缺陷潜伏至 rv3）

- 现象：spec 把草稿占位图按列表顺序 zip 到当前形状，未核内容；p5 中图、p8 左列下、
  p8 右列下三处错误替换/删除正确件，audit 因「目标 shape 存在 + md5 取当前值」误放行，
  直到 rv3 人工比对占位才发现。
- 纪律：每槽必须同时带 **placeholder 指纹（md5/px）** 与 **期望来源 md5**；audit 比对
  目标内嵌图，而不是「shape 存在即过」。列表序只能作排序辅助，不能作绑定依据。

### 实证案例②：spec 溯源字段纪律（rv2 再判 B 一轮）

- `source` 只写**纯路径**；任何说明文字（裁切框、选择理由）放 `crop.note`——路径字段
  混入说明会让 audit/后续脚本解析失败。
- 新裁件必须**落盘到 `文献图片/` 目录**（不得只写临时路径）；修完立即刷新 spec 的
  `manual` 注记与 `figures.json` 映射，否则复审会按旧注记再判一轮（rv2 实案）。

## 4M. 预算与单一真源

- 本册与 02–05、07 各 ≤8192B；同一能力只保留一份实现：schema/命令口径在
  `slides-audit.md`，教义在 `plan-decomposition-visual.md`，本册只做流程与实证。
- staging（`data/大任务/5_开题PPT/学院正式汇报预设.md`）与本册同源同步。

## 4N. audit aesthetics 四行读法（R4）

`compose slides audit '<deck>' --spec deck-spec.yaml` 在 spec 带 `invariants.aesthetics`
时新增四行；无该节时不产生（零回归）：

| 行 | 判定 | 容差 / 失败含义 |
|---|---|---|
| `徽标 pN` | 内容页 p3 重建样例：3.861×6.299 @ (0.079,0.937)，比 0.613；p1/p2 封面特例 | bbox ±0.02 in、比 ±0.01；FAIL=拉伸/移位 |
| `外框样式 pN shape=…` | 粗外框 `prstGeom.prst`（默认 `rect`；spec 带 `adj` 时一并校验） | `adj` ±500；FAIL=样式漂移 |
| `上缘 pN shape=…` | 每个粗外框的 `top` | ±0.02 in；FAIL=切分起点漂移 |
| `标题边距 pN shape=…` | `titles[].shape` 的 left + right(n) | left ±0.02、right ±0.1 in；FAIL=标题框漂移/公式过期 |

- 内容页 `expect_srcRect` **正断言**：`{l: 38699, b: 0, tol: 100}`；裁后素比
  `0.613 ≈ 框比 0.613`（等比无变形）。
- 样例四参数：位置/大小 `0.079, 0.937, 3.861×6.299`；裁剪 `l=38699`；图层＝白卡
  （`矩形: 圆角 2`）上/内容下（spTree 第 3 位）；p4–p10 完全模仿。
- `none` 仍支持（其他 deck 完整图）；p1/p2 封面裁切（`l=49515 b=7343`）不适用内容页。
- 参数只写 `invariants.aesthetics`（见 07 §4O）；改参后重跑即收敛。
- `title_left`/`title_right_formula` 启用时，旧 `标题公式 0.34n+0.55` 行由本行接管。
