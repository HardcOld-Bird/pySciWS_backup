# 单体任务书：预设回退同步（直角＋完整徽标）＋audit 无裁断言

## 背景

用户口头改单（验收后两改）：①圆角回退直角、从预设移除该要求；②内容页徽标保留
**完整图像**（1:1、贴卡缘），不模仿 p1/p2 封面裁切（那是宽边框特例）。deck-spec
aesthetics/装饰指纹已由组长更新（frame_geom: rect；logo_bbox_ref [0.09,0.937,5.926,6.773]，
无 srcRect）。本单同步预设真本与 audit 断言面。

## 工作要求

1. **07 美学参数册**（ppt-preset-formal-07-aesthetics.md）：
   - 圆角条目改「**直角为默认美学**（R4 曾设 roundRect adj 3000，用户裁决太大回退；
     封面/胶囊圆角形除外）」；
   - 徽标条目改「内容页：**完整 2000×2000 圆标、1:1、左缘贴白卡 0.09**（无裁切、
     无白 margin）；**p1/2 封面为宽边框特例**（srcRect l=49515 b=7343 半裁出血）——
     内容页严禁模仿封面裁切」；块底规则/标题公式不变。
2. **06 分册**：§4K 圆角结论改为「维持直角」（撤销 R4 改判）；§4N audit 美学四行
   读法同步（外框样式期望 rect；徽标期望 1:1 完整图）。
3. **04 分册**：无涉圆角处不改（0.94/块底规则保持）。
4. **audit 小改**（slides_audit.py）：`expect_srcRect` 支持 **`none`**（断言徽标
   blipFill 无 srcRect；素比校验自然为 1.0≈1.0）；无该字段行为不变；用例 1。
5. **staging**（学院正式汇报预设.md）同步 ①②；`pysci-dev sync`＋`sync --check`；
   每册 ≤8192B；mdgen-check 无改写；测试绿。

## 白名单

可写 orchestration/skills/、orchestration/pods/、staging、src/pysci/skills/document_writing/tools/slides_audit.py。

## success_criteria

- 预设三册与 staging 同步回退（直角＋完整徽标＋封面特例）；audit none 断言＋用例绿；
  无新字段时零回归；报告附字节数/diff 摘要/sync 输出。
