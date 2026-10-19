# 单体任务书：徽标样例四参数入预设（位置/大小/裁剪/图层高度）

## 背景

用户手动删除重建 p3 徽标并令「完全模仿」；四参数须存入预设（本单），deck 推广另单。
deck-spec aesthetics/装饰指纹已由组长按样例更新（logo_bbox_ref [0.079,0.937,3.861,6.299]、
expect_srcRect {l:38699}）。

## 样例四参数（p3 id=5 实测）

位置 (0.079,0.937)；大小 3.861×6.299；裁剪 srcRect l=38699（右 61.3%，素比＝框比 0.613）；
图层高度＝白卡（矩形: 圆角 2）之上、内容层之下（spTree 第 3 位）；alpha 8000；图源 a5d9de64。

## 工作要求（orchestration/skills/document-writing/references/＋staging）

1. **07 美学参数册**：徽标条目改写为「用户重建样例四参数」（上表全录，含**图层高度**）；
   p4–p10 完全模仿、p1/p2 封面特例（l=49515 b=7343）保留；直角/上缘 0.94/块底规则/
   标题公式不变。
2. **06 分册**：§4N 徽标读法同步（expect_srcRect 正断言 l=38699±100；素比 0.613 校验）。
3. **staging**（学院正式汇报预设.md）同步。
4. `pysci-dev sync`＋`sync --check`；每册 ≤8192B；mdgen-check 无改写；测试绿。
   （audit 代码已有 expect_srcRect {l,b,tol} 正断言，无代码改动预期。）

## 白名单

可写 orchestration/skills/、orchestration/pods/、staging。

## success_criteria

- 四参数（含图层高度）入 07/06/staging；预算合规、sync ✓；报告附字节数与 diff 摘要。
