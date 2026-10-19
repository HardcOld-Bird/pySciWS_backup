# 单体任务书：p3 标题框宽恢复公式值（非任务写入漂移修复）

## 背景

rv4（19:29）后产品 deck 出现**非任务写入**（mtime 20:16:56，sha 643f3676→6be62e71）：
p3 标题（shape 60 / TextBox 59，文本「声学超构表面」6 字）框宽被改为 **2.809 in**，
违反标题框宽公式 0.34n+0.55＝**2.590±0.1**。审计机制捕获该漂移（当前 audit 唯一
FAIL 行）。已查明非 devops 舱所为（L.l3 会话零产品写入、fixture 全在 tmp）。

## 工作要求（最小修复）

1. 将 p3 标题 shape 60 的框宽恢复 **2.590 in**（仅改 cx 一值；位置/字体/其余不动）。
2. 复核：`pysci-compose slides audit --spec` 全量 **零 FAIL**；`slides verify
   --render-dir <明确目录>` PASS；Read p3 目检标题无异常。
3. 交付附：改前/改后读数（shape/位置/宽高）＋audit 全量输出＋成品 sha256。

## 白名单

仅可写 `data/大任务/5_开题PPT/`（spec/素材只读）。

## success_criteria

- p3 标题宽＝2.590±0.1；audit 零 FAIL；其余形状零变动（逐槽 md5 不变）。
