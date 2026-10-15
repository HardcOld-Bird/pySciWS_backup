# lit 组员长期记忆

> 本文件由 lit 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：检索策略、各研究线文献地图、入库/引用踩过的坑。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 lit 组员（文献研究专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。
- 专业知识：部署技能 literature-research（Skill 工具调用）；CLI `pysci-research`。

## 经验

- arxiv MCP（2026-10-10 冒烟实测）：首调可能 50s 超时并随后被 arXiv 限流（HTTP 429，
  retry_after≈60s），等待约 2-3 分钟后即恢复且响应很快。遇到 429/超时不要连续快速重试
  （会加深限流），按 server 提示的 retry_after 等待后再试一次即可。

## 文档提取工具链（2026-10-15 实测可用）

- `.docx` → `python-docx`；`.pdf` → `PyMuPDF (fitz)`（比 pypdf 好，能判每页文本长度以
  识别扫描/纯图页）；`.doc`（旧版）→ **`antiword`（PATH 已含，`/mingw64/bin/antiword`）**，
  中文与表格渲染良好，`-m UTF-8.txt` 指定映射。
- LibreOffice 在 `D:/XiGPrograms/libreOffice/base/program/soffice`（非默认安装路径，
  `which soffice` 找不到，需用绝对路径）；本机**无** `olefile`/`textract`/`catdoc`/`wvText`。
- 图表/流程图页的文本层只给零散节点词，箭头逻辑无法还原——如实标注，勿臆测。
