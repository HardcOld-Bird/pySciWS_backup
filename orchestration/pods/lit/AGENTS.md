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

## 网页截图（playwright，2026-10-16 实测）

- **本机 ms-playwright 浏览器未装**（`C:\Users\antar\AppData\Local\ms-playwright` 不存在），
  直接 `p.chromium.launch()` 报 harness shell 缺失、`executable_path` 指向的文件也不存在。
  对策：**用系统 Chrome 通道** `p.chromium.launch(channel="chrome")`（系统亦装 Edge，
  可 fallback `channel="msedge"`），无需下载浏览器。
- 运行范式：脚本放 `bench/`，用 `uv run --no-sync python <script>`。
- 全页截图遇**粘性/固定元素**会首尾重复渲染；对 GitHub 这类页改用**视口截图**（不传
  `full_page`）更干净。
- 本机对 `cn.comsol.com`、`github.com`、`api.github.com` **均可直连**（无需代理）：
  WebFetch 对 `www.cn.comsol.com` 报 TLS altname 错误时，改 `curl -L` 或 playwright 均正常。

## 领域事实（PPT 页2 调研沉淀）

- COMSOL 官方 **2026-09-16** 宣布 Multiphysics **2027** 版本含 **COMSOL MCP Server**
  （AI 智能体标准化接口），计划 2026 秋发布；官方一手页：
  `cn.comsol.com/press-release/system-level-modeling-and-agentic-ai-take-the-spotlight-in-comsol-multiphysics-version-2027-14722`。
- 社区 COMSOL MCP：旗舰 `wjc9011/COMSOL_Multiphysics_MCP`（论文 DOI
  10.1016/j.neucom.2026.134481）；Python 底座 `MPh-py/MPh`（封装 Java API）。

## 文档提取工具链（2026-10-15 实测可用）

- `.docx` → `python-docx`；`.pdf` → `PyMuPDF (fitz)`（比 pypdf 好，能判每页文本长度以
  识别扫描/纯图页）；`.doc`（旧版）→ **`antiword`（PATH 已含，`/mingw64/bin/antiword`）**，
  中文与表格渲染良好，`-m UTF-8.txt` 指定映射。
- LibreOffice 在 `D:/XiGPrograms/libreOffice/base/program/soffice`（非默认安装路径，
  `which soffice` 找不到，需用绝对路径）；本机**无** `olefile`/`textract`/`catdoc`/`wvText`。
- 图表/流程图页的文本层只给零散节点词，箭头逻辑无法还原——如实标注，勿臆测。
