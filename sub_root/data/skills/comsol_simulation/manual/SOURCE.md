# COMSOL 6.4 手册 RAG 索引（不入仓，安装物可重建）

- **数据源**：`sub_root/deps_management/deps/comsol/6.4/base/doc/pdf/`（COMSOL 6.4 自带
  108 个模块手册 PDF，共约 506 MB）。
- **构建命令**（2026-11-17 首建）：
  ```
  HF_HOME=sub_root/deps_management/deps/cache/huggingface HF_ENDPOINT=https://hf-mirror.com \
  <project-venv-python> scripts/build_knowledge_base.py \
      --pdf-dir <deps/comsol/6.4/base/doc/pdf> --db-dir <本目录>
  ```
  构建脚本来自 `deps_management/deps/repos/comsol-mcp`（pin commit 53c9ca4）。
- **消费方**：COMSOL-MCP 的 `pdf_search` / `pdf_search_status` / `pdf_list_modules`。
  `deps/repos/comsol-mcp/knowledge_base` 为指向本目录的目录 junction（上游自带空
  chroma 骨架 2 个追踪文件已被替换；`git checkout -- knowledge_base` 可恢复上游态）。
- **嵌入模型**：sentence-transformers `all-MiniLM-L6-v2`（缓存随 HF_HOME 自包含）。
- 查询冒烟（2026-11-17）：MCP `pdf_search("periodic boundary condition")` 命中
  Reference Manual《Equation-Based Modeling》Periodic Condition 节点原文（连续/反周期、
  边界选择、约束设置），索引约 130MB（构建中期）时已可用；chroma WAL 并发读安全。
