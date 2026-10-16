# ppt_page4_trend 设计说明

**用途**：汇报 PPT 页 4 的整页单主题大图（16:9，2816×1584，一页全屏放这一张）。

## 构图与报告 B 的对应

从左到右一条**三级台阶**视觉主线，规模由小到大，正是报告 B §三「copilot → agentic → 多 Agent 协同」
三级台阶的图说化：① **Copilot 副驾驶**用一个代码编辑器窗口 + 光标处高亮的「自动补全」，
对应台阶一定义「人在循环里逐行补全，AI 是工具」；② **Agentic 自主智能体**用一个自主节点 + 实线循环箭头
（旁挂文档/终端图标），对应台阶二「单 Agent 自主多步执行：读改文件、跑命令、自纠错，人是验收者」；
③ **Multi-Agent 多智能体编队**用一个居中主节点 + 环形细线连接八个并行子节点并带轻微辉光，
对应台阶三「多个 Agent 分工、并行、互相通信，由 lead agent 或编排程序调度」——也直接对上报告 B
§一的事实锚点（Agents API 的 `multi_agent` 并行、Claude Code Agent Teams、Anthropic Dynamic Workflows
单次累计至多 1000 个 agent）。底部一句观点句「编排，成为 AI 编程的新战场」取自报告 B §四结论 1
（竞争焦点已从「模型多强」转向「多 Agent 编排可靠、可运维」）；其下一行浅灰小字
`Agents API · Claude Agent Teams · Dynamic Workflows` 是三个产品级事实锚点（报告 B §三·甲层），
只做证据点缀、不展开，符合「面向非专业听众」的克制要求。

## 规格与配方

- 尺寸 2816×1584（精确 16:9），浅灰白底、深蓝 `#1F4E79` + 橙 `#D55E00` 两色系（色觉安全）、
  扁平矢量、细线条、大量留白；中文大字（标题与阶段标签为投影可读级字号），与图形一体化。
- 模型 `doubao-seedream-5-0-pro-260628`（即梦 5.0 Pro），文生图；prompt 全文见 LEDGER 与 PNG 内嵌
  `pysci:prompt`。任务书要求 `.png`，5.0 档位只出 jpeg，故经 `imagine adjust` 免费转格式。
- **计费 3 张**（候选 A/B/C，均在 `data/skills/ai_drawing/gallery/ppt_p4_{a,b,c}.png` 归档）；
  交付首选 **C**：标题更大更「大气」、代码窗口内无英文伪码乱码、无 A 中箭头上的多余数字标注，
  同时保留了 B 的证据锚点行。A/B 仅作气质备选。
