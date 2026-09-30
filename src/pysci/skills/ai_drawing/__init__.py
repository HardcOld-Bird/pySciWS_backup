"""pySciWS ai_drawing — AI 图像生成/编辑技能包（视觉审美 + 效果图产出）。

面向两类需求：(1) 生产科研**效果图**（封面 / graphical abstract / 示意图，美学为主、
非数据驱动）；(2) 提供**视觉审美**能力——先让图像模型生成审美范本，再由 Agent
"依葫芦画瓢"用真实数据复现（经 scientific_plotting），打通"美学 × 数据准确"。

架构定位（与 literature_research / comsol_simulation 等一致的四层同构）：
- **ComfyUI 作为"云端 API 工作流编排器"**：本机无头运行（``--cpu``），实际生成交给
  云端模型 API（经 ``ComfyUI-Jimeng-API`` 节点调用火山方舟 / 即梦 Seedream）。ComfyUI +
  torch + 自定义节点由 comfy-cli 装在**独立环境**，绝不进 pysci 依赖；本包只经 HTTP 驱动它。
- **Qoder 内置 ``ImageGen``** 作零安装的 Tier 0 快速草图 / 兜底（由 Agent 直接调用，
  产物经本包 ``ingest`` 入库、``adjust`` 后处理）。

代码在 ``src/pysci/skills/ai_drawing/``，技能级资产在 ``data/skills/ai_drawing/``，
统一 CLI 入口 ``imagine.py``（console script: ``pysci-imagine``）。

设计要点：本包承载"怎么驱动 ComfyUI / 怎么构造工作流 / 怎么后处理 / 怎么记账 /
怎么桥接到绘图复现"的可复用约定；具体研究线的效果图产物落在
``data/research/<n>_<name>/article/artwork/`` 下。
"""
