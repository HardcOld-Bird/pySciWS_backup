"""pySciWS ai_drawing — AI 图像生成/编辑技能包（视觉审美 + 效果图产出）。

面向两类需求：(1) 生产科研**效果图**（封面 / graphical abstract / 示意图，美学为主、
非数据驱动）；(2) 提供**视觉审美**能力——先让图像模型生成审美范本，再由 Agent
"依葫芦画瓢"用真实数据复现（经 scientific_plotting），打通"美学 × 数据准确"。

架构定位（与 literature_research / comsol_simulation 等一致的四层同构）：
- **云端出图：直连火山方舟（即梦 Seedream）的 OpenAI 兼容 API**。只用项目已有的
  ``requests``，零新增核心依赖；单次调用即可用图层拆分、交互编辑、组图生成等复合能力。
  **不经 ComfyUI 编排**：本机无 CUDA GPU，节点图的本地推理优势无从发挥，而第三方
  节点反而把方舟能力面收窄成字段子集（取舍论证见 ``references/backends.md``）。
- **Qoder 内置 ``ImageGen``** 作零安装的 Tier 0 快速草图 / 兜底（由 Agent 直接调用，
  产物经本包 ``ingest`` 入库、``adjust`` 后处理）。
- **本地位图加工**：``postprocess``（Pillow，核心依赖）做几何/格式基础操作；
  ``imaging``（scikit-image + OpenCV，optional extra）做泊松融合、算法 inpaint、抠图、
  形态学、连通域测量、配准等纯 CPU 算子。

代码在 ``src/pysci/skills/ai_drawing/``，技能级资产在 ``data/skills/ai_drawing/``，
统一 CLI 入口 ``imagine.py``（console script: ``pysci-imagine``）。

设计要点：本包承载"怎么调云端出图 / 怎么做位图加工 / 怎么记账 / 怎么桥接到绘图复现"
的可复用约定；具体研究线的效果图产物落在 ``data/research/<n>_<name>/article/artwork/`` 下。

跨技能的流水线（出图 → 看图 → 归档 → 桥接 → ``figures build`` → 迭代）**用 Python 脚本
串 CLI** 表达，配方知识卡片落 ``data/skills/ai_drawing/recipes/*.md``（与其余技能同构）。
"""
