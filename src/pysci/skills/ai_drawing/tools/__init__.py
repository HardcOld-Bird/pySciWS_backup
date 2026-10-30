"""ai_drawing 技能的工具层（CLI 门面 imagine.py + 各功能模块）。

模块划分（按调用方向自下而上）：
- ``config``         —— 统一配置加载（.env）+ 目录锚点 + 脱敏摘要
- ``ark_client``     —— 火山方舟（即梦 Seedream）图像 API 客户端（唯一对外通信面）
- ``postprocess``    —— Pillow 基础后处理（裁剪/缩放/旋转/转换/拼合/主色板抽取）
- ``imaging``        —— 进阶位图算子（scikit-image + OpenCV，optional extra ``imaging``）
- ``ledger``         —— 生成账本（溯源/成本审计）+ PNG tEXt 元数据内嵌
- ``bridge``         —— 审美参考 → scientific_plotting 数据复现桥
- ``imagine``        —— 统一 CLI 门面（console script: ``pysci-imagine``）
"""
