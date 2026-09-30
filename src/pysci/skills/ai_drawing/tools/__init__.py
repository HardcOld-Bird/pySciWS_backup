"""ai_drawing 技能的工具层（CLI 门面 imagine.py + 各功能模块）。

模块划分：
- ``config``         —— 统一配置加载 + ComfyUI 发现 + 脱敏摘要
- ``imagine``        —— 统一 CLI 门面（console script: ``pysci-imagine``）
- ``postprocess``    —— Pillow 简单后处理（裁剪/缩放/旋转/转换/拼合/主色板抽取）
- ``ledger``         —— 生成账本（每次出图的可复现记录）
- ``comfy_client``   —— ComfyUI REST/WS 客户端（提交/轮询/取图/内省）
- ``comfy_session``  —— ComfyUI 服务器生命周期（镜像 comsol_simulation.session）
- ``workflows``      —— 程序化构造 ComfyUI API 格式工作流图
- ``providers``      —— 可插拔 provider 抽象（即梦/方舟优先）
- ``bridge``         —— 审美参考 → scientific_plotting 数据复现桥
"""
