"""ai_drawing 技能测试包。

- 纯 Python 单测（config 发现/脱敏、providers 模型解析、workflows 图构造、postprocess Pillow
  操作、ledger 记账/查询、bridge 抽色/脚手架、paths 锚点）不依赖 ComfyUI/火山方舟 key，随
  ``uv run pytest -m "not hardware"`` 始终运行。
- 外部/联网冒烟（comfy server 可达性、``/object_info`` 内省、真实 t2i/i2i）标记 ``hardware``
  + ``slow``，经 ``conftest.py`` 的 ``comfy_server`` / ``ark_key`` fixture 在不可达/无 key 时
  ``pytest.skip``（对齐 comsol conftest，默认全绿）。
"""
