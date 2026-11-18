"""⑦ AI 生图域薄胶水：纯云生成（方舟 Seedream 主 / 硅基流动副）+ 本地轻后处理。

域边界（用户裁决 2026-11-17）：**⑦ = 生成式图像**（示意 / 概念 / 艺术配图 / 图层），
**③ = 数据图**（matplotlib 出版链）。AI 生成图**不作数据证据**，只作示意/美学用途；
数据图一律走 ``pysci.skills.scientific_plotting``。

模块分工（薄胶水≈500 行，生成本体在云端）：

- :mod:`~pysci.skills.image_generation.providers`   — 双云统一客户端（OpenAI 兼容
  images API：文生 / 图生 / 组图；401 优雅报错不裸崩）；
- :mod:`~pysci.skills.image_generation.guardrails`  — 出图护栏（watermark 显式 false /
  seed 兼容档 / b64 规避 URL 失效 / 成本上限）；
- :mod:`~pysci.skills.image_generation.ledger`      — 生成账本（manifest.jsonl）+
  PNG tEXt 溯源（prompt/seed/模型/时间自带出处）；
- :mod:`~pysci.skills.image_generation.imaging`     — measure（连通域量化）/ align
  （相位相关亚像素配准）；
- :mod:`~pysci.skills.image_generation.postprocess` — rembg 透明底兜底 + Pillow 裁剪/缩放。

选型与裁决：``sub_root/others/community_research/2026-11-17/domain-07-imagegen-selection.md``；
实施计划：``sub_root/others/plans/domains-06-07-3d-imagegen.md``。操作面见技能文档
``.agents/skills/image-generation/SKILL.md``。

状态注记：**真实出图冒烟挂起**（ARK_API_KEY 已过期，2026-11-17）——客户端按两家官方
文档实现，静态测试（mock）恒跑；key 更新后跑
``tests/skills/image_generation/smoke_real.py`` 补真实链路验收。
"""

from __future__ import annotations

__all__ = [
    "guardrails",
    "imaging",
    "ledger",
    "postprocess",
    "providers",
]
