"""中英对照文档生产线（bilingual）——①文献调研域的句级双语产出线。

设计总原则：**除翻译本身外一切琐事程序化**。译者（人 / LLM 代理）只逐句把 work.md 的
英文替换为中文；切分、对账、命名、图片回补、出稿、交付全部机械完成。

三份契约是本包的核心资产（旧项目平移，语义不变）：

1. **work.md 行契约**：每行 ``ID<TAB>英文原文``，行首 ID 与行数不得改动；
2. **manifest 切片契约**：单元（H/S/C/T）+ 透传块（formula/table/image/reference）按
   ``order`` 拼回源文档须无损（``verify_slices`` 硬闸）；
3. **bilingual.md 渲染契约**：英文侧从 manifest 程序化重嵌入（逐字一致，不取 work.md），
   句对 = 中文行 + 斜体英文行、段界 ``---``、图片原位图注紧随、透传块原样。

模块映射（真源 = 各模块 docstring，本处只作导航）：

- ``units.py``    切分（wtpsplit 换接缝 + 缩写哨兵 + 游标逐字）与 markdown-it-py 结构解析；
- ``manifest.py`` work.md / manifest / bilingual.md 三契约与防覆写闸、check-work 结构核验、命名；
- ``images.py``   图片回补（①域提取产物同名回补 → PyMuPDF 页序配对兜底）；
- ``reconcile.py`` 对账三桶（未译/已译/存疑）与占位假译判定；
- ``tm.py``       TM 回填（自研薄 TM：SQLite + RapidFuzz 模糊匹配，升级旧 --reuse 等值回填）；
- ``quality.py``  质量层：回译校验、术语一致性核对（机械闸门，译文引擎可插拔）；
- ``cli.py``      CLI 状态机 prepare / status / submit / check-work / render。

**归属切分（勿越界）**：``renderer.py``（bilingual.md → HTML → PDF 排版）与
``reorder.py``（源 md 文序订正建议）归**并行实施者**维护，本包其余模块不得实现其职责。
接缝契约（本包只按此调用）：``renderer.render_pdf(md_path: Path, pdf_path: Path) -> Path``。

选型 ADR（2026-11-17，用户裁决，详见
``others/plans/bilingual-absorption.md``）：句切 pysbd→**wtpsplit**（豁免 pin，SaT 小模型
vendor 于 ``deps_management/deps/models/wtpsplit/``，CPU 推理）；MD 解析正则状态机→
**markdown-it-py**（token.map 取源文逐字切片）；``--reuse`` 等值回填→**自研薄 TM**（SQLite +
RapidFuzz 模糊匹配回填；原选型 translate-toolkit tmdb 自上游 3.19 移除，用户裁决方案 B
自研替代、translate-toolkit 3.20.6 仅留 TMX/TBX 格式层，详见 ``tm.py`` 原地 ADR）；
旧 pod-guard.mjs（DSH PreToolUse 钩子）→ **submit 前强制 check-work +
显式 ``--skip-check-work`` override**（脱离旧编排体系）。
"""
