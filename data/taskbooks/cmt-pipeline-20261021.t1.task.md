# 任务书：CMT 正问题通用化（cmt-pipeline-20261021 / G1.t1，theory）

## 背景（自包含，看不到组长会话）

- **权威理论文档（必读）**：`data/research/1_gain_ep/theory/参考-透反射管槽结构的S矩阵理论计算/CMT计算与仿真参考.md`
  （MinerU 转换自同名 PDF）。附录 A（衍射模式分布图）与附录 B.2（非厄米非局域反射型 CMT）
  是本任务物理基准；B.1 是 B.2 的平面波模特例；**B.3 透射型不在本任务范围**（用户裁决）。
- **现有实现**：`src/pysci/research/gain_ep/theory/cmt_reflection_s_matrix.py`
  （参照 Fang et al., Phys. Rev. Applied 19, 054003 (2023), Appendix A）。组长已实测事实
  （2026-10-21，可直接信任）：
  - 任意 J 根管槽、逐槽独立复声速 c_complex / 宽 w / 深 h / 位置 d 可用（J=5 冒烟通过）；
  - **互易性恒等式精确成立**：任意非对称几何 S11≡S22 到机器精度（~1e-16），S12 与 S21
    无约束（理论推导：S[out,in]=S[T(in),T(out)] 于周期结构）——这是最强回归锚点；
  - EP/DP 名义参数与 COMSOL 验证吻合 4 位有效数字（见
    `scripts/research/gain_ep/simulation/CMT-COMSOL单点验证.py` 文档头的标定结论）。
- **已知局限（本任务要消除）**：
  1. `MetagratingParams.period` 由光栅方程反推（θᵢ、f₀ 锁死 D=√2λ/2），D 非自由参数；
  2. `compute_s_matrix` 硬编码 ±45°、0/±1 阶的 2×2 提取语义；
  3. `MetagratingParams.__post_init__` 四数组任一为 None 就**全部**回填默认 EP 参数
     （部分指定被静默丢弃）；
  4. 无参数校验（管槽总宽超 D 不报错）；单频求解、无批量频率接口；零单元测试。

## 目标

按文档 B.2 物理对正问题做通用化：

1. **D、θᵢ、f₀ 解耦为自由输入**（不再反推锁死）。
2. **S 矩阵维度自由（用户裁决 2026-10-21，本环节核心）**：传播通道数由附录 A 相图决定
   （WA 线 k_x+m(2π/D)=±n₀k、声锥 k=(1/n₀)k_x），**不恒为 2×2**，可多端口、理论上可趋向
   无穷阶。由 (f₀, θᵢ, D) 判定传播阶次集合 → 端口集合 → S 矩阵形状；每个矩阵元带通道语义
   标注（入射通道 × 出射阶次）；±45°/D=√2λ/2 的 2×2 只是特例，语义与既有
   `compute_s_matrix` 输出保持兼容。
3. 任意 J＋逐槽独立 c_s/w/h/x 语义固化与 docstring 文档化。
4. 参数校验：管槽总宽≤D、逐槽尺寸为正、复声速非零，违例抛明确异常。
5. 修 `__post_init__` 静默回填坑（部分指定不丢弃；全缺省才用默认）。
6. 单频→频率批量接口（一次装配多频求值或逐频便捷封装，形态你定，说明取舍）。
7. 性能非本环节重点（快核是后续环节），但通用化不得使现有单次求值明显劣化。

## 交付

- `src/pysci/research/gain_ep/theory/` 通用化实现；公共 API（`MetagratingParams`、
  `solve_reflection_cmt`、`compute_s_matrix`）保持或平滑迁移，既有调用方
  （`article/figures/ep_*.py`、`scripts/research/gain_ep/...`）不破坏；Google 风格
  docstring＋完整类型注解；符号与文献一致（变量名 Unicode 优先）。
- `tests/research/gain_ep/` 单元测试，锚点：
  - 互易性恒等式（按一般维度的正确推广形式；2×2 特例即 S11≡S22 机器精度）；
  - EP/DP 名义参数回归（S 矩阵数值与现状一致）；
  - 任意 J、D/θᵢ 自由、**至少两组 (f₀, θᵢ, D) 给出不同通道数**的用例；
  - `__post_init__` 行为与参数校验报错。
- 用法写 docstring，不新建 README/说明文档。

## 白名单（仅可写这些目录）

- `src/pysci/research/gain_ep`
- `tests/research/gain_ep`
- `data/research/1_gain_ep/theory`

## 验收预期

`PYTHONPATH=src uv run --no-sync pytest tests -q` 全绿（worktree 口径）；S11≡S22 机器精度
恒等式过；EP/DP 回归值不变；通道维度随 (f₀, θᵢ, D) 变化的用例过。

## 项目约定速查

编辑文件用文件编辑工具（勿 shell 重定向）；运行 `uv run --no-sync python/pytest`；
产物落盘前 `pysci.paths.assert_within_data`；`data/用户的inbox/` 只读（outbox 除外）；
拿不准先 blocked 问组长，勿自行推翻文献符号。
