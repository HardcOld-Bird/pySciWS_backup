# 任务书：高性能 CMT 前向核（cmt-pipeline-20261021 / G2.G2a.t2，theory）

## 背景（自包含）

- **上一环节 G1.t1 已交付通用化正问题**（你若非同一会话请先读交付留档
  `orchestration/state/deliveries/theory-20261021-184701-cmt-pipeline-20261021-G1.t1.md`）：
  `src/pysci/research/gain_ep/theory/cmt_reflection_s_matrix.py` 现有
  `MetagratingParams`（`period`/`x` 自由字段、`MetagratingParameterError` 校验）、
  `solve_s_matrix()`→`ScatteringMatrix`（S 为 |M|×|M|，`Port(order,kx,angle_deg,kz)` 通道
  标注）、`sweep_s_matrix()`；测试在 `tests/research/gain_ep/theory/test_cmt_reflection_s_matrix.py`
  （41 条：EP/DP 回归、互易性多端口推广、通道数 2/3/4 用例）。
- 性能现状：单次 S 求值 **~0.06 s**（G1 顺手矢量化后；最初基线 0.52 s）。
- **用户裁决（2026-10-21，极致性能）**：5 ms 只是地板不是天花板——未来负载未知
  （例如 10+ 维高维参数空间探索），速度越快越好；**授权自由引入 numba 或任何确有帮助
  的依赖**（改 `pyproject.toml` 后必须先 `pysci-dev venv-sync`——入口锁坑，见 basic.md §3）。

## 目标

1. **极致快核**（用户裁决优先级最高）：
   - 解析积分全向量化（M22 正交性闭式对角、fenbu 积分广播化）；
   - **块对角 Schur 消元**：P3 按槽块对角 → 消去槽内振幅，主系统降到 N×N；
   - numba JIT 或其他编译级手段（按 profile 决定，不为用而用）；
   - 同参数族缓存：几何不变量（P2/V3 等）与 β/U 依赖项分离，批量求值不重复装配。
2. **一般通道维度**支持与 G1 语义完全一致（S 为 |M|×|M|、Port 标注不变）。
3. **批量评估接口**（种群/多频/多参数集一次调用），供下一环节逆向设计直接消费。
4. **同一能力不留两份实现**：快核成为唯一物理实现，旧装配路径在同一提交内删除或迁移
   （公共 API 兼容不破：`solve_s_matrix`/`compute_s_matrix`/`sweep_s_matrix`/`make_*_params`）。

## 验收

- 基准（J=3、n_orders=10、k_modes=31）：单次 S 求值 **<5 ms 硬门槛**；目标亚毫秒、
  力争数十微秒级；报告实测**每核吞吐**（目标 ≥2000 次/s/核）；基准数字须有可复跑脚本
  （测试 parametrize 或 `scripts/research/gain_ep/` 下脚本，不新建 README）。
- 与 G1 现实现**逐点对拍 rtol≤1e-10**（先对拍、后删旧路径，留对拍证据）。
- EP/DP 回归、互易性恒等式、通道维度用例全过：`PYTHONPATH=src uv run --no-sync pytest
  tests/research/gain_ep -q` 全绿；`uv run --no-sync ruff format/check` 过。

## 白名单（仅可写这些目录）

- `src/pysci/research/gain_ep`
- `tests/research/gain_ep`
- `scripts/research/gain_ep`（仅基准脚本）

## 备注

- 物理基准文档：`data/research/1_gain_ep/theory/参考-透反射管槽结构的S矩阵理论计算/CMT计算与仿真参考.md`
  （附录 B.2）；符号与文献一致。
- worktree 测试口径 `PYTHONPATH=src`；改 `pyproject.toml` 后 `pysci-dev venv-sync`。
- 拿不准先 blocked 问组长。
