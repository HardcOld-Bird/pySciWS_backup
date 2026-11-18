# Cycles CPU 须知（⑥域渲染纪律）

本机硬件红线：**仅 CPU、无独显、16GB 内存（可用约 5GB）**。由此推出的渲染纪律是
⑥域红线之四，任何渲染脚本/命令都适用：

1. **引擎只有 Cycles CPU**——EEVEE（EEVEE Next）需要 GPU，本机不可用；
2. **单帧低采样**——概念渲染够用即可，不追无噪点成品；
3. **禁并行多实例**——内存预算 ~5GB 只够一个 Blender 同时渲染（`doctor.ensure_single_instance`）。

## 最小渲染配置

```python
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"        # 显式钉死，防止未来加 GPU 后行为漂移
scene.cycles.samples = 32          # 概念渲染 16–64；>128 属于成品需求，先问用户
scene.cycles.use_denoising = True  # 低采样靠降噪救场（OpenImageDenoise，CPU 可用）
scene.render.resolution_x = 1280   # 概念图 1280×960 足够；4K 是内存/时间双杀
scene.render.resolution_y = 960
scene.render.image_settings.file_format = "PNG"
```

## 引擎 ID 版本漂移（坑）

`scene.render.engine` 的合法 ID 随版本浮动：5.1 实测 `BLENDER_EEVEE`；4.2–4.5 为
`BLENDER_EEVEE_NEXT`；旧版 `EEVEE_NEXT`/`EEVEE`。**CYCLES 一路稳定**——这也是钉死
Cycles 的又一理由。若自定义脚本确实要试 EEVEE（例如将来有 GPU 机器），照旧域回退链
逐个 `try: scene.render.engine = cand except TypeError: continue`，`CYCLES` 兜底。

## 内存 ~5GB 纪律（怎么做才不炸）

| 纪律 | 做法 | 原因 |
|---|---|---|
| 单实例 | 跑前 `doctor.ensure_single_instance()` | 每个 Blender 进程常驻数百 MB，渲染峰值数 GB |
| 低采样 | samples ≤ 64 + `use_denoising` | 采样数近似线性吃时间和内存带宽 |
| 低分辨率 | 概念图 ≤ 1280×960 | 帧缓冲 + 降噪暂存随像素数涨 |
| 降 bounces | `scene.cycles.max_bounces = 4`（需要时） | 复杂多次反弹场景的路径树开销 |
| 大网格减面 | 验收门 `judge_render(max_faces=200_000)` | 面数预算就是渲染性能红线 |
| 不并行出图 | 多帧逐张跑，不起多个 `-b` | 见单实例 |

`doctor.available_memory_gb()` 低于预算（5GB）时**暂缓渲染**，先释放内存。

## 时间预期（量级，非承诺）

20mm 级单件、1280×960、samples=32、无复杂材质：CPU 上约 1–5 分钟/帧（视核心数）。
明显更慢 → 检查面数、samples、是否误开 4K/高 bounces。`blender -b` 冷启动本身
~10–30s，批量出图时脚本内循环渲多帧比起多次冷启动划算（但仍是单实例）。

## 验证命令

```powershell
uv run python -c "from pysci.skills.modeling_3d import doctor; print(doctor().format())"
# RESULT OK 且 version 非 N/A → 引擎可用；render engine 一行钉死 CYCLES CPU
& <blender.exe> -b --factory-startup --python-expr "import bpy; bpy.context.scene.render.engine='CYCLES'; print('cycles-ok')"
```

真实冒烟（立方体→Cycles CPU 低采样→GLB 导出）在
`tests/skills/modeling_3d/test_hardware_blender.py`（`-m hardware`，跑前自动单实例确认）。
