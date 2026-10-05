# modeling3d CLI 全量参考

`uv run pysci-model3d <cmd> -h` 永远是最新真值源；本文件补充语义与排障。

## 命令全表

### doctor
配置摘要（路径 / Blender / 渲染默认值 / Tripo，敏感字段脱敏）+ 探测结果：
build123d、trimesh 版本；blender.exe 与 `--version`；blender-mcp addon 是否已装入
（`%APPDATA%\Blender Foundation\Blender\*\scripts\addons*\blendermcp_addon.py`）；
studio 渲染脚本存在性；Tripo key 有效性（有 key 时实调 `/v2/openapi/user/balance`）。

### new \<research\> \<slug\>
- `--kind print`（默认）：脚手架 build123d 打印件脚本（含设计参数区 / 建模区 /
  **几何断言区** / 导出区四段）。
- `--kind scene`：脚手架 Blender 场景脚本（材质 / 灯光 / 相机 / 渲染四段）。
- `--title` 覆盖 docstring 标题；`--force` 覆盖已有脚本。
- 产物：代码脚本 `src/pysci/research/<name>/models/<slug>.py`；
  会话目录 `data/research/<n>_<name>/models/<slug>/{stl,step,renders}` + `notes.md`。
- 模板源：`data/skills/modeling3d/templates/{print_part,scene}.py.tmpl`，
  `%TOKEN%` 替换（TITLE/RESEARCH/SLUG/SESSION_DIR）。模板可自由演化——
  但 **print 模板的 `[model3d] stl:` / `[model3d] volume_mm3:` 标记行是 build 的解析协议，勿改格式**。

### build \<script.py\>
在项目 .venv 中执行建模脚本（捕获 stdout/stderr 后转发），解析标记行定位 STL，
**自动串跑验收门**。选项：
- `--strict-volume`：以脚本报告的 B-rep 解析体积为期望值断言（1% 相对容差，
  覆盖 STL 三角化损失）。**打印件推荐常开**。
- `--expect-volume V`：手动指定期望体积 mm³。
- `--expect-bbox L,W,H`：期望包围盒（mm，排序后逐维比较，默认 0.1mm 容差）。
退出码：脚本失败→透传；验收门 FAIL→1；通过→0。STL 路径还会过
`assert_within_data` 防散落护栏。

### check \<mesh\>
独立验收门（.stl/.glb/.obj/.ply，trimesh 可读即可）。报告项：faces、watertight、
winding consistent、体积为正、体积/包围盒断言（可选）、连通分量数（bodies）、欧拉数。
- `--no-watertight`：渲染用途 / 云生成网格体检时放宽闭合要求（仅报告不判负）。
- `--volume-rtol`（默认 0.01）、`--bbox-atol`（默认 0.1mm）。
语义：watertight=False 的网格**切片必出问题**；euler/bodies 异常提示布尔运算残留。
多壳体（bodies>1）对装配体合法，对单件打印品是红旗。

### render
两种互斥模式：
- **快速模式** `render --mesh <file>`：调用包内 studio 模板
  （`src/pysci/skills/modeling3d/blender_scripts/render_studio.py`，CLI 组成部分，勿手改）：
  清场 → 按扩展名导入（stl/glb/gltf/obj/ply）→ 尺度归一化（最长边=2 单位，布光与输入
  单位解耦）→ 无材质网格补 studio 灰 → 地面 + 三点布光 → 50mm 相机自动取景 →
  引擎回退链 → PNG。选项：`--out`（默认 `<mesh>/../renders/<stem>_<engine>.png`）、
  `--engine EEVEE|CYCLES`、`--samples`、`--width/--height`、
  `--bg studio|transparent|0xRRGGBB`。
- **自定义脚本模式** `render <scene.py> [-- 透传参数...]`：headless 执行场景脚本，
  实时流式输出，默认 30 分钟超时。

### tripo \<source\>
`source` 为存在的文件路径 → 图生 3D；否则（或 `--text`）→ 文生 3D。
流程：upload → create_task → poll（queued/running→success，5s 间隔）→ 立即下载
GLB + 预览图（`<out>.preview.png`，**URL 有时效**）。
- `--out`（默认技能 cache 目录带时间戳）、`--pbr`、`--quad`、`--face-limit`、
  `--model-version`、`--timeout`（默认 600s）。
- `--dry-run`：打印完整请求计划，不联网不计费（key 未就位时演练用）。
- API 契约详见 `tripo_client.py` 模块 docstring（**尚未经实调验证**——首次真实调用
  后如有出入须回写该 docstring）。
- 云生成网格**非打印级**：非流形、尺度随意。打印用途只作视觉参考，须用
  build123d 重建模。

### list \<research\>
列出会话（stl/renders 计数）与代码侧脚本清单。

## 输出路径与协议汇总

| 内容 | 位置 |
|---|---|
| CAD/场景脚本（代码） | `src/pysci/research/<name>/models/<slug>.py` |
| 会话产物 | `data/research/<n>_<name>/models/<slug>/{stl,step,renders,notes.md}` |
| 脚手架模板 | `data/skills/modeling3d/templates/` |
| studio 渲染脚本（功能性） | `src/pysci/skills/modeling3d/blender_scripts/render_studio.py` |
| Tripo 默认落盘 | `data/skills/modeling3d/cache/` |

## Troubleshooting in full

- **`未找到 Blender`**：`.env` 设 `MODELING3D_BLENDER=<绝对路径>`；自动探测只覆盖
  `D:\XiGPrograms\blender\base\Blender *\` 与 `C:\Program Files\Blender Foundation\` 两处惯例位置。
- **引擎 TypeError**：EEVEE 的引擎 ID 随版本变化（`BLENDER_EEVEE_NEXT` → …）；
  studio 模板已带回退链，自定义场景脚本请抄它的 try/except 写法。
- **验收门 FAIL watertight**：回 CAD 脚本查布尔残留/自交（build123d 中常见于
  相切面布尔），**不要用网格修复工具糊 STL**——那会掩盖设计错误且不可复现。
- **build 报「没有标记」**：脚本没按模板打印 `[model3d] stl: <path>`；补上或手动 `check`。
- **Tripo 401/403**：key 缺失/无效；`--dry-run` 先验证管线，再查 platform.tripo3d.ai 控制台。
- **渲染黑图**：多为相机取景落入物体内部或灯光能量与尺度失配——studio 模板已做
  尺度归一化，自定义脚本请自行处理（先归一化再布光）。
- **中文乱码**：`.qoder/rules/basic.md` §3（shell 侧编码），与本技能无关但常见。
