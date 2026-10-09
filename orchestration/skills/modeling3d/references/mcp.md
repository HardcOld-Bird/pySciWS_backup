# blender-mcp 交互通道：设置、用法与边界

## 定位（先读这个）

blender-mcp 是渲染管线的**交互式探索通道**：在 GUI Blender 会话里边看截图边调场景，
效率远高于盲写脚本。但受 SKILL.md 约束性规则管辖：

> **交付物必须出自版本化脚本。** MCP 会话里调好的场景，必须固化回
> `src/pysci/research/<name>/models/<slug>.py` 场景脚本并以
> `pysci-model3d render <scene.py>` headless 重跑，才算交付。

这样 MCP 与 headless 两通道不构成重复实现：MCP 是草稿纸，脚本是定稿。

## 架构与安装状态

社区项目 [ahujasid/blender-mcp](https://github.com/ahujasid/blender-mcp)（约 30k star，
Blender 基金会**无**官方 MCP，此为事实标准）。两部分：

1. **addon**（`blendermcp_addon.py`，装进 Blender 用户脚本目录，起 localhost socket 服务）
   — 已于 2026-10-05 由 Agent headless 装入 Blender 5.1 并写入用户偏好
   （`%APPDATA%\Blender Foundation\Blender\5.1\scripts\addons\blendermcp_addon.py`）。
   升级方式：从仓库重新下载 addon.py，同样命令覆盖安装（`bpy.ops.preferences.addon_install`
   + `addon_enable` + `save_userpref`，全程 `blender -b --python-expr`，无需 GUI）。
2. **MCP server**（`uvx blender-mcp`，stdio 桥）— 已注册于 Qoder user 作用域
   （`~/.qoder-cn/settings.json` 的 `mcpServers.blender-mcp`），与 arxiv/zotero 同级。

## 每次使用前的连接步骤（需用户或桌面自动化配合）

MCP server 连的是 **正在运行的 GUI Blender**，addon 的 socket 服务不会自启：

1. 用户启动 Blender（任意场景）；
2. 3D 视口按 `N` 打开侧栏 → 「MCP for Blender」面板 → 点 **Connect to MCP server**
   （面板里同时能看到 Poly Haven / Hyper3D / Tripo 的开关与 API key 配置位）；
3. Qoder 会话内 `/mcp reload`（或重启会话）后即可调用 `mcp__blender-mcp__*` 工具。

Agent 无法替用户点侧栏按钮——开工前先确认连接（调一个只读工具如 get_scene_info，
失败则请用户完成第 1–2 步）。

## 常用工具面（以会话内 mcp_list 实际枚举为准）

- 场景读取：get_scene_info / get_object_info / get_viewport_screenshot（视觉闭环的关键）
- 对象操作：创建/修改/删除对象、材质应用
- 代码执行：execute_blender_code（任意 bpy 代码——**探索成果据此固化回脚本**）
- 资产生成：Poly Haven（HDRI/材质）、Hyper3D Rodin、Tripo（图生 3D，
  与 CLI `model3d tripo` 共享同一 `TRIPO_API_KEY`；面板内生成的模型直接进当前场景）

## 安全须知

addon 的 socket **无鉴权无加密**，仅绑定 localhost:9876。不要将其暴露到局域网；
远程会话需要时用 SSH 隧道（上游 README 的建议）。

## 与打印管线的边界

blender-mcp **不参与**打印管线：尺寸精确的打印件一律 build123d 脚本 + 验收门。
即使 MCP 会话里"看起来"建好了一个零件，也不得导出 STL 直接交付——网格建模
无尺寸约束体系，无法通过 `model3d check` 的断言保证物理正确。
