# 任务书 tui-polish-20261016 · polish-b：行为两件（VSCode 启动器稳健化 / Esc 路由回进入前视图）

## 背景 / 目标

TUI 抛光计划第 2 环（polish-a 已交付：选中态黑字/`├└` 定宽连线列/标题亮黄/决策行间距，merge `86e2b54`）。本环节修用户实测反馈的 2 条**行为**问题：

**反馈 #4（VSCode 启动器）**：点「用 VSCode 打开原文」显示：
```
[!] 打开失败：FileNotFoundError: [WinError 2] 系统找不到指定的文件。
```
用户环境（**不是用户装错**）：VSCode 装在 `D:\XiGPrograms\vsCode\base\`，其 `bin` 目录（`D:\XiGPrograms\vsCode\base\bin`）已在**系统 PATH**。此坑的典型根因：Windows 的 `code` 启动器是 `code.cmd`/`code.bat`，**`subprocess.Popen` 直呼 `.cmd` 文件会 WinError 2**（CreateProcess 不经 cmd 解释器）；或 `shutil.which("code")` 因 PATH/PATHEXT 未命中返回 None 后仍裸名 Popen。要求修复：
1. 启动器解析链：`shutil.which("code")`（PATHEXT 含 .cmd/.exe/.bat）→ `which("code.cmd")`/`which("code.exe")` → 扫描 PATH 各目录 → 常见安装位兜底（`%LOCALAPPDATA%\Programs\Microsoft VS Code\bin\code.cmd`、Program Files 两处、`D:\XiGPrograms\vsCode\base\bin\code.cmd`）。
2. **`.cmd`/`.bat` 必须经 `cmd /c` 调用**（或 `shell=True` 等价），`.exe` 直呼。
3. 失败消息列出**尝试过的路径清单**（一眼可见卡在哪）；成功消息显示实际使用的启动器路径。
4. 测试：mock `which`/PATH/Popen 覆盖三分支（.cmd 经 cmd /c、.exe 直呼、全路径缺失报尝试清单）；无真实启动 VSCode。

**反馈 #5（Esc 路由）**：任务书**详情页**按 `r` 进驳回面板后按 Esc，现在**直接回主菜单**；预期回**进入前视图**（详情页）。同族问题顺带修：`tb-new` 面板（从任务书列表进）的 Esc 应回**任务书列表**而非主菜单。实现建议：进入面板时记 `self._panel_return`（来源视图），`action_back` 对 `tb-reject`/`tb-new` 分支返回来源视图（详情页↔驳回面板、列表↔新增面板）；a/r 提交成功后的去向维持现状（回列表重建）。

## 输入（只读）

- `src/pysci/skills/orchestration/tools/tui/app.py`：`_open_vscode`（或等价命名，s5 落地的 VSCode 按钮处理）、`_open_reject_panel`/`action_tb_new`/`action_back`（现状回退链见 `action_back`：tb-detail→taskbooks 等，`tb-reject`/`tb-new` 落「其余→menu」分支）。
- `src/pysci/skills/orchestration/tools/taskbooks.py`：`md_path()`（打开目标）。
- 测试面：`test_tui_taskbook_groups.py`（VSCode 三分支既有测试——本环节改判据为新解析链）、`test_tui_tb_defects_fix.py`（驳回面板往返）、s9 `test_tui_overall_s9.py`（导航冒烟）。

## 交付

1. VSCode 启动器解析链＋`.cmd` 经 `cmd /c`＋失败带尝试路径清单＋成功显启动器路径（细节见上）。
2. Esc 路由：`tb-reject` → 进入前视图（tb-detail 或 taskbooks）；`tb-new` → taskbooks；其余回退链不变。
3. 测试：新增/同步用例——VSCode 三分支（含 `.cmd` 走 `cmd /c` 的 argv 断言）；驳回面板 Esc 回详情页、从列表进驳回面板 Esc 回列表、新增面板 Esc 回列表；全量绿。

## 白名单（写权限）

- 可写：src/pysci/skills/orchestration/tools/tui、tests/skills/orchestration

## 验收预期

1. `uv run --no-sync pytest tests` 全绿（含新用例）。
2. 用户实机点「用 VSCode 打开原文」应成功唤起 VSCode（用户 PATH 有 `D:\XiGPrograms\vsCode\base\bin`）；失败时消息可自诊断。
3. 详情页 → 驳回面板 → Esc → 回详情页（用户将实机复看）。
4. s9 跨面钉（键位矩阵/禁语/滚动）与 polish-a 视觉判据仍绿。
