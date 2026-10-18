# 交付全文留档：devops-20261018-155546-wake-bind-20261018-s1-noconsole

- member: devops
- kind: result
- code: 0
- plan: wake-bind-20261018
- step: s1-noconsole
- task_file: orchestration/state/plans/wake-bind-20261018-s1.task.md
- sid: c64eed46-170d-42f7-a9a9-eb089f4a697b
- duration_ms: 2031200
- ts: 2026-10-18T16:29:40+08:00

## 交付正文

空终端根治已完成并合并：钩子链与无控制台可达的 Python spawn 全部隐藏化，A/B 语义逐字一致，全量绿并已推送。

## 改动清单
**1. 钩子脚本 spawn 隐藏化（根治面）**
- `.qoder/hooks/watch-arm.sh`：`PY`/`ORCH` console 启动器全部退役，改 `PYW=.venv/Scripts/pythonw.exe`＋`orch() { PYTHONPATH="$ROOT/src" "$PYW" -m pysci.skills.orchestration.tools.orch "$@"; }`。夜航差分、`wake-claim --check`×2、`watchdog start`、`watch --timeout` 五处调用保持原有 `$(...)`/`2>>LOG`/`2>&1` 重定向与退出码语义。
- `.qoder/hooks/night-gate.sh`、`night-policy.sh`：`night` 调用改 pythonw；here-string/stdin、stderr 注入、`additionalContext` stdout 语义不变；环境缺失仍 fail-open。
- `turnbudget._hook_exe()`：win32 解析到同目录 `pythonw.exe`，PostToolUse 告警不再裸起 console 解释器；非 win32 保持 `sys.executable`。

**2. 无控制台派生链清点（站点 → 处置）**

| 站点 | 处置 |
|---|---|
| watch-arm.sh ×5 / night-gate.sh / night-policy.sh | pythonw.exe（本单修复） |
| `orchestration/tools/turnbudget.py` hook | pythonw.exe（本单修复） |
| devops `dev.py`(5)、`mdgen.py`(1)、`probe.py`(3)、`skilltax.py`(1)、`keyscan.py`(1)、`guard_probe.py`(1) | 并入唯一口径 `procs.run_hidden`（本单修复，共 12 处） |
| `orchestration/tools` 其余 spawn（runner/drain/statecommit/registry/checks/eval…） | 已在 `procs.run_hidden/popen_hidden`（前单清零，本单复核） |
| `tui/app.py` 起 VS Code | **豁免**：interactive-editor，源码留 `spawn-hidden-exempt:` 标记（交互可见属预期） |
| comsol/document_writing/lit/model3d/theory 等包内 spawn | **不可达**：只在成员会话内执行（CLI 由 CREATE_NO_WINDOW 拉起、持有不可见控制台），不在 pythonw tick / DETACHED worker / harness 钩子链上；且不在本单写权限内 |

**3. 回归钉子**
- `test_hidden_spawn_sites.py`：三个钩子脚本非注释行禁止 `python.exe`/`pysci-orch.exe`，并逐条钉住重定向与参数；orch `__main__` 守卫；turnbudget win32 必须解析到 pythonw。
- 扩展 `test_unmerged_worktree_alert.py` 判据①：递归扫 `orchestration/tools + devops/tools`，除 `procs.py` 与显式豁免（须双证：名单+源码标记）外禁裸 spawn。
- 更新 `test_night_diff.py` 桩根到 pythonw/`-m orch` 新契约（端到端 exit 2＋stderr 仍实测）。

## 实测取证（win32，全部落盘可复核）
- **无控制台 A/B**（pythonw 父进程 + `DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW` 起 bash，模拟 `_rearm` 真实形态）：控制组 python.exe 与实验组 pythonw **rc=2、stdout 空、stderr 均为 `EVENT probe-hit`**；四次 spawn 的 `sys.executable` 分别为 `python.exe` / `pythonw.exe`；实验组运行全程 30ms 轮询**零新增可见窗口**。
- **night-gate** 旧/新脚本：rc 2/2、stdout 0B/0B、stderr 265B **逐字一致**；**night-policy**：rc 0/0、stdout 1619B **逐字一致**、stderr 0B。
- **direct 对照**：`python` vs `pythonw` 跑 `orch --help` 输出完全一致；`night --gate` rc2/stderr 一致；policy 输出一致。
- **turnbudget 真 CLI**：hook 命令现为 `...\pythonw.exe ...turnbudget.py`，第 4/5 轮触发后 transcript 可见告警，模型实际读取；marker 落盘正常。
- 全量：worktree `4279 passed, 42 skipped, 6 deselected`；合并后 main `4284 passed, 37 skipped, 6 deselected`（0 failed）。
- `ruff check/format` 全过；`sync --check` 一致；未动 `pyproject.toml`。

## 合并
- 实施提交 `4128702`；合并提交 `351ffea`；已推送 `bec0bba..351ffea main -> main`。
- 合并时 worktree 运行态 90 < main 91，按工具指引走 `--no-cp-state` 保留 main 更丰富的会话池，会话数自检 91→91；worktree 已自动摘除。

<artifact check="none" reason="根治面：重臂监听循环全部 venv 调用改 pythonw，A/B rc/stderr 逐字一致＋零新可见窗口实测">.qoder/hooks/watch-arm.sh</artifact>
<artifact check="none" reason="夜航门禁改 pythonw，旧/新 rc2/stderr 265B 逐字一致">.qoder/hooks/night-gate.sh</artifact>
<artifact check="none" reason="夜航政策注入改 pythonw，旧/新 stdout 1619B 逐字一致">.qoder/hooks/night-policy.sh</artifact>
<artifact check="none" reason="PostToolUse hook 解释器解析到 pythonw；真 CLI 注入实测">src/pysci/skills/orchestration/tools/turnbudget.py</artifact>
<artifact check="none" reason="devops 工具 12 处裸 spawn 并入 procs.run_hidden（本文件 5 处）">src/pysci/skills/devops/tools/dev.py</artifact>
<artifact check="none" reason="新增钩子脚本/新站点机械闸与 pythonw 合约钉子">tests/skills/orchestration/test_hidden_spawn_sites.py</artifact>
<artifact check="none" reason="判据①扩展：递归覆盖 orchestration/tools + devops/tools，豁免双证">tests/skills/orchestration/test_unmerged_worktree_alert.py</artifact>
<artifact check="none" reason="端到端 rewake 测试适配 pythonw/-m 新契约（25 项全绿）">tests/skills/orchestration/test_night_diff.py</artifact>
