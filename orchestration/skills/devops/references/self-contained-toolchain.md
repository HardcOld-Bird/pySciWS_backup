# 自包含工具链细则（sub_root/deps_management，2026-10-25 改造；2026-11-01 布局迁移 deps_management）

basic.md §5 只留公约数；本页是操作细节与踩坑记录（devops/组长按需读）。

## 1. 布局（与 mySimsWS2 同范式，共享 shim 免费路由）

```
sub_root/deps_management/
  toolchain.json      # 版本唯一真源：git/node/uv = version+url+sha256；python/uvTools 记版本
  deps_manager/       # init-deps.cmd(.ps1) 引导；update-deps.cmd(.ps1) 升级单件
  deps/
    git/<ver>/ + git/current.txt
    node/<ver>/ + node/current.txt
    uv/<ver>/ + uv/current.txt
    uv/tools/ + uv/bin/   # UV_TOOL_DIR/UV_TOOL_BIN_DIR：zotero/arxiv/paper-search/blender-mcp
    python/           # UV_PYTHON_INSTALL_DIR（cpython-3.13.15/3.12 各一目录）+ current.txt
    comsol-mcp/       # 社区 COMSOL MCP 仓库（钉 commit，venv 项目内重建）
  cache/              # UV_CACHE_DIR（packages/uv）+ archives/（下载归档缓存）
```

`D:\XXXIIIGGG\common\bin` 共享 shim 从 `%CD%` 向上找
`sub_root\deps_management\toolchain.json` 路由到 `deps\<tool>`——布局同相对路径即零改动复用。
**迁移注意（2026-11-01 实案）**：uv trampoline exe（venv python、uv tool console script）把
绝对路径嵌死在二进制里，依赖树挪位后 `pyvenv.cfg` 改 home 救不回——须 `uv tool install
--force` / 重建 venv 重生成；`deps/python/cpython-<ver>-windows-x86_64-none` 别名 symlink
要用**相对目标**（绝对目标挪位即断）。

## 2. 调用面（三层，各有其位）

1. **程序化 subprocess**：一律 `pysci.toolchain` 解析绝对路径（`git_exe`/`uv_exe`/
   `node_exe`/`bash_exe`/`uv_tool_exe`/`resolve_command`）。Windows 上 CreateProcess
   找不到 PATH 里的 `.cmd` shim——裸名 + shim = FileNotFoundError（2026-10-25 实测）。
2. **hook / guard**：一律 **`.sh` 直跑**——`.qoder/hooks/*.sh`、`orchestration/guards/
   run-guard.sh <guard.mjs>`（settings.json 里直接写相对 `.sh` 路径；run-guard.sh 内部
   用 vendored node 绝对路径）。**`cmd /c …` 形态已淘汰**（2026-10-26 实测：
   `cmd /c <反斜杠路径>` 在钩子执行语境下 `/c` 被 MSYS 路径转换、反斜杠被剥离，cmd 落入
   交互态输出横幅、护栏**静默失效**；细则见 run-guard.sh 头注）。原 `run-hook.cmd`/
   `run-guard.cmd` 两个批处理入口已删除（全仓零引用，死文件，勿再引入）。
3. **会话/测试 PATH**：`toolchain.prepend_toolchain_path()` 前插 vendored 目录
   （tests/conftest.py 与 runner.build_env 已接）——裸 `git`/`node`/`bash` 就地可用，
   pre-commit 等第三方工具同理。

## 3. 引导与升级

- 换机/重建：`sub_root\deps_management\deps_manager\init-deps.cmd`（PowerShell 5.1 执行；
  下载→SHA256→解压→写 current.txt→venv→逐件 --version 对账；`-SkipVenv` 跳过 venv）。
- 升级单件：`update-deps.cmd -Tool <名> -Version <版> -Url <url> -Sha256 <hex>`
  （改 toolchain.json 后重走 init-deps）。
- uv-tool 升级：`toolchain.json` 的 `uvTools` 记版本，装完 `uv tool list` 对账。
  **注意 `uv tool install` 的 venv 基座**：装前设 `UV_PYTHON_INSTALL_DIR`（WSLENV 通道），
  否则工具 venv 基座挂到外部 Python、该 Python 被清后全部 exe 报
  `did not find executable at ...`（2026-10-26 实案，`--reinstall` 重建即愈）。

## 4. 本机 shell 环境硬事实（2026-10-25/26 实测）

- harness Bash 是 WSL bash：**env 前缀传不进 Windows 子进程**（WSL 只共享 `WSLENV`
  列出的变量）——`WSLENV=UV_TOOL_DIR:UV_CACHE_DIR … uv.exe …` 才生效；
- 同一 shell 的 `$var` 会被包装层提前展开（变量赋值/`$?` 观察不可靠）；验证退出码用
  cmd 的 `||` 探针（非零 → 打印标记）或在 Git Bash 内观察；
- `cmd /c` 的内层引号经 WSL 传递会被原样带给 cmd——整段用 bash 单引号包裹；
- MSYS 把 `/c` 之类参数转成路径：cmd 参数写 `//c` 或走 argv 直传。
