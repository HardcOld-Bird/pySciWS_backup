# uv sync 的 Windows 安全配方（pysci-dev venv-sync）

`[project.scripts]` 入口 / 依赖变更后要重建 venv 入口时，**用 `pysci-dev venv-sync`，
不要裸 `uv sync`**。本文放「为什么」与三态处置；命令面在 `dev.py::cmd_venv_sync`。

## 1. 缺陷（backlog 20261012-114239-devops，本机 2026-10-12 实发）

组长挂着 `pysci-orch.exe watch` 时跑 `uv sync`：

```
error: failed to remove file `...\.venv\Scripts\pysci-orch.exe`: 拒绝访问 (os error 5)
```

uv **在报这条错之前，已经把同目录其余入口删光了**。随后的现场是：

```
$ pysci-dev --help        →  command not found
$ pysci-figures …         →  command not found
$ pysci-compose …         →  command not found
```

危害有两层：一是容易被误诊成 PATH/安装问题（venv 里包其实装得好好的，只是 Scripts
入口没了）；二是**正在跑的会话当场失去工具面**——组员的 Stop hook、`pysci-dev`
交付自检全瞎，drain 卡死。裸 `uv sync` 的退出码也不足以判：它不知道自己删了什么。

## 2. Windows 锁语义（2026-10-12 本机实测，配方前提）

| 占住形态 | `os.remove` | `os.replace`（改名） | 可救？ |
| --- | --- | --- | --- |
| 运行中的**镜像**（FILE_SHARE_DELETE） | ✗ winerror 5 | **✓** | ✓ 改名让开，uv 原地重建 |
| 普通打开句柄（无 share-delete） | ✗ winerror 32 | ✗ winerror 32 | ✗ 只能请占用者关 |
| 无人占用 | ✓ | ✓ | 无需让路 |

所以「改名」既是**让路动作**又是**判据**：改名成、删除败 ⇒ 正是镜像锁，原路径已让开；
改名与删除都成 ⇒ 根本没被占（探测无破坏性，工具会按原字节还原，代价只有 mtime 变新）。
`_move_aside_locked()` 三态 `moved / "" / stuck` 即据此。

复现夹具的坑：复制 base venv 的 `python.exe` 当锁**不成立**（子进程找不到 `encodings`
立刻退出）；用 `C:\Windows\System32\ping.exe` 副本 `Popen([fake, "-t", "127.0.0.1"])`
才得到 `REMOVE-DENIED 5 / RENAME-OK` 的真形态——POSIX 的 unlink 看目录权限，模拟不出，
故涉锁测试用例都带 `skipif(os.name != "nt")`。

## 3. 命令面

```bash
pysci-dev venv-sync [--offline 等转发给 uv sync 的参数]
```

做五件事，顺序固定：

1. 读 `pyproject.toml` 的 `[project.scripts]`（读不到 ⇒ rc=2 直接拒，**不**当「无需核对」）；
2. 扫掉往次让路留下的 `<name>.exe.locked-*` 垃圾（占用者已退的才删得动，删不动静默跳过）；
3. 逐个对声明入口做改名探测，被占住的改名为 `<name>.exe.locked-<ts>` 让路，并用
   `wmic process get ProcessId,ExecutablePath /format:csv`（argv 直传，免疫 MSYS 把
   `/format:` 转成路径）**点名占用者**——镜像路径在创建时定格，改名后仍显示原名，
   故按原名匹配；
4. `uv sync`（cwd 锁 `PROJECT_ROOT`，超时 1800 s）；
5. **核对声明过的入口是否全在盘**：缺 ⇒ 打印缺失清单并 rc=2；齐 ⇒ best-effort 删
   本次的 `.locked-*` 残留（删不掉只提示「占用进程退出后可删」，交给下次运行的第 2 步）。

退出码：`0` 入口齐备 / `2` 半坏态未消除或前置条件不满足 / 其余透传 uv 的 rc。
关键点是**把「半坏态」从静默变成显式失败**——旧现场里 uv 报一句锁错误就走，谁也不知道
另外七八个入口已经没了。

## 4. 三态处置

- `moved`：正常路径，继续 sync。残留的 `.locked-*` 是**正在运行的那个进程的本体**，
  等它退出再删；sync 期间别删（删不掉，且没必要）。
- `stuck`：连改名都被拒 ⇒ 有非镜像句柄捏着这文件。**请组长关掉占用程序**（工具已报
  node+PID+路径），❌ **绝不擅自 `kill` 组长进程**：越权，且会让在跑的 drain 会话
  当场失踪（比半坏态更难查）。
- 非 Windows：无 exe 锁形态，工具直接 sync 并仍核对入口（入口名在 POSIX 下是脚本文件，
  `scripts_dir` 布局不同则 rc=2 提示，不误判）。

## 5. 谁能把它跑起来（半坏态自救的关键一环）

`uv run pysci-X` 会**先隐式 sync 项目环境**再执行。所以 pyproject 一变，这条隐式 sync
自己就撞同一把锁——而且**工具本体根本没被跑起来**（2026-10-12 main 实测：
`uv run pysci-dev venv-sync` 只吐出 uv 的一行 `failed to remove file ... pysci-orch.exe:
拒绝访问 (os error 5)`，rc=2，`.venv/Scripts` 当场少掉 6 个入口）。入口被删光时
`pysci-dev.exe` 自己也在被删之列，于是「用新工具救火」这条路的正确走法是**绕开 uv**：

```bash
.venv/Scripts/python.exe -m pysci.skills.devops.tools.dev venv-sync   # python.exe 不是入口，uv 不动它
uv run --no-sync pysci-dev venv-sync                                  # 入口还在盘时可用
```

## 6. main 真仓验收（2026-10-12，backlog 20261012-114239-devops）

组长 `pysci-orch.exe`（PID 9804）真在跑 ⇒ 现场即缺陷本体。pyproject 新增一个 console
script 后跑上面第一条：

```
  [√] pysci-orch.exe 被占住 → 已改名让路（pysci-orch.exe.locked-20261012124035）；占用者=ZHUANZ PID 9804
== uv sync ==  Resolved 217 packages / Installed 1 package
  [i] 残留 pysci-orch.exe.locked-20261012124035 仍被占用（占用进程退出后可删）
[√] 入口齐备（[project.scripts] 11 个全在盘）      rc=0
```

11 个声明入口（含新增的那个）逐个 `--help` 均 rc=0；被让路的那个文件在进程存活期删不掉，
由下一次运行的第 2 步（sweep）腾掉。同场另一枚旧残留 `.locked-20261012` 是本轮改造前手工
删掉的——删得动，正说明「占用者退出后即可回收」这条判据成立。

## 7. 顺带纪律

- worktree 里**不要**跑 `venv-sync`：`PROJECT_ROOT` 就近取 `pyproject.toml`，会去重建
  worktree 那套环境（分钟级、且不是你要的那套）。跨树用 `PYTHONPATH="<wt>/src"` 配方，
  见 `references/worktree-merge.md` §1。
- pyproject 依赖变更只在**主树**同步一次；组员会话中途 `uv sync` 前先想「谁的 exe 正在跑」。
