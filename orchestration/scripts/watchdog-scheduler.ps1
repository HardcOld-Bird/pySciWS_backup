<#
watchdog-scheduler.ps1 — watch 看门狗的 OS 侧持久化（autonomy-fix a1）

目的：`pysci-orch watchdog start` 拉起的看门狗是**会话外**进程，但机器重启 / 用户重登之后
它就不存在了。本脚本用 Windows 任务计划（schtasks）**每 5 分钟**跑一次 `orch watchdog start`——
而 `start` 是**幂等**的（已在跑则不动），于是「每 5 分钟跑一次」同时充当两种自愈：
  ·  开机/重登后没人拉起它 → 下一个 5min 点自动拉起；
  ·  看门狗或 listener 被无声杀死（2026-10-14 02:23 那类）→ 下一个 5min 点自动重臂。
它与**任何**会话/钩子生命周期解耦——这正是那次五小时唤醒真空所缺的最后一环。

安装（幂等，可反复执行；不需管理员）：
    powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode install
卸载（删除任务本身；看门狗**进程**另用 `pysci-orch watchdog stop` 停止）：
    powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode uninstall
查看：
    powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode status

残余风险：`schtasks` 默认以**当前用户 + 交互令牌**注册——用户未登录时任务不会跑（看门狗
因此在未登录窗口内缺席；此时也没有组长会话可被唤醒，影响面为零）。若需免登录运行，给本脚本
加 `/ru <账号> /rp <口令>` 并以**管理员**注册。
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "uninstall", "status")]
    [string]$Mode,

    [string]$TaskName = "pysci-watchdog"
)

$ErrorActionPreference = "Stop"

# PSScriptRoot = <repo>/orchestration/scripts → 上两级 = 仓库根
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Action   = "`"$Python`" -m pysci.skills.orchestration.tools.orch watchdog start"

switch ($Mode) {
    "install" {
        if (-not (Test-Path $Python)) {
            Write-Host "[!] 找不到 venv 解释器：$Python（先 uv sync）"
            exit 2
        }
        # /f = 幂等（重复注册覆盖同名任务，不弹确认）
        schtasks /create /tn $TaskName /sc minute /mo 5 /tr $Action /f
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "[√] 已注册任务 '$TaskName'（每 5 分钟 `orch watchdog start`，幂等）"
        Write-Host "    查看：schtasks /query /tn $TaskName"
        Write-Host "    停看门狗进程：pysci-orch watchdog stop"
        exit 0
    }
    "uninstall" {
        # 任务不存在时 schtasks /delete 会失败——按「幂等可卸载」降级为提示并退出 0。
        schtasks /query /tn $TaskName /fo list | Out-Null
        if ($LASTEXITCODE -eq 0) {
            schtasks /delete /tn $TaskName /f | Out-Null
            Write-Host "[√] 已删除任务 '$TaskName'（看门狗进程请另跑 `pysci-orch watchdog stop`）"
        } else {
            Write-Host "[=] 任务 '$TaskName' 本就不存在（幂等卸载）"
        }
        exit 0
    }
    "status" {
        schtasks /query /tn $TaskName /fo list
        Write-Host "  看门狗进程读数：pysci-orch watchdog status"
        exit 0
    }
}
