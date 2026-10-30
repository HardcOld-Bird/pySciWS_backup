<#
watchdog-scheduler.ps1 - OS-level persistence for the `pysci` watch watchdog (autonomy-fix a1)

Why: `pysci-orch watchdog start` spawns a detached watchdog, but it is gone after a reboot /
re-logon. This script registers a Windows Scheduled Task that runs `orch watchdog start` every
5 minutes. `start` is idempotent (no-op when one is already running), so a 5-minute tick gives
TWO kinds of self-healing at once:
  * nobody spawned it after reboot/re-logon  -> next tick spawns it;
  * the watchdog or its listener was killed silently (the 2026-10-14 02:23 case) -> next tick
    re-arms it.
This is fully decoupled from any session/hook lifetime - the missing last link of that 5-hour
wake-up vacuum.

Install (idempotent; no admin required for the current user):
    powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode install
Uninstall (removes the TASK only; stop the watchdog PROCESS with `pysci-orch watchdog stop`):
    powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode uninstall
Inspect:
    powershell -ExecutionPolicy Bypass -File orchestration\scripts\watchdog-scheduler.ps1 -Mode status

Residual risk: schtasks registers under the current user + interactive token, so the task does
NOT run while the user is logged off (no leader session can be woken then either, so impact is
zero). Add /ru <account> /rp <password> and run elevated for run-whether-logged-on-or-not.

NOTE: this file is deliberately ASCII-only. PowerShell 5.1 decodes UTF-8 WITHOUT a BOM as the
ANSI code page; a mis-decoded CJK byte can render as a backtick, which escapes the next line and
swallows real code - that broke the first version of this script (ParserError on `switch`).
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "uninstall", "status")]
    [string]$Mode,

    [string]$TaskName = "pysci-watchdog"
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot = <repo>/orchestration/scripts  ->  two levels up = repo root
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python   = Join-Path $RepoRoot ".venv\Scripts\python.exe"
# HIDDEN HOST (empty-terminal fix, 2026-10-14 user report): python.exe is a console
# subsystem binary, so a Task Scheduler tick with an interactive token pops a visible
# (empty) console every 5 minutes. pythonw.exe is GUI subsystem -> no console window at
# all, and the action only runs `orch watchdog start` (spawn-and-exit). Fall back to the
# console binary when pythonw.exe is absent (venv layout variance) so the task never
# silently stops working.
$Pythonw  = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
$HostExe  = if (Test-Path $Pythonw) { $Pythonw } else { $Python }
$Action   = "`"$HostExe`" -m pysci.skills.orchestration.tools.orch watchdog start"

switch ($Mode) {
    "install" {
        if (-not (Test-Path $Python)) {
            Write-Host "[!] venv interpreter not found: $Python (run uv sync first)"
            exit 2
        }
        # /f = idempotent (re-register silently over the same task name)
        schtasks /create /tn $TaskName /sc minute /mo 5 /tr $Action /f | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[!] schtasks /create failed (rc=$LASTEXITCODE) - likely needs elevation"
            exit $LASTEXITCODE
        }
        Write-Host "[OK] task '$TaskName' registered (every 5 min: orch watchdog start; idempotent)"
        Write-Host "     inspect : schtasks /query /tn $TaskName"
        Write-Host "     stop    : pysci-orch watchdog stop"
        exit 0
    }
    "uninstall" {
        schtasks /query /tn $TaskName /fo list | Out-Null
        if ($LASTEXITCODE -eq 0) {
            schtasks /delete /tn $TaskName /f | Out-Null
            Write-Host "[OK] task '$TaskName' deleted (also stop the process: pysci-orch watchdog stop)"
        } else {
            Write-Host "[=] task '$TaskName' was not installed (idempotent uninstall)"
        }
        exit 0
    }
    "status" {
        schtasks /query /tn $TaskName /fo list
        Write-Host "  watchdog process readings: pysci-orch watchdog status"
        exit 0
    }
}
