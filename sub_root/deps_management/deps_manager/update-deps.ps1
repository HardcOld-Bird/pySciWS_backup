# update-deps.ps1 - unified dependency entry for pySciWS2.
# Actions: status (read-only), update (reconcile/install), log (toolchain history),
#          rollback -Ref <git-ref> (restore toolchain.json from ref, then reconcile).
# ASCII-only comments (Windows PowerShell 5.1 + no BOM).
param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'update', 'log', 'rollback')]
    [string]$Action = 'status',
    [Parameter(Position = 1)]
    [string]$Ref = ''
)

$ErrorActionPreference = "Stop"
$DepsRoot = Split-Path -Parent $PSScriptRoot
$ProjRoot = Split-Path -Parent (Split-Path -Parent $DepsRoot)
$ToolsRoot = Join-Path $DepsRoot "deps"
$ToolchainPath = Join-Path $DepsRoot "toolchain.json"
$tc = Get-Content -LiteralPath $ToolchainPath -Raw -Encoding UTF8 | ConvertFrom-Json
$gitExe = Join-Path $ToolsRoot ("git\" + $tc.git.version + "\cmd\git.exe")

function Say([string]$msg) { Write-Host "[update-deps] $msg" }
function Row([string]$tool, [string]$homeRel, [string]$exeRel) {
    $toolHome = Join-Path $ToolsRoot $homeRel
    $pointer = Join-Path $toolHome "current.txt"
    $ver = if (Test-Path -LiteralPath $pointer) { (Get-Content -LiteralPath $pointer -Raw).Trim() } else { '-' }
    $ok = $false
    if ($ver -ne '-') { $ok = Test-Path -LiteralPath (Join-Path (Join-Path $toolHome $ver) $exeRel) }
    $installed = if ($ok) { 'ok' } else { 'MISSING' }
    Write-Host ("  {0,-8} manifest={1,-12} installed={2,-12} {3}" -f $tool, $tc.$tool.version, $ver, $installed)
}

if ($Action -eq 'status') {
    Write-Host "=== pySciWS2 dependency status ==="
    Row "git" "git" "cmd\git.exe"
    Row "node" "node\runtime" "node.exe"
    Row "uv" "uv" "uv.exe"
    $pyRoot = Join-Path $ToolsRoot "python"
    $pyPointer = Join-Path $pyRoot "current.txt"
    $pyVer = if (Test-Path -LiteralPath $pyPointer) { (Get-Content -LiteralPath $pyPointer -Raw).Trim() } else { '-' }
    Write-Host ("  {0,-8} manifest={1,-12} installed={2,-12}" -f "python", $tc.python.version, $pyVer)
    $dshCmd = Join-Path $ToolsRoot "node\modules\node_modules\.bin\dsh.cmd"
    Write-Host ("  {0,-8} manifest={1,-12} installed={2}" -f "dsh", $tc.dsh.version, (Test-Path -LiteralPath $dshCmd))
    Write-Host "  dsh_home: $(Join-Path $ToolsRoot 'node\modules\dsh_home')"
    exit 0
}

if ($Action -eq 'log') {
    if (-not (Test-Path -LiteralPath $gitExe)) { throw "project git not installed" }
    & $gitExe -C $ProjRoot log --oneline -20 -- sub_root/deps_management/toolchain.json sub_root/deps_management/deps/node/modules/package.json
    exit $LASTEXITCODE
}

if ($Action -eq 'rollback') {
    if ([string]::IsNullOrWhiteSpace($Ref)) {
        throw "rollback requires a git ref: update-deps.cmd rollback <ref>"
    }
    if (-not (Test-Path -LiteralPath $gitExe)) { throw "project git not installed" }
    Say "restoring toolchain.json and package.json from $Ref"
    & $gitExe -C $ProjRoot checkout $Ref -- sub_root/deps_management/toolchain.json sub_root/deps_management/deps/node/modules/package.json
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
    Say "reconciling from restored manifests"
    & (Join-Path $PSScriptRoot "init-deps.ps1")
    exit $LASTEXITCODE
}

if ($Action -eq 'update') {
    Say "reconciling project toolchain (pinned by toolchain.json)"
    & (Join-Path $PSScriptRoot "init-deps.ps1")
    exit $LASTEXITCODE
}
