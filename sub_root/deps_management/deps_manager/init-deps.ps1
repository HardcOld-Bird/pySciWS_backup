# init-deps.ps1 - bootstrap the project-local toolchain for pySciWS2.
# Layout follows the shared shim layer (D:\XXXIIIGGG\common\bin):
#   deps/git/<ver>/cmd/git.exe
#   deps/node/runtime/<ver>/node.exe
#   deps/node/modules/{package.json,package-lock.json,node_modules,dsh_home}
#   deps/uv/<ver>/uv.exe
#   deps/python/<cpython-dir>/python.exe
# ASCII-only comments (Windows PowerShell 5.1 + no BOM).
#
# Usage:
#   init-deps.cmd [-CheckOnly] [-SkipDsh] [-ProxyUrl http://127.0.0.1:12450]
param(
    [switch]$CheckOnly,
    [switch]$SkipDsh,
    [string]$ProxyUrl = ""
)

$ErrorActionPreference = "Stop"

$DepsRoot = Split-Path -Parent $PSScriptRoot
$ToolsRoot = Join-Path $DepsRoot "deps"
# Layout: <workspace>/sub_root/{pyproject.toml,.venv,src,data} + deps_management/.
# $ProjRoot is the python project root (sub_root); $WsRoot is the workspace/git root.
$ProjRoot = Split-Path -Parent $DepsRoot
$WsRoot = Split-Path -Parent $ProjRoot
$ToolchainPath = Join-Path $DepsRoot "toolchain.json"
if (-not (Test-Path -LiteralPath $ToolchainPath)) {
    throw "toolchain.json not found: $ToolchainPath"
}
$tc = Get-Content -LiteralPath $ToolchainPath -Raw -Encoding UTF8 | ConvertFrom-Json

function Say([string]$msg) { Write-Host "[init-deps] $msg" }
function Warn([string]$msg) { Write-Host "[init-deps][warn] $msg" -ForegroundColor Yellow }

function Get-Archive([string]$tool, [string]$urlTemplate, [string]$sha256) {
    $realUrl = $urlTemplate -replace '\{version\}', $tc.$tool.version
    $leaf = ($realUrl -split '\?')[0]
    $name = Split-Path -Leaf $leaf
    if ([string]::IsNullOrWhiteSpace($name)) { $name = "$tool.archive" }
    $dir = Join-Path $DepsRoot "cache\archives\$tool"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $file = Join-Path $dir $name
    if (Test-Path -LiteralPath $file) {
        $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower()
        if ($hash -eq $sha256.ToLower()) {
            Say "archive cache hit: $name"
            return $file
        }
        Say "cached archive hash mismatch, re-downloading: $name"
        Remove-Item -LiteralPath $file -Force
    }
    Say "downloading $realUrl"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if ($ProxyUrl -ne "") {
        Invoke-WebRequest -UseBasicParsing -Uri $realUrl -OutFile $file -Proxy $ProxyUrl
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $realUrl -OutFile $file
    }
    $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $sha256.ToLower()) {
        Remove-Item -LiteralPath $file -Force
        throw "SHA256 mismatch for $name (got $hash want $sha256)"
    }
    Say "verified SHA256: $name"
    return $file
}

function Test-Installed([string]$homeRel, [string]$exeRel) {
    $toolHome = Join-Path $ToolsRoot $homeRel
    $pointer = Join-Path $toolHome "current.txt"
    if (-not (Test-Path -LiteralPath $pointer)) { return $false }
    $ver = (Get-Content -LiteralPath $pointer -Raw).Trim()
    $exe = Join-Path (Join-Path $toolHome $ver) $exeRel
    return (Test-Path -LiteralPath $exe)
}

function Install-Tool([string]$tool, [string]$homeRel) {
    $spec = $tc.$tool
    if ($null -eq $spec) { throw "toolchain.json missing tool: $tool" }
    $ver = $spec.version
    $toolHome = Join-Path $ToolsRoot $homeRel
    $dest = Join-Path $toolHome $ver
    $pointer = Join-Path $toolHome "current.txt"
    if ((Test-Path -LiteralPath $dest) -and (Test-Path -LiteralPath $pointer) -and ((Get-Content -LiteralPath $pointer -Raw).Trim() -eq $ver)) {
        Say "$tool $ver already installed"
        return
    }
    if ($CheckOnly) {
        Warn "$tool $ver not installed"
        return
    }
    $archive = Get-Archive $tool $spec.url $spec.sha256
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $stage = Join-Path $DepsRoot "cache\stage-$tool"
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    & tar -xf $archive -C $stage
    if ($LASTEXITCODE -ne 0) { throw "extract failed: $archive" }
    $children = @(Get-ChildItem -LiteralPath $stage)
    if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
        Get-ChildItem -LiteralPath $children[0].FullName | Move-Item -Destination $dest
    } else {
        $children | Move-Item -Destination $dest
    }
    Remove-Item -LiteralPath $stage -Recurse -Force
    Set-Content -LiteralPath $pointer -Value $ver -Encoding ascii
    Say "$tool $ver installed at $dest"
}

# Reads one key from a git config file and returns "" when it is absent.
# git config --get exits 1 on a missing key (and PowerShell drops empty-string
# arguments to native commands, so --default "" is not usable here).
function Get-GitKey([string]$exe, [string]$file, [string]$key) {
    $val = ""
    try {
        $out = & $exe config --file $file --get $key 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $val = ([string]($out | Select-Object -First 1)).Trim()
        }
    } catch {
        $val = ""
    }
    return $val
}

# Project-local git global config. The shared shim (common\bin\git.cmd) points
# GIT_CONFIG_GLOBAL at <deps>/git/config/gitconfig and GIT_CONFIG_SYSTEM at a
# file the vendored tarball does not ship, so this generated file is the only
# carrier of git identity and the credential helper inside the project.
# Existing keys are never overwritten: identity is filled in on first init only,
# so later project-specific edits survive update/rollback runs.
function Set-ProjectGitConfig {
    $gitHome = Join-Path $ToolsRoot "git"
    $gitExe = Join-Path $gitHome ($tc.git.version + "\cmd\git.exe")
    $cfgPath = Join-Path (Join-Path $gitHome "config") "gitconfig"
    if (-not (Test-Path -LiteralPath $gitExe)) {
        Warn "project git $($tc.git.version) not installed; git global config skipped"
        return
    }
    if ($CheckOnly) {
        if (Test-Path -LiteralPath $cfgPath) {
            $curName = Get-GitKey $gitExe $cfgPath "user.name"
            $curEmail = Get-GitKey $gitExe $cfgPath "user.email"
            $curHelper = Get-GitKey $gitExe $cfgPath "credential.helper"
            Say "git global config ok: user=$curName <$curEmail> credential.helper=$curHelper"
        } else {
            Warn "git global config missing: $cfgPath"
        }
        return
    }
    $cfgDir = Split-Path -Parent $cfgPath
    if (-not (Test-Path -LiteralPath $cfgDir)) {
        New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
    }
    if (-not (Test-Path -LiteralPath $cfgPath)) {
        $header = "# pySci project-local git global config (target of GIT_CONFIG_GLOBAL`r`n" +
                  "# in the shared shim). Written by init-deps.ps1. Git identity and the`r`n" +
                  "# credential helper live here because the vendored git ships no system`r`n" +
                  "# config. Edit freely: init only fills keys that are still missing.`r`n"
        [IO.File]::WriteAllText($cfgPath, $header, (New-Object Text.UTF8Encoding($false)))
    }
    $hostCfg = ""
    if ($env:USERPROFILE) { $hostCfg = Join-Path $env:USERPROFILE ".gitconfig" }
    $hostName = ""
    $hostEmail = ""
    if ($hostCfg -ne "" -and (Test-Path -LiteralPath $hostCfg)) {
        $hostName = Get-GitKey $gitExe $hostCfg "user.name"
        $hostEmail = Get-GitKey $gitExe $hostCfg "user.email"
    }
    $curName = Get-GitKey $gitExe $cfgPath "user.name"
    if ($curName -eq "") {
        if ($hostName -ne "" -and $hostEmail -ne "") {
            & $gitExe config --file $cfgPath user.name $hostName
            & $gitExe config --file $cfgPath user.email $hostEmail
            Say "git identity seeded from ${hostCfg}: $hostName <$hostEmail>"
        } else {
            Warn "no user.name/user.email in $hostCfg; set them with:"
            Warn "  git config --global user.name `"Your Name`""
            Warn "  git config --global user.email `"you@example.com`""
        }
    } else {
        Say "git identity kept: $curName"
    }
    & $gitExe config --file $cfgPath credential.helper manager
    $gcmOk = $false
    try {
        $null = & $gitExe credential-manager --version 2>&1
        $gcmOk = ($LASTEXITCODE -eq 0)
    } catch {
        $gcmOk = $false
    }
    if (-not $gcmOk) {
        Warn "credential.helper=manager set, but git-credential-manager did not answer; pushes will need credentials"
    }
    Say "git global config: $cfgPath"
}

Say "project root: $ProjRoot"
Say "tools root  : $ToolsRoot"
if ($CheckOnly) { Say "mode        : check only" }

# --- vendored tools ---------------------------------------------------------
Install-Tool "git" "git"
Install-Tool "node" "node\runtime"
Install-Tool "uv" "uv"

# --- project-local git global config (identity + credential helper) ----------
Set-ProjectGitConfig

# --- Python via project uv --------------------------------------------------
$pyVer = $tc.python.version
$pyRoot = Join-Path $ToolsRoot "python"
$pyPointer = Join-Path $pyRoot "current.txt"
$uvExe = Join-Path $ToolsRoot ("uv\" + $tc.uv.version + "\uv.exe")
$env:UV_PYTHON_INSTALL_DIR = $pyRoot
$env:UV_CACHE_DIR = Join-Path $DepsRoot "cache\packages\uv"
$havePy = $false
if (Test-Path -LiteralPath $pyPointer) {
    $pyDirName = (Get-Content -LiteralPath $pyPointer -Raw).Trim()
    $havePy = Test-Path -LiteralPath (Join-Path (Join-Path $pyRoot $pyDirName) "python.exe")
}
if ($havePy) {
    Say "CPython $pyVer already installed"
} elseif ($CheckOnly) {
    Warn "CPython $pyVer not installed"
} else {
    Say "installing CPython $pyVer via project uv"
    & $uvExe python install $pyVer
    if ($LASTEXITCODE -ne 0) { throw "uv python install failed" }
    $pyDir = @(Get-ChildItem -LiteralPath $pyRoot -Directory | Where-Object { $_.Name -like "cpython-*" } | Select-Object -First 1)
    if ($pyDir.Count -eq 0) { throw "no cpython-* dir under $pyRoot" }
    Set-Content -LiteralPath $pyPointer -Value $pyDir[0].Name -Encoding ascii
    Say "CPython installed at $($pyDir[0].FullName)"
}

# --- shared shim layer (project-external routing only) ----------------------
$commonRoot = Split-Path (Split-Path $WsRoot -Parent) -Parent
$shimDir = Join-Path $commonRoot "common\bin"
if (Test-Path -LiteralPath (Join-Path $shimDir "_find_project.cmd")) {
    Say "shared shim layer present: $shimDir"
} else {
    Warn "shared shim layer missing at $shimDir; bare git/node/uv/dsh will not route"
}

# --- DSH (npm, managed) -----------------------------------------------------
$modules = Join-Path $ToolsRoot "node\modules"
$nodeHome = Join-Path $ToolsRoot "node\runtime"
$nodePointer = Join-Path $nodeHome "current.txt"
$dshHome = Join-Path $modules "dsh_home"
if (-not $SkipDsh) {
    if (-not (Test-Path -LiteralPath (Join-Path $modules "package.json"))) {
        throw "missing $modules\package.json"
    }
    $nodeVer = (Get-Content -LiteralPath $nodePointer -Raw).Trim()
    $nodeDir = Join-Path $nodeHome $nodeVer
    $npmCmd = Join-Path $nodeDir "npm.cmd"
    if (-not (Test-Path -LiteralPath $npmCmd)) { throw "npm not found: $npmCmd" }
    $dshCmd = Join-Path $modules "node_modules\.bin\dsh.cmd"
    $dshInstalled = Test-Path -LiteralPath $dshCmd
    if ($CheckOnly) {
        if ($dshInstalled) { Say "DSH already installed: $dshCmd" } else { Warn "DSH not installed" }
    } else {
        $env:NPM_CONFIG_CACHE = Join-Path $DepsRoot "cache\packages\npm"
        $env:NPM_CONFIG_PREFIX = Join-Path $nodeHome "npm-global"
        Push-Location $modules
        try {
            if ($dshInstalled) {
                Say "DSH already installed; running npm install for lock consistency"
            } else {
                Say "installing @deepseek-ai/dsh $($tc.dsh.version)"
            }
            & $npmCmd install --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) { throw "npm install failed in $modules" }
        } finally {
            Pop-Location
        }
    }
    # project DSH home + credentials (never commit)
    if ($CheckOnly) {
        $credSrc = Join-Path $env:USERPROFILE ".dsh\.credentials.yaml"
        $credDst = Join-Path $dshHome ".credentials.yaml"
        if (Test-Path -LiteralPath $credDst) {
            Say "project dsh_home present"
        } elseif (Test-Path -LiteralPath $credSrc) {
            Warn "project dsh_home credentials missing (will copy on real init)"
        } else {
            Warn "no credentials source at $credSrc"
        }
    } else {
        New-Item -ItemType Directory -Force -Path $dshHome | Out-Null
        $credSrc = Join-Path $env:USERPROFILE ".dsh\.credentials.yaml"
        $credDst = Join-Path $dshHome ".credentials.yaml"
        if ((Test-Path -LiteralPath $credSrc) -and -not (Test-Path -LiteralPath $credDst)) {
            Copy-Item -LiteralPath $credSrc -Destination $credDst -Force
            Say "copied credentials into project dsh_home"
        } elseif (-not (Test-Path -LiteralPath $credDst)) {
            Warn "no credentials copied; set DEEPSEEK_API_KEY or copy $credSrc manually"
        }
    }
    if (-not $CheckOnly) {
        $env:DSH_HOME = $dshHome
        $env:DSH_AGENTS_HOME = Join-Path $dshHome "agents"
        $env:PATH = $nodeDir + ";" + $env:PATH
        if (Test-Path -LiteralPath $dshCmd) {
            Say "dsh version:"
            & $dshCmd --version
        }
    }
}

# --- venv (only once pyproject.toml exists) --------------------------------
$pyProject = Join-Path $ProjRoot "pyproject.toml"
if (Test-Path -LiteralPath $pyProject) {
    if ($CheckOnly) {
        Say "pyproject.toml present; venv sync skipped in check-only"
    } else {
        Say "(re)building .venv from pyproject.toml"
        Push-Location $ProjRoot
        try {
            & $uvExe venv --python $pyVer
            if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
            & $uvExe sync
            if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
        } finally {
            Pop-Location
        }
    }
} else {
    Say "pyproject.toml not present yet; venv creation deferred (P4)"
}

# --- verify ----------------------------------------------------------------
if (-not $CheckOnly) {
    $gitExe = Join-Path $ToolsRoot ("git\" + $tc.git.version + "\cmd\git.exe")
    $nodeExe = Join-Path $ToolsRoot ("node\runtime\" + $tc.node.version + "\node.exe")
    $pyDirName = if (Test-Path -LiteralPath $pyPointer) { (Get-Content -LiteralPath $pyPointer -Raw).Trim() } else { "" }
    $pyExe = if ($pyDirName -ne "") { Join-Path (Join-Path $pyRoot $pyDirName) "python.exe" } else { $null }
    if (Test-Path -LiteralPath $gitExe) { & $gitExe --version }
    if (Test-Path -LiteralPath $nodeExe) { & $nodeExe --version }
    if (Test-Path -LiteralPath $uvExe) { & $uvExe --version }
    if ($pyExe -and (Test-Path -LiteralPath $pyExe)) { & $pyExe --version }
    Say "done"
}
