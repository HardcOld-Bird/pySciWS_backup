<#
================================================================================
 Phase 0 setup script -- community COMSOL MCP (wjc9011/COMSOL_Multiphysics_MCP)
================================================================================
 One-shot preparation tool for Phase 0 (validation spike) of the
 "COMSOL skill MCP refactor" plan. It sets up the COMMUNITY comsol MCP only.
 NOTE (post-decision): the in-house `pysci-comsol-extras` companion MCP was
 RETIRED -- mph.Client wedges when booting COMSOL inside the FastMCP/asyncio
 runtime, whereas the plain-script path works. All differentiated capabilities
 now run through the `pysci-simulation` CLI. This script no longer registers
 extras.

 What it does (idempotent, safe to re-run):
   1. Preflight: check git / uv availability.
   2. Clone the community repo and checkout the PINNED commit (detached HEAD).
   3. Create an isolated uv venv (default Python 3.12, avoids the 3.13 wheel
      gaps for torch/chromadb; kept separate from pysci@3.13).
   4. Install: CPU-only torch first (this machine has no NVIDIA GPU), then
      `uv pip install -e .`.
   5. Optional: apply the "shared session" patch (see CRITICAL FINDING below).
      Off by default.
   6. Lightweight verify: comsol-mcp console script exists + `src.server` is
      locatable (does NOT start the JVM, does NOT consume a license).
   7. Build the RAG knowledge base (CPU, all-MiniLM-L6-v2): timed + peak-RAM
      sampled (as the plan requires).
   8. Print the resolved mcpServers registration JSON (copy-paste ready) and
      the Go/No-Go checklist.

 NOTE: This file is intentionally ASCII-only. Windows PowerShell 5.1 reads a
 BOM-less .ps1 as the ANSI code page (GBK on zh-CN), which corrupts UTF-8
 non-ASCII bytes and breaks parsing. Keeping it ASCII makes it robust in any
 shell / PS version, foreground or background.

--------------------------------------------------------------------------------
 Usage
--------------------------------------------------------------------------------
   # Full flow (clone + venv + install + RAG), CPU torch:
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\comsol_mcp\setup_comsol_mcp.ps1

   # Print intended actions only, zero side effects (run this first):
   ... setup_comsol_mcp.ps1 -DryRun

   # Skip RAG (verify install/startup first, build the KB later):
   ... setup_comsol_mcp.ps1 -SkipRag

   # Quick subset build (first 8 PDFs only, proves the pipeline):
   ... setup_comsol_mcp.ps1 -RagLimit 8

   # Read-only: verify the pin against UPSTREAM.lock.json, then report KB status:
   ... setup_comsol_mcp.ps1 -StatusOnly

   # Enable single-license shared session (applies patch; then `pysci-simulation
   # server start` launches comsolmphserver and the community MCP attaches to it):
   ... setup_comsol_mcp.ps1 -ApplySharedSessionPatch

--------------------------------------------------------------------------------
 CRITICAL FINDING -- community MCP "eager start" vs. the single license
--------------------------------------------------------------------------------
 The community src/server.py main() calls session_manager.start() BEFORE
 mcp.run() when transport == "stdio" (an upstream workaround for a Windows +
 JPype stdin deadlock). Consequence: whenever Qoder launches the `comsol` MCP
 over stdio, it IMMEDIATELY starts a standalone COMSOL that grabs the single
 license inside that MCP's own process -- even if you are not running a sim yet.
 Because a standalone JVM is in-process, a separate process (e.g. the pysci
 `simulation` CLI) CANNOT attach to it. Two COMSOL clients at once would need 2
 licenses; we have 1. (The retired `pysci-comsol-extras` MCP is no longer part of
 this picture; differentiated work runs through the plain-script `simulation` CLI.)

 Three resolutions (pick one during Phase 0 based on live verification):
   A) Sequential use (simplest, no patch): only one MCP holds a COMSOL session
      at a time; comsol_disconnect / stop the server before switching.
      Recommended for the spike.
   B) Shared session (single license, community MCP + CLI on one server):
      - Run with -ApplySharedSessionPatch to add a 5-line patch to server.py:
        when env COMSOL_MCP_CONNECT_PORT is set, pre-start does connect(port)
        instead of a standalone start().
      - `pysci-simulation server start` launches the ONE persistent
        comsolmphserver (records pid/port in runs/comsol_server.json).
      - Register the community MCP with env COMSOL_MCP_CONNECT_PORT=2036 so it
        attaches to that same server; run CLI commands with --connect-port 2036.
      - If Qoder launches the community MCP before the persistent server is up,
        the pre-connect fails (non-fatal); just call comsol_connect(port=2036)
        after the CLI has started the server.
   C) HTTP transport (lazy COMSOL start): set COMSOL_MCP_TRANSPORT=http +
      COMSOL_MCP_PORT and register by url -- only if Qoder supports HTTP/SSE MCP
      registration (existing arxiv etc. are stdio; unconfirmed).

 Recommendation: use A for the spike; adopt B as the steady-state once Go.

--------------------------------------------------------------------------------
 Registration (add the community comsol MCP in Qoder)
--------------------------------------------------------------------------------
 User-level config file: C:\Users\antar\.qoder-cn\shared_client\mcp.json
 Shape (stdio): { "mcpServers": { "<name>": { "command":..., "args":[...], "env":{...} } } }
 This script PRINTS the resolved JSON block (real paths) at the end -- copy it in.
 Reference template: scripts\comsol_mcp\mcp_servers.template.json

 Provenance / version-pin SOURCE OF TRUTH: scripts\comsol_mcp\UPSTREAM.lock.json
 It records the upstream URLs, the pinned commit, venv Python, patch state, resolved command
 path, and why this repo deliberately lives OUTSIDE the project tree instead of being a git
 submodule (upstream tracks ~700MB of binaries and has no .gitignore; its 1GB venv is not
 relocatable on Windows). Keep this script's -Commit default in sync with that file;
 `-StatusOnly` verifies the agreement.

--------------------------------------------------------------------------------
 Phase 0 Go/No-Go checklist (drive via CallMcpTool after registering)
--------------------------------------------------------------------------------
   [ ] comsol: comsol_start (or comsol_connect 2036 in shared mode) OK;
       comsol_status shows connected
   [ ] Load a gain_ep .mph -> model_inspect / parameter list correct
   [ ] Set a parameter -> study_solve (async/progress/wait) -> results evaluate
       round-trips
   [ ] Pressure Acoustics is drivable; confirm BPF / EFC far-field
       (acpr.efc1.pext) can be built via the generic physics_add /
       configure_boundary tools (key differentiator check)
   [ ] pdf_search returns sensible hits (e.g. "background pressure field",
       "pressure acoustics")
   [ ] differentiated capabilities via the CLI (not an MCP): pysci-simulation
       export image --mph M --plotgroup PG --out O.png yields a publication-grade
       PNG (Read to preview); `server start/stop` for a persistent session
   [ ] Memory/license etiquette: disconnect/stop when idle; NEVER run alongside
       the COMSOL GUI
   ==> If the community MCP cannot adequately drive our pressure-acoustics model:
       No-Go -- fall back to keeping more of the in-house layer and re-plan.
================================================================================
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string] $RepoDir       = 'D:\XXXIIIGGG\projects\pySci\COMSOL_Multiphysics_MCP',
    # github.com is unreachable from this machine (port 443 timeout / connection reset);
    # default to the gitclone.com mirror. Override with the canonical URL if you use a proxy/VPN:
    #   -RepoUrl https://github.com/wjc9011/COMSOL_Multiphysics_MCP.git
    [string] $RepoUrl       = 'https://gitclone.com/github.com/wjc9011/COMSOL_Multiphysics_MCP',
    # PINNED upstream commit -- must match scripts\comsol_mcp\UPSTREAM.lock.json (that file is the
    # provenance source of truth; this script is the only installer).
    # Re-pinned 2026-10-01: the previous pin 0f6b2c58 is NOT reachable in the shallow/grafted mirror
    # clone, so the guarded checkout below silently fell through to 'staying on HEAD' and the manifest
    # drifted from what actually runs. Now pinned to the VERIFIED on-disk tip => manifest == reality.
    # To recover the original pin: enable a proxy, `git -C $RepoDir fetch --unshallow`, then set both
    # this value and UPSTREAM.lock.json back to 0f6b2c588a08da5ac66915e4f1fce7c07966763e.
    [string] $Commit        = '99172f8f43c6753c2442c406cd5c6055ea8c5bef',
    [string] $PythonVersion = '3.12',
    [string] $PdfDir        = 'D:\XiGPrograms\comsol\6.4\base\doc\pdf',
    [string] $DbDir         = '',           # empty = repo default (matches the server's read path; safest)
    [int]    $RagLimit      = 0,            # >0 = process only the first N PDFs (quick check)
    [bool]   $CpuTorch      = $true,        # no NVIDIA GPU here: pre-install CPU-only torch (smaller, no CUDA dlls)
    # PyPI index for general deps. Default = Tsinghua mirror (fast/reliable in CN; pypi.org is reachable but slow).
    # Outside China override with:  -IndexUrl https://pypi.org/simple
    [string] $IndexUrl      = 'https://pypi.tuna.tsinghua.edu.cn/simple',
    # Primary index for torch CPU wheels. Kept PRIMARY in BOTH install steps so the resolver always
    # prefers the +cpu build; otherwise the `-e .` step re-resolves torch to the CUDA-bundled
    # Windows default (~2.5 GB) via sentence-transformers and overwrites the CPU wheel.
    [string] $TorchIndexUrl = 'https://download.pytorch.org/whl/cpu',
    [switch] $SkipInstall,
    [switch] $SkipRag,
    [switch] $Rebuild,
    [switch] $NoHfMirror,                   # pass --no-mirror to the build script (outside China / no hf-mirror)
    [switch] $ApplySharedSessionPatch,      # see CRITICAL FINDING path B
    [switch] $Shallow,                      # clone with --depth 1 (this repo is huge; shallow is plenty for the spike)
    [switch] $SkipFetch,                    # never run `git fetch` (avoids stalls on a slow mirror when already cloned)
    [switch] $StatusOnly,
    [switch] $DryRun
)

# NOTE: 'Continue', not 'Stop'. git/uv are native commands that write progress and benign
# messages to stderr; under Windows PowerShell 5.1, ErrorActionPreference='Stop' + redirected
# stderr (2>$null, or a caller's *> log) converts that into a terminating NativeCommandError and
# aborts the script. We check $LASTEXITCODE and throw explicitly where a failure must stop us.
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg)  { Write-Host "    $msg" }
function Write-Warn2($msg) { Write-Host "    [!] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host "    [OK] $msg" -ForegroundColor Green }

$venvDir   = Join-Path $RepoDir '.venv'
$venvPy    = Join-Path $venvDir 'Scripts\python.exe'
$comsolExe = Join-Path $venvDir 'Scripts\comsol-mcp.exe'
$buildPy   = Join-Path $RepoDir 'scripts\build_knowledge_base.py'
$serverPy  = Join-Path $RepoDir 'src\server.py'

Write-Host '====================================================================' -ForegroundColor DarkCyan
Write-Host ' Phase 0 :: community COMSOL MCP setup' -ForegroundColor DarkCyan
Write-Host '====================================================================' -ForegroundColor DarkCyan
if ($DryRun) { Write-Warn2 'DryRun mode: prints intended actions only, no side effects.' }
Write-Info "RepoDir   = $RepoDir"
Write-Info "Commit    = $Commit"
Write-Info "Python    = $PythonVersion"
Write-Info "PdfDir    = $PdfDir"

# ---------------------------------------------------------------------------
# -StatusOnly: report existing KB status and exit
# ---------------------------------------------------------------------------
if ($StatusOnly) {
    Write-Step 'Upstream pin vs. lock manifest (read-only)'
    $lockPath = Join-Path $PSScriptRoot 'UPSTREAM.lock.json'
    Write-Info "RepoDir            = $RepoDir"
    Write-Info "script -Commit     = $Commit"
    $lockPin = $null
    if (Test-Path $lockPath) {
        # ReadAllText with an EXPLICIT UTF-8 encoding. PS 5.1's Get-Content decodes a BOM-less UTF-8
        # file as the ANSI code page (GBK on zh-CN), which mangles the lock manifest's CJK prose and
        # swallows the adjacent quotes -> ConvertFrom-Json then dies with a bogus array error.
        # Same trap this script's header documents for .ps1 files; the fix belongs in the reader.
        try {
            $lockJson = [System.IO.File]::ReadAllText($lockPath, [System.Text.Encoding]::UTF8)
            $lockPin = ($lockJson | ConvertFrom-Json).upstream.pinned_commit
        }
        catch { Write-Warn2 "Cannot parse $lockPath -- $($_.Exception.Message)" }
        if ($lockPin) { Write-Info "lock pinned_commit = $lockPin" }
        else { Write-Warn2 'lock manifest parsed but upstream.pinned_commit is empty.' }
    } else { Write-Warn2 "Lock manifest missing: $lockPath" }
    if (Test-Path (Join-Path $RepoDir '.git')) {
        $headFull = (& git -C $RepoDir rev-parse HEAD 2>$null)
        Write-Info "repo HEAD          = $headFull"
        if ($lockPin -and $headFull -eq $lockPin) { Write-Ok 'HEAD agrees with the lock manifest.' }
        elseif ($lockPin) { Write-Warn2 'VERSION DRIFT: HEAD != lock pinned_commit -- re-pin one of them.' }
        if ($headFull -ne $Commit) { Write-Warn2 "VERSION DRIFT: HEAD != script -Commit ($Commit)." }
        if (Test-Path (Join-Path $RepoDir '.git\shallow')) {
            Write-Warn2 'Clone is shallow/grafted: older pins are unreachable without fetch --unshallow.'
        }
    } else { Write-Warn2 "Repo not cloned yet: $RepoDir" }
    if (Test-Path $comsolExe) { Write-Ok "console script: $comsolExe" }
    else { Write-Warn2 "console script missing: $comsolExe (run a full setup)" }

    Write-Step 'RAG knowledge base status'
    if (-not (Test-Path $venvPy)) { throw "venv not created yet: $venvPy (run a full setup first)" }
    $sArgs = @($buildPy, '--status', '--pdf-dir', $PdfDir)
    if ($DbDir) { $sArgs += @('--db-dir', $DbDir) }
    if ($DryRun) { Write-Info "[DRY] & `"$venvPy`" $($sArgs -join ' ')" }
    else { & $venvPy @sArgs }
    return
}

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
Write-Step '1/8 Preflight (git / uv)'
$git = (Get-Command git -ErrorAction SilentlyContinue).Source
$uv  = (Get-Command uv  -ErrorAction SilentlyContinue).Source
if (-not $git) { throw 'git not found. Install Git for Windows first.' }
if (-not $uv)  { throw 'uv not found. Install uv first (https://docs.astral.sh/uv/).' }
Write-Ok "git = $git"
Write-Ok "uv  = $uv"
if (-not (Test-Path $PdfDir)) { Write-Warn2 "PDF dir missing: $PdfDir (RAG build will fail; use -PdfDir)" }
else { Write-Ok "PDF dir exists: $PdfDir" }

# ---------------------------------------------------------------------------
# 2. Clone + pin commit
# ---------------------------------------------------------------------------
Write-Step '2/8 Clone community repo and pin to commit'
if (-not (Test-Path (Join-Path $RepoDir '.git'))) {
    Write-Info "clone $RepoUrl -> $RepoDir"
    $cloneArgs = @('clone', '--quiet')
    if ($Shallow) { $cloneArgs += @('--depth', '1') }
    $cloneArgs += @($RepoUrl, $RepoDir)
    if ($DryRun) { Write-Info "[DRY] git $($cloneArgs -join ' ')" }
    else {
        & git @cloneArgs 2>$null
        if (-not (Test-Path (Join-Path $RepoDir '.git'))) { throw "Clone failed; repo not created. Check network/mirror: $RepoUrl" }
        Write-Ok 'clone complete'
    }
} else {
    Write-Info 'Repo already present; skipping clone (use -SkipFetch is implicit here; no network touch).'
}

# Guarded pin checkout -- never hangs, never throws on a missing pin.
# `rev-parse --verify --quiet` exits non-zero with NO stderr when the object is absent, so the
# PS 5.1 NativeCommandError trap is avoided entirely (unlike `cat-file`).
if (-not $DryRun) {
    $isShallow = Test-Path (Join-Path $RepoDir '.git\shallow')
    $havePin = $false
    & git -C $RepoDir rev-parse --verify --quiet "$Commit^{commit}" 1>$null 2>$null
    if ($LASTEXITCODE -eq 0) { $havePin = $true }
    if (-not $havePin -and -not $isShallow -and -not $SkipFetch) {
        Write-Info 'Pin absent in a FULL clone; bounded fetch (low-speed guarded, aborts ~20s after a stall)...'
        & git -C $RepoDir -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 fetch --quiet --all 2>$null
        & git -C $RepoDir rev-parse --verify --quiet "$Commit^{commit}" 1>$null 2>$null
        if ($LASTEXITCODE -eq 0) { $havePin = $true }
    }
    if ($havePin) {
        & git -C $RepoDir checkout --quiet $Commit 2>$null
        $head = (& git -C $RepoDir rev-parse HEAD 2>$null)
        Write-Ok "Pinned HEAD = $head"
    } else {
        $head = (& git -C $RepoDir rev-parse --short HEAD 2>$null)
        if ($isShallow) { Write-Warn2 "Shallow clone: pin $Commit unreachable; staying on HEAD = $head." }
        else            { Write-Warn2 "Pin $Commit NOT found; staying on HEAD = $head." }
        Write-Warn2 'VERSION DRIFT: the server you are about to use is NOT the pinned commit.'
        Write-Warn2 'Do not trust its results until fixed. Either (a) unshallow behind a proxy'
        Write-Warn2 '(`git -C <RepoDir> fetch --unshallow`) and re-run, or (b) re-pin to the verified'
        Write-Warn2 'HEAD in BOTH this script and scripts\comsol_mcp\UPSTREAM.lock.json.'
        Write-Warn2 'For the exact pin, re-clone via proxy: -RepoUrl https://github.com/wjc9011/COMSOL_Multiphysics_MCP.git'
    }
}

# ---------------------------------------------------------------------------
# 3. Isolated venv @ Python 3.12
# ---------------------------------------------------------------------------
Write-Step "3/8 Create isolated uv venv (Python $PythonVersion)"
if (Test-Path $venvPy) {
    Write-Ok "venv exists: $venvDir (skip)"
} elseif ($DryRun) {
    Write-Info "[DRY] uv venv --python $PythonVersion `"$venvDir`""
} else {
    & uv venv --python $PythonVersion $venvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) { throw "uv venv failed (exit $LASTEXITCODE); is Python $PythonVersion available to uv?" }
    Write-Ok "venv created: $venvDir"
}

# ---------------------------------------------------------------------------
# 4. Install deps (CPU torch + editable install), timed
# ---------------------------------------------------------------------------
if (-not $SkipInstall) {
    Write-Step '4/8 Install deps (CPU torch + `uv pip install -e .`)'
    # Same combined index for BOTH steps: pytorch-cpu primary (torch -> +cpu), Tsinghua extra (everything else).
    $idx = @('--index-url', $TorchIndexUrl, '--extra-index-url', $IndexUrl)
    if ($DryRun) {
        if ($CpuTorch) { Write-Info "[DRY] uv pip install --python `"$venvPy`" $($idx -join ' ') torch" }
        Write-Info "[DRY] uv pip install --python `"$venvPy`" $($idx -join ' ') -e `"$RepoDir`""
    } else {
        $swInstall = [System.Diagnostics.Stopwatch]::StartNew()
        if ($CpuTorch) {
            Write-Info 'Pre-installing CPU-only torch (smaller on a GPU-less machine; skips CUDA deps)...'
            & uv pip install --python $venvPy @idx torch
            if ($LASTEXITCODE -ne 0) { throw "CPU torch install failed (exit $LASTEXITCODE)." }
        }
        Write-Info 'Installing community repo (editable) + deps (chromadb/sentence-transformers/pymupdf/mph/mcp)...'
        & uv pip install --python $venvPy @idx -e $RepoDir
        if ($LASTEXITCODE -ne 0) { throw "Editable install failed (exit $LASTEXITCODE)." }
        $swInstall.Stop()
        Write-Ok ("Install done in {0:n1}s" -f $swInstall.Elapsed.TotalSeconds)
    }
} else {
    Write-Step '4/8 Install deps (-SkipInstall, skipped)'
}

# ---------------------------------------------------------------------------
# 5. Optional shared-session patch (CRITICAL FINDING path B)
# ---------------------------------------------------------------------------
Write-Step '5/8 Shared-session patch (optional)'
if (-not $ApplySharedSessionPatch) {
    Write-Info 'Not enabled (default). For single-license two-MCP sharing, re-run with -ApplySharedSessionPatch.'
} elseif ($DryRun) {
    Write-Info '[DRY] If src\server.py is unpatched, change pre-start to: connect(port) when COMSOL_MCP_CONNECT_PORT is set, else start()'
} else {
    $raw = [System.IO.File]::ReadAllText($serverPy)
    $marker = '# [pysci-shared-session-patch]'
    $old = '        logger.info(f"COMSOL pre-start: {session_manager.start()}")'
    if ($raw.Contains($marker)) {
        Write-Ok 'Patch already present; skip.'
    } elseif ($raw.Contains($old)) {
        $new = @"
        $marker
        _cport = os.environ.get("COMSOL_MCP_CONNECT_PORT")
        if _cport:
            _chost = os.environ.get("COMSOL_MCP_CONNECT_HOST", "localhost")
            logger.info(f"COMSOL pre-connect: {session_manager.connect(port=int(_cport), host=_chost)}")
        else:
            logger.info(f"COMSOL pre-start: {session_manager.start()}")
"@
        $raw = $raw.Replace($old, $new)
        $enc = New-Object System.Text.UTF8Encoding($false)   # BOM-less
        [System.IO.File]::WriteAllText($serverPy, $raw, $enc)
        Write-Ok 'Shared-session patch applied (revert with: git -C $RepoDir checkout -- src/server.py).'
    } else {
        Write-Warn2 'Expected pre-start line not found; patch NOT applied (upstream may differ; check src/server.py manually).'
    }
}

# ---------------------------------------------------------------------------
# 6. Lightweight verify (no JVM, no license)
# ---------------------------------------------------------------------------
Write-Step '6/8 Lightweight verify (console script + src.server locatable)'
if ($DryRun) {
    Write-Info "[DRY] Test-Path `"$comsolExe`""
    Write-Info "[DRY] & `"$venvPy`" -c `"import importlib.util as u; print('SPEC_OK', bool(u.find_spec('src.server')))`"  (cwd=$RepoDir)"
} else {
    if (Test-Path $comsolExe) { Write-Ok "comsol-mcp console script: $comsolExe" }
    else { Write-Warn2 "Not found: $comsolExe (install may be incomplete)" }
    Push-Location $RepoDir
    try {
        $spec = & $venvPy -c "import importlib.util as u; print('SPEC_OK', bool(u.find_spec('src.server')))"
        Write-Info $spec
        if ($spec -match 'SPEC_OK True') { Write-Ok 'src.server locatable (import chain intact).' }
        else { Write-Warn2 'src.server not locatable; check the install.' }
    } finally { Pop-Location }
}

# ---------------------------------------------------------------------------
# 7. Build RAG knowledge base (CPU), timed + peak-RAM sampled
# ---------------------------------------------------------------------------
if (-not $SkipRag) {
    Write-Step '7/8 Build RAG knowledge base (all-MiniLM-L6-v2, CPU)'
    $ragArgs = @($buildPy, '--pdf-dir', $PdfDir)
    if ($DbDir)          { $ragArgs += @('--db-dir', $DbDir) }
    if ($Rebuild)        { $ragArgs += '--rebuild' }
    if ($RagLimit -gt 0) { $ragArgs += @('--limit', "$RagLimit") }
    if ($NoHfMirror)     { $ragArgs += '--no-mirror' }

    if ($DryRun) {
        Write-Info "[DRY] & `"$venvPy`" $($ragArgs -join ' ')"
    } elseif (-not (Test-Path $venvPy)) {
        Write-Warn2 'venv missing; skip RAG (complete the install steps first).'
    } else {
        Write-Info "cmd: $venvPy $($ragArgs -join ' ')"
        # The venv python.exe is a THIN LAUNCHER that spawns the real interpreter as a CHILD, so
        # sampling the launched process under-reports RAM badly (we measured a 4 MB stub while the
        # actual worker used ~830 MB). Run the build in the FOREGROUND (reliable $LASTEXITCODE, and
        # its output is captured by the caller's log/tee) and sample the peak python working set
        # from a background job.
        $peakFile = [System.IO.Path]::GetTempFileName()
        Set-Content -Path $peakFile -Value '0' -Encoding Ascii
        $sampler = Start-Job -ScriptBlock {
            param($pf)
            $peak = 0.0
            while ($true) {
                $m = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Maximum).Maximum
                if ($m) { $mb = [math]::Round($m / 1MB, 1); if ($mb -gt $peak) { $peak = $mb; Set-Content -Path $pf -Value $peak -Encoding Ascii } }
                Start-Sleep -Milliseconds 1500
            }
        } -ArgumentList $peakFile
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $venvPy @ragArgs
        $code = $LASTEXITCODE
        $sw.Stop()
        Stop-Job   $sampler -ErrorAction SilentlyContinue
        Remove-Job $sampler -Force -ErrorAction SilentlyContinue
        $peakMB = 0.0
        try { $peakMB = [double]((Get-Content $peakFile -Raw).Trim()) } catch { $peakMB = 0.0 }
        Remove-Item $peakFile -Force -ErrorAction SilentlyContinue
        Write-Host ''
        if ($code -eq 0) { Write-Ok ("RAG build OK: {0:n1}s, peak python working set ~{1:n1} MB" -f $sw.Elapsed.TotalSeconds, $peakMB) }
        else { Write-Warn2 ("RAG build exit code {0} ({1:n1}s, peak ~{2:n1} MB) -- see log above." -f $code, $sw.Elapsed.TotalSeconds, $peakMB) }
    }
} else {
    Write-Step '7/8 RAG build (-SkipRag, skipped)'
}

# ---------------------------------------------------------------------------
# 8. Print registration JSON + Go/No-Go reminder
# ---------------------------------------------------------------------------
Write-Step '8/8 mcpServers registration block (copy into C:\Users\antar\.qoder-cn\shared_client\mcp.json)'
$comsolEnv = [ordered]@{ COMSOL_MCP_VERSION = '6.4' }
if ($ApplySharedSessionPatch) { $comsolEnv['COMSOL_MCP_CONNECT_PORT'] = '2036'; $comsolEnv['COMSOL_MCP_CONNECT_HOST'] = 'localhost' }
$reg = [ordered]@{
    mcpServers = [ordered]@{
        'comsol' = [ordered]@{ command = $comsolExe; args = @(); env = $comsolEnv }
    }
}
$json = $reg | ConvertTo-Json -Depth 8
Write-Host ''
Write-Host '----8<---- paste this entry into mcpServers (alongside arxiv etc.) ----8<----' -ForegroundColor DarkGray
Write-Host $json
Write-Host '----8<------------------------------------------------------------------------8<----' -ForegroundColor DarkGray
if ($ApplySharedSessionPatch) {
    Write-Warn2 'Shared-session env included: run `pysci-simulation server start` first (comsolmphserver on 2036) so the community MCP can attach.'
} else {
    Write-Warn2 'No shared-session patch: the community MCP over stdio grabs the license IMMEDIATELY on launch (see CRITICAL FINDING). Do NOT run the community MCP and the simulation CLI (or the COMSOL GUI) at the same time -- single license.'
}
Write-Host ''
Write-Ok 'Setup script finished. Next: register the comsol MCP in Qoder, then run the Go/No-Go checklist live.'
