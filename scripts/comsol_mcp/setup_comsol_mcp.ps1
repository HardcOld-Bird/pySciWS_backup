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

   # NOTE: the KB is now FULLY built (35923 chunks / 52 modules). Step 7 rebuilds from
   # scratch and takes ~27 min, so on any re-run pass -SkipRag unless you truly want it.

   # Read-only: verify the pin against UPSTREAM.lock.json, then report KB status:
   ... setup_comsol_mcp.ps1 -StatusOnly

   # Enable single-license shared session (applies patch; then `pysci-simulation
   # server start` launches comsolmphserver and the community MCP attaches to it):
   ... setup_comsol_mcp.ps1 -ApplySharedSessionPatch

--------------------------------------------------------------------------------
 CRITICAL FINDING -- community MCP "eager start" vs. the single license
--------------------------------------------------------------------------------
 VERSION-SCOPED. This describes the PINNED commit 0f6b2c58 (2026-09-18) and
 later. Behaviour genuinely differs by upstream version, so before trusting any
 claim below, read main() in src/server.py OF THE PINNED COMMIT:
   * 0f6b2c58+ (contains upstream 7a4b704): main() honours COMSOL_MCP_TRANSPORT
     and, under the default "stdio", calls session_manager.start() BEFORE
     mcp.run() => COMSOL is started EAGERLY (see the consequence below).
   * 99172f8f and earlier: no transport handling and no pre-start, so COMSOL
     started LAZILY on the first tool call -- but comsol_start was a plain sync
     `def`, and JPype's startJVM hangs forever while another thread is blocked
     reading stdin (fd 0), which the stdio transport always spawns. Net effect:
     no idle license, but a stdio solve HANGS. That fd-0 interlock is the exact
     reason the retired `pysci-comsol-extras` MCP froze, and it is why this pin
     was upgraded on 2026-10-01. (mph's INFO logging during JVM boot triggers
     the same deadlock; upstream now pins that logger to WARNING.)

 Consequence on the pinned version: whenever Qoder launches the `comsol` MCP
 over stdio, it IMMEDIATELY starts a standalone COMSOL that grabs the single
 license inside that MCP's own process -- even if you are not running a sim yet.
 Because that JVM is in-process, a separate process (e.g. the pysci `simulation`
 CLI) CANNOT attach to it. Two COMSOL clients at once would need 2 licenses; we
 have 1. So: disable/stop the `comsol` MCP in Qoder before using the CLI, and
 never run either alongside the interactive GUI.

 OBSERVED SIDE EFFECT of the eager start (2026-10-01, live): pre-start delays the
 MCP handshake by the whole ~30s COMSOL boot, and Qoder rewrote
 shared_client\mcp.json mid-startup (its extension\local\mcp.json mirror carries
 a userConfigMD5 change detector). The reload spawned a SECOND comsol-mcp and
 never reaped the first -- two processes, BOTH with jvm.dll + _jpype.pyd loaded
 (54 threads, ~320MB each), i.e. two JVMs contending for one license. Note also
 that comsol_status reports "standalone": true, so the JVM is IN-PROCESS: there
 is NO comsolmphserver.exe to look for, and "I see no COMSOL process" does NOT
 mean "the license is free". `-StatusOnly` now checks exactly this (process
 count + loaded jvm.dll). Clean fix for duplicates: fully QUIT Qoder, a window
 reload is not enough.

 Three resolutions:
   A) Sequential use (simplest, no patch -- CURRENT DEFAULT): only one driver
      holds a COMSOL session at a time; stop the MCP (or `server stop`) before
      switching.
   B) Shared session (single license, community MCP + CLI on one server):
      - Run with -ApplySharedSessionPatch to add a 5-line patch to server.py:
        when env COMSOL_MCP_CONNECT_PORT is set, pre-start does connect(port)
        instead of a standalone start(). The target line EXISTS on 0f6b2c58
        (it did NOT on 99172f8f, where the patch silently declined to apply).
      - `pysci-simulation server start` launches the ONE persistent
        comsolmphserver (records pid/port in runs/comsol_server.json).
      - Register the community MCP with env COMSOL_MCP_CONNECT_PORT=2036 so it
        attaches to that same server; run CLI commands with --connect-port 2036.
      - If Qoder launches the community MCP before the persistent server is up,
        the pre-connect fails (non-fatal); just call comsol_connect(port=2036)
        after the CLI has started the server.
   C) HTTP transport (lazy COMSOL start, NO idle license -- best fit for a
      single license): added by upstream 7a4b704. Set COMSOL_MCP_TRANSPORT=sse
      (or streamable-http -- NOT "http"; FastMCP only accepts stdio / sse /
      streamable-http) plus COMSOL_MCP_HOST and COMSOL_MCP_PORT (default
      127.0.0.1:8765), then register the MCP BY URL instead of by command. With
      no thread reading stdin, COMSOL starts lazily on the first tool call, no
      license is held while idle, and the handshake is instant -- so the
      duplicate-spawn problem above disappears too.
      Qoder DOES support URL registration. Verified 2026-10-01 against the
      official docs: the IDE "Model Context Protocol" page lists the transport
      as STDIO or SSE with "URL (for SSE or Streamable HTTP)" and states that
      for Streamable HTTP you configure the URL the same way as SSE and Qoder
      auto-detects; the CLI MCP reference lists type = sse / http /
      streamable-http / ws with fields url + headers. mcp.json shape:
        { "mcpServers": { "comsol":
            { "type": "sse", "url": "http://127.0.0.1:8765/sse" } } }
      (FastMCP serves /sse for transport=sse and /mcp for streamable-http --
      keep the two in step.) NOT YET RUN HERE, and it costs you a process:
      nothing spawns the server any more, so you must start it yourself and
      keep it up, and the first tool call pays the ~30s JVM boot (raise the
      server's Request Timeout in Qoder accordingly).

 Recommendation: A today (verified end-to-end 2026-10-01: comsol_status ->
 connected, 6.4, standalone, 8 cores). C is now unblocked and is strictly better
 under one license -- switch when the "start the server yourself" chore is worth
 it, or as soon as the duplicate-JVM problem bites again. Choose B only if you
 need the MCP and the CLI warm at the same time.

--------------------------------------------------------------------------------
 Registration (add the community comsol MCP in Qoder)
--------------------------------------------------------------------------------
 User-level config file: C:\Users\antar\.qoder-cn\shared_client\mcp.json   <-- EDIT THIS ONE
 Qoder mirrors it to ...\shared_client\extension\local\mcp.json and adds a
 userConfigMD5 key there; that mirror is DERIVED, so do not hand-edit it. Only
 one project dir exists under shared_client\projects, so a duplicated `comsol`
 entry is never the cause of two live instances (see OBSERVED SIDE EFFECT above).
 Shape (stdio): { "mcpServers": { "<name>": { "command":..., "args":[...], "env":{...} } } }
 Shape (URL)  : { "mcpServers": { "<name>": { "type":"sse", "url":"http://host:port/sse" } } }
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
    # github.com is reachable once a global proxy is on (verified 2026-10-01: ls-remote in 0.9s, and
    # the existing clone's origin was switched to canonical). The default stays the gitclone.com
    # mirror only because it works with NO proxy -- but it LAGS canonical, and that lag is exactly
    # what stranded the old pin at 99172f8f for weeks (see UPSTREAM.lock.json -> pin_history).
    # With a proxy available, prefer canonical:
    #   -RepoUrl https://github.com/wjc9011/COMSOL_Multiphysics_MCP.git
    [string] $RepoUrl       = 'https://gitclone.com/github.com/wjc9011/COMSOL_Multiphysics_MCP',
    # PINNED upstream commit -- must match scripts\comsol_mcp\UPSTREAM.lock.json (that file is the
    # provenance source of truth; this script is the only installer). `-StatusOnly` checks this value,
    # the lock manifest, and the clone's real HEAD all agree.
    # 2026-10-01: restored to the ORIGINAL intended pin, now reachable. It had been stranded because
    # the mirror lags canonical github.com, so `git rev-parse --verify` failed and the guarded checkout
    # below silently stayed on HEAD (99172f8f) while the manifest claimed otherwise. Behind a proxy a
    # TARGETED fetch makes any SHA reachable in seconds -- no --unshallow needed, because the big blobs
    # (pdf/, comsol_models/) are unchanged across versions and already local (measured: 1.6s):
    #   git -C $RepoDir fetch --depth 1 https://github.com/wjc9011/COMSOL_Multiphysics_MCP.git $Commit
    # What 0f6b2c58 buys over 99172f8f: the JPype stdin/fd-0 deadlock fix plus native
    # COMSOL_MCP_TRANSPORT (7a4b704), four COMSOL 6.x API fixes (this box runs 6.4), COMSOL_MCP_VERSION
    # finally honoured (c0ec1f8), .gitignore (a9de20e), 103 tools (electrochemistry, 8529877), and
    # +1254 lines of upstream tests. pyproject.toml is UNCHANGED => no dependency reinstall, and
    # pdf/ + comsol_models/ + knowledge_base/ are UNCHANGED => the built RAG index survives.
    [string] $Commit        = '0f6b2c588a08da5ac66915e4f1fce7c07966763e',
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
    [switch] $Rebuild,                      # almost never what you want -- read the step 7 warning first
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
            # Informational, not an error: a targeted `git fetch --depth 1 <url> <sha>` keeps the clone
            # shallow while making the pin reachable -- that is how 0f6b2c58 was obtained (1.6s).
            $grafts = @(Get-Content (Join-Path $RepoDir '.git\shallow') -ErrorAction SilentlyContinue).Count
            Write-Warn2 "Clone is shallow ($grafts graft(s)); any other pin needs a targeted fetch first."
        }
    } else { Write-Warn2 "Repo not cloned yet: $RepoDir" }
    if (Test-Path $comsolExe) { Write-Ok "console script: $comsolExe" }
    else { Write-Warn2 "console script missing: $comsolExe (run a full setup)" }

    Write-Step 'RAG knowledge base status (true document count)'
    if (-not (Test-Path $venvPy)) { throw "venv not created yet: $venvPy (run a full setup first)" }
    # Deliberately NOT upstream's `build_knowledge_base.py --status`: that path calls get_stats()
    # without initialize(), and get_stats() returns {initialized: False, count: 0} whenever
    # _collection is None -- so it prints 'Documents: 0' even for a populated index (upstream bug,
    # still present at pin 0f6b2c58; measured 3669 chunks while --status claimed 0). Our own probe
    # reads Chroma directly: no embedding model, no JVM, no license, ~2s. See probe_kb.py.
    $probePy = Join-Path $PSScriptRoot 'probe_kb.py'
    $sArgs = @($probePy, '--repo-dir', $RepoDir, '--pdf-dir', $PdfDir)
    if ($DbDir) { $sArgs += @('--db-dir', $DbDir) }
    if ($DryRun) { Write-Info "[DRY] & `"$venvPy`" $($sArgs -join ' ')" }
    else {
        & $venvPy @sArgs
        if ($LASTEXITCODE -ne 0) { Write-Warn2 "RAG probe exit code $LASTEXITCODE -- read the messages above." }
    }

    Write-Step 'License occupancy (is a COMSOL JVM holding the single license right now?)'
    # The pinned server starts COMSOL EAGERLY under stdio and MPh reports "standalone": true, so the JVM
    # lives IN-PROCESS inside comsol-mcp's python.exe. There is NO comsolmphserver.exe to look for, hence
    # "I see no COMSOL process" does NOT mean "the license is free". Test for a loaded jvm.dll instead.
    # Also: a Windows venv makes ONE instance show up as TWO python.exe -- the Scripts\python.exe launcher
    # shim (a few MB, never loads a JVM) and the base interpreter it re-execs (the real server). So count
    # instances by "has a JVM", never by "command line mentions comsol-mcp" (that double-counts).
    $mcpPy = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine -match 'comsol-mcp' })
    if ($mcpPy.Count -eq 0) {
        Write-Ok 'no comsol-mcp python process running -- the license is free for the CLI / GUI.'
    } else {
        $holders = @()
        foreach ($mp in $mcpPy) {
            $started = ''
            $hasJvm  = $false
            try {
                $proc    = Get-Process -Id $mp.ProcessId -ErrorAction Stop
                $started = $proc.StartTime.ToString('yyyy-MM-dd HH:mm:ss')
                $hasJvm  = [bool]@($proc.Modules | Where-Object { $_.ModuleName -match 'jvm|jpype' }).Count
            } catch { Write-Warn2 "cannot inspect PID $($mp.ProcessId): $($_.Exception.Message)" }
            if ($hasJvm) {
                $holders += $mp.ProcessId
                Write-Warn2 "PID $($mp.ProcessId) (started $started) HAS a JVM => it is HOLDING the license."
            } else {
                Write-Info "PID $($mp.ProcessId) (started $started) no JVM => venv launcher shim, or COMSOL not started."
            }
        }
        Write-Info "COMSOL session(s) alive: $($holders.Count) (of a single license)"
        if ($holders.Count -gt 1) {
            Write-Warn2 "$($holders.Count) separate JVMs are contending for ONE license. Qoder orphans an instance"
            Write-Warn2 'when it rewrites mcp.json mid-startup (see OBSERVED SIDE EFFECT in this header).'
            Write-Warn2 'Fully QUIT Qoder to clear them; a window reload is not enough.'
        } elseif ($holders.Count -eq 0) {
            Write-Ok 'server process(es) up but no COMSOL session => the license is free right now.'
        }
    }
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
    Write-Info "[DRY] & `"$venvPy`" -c `"import src.server; print('IMPORT_OK')`"  (cwd=$RepoDir)"
} else {
    if (Test-Path $comsolExe) { Write-Ok "comsol-mcp console script: $comsolExe" }
    else { Write-Warn2 "Not found: $comsolExe (install may be incomplete)" }
    Push-Location $RepoDir
    try {
        # A REAL import, not find_spec: find_spec only resolves a path, so it cannot catch a missing
        # runtime dependency (e.g. `anyio`, which upstream 7a4b704 introduced). Importing src.server
        # builds the FastMCP app and pulls in every tool module -- but it does NOT start the JVM
        # (session_manager.start() runs only inside main()), so this stays license-free and takes ~2s.
        # A half-installed venv or a badly applied patch fails loudly right here.
        $imp = (& $venvPy -c "import src.server; print('IMPORT_OK')" 2>&1 | Out-String).Trim()
        if ($imp -match 'IMPORT_OK') { Write-Ok 'src.server imports cleanly (full tool chain + deps intact).' }
        else { Write-Warn2 'src.server FAILED to import -- tail below:'; Write-Info $imp }
        # Tool count doubles as a version fingerprint: 93 == pre-0f6b2c58, 103 == this pin.
        # Qoder caches its own count in SERVER_METADATA.json, so a mismatch there means the MCP
        # process still has the OLD code loaded and must be restarted in Qoder.
        $tools = (Get-ChildItem (Join-Path $RepoDir 'src') -Recurse -File -Filter *.py |
                  Select-String -Pattern '@mcp\.tool' | Measure-Object).Count
        Write-Info "registered tools (@mcp.tool count) = $tools"
    } finally { Pop-Location }
}

# ---------------------------------------------------------------------------
# 7. Build RAG knowledge base (CPU), timed + peak-RAM sampled
# ---------------------------------------------------------------------------
if (-not $SkipRag) {
    Write-Step '7/8 Build RAG knowledge base (all-MiniLM-L6-v2, CPU)'
    # MEASURED 2026-10-01 on the full corpus (108 PDFs / 52 modules -> 35923 chunks): ~27 min wall
    # clock, and PDF extraction is only ~31s of that -- the rest is 360 embedding batches. Three traps:
    #  * Do NOT pass -Rebuild. rebuild_from_pdfs() clears the collection ITSELF, but only AFTER it has
    #    extracted every PDF into memory; -Rebuild clears at t=0 instead, stretching the "no usable
    #    index" window from seconds to the whole extraction phase and turning any extraction failure
    #    into total loss of the index you already paid for.
    #  * Launch it from a REAL background terminal when an agent/shell drives it. A child started with
    #    Start-Process stays inside the terminal's job object and gets reaped when that shell exits, so
    #    the build dies silently -- no traceback, no Windows Error Reporting entry (observed live).
    #  * Upstream never calls logging.basicConfig(), so the root logger stays at WARNING and every
    #    "Processing: <file>.pdf" INFO line is dropped: a 27-minute build with no visible progress.
    #    Wrap it (runpy.run_path + logging.basicConfig(level=INFO)) if you want a heartbeat.
    Write-Warn2 'A full build CLEARS the existing index and takes ~27 min. Pass -SkipRag to keep it.'
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
