<#
================================================================================
 Setup script -- community zotero-mcp (54yyyu/zotero-mcp, PyPI: zotero-mcp-server)
================================================================================
 One-shot preparation tool for Phase 3 (library layer) of the
 "literature_research community-infrastructure replacement" plan. It installs and
 registers the COMMUNITY zotero-mcp -- the ZOTERO LIBRARY PRIMARY layer that
 replaces the in-house zotero_bridge.py (hand-written requests + pyzotero).

 Architecture (mirrors comsol-simulation / paper-search-mcp): the community MCP is
 the primary library path (Agent reads/writes Zotero directly: search, read full
 text + PDF page images, add items by DOI/URL/ISBN/BibTeX/file, merge duplicates,
 Scite retraction alerts). The in-house `pysci-research` CLI and document_writing
 refs export are the DIFFERENTIATED + FALLBACK layer: they own the
 frontmatter -> BibTeX normalization contract and the zotero_key/zotero_uri
 write-back, and they delegate the actual Zotero I/O to `zotero-cli --json`
 (the same package's console script). zotero_bridge.py is deleted.

 WHY THIS SUITE DIFFERS FROM scripts/paper_search_mcp:
   paper-search-mcp is stateless and run over `uvx` (ephemeral). zotero-mcp MUST be
   installed persistently (`uv tool install`), because this project's CLI shells out
   to `zotero-cli` -- it has to be on PATH. `uv tool install "zotero-mcp-server[pdf,scite]"`
   provides BOTH console scripts: `zotero-mcp` (the MCP server) and `zotero-cli`.
   There is still NO git clone, NO isolated venv, NO license, NO JVM, NO RAG index.

 What it does (idempotent, safe to re-run):
   1. Preflight: check uv availability.
   2. Install: `uv tool install "zotero-mcp-server[<extras>]"` (default extras pdf,scite;
      semantic is deliberately NOT installed -- it pulls torch). -Reinstall to upgrade.
   3. PATH check: are `zotero-mcp` and `zotero-cli` resolvable? (uv tool bin on PATH?)
   4. Smoke + write authorization reminders (zotero-cli --help; `zotero-mcp authorize-local`).
   5. -StatusOnly: read-only report (lock version, uv, tool installed?, CLIs on PATH?,
      registered in mcp.json?, local vs web mode).
   6. Print the resolved mcpServers registration JSON (copy-paste ready) + Go/No-Go checklist.

 NOTE: This file is intentionally ASCII-only. Windows PowerShell 5.1 reads a
 BOM-less .ps1 as the ANSI code page (GBK on zh-CN), which corrupts UTF-8
 non-ASCII bytes and breaks parsing. (The sibling .json files DO carry CJK prose
 and are read with an explicit UTF-8 encoding below.)

--------------------------------------------------------------------------------
 Usage
--------------------------------------------------------------------------------
   # Default: preflight + install [pdf,scite] + PATH check + print registration JSON
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\zotero_mcp\setup_zotero_mcp.ps1

   # Upgrade / force reinstall:
   ... setup_zotero_mcp.ps1 -Reinstall

   # Pin the version in the printed uvx-variant args (reproduce the spike'd 0.13.1):
   ... setup_zotero_mcp.ps1 -PinVersion

   # Web mode instead of local (older Zotero / no desktop app):
   ... setup_zotero_mcp.ps1 -ZoteroLocal 'false' -ZoteroApiKey 'XXXX' -ZoteroLibraryId '12345'

   # Enable the Scite tool group in the printed env:
   ... setup_zotero_mcp.ps1 -Toolsets 'scite'

   # Smoke the CLI + optionally install the agent skill / run authorize-local:
   ... setup_zotero_mcp.ps1 -Smoke -InstallSkill
   ... setup_zotero_mcp.ps1 -AuthorizeLocal      # INTERACTIVE: pops a Zotero dialog

   # Read-only status:
   ... setup_zotero_mcp.ps1 -StatusOnly

   # Print intended actions only, zero side effects:
   ... setup_zotero_mcp.ps1 -DryRun

--------------------------------------------------------------------------------
 Registration (add the community zotero-mcp in Qoder)
--------------------------------------------------------------------------------
 User-level config file: C:\Users\antar\.qoder-cn\shared_client\mcp.json   <-- EDIT THIS ONE
 Qoder mirrors it to ...\shared_client\extension\local\mcp.json and adds a
 userConfigMD5 key there; that mirror is DERIVED, so do not hand-edit it.
 Shape (stdio): { "mcpServers": { "zotero": { "command":"zotero-mcp", "env":{ "ZOTERO_LOCAL":"true" } } } }
 This script PRINTS the resolved JSON block at the end -- copy it in (comment-free).
 Reference template: scripts\zotero_mcp\mcp_servers.template.json

 Provenance / version SOURCE OF TRUTH: scripts\zotero_mcp\UPSTREAM.lock.json
 It records the upstream project, version (0.13.1), license, deps, extras (pdf+scite
 installed; semantic deferred to avoid torch), the zotero-cli --json envelope contract,
 and the spike findings -- notably that items come back in raw Zotero-API shape (pyzotero
 inside), so refs_bridge's field access is unchanged, and that frontmatter -> BibTeX
 normalization stays in this project (zotero_cli.frontmatter_to_bibtex).

--------------------------------------------------------------------------------
 Phase 3 Go/No-Go checklist (drive via CallMcpTool / zotero-cli after registering)
--------------------------------------------------------------------------------
   [ ] `uv tool install "zotero-mcp-server[pdf,scite]"` succeeded; zotero-mcp + zotero-cli on PATH
   [ ] Zotero desktop running; Settings -> Advanced -> "Allow other applications..." ticked
   [ ] Writes: `zotero-mcp authorize-local` run once, chose "Always Allow" (Zotero 10+)
       (or ZOTERO_API_KEY + ZOTERO_LIBRARY_ID set for web mode)
   [ ] `zotero-cli --json config` returns ok:true (library reachable)
   [ ] Qoder spawns the `zotero` MCP; a search / get metadata tool call returns items
   [ ] `pysci-research add <DOI>` files the item and writes zotero_key/zotero_uri to frontmatter
   [ ] `compose tex refs` exports .bib from the library with citekeys matching the research side
   ==> If zotero-cli cannot be installed / library unreachable: No-Go -- research add
       degrades to skeleton-only, and library/refs report the missing CLI clearly.
================================================================================
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    # Extras to install. semantic is deliberately excluded (pulls sentence-transformers/torch).
    [string] $Extras        = 'pdf,scite',
    # Pin the uvx-variant args to zotero-mcp-server==<lock version> instead of tracking latest.
    [switch] $PinVersion,
    [switch] $Reinstall,       # force reinstall/upgrade of the uv tool
    [switch] $Smoke,           # run `zotero-cli --help` (no network)
    [switch] $InstallSkill,    # run `zotero-mcp install-skill` (agent skill, ~98 tokens)
    [switch] $AuthorizeLocal,  # run `zotero-mcp authorize-local` (INTERACTIVE: Zotero dialog)
    # Registration env (local mode by default).
    [string] $ZoteroLocal      = 'true',
    [string] $ZoteroApiKey     = '',
    [string] $ZoteroLibraryId  = '',
    [string] $ZoteroLibraryType = 'user',
    [string] $Toolsets         = '',
    [switch] $StatusOnly,
    [switch] $DryRun
)

# 'Continue', not 'Stop': uv is a native command that writes progress to stderr;
# under PS 5.1 ErrorActionPreference='Stop' + redirected stderr turns that into a
# terminating NativeCommandError. We check $LASTEXITCODE explicitly where it matters.
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg)  { Write-Host "    $msg" }
function Write-Warn2($msg) { Write-Host "    [!] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host "    [OK] $msg" -ForegroundColor Green }

$lockPath  = Join-Path $PSScriptRoot 'UPSTREAM.lock.json'
$mcpConfig = 'C:\Users\antar\.qoder-cn\shared_client\mcp.json'
$pkgSpec   = "zotero-mcp-server[$Extras]"

# Read the lock's upstream_version with an EXPLICIT UTF-8 encoding (PS 5.1 Get-Content
# would decode the CJK prose as GBK and break ConvertFrom-Json).
function Read-LockVersion {
    if (-not (Test-Path $lockPath)) { return $null }
    try {
        $json = [System.IO.File]::ReadAllText($lockPath, [System.Text.Encoding]::UTF8)
        return ($json | ConvertFrom-Json).upstream.upstream_version
    } catch { Write-Warn2 "Cannot parse $lockPath -- $($_.Exception.Message)"; return $null }
}
$lockVersion = Read-LockVersion

Write-Host '====================================================================' -ForegroundColor DarkCyan
Write-Host ' Phase 3 :: community zotero-mcp setup' -ForegroundColor DarkCyan
Write-Host '====================================================================' -ForegroundColor DarkCyan
if ($DryRun) { Write-Warn2 'DryRun mode: prints intended actions only, no side effects.' }
Write-Info "lock upstream_version = $(if ($lockVersion) { $lockVersion } else { '<unknown>' })"
Write-Info "install spec          = $pkgSpec  (semantic deliberately excluded: avoids torch)"
Write-Info "mode                  = $(if ($ZoteroLocal -eq 'true') { 'local (zotero.sqlite + Zotero 10+ writes)' } else { 'web (ZOTERO_API_KEY + ZOTERO_LIBRARY_ID)' })"

# ---------------------------------------------------------------------------
# -StatusOnly: read-only report, then exit
# ---------------------------------------------------------------------------
if ($StatusOnly) {
    Write-Step 'Toolchain'
    $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if ($uv) { Write-Ok "uv = $uv" } else { Write-Warn2 'uv not found on PATH.' }

    Write-Step 'Persistent tool install'
    if ($uv) {
        $tools = (& uv tool list 2>$null | Out-String)
        if ($tools -match 'zotero-mcp-server') { Write-Ok 'zotero-mcp-server is installed as a uv tool.'; Write-Info ($tools.Trim()) }
        else { Write-Warn2 'zotero-mcp-server not installed as a uv tool. Run this script without -StatusOnly.' }
    }
    $zm  = (Get-Command zotero-mcp  -ErrorAction SilentlyContinue).Source
    $zc  = (Get-Command zotero-cli  -ErrorAction SilentlyContinue).Source
    if ($zm) { Write-Ok "zotero-mcp  = $zm" } else { Write-Warn2 'zotero-mcp not on PATH (needed for MCP registration).' }
    if ($zc) { Write-Ok "zotero-cli = $zc" } else { Write-Warn2 'zotero-cli not on PATH (needed by pysci-research add/library + refs export). Try `uv tool update-shell` then restart the shell.' }

    Write-Step "Registration in $mcpConfig"
    if (Test-Path $mcpConfig) {
        try {
            $cfgText = [System.IO.File]::ReadAllText($mcpConfig, [System.Text.Encoding]::UTF8)
            $cfg = $cfgText | ConvertFrom-Json
            $entry = $cfg.mcpServers.'zotero'
            if ($entry) {
                Write-Ok 'zotero IS registered.'
                Write-Info "command = $($entry.command)  env.ZOTERO_LOCAL = $($entry.env.ZOTERO_LOCAL)"
            } else { Write-Warn2 'zotero NOT registered yet. Run this script without -StatusOnly and paste the printed JSON.' }
        } catch { Write-Warn2 "Cannot parse $mcpConfig -- $($_.Exception.Message)" }
    } else { Write-Warn2 "mcp.json not found at $mcpConfig" }
    return
}

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
Write-Step '1/5 Preflight (uv)'
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw 'uv not found. Install uv first (https://docs.astral.sh/uv/).' }
Write-Ok "uv = $uv"

# ---------------------------------------------------------------------------
# 2. Persistent install (uv tool install "zotero-mcp-server[extras]")
# ---------------------------------------------------------------------------
Write-Step "2/5 Install ($pkgSpec)"
$tArgs = @('tool', 'install', $pkgSpec)
if ($Reinstall) { $tArgs += '--reinstall' }
if ($DryRun) { Write-Info "[DRY] uv $($tArgs -join ' ')" }
else {
    Write-Info 'First install downloads deps (PyMuPDF for the pdf extra; can take a minute)...'
    & uv @tArgs
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "uv tool install exit $LASTEXITCODE -- read messages above (network?)." }
    else { Write-Ok "installed $pkgSpec as a uv tool (provides zotero-mcp + zotero-cli)." }
}

# ---------------------------------------------------------------------------
# 3. PATH check (both console scripts)
# ---------------------------------------------------------------------------
Write-Step '3/5 PATH check (zotero-mcp / zotero-cli)'
if ($DryRun) {
    Write-Info '[DRY] Get-Command zotero-mcp / zotero-cli'
} else {
    $zm = (Get-Command zotero-mcp -ErrorAction SilentlyContinue).Source
    $zc = (Get-Command zotero-cli -ErrorAction SilentlyContinue).Source
    if ($zm) { Write-Ok "zotero-mcp  = $zm" } else { Write-Warn2 'zotero-mcp not on PATH yet.' }
    if ($zc) { Write-Ok "zotero-cli = $zc" } else { Write-Warn2 'zotero-cli not on PATH yet -- pysci-research add/library + refs export need it.' }
    if (-not $zm -or -not $zc) {
        Write-Info 'Fix: run `uv tool update-shell` (adds the uv tool bin to PATH), then restart the shell and re-run -StatusOnly.'
    }
}

# ---------------------------------------------------------------------------
# 4. Smoke + write-authorization reminders
# ---------------------------------------------------------------------------
Write-Step '4/5 Smoke + write authorization'
if ($Smoke) {
    if ($DryRun) { Write-Info '[DRY] zotero-cli --help' }
    else {
        $zc = (Get-Command zotero-cli -ErrorAction SilentlyContinue).Source
        if (-not $zc) { Write-Warn2 'zotero-cli not on PATH; cannot smoke.' }
        else { & zotero-cli --help 2>&1 | Out-String | Write-Host }
    }
} else {
    Write-Info 'Skipped -Smoke. After install, try: zotero-cli --json config   (proves the library is reachable).'
}
if ($InstallSkill) {
    if ($DryRun) { Write-Info '[DRY] zotero-mcp install-skill' }
    else {
        Write-Info 'Installing the agent skill (~98 tokens vs ~13.4k for MCP schemas)...'
        & zotero-mcp install-skill 2>&1 | Out-String | Write-Host
    }
}
if ($AuthorizeLocal) {
    Write-Warn2 'authorize-local is INTERACTIVE: a Zotero dialog will pop up -- choose "Always Allow".'
    if ($DryRun) { Write-Info '[DRY] zotero-mcp authorize-local' }
    else { & zotero-mcp authorize-local 2>&1 | Out-String | Write-Host }
} else {
    Write-Info 'To enable WRITES on Zotero 10+: run `zotero-mcp authorize-local` once and choose "Always Allow".'
    Write-Info 'On older Zotero / no desktop app: set ZOTERO_API_KEY + ZOTERO_LIBRARY_ID (web mode) instead.'
}

# ---------------------------------------------------------------------------
# 5. Print registration JSON + Go/No-Go reminder
# ---------------------------------------------------------------------------
Write-Step '5/5 mcpServers registration block (copy into C:\Users\antar\.qoder-cn\shared_client\mcp.json)'
$envBlock = [ordered]@{
    ZOTERO_LOCAL        = $ZoteroLocal
    ZOTERO_API_KEY      = $ZoteroApiKey
    ZOTERO_LIBRARY_ID   = $ZoteroLibraryId
    ZOTERO_LIBRARY_TYPE = $ZoteroLibraryType
    ZOTERO_MCP_TOOLSETS = $Toolsets
}
$entry = [ordered]@{ command = 'zotero-mcp'; env = $envBlock }
$reg   = [ordered]@{ mcpServers = [ordered]@{ 'zotero' = $entry } }
$json  = $reg | ConvertTo-Json -Depth 8
Write-Host ''
Write-Host '----8<---- paste this entry into mcpServers (alongside arxiv / paper-search-mcp) ----8<----' -ForegroundColor DarkGray
Write-Host $json
Write-Host '----8<-----------------------------------------------------------------------------8<----' -ForegroundColor DarkGray
if ($ZoteroLocal -ne 'true' -and (-not $ZoteroApiKey -or -not $ZoteroLibraryId)) {
    Write-Warn2 'Web mode selected but ZOTERO_API_KEY / ZOTERO_LIBRARY_ID are empty -- writes/reads will fail. Pass -ZoteroApiKey and -ZoteroLibraryId.'
}
Write-Info 'This MCP is the LIBRARY PRIMARY. The pysci-research CLI + refs export delegate to zotero-cli'
Write-Info 'and keep the differentiated frontmatter->BibTeX normalization + zotero_key write-back (zotero_bridge.py deleted).'
Write-Info 'pdf + scite extras installed; semantic deferred (torch). To add it later: uv tool install "zotero-mcp-server[semantic]".'
Write-Host ''
Write-Ok 'Setup script finished. Next: paste the JSON into mcp.json, restart Qoder, run authorize-local, then the Go/No-Go checklist.'
