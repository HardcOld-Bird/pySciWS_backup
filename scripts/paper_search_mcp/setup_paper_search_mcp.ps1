<#
================================================================================
 Setup script -- community paper-search-mcp (openags/paper-search-mcp)
================================================================================
 One-shot preparation tool for Phase 2 (retrieval layer) of the
 "literature_research community-infrastructure replacement" plan. It sets up the
 COMMUNITY paper-search-mcp MCP only -- the MULTI-SOURCE RETRIEVAL PRIMARY layer.

 Architecture (mirrors comsol-simulation): the community MCP is the primary
 retrieval path (Agent calls search_papers for concurrent multi-source search +
 dedup, and download_with_fallback for the OA full-text fallback chain). The
 in-house `pysci-research` CLI is the DIFFERENTIATED + FALLBACK layer: it owns
 the source-output -> frontmatter normalization contract, the WoS enrichment
 overlay, and the arXiv LaTeX-source path (download_source / extract_tex_from_source)
 -- none of which paper-search-mcp provides. Use the CLI's own search only when
 this MCP is unavailable.

 WHY THIS SUITE IS MUCH SIMPLER THAN scripts/comsol_mcp:
   paper-search-mcp is a STATELESS PyPI package run over stdio by `uvx`. There is
   NO git clone, NO isolated venv, NO commit pin, NO RAG index, NO license, NO
   JVM, NO persistent server to supervise. Qoder spawns it on demand. So this
   script has none of comsol's transport/license/origin/duplicate-instance machinery.

 What it does (idempotent, safe to re-run):
   1. Preflight: check uv / uvx availability.
   2. Optional -Prewarm: warm the uvx cache and prove the server module imports
      (NON-BLOCKING; does NOT start the stdio server). Network required.
   3. Optional -InstallTool: `uv tool install paper-search-mcp` (persistent install;
      also provides the `paper-search` CLI). Then the printed registration uses the
      uv-tool variant instead of uvx.
   4. Optional -Smoke: run `paper-search --help` to reveal the real CLI surface
      (no network). A live search is best driven through the MCP after registering.
   5. -StatusOnly: read-only report (lock version, uv present, tool installed?,
      registered in mcp.json?, Unpaywall email set?).
   6. Print the resolved mcpServers registration JSON (copy-paste ready) and the
      Go/No-Go checklist.

 NOTE: This file is intentionally ASCII-only. Windows PowerShell 5.1 reads a
 BOM-less .ps1 as the ANSI code page (GBK on zh-CN), which corrupts UTF-8
 non-ASCII bytes and breaks parsing. Keeping it ASCII makes it robust in any
 shell / PS version, foreground or background. (The sibling .json files DO carry
 CJK prose and are read with an explicit UTF-8 encoding below.)

--------------------------------------------------------------------------------
 Usage
--------------------------------------------------------------------------------
   # Default (no network): preflight + print the uvx registration JSON + checklist
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\paper_search_mcp\setup_paper_search_mcp.ps1

   # Bake your real Unpaywall email (and optional CORE key) into the printed JSON:
   ... setup_paper_search_mcp.ps1 -UnpaywallEmail 'you@example.com' -CoreApiKey 'xxxx'

   # Pin the version in the printed args (reproduce the spike'd 0.1.4 exactly):
   ... setup_paper_search_mcp.ps1 -PinVersion

   # Warm the uvx cache + prove the server imports (network; non-blocking):
   ... setup_paper_search_mcp.ps1 -Prewarm

   # Persistent install (gives the `paper-search` CLI) + print the uv-tool variant:
   ... setup_paper_search_mcp.ps1 -InstallTool -Smoke

   # Read-only status:
   ... setup_paper_search_mcp.ps1 -StatusOnly

   # Print intended actions only, zero side effects:
   ... setup_paper_search_mcp.ps1 -DryRun

--------------------------------------------------------------------------------
 Registration (add the community paper-search-mcp in Qoder)
--------------------------------------------------------------------------------
 User-level config file: C:\Users\antar\.qoder-cn\shared_client\mcp.json   <-- EDIT THIS ONE
 Qoder mirrors it to ...\shared_client\extension\local\mcp.json and adds a
 userConfigMD5 key there; that mirror is DERIVED, so do not hand-edit it.
 Shape (stdio): { "mcpServers": { "paper-search-mcp": { "command":"uvx", "args":["paper-search-mcp"], "env":{...} } } }
 This script PRINTS the resolved JSON block at the end -- copy it in (strip nothing;
 it is already comment-free). Reference template: scripts\paper_search_mcp\mcp_servers.template.json

 Provenance / version SOURCE OF TRUTH: scripts\paper_search_mcp\UPSTREAM.lock.json
 It records the upstream project, version (0.1.4), license, deps, the two-layer tool
 contract, and the spike findings -- notably that WoS + Scopus are STILL UNIMPLEMENTED
 upstream (so the in-house wos_client stays primary for journal metrics), and that the
 OpenAlex connector has NO API-key env var (runs keyless under OpenAlex's ~100 credits/day
 cap; the in-house openalex_client injects OPENALEX_API_KEY and stays the quota-safe fallback).

--------------------------------------------------------------------------------
 Phase 2 Go/No-Go checklist (drive via CallMcpTool after registering)
--------------------------------------------------------------------------------
   [ ] Qoder spawns paper-search-mcp over uvx (first launch downloads deps; needs network)
   [ ] search_papers('non-Hermitian acoustic exceptional point', sources=['arxiv','openalex'])
       returns a standardized, deduped Paper list
   [ ] download_with_fallback on an OA paper resolves a PDF via the fallback chain
   [ ] arxiv-mcp-server STILL registered alongside (complementary: get_paper_latex /
       semantic_search / citation_graph are unique to it) -- do NOT remove it
   [ ] WoS metrics still come from `pysci-research search --enrich` / `get` (wos_client),
       NOT from this MCP (WoS/Scopus unimplemented upstream)
   ==> If the MCP cannot be spawned (network blocked): No-Go for the primary path --
       fall back to `pysci-research search` (openalex_client + arxiv_client) and re-plan.
================================================================================
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    # Baked into the printed registration JSON. Unpaywall is the ONLY source that
    # hard-requires a value (its connector is skipped entirely without an email).
    [string] $UnpaywallEmail = '<your@email.com>',
    [string] $CoreApiKey     = '',
    # Pin args to ["paper-search-mcp==<lock version>"] instead of tracking latest.
    [switch] $PinVersion,
    # Persistent install (uv tool install) instead of ephemeral uvx; prints the uv-tool variant.
    [switch] $InstallTool,
    [switch] $Reinstall,       # with -InstallTool: force reinstall/upgrade
    [switch] $Prewarm,         # warm the uvx cache + prove the server module imports (network, non-blocking)
    [switch] $Smoke,           # run `paper-search --help` (needs -InstallTool; no network)
    [switch] $StatusOnly,
    [switch] $DryRun
)

# 'Continue', not 'Stop': uv/uvx are native commands that write progress to stderr;
# under PS 5.1 ErrorActionPreference='Stop' + redirected stderr turns that into a
# terminating NativeCommandError. We check $LASTEXITCODE explicitly where it matters.
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

function Write-Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Info($msg)  { Write-Host "    $msg" }
function Write-Warn2($msg) { Write-Host "    [!] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host "    [OK] $msg" -ForegroundColor Green }

$lockPath   = Join-Path $PSScriptRoot 'UPSTREAM.lock.json'
$mcpConfig  = 'C:\Users\antar\.qoder-cn\shared_client\mcp.json'

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
Write-Host ' Phase 2 :: community paper-search-mcp setup' -ForegroundColor DarkCyan
Write-Host '====================================================================' -ForegroundColor DarkCyan
if ($DryRun) { Write-Warn2 'DryRun mode: prints intended actions only, no side effects.' }
Write-Info "lock upstream_version = $(if ($lockVersion) { $lockVersion } else { '<unknown>' })"
Write-Info "install mode          = $(if ($InstallTool) { 'uv tool install (persistent + paper-search CLI)' } else { 'uvx (ephemeral, always-latest)' })"

# ---------------------------------------------------------------------------
# -StatusOnly: read-only report, then exit
# ---------------------------------------------------------------------------
if ($StatusOnly) {
    Write-Step 'Toolchain'
    $uv  = (Get-Command uv  -ErrorAction SilentlyContinue).Source
    $uvx = (Get-Command uvx -ErrorAction SilentlyContinue).Source
    if ($uv)  { Write-Ok "uv  = $uv" }  else { Write-Warn2 'uv not found on PATH.' }
    if ($uvx) { Write-Ok "uvx = $uvx" } else { Write-Warn2 'uvx not found on PATH (uvx ships with uv; reinstall uv if missing).' }

    Write-Step 'Persistent tool install (only if -InstallTool was used)'
    if ($uv) {
        $tools = (& uv tool list 2>$null | Out-String)
        if ($tools -match 'paper-search-mcp') { Write-Ok 'paper-search-mcp is installed as a uv tool.'; Write-Info ($tools.Trim()) }
        else { Write-Info 'paper-search-mcp not installed as a uv tool (fine if you use the uvx path).' }
        $ps = (Get-Command paper-search -ErrorAction SilentlyContinue).Source
        if ($ps) { Write-Ok "paper-search CLI = $ps" } else { Write-Info 'paper-search CLI not on PATH.' }
    }

    Write-Step "Registration in $mcpConfig"
    if (Test-Path $mcpConfig) {
        try {
            $cfgText = [System.IO.File]::ReadAllText($mcpConfig, [System.Text.Encoding]::UTF8)
            $cfg = $cfgText | ConvertFrom-Json
            $entry = $cfg.mcpServers.'paper-search-mcp'
            if ($entry) {
                Write-Ok 'paper-search-mcp IS registered.'
                Write-Info "command = $($entry.command)  args = $($entry.args -join ' ')"
                $em = $entry.env.'PAPER_SEARCH_MCP_UNPAYWALL_EMAIL'
                if ($em -and $em -ne '<your@email.com>') { Write-Ok "Unpaywall email set: $em" }
                else { Write-Warn2 'Unpaywall email is a placeholder/empty -- the Unpaywall source will be skipped. Set a real email.' }
                if ($lockVersion -and ($entry.args -join ' ') -notmatch [regex]::Escape($lockVersion) -and ($entry.args -join ' ') -notmatch 'paper-search-mcp$') {
                    Write-Info 'args do not pin a version => tracking latest (capability-first; expected unless -PinVersion was used).'
                }
            } else { Write-Warn2 'paper-search-mcp NOT registered yet. Run this script without -StatusOnly and paste the printed JSON.' }
            $arxiv = $cfg.mcpServers.'arxiv'
            if ($arxiv) { Write-Ok 'arxiv-mcp-server still registered (complementary -- keep it).' }
            else { Write-Warn2 'arxiv-mcp-server not found -- it should be KEPT (deep arXiv tools are unique to it).' }
        } catch { Write-Warn2 "Cannot parse $mcpConfig -- $($_.Exception.Message)" }
    } else { Write-Warn2 "mcp.json not found at $mcpConfig" }
    return
}

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
Write-Step '1/4 Preflight (uv / uvx)'
$uv  = (Get-Command uv  -ErrorAction SilentlyContinue).Source
$uvx = (Get-Command uvx -ErrorAction SilentlyContinue).Source
if (-not $uv) { throw 'uv not found. Install uv first (https://docs.astral.sh/uv/).' }
Write-Ok "uv  = $uv"
if ($InstallTool) {
    # uv tool path only needs uv.
    Write-Ok 'uv tool mode: uvx not strictly required.'
} elseif (-not $uvx) {
    Write-Warn2 'uvx not found on PATH. uvx ships with uv; if missing, reinstall uv or use -InstallTool.'
} else {
    Write-Ok "uvx = $uvx"
}

# ---------------------------------------------------------------------------
# 2. Optional prewarm / persistent install
# ---------------------------------------------------------------------------
if ($InstallTool) {
    Write-Step '2/4 Persistent install (uv tool install paper-search-mcp)'
    $tArgs = @('tool', 'install', 'paper-search-mcp')
    if ($Reinstall) { $tArgs += '--reinstall' }
    if ($DryRun) { Write-Info "[DRY] uv $($tArgs -join ' ')" }
    else {
        & uv @tArgs
        if ($LASTEXITCODE -ne 0) { Write-Warn2 "uv tool install exit $LASTEXITCODE -- read messages above (network?)." }
        else {
            Write-Ok 'paper-search-mcp installed as a uv tool.'
            $ps = (Get-Command paper-search -ErrorAction SilentlyContinue).Source
            if ($ps) { Write-Ok "paper-search CLI = $ps" } else { Write-Warn2 'paper-search CLI not on PATH yet; you may need to restart the shell (uv tool dir).' }
        }
    }
} elseif ($Prewarm) {
    Write-Step '2/4 Prewarm uvx cache + import check (network, non-blocking)'
    # Prove uvx can resolve the package AND that the server module (where the FastMCP
    # app + @mcp.tool defs live) imports cleanly -- pulling every runtime dep (fastmcp,
    # mcp, feedparser, bs4, lxml, httpx, pypdf, requests). This does NOT start the stdio
    # server (that would block). Analogous to comsol's `import src.server` license-free check.
    $imp = 'import paper_search_mcp.server as s; print("PAPER_SEARCH_MCP_IMPORT_OK")'
    if ($DryRun) { Write-Info "[DRY] uvx --from paper-search-mcp python -c `"$imp`"" }
    else {
        Write-Info 'First run downloads deps (can take a minute on a cold cache)...'
        $out = (& uvx --from paper-search-mcp python -c $imp 2>&1 | Out-String)
        if ($out -match 'PAPER_SEARCH_MCP_IMPORT_OK') { Write-Ok 'uvx resolved the package and paper_search_mcp.server imports cleanly.' }
        else { Write-Warn2 'Import check did not confirm OK -- tail below (likely network):'; Write-Info $out.Trim() }
    }
} else {
    Write-Step '2/4 Install / prewarm (skipped)'
    Write-Info 'Default: no install. Qoder will resolve paper-search-mcp via uvx on first launch.'
    Write-Info 'Add -Prewarm to warm the cache now, or -InstallTool for a persistent install + CLI.'
}

# ---------------------------------------------------------------------------
# 3. Optional smoke (CLI help; no network)
# ---------------------------------------------------------------------------
Write-Step '3/4 Smoke (paper-search --help)'
if (-not $InstallTool) {
    Write-Info 'Skipped (needs -InstallTool to provide the paper-search CLI).'
} elseif ($DryRun) {
    Write-Info '[DRY] paper-search --help'
} else {
    $ps = (Get-Command paper-search -ErrorAction SilentlyContinue).Source
    if (-not $ps) { Write-Warn2 'paper-search CLI not on PATH; cannot smoke. Restart the shell after -InstallTool.' }
    else {
        & paper-search --help 2>&1 | Out-String | Write-Host
        Write-Info 'A live search is best driven through the MCP (CallMcpTool search_papers) after registering.'
    }
}

# ---------------------------------------------------------------------------
# 4. Print registration JSON + Go/No-Go reminder
# ---------------------------------------------------------------------------
Write-Step '4/4 mcpServers registration block (copy into C:\Users\antar\.qoder-cn\shared_client\mcp.json)'
$pkgArg = if ($PinVersion -and $lockVersion) { "paper-search-mcp==$lockVersion" } else { 'paper-search-mcp' }
$envBlock = [ordered]@{
    PAPER_SEARCH_MCP_UNPAYWALL_EMAIL       = $UnpaywallEmail
    PAPER_SEARCH_MCP_CORE_API_KEY          = $CoreApiKey
    PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY = ''   # deliberately empty: campus net + S2 removed in Phase 1a
    PAPER_SEARCH_MCP_GOOGLE_SCHOLAR_PROXY_URL = ''
    PAPER_SEARCH_MCP_DOAJ_API_KEY          = ''
    PAPER_SEARCH_MCP_ZENODO_ACCESS_TOKEN   = ''
}
if ($InstallTool) {
    $entry = [ordered]@{ command = 'uv'; args = @('tool', 'run', 'paper-search-mcp'); env = $envBlock }
} else {
    $entry = [ordered]@{ command = 'uvx'; args = @($pkgArg); env = $envBlock }
}
$reg  = [ordered]@{ mcpServers = [ordered]@{ 'paper-search-mcp' = $entry } }
$json = $reg | ConvertTo-Json -Depth 8
Write-Host ''
Write-Host '----8<---- paste this entry into mcpServers (alongside arxiv etc.) ----8<----' -ForegroundColor DarkGray
Write-Host $json
Write-Host '----8<------------------------------------------------------------------------8<----' -ForegroundColor DarkGray
if ($UnpaywallEmail -eq '<your@email.com>') {
    Write-Warn2 'Unpaywall email is still the placeholder. Re-run with -UnpaywallEmail ''you@example.com'' or edit it in mcp.json -- the Unpaywall source is skipped without it.'
}
Write-Info 'This MCP is the RETRIEVAL PRIMARY. The pysci-research CLI stays the fallback + differentiated layer'
Write-Info '(frontmatter normalization, WoS enrichment, arXiv LaTeX source). KEEP arxiv-mcp-server (complementary).'
Write-Info 'WoS/Scopus are UNIMPLEMENTED upstream -> journal metrics keep coming from wos_client (search --enrich).'
Write-Host ''
Write-Ok 'Setup script finished. Next: paste the JSON into mcp.json, restart Qoder, then run the Go/No-Go checklist live.'
