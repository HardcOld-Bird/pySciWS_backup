# tests.ps1 - one entry for the mechanical layer.
# Actions: all | fast | harness | hardware | cov
# ASCII-only comments (Windows PowerShell 5.1 + no BOM).
param(
    [Parameter(Position = 0)]
    [ValidateSet('all', 'fast', 'harness', 'hardware', 'cov')]
    [string]$Action = 'all'
)

$ErrorActionPreference = 'Stop'
$SubRoot = Split-Path -Parent $PSScriptRoot
$ProjRoot = Split-Path -Parent $SubRoot
$Python = Join-Path $ProjRoot '.venv\Scripts\python.exe'
$Ruff = Join-Path $ProjRoot '.venv\Scripts\ruff.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "missing venv python: $Python (run init-deps.cmd)"
}

Push-Location $ProjRoot
try {
    switch ($Action) {
        'all' {
            if (Test-Path -LiteralPath $Ruff) {
                & $Ruff check .
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            & $Python -m pytest
            exit $LASTEXITCODE
        }
        'fast' {
            & $Python -m pytest -m "not slow and not hardware"
            exit $LASTEXITCODE
        }
        'harness' {
            & $Python -m pytest sub_root/tests/harness
            exit $LASTEXITCODE
        }
        'hardware' {
            & $Python -m pytest -m hardware
            exit $LASTEXITCODE
        }
        'cov' {
            & $Python -m pytest --cov
            exit $LASTEXITCODE
        }
    }
} finally {
    Pop-Location
}
