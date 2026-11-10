$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Please run setup.cmd first.' }
if (-not (Test-Path -LiteralPath 'web\dist\index.html')) { throw 'UI has not been built. Please run setup.cmd first.' }
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2
    if ($health.ok -and $health.local) {
        Start-Process 'http://127.0.0.1:8765'
        exit 0
    }
} catch { }
Write-Host 'Tingji: http://127.0.0.1:8765'
Write-Host 'Keep this window open while processing. Press Ctrl+C to stop.'
& $pythonExe scripts/run.py
exit $LASTEXITCODE
