$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    if (Test-Path -LiteralPath $bundledPython) {
        & $bundledPython -m venv --without-pip .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
    } else {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 and run setup.cmd again.' }
    }
}
if (Test-Path -LiteralPath $bundledPython) {
    & $bundledPython -m pip --python $pythonExe install -r requirements.txt
} else {
    & $pythonExe -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Push-Location web
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Setup complete. Download models, then start the app.'
