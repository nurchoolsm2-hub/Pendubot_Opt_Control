$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$appPython = Join-Path $appRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    Write-Host 'Preparing the Pendubot environment...'
    & python -m venv (Join-Path $appRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python could not create the environment. Install Python 3.12 or newer and try again.' }
}
Write-Host 'Checking the simulation libraries...'
& $appPython -c 'import jax, numpy, scipy, matplotlib, PIL' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing simulation libraries. This first setup needs an internet connection.'
    & $appPython -m pip install -r (Join-Path $appRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Library installation did not finish. Check your internet connection and try again.' }
}
Set-Content -LiteralPath (Join-Path $appRoot '.venv\pendubot-ready.txt') -Value 'Pendubot libraries installed' -Encoding utf8
Write-Host 'Pendubot is installed. Open Launch Pendubot.cmd to start.'
