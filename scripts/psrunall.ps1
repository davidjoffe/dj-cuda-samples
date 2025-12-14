# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE
# Helper to test run all built applications (say after build)
# PowerShell version (ps1)

Write-Host "================================"
Write-Host "dj: psrunall: Run all built applications"

# Iterate automatically over all subfolders under samples and try run built app if present
$BuildDir = "build-windows\samples"

Get-ChildItem samples -Directory | ForEach-Object {
    $name = $_.Name
    $exe  = Join-Path $BuildDir "$name\Release\dj${name}.exe"

    if (-not (Test-Path $exe)) {
        Write-Host "================================" -ForegroundColor Yellow
        Write-Host "dj: WARNING: EXECUTABLE NOT FOUND"    -ForegroundColor Yellow
        Write-Host "dj: EXPECTED: $exe"                  -ForegroundColor Yellow
        Write-Host "================================" -ForegroundColor Yellow
        Write-Host ""
    }
    else {
        Write-Host "dj: Running $name..." -ForegroundColor Green
        & $exe @args
    }
}