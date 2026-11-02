$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '8'
$started = (Get-Date).ToString('o')
$statusPath = Join-Path $PSScriptRoot 'run_status.json'
@{ status = 'running'; started_at = $started; supervisor_pid = $PID; expected_runs = 175 } |
    ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
try {
    & $pythonPath -u -B system/run_incomplete.py --rates 0 0.1 0.3 0.5 0.7 --seeds 42 2024 2025 2026 2027 --variants completed --compression none --device cuda --results-dir results
    if ($LASTEXITCODE -ne 0) { throw "Matrix training exited with code $LASTEXITCODE" }
    & $pythonPath -u -B system/summarize_matrix.py --results-dir results
    if ($LASTEXITCODE -ne 0) { throw "Statistics generation exited with code $LASTEXITCODE" }
    @{ status = 'completed'; started_at = $started; finished_at = (Get-Date).ToString('o'); expected_runs = 175; exit_code = 0 } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
} catch {
    @{ status = 'failed'; started_at = $started; finished_at = (Get-Date).ToString('o'); error = $_.Exception.Message } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Write-Error $_
    exit 1
}
