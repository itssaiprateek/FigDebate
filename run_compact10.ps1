# Compatibility entry point. Prefer scripts/run_development_checks.py for new work.
param(
    [string]$PythonExecutable = 'python',
    [switch]$SelectionOnly,
    [string]$RunDir
)
$ErrorActionPreference = 'Stop'
$launcherArgs = @((Join-Path $PSScriptRoot 'scripts/run_development_checks.py'))
if ($SelectionOnly) { $launcherArgs += '--selection-only' }
if ($RunDir) { $launcherArgs += @('--run-dir', $RunDir) }
& $PythonExecutable @launcherArgs
if ($LASTEXITCODE -ne 0) { throw "Development check exited with code $LASTEXITCODE." }
