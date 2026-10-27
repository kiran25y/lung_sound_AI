param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [int]$Workers = 4,
    [int]$StressPerSourceClass = 75,
    [string]$WslDistribution = "Ubuntu",
    [string]$LinuxPython = "/home/user/.venvs/hani_eval/bin/python"
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataset = (Resolve-Path -LiteralPath $DatasetRoot).Path
$projectWsl = (wsl.exe -d $WslDistribution -- wslpath -a $project).Trim()
$datasetWsl = (wsl.exe -d $WslDistribution -- wslpath -a $dataset).Trim()

$env:WORKERS = "$Workers"
$env:STRESS_PER_SOURCE_CLASS = "$StressPerSourceClass"

$wslArgs = @(
    "-d", $WslDistribution, "--", "env",
    "PYTHON_BIN=$LinuxPython", "WORKERS=$Workers",
    "STRESS_PER_SOURCE_CLASS=$StressPerSourceClass",
    "bash", "$projectWsl/scripts/run_end_to_end_comparison.sh", $datasetWsl
)
& wsl.exe @wslArgs

if ($LASTEXITCODE -ne 0) {
    throw "End-to-end comparison failed with exit code $LASTEXITCODE"
}
