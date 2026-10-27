param(
    [string]$BuildDirectory = "build_hanil_ai_package",
    [string]$PackageDirectory = "release/hanil_ai_esp32s3",
    [string]$ZipPath = "release/hanil_ai_esp32s3.zip"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BuildDirectory))
$PackageRoot = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $PackageDirectory))
$ZipFile = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ZipPath))
$SourceHeaders = Join-Path $RepositoryRoot "components/hanil_ai/include"
$BuiltLibrary = Join-Path $BuildRoot "esp-idf/hanil_ai/libhanil_ai.a"

if (-not (Test-Path -LiteralPath $BuiltLibrary -PathType Leaf)) {
    throw "Built library not found: $BuiltLibrary. Build with HANIL_AI_LIBRARY_EXAMPLE=ON first."
}
if (-not (Test-Path -LiteralPath $PackageRoot -PathType Container)) {
    throw "Package directory not found: $PackageRoot"
}

$PackageLib = Join-Path $PackageRoot "lib"
$PackageInclude = Join-Path $PackageRoot "include"
New-Item -ItemType Directory -Force -Path $PackageLib, $PackageInclude | Out-Null
Copy-Item -LiteralPath $BuiltLibrary -Destination (Join-Path $PackageLib "libhanil_ai.a") -Force

$Headers = @(
    "hanil_ai_types.h",
    "hanil_microphone.h",
    "hanil_quality_gate.h",
    "hanil_frontend.h",
    "hanil_classifier.h",
    "hanil_pipeline.h"
)
foreach ($Header in $Headers) {
    Copy-Item -LiteralPath (Join-Path $SourceHeaders $Header) `
              -Destination (Join-Path $PackageInclude $Header) -Force
}

$HashTargets = Get-ChildItem -LiteralPath $PackageRoot -Recurse -File |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Sort-Object FullName
$HashLines = foreach ($File in $HashTargets) {
    $Relative = [System.IO.Path]::GetRelativePath($PackageRoot, $File.FullName).Replace('\', '/')
    $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $Relative"
}
Set-Content -LiteralPath (Join-Path $PackageRoot "SHA256SUMS.txt") `
            -Value $HashLines -Encoding utf8NoBOM

if (Test-Path -LiteralPath $ZipFile -PathType Leaf) {
    Remove-Item -LiteralPath $ZipFile -Force
}
Compress-Archive -Path (Join-Path $PackageRoot "*") -DestinationPath $ZipFile `
                 -CompressionLevel Optimal

$Archive = Get-Item -LiteralPath (Join-Path $PackageLib "libhanil_ai.a")
$Zip = Get-Item -LiteralPath $ZipFile
$ArchiveHash = (Get-FileHash -LiteralPath $Archive.FullName -Algorithm SHA256).Hash
$ZipHash = (Get-FileHash -LiteralPath $Zip.FullName -Algorithm SHA256).Hash

Write-Host "Package refreshed successfully."
Write-Host "Static library: $($Archive.FullName) ($($Archive.Length) bytes)"
Write-Host "Library SHA256: $ArchiveHash"
Write-Host "ZIP: $($Zip.FullName) ($($Zip.Length) bytes)"
Write-Host "ZIP SHA256: $ZipHash"
