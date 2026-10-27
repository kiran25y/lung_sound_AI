param(
    [string]$PythonExe = "python",
    [string]$Manifest = "data/manifest_local.csv",
    [string]$InitialCheckpoint = "runs/improved_specificity_s1337/best.pt",
    [string]$Output = "runs/improved_subtype_s1337",
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    & $PythonExe phase1_baseline.py `
        --manifest $Manifest `
        --multilabel `
        --model dscnn_tcn `
        --width 2.0 `
        --init-checkpoint $InitialCheckpoint `
        --subtype-loss-weight 0.25 `
        --selection-metric product_composite `
        --imbalance-strategy natural_sqrt `
        --multilabel-pos-weight-power 0.5 `
        --multilabel-pos-weight-cap 4.0 `
        --sungbook-train-frac 0.30 `
        --threshold-external-frac 0.30 `
        --threshold-mode joint `
        --target-abnormal-sensitivity 0.80 `
        --threshold-grid-step 0.01 `
        --specaugment `
        --bandlimit-prob 0.50 `
        --event-overlay 0.20 `
        --mixup-alpha 0.20 `
        --mixstyle `
        --lr 0.0001 `
        --epochs 40 `
        --warmup-epochs 1 `
        --min-epochs 10 `
        --patience 10 `
        --batch_size 16 `
        --amp `
        --workers 0 `
        --seed $Seed `
        --out $Output
}
finally {
    Pop-Location
}
