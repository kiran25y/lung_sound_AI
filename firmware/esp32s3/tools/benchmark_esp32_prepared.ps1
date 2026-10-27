param(
    [Parameter(Mandatory = $true)] [string] $Manifest,
    [Parameter(Mandatory = $true)] [string] $Out,
    [string] $Port = "COM4"
)

$ErrorActionPreference = "Stop"
$manifestData = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
$rows = @()
$memory = [ordered]@{}

$serial = [System.IO.Ports.SerialPort]::new($Port, 115200)
$serial.ReadTimeout = 250
$serial.WriteTimeout = 30000
$serial.NewLine = "`n"

try {
    $serial.Open()
    $serial.DtrEnable = $false
    $serial.RtsEnable = $false
    $serial.DiscardInBuffer()
    $serial.RtsEnable = $true
    Start-Sleep -Milliseconds 150
    $serial.RtsEnable = $false

    $readyDeadline = (Get-Date).AddSeconds(30)
    $ready = $false
    while ((Get-Date) -lt $readyDeadline) {
        try {
            $line = $serial.ReadLine().Trim()
        } catch [System.TimeoutException] {
            continue
        }
        if ($line -match 'CLASSIFIER_MEMORY arena_used=(\d+).*arena_capacity=(\d+).*model_runtime=(\S+) psram_allocations=(\d+)') {
            $memory.arena_used_bytes = [int]$Matches[1]
            $memory.arena_capacity_bytes = [int]$Matches[2]
            $memory.model_runtime = $Matches[3]
            $memory.psram_allocations = [int]$Matches[4]
            Write-Output "[esp32] $line"
        }
        if ($line -match 'SRAM stage=tensors_ready free=(\d+) largest=(\d+) minimum_free=(\d+) total=(\d+)') {
            $memory.free_internal_bytes = [int]$Matches[1]
            $memory.largest_internal_block_bytes = [int]$Matches[2]
            $memory.minimum_free_internal_bytes = [int]$Matches[3]
            $memory.total_internal_capable_bytes = [int]$Matches[4]
            Write-Output "[esp32] $line"
        }
        if ($line -match 'input_scale=([0-9.eE+-]+) input_zp=(-?\d+)') {
            $memory.input_scale = [double]$Matches[1]
            $memory.input_zero_point = [int]$Matches[2]
        }
        if ($line -match 'READY classification_only=yes') {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        throw "ESP32 classifier did not become ready"
    }
    if ($memory.Contains("input_scale")) {
        if ([math]::Abs($memory.input_scale - [double]$manifestData.input_scale) -gt 1e-8 -or
            $memory.input_zero_point -ne [int]$manifestData.input_zero_point) {
            throw "Prepared features and device input quantization do not match"
        }
    } else {
        Write-Warning "The port attached after boot; device quantization was not reprinted. Payload hashes still protect feature parity."
    }

    $index = 0
    foreach ($expected in $manifestData.rows) {
        $index++
        [byte[]]$payload = [System.IO.File]::ReadAllBytes($expected.payload)
        [byte[]]$header = New-Object byte[] 8
        [byte[]]$magic = [System.Text.Encoding]::ASCII.GetBytes("FEA1")
        [byte[]]$length = [System.BitConverter]::GetBytes([uint32]$payload.Length)
        [System.Array]::Copy($magic, 0, $header, 0, 4)
        [System.Array]::Copy($length, 0, $header, 4, 4)
        $serial.Write($header, 0, $header.Length)
        $serial.Write($payload, 0, $payload.Length)

        $result = $null
        $resultDeadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $resultDeadline) {
            try {
                $line = $serial.ReadLine().Trim()
            } catch [System.TimeoutException] {
                continue
            }
            if ($line -match 'RESULT source=host_features p_crackle=([0-9.]+) p_wheeze=([0-9.]+) p_abnormal=([0-9.]+) class2=(\S+) class3=(\S+) latency_ms=([0-9.]+) latency_us=(\d+) qhash=([0-9a-fA-F]{8}) out_q=\[(-?\d+),(-?\d+)\]') {
                $deviceOut = @([int]$Matches[9], [int]$Matches[10])
                $truth3Known = $null -ne $expected.truth3
                $result = [ordered]@{
                    wav = $expected.wav
                    label = $expected.label
                    truth2 = $expected.truth2
                    truth3 = $expected.truth3
                    input_qhash = $expected.input_qhash
                    host_out_q = @([int]$expected.host_out_q[0], [int]$expected.host_out_q[1])
                    exact_host_device_parity = (
                        $Matches[8].ToLower() -eq $expected.input_qhash -and
                        $deviceOut[0] -eq [int]$expected.host_out_q[0] -and
                        $deviceOut[1] -eq [int]$expected.host_out_q[1]
                    )
                    p_crackle = [double]$Matches[1]
                    p_wheeze = [double]$Matches[2]
                    p_abnormal = [double]$Matches[3]
                    class2 = $Matches[4]
                    class3 = $Matches[5]
                    latency_ms = [double]$Matches[6]
                    latency_us = [int64]$Matches[7]
                    qhash = $Matches[8].ToLower()
                    out_q = $deviceOut
                    correct2 = ($Matches[4] -eq $expected.truth2)
                    correct3 = ($truth3Known -and $Matches[5] -eq $expected.truth3)
                }
                break
            }
        }
        if ($null -eq $result) {
            throw "ESP32 returned no result for $($expected.wav)"
        }
        $rows += [pscustomobject]$result
        Write-Output ("[{0:D2}/{1:D2}] {2} label={3} pred={4}/{5} latency={6:F3}ms parity={7}" -f
            $index, $manifestData.rows.Count, [System.IO.Path]::GetFileName($expected.wav),
            $expected.label, $result.class2, $result.class3, $result.latency_ms,
            $result.exact_host_device_parity)
    }
} finally {
    if ($serial.IsOpen) {
        $serial.Close()
    }
}

$latencies = @($rows | ForEach-Object { [double]$_.latency_ms } | Sort-Object)
$middle = [int]($latencies.Count / 2)
$median = if (($latencies.Count % 2) -eq 0) {
    ($latencies[$middle - 1] + $latencies[$middle]) / 2.0
} else {
    $latencies[$middle]
}
$threeClassRows = @($rows | Where-Object { $null -ne $_.truth3 })
$summary = [ordered]@{
    board = "ESP32-S3 240 MHz"
    runtime = "portable TFLM INT8, model and tensor arena in internal SRAM"
    retested_at = (Get-Date).ToString("o")
    samples = $rows.Count
    exact_parity = @($rows | Where-Object exact_host_device_parity).Count
    two_class_accuracy = @($rows | Where-Object correct2).Count / $rows.Count
    three_class_accuracy_excluding_both = @($threeClassRows | Where-Object correct3).Count / $threeClassRows.Count
    latency_ms = [ordered]@{
        mean = ($latencies | Measure-Object -Average).Average
        median = $median
        min = $latencies[0]
        max = $latencies[-1]
    }
    memory = $memory
    rows = $rows
}
$parent = Split-Path -Parent $Out
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Out -Encoding utf8
$summaryWithoutRows = [ordered]@{}
foreach ($entry in $summary.GetEnumerator()) {
    if ($entry.Key -ne "rows") {
        $summaryWithoutRows[$entry.Key] = $entry.Value
    }
}
$summaryWithoutRows | ConvertTo-Json -Depth 8
Write-Output "[saved] $Out"
