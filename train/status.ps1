# Training-loop status at a glance. Run from anywhere:
#   powershell -NoProfile -File "E:\Programing Projects\llm game try 3\train\status.ps1"
$proj = Split-Path $PSScriptRoot -Parent
$log = Join-Path $proj "train\expert_iter_v5.log"
$summary = Join-Path $proj "train\rounds\summary.jsonl"

$loop = Get-CimInstance Win32_Process -Filter "CommandLine like '%expert_iter%'" -ErrorAction SilentlyContinue
Write-Host ("loop running : " + [bool]$loop)
if (Test-Path $log) {
    $age = (Get-Date) - (Get-Item $log).LastWriteTime
    Write-Host ("log activity : last write {0:N0}s ago" -f $age.TotalSeconds)
    $raw = Get-Content $log -Tail 4 -ErrorAction SilentlyContinue
    $phase = ($raw -split "`r" | Where-Object { $_ -match "^>>>|%\||done: " } | Select-Object -Last 1)
    if ($phase) { Write-Host ("current phase: " + $phase.Trim().Substring(0, [Math]::Min(110, $phase.Trim().Length))) }
}
try { Write-Host ("gpu          : " + (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)) } catch {}
Write-Host ""
Write-Host "rounds so far:"
if (Test-Path $summary) {
    Get-Content $summary | ForEach-Object {
        $r = $_ | ConvertFrom-Json
        Write-Host ("  round {0}: {1,-22} mean={2,-7} survivors={3}/{4}" -f
            $r.round, $r.policy, $r.mean_return, ($r.npcs - $r.deaths), $r.npcs)
    }
} else { Write-Host "  (none yet)" }
Write-Host ""
Write-Host "stop cleanly : New-Item '$proj\train\rounds\STOP' -ItemType File"
