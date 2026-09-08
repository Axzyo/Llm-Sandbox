# The training loop's control panel. One stable command per job — no per-run
# parameters, no version-numbered logs. Resume state is derived from disk:
# next round = last summary round + 1; policy = newest student-rN in Ollama.
#
#   powershell -NoProfile -File train\loop.ps1 start    # launch / resume (detached, survives closing everything)
#   powershell -NoProfile -File train\loop.ps1 stop     # finish current round, then exit cleanly
#   powershell -NoProfile -File train\loop.ps1 kill     # stop immediately (current round's work is lost)
#   powershell -NoProfile -File train\loop.ps1 report   # full progress report
#   powershell -NoProfile -File train\loop.ps1 plan     # show what `start` would do, without starting
param([Parameter(Position = 0)][string]$Command = "report")

$proj = Split-Path $PSScriptRoot -Parent
$roundsDir = Join-Path $proj "train\rounds"
$summary = Join-Path $roundsDir "summary.jsonl"
$log = Join-Path $proj "train\expert_iter.log"
$stopFile = Join-Path $roundsDir "STOP"

# fixed training configuration — edit here, never per launch
$episodes = 24; $npcs = 3; $budget = 240; $seed = 300; $top = 0.4; $temp = 0.7

function Get-LoopProcess {
    Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "expert_iter\.py" }
}

function Get-NextRound {
    if (Test-Path $summary) {
        $max = (Get-Content $summary | ForEach-Object { ($_ | ConvertFrom-Json).round } |
            Measure-Object -Maximum).Maximum
        return [int]$max + 1
    }
    return 1
}

function Get-InitModel {
    # the newest student generation is always the policy to resume with
    $best = -1; $name = $null
    try {
        (& ollama list) | ForEach-Object {
            if ($_ -match "^(student-g(\d+))") {
                $k = [int]$Matches[2]
                if ($k -gt $best) { $best = $k; $name = $Matches[1] }
            }
        }
    } catch {}
    return $name
}

function Get-Plan {
    $next = Get-NextRound
    $init = Get-InitModel
    $extra = if ($init) { " --init-model $init" } else { "" }
    @{
        Round  = $next
        Policy = if ($init) { $init } else { "gemma4 (baseline)" }
        Args   = "--rounds 0 --start-round $next --episodes $episodes --npcs $npcs " +
                 "--budget $budget --seed $seed --top $top --temperature $temp$extra"
    }
}

switch ($Command) {
    "start" {
        if (Get-LoopProcess) { Write-Host "already running - nothing started (use 'report')"; break }
        if (Test-Path $stopFile) { Remove-Item $stopFile }  # a stale STOP would end it instantly
        $plan = Get-Plan
        "`n===== launched $(Get-Date -Format u): round $($plan.Round) onward, policy $($plan.Policy) =====" |
            Out-File -Append $log
        $inner = "`$env:HF_HOME='E:\hf-cache'; python train\expert_iter.py $($plan.Args) *>> '$log'"
        Start-Process -WindowStyle Hidden -WorkingDirectory $proj powershell `
            -ArgumentList '-NoProfile', '-Command', $inner
        Write-Host ("started (detached): round {0} onward with {1}, endless until 'stop'" -f $plan.Round, $plan.Policy)
        Write-Host "log: $log"
    }
    "stop" {
        if (-not (Get-LoopProcess)) { Write-Host "not running - nothing to stop"; break }
        New-Item $stopFile -ItemType File -Force | Out-Null
        Write-Host "STOP placed - the loop finishes its current round (can be hours), then exits cleanly."
        Write-Host "To stop immediately instead: loop.ps1 kill"
    }
    "kill" {
        $p = Get-LoopProcess
        if (-not $p) { Write-Host "not running"; break }
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $($p.ProcessId)" -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Stop-Process -Id $p.ProcessId -Force
        Write-Host "killed. The interrupted round's rollout data is on disk; its training is lost (harmless - the next round retrains from the pool)."
    }
    { $_ -in "report", "status" } {
        $p = Get-LoopProcess
        Write-Host ("running      : " + [bool]$p)
        if (Test-Path $log) {
            $age = (Get-Date) - (Get-Item $log).LastWriteTime
            Write-Host ("log activity : last write {0:N0}s ago" -f $age.TotalSeconds)
            $tail = Get-Content $log -Tail 4 -ErrorAction SilentlyContinue
            $phase = ($tail -split "`r" | Where-Object { $_ -match "^>>>|%\||done: |STOP file" } | Select-Object -Last 1)
            if ($phase) {
                Write-Host ("current phase: " + $phase.Trim().Substring(0, [Math]::Min(110, $phase.Trim().Length)))
            }
        }
        try { Write-Host ("gpu          : " + (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)) } catch {}
        Write-Host ""
        if (Test-Path $summary) {
            Write-Host ("{0,-6}{1,-14}{2,7}{3,9}{4,8}{5,8}{6,8}{7,8}{8,8}" -f
                "round", "policy", "surv", "mean", "s=1", "s=.75", "s=.5", "s=.25", "wizard")
            Get-Content $summary | ForEach-Object {
                $r = $_ | ConvertFrom-Json
                $pp = $r.per_profile_return
                Write-Host ("{0,-6}{1,-14}{2,7}{3,9:N1}{4,8:N1}{5,8:N1}{6,8:N1}{7,8:N1}{8,8:N2}" -f
                    $r.round, $r.policy.Substring(0, [Math]::Min(13, $r.policy.Length)),
                    "$($r.npcs - $r.deaths)/$($r.npcs)", $r.mean_return,
                    $pp.'curiosity=0,survival=1', $pp.'curiosity=0.25,survival=0.75',
                    $pp.'curiosity=0.5,survival=0.5', $pp.'curiosity=0.75,survival=0.25',
                    $pp.'curiosity=1,survival=0')
            }
        } else { Write-Host "no rounds recorded yet" }
        Write-Host ""
        $plan = Get-Plan
        Write-Host ("next 'start' : round {0} with {1}" -f $plan.Round, $plan.Policy)
    }
    "plan" {
        $plan = Get-Plan
        Write-Host ("would start round {0} with policy {1}" -f $plan.Round, $plan.Policy)
        Write-Host ("args: " + $plan.Args)
    }
    default { Write-Host "usage: loop.ps1 start | stop | kill | report | plan" }
}
