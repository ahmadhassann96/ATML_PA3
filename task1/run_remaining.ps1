# Runs every Task 1 step after training: diagnostics, qualitative generations, the 7 benchmark
# evaluations, then the report. Steps whose output already exists are skipped, so re-running after
# an interruption picks up where it stopped. Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File task1\run_remaining.ps1
# Set $env:PA3_PY to use a different Python (default: the PA2 venv used for all local runs).
# -Skip full_ft_l2sp,oplora16 leaves those evaluations out (e.g. when they run on Kaggle instead).
param([string[]]$Skip = @())
$Skip = @($Skip | ForEach-Object { $_ -split "," } | Where-Object { $_ })

$ErrorActionPreference = "Continue"
$py = $env:PA3_PY
if (-not $py) { $py = "D:\Junior\CS6304 - ATML\PA2\ATML-PA2-LLM-PostTraining\.venv\Scripts\python.exe" }
$env:HF_HUB_OFFLINE = "1"   # everything is cached by task1.prefetch; avoids network stalls
$out = "outputs\task1"
$conds = @("baseline", "full_ft@625", "full_ft", "full_ft_l2sp", "lora4", "lora16", "oplora16")

function Done($path, $marker) {
    # evaluate.py saves summary.json after GSM8K too; only the final save has the "primary" key
    (Test-Path $path) -and (-not $marker -or (Select-String -Path $path -Pattern $marker -Quiet))
}

function Step($name, $done, [string[]]$cmd, $marker = $null) {
    if (Done $done $marker) { Write-Host "== skip $name (found $done)"; return }
    Write-Host "== $name  [$(Get-Date -Format 'HH:mm')]"
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "!! $name failed (exit $LASTEXITCODE)" -ForegroundColor Red }
}

Step "diagnostics" "$out\diagnostics\diagnostics.csv" @("-m", "task1.diagnostics")

foreach ($c in $conds) {
    Step "qualitative $c" "$out\qualitative\$($c.Replace('@', '_at_')).jsonl" `
        @("-m", "task1.qualitative", "--yaml", "first_plot_questions.yaml", "--condition", $c)
}
& $py -m task1.qualitative --collect

foreach ($c in $conds) {
    if ($Skip -contains $c) { Write-Host "== skip evaluate $c (-Skip)"; continue }
    $sum = "$out\eval\$($c.Replace('@', '_at_'))\summary.json"
    $cmd = @("-m", "task1.evaluate", "--condition", $c)
    # GSM8K already finished but the log-likelihood tasks did not: keep GSM8K, run only the rest
    if ((Done $sum '"gsm8k_cot"') -and -not (Done $sum '"primary"')) { $cmd += @("--skip", "gsm8k_cot") }
    Step "evaluate $c" $sum $cmd '"primary"'
}

& $py -m task1.analyze
Write-Host "== all done  [$(Get-Date -Format 'HH:mm')]"
