# Run from VS Code PowerShell in the CURRENT, post-audit merit-covid-ct repo.
# This script does not change source code, config, split, or original datasets.
param(
    [ValidateSet('Info','Tests','QuickTests','Split','SampleQC','SmokeGPU','PilotGPU','AllSafe')]
    [string]$Step = 'Tests',
    [string]$RepoPath = '.',
    [switch]$AssumeMillimeters
)

$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'

try { $repo = (Resolve-Path -LiteralPath $RepoPath -ErrorAction Stop).Path }
catch { throw "Repository folder not found: $RepoPath" }
if (-not (Test-Path (Join-Path $repo 'pyproject.toml'))) {
    throw "No pyproject.toml in $repo. Open the merit-covid-ct repository root."
}
Set-Location -LiteralPath $repo

$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
if (Test-Path $venvPython) { $script:PythonExe = $venvPython }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $script:PythonExe = 'python' }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $script:PythonExe = 'py' }
else { throw 'Python not found. Activate your environment or install Python.' }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$script:LogFolder = Join-Path $repo "outputs\manual_review_$stamp"
New-Item -ItemType Directory -Path $script:LogFolder -Force | Out-Null

function Invoke-Python {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $script:LogFolder "$Name.log"
    Write-Host "`n>>> $Name" -ForegroundColor Cyan
    Write-Host "Command: $script:PythonExe $($Arguments -join ' ')"
    # Windows PowerShell 5.1 may wrap native stderr as non-terminating ErrorRecords.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $script:PythonExe @Arguments 2>&1 | Tee-Object -FilePath $log | Out-Host
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    Write-Host "Exit code: $code | Log: $log" -ForegroundColor $(if ($code -eq 0) {'Green'} else {'Red'})
    return $code
}

function Show-Info {
    Write-Host "`nRepository: $repo"
    Write-Host "Logs: $script:LogFolder"
    Write-Host "Python: $script:PythonExe"
    & $script:PythonExe --version
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host 'Git HEAD:'; git rev-parse HEAD
        Write-Host 'Uncommitted changes:'; git status --short
    }
    $newFile = 'tests/test_merit_audit_options.py'
    if (Test-Path $newFile) {
        Write-Host "Post-Codex marker present: $newFile" -ForegroundColor Green
    } else {
        Write-Host "WARNING: $newFile is missing. You might have opened the OLD ZIP, not the post-Codex repo." -ForegroundColor Yellow
    }
    if (Test-Path 'data/covid19_20/cases.json') {
        Write-Host 'Dataset manifest found: data/covid19_20/cases.json' -ForegroundColor Green
    } else {
        Write-Host 'Dataset manifest missing: data/covid19_20/cases.json' -ForegroundColor Yellow
    }
}

function Assert-Data {
    if (-not (Test-Path 'data/covid19_20/cases.json')) {
        Write-Host 'BLOCKED: data/covid19_20/cases.json missing. Check the dataset paths first.' -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Run-Tests {
    return (Invoke-Python -Name 'full_pytest_first_error' -Arguments @('-m','pytest','-q','-x','--tb=short'))
}
function Run-QuickTests {
    $names = @('tests/test_merit_unetrpp.py','tests/test_merit_pipeline.py','tests/test_preprocessing.py','tests/test_reconstruction.py','tests/test_splitting.py','tests/test_merit_audit_options.py')
    foreach ($name in $names) {
        if (-not (Test-Path $name)) { throw "Missing test: $name. Is this the post-Codex version?" }
    }
    return (Invoke-Python -Name 'selected_pytest' -Arguments (@('-m','pytest') + $names + @('-q','--tb=short')))
}
function Run-Split {
    return (Invoke-Python -Name 'check_split' -Arguments @('scripts/merit_pipeline.py','check-split','--config','configs/merit_index.yaml'))
}
function Run-SampleQC {
    if (-not (Assert-Data)) { return 2 }
    $cmdArgs = @('scripts/merit_preflight.py','inspect','--config','configs/merit_index.yaml','--fold','0','--sizes','128','256','--block-depth','8','--limit','3','--output', (Join-Path $script:LogFolder 'sample_qc'))
    if ($AssumeMillimeters) {
        Write-Host 'WARNING: --assume-mm is an explicit assumption, not proof of real-world units.' -ForegroundColor Yellow
        $cmdArgs += '--assume-mm'
    }
    return (Invoke-Python -Name 'sample_qc' -Arguments $cmdArgs)
}
function Run-SmokeGPU {
    return (Invoke-Python -Name 'cuda_smoke' -Arguments @('scripts/merit_pipeline.py','smoke','--device','cuda','--output',(Join-Path $script:LogFolder 'cuda_smoke.json')))
}
function Run-PilotGPU {
    if (-not (Assert-Data)) { return 2 }
    return (Invoke-Python -Name 'real_ct_pilot' -Arguments @('scripts/merit_preflight.py','pilot','--config','configs/merit_index.yaml','--fold','0','--case-id','volume-covid19-A-0025','--size','128','--steps','4','--device','cuda','--output',(Join-Path $script:LogFolder 'real_ct_pilot')))
}

Show-Info
$exitCode = 0
switch ($Step) {
    'Info'       { $exitCode = 0 }
    'Tests'      { $exitCode = Run-Tests }
    'QuickTests' { $exitCode = Run-QuickTests }
    'Split'      { $exitCode = Run-Split }
    'SampleQC'   { $exitCode = Run-SampleQC }
    'SmokeGPU'   { $exitCode = Run-SmokeGPU }
    'PilotGPU'   { $exitCode = Run-PilotGPU }
    'AllSafe' {
        $exitCode = Run-Tests
        if ($exitCode -eq 0) { $exitCode = Run-Split }
        if ($exitCode -eq 0) { $exitCode = Run-SampleQC }
    }
}
Write-Host "`nLog folder: $script:LogFolder" -ForegroundColor Cyan
if ($exitCode -ne 0) {
    Write-Host 'Not PASS. Send the relevant .log file or paste the first traceback.' -ForegroundColor Yellow
}
exit $exitCode
