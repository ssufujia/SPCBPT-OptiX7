[CmdletBinding()]
param(
    [string]$BuildDir = 'build/release-optix9',
    [string]$Scene = '',
    [string]$Eye = '',
    [string]$Lookat = '',
    [string]$Up = '',
    [double]$Fov = 0,
    [uint32]$ExperimentSeed = 0,
    [string]$OutputDir = 'build/optimal-e-validation',
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$TorchDevice = 'auto',
    [ValidateRange(1, 10000)]
    [int]$PyTorchSteps = 20,
    [ValidateScript({
        [double]::IsFinite($_) -and [double]$_ -gt 0.0
    })]
    [double]$CudaLearningRate = 1.0,
    [string]$PythonCommand = 'python',
    [string]$ProvenanceId = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine($PSScriptRoot, '..')
)

function Resolve-ProjectPath {
    param([Parameter(Mandatory)][string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($repoRoot, $Path)
    )
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    & $FilePath @Arguments
    $nativeExitCode = $LASTEXITCODE
    if ($nativeExitCode -ne 0) {
        throw "$FilePath failed with exit code $nativeExitCode"
    }
}

$resolvedBuildDir = Resolve-ProjectPath -Path $BuildDir
$resolvedOutputDir = Resolve-ProjectPath -Path $OutputDir
$activeEnvironmentPython = if ($env:VIRTUAL_ENV) {
    Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
}
if ($PythonCommand -eq 'python' -and
    $activeEnvironmentPython -and
    (Test-Path -LiteralPath $activeEnvironmentPython -PathType Leaf)) {
    $PythonCommand = $activeEnvironmentPython
}
if ($PythonCommand -eq 'python') {
    $pythonCandidates = Get-Command python.exe -All -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandType -eq 'Application' -and
            $_.Path -notlike '*\WindowsApps\python.exe'
        } |
        Select-Object -ExpandProperty Path -Unique
    foreach ($candidate in $pythonCandidates) {
        & $candidate -c 'import numpy, torch' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonCommand = $candidate
            break
        }
    }
    if ($PythonCommand -eq 'python') {
        throw 'No Python interpreter with NumPy and PyTorch was found. Use -PythonCommand.'
    }
}
$smokeExe = Join-Path $resolvedBuildDir 'bin/spcbpt_smoke.exe'
$cudaValidatorExe = Join-Path $resolvedBuildDir 'bin/spcbpt_optimal_e_optimizer_test.exe'
if (-not (Test-Path -LiteralPath $smokeExe -PathType Leaf)) {
    throw "Missing $smokeExe. Build spcbpt_smoke first."
}
if (-not (Test-Path -LiteralPath $cudaValidatorExe -PathType Leaf)) {
    throw "Missing $cudaValidatorExe. Build spcbpt_optimal_e_optimizer_test first."
}

New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
$snapshotPath = Join-Path $resolvedOutputDir 'problem.spcoe'
$cudaResultPath = Join-Path $resolvedOutputDir 'cuda.spcor'
$pytorchOutputPath = Join-Path $resolvedOutputDir 'pytorch_base_q.npy'
$pytorchRawOutputPath = Join-Path $resolvedOutputDir 'pytorch_base_q.f32'
$pytorchCudaObjectivePath = Join-Path $resolvedOutputDir 'pytorch_cuda_objective.txt'
$referenceMetricsPath = Join-Path $resolvedOutputDir 'reference_metrics.json'
$summaryPath = Join-Path $resolvedOutputDir 'summary.json'

$captureArgs = @("--export-optimal-e=$snapshotPath")
if ($Scene) {
    $captureArgs += "--scene=$(Resolve-ProjectPath -Path $Scene)"
}
$cameraArgumentCount = @($Eye, $Lookat, $Up) |
    Where-Object { [bool]$_ } |
    Measure-Object |
    Select-Object -ExpandProperty Count
if (($cameraArgumentCount -ne 0 -or $Fov -ne 0) -and
    ($cameraArgumentCount -ne 3 -or $Fov -eq 0)) {
    throw '-Eye, -Lookat, -Up and -Fov must be provided together.'
}
if ($Eye) {
    $captureArgs += "--eye=$Eye"
    $captureArgs += "--lookat=$Lookat"
    $captureArgs += "--up=$Up"
    $captureArgs += "--fov=$($Fov.ToString([System.Globalization.CultureInfo]::InvariantCulture))"
}
$captureArgs += "--experiment-seed=$ExperimentSeed"
$captureTimer = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-NativeChecked -FilePath $smokeExe -Arguments $captureArgs
$captureTimer.Stop()

$cudaArgs = @(
    "--snapshot=$snapshotPath"
    "--result=$cudaResultPath"
    "--learning-rate=$($CudaLearningRate.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture))"
)
$cudaTimer = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-NativeChecked -FilePath $cudaValidatorExe -Arguments $cudaArgs
$cudaTimer.Stop()

$pythonArgs = @(
    (Join-Path $repoRoot 'scripts/optimal_e_reference.py')
    '--input'
    $snapshotPath
    '--cuda-result'
    $cudaResultPath
    '--output'
    $pytorchOutputPath
    '--raw-output'
    $pytorchRawOutputPath
    '--metrics-output'
    $referenceMetricsPath
    '--steps'
    $PyTorchSteps.ToString()
    '--device'
    $TorchDevice
)
$pytorchTimer = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-NativeChecked -FilePath $PythonCommand -Arguments $pythonArgs
$pytorchTimer.Stop()

$referenceMetrics = Get-Content -LiteralPath $referenceMetricsPath -Raw |
    ConvertFrom-Json
if ([uint32]$referenceMetrics.experiment_seed -ne $ExperimentSeed) {
    throw (
        'Snapshot experiment seed does not match the requested seed: ' +
        "$($referenceMetrics.experiment_seed) != $ExperimentSeed"
    )
}
$reverseCudaArgs = @(
    "--snapshot=$snapshotPath"
    "--result=$cudaResultPath"
    "--candidate-q=$pytorchRawOutputPath"
    "--candidate-objective=$pytorchCudaObjectivePath"
    "--learning-rate=$($CudaLearningRate.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture))"
)
$reverseCudaTimer = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-NativeChecked -FilePath $cudaValidatorExe -Arguments $reverseCudaArgs
$reverseCudaTimer.Stop()
$pytorchCudaObjective = [double]::Parse(
    (Get-Content -LiteralPath $pytorchCudaObjectivePath -Raw).Trim(),
    [System.Globalization.CultureInfo]::InvariantCulture
)
$pytorchFinalLoss = [double]$referenceMetrics.pytorch.final_loss
$reverseAbsoluteError = [Math]::Abs(
    $pytorchCudaObjective - $pytorchFinalLoss
)
$reverseTolerance = 1e-4 + 2e-5 * [Math]::Abs($pytorchFinalLoss)
if ($reverseAbsoluteError -gt $reverseTolerance) {
    throw (
        'CUDA objective disagrees with PyTorch-final q: ' +
        "abs_error=$reverseAbsoluteError tolerance=$reverseTolerance"
    )
}
$resolvedScene = if ($Scene) {
    Resolve-ProjectPath -Path $Scene
} else {
    'bedroom.scene (default)'
}
$summary = [ordered]@{
    schema_version = 4
    scene = $resolvedScene
    eye = $Eye
    lookat = $Lookat
    up = $Up
    fov = $Fov
    experiment_seed = $ExperimentSeed
    provenance_id = $ProvenanceId
    cuda_learning_rate = $CudaLearningRate
    timings_seconds = [ordered]@{
        capture = $captureTimer.Elapsed.TotalSeconds
        cuda = $cudaTimer.Elapsed.TotalSeconds
        pytorch = $pytorchTimer.Elapsed.TotalSeconds
        cuda_reverse = $reverseCudaTimer.Elapsed.TotalSeconds
    }
    reference = $referenceMetrics
    reverse_cross_check = [ordered]@{
        pytorch_final_objective = $pytorchFinalLoss
        cuda_objective = $pytorchCudaObjective
        absolute_error = $reverseAbsoluteError
    }
}
[System.IO.File]::WriteAllText(
    $summaryPath,
    (($summary | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host "OPTIMAL_E_VALIDATION_OK: $resolvedOutputDir"
