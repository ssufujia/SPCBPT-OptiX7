[CmdletBinding()]
param(
    [string]$BuildDir = 'build/release-optix9',
    [string]$Scene = '',
    [string]$OutputDir = 'build/optimal-e-bakeoff-bedroom',
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$TorchDevice = 'cuda',
    [ValidateRange(1, 10000)]
    [int]$PyTorchSteps = 20,
    [ValidateScript({
        [double]::IsFinite($_) -and [double]$_ -gt 0.0
    })]
    [double]$CudaLearningRate = 1.0,
    [string]$PythonCommand = 'python',
    [ValidateRange(0, 9)]
    [int]$StartIndex = 0,
    [ValidateRange(1, 10)]
    [int]$Count = 10,
    [switch]$Force
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

$cameras = @(
    [ordered]@{ index = 0; name = 'base'; eye = '-0.913696,1.740873,2.264472'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 1; name = 'yaw_m12'; eye = '-1.184834,1.740873,2.346306'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 2; name = 'yaw_p12'; eye = '-0.665497,1.740873,2.128054'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 3; name = 'yaw_m24'; eye = '-1.467061,1.740873,2.369979'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 4; name = 'yaw_p24'; eye = '-0.451086,1.740873,1.943013'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 5; name = 'pitch_m10'; eye = '-0.888442,1.498121,2.324565'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 6; name = 'pitch_p10'; eye = '-0.954898,1.968618,2.166431'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 7; name = 'dolly_in'; eye = '-0.992426,1.666788,2.077132'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 8; name = 'dolly_out'; eye = '-0.834966,1.814958,2.451812'; lookat = '-1.438563,1.246972,1.015537' }
    [ordered]@{ index = 9; name = 'yaw_p18_pitch_p7'; eye = '-0.599366,1.902293,1.987939'; lookat = '-1.438563,1.246972,1.015537' }
)
$cameraUp = '0,1,0'
$cameraFov = 45

$endIndex = [Math]::Min($StartIndex + $Count, $cameras.Count)
$resolvedOutputDir = Resolve-ProjectPath -Path $OutputDir
$validationScript = Join-Path $PSScriptRoot 'validate_optimal_e.ps1'
New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
$resolvedScene = if ($Scene) {
    Resolve-ProjectPath -Path $Scene
} else {
    'bedroom.scene (default)'
}

$manifest = [ordered]@{
    schema_version = 2
    scene = $resolvedScene
    pytorch_steps = $PyTorchSteps
    cuda_learning_rate = $CudaLearningRate
    up = $cameraUp
    fov = $cameraFov
    torch_device = $TorchDevice
    cameras = $cameras
}
$manifestPath = Join-Path $resolvedOutputDir 'cameras.json'
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    $existingManifest = Get-Content -LiteralPath $manifestPath -Raw |
        ConvertFrom-Json
    $existingIdentity = $existingManifest | ConvertTo-Json -Depth 6 -Compress
    $requestedIdentity = $manifest | ConvertTo-Json -Depth 6 -Compress
    if ($existingIdentity -ne $requestedIdentity) {
        throw (
            'OutputDir contains an incompatible experiment manifest. ' +
            'Use a new -OutputDir; -Force only reruns cameras within the same manifest.'
        )
    }
} else {
    $orphanedSummaries = Get-ChildItem -LiteralPath $resolvedOutputDir -Directory |
        Where-Object {
            Test-Path -LiteralPath (Join-Path $_.FullName 'summary.json') -PathType Leaf
        }
    if ($orphanedSummaries) {
        throw 'OutputDir contains summaries but no cameras.json manifest.'
    }
    [System.IO.File]::WriteAllText(
        $manifestPath,
        (($manifest | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false)
    )
}

$resolvedSummaryDevice = $null

function Read-ValidatedSummary {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Camera,
        [Parameter(Mandatory)][string]$SummaryPath
    )

    $summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if ([int]$summary.schema_version -ne 4 -or
        $summary.scene -ne $resolvedScene -or
        $summary.eye -ne $Camera.eye -or
        $summary.lookat -ne $Camera.lookat -or
        $summary.up -ne $cameraUp -or
        [double]$summary.fov -ne $cameraFov -or
        [double]$summary.cuda_learning_rate -ne $CudaLearningRate) {
        throw "Summary identity does not match the manifest: $SummaryPath"
    }
    if ([int]$summary.reference.pytorch.steps -ne $PyTorchSteps) {
        throw "Summary PyTorch step count does not match the manifest: $SummaryPath"
    }
    $actualDevice = [string]$summary.reference.device
    if (-not $actualDevice) {
        throw "Summary is missing the resolved torch device: $SummaryPath"
    }
    if ($TorchDevice -eq 'auto') {
        if ($null -eq $script:resolvedSummaryDevice) {
            $script:resolvedSummaryDevice = $actualDevice
        } elseif ($actualDevice -ne $script:resolvedSummaryDevice) {
            throw (
                'Summaries resolved -TorchDevice auto to different devices: ' +
                "$($script:resolvedSummaryDevice) and $actualDevice"
            )
        }
    } elseif ($actualDevice -ne $TorchDevice) {
        throw "Summary torch device does not match the manifest: $SummaryPath"
    }
    foreach ($metricName in @(
        'cuda_initial_objective',
        'cuda_final_objective',
        'accepted_steps',
        'initial_objective_abs_error',
        'gradient_max_abs_error',
        'final_objective_abs_error'
    )) {
        if ($null -eq $summary.reference.cross_check.$metricName) {
            throw "Summary is missing cross-check metric '$metricName': $SummaryPath"
        }
    }
    foreach ($metricName in @('initial_loss', 'final_loss')) {
        if ($null -eq $summary.reference.pytorch.$metricName) {
            throw "Summary is missing PyTorch metric '$metricName': $SummaryPath"
        }
    }
    foreach ($metricName in @(
        'pytorch_final_objective',
        'cuda_objective',
        'absolute_error'
    )) {
        if ($null -eq $summary.reverse_cross_check.$metricName) {
            throw "Summary is missing reverse metric '$metricName': $SummaryPath"
        }
    }
    return $summary
}

foreach ($camera in $cameras) {
    $runName = '{0:D2}-{1}' -f $camera.index, $camera.name
    $summaryPath = Join-Path (Join-Path $resolvedOutputDir $runName) 'summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $null = Read-ValidatedSummary -Camera $camera -SummaryPath $summaryPath
    }
}

foreach ($camera in $cameras[$StartIndex..($endIndex - 1)]) {
    $runName = '{0:D2}-{1}' -f $camera.index, $camera.name
    $runDir = Join-Path $resolvedOutputDir $runName
    $summaryPath = Join-Path $runDir 'summary.json'
    if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and -not $Force) {
        $null = Read-ValidatedSummary -Camera $camera -SummaryPath $summaryPath
        Write-Host "SKIP $runName (summary exists)"
        continue
    }

    Write-Host "RUN $runName eye=$($camera.eye) lookat=$($camera.lookat)"
    $validationArguments = @{
        BuildDir = $BuildDir
        Scene = $Scene
        Eye = $camera.eye
        Lookat = $camera.lookat
        Up = $cameraUp
        Fov = $cameraFov
        OutputDir = $runDir
        TorchDevice = $TorchDevice
        PyTorchSteps = $PyTorchSteps
        CudaLearningRate = $CudaLearningRate
        PythonCommand = $PythonCommand
    }
    & $validationScript @validationArguments
    $null = Read-ValidatedSummary -Camera $camera -SummaryPath $summaryPath
}

$rows = foreach ($camera in $cameras) {
    $runName = '{0:D2}-{1}' -f $camera.index, $camera.name
    $summaryPath = Join-Path (Join-Path $resolvedOutputDir $runName) 'summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        continue
    }
    $summary = Read-ValidatedSummary -Camera $camera -SummaryPath $summaryPath
    $crossCheck = $summary.reference.cross_check
    $pytorch = $summary.reference.pytorch
    [pscustomobject][ordered]@{
        index = $camera.index
        camera = $camera.name
        cuda_initial_loss = [double]$crossCheck.cuda_initial_objective
        cuda_final_loss = [double]$crossCheck.cuda_final_objective
        cuda_improvement = [double]$crossCheck.cuda_initial_objective - [double]$crossCheck.cuda_final_objective
        cuda_accepted_steps = [int]$crossCheck.accepted_steps
        pytorch_initial_loss = [double]$pytorch.initial_loss
        pytorch_final_loss = [double]$pytorch.final_loss
        pytorch_final_cuda_loss = [double]$summary.reverse_cross_check.cuda_objective
        pytorch_improvement = [double]$pytorch.initial_loss - [double]$pytorch.final_loss
        pytorch_reverse_abs_error = [double]$summary.reverse_cross_check.absolute_error
        objective_abs_error = [double]$crossCheck.initial_objective_abs_error
        gradient_max_abs_error = [double]$crossCheck.gradient_max_abs_error
        cuda_final_abs_error = [double]$crossCheck.final_objective_abs_error
        capture_seconds = [double]$summary.timings_seconds.capture
        cuda_seconds = [double]$summary.timings_seconds.cuda
        pytorch_seconds = [double]$summary.timings_seconds.pytorch
        cuda_reverse_seconds = [double]$summary.timings_seconds.cuda_reverse
    }
}

$results = [ordered]@{
    schema_version = 3
    completed = @($rows).Count
    expected = $cameras.Count
    cuda_learning_rate = $CudaLearningRate
    rows = @($rows)
}
[System.IO.File]::WriteAllText(
    (Join-Path $resolvedOutputDir 'results.json'),
    (($results | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
@($rows) | Export-Csv -LiteralPath (Join-Path $resolvedOutputDir 'results.csv') -NoTypeInformation
@($rows) | Format-Table index, camera, cuda_final_loss, pytorch_final_loss, pytorch_reverse_abs_error -AutoSize
Write-Host "OPTIMAL_E_BAKEOFF_PROGRESS: $(@($rows).Count)/$($cameras.Count) in $resolvedOutputDir"
