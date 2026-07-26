[CmdletBinding()]
param(
    [string]$BuildDir = 'build/release-optix9',
    [string]$OutputDir = 'build/experiments/optimal-e-multiscene-v4',
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$TorchDevice = 'cuda',
    [ValidateRange(1, 10000)]
    [int]$PyTorchSteps = 20,
    [ValidateScript({
        [double]::IsFinite($_) -and [double]$_ -gt 0.0
    })]
    [double]$CudaLearningRate = 1.0,
    [string]$PythonCommand = 'python',
    [ValidateRange(0, 113)]
    [int]$StartIndex = 0,
    [ValidateRange(1, 114)]
    [int]$Count = 114,
    [switch]$SmokeOnly,
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

function Get-FileProvenance {
    param([Parameter(Mandatory)][string]$Path)
    $resolvedPath = Resolve-ProjectPath -Path $Path
    $item = Get-Item -LiteralPath $resolvedPath -ErrorAction Stop
    return [ordered]@{
        path = [System.IO.Path]::GetRelativePath($repoRoot, $resolvedPath).Replace('\', '/')
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedPath).Hash.ToLowerInvariant()
        length = $item.Length
    }
}

function Get-DirectoryTreeProvenance {
    param([Parameter(Mandatory)][string]$Path)
    $resolvedPath = Resolve-ProjectPath -Path $Path
    $root = Get-Item -LiteralPath $resolvedPath -ErrorAction Stop
    if (-not $root.PSIsContainer) {
        throw "Provenance root is not a directory: $resolvedPath"
    }

    $files = [System.Collections.Generic.SortedDictionary[
        string,
        System.IO.FileInfo
    ]]::new([System.StringComparer]::Ordinal)
    foreach ($file in Get-ChildItem -LiteralPath $resolvedPath -Recurse -File) {
        $relativePath = [System.IO.Path]::GetRelativePath(
            $resolvedPath,
            $file.FullName
        ).Replace('\', '/')
        $files.Add($relativePath, $file)
    }
    $identity = [System.Text.StringBuilder]::new()
    [long]$totalLength = 0
    foreach ($entry in $files.GetEnumerator()) {
        $relativePath = $entry.Key
        $file = $entry.Value
        $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).
            Hash.ToLowerInvariant()
        $null = $identity.Append($relativePath.Length).Append(':').
            Append($relativePath).Append(':').
            Append($file.Length).Append(':').
            Append($sha256).Append("`n")
        $totalLength += $file.Length
    }
    $identityBytes = [System.Text.Encoding]::UTF8.GetBytes(
        $identity.ToString()
    )
    return [ordered]@{
        path = [System.IO.Path]::GetRelativePath(
            $repoRoot,
            $resolvedPath
        ).Replace('\', '/')
        file_count = $files.Count
        total_length = $totalLength
        sha256 = [System.Convert]::ToHexString(
            [System.Security.Cryptography.SHA256]::HashData($identityBytes)
        ).ToLowerInvariant()
    }
}

function Invoke-CapturedProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$StandardOutputPath,
        [Parameter(Mandatory)][string]$StandardErrorPath
    )
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        $startInfo.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw "Failed to start $FilePath"
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        [System.IO.File]::WriteAllText(
            $StandardOutputPath,
            $stdoutTask.GetAwaiter().GetResult(),
            [System.Text.UTF8Encoding]::new($false)
        )
        [System.IO.File]::WriteAllText(
            $StandardErrorPath,
            $stderrTask.GetAwaiter().GetResult(),
            [System.Text.UTF8Encoding]::new($false)
        )
        return $process.ExitCode
    } finally {
        $process.Dispose()
    }
}

function New-Camera {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Eye,
        [Parameter(Mandatory)][string]$Lookat,
        [Parameter(Mandatory)][string]$Up,
        [Parameter(Mandatory)][double]$Fov,
        [Parameter(Mandatory)][string]$Source
    )
    return [ordered]@{
        name = $Name
        eye = $Eye
        lookat = $Lookat
        up = $Up
        fov = $Fov
        source = $Source
    }
}

$scenes = @(
    [ordered]@{
        id = 'bedroom'
        requested_scene = 'assets/bedroom.scene'
        actual_scene = 'assets/bedroom.scene'
        cameras = @(
            (New-Camera base '-0.913696,1.740873,2.264472' '-1.438563,1.246972,1.015537' '0,1,0' 45 authored)
            (New-Camera authored_alt_1 '1.2,1.41990197,2.72466350' '0.623937607,1.21846092,1.41820455' '0,1,0' 45 authored_commented)
            (New-Camera authored_alt_2 '2.653059,0.733602,1.036655' '2.555092,0.751108,1.026866' '0,1,0' 45 authored_commented)
        )
    }
    [ordered]@{
        id = 'white_room'
        requested_scene = 'assets/white-room/white-room-obj.scene'
        actual_scene = 'assets/white-room/white-room-obj.scene'
        cameras = @(
            (New-Camera base '2.8,2.0,6.9' '0.0,1.4,2' '0,1,0' 35 authored)
            (New-Camera yaw_m6 '2.8,2.0,6.9' '0.527528,1.4,1.734163' '0,1,0' 35 generated_yaw)
            (New-Camera yaw_p12 '2.8,2.0,6.9' '-0.957581,1.4,2.689229' '0,1,0' 35 generated_yaw)
        )
    }
    [ordered]@{
        id = 'showcase'
        requested_scene = 'assets/showcase/showcase.scene'
        actual_scene = 'assets/showcase/showcase.scene'
        cameras = @(
            (New-Camera base '0.102131,0.077978,0.027743' '-24.710585,0.078562,0.027675' '0,1,0' 20 authored)
            (New-Camera authored_overview '-0.309680,1.189867,-7.197319' '0,1,0' '0,1,0' 20 authored_commented)
        )
    }
    [ordered]@{
        id = 'cornell'
        requested_scene = 'assets/cornell_box/cornell.scene'
        actual_scene = 'assets/cornell_box/cornell.scene'
        cameras = @(
            (New-Camera base '2.78,2.79,-8.70953491' '2.78,2.79,-8.70753516' '0,1,0' 35 authored)
            (New-Camera yaw_p12 '2.78,2.79,-8.70953491' '2.780416,2.79,-8.707579' '0,1,0' 35 generated_yaw)
        )
    }
    [ordered]@{
        id = 'conference'
        requested_scene = 'assets/conference/conference2.scene'
        actual_scene = 'assets/conference/conference.scene'
        cameras = @(
            (New-Camera base '11.005541,2.400580,5.180362' '10.930385,2.395472,5.114594' '0,1,0' 50 authored)
            (New-Camera authored_conference3 '7.492245,3.657255,-1.611869' '7.400266,3.650079,-1.573288' '0,1,0' 50 authored_sibling)
            (New-Camera yaw_m12 '11.005541,2.400580,5.180362' '10.945701,2.395472,5.100405' '0,1,0' 50 generated_yaw)
            (New-Camera yaw_p12 '11.005541,2.400580,5.180362' '10.918353,2.395472,5.131657' '0,1,0' 50 generated_yaw)
        )
    }
    [ordered]@{
        id = 'glassroom'
        requested_scene = 'assets/glassroom/glassroom_project_final.scene'
        actual_scene = 'assets/glassroom/glassroom_project_final.scene'
        cameras = @(
            (New-Camera base '-47.900162,2.651114,17.076302' '-46.219166,2.541034,18.955288' '0,1,0' 45 authored)
            (New-Camera yaw_m12 '-47.900162,2.651114,17.076302' '-46.646563,2.541034,19.263726' '0,1,0' 45 generated_yaw)
            (New-Camera authored_alt_2 '-48.133595,2.471888,16.947020' '-45.965446,2.312077,19.091230' '0,1,0' 45 authored_commented)
            (New-Camera authored_alt_3 '-48.243561,3.094823,16.479696' '-46.303257,2.736173,18.432236' '0,1,0' 45 authored_commented)
            (New-Camera authored_alt_4 '-48.345745,3.020571,16.293541' '-46.313599,2.975809,18.458923' '0,1,0' 45 authored_commented)
        )
    }
    [ordered]@{
        id = 'bathroom'
        requested_scene = 'assets/bathroom_b/scene_v4.scene'
        actual_scene = 'assets/bathroom_b/scene_v4.scene'
        cameras = @(
            (New-Camera base '-0.318746,2.204264,-1.649909' '-2.179732,-1.415354,-2.400182' '0,1,0' 60 authored)
            (New-Camera authored_alt '-0.392655,2.051186,-1.729190' '-2.608369,-0.913501,-2.404806' '0,1,0' 60 authored_commented)
            (New-Camera yaw_m12 '-0.318746,2.204264,-1.649909' '-1.983074,-1.415354,-2.770707' '0,1,0' 60 generated_yaw)
            (New-Camera yaw_p12 '-0.318746,2.204264,-1.649909' '-2.295056,-1.415354,-1.996866' '0,1,0' 60 generated_yaw)
        )
    }
    [ordered]@{
        id = 'breakfast'
        requested_scene = 'assets/breafast_2.0/breafast_final.scene'
        actual_scene = 'assets/breafast_2.0/breafast_final.scene'
        cameras = @(
            (New-Camera base '-0.623726,-6.587055,1.204726' '-0.623726,-5.587055,1.204728' '0,0,1' 35 authored)
            (New-Camera authored_alt '-0.624943,-6.584619,1.134983' '-0.623726,-5.587055,1.204728' '0,0,1' 35 authored_commented)
            (New-Camera yaw_p12 '-0.623726,-6.587055,1.204726' '-0.831638,-5.608907,1.204728' '0,0,1' 35 generated_yaw)
        )
    }
    [ordered]@{
        id = 'projector'
        requested_scene = 'assets/projector/projector1.scene'
        actual_scene = 'assets/projector/projector1.scene'
        cameras = @(
            (New-Camera base '1.888024,0.870315,2.298156' '1.883880,0.855320,2.199373' '0,1,0' 65 authored)
            (New-Camera yaw_m12 '1.888024,0.870315,2.298156' '1.904509,0.855320,2.200670' '0,1,0' 65 generated_yaw)
            (New-Camera yaw_p12 '1.888024,0.870315,2.298156' '1.863432,0.855320,2.202393' '0,1,0' 65 generated_yaw)
        )
    }
    [ordered]@{
        id = 'kitchen'
        requested_scene = 'assets/kitchen/kitchen_final.scene'
        actual_scene = 'assets/kitchen/kitchen_final.scene'
        cameras = @(
            (New-Camera base '0.483840,1.575793,2.749950' '0.443696,1.576540,2.658364' '0,1,0' 45 authored)
            (New-Camera yaw_m12 '0.483840,1.575793,2.749950' '0.463615,1.576540,2.652019' '0,1,0' 45 generated_yaw)
            (New-Camera yaw_p12 '0.483840,1.575793,2.749950' '0.425531,1.576540,2.668712' '0,1,0' 45 generated_yaw)
            (New-Camera pitch_m8 '0.483840,1.575793,2.749950' '0.444045,1.562616,2.659160' '0,1,0' 45 generated_pitch)
            (New-Camera pitch_p8 '0.483840,1.575793,2.749950' '0.444128,1.590450,2.659351' '0,1,0' 45 generated_pitch)
        )
    }
    [ordered]@{
        id = 'hallway'
        requested_scene = 'assets/hallway/hallway-teaser_final.scene'
        actual_scene = 'assets/hallway/hallway-teaser.scene'
        fallback_reason = 'final reproducibly triggers CUDA illegal memory access; stable teaser retained'
        quarantined_final_dependencies = @(
            'assets/hallway/geometry/newLight'
            'assets/hallway/geometry/lens2.obj'
            'assets/hallway/geometry/patch.obj'
            'assets/envmap/cloudy_sky.hdr'
        )
        cameras = @(
            (New-Camera base '1.888024,0.870315,2.298156' '1.852158,0.851879,2.206647' '0,1,0' 65 authored)
            (New-Camera yaw_m12 '1.888024,0.870315,2.298156' '1.871968,0.851879,2.201190' '0,1,0' 65 generated_yaw)
            (New-Camera yaw_p12 '1.888024,0.870315,2.298156' '1.833916,0.851879,2.216104' '0,1,0' 65 generated_yaw)
            (New-Camera pitch_p8 '1.888024,0.870315,2.298156' '1.851571,0.865737,2.205149' '0,1,0' 65 generated_pitch)
        )
    }
)

$seeds = @(11, 29, 47)
$cameraCount = @($scenes | ForEach-Object { $_.cameras }).Count
if ($scenes.Count -ne 11 -or $cameraCount -ne 38) {
    throw "Internal experiment definition mismatch: $($scenes.Count) scenes, $cameraCount cameras"
}

$runs = @()
$runIndex = 0
foreach ($scene in $scenes) {
    for ($cameraIndex = 0; $cameraIndex -lt $scene.cameras.Count; ++$cameraIndex) {
        $camera = $scene.cameras[$cameraIndex]
        foreach ($seed in $seeds) {
            $runs += [ordered]@{
                index = $runIndex++
                scene_id = $scene.id
                scene = $scene.actual_scene
                camera_index = $cameraIndex
                camera = $camera.name
                eye = $camera.eye
                lookat = $camera.lookat
                up = $camera.up
                fov = $camera.fov
                experiment_seed = $seed
            }
        }
    }
}
if ($runs.Count -ne 114) {
    throw "Internal run count mismatch: $($runs.Count)"
}

$resolvedBuildDir = Resolve-ProjectPath -Path $BuildDir
$resolvedOutputDir = Resolve-ProjectPath -Path $OutputDir
$smokeExe = Join-Path $resolvedBuildDir 'bin/spcbpt_smoke.exe'
$cudaValidatorExe = Join-Path $resolvedBuildDir 'bin/spcbpt_optimal_e_optimizer_test.exe'
$raygenOptixIr = Join-Path $resolvedBuildDir 'bin/optix-ir/raygen.optixir'
$hitProgramOptixIr = Join-Path $resolvedBuildDir 'bin/optix-ir/hit_program.optixir'
$validationScript = Join-Path $PSScriptRoot 'validate_optimal_e.ps1'
foreach ($requiredFile in @(
    $smokeExe,
    $cudaValidatorExe,
    $raygenOptixIr,
    $hitProgramOptixIr,
    $validationScript
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing required experiment input: $requiredFile"
    }
}
New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null

$provenance = [ordered]@{
    schema_version = 2
    smoke = Get-FileProvenance -Path $smokeExe
    cuda_validator = Get-FileProvenance -Path $cudaValidatorExe
    raygen_optixir = Get-FileProvenance -Path $raygenOptixIr
    hit_program_optixir = Get-FileProvenance -Path $hitProgramOptixIr
    generator_script = Get-FileProvenance -Path $PSCommandPath
    python_reference = Get-FileProvenance -Path (Join-Path $PSScriptRoot 'optimal_e_reference.py')
    validation_script = Get-FileProvenance -Path $validationScript
    asset_tree = Get-DirectoryTreeProvenance -Path 'assets'
    actual_scenes = @(
        foreach ($scene in $scenes) {
            [ordered]@{
                id = $scene.id
                descriptor = Get-FileProvenance -Path $scene.actual_scene
            }
        }
    )
}
$provenanceBytes = [System.Text.Encoding]::UTF8.GetBytes(
    ($provenance | ConvertTo-Json -Depth 8 -Compress)
)
$provenanceId = [System.Convert]::ToHexString(
    [System.Security.Cryptography.SHA256]::HashData($provenanceBytes)
).ToLowerInvariant()

$manifest = [ordered]@{
    schema_version = 3
    scene_count = $scenes.Count
    camera_count = $cameraCount
    seeds = $seeds
    run_count = $runs.Count
    pytorch_steps = $PyTorchSteps
    cuda_learning_rate = $CudaLearningRate
    torch_device = $TorchDevice
    scenes = $scenes
    runs = $runs
    provenance_id = $provenanceId
    provenance = $provenance
}
$manifestPath = Join-Path $resolvedOutputDir 'manifest.json'
$manifestJson = ($manifest | ConvertTo-Json -Depth 12)
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    $existingJson = Get-Content -LiteralPath $manifestPath -Raw |
        ConvertFrom-Json |
        ConvertTo-Json -Depth 12
    if ($existingJson -ne $manifestJson) {
        throw 'OutputDir contains an incompatible manifest. Use a new -OutputDir.'
    }
} else {
    [System.IO.File]::WriteAllText(
        $manifestPath,
        ($manifestJson + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false)
    )
}

if ($SmokeOnly) {
    $smokeRoot = Join-Path $resolvedOutputDir 'smoke'
    New-Item -ItemType Directory -Path $smokeRoot -Force | Out-Null
    $smokeRows = @()
    foreach ($scene in $scenes) {
        $scenePath = Resolve-ProjectPath -Path $scene.actual_scene
        for ($cameraIndex = 0; $cameraIndex -lt $scene.cameras.Count; ++$cameraIndex) {
            $camera = $scene.cameras[$cameraIndex]
            $name = '{0}-c{1:D2}-{2}' -f $scene.id, $cameraIndex, $camera.name
            $stdoutPath = Join-Path $smokeRoot "$name.stdout.log"
            $stderrPath = Join-Path $smokeRoot "$name.stderr.log"
            $arguments = @(
                "--scene=$scenePath"
                "--eye=$($camera.eye)"
                "--lookat=$($camera.lookat)"
                "--up=$($camera.up)"
                "--fov=$($camera.fov)"
                '--experiment-seed=11'
                '--validate-frame'
            )
            $timer = [System.Diagnostics.Stopwatch]::StartNew()
            $exitCode = Invoke-CapturedProcess `
                -FilePath $smokeExe `
                -Arguments $arguments `
                -StandardOutputPath $stdoutPath `
                -StandardErrorPath $stderrPath
            $timer.Stop()
            $smokeRows += [pscustomobject][ordered]@{
                scene = $scene.id
                camera = $camera.name
                exit_code = $exitCode
                seconds = $timer.Elapsed.TotalSeconds
                stdout = [System.IO.Path]::GetRelativePath($resolvedOutputDir, $stdoutPath)
                stderr = [System.IO.Path]::GetRelativePath($resolvedOutputDir, $stderrPath)
            }
            Write-Host "SMOKE $name exit=$exitCode"
        }
    }
    $smokeReport = [ordered]@{
        provenance_id = $provenanceId
        completed = $smokeRows.Count
        passed = @($smokeRows | Where-Object exit_code -eq 0).Count
        failed = @($smokeRows | Where-Object exit_code -ne 0).Count
        rows = $smokeRows
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $resolvedOutputDir 'smoke-results.json'),
        (($smokeReport | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false)
    )
    if ($smokeReport.failed -ne 0) {
        throw "$($smokeReport.failed) camera smoke checks failed"
    }
    Write-Host "MULTISCENE_SMOKE_OK: $($smokeReport.passed)/$cameraCount"
    return
}

$smokeResultPath = Join-Path $resolvedOutputDir 'smoke-results.json'
if (-not (Test-Path -LiteralPath $smokeResultPath -PathType Leaf)) {
    throw 'Run this script once with -SmokeOnly before data generation.'
}
$smokeResult = Get-Content -LiteralPath $smokeResultPath -Raw | ConvertFrom-Json
if ($smokeResult.provenance_id -ne $provenanceId -or
    [int]$smokeResult.passed -ne $cameraCount -or
    [int]$smokeResult.failed -ne 0) {
    throw 'Camera smoke report is incomplete or contains failures.'
}

$script:resolvedTorchDevice = $null
function Read-ValidatedSummary {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Run,
        [Parameter(Mandatory)][string]$SummaryPath
    )
    $summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $expectedScene = Resolve-ProjectPath -Path $Run.scene
    if ([int]$summary.schema_version -ne 4 -or
        $summary.scene -ne $expectedScene -or
        $summary.eye -ne $Run.eye -or
        $summary.lookat -ne $Run.lookat -or
        $summary.up -ne $Run.up -or
        [double]$summary.fov -ne [double]$Run.fov -or
        [uint32]$summary.experiment_seed -ne [uint32]$Run.experiment_seed -or
        [uint32]$summary.reference.experiment_seed -ne [uint32]$Run.experiment_seed -or
        $summary.provenance_id -ne $provenanceId -or
        [double]$summary.cuda_learning_rate -ne $CudaLearningRate -or
        [int]$summary.reference.pytorch.steps -ne $PyTorchSteps) {
        throw "Summary identity does not match the manifest: $SummaryPath"
    }
    $actualDevice = [string]$summary.reference.device
    if ($TorchDevice -eq 'auto') {
        if ($null -eq $script:resolvedTorchDevice) {
            $script:resolvedTorchDevice = $actualDevice
        } elseif ($actualDevice -ne $script:resolvedTorchDevice) {
            throw "Summaries resolved -TorchDevice auto inconsistently: $SummaryPath"
        }
    } elseif ($actualDevice -ne $TorchDevice) {
        throw "Summary torch device does not match the manifest: $SummaryPath"
    }
    return $summary
}

$endIndex = [Math]::Min($StartIndex + $Count, $runs.Count)
foreach ($run in $runs[$StartIndex..($endIndex - 1)]) {
    $runName = '{0:D3}-{1}-c{2:D2}-{3}-s{4}' -f
        $run.index, $run.scene_id, $run.camera_index, $run.camera, $run.experiment_seed
    $runDir = Join-Path $resolvedOutputDir $runName
    $summaryPath = Join-Path $runDir 'summary.json'
    if ((Test-Path -LiteralPath $summaryPath -PathType Leaf) -and -not $Force) {
        $null = Read-ValidatedSummary -Run $run -SummaryPath $summaryPath
        Write-Host "SKIP $runName"
        continue
    }
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $runLog = Join-Path $runDir 'run.log'
    Write-Host "RUN $runName"
    $validationArguments = @{
        BuildDir = $BuildDir
        Scene = $run.scene
        Eye = $run.eye
        Lookat = $run.lookat
        Up = $run.up
        Fov = $run.fov
        ExperimentSeed = $run.experiment_seed
        OutputDir = $runDir
        TorchDevice = $TorchDevice
        PyTorchSteps = $PyTorchSteps
        CudaLearningRate = $CudaLearningRate
        PythonCommand = $PythonCommand
        ProvenanceId = $provenanceId
    }
    try {
        & $validationScript @validationArguments *> $runLog
        $null = Read-ValidatedSummary -Run $run -SummaryPath $summaryPath
        $failurePath = Join-Path $runDir 'failure.json'
        if (Test-Path -LiteralPath $failurePath -PathType Leaf) {
            Remove-Item -LiteralPath $failurePath
        }
    } catch {
        $failure = [ordered]@{
            run = $run
            error = $_.Exception.Message
            log = $runLog
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $runDir 'failure.json'),
            (($failure | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
            [System.Text.UTF8Encoding]::new($false)
        )
        throw
    }
}

$rows = @()
foreach ($run in $runs) {
    $runName = '{0:D3}-{1}-c{2:D2}-{3}-s{4}' -f
        $run.index, $run.scene_id, $run.camera_index, $run.camera, $run.experiment_seed
    $summaryPath = Join-Path (Join-Path $resolvedOutputDir $runName) 'summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        continue
    }
    $summary = Read-ValidatedSummary -Run $run -SummaryPath $summaryPath
    $rows += [pscustomobject][ordered]@{
        index = $run.index
        scene = $run.scene_id
        camera = $run.camera
        seed = $run.experiment_seed
        cuda_initial_loss = [double]$summary.reference.cross_check.cuda_initial_objective
        cuda_final_loss = [double]$summary.reference.cross_check.cuda_final_objective
        pytorch_initial_loss = [double]$summary.reference.pytorch.initial_loss
        pytorch_final_loss = [double]$summary.reference.pytorch.final_loss
        pytorch_final_cuda_loss = [double]$summary.reverse_cross_check.cuda_objective
        objective_abs_error = [double]$summary.reference.cross_check.initial_objective_abs_error
        gradient_max_abs_error = [double]$summary.reference.cross_check.gradient_max_abs_error
        cuda_final_abs_error = [double]$summary.reference.cross_check.final_objective_abs_error
        pytorch_reverse_abs_error = [double]$summary.reverse_cross_check.absolute_error
        capture_seconds = [double]$summary.timings_seconds.capture
        cuda_seconds = [double]$summary.timings_seconds.cuda
        pytorch_seconds = [double]$summary.timings_seconds.pytorch
        cuda_reverse_seconds = [double]$summary.timings_seconds.cuda_reverse
    }
}
$results = [ordered]@{
    schema_version = 2
    completed = $rows.Count
    expected = $runs.Count
    cuda_learning_rate = $CudaLearningRate
    rows = $rows
}
[System.IO.File]::WriteAllText(
    (Join-Path $resolvedOutputDir 'results.json'),
    (($results | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
$rows | Export-Csv -LiteralPath (Join-Path $resolvedOutputDir 'results.csv') -NoTypeInformation
Write-Host "MULTISCENE_PROGRESS: $($rows.Count)/$($runs.Count) in $resolvedOutputDir"
