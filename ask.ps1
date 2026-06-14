param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Question,

    [Parameter(Position = 1)]
    [Alias("Image", "File")]
    [string[]]$InputFile = @(),

    [string]$Model,

    [string]$Provider = "qwen",

    [string]$SearchProvider = "tavily",

    [string]$BrowserBackend = "visible",

    [int]$MaxSteps = 30,

    [int]$MaxRuntimeSeconds = 3600,

    [int]$MaxNoProgressSeconds = 300,

    [int]$MaxDrainSeconds = 1200,

    [switch]$Json
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $RepoRoot
$env:INSIGHTSWARM_MODEL_PROVIDER = $Provider

if ($Model) {
    $env:INSIGHTSWARM_QWEN_TEXT_MODEL = $Model
    $env:INSIGHTSWARM_BROWSER_QWEN_TEXT_MODEL = $Model
}

if (-not $env:INSIGHTSWARM_BROWSER_PROFILE_ROOT) {
    $env:INSIGHTSWARM_BROWSER_PROFILE_ROOT = Join-Path $RepoRoot ".tmp\browser-profiles"
}

$argsList = @(
    "-m", "insightswarm.cli",
    "--model-provider", $Provider,
    "run", "ask", $Question,
    "--search-provider", $SearchProvider,
    "--browser-backend", $BrowserBackend,
    "--max-steps", "$MaxSteps",
    "--max-runtime-seconds", "$MaxRuntimeSeconds",
    "--max-no-progress-seconds", "$MaxNoProgressSeconds",
    "--max-drain-seconds", "$MaxDrainSeconds"
)

foreach ($path in $InputFile) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Input file does not exist: $path"
    }
    $argsList += @("--input-file", $path)
}

if ($Json) {
    $argsList += "--json"
}

Push-Location $RepoRoot
try {
    & python @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
