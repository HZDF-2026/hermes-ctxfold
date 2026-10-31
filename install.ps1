# install.ps1 — drop the ctxfold context engine into a Hermes profile.
#
#   .\install.ps1                      # uses $HERMES_HOME, else the platform default
#   .\install.ps1 -HermesHome D:\hermes
#   .\install.ps1 -Build               # compile the worker first
param(
    [string]$HermesHome = "",
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $HermesHome) {
    if ($env:HERMES_HOME) {
        $HermesHome = $env:HERMES_HOME
    } elseif ($env:OS -eq "Windows_NT") {
        $HermesHome = Join-Path $env:LOCALAPPDATA "hermes"
    } else {
        $HermesHome = Join-Path $HOME ".hermes"
    }
}

$target = Join-Path $HermesHome "plugins/ctxfold"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
New-Item -ItemType Directory -Force -Path $target | Out-Null

Copy-Item -Path (Join-Path $root "plugins/context_engine/ctxfold/*") -Destination $target -Recurse -Force
Write-Host "installed -> $target"

$exe = Join-Path $target "bin/ctxfold.exe"
if ($Build -or -not (Test-Path $exe)) {
    & (Join-Path $root "build.ps1")
    Copy-Item -Path (Join-Path $root "plugins/context_engine/ctxfold/bin/*") -Destination (Join-Path $target "bin") -Force
}

if (Test-Path $exe) {
    Write-Host "worker: $exe"
} else {
    Write-Warning "worker binary missing — run .\build.ps1, then copy bin/ into $target"
}

Write-Host @"

Enable it in $HermesHome/config.yaml:

context:
  engine: "ctxfold"

Then restart Hermes. Check with: hermes plugins    (Context Engine -> ctxfold)
"@
