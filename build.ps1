# build.ps1 — compile the ctxfold worker (C++17, zero dependencies).
#
#   .\build.ps1              # g++ (MinGW/MSYS2), falls back to cl.exe
#   .\build.ps1 -Compiler cl # force MSVC (run inside a Developer Prompt)
#
# Output: plugins/context_engine/ctxfold/bin/ctxfold.exe (or `ctxfold` on Unix).
param(
    [ValidateSet("auto", "g++", "cl")]
    [string]$Compiler = "auto"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir = Join-Path $root "plugins/context_engine/ctxfold/bin"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$sources = @(
    "src/core.cpp",
    "src/compress.cpp",
    "src/main.cpp",
    "src/sha256.cpp"
) | ForEach-Object { Join-Path $root $_ }

$exe = if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Join-Path $outDir "ctxfold.exe"
} else {
    Join-Path $outDir "ctxfold"
}

function Test-Tool($name) {
    return (Get-Command $name -ErrorAction SilentlyContinue) -ne $null
}

$useGpp = ($Compiler -eq "g++") -or ($Compiler -eq "auto" -and (Test-Tool g++))
if ($useGpp) {
    Write-Host "compiling with g++ -> $exe"
    Remove-Item $exe -Force -ErrorAction SilentlyContinue  # never report a stale binary
    & g++ -std=c++17 -O2 -Wall -o $exe @sources
    if ($LASTEXITCODE -ne 0) { throw "g++ failed with exit code $LASTEXITCODE" }
} elseif (Test-Tool cl) {
    Write-Host "compiling with cl.exe -> $exe"
    Remove-Item $exe -Force -ErrorAction SilentlyContinue
    Push-Location $root
    try {
        & cl /nologo /std:c++17 /O2 /EHsc /Fe:$exe @sources
        if ($LASTEXITCODE -ne 0) { throw "cl.exe failed with exit code $LASTEXITCODE" }
        Remove-Item *.obj -Force -ErrorAction SilentlyContinue
    } finally {
        Pop-Location
    }
} else {
    throw "no C++ compiler found: install MinGW-w64 (g++) or run from a Developer Prompt with cl.exe"
}

if (-not (Test-Path $exe)) { throw "build produced no binary at $exe" }
Write-Host ("built {0} ({1:N0} bytes)" -f $exe, (Get-Item $exe).Length)
