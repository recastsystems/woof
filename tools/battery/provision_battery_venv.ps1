<#
.SYNOPSIS
  Build a venv bound to THIS checkout that can run the CPU battery legs.

.DESCRIPTION
  The stage-1 and always legs need more than the base install: six
  stage-1 files import cupy at collection, the packaging gates fail loudly
  (by design) without setuptools, the high-res geog probe compares itself
  against rasterio/pyproj, and the MCP tests need the [mcp] extra.  On
  2026-09-01 three separate lanes re-discovered each of these gaps; one
  closing battery reported 16 failures that were all provisioning.

  Python 3.13 venvs ship WITHOUT setuptools, so it is installed explicitly.

.EXAMPLE
  powershell -File tools\battery\provision_battery_venv.ps1 -Target C:\path\to\venv
  Then, from a NEUTRAL cwd (never the repo root, which shadows the venv):
    <venv>\Scripts\python -c "import woof; print(woof.__file__)"
  must print a path inside this checkout.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [string]$Python = 'python',
    [string]$Extras = 'all,mcp,geog'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& $Python -m venv $Target
$py = Join-Path $Target 'Scripts\python.exe'
& $py -m pip install -q --upgrade pip
& $py -m pip install -q 'setuptools>=77' build wheel 'pytest==8.4.2'
$env:WOOF_ALLOW_UNPINNED_WHEEL = '1'
& $py -m pip install -q -e (Join-Path $repo 'recast-woof-data') -e (Join-Path $repo 'gpuwm-kernels') -e "$repo[$Extras]"
Push-Location $env:TEMP
try {
    $bound = & $py -c "import woof, importlib.metadata as m; print(m.version('woof'), m.version('recast-woof-data'), m.version('gpuwm-kernels'), woof.__file__)"
} finally { Pop-Location }
"provisioned: $bound"
if ($bound -notmatch [regex]::Escape($repo)) { throw "the venv resolves woof outside this checkout: $bound" }
