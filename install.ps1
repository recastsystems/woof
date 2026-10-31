# One-command install for a WOOF developer checkout (Windows
# PowerShell 5.1+ or PowerShell 7 on Windows; Linux and macOS use
# install.sh, and this script refuses there, see below).
#
#   .\install.ps1 [-Yes] [-NoRender] [-Cuda 12|13] [-Python PY]
#                                           -- from a checkout root
#
# The standalone (iwr | iex) form clones the public repository into
# .\woof when run outside a checkout; WOOF_REPO_URL overrides the
# clone source (fork or mirror).
#
# What it does, in order (every step is re-run safe):
#   1. finds the checkout (or clones $env:WOOF_REPO_URL into .\woof);
#   2. makes sure a Rust toolchain is present, offering to install
#      rustup when `cargo` is missing (prompts first; -Yes or
#      WOOF_INSTALL_YES=1 consents non-interactively), and that it is
#      at least the newest `rust-version` tools\*\Cargo.toml declares:
#      an older rustc under rustup gets that toolchain added beside it
#      (same consent) and used for this install via RUSTUP_TOOLCHAIN;
#      an older rustc without rustup stops with the version it needs;
#   3. creates .venv if absent on free-threaded CPython 3.14t (see
#      "Interpreter" below), installs the checkout's recast-woof-data
#      companion, then installs -e ".[gpu-cuNN,render]" into it,
#      where NN is the CUDA major this box's driver reports (CuPy
#      ships one wheel per major and the wrong one dies at its first
#      cuBLAS load); -Cuda overrides the detection, and an undetectable
#      major is announced rather than defaulted quietly;
#   4. stages the externalized Thompson tables with `woof fetch-tables`
#      (downloads only what is absent -- ~243 MiB from a checkout --
#      SHA-256 verified before install; a no-op when already staged;
#      skip with -NoFetchTables or WOOF_INSTALL_NO_FETCH_TABLES=1);
#   5. builds the vendored Rust GRIB bridges offline in
#      tools\grib1_bridge;
#   6. builds the vendored production render engine offline in
#      tools\rustwx (skip with -NoRender or WOOF_INSTALL_NO_RENDER=1);
#   7. (the terminal workspace is not part of this engine);
#   8. builds the regular-grid Zarr reader offline in tools\zarr_bridge,
#      the mapped-source decode engine in tools\rw_wps and the velocity
#      dealiasing library in tools\region_global_dealias (the default
#      decode path and the default dealiasing engine);
#   9. finishes with `woof doctor`, with .venv\Scripts on its Path,
#      and exits with doctor's status.
#
# Environment:
#   WOOF_REPO_URL     clone source when run outside a checkout
#                      (default:
#                      https://github.com/recastsystems/woof).
#   WOOF_PYTHON       interpreter used to create .venv, same as
#                      -Python (default: free-threaded CPython 3.14t,
#                      see "Interpreter" below)
#   WOOF_INSTALL_YES  "1" behaves like -Yes
#   WOOF_INSTALL_NO_RENDER  "1" behaves like -NoRender
#   WOOF_INSTALL_NO_FETCH_TABLES  "1" behaves like -NoFetchTables
#   WOOF_INSTALL_CUDA  "12" or "13" behaves like -Cuda
#
# Interpreter.  A new .venv is made on free-threaded CPython 3.14t
# (PEP 703): a python3.14t already on PATH (python.org's free-threaded
# option and the Python install manager both provide one), else the one
# uv finds or installs (`uv python install 3.14t`), else, with consent
# (-Yes, or the prompt), uv itself from https://astral.sh/uv first.  The
# woof command line then re-runs itself once with PYTHON_GIL=0 (see
# woof\free_threading.py), so nothing has to be set by hand, and no
# PYTHON_GIL is left in the user's environment, where it would stop
# every Python with the interpreter lock from starting.
# THE BREAKAGE THIS PREVENTS: every [devices] rank of a multi-card
# forecast steps its card from its own Python thread, and under an
# interpreter lock those threads take turns -- measured 2026-10-03,
# 4 cards ran 289 s per forecast hour with the lock and 130 s on 3.14t,
# so a plain multi-card install ran about 2x slower than the published
# benchmarks.  -Python (or WOOF_PYTHON) picks another interpreter;
# when no 3.14t can be had, none can make a venv, or a dependency will
# not install under it, the install continues on python (then py -3) and
# says, by name, what that costs and how to get 3.14t.
#
# No param() block: the script must also run when piped through iex,
# where param() is unavailable; flags arrive via $args or environment.
#
# Windows only.  Every step below assumes a Windows host: the venv
# interpreter is .venv\Scripts\python.exe, Rust lands in
# $env:USERPROFILE\.cargo\bin via win.rustup.rs's rustup-init.exe, and
# Path entries are joined with ';'.  Under PowerShell 7 on Linux or
# macOS none of that holds and the run broke partway, after the clone
# or .venv step had already changed the tree: on the ubuntu-24.04 CI
# runner's pwsh the .venv\Scripts\python.exe it invoked for pip was
# handed to xdg-open, and the run then died at Join-Path on the unset
# $env:USERPROFILE.  install.sh is the installer for those hosts, so
# this script refuses there before it touches anything.  Windows
# PowerShell 5.1 defines no $IsWindows and only runs on Windows, so an
# absent $IsWindows means Windows.

if ((Test-Path variable:IsWindows) -and -not $IsWindows) {
    $refusal = ('install: ERROR: install.ps1 is the Windows installer and this host is not Windows ' +
                '(it builds a .venv\Scripts layout, installs Rust under %USERPROFILE% and joins Path with '';''). ' +
                'Nothing was changed.  On Linux or macOS run the POSIX installer from the checkout root ' +
                'instead: bash install.sh  (same options: --yes, --no-render, --no-fetch-tables, --cuda 12|13)')
    if ($MyInvocation.MyCommand.Path) {
        [Console]::Error.WriteLine($refusal)
        exit 2
    }
    # Piped (iwr | iex): `exit` would close the caller's console.
    throw $refusal
}

$ErrorActionPreference = 'Stop'

$Yes = ($env:WOOF_INSTALL_YES -eq '1')
$NoRender = ($env:WOOF_INSTALL_NO_RENDER -eq '1')
$NoFetchTables = ($env:WOOF_INSTALL_NO_FETCH_TABLES -eq '1')
$CudaMajor = $env:WOOF_INSTALL_CUDA
$ExplicitPython = $env:WOOF_PYTHON
$scriptArgs = @()
if (Test-Path variable:args) { $scriptArgs = @($args) }
$wantCuda = $false
$wantPython = $false
foreach ($arg in $scriptArgs) {
    if ($wantCuda) { $CudaMajor = "$arg"; $wantCuda = $false; continue }
    if ($wantPython) { $ExplicitPython = "$arg"; $wantPython = $false; continue }
    switch -Regex ($arg) {
        '^(-y|-Yes|--yes)$' { $Yes = $true }
        '^(-NoRender|--no-render)$' { $NoRender = $true }
        '^(-NoFetchTables|--no-fetch-tables)$' { $NoFetchTables = $true }
        '^(-Cuda|--cuda)$' { $wantCuda = $true }
        '^(-Cuda|--cuda)[:=](.+)$' { $CudaMajor = $Matches[2] }
        '^(-Python|--python)$' { $wantPython = $true }
        '^(-Python|--python)[:=](.+)$' { $ExplicitPython = $Matches[2] }
        default {
            throw ("install.ps1: unknown argument '$arg' " +
                   "(-Yes, -NoRender, -NoFetchTables, -Cuda and -Python)")
        }
    }
}
if ($wantCuda) { throw 'install.ps1: -Cuda needs a value (12 or 13)' }
if ($wantPython) { throw 'install.ps1: -Python needs an interpreter' }
if ($CudaMajor -and @('12', '13') -notcontains "$CudaMajor") {
    throw "install.ps1: -Cuda takes 12 or 13, not '$CudaMajor'"
}

function Say([string]$Message) { Write-Host "install: $Message" }
function Fail([string]$Message) { throw "install: ERROR: $Message" }
function Invoke-Step([string]$What, [scriptblock]$Step) {
    & $Step
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------- checkout
$repoUrl = if ($env:WOOF_REPO_URL) { $env:WOOF_REPO_URL }
           else { 'https://github.com/recastsystems/woof' }
if ((Test-Path 'pyproject.toml') -and (Test-Path 'woof') -and
        (Test-Path 'tools\grib1_bridge')) {
    Say "using the existing checkout at $(Get-Location)"
} elseif ((Test-Path 'woof\pyproject.toml') -and
        (Test-Path 'woof\woof')) {
    Set-Location 'woof'
    Say "using the existing checkout at $(Get-Location)"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Fail 'git is required to clone'
    }
    Say "cloning $repoUrl into .\woof"
    Invoke-Step 'git clone' { git clone $repoUrl woof }
    Set-Location 'woof'
}

# ---------------------------------------------------------------- rust/cargo
# Before the .venv: on free-threaded 3.14t the [render] extra's wrf-rust
# has no published wheel (0.2.39 ships cp310-cp314 only, read off the
# index 2026-10-08), so pip builds it from its sdist, and that needs
# cargo on Path.
#
# A rustup installed earlier in this same run (or a previous one) lives in
# ~\.cargo\bin before it reaches PATH, so look there too.
$cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path (Join-Path $cargoBin 'cargo.exe')) {
    $env:Path = "$cargoBin;$env:Path"
}
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($cargo) {
    Say "cargo found: $($cargo.Source)"
} else {
    Say 'cargo was not found; the Rust GRIB bridges need a Rust toolchain.'
    $rustupYes = $Yes
    if (-not $rustupYes) {
        $answer = ''
        try { $answer = Read-Host 'install: install rustup (https://rustup.rs) now? [y/N]' }
        catch { $answer = '' }
        if ($answer -match '^(y|yes)$') { $rustupYes = $true }
    }
    if (-not $rustupYes) {
        Fail ('cargo is missing and consent to install rustup was not ' +
              'given; re-run with -Yes (or WOOF_INSTALL_YES=1), or ' +
              'install a Rust toolchain yourself and re-run')
    }
    Say 'installing rustup (stable toolchain, PATH left unmodified)'
    $rustupInit = Join-Path $env:TEMP 'rustup-init.exe'
    Invoke-WebRequest -UseBasicParsing 'https://win.rustup.rs/x86_64' `
        -OutFile $rustupInit
    Invoke-Step 'rustup-init' {
        & $rustupInit -y --no-modify-path
    }
    Remove-Item $rustupInit -ErrorAction SilentlyContinue
    $env:Path = "$cargoBin;$env:Path"
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Fail 'rustup finished but cargo is still not on PATH'
    }
}

# ------------------------------------------------------------ rust version
# THE BREAKAGE THIS PREVENTS: a box whose default Rust was 1.93.1 ran the
# installer past the .venv and then stopped in tools\rustwx with "rustc
# 1.93.1 is not supported by the following packages: ... rw-simradar@0.1.0
# requires rustc 1.94" (2.8.9 candidate, 2026-10-10); the same install
# passed with RUSTUP_TOOLCHAIN=1.94.0.  The minimum is not a constant here:
# it is the newest `rust-version` the workspaces below declare in their
# tools\<name>\Cargo.toml, the same field cargo enforces, so this check and
# cargo cannot disagree.  install.sh reads the same field the same way.
$rustWorkspaces = @('grib1_bridge', 'rustwx', 'arwen-tui', 'zarr_bridge',
                    'rw_wps', 'region_global_dealias',
                    'nextgen_hydrofabric', 'nextgen_forcing', 'nextgen_verify',
                    'hydro_world', 'hydro_world_params', 'nextgen_reservoir_prepare', 'hydro_world_init',
                    'ocean_prepare', 'rw-bathy', 'wind-table', 'nextgen_output', 'rw_icechunk')
function ConvertTo-RustVersion([string]$Text) {
    # '1.94', '1.94.0' or rustc's '1.95.0-nightly' as a comparable [version].
    if ($Text -match '^(\d+)\.(\d+)(?:\.(\d+))?') {
        $patch = 0
        if ($Matches[3]) { $patch = [int]$Matches[3] }
        return New-Object System.Version ([int]$Matches[1]), ([int]$Matches[2]), $patch
    }
    return $null
}
function Get-RustcVersion {
    $out = $null
    try { $out = & rustc --version 2>$null } catch { $out = $null }
    if ("$out" -match '^rustc\s+(\S+)') { return $Matches[1] }
    return $null
}
$rustMin = $null
$rustMinText = $null
foreach ($ws in $rustWorkspaces) {
    $manifest = Join-Path 'tools' (Join-Path $ws 'Cargo.toml')
    if (-not (Test-Path $manifest)) { continue }
    foreach ($line in (Get-Content $manifest)) {
        if ($line -match '^rust-version\s*=\s*"([0-9][0-9.]*)"') {
            $text = $Matches[1]
            $found = ConvertTo-RustVersion $text
            if ($found -and ((-not $rustMin) -or $found -gt $rustMin)) {
                $rustMin = $found
                $rustMinText = $text
            }
        }
    }
}
$rustPinned = $false
$callerToolchain = $env:RUSTUP_TOOLCHAIN
if ($rustMin) {
    $rustcHave = Get-RustcVersion
    $haveVersion = $null
    if ($rustcHave) { $haveVersion = ConvertTo-RustVersion $rustcHave }
    $shown = if ($rustcHave) { $rustcHave } else { '(none)' }
    if ($haveVersion -and $haveVersion -ge $rustMin) {
        Say "rustc $rustcHave meets the Rust $rustMinText the workspaces declare"
    } elseif (Get-Command rustup -ErrorAction SilentlyContinue) {
        Say "rustc $shown is older than Rust $rustMinText, the newest"
        Say "rust-version in tools\*\Cargo.toml; rustup can add $rustMinText beside it."
        $toolchainYes = $Yes
        if (-not $toolchainYes) {
            $answer = ''
            try { $answer = Read-Host "install: install Rust $rustMinText with rustup now (your default stays as it is)? [y/N]" }
            catch { $answer = '' }
            if ($answer -match '^(y|yes)$') { $toolchainYes = $true }
        }
        if (-not $toolchainYes) {
            Fail ("rustc $shown is older than the Rust $rustMinText the " +
                  'workspaces need and consent to install it was not given; ' +
                  're-run with -Yes (or WOOF_INSTALL_YES=1), or run ' +
                  "'rustup toolchain install $rustMinText --profile minimal' " +
                  'yourself and re-run')
        }
        Say "installing Rust $rustMinText (rustup, minimal profile, default unchanged)"
        Invoke-Step "rustup toolchain install $rustMinText" {
            rustup toolchain install $rustMinText --profile minimal
        }
        $env:RUSTUP_TOOLCHAIN = $rustMinText
        $rustPinned = $true
        $rustcHave = Get-RustcVersion
        $haveVersion = $null
        if ($rustcHave) { $haveVersion = ConvertTo-RustVersion $rustcHave }
        if (-not ($haveVersion -and $haveVersion -ge $rustMin)) {
            Fail ("rustup installed Rust $rustMinText but rustc reports " +
                  "'$rustcHave' under RUSTUP_TOOLCHAIN=$rustMinText")
        }
        Say "building with rustc $rustcHave (RUSTUP_TOOLCHAIN=$rustMinText, this install only)"
    } else {
        Fail ("rustc $shown is older than Rust $rustMinText, the minimum " +
              'the workspaces in tools\ declare, and there is no rustup to ' +
              "add it; install Rust $rustMinText or newer (https://rustup.rs) " +
              'and re-run')
    }
}

# -------------------------------------------------------------------- venv
# The interpreter: free-threaded CPython 3.14t unless -Python (or
# WOOF_PYTHON) names another one.  See "Interpreter" in the header for
# the breakage this prevents and the measurement behind it.  An
# interpreter is held as an array: the program, then any arguments it
# needs ahead of the script's own (the launcher's `py -3.14t`).
$ftProbe = "import sys, sysconfig; sys.exit(0 if sysconfig.get_config_var('Py_GIL_DISABLED') else 1)"
function Invoke-Interpreter([object[]]$Interpreter, [object[]]$Arguments) {
    $exe = $Interpreter[0]
    $lead = @($Interpreter | Select-Object -Skip 1)
    & $exe @lead @Arguments
}
function Test-FreeThreaded([object[]]$Interpreter) {
    if (-not ((Get-Command $Interpreter[0] -ErrorAction SilentlyContinue) -or
              (Test-Path -LiteralPath $Interpreter[0] -PathType Leaf))) { return $false }
    $global:LASTEXITCODE = 1
    try {
        & {
            $ErrorActionPreference = 'Continue'
            Invoke-Interpreter $Interpreter @('-c', $ftProbe) 2>$null | Out-Null
        }
    } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}
function Show-Interpreter([object[]]$Interpreter) { return ($Interpreter -join ' ') }
function Write-LockedWarning([string]$Python, [string]$Why, [string]$Remedy) {
    Say "WARNING: .venv runs on $Python, a Python with the interpreter lock,"
    Say "not free-threaded CPython 3.14t: $Why."
    Say 'Multi-card [devices] forecasts run about 2x slower on it, because'
    Say 'their slab threads take turns on the lock (measured: 4 cards at'
    Say '289 s per forecast hour against 130 s on 3.14t).  One-card'
    Say "forecasts are unaffected.  To fix: $Remedy"
}
function Find-Uv {
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { return 'uv' }
    foreach ($dir in @('.local\bin', '.cargo\bin')) {
        if (-not $env:USERPROFILE) { break }
        $candidate = Join-Path $env:USERPROFILE (Join-Path $dir 'uv.exe')
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}
$script:ftWhy = ''
$script:noneFound = 'no python3.14t or uv was found'
function Install-Uv {
    $consent = $Yes
    # Enter means yes at the prompt, so a script whose input is redirected
    # (where Read-Host reads an empty line at end of input) is never asked.
    $interactive = $true
    try { $interactive = -not [Console]::IsInputRedirected } catch { $interactive = $false }
    if (-not $consent -and $interactive) {
        $answer = 'n'
        try {
            $answer = Read-Host 'install: install uv (https://astral.sh/uv) to provide free-threaded Python 3.14t? [Y/n]'
        } catch { $answer = 'n' }
        if ($answer -match '^(|y|yes)$') { $consent = $true }
    }
    if (-not $consent) {
        $script:ftWhy = "$($script:noneFound) and consent to install uv was not given"
        return $null
    }
    Say 'installing uv (https://astral.sh/uv; Path and profiles left unmodified)'
    $hadNoModify = $env:UV_NO_MODIFY_PATH
    $env:UV_NO_MODIFY_PATH = '1'
    try {
        & powershell -NoProfile -ExecutionPolicy ByPass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
        $uvExit = $LASTEXITCODE
    } catch {
        $uvExit = 1
    } finally {
        $env:UV_NO_MODIFY_PATH = $hadNoModify
    }
    if ($uvExit -ne 0) {
        $script:ftWhy = 'the uv installer failed (see its output above)'
        return $null
    }
    $uv = Find-Uv
    if (-not $uv) { $script:ftWhy = 'uv installed but could not be found afterwards' }
    return $uv
}
function Find-UvFreeThreaded([string]$Uv) {
    $found = & {
        $ErrorActionPreference = 'Continue'
        & $Uv python find 3.14t 2>$null
    }
    if ($LASTEXITCODE -eq 0 -and $found) {
        $path = "$(@($found)[0])".Trim()
        if ($path -and (Test-FreeThreaded @($path))) { return $path }
    }
    return $null
}
function Get-FreeThreadedPython([switch]$SkipPath) {
    # python.org's free-threaded option and the Python install manager
    # both put a python3.14t on PATH.  -SkipPath passes over it (one that
    # could not make a venv) and takes only a uv-managed 3.14t: by default
    # `uv python find` answers with a system 3.14t first, which on the
    # retry is the very one that could not make a venv (seen 2026-10-08
    # with deadsnakes' python3.14-nogil on Linux).
    if (-not $SkipPath) {
        if (Test-FreeThreaded @('python3.14t')) { return ,@('python3.14t') }
        $script:noneFound = 'no python3.14t or uv was found'
        $found = Get-UvFreeThreadedPython
    } else {
        $script:noneFound = 'no uv was found'
        $hadPreference = $env:UV_PYTHON_PREFERENCE
        $env:UV_PYTHON_PREFERENCE = 'only-managed'
        try {
            $found = Get-UvFreeThreadedPython
        } finally {
            $env:UV_PYTHON_PREFERENCE = $hadPreference
        }
    }
    # The comma keeps a one-element interpreter an array through the return.
    if ($found) { return ,@($found) }
    return $null
}
function Get-UvFreeThreadedPython {
    $uv = Find-Uv
    if (-not $uv) { $uv = Install-Uv }
    if (-not $uv) { return $null }
    $found = Find-UvFreeThreaded $uv
    if ($found) { return ,@($found) }
    Say 'installing free-threaded CPython 3.14t with uv'
    $hadBin = $env:UV_PYTHON_INSTALL_BIN
    $env:UV_PYTHON_INSTALL_BIN = '0'
    try {
        & $uv python install 3.14t
        $installExit = $LASTEXITCODE
    } catch {
        $installExit = 1
    } finally {
        $env:UV_PYTHON_INSTALL_BIN = $hadBin
    }
    if ($installExit -ne 0) {
        $script:ftWhy = 'uv could not install Python 3.14t (see its output above)'
        return $null
    }
    $found = Find-UvFreeThreaded $uv
    if ($found) { return ,@($found) }
    $script:ftWhy = 'uv installed Python 3.14t but did not report a free-threaded interpreter'
    return $null
}
function Get-LockedPython {
    if (Get-Command python -ErrorAction SilentlyContinue) { return ,@('python') }
    if (Get-Command py -ErrorAction SilentlyContinue) { return ,@('py', '-3') }
    Fail 'no python/py on PATH (Python 3.11+ is required)'
}
# A venv that fails part-way leaves .venv\Scripts\python.exe behind, and
# the next run would take the "reused" route and stop at the pip upgrade,
# so a failed attempt is removed.  THE BREAKAGE THIS PREVENTS: a
# python3.14t whose venv step fails (no ensurepip in a trimmed or
# repackaged build) threw through Fail and ended the whole install
# instead of falling back.
function Test-NewVenv([object[]]$Interpreter) {
    Say "creating .venv with $(Show-Interpreter $Interpreter)"
    $made = $false
    try {
        # Out-Host: anything venv prints must not join this function's
        # return value, which callers read as true or false.
        Invoke-Step 'venv creation' { Invoke-Interpreter $Interpreter @('-m', 'venv', '.venv') | Out-Host }
        $made = $true
    } catch {
        Say "$($_.Exception.Message)"
    }
    if (-not $made -and (Test-Path -LiteralPath '.venv')) {
        Remove-Item -Recurse -Force -LiteralPath '.venv'
    }
    return $made
}
function New-Venv([object[]]$Interpreter) {
    if (-not (Test-NewVenv $Interpreter)) {
        Fail ("$(Show-Interpreter $Interpreter) could not create .venv (see its output above)")
    }
}

$venvPython = Join-Path '.venv' 'Scripts\python.exe'
$venvRoute = 'reused'
$venvFreeThreaded = $false
if (Test-Path $venvPython) {
    Say 'reusing the existing .venv'
    if (Test-FreeThreaded @($venvPython)) {
        $venvFreeThreaded = $true
        Say 'the existing .venv is free-threaded'
    } elseif (-not $ExplicitPython) {
        Write-LockedWarning 'the existing .venv''s Python' `
            'that .venv was made before this installer provisioned 3.14t, or with another interpreter' `
            'remove .venv (Remove-Item -Recurse -Force .venv) and re-run .\install.ps1.'
    }
} else {
    if ($ExplicitPython) {
        $python = @($ExplicitPython)
        $venvRoute = 'explicit'
    } else {
        $python = Get-FreeThreadedPython
        if ($python) {
            $venvRoute = 'free-threaded'
        } else {
            $python = Get-LockedPython
            $venvRoute = 'locked'
        }
    }
    $ftRemedy = ''
    if ($venvRoute -ne 'free-threaded') {
        New-Venv $python
    } elseif (-not (Test-NewVenv $python)) {
        $venvWhy = "$(Show-Interpreter $python) could not create a venv (see its output above)"
        $venvMade = $false
        # A system python3.14t that cannot make a venv does not rule out
        # the 3.14t uv manages, which carries its own ensurepip.
        $retry = Get-FreeThreadedPython -SkipPath
        if ($retry) {
            $python = $retry
            if (Test-NewVenv $python) {
                $venvMade = $true
            } else {
                $venvWhy = "$venvWhy, and neither could $(Show-Interpreter $python)"
            }
        } else {
            $venvWhy = "$venvWhy, and $($script:ftWhy)"
        }
        if (-not $venvMade) {
            $script:ftWhy = $venvWhy
            $ftRemedy = ('reinstall that Python with its venv support, or install uv (https://docs.astral.sh/uv/), ' +
                         'remove .venv (Remove-Item -Recurse -Force .venv) and re-run .\install.ps1.')
            $python = Get-LockedPython
            $venvRoute = 'locked'
            New-Venv $python
        }
    }
    if (Test-FreeThreaded @($venvPython)) {
        $venvFreeThreaded = $true
        Say '.venv is free-threaded: multi-card slab threads run at once'
    } elseif ($venvRoute -eq 'explicit') {
        Write-LockedWarning (Show-Interpreter $python) 'it was chosen with -Python or WOOF_PYTHON' `
            'remove .venv and re-run without -Python and WOOF_PYTHON.'
    } else {
        $why = if ($script:ftWhy) { $script:ftWhy } else { 'no free-threaded interpreter was found' }
        if (-not $ftRemedy) {
            $ftRemedy = ('install uv (https://docs.astral.sh/uv/) or a free-threaded Python 3.14, remove .venv ' +
                         '(Remove-Item -Recurse -Force .venv) and re-run .\install.ps1.')
        }
        Write-LockedWarning (Show-Interpreter $python) $why $ftRemedy
    }
}
# ------------------------------------------------------------ CUDA major
# CuPy ships ONE wheel per CUDA major and pip cannot detect the major, so
# the extra has to name it.  Through 1.8.0 this line pasted
# ".[gpu,render]" unconditionally -- the cu12 wheel -- so a CUDA-13-only
# box got a CuPy that imports cleanly, compiles kernels, and then dies at
# its first cuBLAS load, with nothing in the install saying so.  Read the
# major off the driver instead, and when it cannot be read, SAY that
# rather than defaulting in silence.
function Get-CudaMajor {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }
    try { $smi = & nvidia-smi } catch { return $null }
    # The header label is not one string: Linux drivers print
    # "CUDA Version: 12.4" and the Windows driver on the reference box
    # prints "CUDA UMD Version: 13.3".  Matching only the first
    # spelling read as "no NVIDIA driver" on a machine that plainly
    # had one.
    foreach ($line in @($smi)) {
        if ("$line" -match 'CUDA[A-Za-z ]*Version:\s*(\d+)') {
            return $Matches[1]
        }
    }
    return $null
}

if ($CudaMajor) {
    Say "CUDA major $CudaMajor was given on the command line"
} else {
    $CudaMajor = Get-CudaMajor
    if ($CudaMajor) { Say "nvidia-smi reports CUDA $CudaMajor" }
}
if (@('12', '13') -contains "$CudaMajor") {
    $gpuExtra = "gpu-cu$CudaMajor"
} else {
    $gpuExtra = 'gpu-cu12'
    Say 'the box''s CUDA major could not be read (no nvidia-smi, or no'
    Say 'driver answered), so this install falls back to [gpu-cu12].'
    Say 'IF THIS BOX''S CUDA IS 13-ONLY THAT WHEEL IS WRONG: it will'
    Say 'import fine and fail at the first cuBLAS load.  Re-run with'
    Say '-Cuda 13 in that case; woof doctor judges the pairing at the'
    Say 'end of this script either way.'
}
# ------------------------------------------------ one CuPy build, not two
# cupy-cuda12x and cupy-cuda13x both install the same `cupy` package
# files, and pip treats them as unrelated distributions.  Re-running this
# installer with another major into the reused .venv left BOTH registered
# over one set of files: switching back then said "already satisfied"
# while the other major's build answered, and uninstalling either broke
# the other.  So when any CuPy other than the chosen one is present,
# every CuPy is removed and the chosen one is installed clean below.
# --prefer-binary on a free-threaded .venv: netCDF4's cftime publishes no
# cp314t wheel for its newest release (1.6.6; 1.6.5 has one, read off the
# index 2026-10-08), and without the flag pip builds the newest from
# source, which needs a C compiler and Cython.  A locked .venv installs
# exactly as before.
function Install-PythonPackages {
    $preferBinary = @()
    if ($venvFreeThreaded) { $preferBinary = @('--prefer-binary') }
    Invoke-Step 'pip upgrade' {
        & $venvPython -m pip install --upgrade pip
    }
    $cupyWanted = 'cupy-cuda' + $gpuExtra.Substring(6) + 'x'
    # The list is read with 'Continue' in its own scope: under 'Stop',
    # Windows PowerShell 5.1 turns the first line pip writes to stderr (a
    # warning about a half-removed package, left when an uninstall hit a
    # locked DLL) into a thrown error, and the old catch read that as "no
    # CuPy" so both builds stayed.  A pip that cannot list the .venv now
    # stops the install instead.
    $cupyListed = & {
        $ErrorActionPreference = 'Continue'
        $lines = & $venvPython -m pip list --format=freeze --disable-pip-version-check 2>$null
        [pscustomobject]@{ Lines = @($lines); Exit = $LASTEXITCODE }
    }
    if ($cupyListed.Exit -ne 0) {
        Fail ("pip could not list the packages in .venv, so this install cannot tell which CuPy it holds " +
              "(run $venvPython -m pip list to see why)")
    }
    $cupyHave = @()
    foreach ($line in $cupyListed.Lines) {
        if ("$line" -match '^(cupy(-cuda\d+x)?)\s*[=@]') { $cupyHave += $Matches[1].ToLower() }
    }
    if (@($cupyHave | Where-Object { $_ -ne $cupyWanted }).Count -gt 0) {
        Say ("this .venv holds another CUDA major's CuPy (" + ($cupyHave -join ', ') + ');')
        Say "removing every CuPy build so $cupyWanted installs clean"
        Invoke-Step 'pip uninstall cupy' {
            & $venvPython -m pip uninstall -y @cupyHave
        }
    }
    Say 'installing the matching recast-woof-data companion from this checkout (editable)'
    Invoke-Step 'pip install recast-woof-data' {
        & $venvPython -m pip install @preferBinary -e recast-woof-data
    }
    Say "installing woof with the [$gpuExtra,render] extras (editable)"
    Invoke-Step 'pip install' {
        & $venvPython -m pip install @preferBinary -e ".[$gpuExtra,render]"
    }
}

# A dependency that will not install under 3.14t (no cp314t wheel and an
# sdist that does not build) must not cost the user the whole install: a
# .venv this run made on 3.14t by default is remade on python and the
# install goes on, saying what that costs.  A reused .venv, or one made
# from an interpreter the user named, is never removed.
try {
    Install-PythonPackages
} catch {
    $message = "$($_.Exception.Message)"
    if ($venvRoute -ne 'free-threaded' -or $message -like '*could not list the packages*') { throw }
    Say "pip failed under 3.14t: $message"
    $locked = Get-LockedPython
    Write-LockedWarning (Show-Interpreter $locked) `
        ("a dependency did not install under 3.14t (see pip's output above), so this run remade .venv on " +
         (Show-Interpreter $locked)) `
        'once that package installs under 3.14t, remove .venv (Remove-Item -Recurse -Force .venv) and re-run .\install.ps1.'
    Remove-Item -Recurse -Force '.venv'
    New-Venv $locked
    $venvFreeThreaded = $false
    Install-PythonPackages
}

# ------------------------------------------------------- externalized tables
# The two largest Thompson tables ship as GitHub release assets rather
# than in the wheel (freezeH2O.dat, 243 MiB, is not in git either);
# fetch-tables downloads only what is absent and verifies SHA-256
# against the packaged pins before installing.
if ($NoFetchTables) {
    Say 'skipping the externalized table fetch (-NoFetchTables);'
    Say 'woof doctor prints the exact fetch command while they are missing'
} else {
    Say 'staging the externalized Thompson tables (downloads only what'
    Say 'is absent -- ~243 MiB from a checkout; SHA-256 verified)'
    Invoke-Step 'woof fetch-tables' {
        & (Join-Path '.venv' 'Scripts\woof.exe') fetch-tables
    }
}

# ------------------------------------------------------- offline Rust build
Say 'building the vendored Rust GRIB bridges (offline, locked)'
Push-Location 'tools\grib1_bridge'
try {
    Invoke-Step 'cargo build' {
        cargo build --release --locked --offline
    }
} finally {
    Pop-Location
}
if ($NoRender) {
    Say 'skipping the tools\rustwx render engine (-NoRender);'
    Say 'stage it with woof fetch-bridges, or request --engine matplotlib'
} else {
    Say 'building the vendored render engine in tools\rustwx (offline,'
    Say 'locked; the long pole of install -- skip with -NoRender)'
    Push-Location 'tools\rustwx'
    try {
        Invoke-Step 'cargo build (rustwx)' {
            cargo build --release --locked --offline
        }
    } finally {
        Pop-Location
    }
}
Say 'building the regular-grid Zarr reader (offline, locked)'
Push-Location 'tools\zarr_bridge'
try {
    Invoke-Step 'cargo build (rw_zarr)' {
        cargo build --release --locked --offline
    }
} finally {
    Pop-Location
}
# The optional Icechunk output store writer; its crates are vendored with rustwx.
Say 'building the Icechunk store writer in tools\rw_icechunk (offline, locked)'
Push-Location 'tools\rw_icechunk'
try {
    Invoke-Step 'cargo build (rw_icechunk)' {
        cargo build --release --locked --offline
    }
} finally {
    Pop-Location
}
# Every mapped source decodes in gpuwm_mapped_engine and every radar ingest
# dealiases through region_global_dealias by default; a checkout that skips
# either build reports both MISSING and cannot run those default routes.
Say 'building the mapped-source decode engine in tools\rw_wps (offline, locked)'
Push-Location 'tools\rw_wps'
try {
    Invoke-Step 'cargo build (rw_wps)' {
        cargo build --release --locked --offline
    }
} finally {
    Pop-Location
}
Say 'building the velocity dealiasing library (offline, locked)'
Push-Location 'tools\region_global_dealias'
try {
    Invoke-Step 'cargo build (region_global_dealias)' {
        cargo build --release --locked --offline
    }
} finally {
    Pop-Location
}
foreach ($workspace in 'nextgen_hydrofabric', 'nextgen_forcing', 'nextgen_verify', 'hydro_world', 'hydro_world_params', 'nextgen_reservoir_prepare', 'hydro_world_init', 'ocean_prepare', 'rw-bathy', 'wind-table', 'nextgen_output') {
    Say "building NextGen tools\$workspace (offline, locked)"
    Push-Location "tools\$workspace"
    try {
        # Hydro's registry dependencies are not all vendored with rustwx.
        Invoke-Step "cargo fetch ($workspace)" {
            cargo fetch --locked
        }
        Invoke-Step "cargo build ($workspace)" {
            cargo build --release --locked --offline
        }
    } finally {
        Pop-Location
    }
}
if ($rustPinned) {
    Say "built with Rust $rustMinText; a later hand-run cargo build in tools\ needs"
    Say "it too: set RUSTUP_TOOLCHAIN=$rustMinText first, or rustup default $rustMinText"
    # The piped (iwr | iex) form runs in the caller's session: hand back
    # the toolchain choice it came in with.
    $env:RUSTUP_TOOLCHAIN = $callerToolchain
}

# ------------------------------------------------------------------ doctor
# Doctor judges the environment this script just made, as it stands once
# activated: without .venv\Scripts on Path its console-script check reports
# a gap this script created and the install exits nonzero for it.  The
# caller's Path comes back afterwards, because the piped (iwr | iex) form
# runs in the caller's own session.
Say 'running woof doctor'
$callerPath = $env:Path
try {
    $env:Path = (Join-Path (Get-Location).Path '.venv\Scripts') + ';' + $callerPath
    & (Join-Path '.venv' 'Scripts\woof.exe') doctor
    $doctorExit = $LASTEXITCODE
} finally {
    $env:Path = $callerPath
}
if ($doctorExit -eq 0) {
    Say 'done -- doctor is clean.  Activate with: .\.venv\Scripts\Activate.ps1'
} else {
    Say 'install steps completed; doctor reports gaps above (each line'
    Say 'prints its own remedy).  Re-run .\install.ps1 any time.'
}
# `exit` would close an interactive console when this script arrives via
# `iwr | iex`, so only a file invocation propagates doctor's exit code that
# way; the piped form signals a doctor gap through a terminating error.
if ($MyInvocation.MyCommand.Path) {
    exit $doctorExit
} elseif ($doctorExit -ne 0) {
    throw "woof doctor exited $doctorExit"
}
