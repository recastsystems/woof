#!/bin/sh
# One-command install for a WOOF developer checkout (POSIX sh).
#
#   ./install.sh [--yes] [--no-render] [--cuda 12|13] [--python PY]
#                                           -- from a checkout root
#
# The standalone (curl | sh) form clones the public repository into
# ./woof when run outside a checkout; WOOF_REPO_URL overrides the
# clone source (fork or mirror).
#
# What it does, in order (every step is re-run safe):
#   1. finds the checkout (or clones $WOOF_REPO_URL into ./woof);
#   2. makes sure a Rust toolchain is present, offering to install
#      rustup when `cargo` is missing (prompts first; --yes or
#      WOOF_INSTALL_YES=1 consents non-interactively), and that it is
#      at least the newest `rust-version` tools/*/Cargo.toml declares:
#      an older rustc under rustup gets that toolchain added beside it
#      (same consent) and used for this install via RUSTUP_TOOLCHAIN;
#      an older rustc without rustup (a distribution's Rust) gets rustup
#      installed in ~/.cargo with that version as its default (same
#      consent; system Rust and PATH unchanged), or stops naming it;
#   3. creates .venv if absent on free-threaded CPython 3.14t (see
#      "Interpreter" below), installs the checkout's recast-woof-data
#      companion, then installs -e '.[gpu-cuNN,render]' into it,
#      where NN is the CUDA major this box's driver reports (CuPy
#      ships one wheel per major and the wrong one dies at its first
#      cuBLAS load); --cuda overrides the detection, and an
#      undetectable major is announced rather than defaulted quietly;
#   4. stages the externalized Thompson tables with `woof fetch-tables`
#      (downloads only what is absent -- ~243 MiB from a checkout --
#      SHA-256 verified before install; a no-op when already staged;
#      skip with --no-fetch-tables or WOOF_INSTALL_NO_FETCH_TABLES=1);
#   5. builds the vendored Rust GRIB bridges offline in
#      tools/grib1_bridge;
#   6. builds the vendored production render engine offline in
#      tools/rustwx (skip with --no-render or
#      WOOF_INSTALL_NO_RENDER=1);
#   7. (the terminal workspace is not part of this engine);
#   8. builds the regular-grid Zarr reader offline in tools/zarr_bridge,
#      the mapped-source decode engine in tools/rw_wps and the velocity
#      dealiasing library in tools/region_global_dealias (the default
#      decode path and the default dealiasing engine);
#   9. finishes with `woof doctor`, with .venv/bin on its PATH, and
#      exits with doctor's status.
#
# Environment:
#   WOOF_REPO_URL     clone source when run outside a checkout
#                      (default:
#                      https://github.com/recastsystems/woof).
#   WOOF_PYTHON       interpreter used to create .venv, same as
#                      --python (default: free-threaded CPython 3.14t,
#                      see "Interpreter" below)
#   WOOF_INSTALL_YES  "1" behaves like --yes
#   WOOF_INSTALL_NO_RENDER  "1" behaves like --no-render
#   WOOF_INSTALL_NO_FETCH_TABLES  "1" behaves like --no-fetch-tables
#   WOOF_INSTALL_CUDA  "12" or "13" behaves like --cuda
#
# Interpreter.  A new .venv is made on free-threaded CPython 3.14t
# (PEP 703): a python3.14t already on PATH, else the one uv finds or
# installs (`uv python install 3.14t`), else, with consent (--yes, or
# the prompt), uv itself from https://astral.sh/uv first.  The woof
# command line then re-executes itself once with PYTHON_GIL=0 (see
# woof/free_threading.py), so nothing has to be exported by hand.
# THE BREAKAGE THIS PREVENTS: every [devices] rank of a multi-card
# forecast steps its card from its own Python thread, and under an
# interpreter lock those threads take turns -- measured 2026-10-03,
# 4 cards ran 289 s per forecast hour with the lock and 130 s on
# 3.14t, so a plain multi-card install ran about 2x slower than the
# published benchmarks.  --python (or WOOF_PYTHON) picks another
# interpreter; when no 3.14t can be had, none can make a venv, or a
# dependency will not install under it, the install continues on
# python3 (then python) and says, by name, what that costs and how to
# get 3.14t.

set -eu

YES="${WOOF_INSTALL_YES:-0}"
NO_RENDER="${WOOF_INSTALL_NO_RENDER:-0}"
NO_FETCH_TABLES="${WOOF_INSTALL_NO_FETCH_TABLES:-0}"
CUDA_MAJOR="${WOOF_INSTALL_CUDA:-}"
EXPLICIT_PYTHON="${WOOF_PYTHON:-}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        -y|--yes) YES=1 ;;
        --no-render) NO_RENDER=1 ;;
        --no-fetch-tables) NO_FETCH_TABLES=1 ;;
        --cuda)
            [ "$#" -ge 2 ] || {
                echo "install.sh: --cuda needs a value (12 or 13)" >&2
                exit 2; }
            CUDA_MAJOR="$2"; shift ;;
        --cuda=*) CUDA_MAJOR="${1#--cuda=}" ;;
        --python)
            [ "$#" -ge 2 ] || {
                echo "install.sh: --python needs an interpreter" >&2
                exit 2; }
            EXPLICIT_PYTHON="$2"; shift ;;
        --python=*) EXPLICIT_PYTHON="${1#--python=}" ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" 2>/dev/null || true
            exit 0 ;;
        *)
            echo "install.sh: unknown argument '$1'" \
                 "(--yes, --no-render, --no-fetch-tables, --cuda and --python)" >&2
            exit 2 ;;
    esac
    shift
done
case "$CUDA_MAJOR" in
    ''|12|13) ;;
    *)  echo "install.sh: --cuda takes 12 or 13, not '$CUDA_MAJOR'" >&2
        exit 2 ;;
esac

say()  { printf 'install: %s\n' "$*"; }
fail() { printf 'install: ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checkout
REPO_URL="${WOOF_REPO_URL:-https://github.com/recastsystems/woof}"
if [ -f pyproject.toml ] && [ -d woof ] && [ -d tools/grib1_bridge ]; then
    say "using the existing checkout at $(pwd)"
elif [ -f woof/pyproject.toml ] && [ -d woof/woof ]; then
    cd woof
    say "using the existing checkout at $(pwd)"
else
    command -v git >/dev/null 2>&1 || fail "git is required to clone"
    say "cloning $REPO_URL into ./woof"
    git clone "$REPO_URL" woof
    cd woof
fi

# ---------------------------------------------------------------- rust/cargo
# Before the .venv: on free-threaded 3.14t the [render] extra's wrf-rust
# has no published wheel (0.2.39 ships cp310-cp314 only, read off the
# index 2026-10-08), so pip builds it from its sdist, and that needs
# cargo on PATH.  It builds in about 20 s and imports without turning
# the interpreter lock back on.
#
# A rustup installed earlier in this same run (or a previous one) lives in
# ~/.cargo/bin before it reaches PATH, so look there too.
PATH="$HOME/.cargo/bin:$PATH"
if command -v cargo >/dev/null 2>&1; then
    say "cargo found: $(command -v cargo)"
else
    say "cargo was not found; the Rust GRIB bridges need a Rust toolchain."
    RUSTUP_YES="$YES"
    if [ "$RUSTUP_YES" != 1 ]; then
        if [ -r /dev/tty ]; then
            printf 'install: install rustup (https://rustup.rs) now? [y/N] '
            read -r answer < /dev/tty || answer=""
            case "$answer" in
                y|Y|yes|YES) RUSTUP_YES=1 ;;
            esac
        fi
    fi
    if [ "$RUSTUP_YES" != 1 ]; then
        fail "cargo is missing and consent to install rustup was not \
given; re-run with --yes (or WOOF_INSTALL_YES=1), or install a Rust \
toolchain yourself and re-run"
    fi
    command -v curl >/dev/null 2>&1 || fail "curl is required to fetch rustup"
    say "installing rustup (stable toolchain, PATH left unmodified)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
    command -v cargo >/dev/null 2>&1 \
        || fail "rustup finished but cargo is still not on PATH"
fi

# ------------------------------------------------------------ rust version
# THE BREAKAGE THIS PREVENTS: a box whose default Rust was 1.93.1 ran
# this script past the .venv and then stopped in tools/rustwx with
# "rustc 1.93.1 is not supported by the following packages: ...
# rw-simradar@0.1.0 requires rustc 1.94" (2.8.9 candidate,
# 2026-10-10); the same install passed with RUSTUP_TOOLCHAIN=1.94.0.
# The minimum is not a constant here: it is the newest `rust-version`
# the workspaces below declare in their tools/<name>/Cargo.toml, the
# same field cargo enforces, so this check and cargo cannot disagree.
RUST_WORKSPACES='grib1_bridge rustwx arwen-tui zarr_bridge rw_wps region_global_dealias nextgen_hydrofabric nextgen_forcing nextgen_verify hydro_world hydro_world_params nextgen_reservoir_prepare hydro_world_init ocean_prepare rw-bathy wind-table nextgen_output rw_icechunk'
rust_minimum() {
    for ws in $RUST_WORKSPACES; do
        [ -f "tools/$ws/Cargo.toml" ] || continue
        sed -n 's/^rust-version[[:space:]]*=[[:space:]]*"\([0-9][0-9.]*\)".*/\1/p' \
            "tools/$ws/Cargo.toml"
    done | awk -F. '{ v = $1 * 1000000 + $2 * 1000 + $3
                      if (v > best) { best = v; text = $0 } }
                    END { print text }'
}
# version_ok HAVE NEED: rustc's HAVE (x.y.z) is NEED (x.y or x.y.z) or newer.
version_ok() {
    awk -v have="$1" -v need="$2" 'BEGIN {
        split(have, h, "."); split(need, n, ".")
        exit !(h[1] * 1000000 + h[2] * 1000 + h[3] >= n[1] * 1000000 + n[2] * 1000 + n[3]) }'
}
rustc_version() { rustc --version 2>/dev/null | awk '{ print $2 }'; }
RUST_MIN="$(rust_minimum)"
RUST_PINNED=0
if [ -n "$RUST_MIN" ]; then
    RUSTC_HAVE="$(rustc_version || true)"
    if version_ok "${RUSTC_HAVE:-0}" "$RUST_MIN"; then
        say "rustc $RUSTC_HAVE meets the Rust $RUST_MIN the workspaces declare"
    elif command -v rustup >/dev/null 2>&1; then
        say "rustc ${RUSTC_HAVE:-(none)} is older than Rust $RUST_MIN, the newest"
        say "rust-version in tools/*/Cargo.toml; rustup can add $RUST_MIN beside it."
        TOOLCHAIN_YES="$YES"
        if [ "$TOOLCHAIN_YES" != 1 ]; then
            if [ -r /dev/tty ]; then
                printf 'install: install Rust %s with rustup now (your default stays as it is)? [y/N] ' "$RUST_MIN"
                read -r answer < /dev/tty || answer=""
                case "$answer" in
                    y|Y|yes|YES) TOOLCHAIN_YES=1 ;;
                esac
            fi
        fi
        if [ "$TOOLCHAIN_YES" != 1 ]; then
            fail "rustc ${RUSTC_HAVE:-(none)} is older than the Rust $RUST_MIN \
the workspaces need and consent to install it was not given; re-run with \
--yes (or WOOF_INSTALL_YES=1), or run 'rustup toolchain install $RUST_MIN \
--profile minimal' yourself and re-run"
        fi
        say "installing Rust $RUST_MIN (rustup, minimal profile, default unchanged)"
        rustup toolchain install "$RUST_MIN" --profile minimal
        RUSTUP_TOOLCHAIN="$RUST_MIN"
        export RUSTUP_TOOLCHAIN
        RUST_PINNED=1
        RUSTC_HAVE="$(rustc_version || true)"
        version_ok "${RUSTC_HAVE:-0}" "$RUST_MIN" \
            || fail "rustup installed Rust $RUST_MIN but rustc reports \
'${RUSTC_HAVE:-nothing}' under RUSTUP_TOOLCHAIN=$RUST_MIN"
        say "building with rustc $RUSTC_HAVE (RUSTUP_TOOLCHAIN=$RUST_MIN, this install only)"
    else
        # THE BREAKAGE THIS PREVENTS: a distribution's Rust older than the
        # minimum, with no rustup beside it (Ubuntu's apt rustc, or a
        # source-built 1.93.1), stopped the published 2.9.4 install.sh dead
        # after the clone, even with --yes, while the same script installs
        # rustup with that consent when cargo is missing altogether
        # (seen 2026-10-22).  With the same consent, rustup goes into
        # ~/.cargo with the minimum as its default; the system Rust and the
        # shell profile are left as they are.
        say "rustc ${RUSTC_HAVE:-(none)} ($(command -v rustc 2>/dev/null || echo 'not on PATH')) is older"
        say "than Rust $RUST_MIN, the newest rust-version in tools/*/Cargo.toml, and"
        say "no rustup manages it; rustup can install $RUST_MIN in ~/.cargo beside it."
        RUSTUP_YES="$YES"
        if [ "$RUSTUP_YES" != 1 ]; then
            if [ -r /dev/tty ]; then
                printf 'install: install rustup with Rust %s in ~/.cargo now (system Rust and PATH unchanged)? [y/N] ' "$RUST_MIN"
                read -r answer < /dev/tty || answer=""
                case "$answer" in
                    y|Y|yes|YES) RUSTUP_YES=1 ;;
                esac
            fi
        fi
        if [ "$RUSTUP_YES" != 1 ]; then
            fail "rustc ${RUSTC_HAVE:-(none)} is older than Rust $RUST_MIN, the \
minimum the workspaces in tools/ declare, and consent to install rustup was \
not given; re-run with --yes (or WOOF_INSTALL_YES=1), or install Rust \
$RUST_MIN or newer (https://rustup.rs, or a newer distribution package) and \
re-run"
        fi
        command -v curl >/dev/null 2>&1 || fail "curl is required to fetch rustup"
        # Fetched to a file first: piped straight into sh, a failed download
        # runs an empty script that exits 0.
        RUSTUP_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/gpuwm-rustup-init.XXXXXX") \
            || fail "no temporary file for the rustup installer could be made"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$RUSTUP_SCRIPT" \
            || { rm -f "$RUSTUP_SCRIPT"; fail "the rustup installer could not be downloaded from https://sh.rustup.rs"; }
        say "installing rustup with Rust $RUST_MIN (minimal profile, PATH left unmodified)"
        # RUSTUP_INIT_SKIP_PATH_CHECK: rustup-init otherwise stops to ask
        # about the Rust already on PATH, the very one being worked around.
        env RUSTUP_INIT_SKIP_PATH_CHECK=yes sh "$RUSTUP_SCRIPT" -y --no-modify-path \
            --profile minimal --default-toolchain "$RUST_MIN" \
            || { rm -f "$RUSTUP_SCRIPT"; fail "the rustup installer failed (see its output above)"; }
        rm -f "$RUSTUP_SCRIPT"
        PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
        command -v rustup >/dev/null 2>&1 \
            || fail "rustup finished but is not in ${CARGO_HOME:-$HOME/.cargo}/bin"
        RUSTUP_TOOLCHAIN="$RUST_MIN"
        export RUSTUP_TOOLCHAIN
        RUST_PINNED=1
        RUSTC_HAVE="$(rustc_version || true)"
        version_ok "${RUSTC_HAVE:-0}" "$RUST_MIN" \
            || fail "rustup installed Rust $RUST_MIN but rustc reports \
'${RUSTC_HAVE:-nothing}' under RUSTUP_TOOLCHAIN=$RUST_MIN"
        say "building with rustc $RUSTC_HAVE from ${CARGO_HOME:-$HOME/.cargo}/bin (this install only;"
        say "add it to PATH for hand-run cargo builds: export PATH=\"${CARGO_HOME:-\$HOME/.cargo}/bin:\$PATH\")"
    fi
fi

# -------------------------------------------------------------------- venv
# The interpreter: free-threaded CPython 3.14t unless --python (or
# WOOF_PYTHON) names another one.  See "Interpreter" in the header for
# the breakage this prevents and the measurement behind it.
FT_PROBE='import sys, sysconfig; sys.exit(0 if sysconfig.get_config_var("Py_GIL_DISABLED") else 1)'
is_free_threaded() { "$1" -c "$FT_PROBE" >/dev/null 2>&1; }

warn_locked() {
    # $1 = the interpreter, $2 = why it is not 3.14t, $3 = the remedy
    say "WARNING: .venv runs on $1, a Python with the interpreter lock,"
    say "not free-threaded CPython 3.14t: $2."
    say "Multi-card [devices] forecasts run about 2x slower on it, because"
    say "their slab threads take turns on the lock (measured: 4 cards at"
    say "289 s per forecast hour against 130 s on 3.14t).  One-card"
    say "forecasts are unaffected.  To fix: $3"
}

find_uv() {
    for candidate in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            UV="$candidate"
            return 0
        fi
    done
    return 1
}

bootstrap_uv() {
    UV_YES="$YES"
    # Opened, not just tested with -r: in a container or under a service
    # /dev/tty is readable by mode but has no terminal behind it.
    if [ "$UV_YES" != 1 ] && ( : < /dev/tty ) 2>/dev/null; then
        printf 'install: install uv (https://astral.sh/uv) to provide free-threaded Python 3.14t? [Y/n] '
        read -r answer < /dev/tty || answer="n"
        case "$answer" in
            ''|y|Y|yes|YES) UV_YES=1 ;;
        esac
    fi
    if [ "${SKIP_PATH_FT:-0}" = 1 ]; then NONE_FOUND="no uv was found"
    else NONE_FOUND="no python3.14t or uv was found"; fi
    if [ "$UV_YES" != 1 ]; then
        FT_WHY="$NONE_FOUND and consent to install uv was not given"
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        FT_WHY="$NONE_FOUND and curl, needed to fetch uv, is missing"
        return 1
    fi
    say "installing uv (https://astral.sh/uv; shell profiles left unmodified)"
    # Fetched to a file first: piped straight into sh, a failed download
    # runs an empty script that exits 0.
    UV_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/gpuwm-uv-install.XXXXXX") \
        || { FT_WHY="no temporary file for the uv installer could be made"; return 1; }
    if ! curl -LsSf https://astral.sh/uv/install.sh -o "$UV_SCRIPT"; then
        rm -f "$UV_SCRIPT"
        FT_WHY="the uv installer could not be downloaded from https://astral.sh/uv/install.sh"
        return 1
    fi
    if ! env UV_NO_MODIFY_PATH=1 sh "$UV_SCRIPT"; then
        rm -f "$UV_SCRIPT"
        FT_WHY="the uv installer failed (see its output above)"
        return 1
    fi
    rm -f "$UV_SCRIPT"
    find_uv || { FT_WHY="uv installed but could not be found afterwards"; return 1; }
}

# uv, asked for a managed interpreter only when SKIP_PATH_FT=1: by default
# `uv python find` answers with a system python3.14t first, which on the
# retry is the very one that could not make a venv (seen 2026-10-08 with
# deadsnakes' python3.14-nogil: uv found /usr/bin/python3.14t again).
uv_python() {
    if [ "${SKIP_PATH_FT:-0}" = 1 ]; then
        UV_PYTHON_PREFERENCE=only-managed "$UV" python "$@"
    else
        "$UV" python "$@"
    fi
}

provision_free_threaded() {
    # Sets FT_PYTHON on success and FT_WHY on failure.  SKIP_PATH_FT=1
    # passes over a python3.14t on PATH (one that could not make a venv)
    # and takes only a uv-managed 3.14t.
    if [ "${SKIP_PATH_FT:-0}" != 1 ] \
        && command -v python3.14t >/dev/null 2>&1 && is_free_threaded python3.14t; then
        FT_PYTHON=python3.14t
        return 0
    fi
    find_uv || bootstrap_uv || return 1
    if FT_PYTHON=$(uv_python find 3.14t 2>/dev/null) && is_free_threaded "$FT_PYTHON"; then
        return 0
    fi
    say "installing free-threaded CPython 3.14t with uv"
    if ! UV_PYTHON_INSTALL_BIN=0 uv_python install 3.14t; then
        FT_WHY="uv could not install Python 3.14t (see its output above)"
        return 1
    fi
    if FT_PYTHON=$(uv_python find 3.14t 2>/dev/null) && is_free_threaded "$FT_PYTHON"; then
        return 0
    fi
    FT_WHY="uv installed Python 3.14t but did not report a free-threaded interpreter"
    return 1
}

locked_python() {
    if command -v python3 >/dev/null 2>&1; then
        LOCKED_PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        LOCKED_PYTHON=python
    else
        fail "no python3/python on PATH (Python 3.11+ is required)"
    fi
}

# A venv that fails part-way leaves .venv/bin/python behind, and the next
# run would take the "reused" route and stop at the pip upgrade, so a
# failed attempt is removed.  THE BREAKAGE THIS PREVENTS: a distro
# python3.14t without its venv package (deadsnakes without
# python3.14-venv, for one) has no ensurepip, so `-m venv` stops after
# writing bin/python; under `set -eu` that ended the whole install.
make_venv() {
    say "creating .venv with $1"
    if "$1" -m venv .venv; then
        return 0
    fi
    rm -rf .venv
    return 1
}

VENV_PY=.venv/bin/python
VENV_ROUTE=reused
VENV_FT=0
if [ -x "$VENV_PY" ]; then
    say "reusing the existing .venv"
    if is_free_threaded "$VENV_PY"; then
        VENV_FT=1
        say "the existing .venv is free-threaded"
    elif [ -z "$EXPLICIT_PYTHON" ]; then
        warn_locked "the existing .venv's Python" \
            "that .venv was made before this installer provisioned 3.14t, or with another interpreter" \
            "remove .venv (rm -rf .venv) and re-run ./install.sh."
    fi
else
    if [ -n "$EXPLICIT_PYTHON" ]; then
        PYTHON="$EXPLICIT_PYTHON"
        VENV_ROUTE=explicit
    elif provision_free_threaded; then
        PYTHON="$FT_PYTHON"
        VENV_ROUTE=free-threaded
    else
        locked_python
        PYTHON="$LOCKED_PYTHON"
        VENV_ROUTE=locked
    fi
    if ! make_venv "$PYTHON"; then
        [ "$VENV_ROUTE" = free-threaded ] \
            || fail "$PYTHON could not create .venv (see its output above; Ubuntu ships the venv module as python3-venv)"
        VENV_WHY="$PYTHON could not create a venv (see its output above; a distro python3.14t without its venv package does this)"
        VENV_MADE=0
        # A system python3.14t that cannot make a venv does not rule out
        # the 3.14t uv manages, which carries its own ensurepip.
        SKIP_PATH_FT=1
        if provision_free_threaded; then
            PYTHON="$FT_PYTHON"
            if make_venv "$PYTHON"; then
                VENV_MADE=1
            else
                VENV_WHY="$VENV_WHY, and neither could $PYTHON"
            fi
        else
            VENV_WHY="$VENV_WHY, and $FT_WHY"
        fi
        if [ "$VENV_MADE" != 1 ]; then
            FT_WHY="$VENV_WHY"
            FT_REMEDY="install that Python's venv package (python3.14-venv on deadsnakes) or uv (https://docs.astral.sh/uv/), remove .venv (rm -rf .venv) and re-run ./install.sh."
            locked_python
            PYTHON="$LOCKED_PYTHON"
            VENV_ROUTE=locked
            make_venv "$PYTHON" \
                || fail "$PYTHON could not create .venv either (see its output above; Ubuntu ships the venv module as python3-venv)"
        fi
    fi
    if is_free_threaded "$VENV_PY"; then
        VENV_FT=1
        say ".venv is free-threaded: multi-card slab threads run at once"
    elif [ "$VENV_ROUTE" = explicit ]; then
        warn_locked "$PYTHON" "it was chosen with --python or WOOF_PYTHON" \
            "remove .venv and re-run without --python and WOOF_PYTHON."
    else
        warn_locked "$PYTHON" "${FT_WHY:-no free-threaded interpreter was found}" \
            "${FT_REMEDY:-install uv (https://docs.astral.sh/uv/) or a python3.14t, remove .venv (rm -rf .venv) and re-run ./install.sh.}"
    fi
fi
# ------------------------------------------------------------ CUDA major
# CuPy ships ONE wheel per CUDA major and pip cannot detect the major, so
# the extra has to name it.  Through 1.8.0 this line pasted
# `.[gpu,render]` unconditionally -- the cu12 wheel -- so a CUDA-13-only
# box got a CuPy that imports cleanly, compiles kernels, and then dies at
# its first cuBLAS load, with nothing in the install saying so.  Read the
# major off the driver instead, and when it cannot be read, SAY that
# rather than defaulting in silence.
# The header label is not one string: Linux drivers print
# "CUDA Version: 12.4" and the Windows driver on the reference box
# prints "CUDA UMD Version: 13.3".  Matching only the first spelling
# read as "no NVIDIA driver" on a machine that plainly had one.
detect_cuda_major() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    detected=$(nvidia-smi 2>/dev/null |
        sed -n 's/.*CUDA[A-Za-z ]*Version: *\([0-9][0-9]*\).*/\1/p' |
        head -n 1)
    case "$detected" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s\n' "$detected" ;;
    esac
}

if [ -n "$CUDA_MAJOR" ]; then
    say "CUDA major $CUDA_MAJOR was given on the command line"
else
    CUDA_MAJOR=$(detect_cuda_major || true)
    if [ -n "$CUDA_MAJOR" ]; then
        say "nvidia-smi reports CUDA $CUDA_MAJOR"
    fi
fi
case "$CUDA_MAJOR" in
    12|13) GPU_EXTRA="gpu-cu$CUDA_MAJOR" ;;
    *)
        GPU_EXTRA="gpu-cu12"
        say "the box's CUDA major could not be read (no nvidia-smi, or no"
        say "driver answered), so this install falls back to [gpu-cu12]."
        say "IF THIS BOX'S CUDA IS 13-ONLY THAT WHEEL IS WRONG: it will"
        say "import fine and fail at the first cuBLAS load.  Re-run with"
        say "--cuda 13 in that case; woof doctor judges the pairing at"
        say "the end of this script either way."
        ;;
esac
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
install_python_packages() {
    if [ "$VENV_FT" = 1 ]; then PREFER_BINARY=--prefer-binary; else PREFER_BINARY=; fi
    "$VENV_PY" -m pip install --upgrade pip || return 1
    CUPY_WANTED="cupy-cuda${GPU_EXTRA#gpu-cu}x"
    # A pip that cannot list the .venv stops here: reading its failure as
    # "no CuPy" would install beside a build nobody saw.
    if ! CUPY_LISTED=$("$VENV_PY" -m pip list --format=freeze --disable-pip-version-check 2>/dev/null); then
        fail "pip could not list the packages in .venv, so this install cannot tell which CuPy it holds (run $VENV_PY -m pip list to see why)"
    fi
    CUPY_HAVE=$(printf '%s\n' "$CUPY_LISTED" |
        tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz' |
        sed -n 's/^\(cupy\(-cuda[0-9][0-9]*x\)\{0,1\}\) *[=@].*/\1/p')
    CUPY_OTHER=0
    for dist in $CUPY_HAVE; do
        [ "$dist" = "$CUPY_WANTED" ] || CUPY_OTHER=1
    done
    if [ "$CUPY_OTHER" = 1 ]; then
        say "this .venv holds another CUDA major's CuPy ($(echo $CUPY_HAVE));"
        say "removing every CuPy build so $CUPY_WANTED installs clean"
        # shellcheck disable=SC2086
        "$VENV_PY" -m pip uninstall -y $CUPY_HAVE || return 1
    fi
    say "installing the matching recast-woof-data companion from this checkout (editable)"
    # shellcheck disable=SC2086
    "$VENV_PY" -m pip install $PREFER_BINARY -e recast-woof-data || return 1
    say "installing woof with the [$GPU_EXTRA,render] extras (editable)"
    # shellcheck disable=SC2086
    "$VENV_PY" -m pip install $PREFER_BINARY -e ".[$GPU_EXTRA,render]" || return 1
}

# A dependency that will not install under 3.14t (no cp314t wheel and an
# sdist that does not build) must not cost the user the whole install: a
# .venv this run made on 3.14t by default is remade on python3 and the
# install goes on, saying what that costs.  A reused .venv, or one made
# from an interpreter the user named, is never removed.
if ! install_python_packages; then
    [ "$VENV_ROUTE" = free-threaded ] \
        || fail "pip could not install woof into .venv (see its output above)"
    locked_python
    warn_locked "$LOCKED_PYTHON" \
        "a dependency did not install under 3.14t (see pip's output above), so this run remade .venv on $LOCKED_PYTHON" \
        "once that package installs under 3.14t, remove .venv (rm -rf .venv) and re-run ./install.sh."
    rm -rf .venv
    make_venv "$LOCKED_PYTHON" \
        || fail "$LOCKED_PYTHON could not create .venv (see its output above; Ubuntu ships the venv module as python3-venv)"
    VENV_FT=0
    install_python_packages \
        || fail "pip could not install woof into .venv (see its output above)"
fi

# ------------------------------------------------------- externalized tables
# The two largest Thompson tables ship as GitHub release assets rather
# than in the wheel (freezeH2O.dat, 243 MiB, is not in git either);
# fetch-tables downloads only what is absent and verifies SHA-256
# against the packaged pins before installing.
if [ "$NO_FETCH_TABLES" = 1 ]; then
    say "skipping the externalized table fetch (--no-fetch-tables);"
    say "woof doctor prints the exact fetch command while they are missing"
else
    say "staging the externalized Thompson tables (downloads only what"
    say "is absent -- ~243 MiB from a checkout; SHA-256 verified)"
    .venv/bin/woof fetch-tables
    # The default mixed-layer ocean's 12 WOA18 MLD months (~85 MB), staged
    # now so no forecast reads NCEI at start.  Not fatal here: a forecast
    # still fetches its month on a miss, and doctor names the gap.
    say "staging the WOA18 ocean mixed-layer climatology (~85 MB)"
    .venv/bin/woof fetch-tables --ocean --ocean-only || say "WOA18 ocean climatology not staged yet; re-run: woof fetch-tables --ocean"
fi

# ------------------------------------------------------- offline Rust build
say "building the vendored Rust GRIB bridges (offline, locked)"
( cd tools/grib1_bridge && cargo build --release --locked --offline )
if [ "$NO_RENDER" = 1 ]; then
    say "skipping the tools/rustwx render engine (--no-render);"
    say "stage it with woof fetch-bridges, or request --engine matplotlib"
else
    say "building the vendored render engine in tools/rustwx (offline,"
    say "locked; the long pole of install -- skip with --no-render)"
    ( cd tools/rustwx && cargo build --release --locked --offline )
    # The sounding renderer draws on the GPU through the system Vulkan loader
    # (libvulkan1) and falls back to its CPU rasteriser without it; headless
    # server images often leave the loader out, so name it here.
    if [ "$(uname -s)" = Linux ] && ! { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q "libvulkan.so.1"; then
        say "no Vulkan loader (libvulkan.so.1): woof soundings will draw on the CPU;"
        say "for GPU rendering: sudo apt-get install -y libvulkan1 (dnf: vulkan-loader)"
    fi
fi
say "building the regular-grid Zarr reader (offline, locked)"
( cd tools/zarr_bridge && cargo build --release --locked --offline )
# The optional Icechunk output store writer; its crates are vendored with rustwx.
say "building the Icechunk store writer in tools/rw_icechunk (offline, locked)"
( cd tools/rw_icechunk && cargo build --release --locked --offline )
# Every mapped source decodes in gpuwm_mapped_engine and every radar ingest
# dealiases through region_global_dealias by default; a checkout that skips
# either build reports both MISSING and cannot run those default routes.
say "building the mapped-source decode engine in tools/rw_wps (offline, locked)"
( cd tools/rw_wps && cargo build --release --locked --offline )
say "building the velocity dealiasing library (offline, locked)"
( cd tools/region_global_dealias && cargo build --release --locked --offline )
# nextgen_output (librw_nextgen_output) writes a default-on hydro forecast's history, restart and
# products; its crates.io dependencies resolve to the vendored rustwx sources.
for workspace in nextgen_hydrofabric nextgen_forcing nextgen_verify hydro_world hydro_world_params nextgen_reservoir_prepare hydro_world_init ocean_prepare rw-bathy wind-table nextgen_output; do
    # These workspaces include locked registry dependencies not all vendored
    # with rustwx. Populate Cargo's cache before the offline build.
    say "fetching locked NextGen dependencies in tools/$workspace"
    ( cd "tools/$workspace" && cargo fetch --locked )
    say "building NextGen tools/$workspace (offline, locked)"
    ( cd "tools/$workspace" && cargo build --release --locked --offline )
done
if [ "$RUST_PINNED" = 1 ]; then
    say "built with Rust $RUST_MIN; a later hand-run cargo build in tools/ needs"
    say "it too: RUSTUP_TOOLCHAIN=$RUST_MIN cargo build, or rustup default $RUST_MIN"
fi

# ------------------------------------------------------------------ doctor
# Doctor judges the environment this script just made, as it stands once
# activated: without .venv/bin on PATH its console-script check reports a
# gap this script created and the install exits nonzero for it.
say "running woof doctor"
# Doctor's lines are kept so the closing line can say what it found: a
# doctor that exits 0 can still name a MISSING dataset (the geography a
# forecast is refused without is an opt-in download, not a broken install),
# and "doctor is clean" over that line sent a reader straight into the
# refusal at their first `woof go` (2.9.2 box door test).  The exit status
# is doctor's own, unchanged.
doctor_log="$(mktemp)"
doctor_status="$(mktemp)"
{ status=0; PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/woof doctor || status=$?; echo "$status" > "$doctor_status"; } 2>&1 | tee "$doctor_log"
status="$(cat "$doctor_status")"
rm -f "$doctor_status"
if [ "$status" -eq 0 ]; then
    if grep -q '^  MISSING' "$doctor_log"; then
        say "done -- doctor passes, with these gaps still to stage:"
        grep '^  MISSING' "$doctor_log" | sed 's/^  MISSING /install:   missing /' | cut -c1-160
        say "each gap's line above prints its remedy (geography: woof fetch-geog);"
        say "a forecast is refused at its start until the geography is staged."
    else
        say "done -- doctor is clean."
    fi
    say "Activate with: . .venv/bin/activate"
    rm -f "$doctor_log"
else
    rm -f "$doctor_log"
    say "install steps completed; doctor reports gaps above (each line"
    say "prints its own remedy).  Re-run ./install.sh any time."
    exit "$status"
fi
