"""The Earth2Studio interpreter and TempestExtremes tools that the cyclone and Earth-2 doors run.

Earth2Studio and its torch stack never load in the woof process. The doors run their runners in a
separate interpreter, and ``woof tc-tracks --backend tempestextremes`` calls the TempestExtremes
executables. Breakage this module prevents (2.9.3 door 6 on G12, 2026-10-20): on a fresh install
nothing provided either one. ``woof tc-tracks`` refused on every backend, after it had already done
its extraction work. The GPU backends said only "WOOF_EARTH2_PYTHON must name the Earth2Studio
interpreter", tempestextremes printed a raw ``[Errno 2] ... 'DetectNodes'``, ``woof earth2-diagnose``
stopped the same way, and ``woof doctor`` said nothing about any of it.

This is now the one place that finds both, and builds them from pins when they are missing:

* interpreter: ``WOOF_EARTH2_PYTHON`` when it is set (used as named, never changed), otherwise the
  managed environment, otherwise this interpreter if it can import earth2studio;
* ``DetectNodes`` / ``StitchNodes``: ``WOOF_DETECTNODES`` / ``WOOF_STITCHNODES``, otherwise PATH,
  otherwise the managed environment.

The managed environment is built by ``woof setup --with-earth2``, or on first use by a door that
needs it (``WOOF_EARTH2_FETCH=0`` turns that off, and the door then refuses with the remedy). Its pins
are ``authorities/earth2-env.v1.json`` plus ``runtime`` in ``earth2-models.v1.json``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

AUTHORITIES = Path(__file__).with_name("authorities")
TABLE = AUTHORITIES / "earth2-env.v1.json"
MODELS = AUTHORITIES / "earth2-models.v1.json"
PYTHON_ENV = "WOOF_EARTH2_PYTHON"
ROOT_ENV = "WOOF_EARTH2_ENV"
FETCH_ENV = "WOOF_EARTH2_FETCH"
RECIPE_ENV = "WOOF_EARTH2_RECIPE"
SETUP_COMMAND = "woof setup --with-earth2"
TEMPEST_TOOLS = ("DetectNodes", "StitchNodes")


def table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def runtime() -> dict:
    return json.loads(MODELS.read_text(encoding="utf-8"))["runtime"]


def pinned_version() -> str:
    return runtime()["pinned_version"]


def root() -> Path:
    named = os.environ.get(ROOT_ENV)
    if named:
        return Path(named).expanduser().resolve()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "woof" / "earth2-env"


def _windows() -> bool:
    return os.name == "nt"


def managed_python() -> Path:
    venv = root() / "venv"
    return venv / "Scripts" / "python.exe" if _windows() else venv / "bin" / "python"


def managed_tool_dir() -> Path:
    prefix = root() / "tempest"
    return prefix / "Library" / "bin" if _windows() else prefix / "bin"


def fetch_allowed() -> bool:
    return os.environ.get(FETCH_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _extras(extras) -> list[str]:
    return [e for e in dict.fromkeys(x for x in (extras or ()) if x)]


def install_line(extras=()) -> str:
    wanted = _extras(extras)
    spec = "earth2studio" + (f"[{','.join(wanted)}]" if wanted else "") + f"=={pinned_version()}"
    return f"pip install '{spec}'"


def required_modules(extras=()) -> list[str]:
    modules = table()["extra_modules"]
    names = list(modules[""])
    for extra in _extras(extras):
        names.extend(modules.get(extra, ()))
    return list(dict.fromkeys(names))


_PROBE = (
    "import importlib.metadata as m, importlib.util as u, json, sys\n"
    "wanted = json.loads(sys.argv[1])\n"
    "try:\n    version = m.version('earth2studio')\nexcept Exception:\n    version = None\n"
    "print(json.dumps({'version': version, 'missing': [n for n in wanted if u.find_spec(n) is None]}))\n"
)


def probe(python: str, extras=(), *, exact: bool = False) -> tuple[bool, str]:
    """Does ``python`` hold earth2studio (the pinned version when ``exact``) and the extras' modules?

    Reads package metadata and module specs only; nothing is imported, so this costs a
    subprocess start, not a torch load.
    """
    try:
        done = subprocess.run([str(python), "-c", _PROBE, json.dumps(required_modules(extras))],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"cannot run it ({error})"
    if done.returncode != 0:
        return False, f"probe exited {done.returncode}: {done.stderr.strip()[-300:]}"
    found = json.loads(done.stdout.strip().splitlines()[-1])
    if found["version"] is None:
        return False, "earth2studio is not installed"
    if exact and found["version"] != pinned_version():
        return False, f"has earth2studio {found['version']}, needs {pinned_version()}"
    if found["missing"]:
        return False, (f"lacks {', '.join(found['missing'])} (earth2studio extra "
                       f"[{','.join(_extras(extras))}])")
    return True, f"earth2studio {found['version']}"


def find_python(extras=(), *, exact: bool = False) -> tuple[str | None, list[str]]:
    """The interpreter to run Earth2Studio in, or ``None`` plus why each candidate failed."""
    named = os.environ.get(PYTHON_ENV)
    if named:
        candidates = [named]
    else:
        candidates = [str(p) for p in (managed_python(),) if p.exists()] + [sys.executable]
    notes = []
    for candidate in candidates:
        ok, why = probe(candidate, extras, exact=exact)
        if ok:
            return candidate, notes
        notes.append(f"{candidate}: {why}")
    return None, notes


def remedy(extras=()) -> str:
    return (f"run `{SETUP_COMMAND}` (builds the managed Earth2Studio environment at {root()}), or set "
            f"{PYTHON_ENV} to a Python where `{install_line(extras)}` is installed")


def earth2_python(door: str, extras=(), *, exact: bool = False, provision: bool = False,
                  progress=None) -> str:
    """Resolve the Earth2Studio interpreter for ``door``, building the managed one when allowed."""
    found, notes = find_python(extras, exact=exact)
    if found:
        return found
    named = os.environ.get(PYTHON_ENV)
    if named:
        raise ValueError(
            f"{door}: {PYTHON_ENV}={named} cannot run it: {notes[-1].split(': ', 1)[-1]}. Install "
            f"`{install_line(extras)}` into that interpreter, or unset {PYTHON_ENV} to use the "
            f"managed environment (`{SETUP_COMMAND}`)")
    if provision and fetch_allowed() and provision_supported():
        ensure(extras=list(table()["default_extras"]) + _extras(extras), tempest=False, progress=progress)
        found, more = find_python(extras, exact=exact)
        if found:
            return found
        notes += more
    reason = "; ".join(notes)
    blocked = (f" {FETCH_ENV}=0 stopped the first-use build." if provision and not fetch_allowed()
               else "")
    raise ValueError(
        f"{door} needs an Earth2Studio interpreter (earth2studio=={pinned_version()}"
        + (f" with the [{','.join(_extras(extras))}] extra" if _extras(extras) else "")
        + f") and none was found ({reason}).{blocked} Remedy: {remedy(extras)}")


def find_tool(name: str) -> str | None:
    """A TempestExtremes executable: its GPUWM_<NAME> override, PATH, then the managed env."""
    env = "GPUWM_" + name.upper()
    named = os.environ.get(env)
    if named:
        resolved = shutil.which(named) or (named if Path(named).is_file() else None)
        if resolved is None:
            raise ValueError(f"{env}={named} names no executable; point it at a real {name} or unset it")
        return resolved
    managed = managed_tool_dir() / (name + (".exe" if _windows() else ""))
    return shutil.which(name) or (str(managed) if managed.is_file() else None)


def tempest_remedy() -> str:
    spec = table()["tempestextremes"]
    return (f"run `{SETUP_COMMAND}` (installs {spec['conda_spec']} from {spec['channel']}, "
            f"{spec['licence']}, into {root() / 'tempest'}), or install TempestExtremes yourself "
            f"(`conda install -c {spec['channel']} tempest-extremes`) and put DetectNodes and StitchNodes "
            "on PATH or name them in WOOF_DETECTNODES / WOOF_STITCHNODES")


def tempest_tools(door: str, *, provision: bool = False, progress=None) -> tuple[str, str]:
    """``(DetectNodes, StitchNodes)``, building the managed copies when allowed."""
    found = [find_tool(name) for name in TEMPEST_TOOLS]
    if all(found):
        return found[0], found[1]
    if provision and fetch_allowed() and provision_supported():
        ensure(extras=(), python=False, tempest=True, progress=progress)
        found = [find_tool(name) for name in TEMPEST_TOOLS]
        if all(found):
            return found[0], found[1]
    missing = [name for name, path in zip(TEMPEST_TOOLS, found) if not path]
    blocked = (f" {FETCH_ENV}=0 stopped the first-use build." if provision and not fetch_allowed()
               else "")
    raise ValueError(
        f"{door} needs the TempestExtremes executables and {' and '.join(missing)} "
        f"{'is' if len(missing) == 1 else 'are'} not in WOOF_DETECTNODES/WOOF_STITCHNODES, on PATH, "
        f"or in {managed_tool_dir()}.{blocked} Remedy: {tempest_remedy()}")


# ---------------------------------------------------------------------------------------------
# Building the managed environment.


def provision_supported() -> bool:
    """The pinned recipes are Linux recipes (``ENV/bin/python``, linux micromamba assets)."""
    return sys.platform.startswith("linux") and _platform_key() in table()["tempestextremes"]["micromamba"]["assets"]


def _platform_key() -> str:
    machine = platform.machine().lower()
    return {"x86_64": "linux-64", "amd64": "linux-64", "aarch64": "linux-aarch64",
            "arm64": "linux-aarch64"}.get(machine, f"linux-{machine}")


def recipe_name() -> str:
    named = os.environ.get(RECIPE_ENV)
    recipes = runtime()["install_recipes"]
    if named:
        if named not in recipes:
            raise ValueError(f"{RECIPE_ENV}={named} is not one of {', '.join(sorted(recipes))}")
        return named
    from woof.doctor import _driver_cuda_major
    major = _driver_cuda_major()
    by_major = table()["recipe_by_driver_major"]
    return by_major["13"] if major is None or major >= 13 else by_major["12"]


def _say(progress, text: str) -> None:
    (progress or (lambda line: print(line, file=sys.stderr, flush=True)))(text)


def _run(argv, log: Path, env=None) -> None:
    argv = [str(a) for a in argv]
    with log.open("a") as stream:
        stream.write("$ " + shlex.join(argv) + "\n")
        stream.flush()
        done = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if done.returncode != 0:
        tail = log.read_text(errors="replace")[-3000:]
        raise RuntimeError(f"`{shlex.join(argv)}` exited {done.returncode}; log {log}\n{tail}")


def _uv(base: Path, log: Path) -> str:
    found = shutil.which("uv")
    if found:
        return found
    target = base / ".uv"
    exe = target / "bin" / "uv"
    if not exe.is_file():
        _run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
              "--target", target, table()["uv_bootstrap"]["package"]], log)
    if not exe.is_file():
        raise RuntimeError(f"installing {table()['uv_bootstrap']['package']} left no {exe}")
    return str(exe)


def _build_python(extras, progress, log: Path) -> dict:
    base = root()
    env = base / "venv"
    recipe_key = recipe_name()
    recipe = runtime()["install_recipes"][recipe_key]
    substitutes = table().get("extra_substitutes", {}).get(recipe_key, {})
    wanted = _extras(extras)
    pip_extras = [e for e in wanted if not isinstance(substitutes.get(e), list)]
    packages = [p for e in wanted if isinstance(substitutes.get(e), list) for p in substitutes[e]]
    uv = _uv(base, log)
    start = time.monotonic()
    _say(progress, f"earth2: building {env} (recipe {recipe_key}, earth2studio=={pinned_version()}"
         f"[{','.join(wanted)}]); log {log}")
    toolkit = None
    for command in recipe["commands"]:
        argv = shlex.split(command.replace("{extra}", ",".join(pip_extras)))
        if argv[:2] == ["uv", "venv"] and managed_python().exists():
            continue
        argv = [uv if a == "uv" else a.replace("ENV", str(env)) for a in argv]
        if not pip_extras:
            argv = [a.replace("earth2studio[]", "earth2studio") for a in argv]
        if packages and any(a.startswith("earth2studio") for a in argv):
            argv += packages
            toolkit = _compiler_toolkit(uv, recipe, env, log)
        _run(argv, log, env=toolkit)
    return dict(python=str(managed_python()), recipe=recipe_key, extras=wanted,
                substituted_packages=packages, seconds=time.monotonic() - start)


def _compiler_toolkit(uv: str, recipe: dict, env: Path, log: Path):
    """The build environment for substituted packages that compile a CUDA extension.

    A recipe with ``compiler_packages`` (cu13) installs that private CUDA compiler into the venv
    and builds against it (``cuda_home``), as its ``before_model_extra`` says. Breakage it prevents
    (2.9.4 follow-up, test host 2026-10-21): torch-harmonics, which FCN3 needs, built against the box's
    /usr/local/cuda 12.8 and refused ("detected CUDA version 12.8 mismatches ... 13.0") under the
    cu13 torch. A recipe without them (cu12) builds against whatever CUDA torch finds. Returns the
    environment for the install, or ``None`` for this process's own.
    """
    packages = recipe.get("compiler_packages")
    if not packages:
        return None
    _run([uv, "pip", "install", "--python", managed_python(), *packages], log)
    home = Path(recipe["cuda_home"].replace("ENV", str(env)))
    if _glibc_rsqrt_compat(home):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"earth2: {home}/include/crt/math_functions.h: rsqrt/rsqrtf take glibc 2.42's "
                         "noexcept (the declaration CUDA 13.2 ships)\n")
    library = home / "lib"
    runtime_so = library / "libcudart.so"
    versioned = sorted(library.glob("libcudart.so.[0-9]*"))
    if not runtime_so.exists() and versioned:
        runtime_so.symlink_to(versioned[0].name)
    build = dict(os.environ, CUDA_HOME=str(home), CUDA_PATH=str(home))
    build["PATH"] = str(home / "bin") + os.pathsep + build.get("PATH", "")
    return build


#: CUDA 13.2's own guard: glibc 2.42 declares the C23 rsqrt/rsqrtf noexcept, and the device declarations
#: must agree.
_RSQRT_GUARD = """/* gpuwm earth2_env: the glibc 2.42+ rsqrt guard CUDA 13.2 ships, for an earlier 13.x header. */
#if !defined(_NV_RSQRT_SPECIFIER)
#  if defined(__GNUC__) && !defined(__ANDROID__) && !defined(__QNX__) && !defined(__APPLE__) && !defined(__HORIZON__)
#    include <features.h>
#    if defined(__GLIBC_PREREQ)
#      if __GLIBC_PREREQ(2,42)
#        define _NV_RSQRT_SPECIFIER noexcept(true)
#      endif
#    endif
#  endif
#  if !defined(_NV_RSQRT_SPECIFIER)
#    define _NV_RSQRT_SPECIFIER
#  endif
#endif
"""
_RSQRT_DECLARATIONS = ("rsqrt(double x);", "rsqrtf(float x);")


def _glibc_rsqrt_compat(home: Path) -> bool:
    """Give the private compiler's rsqrt/rsqrtf declarations glibc 2.42's noexcept, as CUDA 13.2 does.

    Breakage it prevents (2.9.4 R15, RTX 4090 and RTX 5070 Ti 2026-10-22, Ubuntu 26.04, glibc 2.43): with the
    pinned 13.0.88 compiler, torch-harmonics (FCN3) failed "exception specification is incompatible
    with that of previous function rsqrt" in bits/mathcalls.h, so ``woof setup --with-earth2`` could
    not build the environment. The compiler stays the one torch's 13.0 headers match: a 13.2 nvcc
    passes this header but its CCCL refuses torch's 13.0 runtime headers ("CUDA compiler and CUDA
    toolkit headers are incompatible"). Only the two declarations change; a header that already
    carries the guard (13.2 and later) is left as it is. True when the header was changed.
    """
    header = home / "include" / "crt" / "math_functions.h"
    if not header.is_file():
        return False
    text = header.read_text(encoding="utf-8")
    if "_NV_RSQRT_SPECIFIER" in text:
        return False
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines)
            if line.lstrip().startswith("extern") and line.rstrip().endswith(_RSQRT_DECLARATIONS)]
    if len(hits) != len(_RSQRT_DECLARATIONS):
        return False
    for i in hits:
        body = lines[i].rstrip()
        lines[i] = body[:-1] + " _NV_RSQRT_SPECIFIER;" + lines[i][len(body):]
    lines.insert(hits[0], _RSQRT_GUARD)
    header.write_text("".join(lines), encoding="utf-8")
    return True


def _download(url: str, sha256: str, destination: Path) -> None:
    part = destination.with_suffix(".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-earth2-env/1"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as stream:
        for block in iter(lambda: response.read(1 << 20), b""):
            digest.update(block)
            stream.write(block)
    if digest.hexdigest() != sha256:
        part.unlink()
        raise RuntimeError(f"{url}: sha256 {digest.hexdigest()} is not the pinned {sha256}")
    part.chmod(0o755)
    part.replace(destination)


def _build_tempest(progress, log: Path) -> dict:
    base = root()
    spec = table()["tempestextremes"]
    asset = spec["micromamba"]["assets"][_platform_key()]
    start = time.monotonic()
    micromamba = base / ".micromamba" / "micromamba"
    micromamba.parent.mkdir(parents=True, exist_ok=True)
    if not micromamba.is_file():
        _download(asset["url"], asset["sha256"], micromamba)
    prefix = base / "tempest"
    if prefix.exists():
        shutil.rmtree(prefix)
    _say(progress, f"earth2: installing {spec['conda_spec']} ({spec['licence']}) into {prefix}; log {log}")
    _run([micromamba, "create", "--yes", "--quiet", "--prefix", prefix, "--root-prefix", base / ".mamba",
          "--override-channels", "--channel", spec["channel"], spec["conda_spec"]], log)
    missing = [t for t in spec["executables"] if not (managed_tool_dir() / t).is_file()]
    if missing:
        raise RuntimeError(f"{spec['conda_spec']} installed no {', '.join(missing)} in {managed_tool_dir()}")
    return dict(prefix=str(prefix), spec=spec["conda_spec"], licence=spec["licence"],
                micromamba=spec["micromamba"]["version"], seconds=time.monotonic() - start)


@contextlib.contextmanager
def _lock(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".lock").open("w") as handle:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:
            pass
        yield


def ensure(extras=None, *, python: bool = True, tempest: bool = True, progress=None) -> dict:
    """Build whatever of the managed environment is missing; a complete one is verified and kept.

    The interpreter (torch + earth2studio wheels) and TempestExtremes (conda-forge) are independent
    downloads, so they build at the same time. Breakage the lock prevents: two doors started on a
    fresh box would both write one venv.
    """
    if not provision_supported():
        raise ValueError(f"the managed Earth2Studio environment is built from Linux recipes; on "
                         f"{sys.platform}/{platform.machine()} set {PYTHON_ENV} and put DetectNodes/"
                         "StitchNodes on PATH instead")
    wanted = _extras(table()["default_extras"] if extras is None else extras)
    base = root()
    with _lock(base):
        jobs = {}
        if python and not (managed_python().exists() and probe(str(managed_python()), wanted, exact=True)[0]):
            jobs["python"] = lambda: _build_python(wanted, progress, base / "build-python.log")
        if tempest and not all((managed_tool_dir() / t).is_file() for t in TEMPEST_TOOLS):
            jobs["tempest"] = lambda: _build_tempest(progress, base / "build-tempest.log")
        if not jobs:
            return dict(root=str(base), built=[])
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {name: pool.submit(job) for name, job in jobs.items()}
            results, errors = {}, []
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as error:  # every build reports, then the first failure raises
                    errors.append(f"{name}: {error}")
        receipt = dict(schema="gpuwm.earth2-env-receipt.v1", root=str(base), built=sorted(results),
                       results=results, errors=errors, total_seconds=time.monotonic() - start,
                       parallelism=f"{len(jobs)} concurrent builds (interpreter wheels; conda-forge tools)")
        receipt_path = base / "earth2-env.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        receipt_path.write_text(json.dumps(dict(previous, **receipt, history=previous.get("history", [])
                                                + [dict(built=sorted(results), errors=errors,
                                                        total_seconds=receipt["total_seconds"])]),
                                           indent=2) + "\n")
        if errors:
            raise ValueError("building the managed Earth2Studio environment failed: " + " | ".join(errors))
        _say(progress, f"earth2: built {', '.join(sorted(results))} in {receipt['total_seconds']:.0f} s "
             f"at {base}")
        return receipt


__all__ = ["earth2_python", "ensure", "find_python", "find_tool", "managed_python", "probe",
           "remedy", "root", "tempest_remedy", "tempest_tools"]
