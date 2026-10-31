"""Source-bound cu12/cu13 kernel packs, using the common hex pack contract.

Only CUDA driver orchestration lives here. Cache misses keep the ordinary compiler.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

DEFAULT_PACK = Path(__file__).resolve().parent / "data" / "kernel-pack.json"
#: Ada (sm_89: RTX 4090, L40S), Hopper (sm_90: H100, H200), Blackwell
#: datacenter (sm_100: B200) and Blackwell RTX (sm_120: RTX 5090, RTX PRO 6000).
ARCHITECTURES = ("sm_89", "sm_90", "sm_100", "sm_120")
_BRIDGE_LOCK = threading.RLock()
_SITE_TOKEN = "@SITE@"
_PACKS: dict = {}
_WARNED: set = set()


def effective_options(options):
    """Apply the engine's compiler boundary when that engine provides one."""
    try:
        from woof.numerics import effective_options as engine_options
    except ModuleNotFoundError as error:
        if error.name != "woof.numerics":
            raise
        # Older supported engine wheels do not rewrite compiler options.
        return tuple(options)
    return engine_options(options)


def _warn_once(key, text):
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"woof: {text}", file=sys.stderr, flush=True)


def _site_roots():
    """The install roots CuPy writes into its own ``-I`` options (site-packages)."""
    try:
        import cupy
    except ImportError:
        return ()
    roots = []
    for package in (os.path.dirname(cupy.__file__), os.path.realpath(os.path.dirname(cupy.__file__))):
        root = os.path.dirname(package)
        if root and root not in roots:
            roots.append(root)
    return tuple(roots)


def portable_options(options, roots=None):
    """Options with the install's site-packages root replaced by a token.

    CuPy adds ``-I<site-packages>/cupy/_core/include`` (and the pip CUDA
    headers under ``<site-packages>/nvidia``) to every compile, so the same
    request from two venvs differs only in that prefix.  The headers under it
    are the bytes of the same pinned wheels, so the prefix is not part of what
    the compiler produces; an install whose headers come from somewhere else
    (a system toolkit) keeps its own path and misses by name.
    """
    roots = _site_roots() if roots is None else roots
    result = []
    for option in options:
        option = str(option)
        for root in roots:
            if option.startswith("-I" + root + "/"):
                option = "-I" + _SITE_TOKEN + option[2 + len(root):]
                break
        result.append(option)
    return result


def local_options(options, roots=None):
    roots = _site_roots() if roots is None else roots
    if not roots:
        return [str(option) for option in options]
    return [str(option).replace("-I" + _SITE_TOKEN, "-I" + roots[0], 1) for option in options]
_FIELDS = ("source", "options", "arch", "extra_source", "backend",
           "enable_cooperative_groups", "name_expressions", "log_stream",
           "cache_in_memory", "jitify", "to_ltoir")
#: cupy.cuda.compiler.compile_using_nvrtc, the route woof's direct-NVRTC
#: units (radiation, land, drag; woof.nvrtc_ptx_cache) take to keep
#: ``--ftz=false``.  It bypasses _compile_with_cache_cuda, so it is bridged
#: separately: 50 such modules on the default global suite.
_DIRECT_FIELDS = ("source", "options", "arch", "filename", "name_expressions",
                  "log_stream", "cache_in_memory", "jitify", "method")
DIRECT_ROUTE = "direct-nvrtc"
RAW_ROUTE = "raw-nvrtc"


def _distribution(names):
    """``name==version`` of the first installed distribution in ``names``, else None."""
    from importlib import metadata
    for name in names:
        try:
            return f"{name}=={metadata.version(name)}"
        except metadata.PackageNotFoundError:
            continue
    return None


def compiler_identity():
    import cupy
    from cupy.cuda import compiler
    from woof.certify.compile_platform import nvrtc_library

    # getVersion loads NVRTC without compiling a probe or using a device.
    version = list(compiler._get_nvrtc_version())
    library = nvrtc_library()
    if library.get("status") != "resolved":
        raise ValueError("loaded NVRTC library digest is unavailable")
    return {"cupy": cupy.__version__, "nvrtc": version,
            "cupy_package": _distribution(("cupy-cuda13x", "cupy-cuda12x", "cupy")),
            "nvrtc_package": _distribution(("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu13",
                                            "nvidia-cuda-nvrtc-cu12")),
            "nvrtc_library_sha256": library["sha256"],
            "cache_key": str(compiler._get_cupy_cache_key()),
            "use_ptx": bool(compiler._use_ptx), "use_pch": bool(compiler._use_pch),
            "debug": os.environ.get("CUPY_CUDA_COMPILE_WITH_DEBUG", "0")}


def source_hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def request_metadata(bound):
    # NVRTC never reads extra_source (CuPy hashes it into its cache key: the
    # concatenated CuPy headers, about 190 KB on every request).  Its digest
    # binds the row to the same headers at a fraction of the pack size.
    extra = bound.get("extra_source")
    return {"extra_source": extra if extra is None or str(extra).startswith("sha256:")
            else "sha256:" + source_hash(extra),
            "backend": bound.get("backend", "nvrtc"),
            "enable_cooperative_groups": bool(bound.get("enable_cooperative_groups", False)),
            "name_expressions": list(bound["name_expressions"]) if bound.get("name_expressions") else None,
            "jitify": bool(bound.get("jitify", False)),
            "to_ltoir": bool(bound.get("to_ltoir", False))}


def direct_metadata(filename):
    return {"route": DIRECT_ROUTE, "filename": str(filename or "kern.cu")}


def raw_metadata(bound):
    """Identity of an already-finalized NVRTC request, including its aliases."""
    return {"route": RAW_ROUTE, "filename": str(bound.get("filename", "kern.cu")),
            "name_expressions": (list(bound["name_expressions"])
                                 if bound.get("name_expressions") else None),
            "cache_in_memory": bool(bound.get("cache_in_memory", False)),
            "method": bound.get("method")}


def recipe(entry):
    """Turn a common recorded compiler argument into an AOT build entry."""
    source = entry["source"]
    if entry.get("route") == RAW_ROUTE:
        if (entry.get("jitify") or entry.get("log_stream") is not None
                or entry.get("cache_in_memory") or entry.get("method") is not None):
            raise ValueError("recorded route requires a different compiler backend or memory cache")
        return {"module_key": "nvrtc.raw." + source_hash(source), "source": source,
                "source_sha256": source_hash(source),
                "options": portable_options(entry.get("options", [])),
                "compiler_route": RAW_ROUTE,
                "compile_request": raw_metadata(entry)}
    if entry.get("route") == DIRECT_ROUTE:
        if entry.get("name_expressions") or entry.get("jitify") or entry.get("method") is not None:
            raise ValueError("recorded route requires linking or a different compiler backend")
        return {"module_key": "nvrtc.direct." + source_hash(source), "source": source,
                "source_sha256": source_hash(source),
                "options": portable_options(entry.get("options", [])),
                "compiler_route": DIRECT_ROUTE,
                "compile_request": direct_metadata(entry.get("filename"))}
    metadata = request_metadata(entry)
    if (metadata["backend"] != "nvrtc" or metadata["jitify"]
            or metadata["to_ltoir"] or metadata["enable_cooperative_groups"]):
        raise ValueError("recorded route requires linking or a different compiler backend")
    return {"module_key": "cupy.cache." + source_hash(source), "source": source,
            "source_sha256": source_hash(source),
            "options": portable_options(entry.get("options", [])),
            "compile_request": metadata}


@contextmanager
def capture_requests():
    """Record the complete real CuPy requests for a build catalog or a test."""
    from cupy.cuda import compiler
    original = compiler._compile_with_cache_cuda
    rows = {}
    def recording(*args, **kwargs):
        bound = dict(zip(_FIELDS, args))
        bound.update(kwargs)
        if isinstance(bound.get("source"), str):
            entry = {"source": bound["source"], "options": list(effective_options(bound.get("options") or ())),
                     "arch": bound.get("arch"), **request_metadata(bound)}
            key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            rows[key] = entry
        return original(*args, **kwargs)
    compiler._compile_with_cache_cuda = recording
    try:
        yield rows
    finally:
        compiler._compile_with_cache_cuda = original


class Pack:
    def __init__(self, path=None, identity=None):
        self.rows = []
        self.compiler = None
        self.reason = None
        self._by_key = {}
        self._architectures = set()
        self._header_snapshots = {}
        if os.environ.get("WOOF_AOT", "").lower() in ("off", "0", "false", "no"):
            self.reason = "aot_disabled"
            return
        requested = path or os.environ.get("WOOF_AOT_PACK")
        path = Path(requested or DEFAULT_PACK)
        from woof.kernel_pack import compiler_of, read
        try:
            measured = compiler_identity() if identity is None else identity
        except Exception as error:
            self.reason = "compiler_identity_unavailable"
            self.detail = type(error).__name__
            _warn_once(("identity",), "the CUDA compiler identity could not be measured "
                       f"({self.detail}); the prebuilt kernel pack is not used and modules compile on first use")
            return
        if not requested:
            # The default GPU extra uses CUDA 12, while CUDA 13 remains
            # supported. Ship a compiler-family sidecar without replacing
            # the other family's pack. The exact identity and every entry
            # digest are still checked by read below; the name is only a
            # selection hint, never permission to load incompatible cubins.
            version = measured.get("nvrtc")
            if isinstance(version, (list, tuple)) and version:
                candidate = path.with_name(f"kernel-pack-cuda{int(version[0])}.json")
                if candidate.is_file():
                    path = candidate
        if not path.is_file():
            self.reason = "pack_missing"
            _warn_once(("missing", str(path)),
                       f"no prebuilt kernel pack at {path}; retained images and CuPy's disk cache "
                       "remain available; missing requests compile through NVRTC")
            return
        try:
            stat = path.stat()
            # Some container filesystems give consecutive same-size writes
            # identical timestamps. Bind the memo to bytes as well, so an
            # edited pack cannot bypass its entry integrity checks.
            with path.open("rb") as stream:
                pack_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            memo = (str(path), stat.st_size, stat.st_mtime_ns, pack_digest,
                    json.dumps(measured, sort_keys=True))
            rows = _PACKS.get(memo)
            if rows is None:
                rows = read(path, measured)
                _PACKS[memo] = rows
            self.rows = rows
            self.compiler = measured
        except (OSError, ValueError, TypeError, KeyError, lzma.LZMAError) as error:
            self.reason = "pack_invalid_or_compiler_mismatch"
            self.detail = str(error)
            built = compiler_of(path)
            if not built:
                text = (f"the prebuilt kernel pack is unreadable ({self.detail}); retained images and "
                        "CuPy's disk cache remain available; missing requests compile through NVRTC")
            else:
                wanted = " ".join(str(built[key]) for key in ("cupy_package", "nvrtc_package")
                                  if built.get(key))
                text = ("the prebuilt kernel pack was built with "
                        f"{built.get('cupy_package') or 'CuPy ' + str(built.get('cupy'))} and "
                        f"{built.get('nvrtc_package') or 'NVRTC ' + str(built.get('nvrtc'))}; this install has "
                        f"{measured.get('cupy_package') or 'CuPy ' + str(measured.get('cupy'))} and "
                        f"{measured.get('nvrtc_package') or 'NVRTC ' + str(measured.get('nvrtc'))}; retained images and "
                        "CuPy's disk cache remain available; missing requests compile through NVRTC"
                        + (f". To start from the pack: pip install {wanted}" if wanted else ""))
            _warn_once(("mismatch", str(path)), text)
            return
        for row in self.rows:
            self._by_key.setdefault(row.get("module_key"), []).append(row)
            self._architectures.add(row.get("arch"))

    def select(self, module_key, source, options, sm, *, metadata=None):
        if self.reason:
            return None, self.reason
        if sm not in ARCHITECTURES or sm not in self._architectures:
            held = ", ".join(sorted(self._architectures)) or "no architecture"
            _warn_once(("arch", sm), f"the prebuilt kernel pack holds {held}, not this "
                       f"card's {sm}; matching retained images and CuPy's disk cache remain available; "
                       "missing requests compile through NVRTC")
            return None, "architecture_mismatch"
        modules = self._by_key.get(module_key)
        if not modules:
            return None, "module_missing"
        architectures = [r for r in modules if r.get("arch") == sm]
        if not architectures:
            return None, "architecture_mismatch"
        # Shipped rows carry the source digest only (the text is in the recipe
        # catalog); a row that does carry its text must match it too.
        sources = [r for r in architectures if r.get("source_sha256") == source_hash(source)
                   and r.get("source", source) == source]
        if not sources:
            return None, "source_hash_mismatch"
        portable = portable_options(options)
        compatible_options = [row for row in sources if row.get("options") == portable]
        if not compatible_options:
            return None, "options_mismatch"
        for row in compatible_options:
            if row.get("compile_request") == metadata:
                from woof.kernel_dependencies import (
                    forced_include, header_dependencies, include_tree_dependencies)
                try:
                    if "include_trees" in row:
                        # The retained-image cache already uses this full-tree
                        # binding for CuPy/CCCL's conditional includes. Port it
                        # to shipped images instead of bypassing their guards.
                        if (forced_include(options)
                                or re.search(r'["<](?:/|\.\./)', source + ' '.join(options))):
                            raise ValueError("header dependency may escape the include roots")
                        with _BRIDGE_LOCK:
                            actual = include_tree_dependencies(options, self._header_snapshots)
                        if actual != row["include_trees"]:
                            return None, "header_tree_mismatch"
                    else:
                        # Legacy rows lack external-header bindings.
                        dependencies = header_dependencies(source, options)
                        if dependencies:
                            return None, "external_header_dependency"
                except (OSError, ValueError) as error:
                    _warn_once(("unbound-header", str(error)), f"prebuilt image dependency is unbound; using ordinary compiler/cache: {error}")
                    return None, "unbound_header_dependency"
                return row, None
        return None, "compile_request_mismatch"

    @staticmethod
    def image(row):
        """The row's verified cubin bytes, or None when they fail their digest."""
        try:
            if "cubin_xz_base64" in row:
                blob = lzma.decompress(base64.b64decode(row["cubin_xz_base64"], validate=True))
            else:
                blob = base64.b64decode(row["cubin_base64"], validate=True)
        except (KeyError, ValueError, lzma.LZMAError):
            return None
        if blob[:4] != b"\x7fELF" or hashlib.sha256(blob).hexdigest() != row.get("cubin_sha256"):
            return None
        return blob

    def load(self, cp, cache_directory, module_key, source, options, sm, *, metadata=None):
        row, reason = self.select(module_key, source, options, sm, metadata=metadata)
        if row is None:
            return None, reason, None
        try:
            if "cubin_xz_base64" in row:
                blob = lzma.decompress(base64.b64decode(row["cubin_xz_base64"], validate=True))
            else:
                blob = base64.b64decode(row["cubin_base64"], validate=True)
            digest = hashlib.sha256(blob).hexdigest()
            if blob[:4] != b"\x7fELF" or digest != row["cubin_sha256"]:
                return None, "cubin_digest_mismatch", None
            root = Path(cache_directory) / "gpuwm-aot"
            root.mkdir(parents=True, exist_ok=True)
            path = root / (digest + ".cubin")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                fd, temporary = tempfile.mkstemp(prefix=".cubin-", dir=root)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(blob)
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            with _BRIDGE_LOCK:
                module = cp.RawModule(path=str(path))
                # CuPy memoizes path loads. Typedef name expressions can share
                # one image while carrying different aliases for its symbols.
                # Retain every compatible alias, including parallel prewarm loads.
                loaded = module.module
                if row.get("name_mapping"):
                    existing = dict(getattr(loaded, "mapping", None) or {})
                    if any(name in existing and existing[name] != symbol
                           for name, symbol in row["name_mapping"].items()):
                        return None, "name_mapping_conflict", None
                    loaded._set_mapping({**existing, **row["name_mapping"]})
            return module, None, row
        except (KeyError, ValueError, OSError, RuntimeError, lzma.LZMAError) as error:
            return None, "cubin_load_failed:" + type(error).__name__, None


def cache_compiler_banner(pack):
    """Reuse measured build metadata only for the same loaded compiler bytes.

    The shared fingerprint function still measures every other field itself.
    Its empty NVRTC program is replaced by the wheel's measured banner only
    while the live library digest matches. A changed library takes the original
    probe. No fingerprint fields are duplicated or inferred here.
    """
    banners = {row["nvrtc_banner"] for row in pack.rows
               if isinstance(row.get("nvrtc_banner"), str) and row["nvrtc_banner"]}
    if len(banners) != 1 or not pack.compiler:
        return
    from woof.certify import compile_platform
    banner = banners.pop()
    version = pack.compiler["nvrtc"]
    if not compile_platform.nvrtc_build(banner).startswith(f"{version[0]}.{version[1]}."):
        return
    if not compile_platform.nvrtc_build_id(banner).startswith("CL-"):
        return
    with _BRIDGE_LOCK:
        original = compile_platform.nvrtc_banner
        cached = getattr(original, "_gpuwm_aot_banners", None)
        if cached is None:
            cached = {}
            def measured_banner():
                library = compile_platform.nvrtc_library()
                if library.get("status") == "resolved":
                    known = cached.get(library["sha256"])
                    if known is not None:
                        return known
                return original()
            measured_banner._gpuwm_aot_banners = cached
            measured_banner.__wrapped__ = original
            compile_platform.nvrtc_banner = measured_banner
        cached[pack.compiler["nvrtc_library_sha256"]] = banner


def install(cp, pack=None):
    """Use shipped entries for CuPy-generated and engine RawModule requests.

    The common pack stays the format authority. This bridge handles the same
    source-bearing arguments the unchanged prewarm recorder already writes.
    It never changes CUDA source, launch operands or the step loop.
    """
    with _BRIDGE_LOCK:
        try:
            selected = pack if pack is not None else Pack()
            _bridge(cp, selected)
            _raw_bridge(cp, selected)
        finally:
            _install_recorder()


def _cache_directory(compiler):
    from woof.nvrtc_ptx_cache import _cache_directory as resolve
    return resolve(compiler)


def _bridge(cp, pack):
    selected = pack or Pack()
    cache_compiler_banner(selected)
    if not any("compile_request" in row for row in selected.rows):
        return
    from cupy.cuda import compiler
    previous = compiler._compile_with_cache_cuda
    while previous is not None:
        if getattr(previous, "_gpuwm_aot_bridge", False):
            return
        previous = getattr(previous, "__wrapped__", None)
    original = compiler._compile_with_cache_cuda
    def loading(*args, **kwargs):
        bound = dict(zip(_FIELDS, args))
        bound.update(kwargs)
        source = bound.get("source")
        if not isinstance(source, str):
            return original(*args, **kwargs)
        # This bridge sits outside woof's cached-compiler wrapper. Apply
        # the same numerical boundary before selecting a prebuilt image;
        # otherwise fast runs can load a no-contraction recipe that the
        # ordinary runtime would rewrite before compiling or caching.
        options = effective_options(bound.get("options") or ())
        metadata = request_metadata(bound)
        key = "cupy.cache." + source_hash(source)
        arch = bound.get("arch") or cp.cuda.Device().compute_capability
        sm = str(arch) if str(arch).startswith("sm_") else "sm_" + str(arch)
        started = time.perf_counter()
        module = None
        reason = "unsupported_compile_route"
        disabled = os.environ.get("WOOF_AOT", "").lower() in ("off", "0", "false", "no")
        if disabled:
            reason = "aot_disabled"
        elif bound.get("log_stream") is not None:
            reason = "compiler_log_requested"
        elif bound.get("cache_in_memory"):
            reason = "memory_only_cache_requested"
        elif (metadata["backend"] == "nvrtc" and not metadata["jitify"]
                and not metadata["to_ltoir"] and not metadata["enable_cooperative_groups"]):
            module, reason, _ = selected.load(
                cp, _cache_directory(compiler),
                key, source, options, sm, metadata=metadata)
            if reason == "module_missing":
                reason = "source_hash_not_in_pack"
        if module is not None:
            mark("aot", key, time.perf_counter() - started)
            return module.module
        result = original(*args, **kwargs)
        mark("jit", key, time.perf_counter() - started, reason=reason)
        return result
    loading._gpuwm_aot_bridge = True
    loading.__wrapped__ = original
    compiler._compile_with_cache_cuda = loading
    original_direct = compiler.compile_using_nvrtc
    def direct(*args, **kwargs):
        bound = dict(zip(_DIRECT_FIELDS, args))
        bound.update(kwargs)
        source = bound.get("source")
        if (not isinstance(source, str) or bound.get("name_expressions")
                or bound.get("log_stream") is not None or bound.get("cache_in_memory")
                or bound.get("jitify") or bound.get("method") is not None
                or os.environ.get("WOOF_AOT", "").lower() in ("off", "0", "false", "no")):
            return original_direct(*args, **kwargs)
        key = "nvrtc.direct." + source_hash(source)
        arch = bound.get("arch") or compiler._get_arch()
        sm = str(arch) if str(arch).startswith("sm_") else "sm_" + str(arch)
        started = time.perf_counter()
        row, reason = selected.select(key, source, effective_options(bound.get("options") or ()), sm,
                                      metadata=direct_metadata(bound.get("filename")))
        blob = None if row is None else selected.image(row)
        if blob is not None:
            mark("aot", key, time.perf_counter() - started)
            return blob, row.get("name_mapping")
        if row is not None:
            reason = "cubin_digest_mismatch"
        elif reason == "module_missing":
            reason = "source_hash_not_in_pack"
        result = original_direct(*args, **kwargs)
        mark("jit", key, time.perf_counter() - started, reason=reason)
        return result
    direct._gpuwm_aot_bridge = True
    direct.__wrapped__ = original_direct
    compiler.compile_using_nvrtc = direct


def _raw_bridge(cp, pack):
    """Serve final requests too: elementwise/reduction modules can bypass both outer routes."""
    from cupy.cuda import compiler
    original = compiler._compile_using_nvrtc_no_warning
    previous = original
    while previous is not None:
        if getattr(previous, "_gpuwm_aot_raw_bridge", False):
            return
        previous = getattr(previous, "__wrapped__", None)
    selected = pack or Pack()
    if not any(row.get("compiler_route") == RAW_ROUTE for row in selected.rows):
        return

    def raw(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
            log_stream=None, cache_in_memory=False, jitify=None, method=None):
        options = effective_options(options)
        def call():
            return original(source, options, arch, filename, name_expressions,
                            log_stream, cache_in_memory, jitify, method)
        if (not isinstance(source, str) or log_stream is not None or cache_in_memory
                or jitify or method is not None
                or os.environ.get("WOOF_AOT", "").lower() in ("off", "0", "false", "no")):
            return call()
        target = str(arch or compiler._get_arch()).removeprefix("sm_")
        # A cubin must not substitute for a PTX request, even on a supported SM.
        if compiler._get_arch_for_options_for_nvrtc(target)[1] != "cubin":
            return call()
        key = "nvrtc.raw." + source_hash(source)
        metadata = raw_metadata(dict(filename=filename, name_expressions=name_expressions,
                                     cache_in_memory=cache_in_memory, method=method))
        started = time.perf_counter()
        row, reason = selected.select(key, source, options, "sm_" + target, metadata=metadata)
        blob = None if row is None else selected.image(row)
        if blob is not None:
            mark("aot", key, time.perf_counter() - started)
            mapping = row.get("name_mapping")
            return blob, (dict(mapping) if isinstance(mapping, dict) else mapping)
        result = call()
        mark("jit", key, time.perf_counter() - started,
             reason="cubin_digest_mismatch" if row is not None else reason)
        return result
    raw._gpuwm_aot_raw_bridge = True
    raw.__wrapped__ = original
    compiler._compile_using_nvrtc_no_warning = raw


_RECORDED: dict = {}


def record_raw(bound):
    """Record a final parent request even when its process future serves it."""
    if (not os.environ.get("WOOF_AOT_RECORD") or not isinstance(bound.get("source"), str)
            or bound.get("jitify") or bound.get("log_stream") is not None
            or bound.get("cache_in_memory") or bound.get("method") is not None):
        return
    entry = {"source": bound["source"],
             "options": list(effective_options(bound.get("options") or ())),
             **raw_metadata(bound)}
    _RECORDED[hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()] = entry


def _install_recorder():
    """``WOOF_AOT_RECORD=<file>``: write every CUDA compile request this run makes.

    ``tools/build_kernel_aot.py`` builds this catalog into a family pack.
    It wraps outside the pack bridge, so served and compiled
    requests are both recorded; it never changes what is compiled or loaded.
    """
    target = os.environ.get("WOOF_AOT_RECORD")
    if not target:
        return
    # Parallel profile/ensemble workers must not replace each other's catalogs.
    # A literal {pid} in the requested filename selects one shard per process.
    target = target.replace("{pid}", str(os.getpid()))
    from cupy.cuda import compiler
    if getattr(compiler._compile_with_cache_cuda, "_gpuwm_aot_recorder", False):
        return
    cached, direct = compiler._compile_with_cache_cuda, compiler.compile_using_nvrtc
    raw = compiler._compile_using_nvrtc_no_warning

    def keep(entry):
        _RECORDED[hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()] = entry

    def recording_cached(*args, **kwargs):
        bound = dict(zip(_FIELDS, args))
        bound.update(kwargs)
        if isinstance(bound.get("source"), str):
            keep({"source": bound["source"], "options": list(effective_options(bound.get("options") or ())),
                  **request_metadata(bound)})
        return cached(*args, **kwargs)

    def recording_direct(*args, **kwargs):
        bound = dict(zip(_DIRECT_FIELDS, args))
        bound.update(kwargs)
        if isinstance(bound.get("source"), str) and not bound.get("name_expressions"):
            keep({"source": bound["source"], "options": list(effective_options(bound.get("options") or ())),
                  **direct_metadata(bound.get("filename"))})
        return direct(*args, **kwargs)

    def recording_raw(*args, **kwargs):
        bound = dict(zip(_DIRECT_FIELDS, args))
        bound.update(kwargs)
        record_raw(bound)
        return raw(*args, **kwargs)

    for wrapper, original in ((recording_cached, cached), (recording_direct, direct),
                              (recording_raw, raw)):
        wrapper._gpuwm_aot_recorder = True
        wrapper.__wrapped__ = original
    compiler._compile_with_cache_cuda = recording_cached
    compiler.compile_using_nvrtc = recording_direct
    compiler._compile_using_nvrtc_no_warning = recording_raw

    def save():
        from woof.kernel_pack import write
        try:
            identity = compiler_identity()
        except Exception:
            identity = {}
        write(target, identity, list(_RECORDED.values()))
    import atexit
    atexit.register(save)


_EVENTS = []


def mark(source, module_key, seconds, *, reason=None):
    # Startup only: compilation/cache loading never occurs on a warm step.
    with _BRIDGE_LOCK:
        _EVENTS.append({"source": source, "module_key": module_key,
                        "seconds": seconds, "reason": reason})


def receipt():
    with _BRIDGE_LOCK:
        return [dict(row) for row in _EVENTS]
