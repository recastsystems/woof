"""Persist exact compiler results, including CuPy's empty preprocessing probes.

The ordinary CuPy disk cache does not cache its NVRTC preprocessing probe, and
in-memory physics compilers do not necessarily write CuPy's cache at all. This
layer serves those exact requests from binary artifacts. It changes no source,
options, launch operands or numerical defaults. The compiler family/build, run
configuration, architecture, source, headers and final options bind each entry.
Cold writes run beside compilation on an IO pool; warm reads address one entry,
never scan other plans. Unreadable entries fall back to the original compiler.
"""
from __future__ import annotations

import atexit
import concurrent.futures
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import warnings

SCHEMA = "gpuwm-hex.kernel-warm/v2"
DIRECTORY_ENVIRONMENT = "WOOF_HEX_WARM_PACK_DIR"
#: Read-only stores a run also reads after its own (os.pathsep list); a box image bakes its warm store there.
SYSTEM_ENVIRONMENT = "WOOF_HEX_SYSTEM_KERNEL_CACHE"
SYSTEM_DEFAULT = "/opt/gpuwm-hex/kernel-packs"
_MAGIC = b"HEXWARM2"
_MAX_HEADER = 1024 * 1024
_STATE = None
_LOCK = threading.RLock()


#: CuPy's cached route names its NVRTC program after its own cache entry, a SHA-1 of the
#: request including the install's absolute include paths (woof.hex.kernel_plan._CUPY_PROGRAM).
_CUPY_PROGRAM = re.compile(r"^[0-9a-f]{40}\.(cubin|ltoir)\.cu$")
#: The install root (site-packages) in a portable key, as in the prebuilt pack's recipes.
_SITE_TOKEN = "@SITE@"


def _install_roots():
    """The site-packages roots of this install (CuPy's, and its resolved form)."""
    try:
        from .cuda_backend.aot import _site_roots
    except ImportError:
        return ()
    return tuple(_site_roots())


def _portable_path(path, roots):
    text = str(path)
    for root in roots:
        for separator in ("/", os.sep):
            if text == root or text.startswith(root + separator):
                return _SITE_TOKEN + text[len(root):].replace(os.sep, "/")
    return text


def _portable_options(options, roots):
    """``-I<root>/...`` (and ``--include-path``) with the install root as one token."""
    result = []
    iterator = iter(options)
    for option in iterator:
        option = str(option)
        if option in ("-I", "--include-path"):
            result.append(option)
            following = next(iterator, None)
            if following is not None:
                result.append(_portable_path(following, roots))
        elif option.startswith("--include-path="):
            result.append("--include-path=" + _portable_path(option.split("=", 1)[1], roots))
        elif option.startswith("-I"):
            result.append("-I" + _portable_path(option[2:], roots))
        else:
            result.append(option)
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _copy(value):
    if isinstance(value, tuple):
        return value[0], None if value[1] is None else dict(value[1])
    return value


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".warm-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _include_metadata(source):
    """Read object-like definitions without evaluating conditional branches.

    Every definition is retained, so conditional alternatives over-bind rather
    than omit a possible file input. Function-like computed include names keep
    ordinary compilation until their dependency closure can be bound.
    """
    source = re.sub(r"\\\r?\n", "", source)
    tokens = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*'
    source = re.sub(tokens, lambda m: m[0] if m[0][0] in "\"'" else
                    " " + "\n" * m[0].count("\n"), source, flags=re.S)
    macros = {}
    for name, rest in re.findall(r'^[^\S\n]*#[^\S\n]*define[^\S\n]+([A-Za-z_]\w*)([^\n]*)', source, re.M):
        macros.setdefault(name, set()).add(None if rest.startswith("(") else rest.strip())
    includes = re.findall(r'^[^\S\n]*#[^\S\n]*include[^\S\n]+([^\n]+)', source, re.M)
    return macros, [value.strip() for value in includes]


def _tree_listing(root, files):
    """Each file of a bound header tree by name, size, inode, mtime and ctime (no reads)."""
    rows = []
    for file in files:
        info = file.stat()
        rows.append([str(file.relative_to(root)), info.st_size, info.st_dev, info.st_ino,
                     info.st_mtime_ns, info.st_ctime_ns])
    return rows


def _expanded_headers(operand, macros):
    """Conservatively expand finite object-like header alternatives."""
    if operand.startswith(('"', '<')):
        values = {operand}
    else:
        values = {operand}
        token_pattern = r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|.'
        for _ in range(32):
            expanded = set()
            for value in values:
                pieces = {""}
                for token in re.findall(token_pattern, value):
                    replacements = macros.get(token, {token})
                    if None in replacements:
                        raise ValueError("function-like computed include cannot be bound")
                    pieces = {left + right for left in pieces for right in replacements}
                    if len(pieces) > 256:
                        raise ValueError("computed include has too many alternatives")
                expanded.update(pieces)
            if expanded == values:
                break
            values = expanded
            if len(values) > 256:
                raise ValueError("computed include has too many alternatives")
        else:
            raise ValueError("recursive computed include cannot be bound")
    names = []
    for value in values:
        match = re.fullmatch(r'"([^"\\\n]+)"|<([^>\n]+)>', value.strip())
        if match is None:
            raise ValueError("unresolved computed include cannot be bound")
        names.append(match[1] or match[2])
    return names


def box_roots(primary):
    """Read-only stores a run reads after its own: the box's default store and an image-baked one.

    THE BREAKAGE THIS ENDS (V8X6000 and an RTX 5090 host, 2026-10-29): a run that isolates its cache root
    (WOOF_HEX_CACHE_ROOT or WOOF_HEX_KERNEL_CACHE_ROOT, as lanes and members do) found none of the
    box's compiled images and compiled every module again.  Entries are content-addressed and every read
    checks the entry's own digest and request key, so another store can only serve the identical request;
    this run still writes only to its own store.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    roots = [(Path(base).expanduser() if base else Path.home() / ".cache") / "woof hex" / "prep" / "kernel-packs"]
    declared = os.environ.get(SYSTEM_ENVIRONMENT)
    roots += [Path(part).expanduser() for part in (declared.split(os.pathsep) if declared is not None
                                                   else [SYSTEM_DEFAULT]) if part]
    result = []
    for root in roots:
        try:
            if root.resolve() != Path(primary).resolve() and root not in result:
                result.append(root)
        except (OSError, RuntimeError):
            continue
    return result


class Store:
    def __init__(self, root, identity, tag, workers=None, fallbacks=()):
        self.identity = identity
        self.tag = str(tag)
        self.root = Path(root)
        # Content-addressed: an artifact's key digests its whole request (source, options, arch, filename and
        # the bound header bytes) and get() refuses any file whose header does not name that key, so one
        # directory per compiler identity serves every run.  THE BREAKAGE THIS PREVENTS: the directory was also
        # keyed by the run's argv (the plan tag), so every new storm-following member (a new --mesh/--grid)
        # missed every artifact and recompiled the MPAS suite's fast-mode kernels inline, 15 s at the start of
        # each moving segment on an RTX 4090 host (MPAS-ROUTE-SPEED-2026-10-24).
        self.directory = (self.root / ("cuda" + str(identity["nvrtc"][0])) /
                          _digest(identity) / "content")
        self.fallbacks = [Path(fallback) / ("cuda" + str(identity["nvrtc"][0])) / _digest(identity) / "content"
                          for fallback in fallbacks]
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        self.workers = workers or cores
        self.pool = concurrent.futures.ThreadPoolExecutor(self.workers, thread_name_prefix="hex-warm-io")
        self.lock = threading.RLock()
        self.pending = {}
        self.headers = {}
        self.bindings = {}
        self.warned_bypasses = set()
        self.counts = dict(hits=0, box_hits=0, misses=0, invalid=0, writes=0, write_failures=0,
                           bypasses=0, lookup_seconds=0.0, compiles=0,
                           shared_hits=0, compile_lock_seconds=0.0, lock_failures=0)

    def _includes(self, options, source):
        """Bind explicit include trees once per process, with parallel byte hashing.

        Existing NVRTC caches assume installed compiler/header inputs stay fixed
        while a process runs. We make the same assumption. A fresh process reuses
        a tree's stored binding only when every file has the same size, inode,
        mtime and ctime (``_stored_tree``); an edited file always has a new ctime,
        including one whose mtime was reset, so it is reparsed and rehashed.
        """
        roots = []
        source_macros, source_includes = _include_metadata(source)
        iterator = iter(options)
        for option in iterator:
            option = str(option)
            if option in ("-D", "--define-macro"):
                definition = str(next(iterator))
                name, _, value = definition.partition("=")
                source_macros.setdefault(name, set()).add(value or "1")
            elif option.startswith(("-D", "--define-macro=")):
                definition = option[2:] if option.startswith("-D") else option.split("=", 1)[1]
                name, _, value = definition.partition("=")
                source_macros.setdefault(name, set()).add(value or "1")
            elif option in ("-I", "--include-path"):
                roots.append(str(next(iterator)))
            elif option.startswith("-I"):
                roots.append(option[2:])
            elif option.startswith("--include-path="):
                roots.append(option.split("=", 1)[1])
            elif option.startswith(("--pre-include", "-include", "-pch", "--pch")):
                # These additional file inputs must keep the compiler path until
                # their dependency closure can be bound explicitly.
                raise ValueError("external preinclude/PCH input")
        # Bind source-local literal or object-macro headers before any artifact
        # lookup. The explicit trees supply definitions used by nested headers.
        for operand in source_includes:
            try:
                names = _expanded_headers(operand, source_macros)
            except ValueError:
                continue  # Retry once explicit header definitions are known.
            for name in names:
                path = Path(name)
                if path.is_file():
                    roots.append(str(path.resolve().parent))
        # Different generated kernels commonly have the same include directives
        # and definitions. Installed headers are already fixed for this Store;
        # reuse their validated closure, still binding every request's roots.
        # The full source remains in request(), so executable keys do not change.
        binding = _digest({"roots": roots, "cwd": str(Path.cwd()),
                           "macros": {name: sorted(values, key=str)
                                      for name, values in source_macros.items()},
                           "includes": source_includes})
        with self.lock:
            future = self.bindings.get(binding)
            owner = future is None
            if owner:
                if len(self.bindings) >= 1024:
                    self.bindings.clear()
                future = concurrent.futures.Future()
                self.bindings[binding] = future
        if owner:
            try:
                future.set_result(self._bind_includes(roots, source_macros, source_includes))
            except BaseException as error:
                future.set_exception(error)
                with self.lock:
                    if self.bindings.get(binding) is future:
                        del self.bindings[binding]
                raise
        return list(future.result())

    def _bind_includes(self, roots, source_macros, source_includes):
        macros = {name: set(values) for name, values in source_macros.items()}
        indirect = []
        result = []
        bounded_paths = tuple(Path(root).resolve() for root in dict.fromkeys(roots))
        for root in dict.fromkeys(roots):
            path = Path(root).resolve()
            token = str(path)
            with self.lock:
                future = self.headers.get(token)
                owner = future is None
                if owner:
                    future = concurrent.futures.Future()
                    self.headers[token] = future
            if owner:
                try:
                    files = []
                    def failed(error):
                        raise error
                    for base, dirs, names in os.walk(path, onerror=failed):
                        if any((Path(base) / name).is_symlink() for name in dirs):
                            raise ValueError("include tree contains a directory symlink")
                        files.extend(Path(base) / name for name in names
                                     if (Path(base) / name).is_file())
                    files.sort()
                    def hashed(file):
                        content = file.read_bytes()
                        definitions, includes = _include_metadata(content.decode("utf-8", "replace"))
                        computed = []
                        for operand in includes:
                            if not operand.startswith(('"', '<')):
                                computed.append((str(file.parent), operand))
                                continue
                            for name in _expanded_headers(operand, {}):
                                include = Path(name)
                                if include.is_absolute() or ".." in include.parts:
                                    # Root metadata is shared by different option sets.
                                    # Revalidate these operands against each request's
                                    # bound roots, not only the first request's roots.
                                    computed.append((str(file.parent), operand))
                        return ((str(file.relative_to(path)), hashlib.sha256(content).hexdigest()),
                                definitions, computed)
                    began_ns = time.time_ns()
                    listing = _tree_listing(path, files)
                    bound = self._stored_tree(token, listing)
                    if bound is None:
                        rows = list(self.pool.map(hashed, files))
                        bound = (_digest([row[0] for row in rows]),
                                 [row[1] for row in rows],
                                 [entry for row in rows for entry in row[2]])
                        self._store_tree(token, listing, bound, began_ns)
                    future.set_result(bound)
                except BaseException as error:
                    future.set_exception(error)
                    raise
            digest, definitions, computed = future.result()
            result.append((token, digest))
            indirect.extend(computed)
            for definition in definitions:
                for name, values in definition.items():
                    macros.setdefault(name, set()).update(values)
        # Computed header operands can otherwise escape every hashed tree.
        # No compiler option is changed; unsupported cases retain compilation.
        checks = [(str(Path.cwd()), operand) for operand in source_includes]
        checks.extend(indirect)
        for parent, operand in checks:
            for name in _expanded_headers(operand, macros):
                include = Path(name)
                if include.is_absolute() or ".." in include.parts:
                    target = (Path(parent) / include).resolve()
                    if not any(target.is_relative_to(base) for base in bounded_paths):
                        raise ValueError("computed include escapes the bound header trees")
                else:
                    local = Path(parent) / include
                    if local.is_file() and not any(local.resolve().is_relative_to(base)
                                                   for base in bounded_paths):
                        raise ValueError("computed local include is outside bound header trees")
        return result

    def _tree_path(self, token):
        return self.root / "header-trees" / (hashlib.sha256(token.encode()).hexdigest() + ".json")

    def _stored_tree(self, token, listing):
        """A header tree's binding from an earlier process, when every file in it is the same file.

        THE COST THIS REMOVES (V5090 box, 2026-10-27): every forecast process read and regex-parsed
        each file of every bound include tree (the CUDA/CCCL headers: thousands of files) on the
        hex-warm-io pool.  The parse is Python and holds the GIL, so for 4-18 s of each start the
        pool starved the main thread's host preparation (a 3 km region's dycore inputs took 7.7 s
        with 1.3 s of main-thread CPU).  A stored binding is used only when the tree lists the same
        files with the same size, inode, mtime and ctime.  Any write to a file changes its ctime,
        which no edit can preserve, so an edited header is reparsed and rehashed exactly as before.
        File times are coarse (one clock tick), so an edit in the same tick as the file's last
        change could keep every stamp: a binding is stored as usable only when every file's ctime
        was more than a second older than the parse that made it (git's racy-clean rule).  A file
        changed after that parse then has a later ctime, and one changed before it was parsed.
        """
        try:
            stored = json.loads(self._tree_path(token).read_text())
            if (stored.get("schema") != SCHEMA or stored.get("token") != token
                    or stored.get("listing") != listing):
                return None
            definitions = [{name: {value for value in values} for name, values in row.items()}
                           for row in stored["definitions"]]
            computed = [(parent, operand) for parent, operand in stored["computed"]]
            return stored["digest"], definitions, computed
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def _store_tree(self, token, listing, bound, began_ns):
        digest, definitions, computed = bound
        if any(row[5] >= began_ns - 1_000_000_000 for row in listing):
            return  # racily clean: a file changed within a second of the parse; parse again next time
        try:
            _atomic(self._tree_path(token), json.dumps({
                "schema": SCHEMA, "token": token, "listing": listing, "digest": digest,
                "definitions": [{name: sorted(values, key=lambda v: (v is not None, v or ""))
                                 for name, values in row.items()} for row in definitions],
                "computed": [list(entry) for entry in computed]}).encode())
        except OSError:
            pass  # the next process parses the tree again, as before

    def request(self, route, arguments):
        """The artifact key: the request's content, not where this install lives.

        All words remain in order: duplicate compiler flags can affect the
        final value, and names can occur in __FILE__ and diagnostic metadata.

        THE BREAKAGE THIS ENDS (VRFS 8x RTX 5090, 2026-10-28, pc-after-8c): a run
        from a second venv of the same release, on a box whose store already held
        every image, missed all of them and compiled 53 modules inline on one core
        (first step 37.3 s against 4.3 s warm). The key carried where that venv lives
        in three places: CuPy's ``-I<site-packages>/...`` options, the absolute
        paths of the bound header trees, and the program name CuPy derives from
        a digest of those same options (``<sha1>.cubin.cu``). The install root is
        now one token in the options and the tree names (the trees' bytes stay in
        the key, so a different header set still misses), and CuPy's derived name
        is one token unless the source or options can embed the program name
        (``assert``, ``__FILE__``, line or debug info), where it stays bound.
        """
        bound = dict(arguments)
        options = tuple(bound.get("options") or ())
        headers = self._includes(options, bound.get("source", ""))
        roots = _install_roots()
        bound["options"] = _portable_options(options, roots)
        bound["headers_digest"] = [(_portable_path(token, roots), digest) for token, digest in headers]
        filename = bound.get("filename")
        if isinstance(filename, str) and _CUPY_PROGRAM.match(filename):
            from .kernel_plan import _name_sensitive
            if not _name_sensitive(str(bound.get("source", "")), tuple(str(o) for o in options)):
                bound["filename"] = _CUPY_PROGRAM.sub(r"@CUPY_ENTRY@.\1.cu", filename)
        return _digest({"schema": SCHEMA, "route": route, "arguments": bound})

    def path(self, key):
        return self.directory / key[:2] / (key + ".bin")

    def held(self, key):
        """Whether this store or a read-only box store holds an entry under this request key."""
        return any((directory / key[:2] / (key + ".bin")).is_file()
                   for directory in [self.directory, *self.fallbacks])

    def get(self, key):
        value = self._get(key, self.path(key))
        if value is None and self.fallbacks:
            for directory in self.fallbacks:
                path = directory / key[:2] / (key + ".bin")
                if path.is_file():
                    value = self._get(key, path, counted=False)
                    if value is not None:
                        with self.lock:
                            self.counts["box_hits"] += 1
                        break
        return value

    def _get(self, key, path, counted=True):
        started = time.perf_counter()
        with self.lock:
            pending = self.pending.get(key)
        if pending is not None:
            with self.lock:
                self.counts["hits"] += 1
            return _copy(pending[0])
        try:
            data = path.read_bytes()
            if data[:8] != _MAGIC:
                raise ValueError("artifact magic")
            size = struct.unpack("<I", data[8:12])[0]
            if size > _MAX_HEADER or len(data) < 12 + size:
                raise ValueError("artifact header")
            header = json.loads(data[12:12 + size])
            if not isinstance(header, dict):
                raise ValueError("artifact header type")
            payload = data[12 + size:]
            # Bind the mapping and type too, not only executable bytes.
            proof = header.pop("sha256")
            if (header.get("schema") != SCHEMA or header.get("request") != key or
                    hashlib.sha256(_json(header) + payload).hexdigest() != proof):
                raise ValueError("artifact digest")
            if header["kind"] == "text":
                value = payload.decode("utf-8")
            elif header["kind"] == "image":
                mapping = header["mapping"]
                if mapping is not None and (not isinstance(mapping, dict) or
                        not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items())):
                    raise ValueError("artifact mapping")
                value = payload, mapping
            else:
                raise ValueError("artifact kind")
            if counted:
                with self.lock:
                    self.counts["hits"] += 1
            return value
        except FileNotFoundError:
            if counted:
                with self.lock:
                    self.counts["misses"] += 1
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            with self.lock:
                self.counts["invalid"] += 1
        finally:
            with self.lock:
                self.counts["lookup_seconds"] += time.perf_counter() - started
        return None

    def put(self, key, value):
        value = _copy(value)
        if isinstance(value, str):
            payload, kind, mapping = value.encode("utf-8"), "text", None
        elif isinstance(value, tuple) and isinstance(value[0], bytes):
            payload, mapping = value
            kind = "image"
        else:
            return
        header = {"schema": SCHEMA, "request": key, "kind": kind, "mapping": mapping}
        header["sha256"] = hashlib.sha256(_json(header) + payload).hexdigest()
        encoded = _json(header)
        content = _MAGIC + struct.pack("<I", len(encoded)) + encoded + payload
        def write():
            try:
                _atomic(self.path(key), content)
                with self.lock:
                    self.counts["writes"] += 1
            except OSError as error:
                with self.lock:
                    self.counts["write_failures"] += 1
                warnings.warn(f"hex warm artifact write failed at {self.path(key)}: {error}; "
                              "ordinary compilation retained")
        with self.lock:
            if key not in self.pending:
                self.pending[key] = (value, self.pool.submit(write))
            return self.pending[key][1]

    @contextmanager
    def _compile_lock(self, key):
        # Never unlink lock files: a waiter holds the old inode, and a third
        # process could otherwise create a new lock and compile concurrently.
        # A process exiting or crashing releases the OS lock automatically.
        from .file_lock import held
        began = time.perf_counter()
        with ExitStack() as stack:
            try:
                stack.enter_context(held(self.path(key).with_suffix(".lock")))
            except OSError as error:
                with self.lock:
                    self.counts["lock_failures"] += 1
                warnings.warn(f"hex warm artifact lock unavailable: {error}; ordinary compilation retained")
            with self.lock:
                self.counts["compile_lock_seconds"] += time.perf_counter() - began
            yield

    def call(self, route, arguments, function):
        try:
            key = self.request(route, arguments)
        except (OSError, ValueError, TypeError, StopIteration) as error:
            reason = str(error) or type(error).__name__
            with self.lock:
                self.counts["bypasses"] += 1
                warn = reason not in self.warned_bypasses
                self.warned_bypasses.add(reason)
            if warn:
                warnings.warn(f"hex warm artifact bypass: {reason}; ordinary compilation retained")
            return function()
        value = self.get(key)
        if value is not None and not (route == "compiler-banner" and value == ""):
            return value
        # The in-process plan owns thread single-flight; this lock also covers
        # independent forecast members and spawned compiler workers. Recheck
        # after acquiring it, then publish before releasing it. Warm hits above
        # do not take the lock and different kernels still compile in parallel.
        with self._compile_lock(key):
            if self.path(key).is_file():
                value = self.get(key)
                if value is not None and not (route == "compiler-banner" and value == ""):
                    with self.lock:
                        self.counts["shared_hits"] += 1
                    return value
            with self.lock:
                self.counts["compiles"] += 1
            value = function()
            if not (route == "compiler-banner" and value == ""):
                written = self.put(key, value)
                if written is not None:
                    written.result()
            return value

    def flush(self):
        with self.lock:
            futures = [row[1] for row in self.pending.values()]
        for future in futures:
            future.result()

    def receipt(self):
        with self.lock:
            return {"schema": SCHEMA, "directory": str(self.directory),
                    "compiler": self.identity, "configuration": self.tag,
                    "io_workers": self.workers, **self.counts}


def cache_root():
    """A box cache independent of a member's host-preparation and CuPy paths.

    Keep the former default location so existing default caches stay warm.
    A service may set KERNEL_CACHE_ROOT once for every member on its box;
    CACHE_ROOT remains the general application override. Compiler identity
    and exact request bytes, not these directory names, bind every image.
    """
    for name in ("WOOF_HEX_KERNEL_CACHE_ROOT", "WOOF_HEX_CACHE_ROOT"):
        declared = os.environ.get(name)
        if declared:
            return Path(declared).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "woof hex/prep"


def _root():
    if os.environ.get("WOOF_HEX_KERNEL_PREWARM", "").strip().lower() in ("off", "0", "no", "false"):
        return None
    explicit = os.environ.get(DIRECTORY_ENVIRONMENT)
    if explicit:
        return Path(explicit)
    return cache_root() / "kernel-packs"


def install(*, root=None, tag=None, identity=None):
    """Install before the plan funnel, preserving any numerical option wrapper."""
    global _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE
        root = _root() if root is None else root
        if root is None:
            return None
        from cupy.cuda import compiler
        from . import kernel_plan
        from .cuda_backend.aot import compiler_identity, effective_options
        from woof.certify import compile_platform
        identity = compiler_identity() if identity is None else identity
        tag = kernel_plan.default_tag_text() if tag is None else tag
        store = Store(root, identity, tag, workers=kernel_plan._workers(), fallbacks=box_roots(root))
        original = compiler._compile_using_nvrtc_no_warning
        def compile_cached(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
                           log_stream=None, cache_in_memory=False, jitify=None, method=None):
            options = effective_options(options)
            def call():
                return original(source, options, arch, filename, name_expressions,
                                log_stream, cache_in_memory, jitify, method)
            if log_stream is not None or jitify or not isinstance(source, str):
                return call()
            # CuPy embeds a random temporary program path in these images.
            # Retaining a previous path would change observable __FILE__ data.
            if not cache_in_memory and kernel_plan._name_sensitive(source, options):
                with store.lock:
                    store.counts["bypasses"] += 1
                return call()
            selected = str(arch or compiler._get_arch())
            arguments = dict(source=source, options=list(options), arch=selected, filename=filename,
                             name_expressions=name_expressions, cache_in_memory=bool(cache_in_memory),
                             method=method)
            return store.call("nvrtc", arguments, call)
        compile_cached.__wrapped__ = original
        compiler._compile_using_nvrtc_no_warning = compile_cached
        preprocess = compiler._preprocess
        def preprocess_cached(source, options, arch, backend):
            if backend != "nvrtc":
                return preprocess(source, options, arch, backend)
            options = effective_options(options)
            return store.call("preprocess", dict(source=source, options=list(options), arch=str(arch)),
                              lambda: preprocess(source, options, arch, backend))
        preprocess_cached.__wrapped__ = preprocess
        compiler._preprocess = preprocess_cached
        banner = compile_platform.nvrtc_banner
        def banner_cached():
            return store.call("compiler-banner", {}, banner)
        banner_cached.__wrapped__ = banner
        compile_platform.nvrtc_banner = banner_cached
        _STATE = store
        atexit.register(store.flush)
        return store


def worker_config():
    return None if _STATE is None else dict(root=str(_STATE.root), tag=_STATE.tag, identity=_STATE.identity)


def restore(arguments, arch):
    """Load a plan entry directly, before creating any compiler processes."""
    if _STATE is None:
        return None
    from . import kernel_plan
    from .cuda_backend.aot import effective_options
    options = effective_options(arguments["options"])
    if (not arguments["cache_in_memory"] and
            kernel_plan._name_sensitive(arguments["source"], options)):
        return None
    bound = dict(arguments, arch=str(arch), options=list(options))
    try:
        return _STATE.get(_STATE.request("nvrtc", bound))
    except (OSError, ValueError, TypeError, StopIteration):
        return None


def raw_image(source, options, arch, filename, compile_image):
    """Cache raw NVRTC without adding CuPy's implicit compilation options."""
    store = _STATE
    if store is None:
        try:
            store = install()
        except (ImportError, OSError, ValueError) as error:
            warnings.warn(f"hex raw warm pack unavailable: {error}; ordinary compilation retained")
    if store is None:
        return compile_image()
    arguments = dict(source=source, options=list(options), arch=str(arch), filename=filename)
    image, _ = store.call("raw-nvrtc", arguments, lambda: (compile_image(), None))
    return image


def raw_cached(source, options, arch, filename):
    """Whether this process's warm store already holds the raw NVRTC image ``raw_image`` would ask for."""
    store = _STATE
    if store is None:
        return False
    arguments = dict(source=source, options=list(options), arch=str(arch), filename=filename)
    try:
        return store.held(store.request("raw-nvrtc", arguments))
    except (OSError, ValueError, TypeError, StopIteration):
        return False


def receipt():
    return None if _STATE is None else _STATE.receipt()
