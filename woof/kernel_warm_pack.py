"""Exact-request cubins retained automatically by the model's compile plan.

The index belongs to one compiler/SM tuple. Cubins are
immutable and content addressed. Only actual successful compiler results enter
it; a missing or damaged image takes the ordinary parallel compiler path.
"""
from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import warnings

from woof.compiler_metadata import _atomic, _digest


@contextmanager
def _index_lock(path):
    """Exclusive process lock on ``path``: flock on POSIX, msvcrt on Windows.

    The index merge imported fcntl unconditionally, so on Windows every save
    failed with "No module named 'fcntl'" and no warm image was ever indexed
    (tests/test_kernel_warm_pack.py: six failures on a Windows host).
    """
    with Path(path).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


SCHEMA = "gpuwm.warm-images/v2"


def cache_root():
    """Persistent image store, independent of a forecast's plan or output."""
    return Path(os.environ.get("WOOF_KERNEL_CACHE") or
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "woof" / "kernels").expanduser()


def program_name(source, filename):
    # CuPy's SHA1 filename includes absolute -I paths. Normalize the actual
    # compiler input, not just its key: NVRTC embeds basenames in symbols.
    if re.fullmatch(r"[0-9a-f]{40}\.(?:cubin|ltoir)\.cu", str(filename)):
        return "gpuwm-" + hashlib.sha256(source.encode()).hexdigest() + ".cu"
    return str(filename)


def _portable_options(options):
    result = []
    roots = 0
    i = 0
    while i < len(options):
        option = str(options[i])
        if option in ("-I", "--include-path") and i + 1 < len(options):
            i += 1
            option = f"-I@{roots}"
            roots += 1
        elif option.startswith("-I") or option.startswith("--include-path="):
            option = f"-I@{roots}"
            roots += 1
        result.append(option)
        i += 1
    return result


class WarmImages:
    def __init__(self, root, identity, tag, arch):
        self.binding = dict(schema=SCHEMA, compiler=identity, arch=str(arch).removeprefix("sm_"))
        self.root = Path(root) / "images" / _digest(self.binding)
        self.index = self.root / "index.json"
        if not self.index.exists() and any(self.root.parent.glob("*/index.json")):
            warnings.warn("kernel cache has no matching compiler/SM binding; missing requests will recompile",
                          RuntimeWarning)
        self.rows = self._read()
        self.memory = {}
        self.pending = {}
        self.lock = threading.RLock()
        self.writer = None
        self.unbound = set()
        self.jobs = []
        self.header_snapshots = {}
        self.request_keys = {}
        self.counts = dict(index_entries=len(self.rows), hits=0, misses=0, stored=0, invalid=0, errors=0, unbound=0)

    def _read(self):
        try:
            document = json.loads(self.index.read_text())
            if document["binding"] == self.binding and isinstance(document["entries"], dict):
                return document["entries"]
            raise ValueError("index binding mismatch")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError):
            warnings.warn("warm kernel index is damaged or incompatible; recompiling requests", RuntimeWarning)
        return {}

    @staticmethod
    def key(arguments, snapshots=None):
        # Source/defines retain every code-dependent extent. Run tags do not
        # participate; include roots are represented by their actual contents.
        from woof.kernel_dependencies import header_dependencies, include_tree_dependencies, include_roots
        source = arguments.get("source", "")
        options = list(arguments.get("options", ()))
        from woof.hip_backend import active, header_identity, compiler_dependencies, toolchain_identity
        if include_roots(options):
            from woof.kernel_dependencies import forced_include
            if forced_include(options) or re.search(r'["<](?:/|\.\./)', source + ' '.join(options)):
                raise ValueError("header dependency may escape the include roots")
            if active():
                # The HIP compiler binding already hashes its entire trusted
                # ROCm tree. Reuse that boundary rather than repeatedly
                # treating compiler config path macros as escaping includes.
                dependencies = [compiler_dependencies(toolchain_identity()[0]),
                                *header_identity(options)]
            else:
                dependencies = include_tree_dependencies(options, snapshots if snapshots is not None else {})
        else:
            dependencies = header_dependencies(source, options, arguments.get("filename", "kern.cu"), portable=True)
        request = dict(arguments, options=_portable_options(options), source=hashlib.sha256(source.encode()).hexdigest())
        return _digest(dict(request=request, headers=dependencies))

    def _request_key(self, arguments):
        try:
            if arguments.get("arch") == "device":
                arguments = dict(arguments, arch=self.binding["arch"])
            request = _digest(arguments)
            with self.lock:
                if request not in self.request_keys:
                    try:
                        self.request_keys[request] = self.key(arguments, self.header_snapshots)
                    except ValueError:
                        # A refusal is immutable for this compiler session,
                        # just like a successful include-tree snapshot. Never
                        # serve/store an unbound image, but do not rescan its
                        # headers on every get/put or wait behind that scan.
                        self.request_keys[request] = None
                        raise
                return self.request_keys[request]
        except (OSError, ValueError) as error:
            with self.lock:
                self.counts["unbound"] += 1
                detail = str(error)
                if detail not in self.unbound:
                    self.unbound.add(detail)
                    warnings.warn(f"warm image dependency is unbound; using ordinary compiler/cache: {detail}", RuntimeWarning)
            return None

    def get(self, arguments):
        key = self._request_key(arguments)
        if key is None:
            return None
        with self.lock:
            if key in self.memory:
                image, mapping = self.memory[key]
                self.counts["hits"] += 1
                return image, dict(mapping) if isinstance(mapping, dict) else mapping
            row = self.rows.get(key)
            if row is None:
                self.counts["misses"] += 1
                return None
            try:
                if row["sha256"] != _digest(dict(key=key, image=row["image"], mapping=row["mapping"])):
                    raise ValueError("index entry digest mismatch")
                digest = row["image"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid image digest")
                image = (self.root / (digest + ".cubin")).read_bytes()
                if not image.startswith(b"\x7fELF") or hashlib.sha256(image).hexdigest() != digest:
                    raise ValueError("image digest mismatch")
                mapping = row["mapping"]
                if mapping is not None and (not isinstance(mapping, dict) or
                        any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())):
                    raise ValueError("invalid symbol mapping")
            except (OSError, ValueError, KeyError, TypeError):
                self.counts["invalid"] += 1
                self.rows.pop(key, None)
                warnings.warn("warm kernel image is missing or damaged; recompiling the exact request", RuntimeWarning)
                return None
            self.memory[key] = image, mapping
            self.counts["hits"] += 1
            return image, dict(mapping) if isinstance(mapping, dict) else mapping

    def put(self, arguments, result):
        image, mapping = result
        if not isinstance(image, bytes) or not image.startswith(b"\x7fELF"):
            return  # PTX/linking routes keep their existing cache and compiler.
        key = self._request_key(arguments)
        if key is None:
            return None
        with self.lock:
            if key in self.memory:
                return
            mapping = dict(mapping) if isinstance(mapping, dict) else mapping
            self.memory[key] = image, mapping
            if self.writer is None:
                self.writer = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="warm-image-write")
            self.jobs.append(self.writer.submit(self._write_image, key, image, mapping))

    def _write_image(self, key, image, mapping):
        try:
            digest = hashlib.sha256(image).hexdigest()
            _atomic(self.root / (digest + ".cubin"), image)
            with self.lock:
                row = dict(image=digest, mapping=mapping,
                           sha256=_digest(dict(key=key, image=digest, mapping=mapping)))
                self.rows[key] = row
                self.pending[key] = row
                self.counts["stored"] += 1
        except Exception as error:
            with self.lock:
                self.counts["errors"] += 1
            warnings.warn(f"warm image write failed: {error}", RuntimeWarning)

    def save(self):
        with self.lock:
            jobs = list(self.jobs)
        for job in jobs:
            job.result()
        with self.lock:
            fresh = dict(self.pending)
        if not fresh:
            return
        try:
            # Other members of an ensemble may publish this same shape. Merge
            # under a process lock; never discard another member's new requests.
            self.root.mkdir(parents=True, exist_ok=True)
            with _index_lock(self.root / "index.lock"):
                rows = self._read()
                rows.update(fresh)
                _atomic(self.index, (json.dumps(dict(binding=self.binding, entries=rows), sort_keys=True) + "\n").encode())
            with self.lock:
                for key, row in fresh.items():
                    if self.pending.get(key) == row:
                        self.pending.pop(key, None)
        except Exception as error:
            with self.lock:
                self.counts["errors"] += 1
            warnings.warn(f"warm image index write failed: {error}", RuntimeWarning)

    def close(self):
        self.save()
        if self.writer is not None:
            self.writer.shutdown(wait=True)

    def receipt(self):
        with self.lock:
            return dict(path=str(self.index), unbound_reasons=sorted(self.unbound), **self.counts)



def install_metadata(compiler, root, identity):
    """Intercept only compiler metadata probes, keeping real compilation intact."""
    original = compiler._preprocess
    existing = getattr(original, "_gpuwm_metadata", None)
    if existing is not None:
        return existing
    from woof.compiler_metadata import CompilerMetadata
    cache = CompilerMetadata(root, identity)

    def preprocess(source, options, arch, backend):
        if source != "" or backend != "nvrtc":
            return original(source, options, arch, backend)
        from woof.kernel_dependencies import forced_include
        if forced_include(options):
            warnings.warn("forced-include metadata requires ordinary NVRTC preprocessing", RuntimeWarning)
            return original(source, options, arch, backend)
        arguments = dict(kind="empty-preprocess", options=_portable_options(list(options)), arch=str(arch), backend=backend)
        return cache.resolve(arguments, lambda: original(source, options, arch, backend))

    preprocess.__wrapped__ = original
    preprocess._gpuwm_metadata = cache
    compiler._preprocess = preprocess
    return cache


PACK_SCHEMA = "gpuwm.warm-pack/v1"


def _checked_files(root, architectures=()):
    """Read only complete, verified published image sets; never export locks."""
    files = {}
    wanted = {str(arch).removeprefix("sm_") for arch in architectures}
    for index in sorted((root / "images").glob("*/index.json")):
        index_bytes = index.read_bytes()
        document = json.loads(index_bytes)
        binding = document["binding"]
        if binding.get("schema") != SCHEMA or (wanted and binding["arch"] not in wanted):
            continue
        if index.parent.name != _digest(binding):
            raise ValueError(f"kernel binding digest mismatch: {index}")
        # Validate rows through the same contract as get(), without needing
        # source requests or importing CuPy on a provisioning host.
        for key, row in document["entries"].items():
            _validate_row(key, row)
            blob = index.parent / (row["image"] + ".cubin")
            data = blob.read_bytes()
            if not data.startswith(b"\x7fELF") or hashlib.sha256(data).hexdigest() != row["image"]:
                raise ValueError(f"kernel image digest mismatch: {blob}")
            files[blob.relative_to(root).as_posix()] = data
        files[index.relative_to(root).as_posix()] = index_bytes
    for path in sorted((root / "compiler-metadata").glob("*/*.json")):
        data = path.read_bytes()
        row = json.loads(data)
        material = {key: row[key] for key in ("binding", "arguments", "value")}
        if (row["sha256"] != _digest(material) or path.stem != _digest(row["arguments"])
                or path.parent.name != _digest(row["binding"])):
            raise ValueError(f"compiler metadata digest mismatch: {path}")
        files[path.relative_to(root).as_posix()] = data
    return files


def _validate_row(key, row):
    if not re.fullmatch(r"[0-9a-f]{64}", key) or not re.fullmatch(r"[0-9a-f]{64}", row["image"]):
        raise ValueError("invalid kernel image key")
    if row["sha256"] != _digest(dict(key=key, image=row["image"], mapping=row["mapping"])):
        raise ValueError("kernel index entry digest mismatch")
    mapping = row["mapping"]
    if mapping is not None and (not isinstance(mapping, dict) or
            any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())):
        raise ValueError("invalid kernel symbol mapping")


def pack(output, *, root=None, architectures=()):
    import io
    import tarfile
    import tempfile
    root = Path(root) if root is not None else cache_root()
    files = _checked_files(root, architectures)
    if not any(name.endswith("index.json") for name in files):
        raise ValueError("no verified compiled images for the requested architectures")
    manifest = dict(schema=PACK_SCHEMA, files={name: hashlib.sha256(data).hexdigest()
                                             for name, data in files.items()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".kernel-pack-", dir=output.parent)
    os.close(fd)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name, data in [("manifest.json", (json.dumps(manifest, sort_keys=True) + "\n").encode()), *files.items()]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    _atomic(Path(str(output) + ".sha256"), (digest + "\n").encode())
    return dict(schema=PACK_SCHEMA, path=str(output), sha256=digest, files=len(files))


def seed(pack_path, *, root=None, sha256=None):
    import tarfile
    root = Path(root) if root is not None else cache_root()
    pack_path = Path(pack_path)
    digest = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    if sha256 is None:
        sha256 = Path(str(pack_path) + ".sha256").read_text().strip()
    if digest != sha256:
        raise ValueError("kernel pack SHA256 mismatch; seed refused, ordinary compilation remains available")
    with tarfile.open(pack_path, "r:gz") as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
            raise ValueError("kernel pack contains duplicate or non-file members")
        files = {m.name: archive.extractfile(m).read() for m in members}
    manifest = json.loads(files.pop("manifest.json"))
    if manifest["schema"] != PACK_SCHEMA or manifest["files"] != {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()}:
        raise ValueError("kernel pack manifest mismatch")
    pattern = r"(?:images/[0-9a-f]{64}/(?:index\.json|[0-9a-f]{64}\.cubin)|compiler-metadata/[0-9a-f]{64}/[0-9a-f]{64}\.json)"
    if any(not re.fullmatch(pattern, name) for name in files):
        raise ValueError("invalid kernel pack member path")
    # Stage and verify the whole pack before touching the destination. No
    # archive path is extracted, and a failed pack cannot partially seed it.
    import tempfile
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seed-", dir=root) as temporary:
        stage = Path(temporary)
        for name, data in files.items():
            _atomic(stage / name, data)
        if _checked_files(stage) != files:
            raise ValueError("kernel pack has unreferenced or incompatible images")
        for name, data in files.items():
            if name.endswith("index.json"):
                continue
            _atomic(root / name, data)
        # Merge under the publisher's lock, preserving kernels warmed locally.
        for name, data in files.items():
            if not name.endswith("index.json"):
                continue
            document = json.loads(data)
            binding = document["binding"]
            images = WarmImages(root, binding["compiler"], None, binding["arch"])
            images.pending.update(document["entries"])
            images.save()
            if images.counts["errors"]:
                raise OSError("kernel pack index publication failed")
    return dict(schema=PACK_SCHEMA, path=str(pack_path), cache=str(root), sha256=digest, files=len(files))


def register_cli(sub):
    parser = sub.add_parser("kernels", help="export or seed verified persistent CUDA images")
    commands = parser.add_subparsers(dest="kernel_action", required=True)
    export = commands.add_parser("pack", help="export images already compiled on this box")
    export.add_argument("output")
    export.add_argument("--arch", action="append", default=[], help="SM filter, e.g. sm_120 or sm_89; repeatable")
    restore = commands.add_parser("seed", help="verify and import a local image pack")
    restore.add_argument("pack")
    restore.add_argument("--sha256", help="expected SHA256; defaults to PACK.sha256")
    for command in (export, restore):
        command.add_argument("--cache", help="override WOOF_KERNEL_CACHE")
        command.set_defaults(func=_cli)


def _cli(args):
    import tarfile
    try:
        result = (pack(args.output, root=args.cache, architectures=args.arch) if args.kernel_action == "pack"
                  else seed(args.pack, root=args.cache, sha256=args.sha256))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        raise SystemExit(f"kernel cache: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0
