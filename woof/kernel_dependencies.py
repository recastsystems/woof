"""Bind literal external includes; refuse dependencies requiring preprocessing.

This performs no compiler or device work. Builtin NVRTC headers are bound by
loaded compiler bytes. All matching external header candidates are hashed,
including nested literal includes, so search-order shadowing changes the key.
Macro includes and forced include options keep the ordinary compiler route.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


def include_roots(options):
    roots = []
    options = list(map(str, options))
    for i, option in enumerate(options):
        if option in ("-I", "--include-path") and i + 1 < len(options):
            roots.append(Path(options[i + 1]))
        elif option.startswith("-I") and len(option) > 2:
            roots.append(Path(option[2:]))
        elif option.startswith("--include-path="):
            roots.append(Path(option.split("=", 1)[1]))
    return roots


#: CUDA host-library wheels that install their headers into the toolkit's shared
#: include folder (CUDA 13: site-packages/nvidia/cu13/include) beside the compiler's
#: own headers. ``pip install cupy-cuda13x[ctk]`` adds the first six; other stacks add
#: more. A device compile opens none of them unless the kernel source or a bound header
#: names one, so :func:`include_tree_dependencies` with ``source=`` leaves out the
#: unnamed ones. Matched with and without a -cu12/-cu13 suffix.
LIBRARY_HEADER_DISTRIBUTIONS = (
    "nvidia-cublas", "nvidia-cufft", "nvidia-curand", "nvidia-cusolver", "nvidia-cusparse",
    "nvidia-nvjitlink", "nvidia-cusparselt", "nvidia-npp", "nvidia-nvjpeg", "nvidia-cufile",
    "nvidia-nccl", "nvidia-cudnn", "nvidia-cutensor", "nvidia-nvshmem", "nvidia-cuda-cupti",
    "nvidia-nvfatbin", "nvidia-nvml-dev")

_LIBRARY_FILES = None


def library_header_files():
    """Resolved paths of every file the installed library wheels above own."""
    global _LIBRARY_FILES
    if _LIBRARY_FILES is None:
        from importlib import metadata
        owned = set()
        for name in LIBRARY_HEADER_DISTRIBUTIONS:
            for variant in (name, name + "-cu12", name + "-cu13"):
                try:
                    distribution = metadata.distribution(variant)
                except metadata.PackageNotFoundError:
                    continue
                for item in distribution.files or ():
                    owned.add(Path(distribution.locate_file(item)).resolve())
        _LIBRARY_FILES = frozenset(owned)
    return _LIBRARY_FILES


def include_tree_dependencies(options, snapshots, *, exclude_python_bytecode=False, source=None):
    """Conservative binding for header macros, including CCCL dispatch.

    NVRTC has no host system-header search. Bind all files in every explicit
    include root, in search order, including extensionless C++ headers. Roots
    are snapshotted once per compiler session, just like its loaded libraries;
    a new process rehashes bytes and detects changed or added headers.

    With ``source`` (the kernel text), bind what the compile can open rather
    than the whole folder: headers that CUDA library wheels drop into the
    shared toolkit include folder (:data:`LIBRARY_HEADER_DISTRIBUTIONS`) are
    left out unless the source or a bound header names them. THE BREAKAGE
    (2.9.12 R11, 2026-11-04): shipped startup images were built where that
    folder held only the runtime and NVRTC headers; ``pip install
    recast-woof[all-cu13]`` (cupy[ctk]) adds cuBLAS/cuFFT/cuRAND/cuSOLVER/cuSPARSE/
    nvJitLink headers there, the whole-folder hash changed, and every image
    missed (header_tree_mismatch: each module compiled on one core at the
    first forecast step). Every other file stays bound by bytes, so a changed,
    added or shadowing compiler header still misses. A header a compile can
    open is spelled in the source or a bound header, literally or in the
    macro that computes it (thrust's system headers), so the name check is a
    substring check over their bytes. In a folder with no library headers the
    result equals the whole-folder binding, so packs built there need no
    rebuild.
    """
    from concurrent.futures import ThreadPoolExecutor
    result = []
    roots = include_roots(options)
    resolved_roots = tuple(root.resolve() for root in roots)
    library = library_header_files() if source is not None else frozenset()
    library_rows = {}
    for root in roots:
        # Match the fold's complete search-root identity and memoized refusals.
        # HIP's bytecode exclusion gets its own cache entry.
        key = (resolved_roots, root.resolve())
        if exclude_python_bytecode:
            key = (key, True)
        previous = snapshots.get(key)
        if isinstance(previous, ValueError):
            raise ValueError(str(previous)) from None
        if key not in snapshots:
            paths = sorted(p for p in root.rglob("*") if p.is_file()
                           and not (exclude_python_bytecode and
                                    ("__pycache__" in p.relative_to(root).parts
                                     or p.suffix in (".pyc", ".pyo"))))
            def read(path):
                data = path.read_bytes()
                # A computed dependency outside these roots cannot safely be
                # served from a tree snapshot. Ordinary NVRTC still handles it.
                directives = re.findall(rb'^\s*#\s*(?:include|define)\b[^\n]*', data, re.M)
                for directive in directives:
                    # Tokenize whole strings before looking for paths. NVTX's
                    # Windows linker pragma contains an escaped "/EXPORT:"
                    # inside a string; that is not an absolute header name.
                    tokens = re.findall(rb'"((?:\\.|[^"\\\n])*)"|<([^>\n]*)>', directive)
                    for quoted, angled in tokens:
                        name = quoted or angled
                        if not name.startswith((b"/", b"../")):
                            continue
                        target = (path.parent / name.decode()).resolve()
                        if not any(target.is_relative_to(base) for base in resolved_roots):
                            raise ValueError(f"header may escape include roots: {path.name}")
                return (path.relative_to(root).as_posix(), hashlib.sha256(data).hexdigest())
            try:
                with ThreadPoolExecutor(min(16, max(1, len(paths)))) as pool:
                    rows = list(pool.map(read, paths))
            except ValueError as error:
                snapshots[key] = ValueError(str(error))
                raise
            from woof.compiler_metadata import _digest
            snapshots[key] = _digest(rows)
            snapshots[("files", key)] = (paths, rows, tuple(path.resolve() for path in paths))
        files = snapshots.get(("files", key))
        if library and files is not None:
            library_rows[len(result)] = files
        result.append(dict(tree_sha256=snapshots[key]))
    if not result:
        raise ValueError("computed header dependency has no explicit include roots")
    if library_rows:
        _leave_out_unnamed_library_headers(result, library_rows, library, source, snapshots,
                                           (resolved_roots, exclude_python_bytecode))
    return result


def _leave_out_unnamed_library_headers(result, library_rows, library, source, snapshots, roots_key):
    """Rebind each root without the library headers nothing names (see above)."""
    from woof.compiler_metadata import _digest
    candidates = {}
    for paths, _, resolved in library_rows.values():
        for path, real in zip(paths, resolved):
            if real in library:
                candidates.setdefault(path.name, set()).add(real)
    if not candidates:
        return
    names = tuple(sorted(candidates, key=len, reverse=True))
    pattern = re.compile(b"|".join(re.escape(name.encode()) for name in names))
    # Per compiler session, like the tree snapshot (a new process rereads bytes).
    session = snapshots.setdefault(("library-names", roots_key, names), {})
    if "mentions" not in session:
        mentions, base = {}, set()
        for paths, _, resolved in library_rows.values():
            for path, real in zip(paths, resolved):
                if real not in mentions:
                    mentions[real] = frozenset(match.decode() for match in pattern.findall(path.read_bytes()))
                    if real not in library:
                        base |= mentions[real]
        session.update(mentions=mentions, base=frozenset(base))
    mentions = session["mentions"]
    # Names spelled by the source or any bound (non-library) header, then
    # every library header those names reach, to a fixed point.
    spelled = frozenset(match.decode() for match in pattern.findall(source.encode("utf-8")))
    digests = session.get(spelled)
    if digests is None:
        named = set(session["base"] | spelled)
        pending = list(named)
        while pending:
            for real in candidates.get(pending.pop(), ()):
                for name in mentions[real] - named:
                    named.add(name)
                    pending.append(name)
        digests = {}
        for index, (paths, rows, resolved) in library_rows.items():
            kept = [row for path, row, real in zip(paths, rows, resolved)
                    if real not in library or path.name in named]
            digests[index] = _digest(kept)
        session[spelled] = digests
    for index, digest in digests.items():
        result[index] = dict(tree_sha256=digest)


_BUILTIN = frozenset({"assert.h", "ctype.h", "float.h", "limits.h", "math.h",
    "stddef.h", "stdint.h", "stdbool.h", "stdio.h", "stdlib.h", "string.h",
    "cmath", "cstddef", "cstdint", "cfloat", "climits", "cstring"})


def forced_include(options):
    return any(str(option).startswith(("--pre-include", "-include")) for option in options)


def header_dependencies(source, options=(), filename="kern.cu", *, portable=False):
    """Return actual header digests, or raise when their identity is unbound."""
    if forced_include(options):
        raise ValueError("forced include options require the ordinary compiler")
    roots = [Path.cwd()]
    path = Path(str(filename))
    if path.is_absolute():
        roots.insert(0, path.parent)
    options = list(map(str, options))
    for i, option in enumerate(options):
        if option in ("-I", "--include-path") and i + 1 < len(options):
            roots.append(Path(options[i + 1]))
        elif option.startswith("-I") and len(option) > 2:
            roots.append(Path(option[2:]))
        elif option.startswith("--include-path="):
            roots.append(Path(option.split("=", 1)[1]))
    found = {}
    missing = set()
    texts = []
    aliases = {"__has_include"}
    discoveries = set()
    def resolve(name, local, builtin=False):
        candidates = []
        for directory in [local, *roots]:
            candidate = directory / name
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in candidates:
                    candidates.append(resolved)
        if not candidates:
            if builtin and name in _BUILTIN:
                return
            if not portable:
                raise ValueError(f"external header cannot be bound: {name}")
            # Includes inside false guards (including already inlined headers)
            # are not dependencies. Bind absence too: installing the header
            # must invalidate a previous __has_include decision.
            missing.add(name)
        for candidate in candidates:
            key = str(candidate)
            if key in found:
                continue
            if len(found) >= 4096:
                raise ValueError("header dependency limit exceeded")
            data = candidate.read_bytes()
            found[key] = hashlib.sha256(data).hexdigest()
            visit(data.decode("utf-8", "replace"), candidate.parent)

    def visit(text, local):
        text = re.sub(r"\\\r?\n", "", text)
        if 'R"' in text or '??' in text:
            raise ValueError("nonstandard directive spelling requires the ordinary compiler")
        tokens = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|/\*.*?\*/|//[^\n]*", re.S)
        text = tokens.sub(lambda match: " " + "\n" * match.group().count("\n")
                          if match.group().startswith(("/*", "//")) else match.group(), text)
        text = text.replace("%:", "#")
        if "__has_include" in text and not portable:
            raise ValueError("conditional header discovery requires the ordinary compiler")
        texts.append((text, local))
        for directive, operand in re.findall(r"^[^\S\n]*#[^\S\n]*(include_next|include|import)\b([^\n]*)", text, re.M):
            match = re.fullmatch(r'\s*(?:"([^"\n]+)"|<([^>\n]+)>)\s*', operand)
            if directive != "include" or match is None:
                raise ValueError(f"macro or nonliteral include requires the ordinary compiler: {directive} {operand}")
            name = match.group(1) or match.group(2)
            resolve(name, local, bool(match.group(2)))
    visit(source, roots[0])
    if portable:
        # CCCL wraps __has_include in a single-argument identity macro. Bind
        # every literal discovery, even in an inactive branch; refuse computed
        # names rather than guessing a preprocessor result.
        while True:
            before = (len(texts), len(aliases), len(discoveries))
            for text, local in list(texts):
                for alias, arg, target, value in re.findall(
                        r'^\s*#\s*define\s+(\w+)\((\w+)\)\s+(\w+)\((\w+)\)\s*$', text, re.M):
                    if target in aliases and arg == value:
                        aliases.add(alias)
                for line in text.splitlines():
                    if re.match(r'\s*#\s*define\s+(?:' + '|'.join(aliases) + r')\b', line):
                        continue
                    for operand in re.findall(r'\b(?:' + '|'.join(aliases) + r')\s*\(([^)]*)\)', line):
                        match = re.fullmatch(r'\s*(?:"([^"\n]+)"|<([^>\n]+)>)\s*', operand)
                        if match is None:
                            raise ValueError("computed header discovery requires the ordinary compiler")
                        name = match.group(1) or match.group(2)
                        discovery = (name, str(local))
                        if discovery not in discoveries:
                            discoveries.add(discovery)
                            resolve(name, local, bool(match.group(2)))
            if before == (len(texts), len(aliases), len(discoveries)):
                break
        def portable_path(path):
            for i, root in enumerate(roots):
                try:
                    return f"root{i}/" + Path(path).relative_to(root.resolve()).as_posix()
                except ValueError:
                    pass
            return path
        return ([{"path": portable_path(path), "sha256": digest} for path, digest in sorted(found.items())]
                + [{"missing": name} for name in sorted(missing)])
    return [{"path": path, "sha256": digest} for path, digest in sorted(found.items())]
