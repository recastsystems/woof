"""Per-mesh preparation store: build a mesh, its static and its cell ordering once, fetch them everywhere.

THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-12/13).  Nothing a mesh, its
static or its cell ordering contains depends on the forecast date, yet every
box rebuilt them before its cards could start: the 20 -> 2.5 km variable mesh
was generated on a 192-core CPU box in 52 min, relayed to two GPU boxes (6 min
each) and generated again on a workstation in 30.7 min; each box then rebuilt
the same static (73 s to 206 s) and the same Hilbert ordering (171 s) while
its GPUs sat at 0 %.  Every rebuild produced the same bytes (grid 8a113c68,
static 4164c5ed, ordered grid cda1b669 on all three boxes).

WHAT IS STORED, AND THE KEY THAT NAMES IT.  An entry is keyed by everything
its bytes depend on, so the same key means the same bytes:

  mesh    generator identity + the canonical resolution request + every flag
          that changes the output                 -> grid.nc, receipt.json
  static  static-builder identity + grid SHA-256 + the geography it reads
          (fingerprint of the 12 dataset directories, resolved exactly as
          rw-mpas ``resolve_geog_dataset_dir`` resolves them) + options
                                                  -> static.nc, receipt.json
  order   hex_order identity + grid SHA-256 + static SHA-256
                                                  -> ordering.nc, ordered.grid.nc, ordered.static.nc

A fetched file is published at its destination only after its SHA-256 equals
the entry's, so a hit is byte-identical to what the named engine wrote.

ENGINE IDENTITY.  By default an engine is its binary's SHA-256.  Builds of the
same source at different paths differ in bytes (panic locations), so the store
carries an ``engines`` registry mapping attested binaries to a source class
(``rw-mpas@<commit>``).  Attesting is a deliberate act on the desk tool, with
evidence; nothing here infers equivalence.  An unattested binary still works:
it simply only hits entries its own bytes built.

WHERE THE STORE IS.  ``WOOF_HEX_PREP_STORE`` lists stores (os.pathsep), first
hit wins.  Each is either a directory laid out ``<root>/<kind>/<key>/entry.json``
plus the files (a shared disk or a local mirror), or a ticket ``*.json`` of
time-limited GET URLs written by the desk tool -- a box never holds object-store
credentials.  ``WOOF_HEX_PREP_STORE_OUTBOX`` names a directory store a fresh
build is staged into (hard links, else copies) for the desk tool to publish.
``off`` in either disables it.  With neither set nothing changes: every call
builds exactly as before.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen

from .file_digest import record as _record_digest, sha256_file as _sha256_file

ENTRY_SCHEMA = "gpuwm-hex.prep-store-entry/v1"
KEY_SCHEMA = "gpuwm-hex.prep-store-key/v1"
TICKET_SCHEMA = "gpuwm-hex.prep-store-ticket/v1"
ENGINES_SCHEMA = "gpuwm-hex.prep-store-engines/v1"
ENV_STORE = "WOOF_HEX_PREP_STORE"
ENV_OUTBOX = "WOOF_HEX_PREP_STORE_OUTBOX"
ENV_ENGINES = "WOOF_HEX_PREP_STORE_ENGINES"
ENV_STREAMS = "WOOF_HEX_PREP_STORE_STREAMS"
KINDS = ("mesh", "static", "order", "ugwp_oro")
_CHUNK = 64 << 20


class PrepStoreError(RuntimeError):
    """A store answered with bytes that are not the entry's; nothing was published.

    ``fetch_entry`` catches it per entry and the caller builds instead."""


#: Digests this process already proved, by exact file version: a file this
#: module just fetched and verified is not read a second time to key the next
#: artifact (the static key needs the grid's SHA-256, the order key both).
_VERIFIED: dict[tuple[int, int, int, int, int], str] = {}


def _version(path: Path) -> tuple[int, int, int, int, int]:
    st = Path(path).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def sha256_file(path: str | Path) -> str:
    known = _VERIFIED.get(_version(Path(path)))
    return known if known is not None else _sha256_file(path)


def _log(message: str) -> None:
    print(f"prep-store: {message}", file=sys.stderr, flush=True)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def key_of(material: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(material).encode("ascii")).hexdigest()


def _disabled(text: str | None) -> bool:
    return (text or "").strip().lower() in ("", "off", "0", "none")


# ---------------------------------------------------------------------------
# engine identity
# ---------------------------------------------------------------------------
def _load_registry_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("schema") != ENGINES_SCHEMA:
        return {}
    return dict(data.get("engines", {}))


def engine_registry(stores: Sequence["Store"] | None = None) -> dict[str, Any]:
    """Attested binary SHA-256 -> {class, name, evidence}, merged from every configured store."""
    merged: dict[str, Any] = {}
    for store in (stores if stores is not None else configured_stores()):
        for sha, row in store.engines().items():
            merged.setdefault(sha, row)
    extra = os.environ.get(ENV_ENGINES)
    if extra and not _disabled(extra):
        for sha, row in _load_registry_file(Path(extra)).items():
            merged.setdefault(sha, row)
    return merged


def engine_identity(binary: str | Path, registry: Mapping[str, Any]) -> dict[str, str]:
    sha = sha256_file(Path(binary))
    row = registry.get(sha)
    if row and row.get("class"):
        return {"class": str(row["class"]), "binary_sha256": sha}
    return {"class": f"sha256:{sha}", "binary_sha256": sha}


# ---------------------------------------------------------------------------
# geography fingerprint (mirrors rw-mpas static_builder.rs, 7008877b6)
# ---------------------------------------------------------------------------
#: GEOG_DATASET_TABLE: (slot, directory names the static builders read for it, newest first).
#: A slot is keyed over every one of its directories that exists, and is missing only when none does.
#: BREAKAGE THIS PREVENTS: the table named one directory per slot (rw-mpas 7008877b6), and woof 2.9.5's
#: rw_mpas_static reads three others (texture_top in soilgrids, soiltemp_era5_1deg, and the lake-free MODIS land
#: use). On a box laid out for it (the WOOF run-box image) soiltype_top_30s and soiltemp_1deg do not exist, so
#: every static request read as "missing", was never keyed, and was rebuilt on every run: 318 s for the 30 km
#: region's parent, 2026-10-23 (`prep_store stage`: "this command is built, never stored").
GEOG_DATASETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("terrain", ("topo_gmted2010_30s",)),
    ("landuse", ("modis_landuse_20class_30s", "modis_landuse_20class_30s_with_lakes")),
    ("soilcat", ("texture_top", "soiltype_top_30s")),
    ("greenfrac", ("greenfrac_fpar_modis",)),
    ("albedo", ("albedo_modis",)),
    ("snow_albedo", ("maxsnowalb_modis",)),
    ("soil_temperature", ("soiltemp_era5_1deg", "soiltemp_1deg")),
    ("soilcomp", ("soilcomp",)),
    ("soilcl1", ("texture_layer1",)),
    ("soilcl2", ("texture_layer2",)),
    ("soilcl3", ("texture_layer3",)),
    ("soilcl4", ("texture_layer4",)),
)
#: GEOG_CONTAINER_TABLE: (directory, container it may sit inside).
GEOG_CONTAINERS: dict[str, str] = {
    "soilcomp": "soilgrids", "texture_top": "soilgrids", "texture_bot": "soilgrids",
    "texture_layer1": "soilgrids", "texture_layer2": "soilgrids",
    "texture_layer3": "soilgrids", "texture_layer4": "soilgrids",
}


def resolve_geog_dataset_dir(root: Path, dataset: str) -> Path:
    """Flat ``root/<dataset>`` when it has an ``index``, else the container's, else flat."""
    flat = root / dataset
    if (flat / "index").is_file():
        return flat
    container = GEOG_CONTAINERS.get(dataset)
    if container is not None:
        nested = root / container / dataset
        if (nested / "index").is_file():
            return nested
    return flat


def geog_fingerprint(root: str | Path) -> dict[str, Any]:
    """Which directory each slot reads and what is in it: every file's name and size, the ``index`` digest.

    Names and sizes of every tile, plus the SHA-256 of each ``index``, from a
    directory walk that takes about a second; hashing the tiles themselves
    (tens of GB) would cost more than the build it saves.  A pinned archive
    re-published with different tiles of identical names and sizes would not be
    seen; the pins in woof/geog_assets.py are what keep that from happening.
    """
    root = Path(root).expanduser().resolve()
    slots: dict[str, Any] = {}
    dirs: dict[str, Any] = {}
    for slot, candidates in GEOG_DATASETS:
        found = {}
        for dataset in candidates:
            where = resolve_geog_dataset_dir(root, dataset)
            if not where.is_dir():
                continue
            listing = []
            for dirpath, dirnames, filenames in os.walk(where):
                dirnames.sort()
                for name in sorted(filenames):
                    path = Path(dirpath) / name
                    listing.append([str(path.relative_to(where)).replace(os.sep, "/"), path.stat().st_size])
            index = where / "index"
            found[dataset] = {
                "dir": str(where.relative_to(root)).replace(os.sep, "/"),
                "files": hashlib.sha256(_canonical(listing).encode("ascii")).hexdigest(),
                "n_files": len(listing),
                "bytes": sum(size for _, size in listing),
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None,
            }
        if not found:
            slots[slot] = {"missing": True, "candidates": list(candidates)}
            dirs[slot] = None
            continue
        # One candidate keeps the slot's old key material exactly (an existing store's keys stay valid when only
        # one directory is present); several are keyed together.
        if len(found) == 1:
            slots[slot] = {k: v for k, v in next(iter(found.values())).items() if k != "dir"}
        else:
            slots[slot] = {name: {k: v for k, v in row.items() if k != "dir"} for name, row in sorted(found.items())}
        dirs[slot] = next(iter(found.values()))["dir"] if len(found) == 1 else [row["dir"] for row in found.values()]
    # Where a dataset sits (flat or inside its container) does not change what
    # is read, so the location is reported but not keyed.
    keyed = slots
    return {"slots": keyed, "dirs": dirs, "sha256": key_of(keyed)}


# ---------------------------------------------------------------------------
# stores
# ---------------------------------------------------------------------------
class Store:
    name = "store"

    def engines(self) -> dict[str, Any]:
        return {}

    def pinned_geog(self) -> str | None:
        return None

    def entry(self, kind: str, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def fetch(self, kind: str, key: str, entry: Mapping[str, Any], targets: Mapping[str, Path]) -> int:
        raise NotImplementedError


class DirStore(Store):
    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.name = str(self.root)

    def engines(self) -> dict[str, Any]:
        return _load_registry_file(self.root / "engines.json")

    def pinned_geog(self):
        try:
            return json.loads((self.root / "geog.json").read_text(encoding="utf-8")).get("pinned")
        except (OSError, ValueError):
            return None

    def entry(self, kind, key):
        path = self.root / kind / key / "entry.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if data.get("schema") == ENTRY_SCHEMA and data.get("key") == key else None

    def fetch(self, kind, key, entry, targets):
        moved = 0
        for name, dest in targets.items():
            partial = _partial(dest)
            shutil.copyfile(self.root / kind / key / name, partial)
            moved += partial.stat().st_size
        return moved


class TicketStore(Store):
    """Time-limited GET URLs for every object of every entry; no credentials."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self.name = f"ticket {self.path.name}"
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema") != TICKET_SCHEMA:
            raise PrepStoreError(f"{self.path} is not a {TICKET_SCHEMA} ticket")
        self.data = data
        expires = data.get("expires_epoch")
        self.expired = bool(expires) and time.time() > float(expires)
        if self.expired:
            _log(f"{self.path} expired at {data.get('expires')}; ask the desk tool for a new ticket")

    def engines(self):
        return dict(self.data.get("engines", {}))

    def pinned_geog(self):
        return self.data.get("geog", {}).get("pinned")

    def entry(self, kind, key):
        if self.expired:
            return None
        row = self.data.get("entries", {}).get(f"{kind}/{key}")
        if not row:
            return None
        entry = row.get("entry", {})
        return entry if entry.get("schema") == ENTRY_SCHEMA and entry.get("key") == key else None

    def fetch(self, kind, key, entry, targets):
        urls = self.data["entries"][f"{kind}/{key}"]["urls"]
        jobs = []
        streams = int(os.environ.get(ENV_STREAMS) or min(64, max(8, 2 * (os.cpu_count() or 4))))
        for name, dest in targets.items():
            size = int(entry["files"][name]["bytes"])
            partial = _partial(dest)
            with open(partial, "wb") as stream:
                stream.truncate(size)
            # Enough ranges to fill every stream on a small file (measured on a
            # workstation: a 137 MB entry in 64 MiB ranges ran on 3 streams at 8.5 MB/s).
            chunk = max(min(_CHUNK, -(-size // streams)), 4 << 20)
            for start in range(0, size, chunk):
                jobs.append((urls[name], partial, start, min(start + chunk, size) - 1))
        with ThreadPoolExecutor(max_workers=max(1, min(streams, len(jobs) or 1))) as pool:
            return sum(pool.map(lambda job: _get_range(*job), jobs))


def _get_range(url: str, partial: Path, first: int, last: int) -> int:
    want = last - first + 1
    error: Exception | None = None
    for attempt in range(5):
        try:
            request = Request(url, headers={"Range": f"bytes={first}-{last}"})
            got = 0
            with urlopen(request, timeout=120) as response, open(partial, "r+b") as stream:
                if response.status != 206:
                    raise PrepStoreError(f"ranged GET answered {response.status}")
                stream.seek(first)
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    stream.write(block)
                    got += len(block)
            if got != want:
                raise PrepStoreError(f"short range {got} of {want} bytes")
            return got
        except Exception as caught:  # noqa: BLE001 - a transfer is retried, then refused by name
            error = caught
            time.sleep(min(8.0, 0.5 * 2 ** attempt))
    raise PrepStoreError(f"range {first}-{last} of {partial.name} failed after 5 attempts: {error}")


def _partial(dest: Path) -> Path:
    return dest.with_name(f".{dest.name}.prep-store-partial")


def configured_stores() -> list[Store]:
    text = os.environ.get(ENV_STORE, "")
    if _disabled(text):
        return []
    stores: list[Store] = []
    for item in text.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        try:
            stores.append(TicketStore(path) if path.is_file() else DirStore(path))
        except (OSError, ValueError, PrepStoreError) as error:
            _log(f"store {item} unusable ({error}); building instead")
    return stores


def outbox() -> Path | None:
    text = os.environ.get(ENV_OUTBOX, "")
    return None if _disabled(text) else Path(text).expanduser()


# ---------------------------------------------------------------------------
# fetch and stage
# ---------------------------------------------------------------------------
def _hash_many(paths: Iterable[Path]) -> list[str]:
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=max(1, len(paths))) as pool:
        return list(pool.map(lambda p: _plain_sha(p), paths))


def _plain_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_entry(kind: str, key: str, targets: Mapping[str, Path],
                stores: Sequence[Store] | None = None) -> dict[str, Any] | None:
    """Publish the entry's files at ``targets`` (file name -> destination), or return None on a miss."""
    for store in (stores if stores is not None else configured_stores()):
        entry = store.entry(kind, key)
        if entry is None:
            continue
        if not set(targets) <= set(entry.get("files", {})):
            continue  # an entry without a file this caller needs is a miss, never a partial hit
        started = time.perf_counter()
        for dest in targets.values():
            dest.parent.mkdir(parents=True, exist_ok=True)
        names = list(targets)
        partials = [_partial(targets[name]) for name in names]
        # A bad entry (bytes that are not the entry's, a cut transfer, a file
        # the store no longer holds) is refused and never published, and the
        # caller builds: on 2026-10-23 one entry whose receipt was not its
        # entry.json's bytes stopped six region doors instead of letting them build.
        try:
            moved = store.fetch(kind, key, entry, targets)
            fetched = time.perf_counter() - started
            digests = _hash_many(partials)
            bad = [n for n, d in zip(names, digests) if d != entry["files"][n]["sha256"]]
            if bad:
                raise PrepStoreError(f"{store.name} {kind}/{key[:12]}: {', '.join(bad)} differ from the "
                                     f"entry's SHA-256; nothing published")
        except (OSError, ValueError, KeyError, TypeError, PrepStoreError) as error:
            for partial in partials:
                partial.unlink(missing_ok=True)
            _log(f"BAD STORE ENTRY REFUSED: {kind} {key[:12]} from {store.name}: {error}. "
                 f"Nothing from it was published; trying the next store, else building.")
            continue
        for name, partial in zip(names, partials):
            os.replace(partial, targets[name])
            _VERIFIED[_version(targets[name])] = entry["files"][name]["sha256"]
            _record_digest(targets[name], entry["files"][name]["sha256"])
        seconds = time.perf_counter() - started
        _log(f"{kind} {key[:12]} from {store.name}: {moved / 1e9:.2f} GB in {fetched:.1f} s, "
             f"SHA-256 verified, {seconds:.1f} s total")
        return {"store": store.name, "seconds": round(seconds, 2), "transfer_seconds": round(fetched, 2),
                "bytes": moved, "entry": entry}
    return None


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def stage_entry(kind: str, key: str, material: Mapping[str, Any], files: Mapping[str, Path],
                *, root: Path | None = None, built: Mapping[str, Any] | None = None) -> Path | None:
    """Place a finished build into the outbox (or ``root``) as a store entry; entry.json lands last."""
    root = root if root is not None else outbox()
    if root is None:
        return None
    where = Path(root) / kind / key
    where.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, path in files.items():
        _link_or_copy(Path(path), where / name)
        digest = sha256_file(Path(path))
        _VERIFIED[_version(Path(path))] = digest
        rows[name] = {"bytes": Path(path).stat().st_size, "sha256": digest}
    entry = {"schema": ENTRY_SCHEMA, "kind": kind, "key": key, "material": dict(material), "files": rows,
             "built": {"host": socket.gethostname(),
                       "at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       **dict(built or {})}}
    partial = where / ".entry.json.partial"
    partial.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, where / "entry.json")
    _log(f"{kind} {key[:12]} staged for publication at {where}")
    return where


# ---------------------------------------------------------------------------
# the requests the binaries take, parsed into keys
# ---------------------------------------------------------------------------
_MESH_DEFAULTS = {"--triangulation": "rebuild", "--fit-spacing": "no", "--sweeps": "300",
                  "--tolerance": str(1e-3), "--omega": str(1.4)}
_MESH_VALUED = {"--spec", "--background-km", "--cells", "--card", "--vram-gib", "--fit-spacing",
                "--sweeps", "--tolerance", "--omega", "--triangulation", "--out", "--receipt"}
_MESH_SWITCHES = {"--clobber"}
_STATIC_VALUED = {"--grid", "--out", "--geog", "--nominal-dx-m", "--valid-time", "--receipt", "--supersample-30s"}
# --native-init and --native-gwd-static (the native static stream the global and variable-resolution doors ask for)
# change the bytes, so they are keyed; before, a static asked with them was never stored.
_STATIC_SWITCHES = {"--clobber", "--terrain-only", "--native-init", "--native-gwd-static"}
_FLOATS = {"--background-km", "--vram-gib", "--tolerance", "--omega", "--nominal-dx-m"}


def _parse(argv: Sequence[str], valued: set[str], switches: set[str]) -> dict[str, Any] | None:
    """Flags -> values; None when any argument is one this store does not understand."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in switches:
            out[flag] = True
            i += 1
        elif flag in valued and i + 1 < len(argv):
            value = argv[i + 1]
            out[flag] = repr(float(value)) if flag in _FLOATS else str(value)
            i += 2
        else:
            return None
    return out


def canonical_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The request exactly as written, with the two keys the generator defaults made explicit.

    Measured 2026-10-13: ``--spec`` holding
    ``{"background_km": 20.0, "name": null, "regions": []}`` writes the same
    grid (0e04546d) as ``--background-km 20`` did on three other boxes.
    """
    spec = dict(spec)
    spec.setdefault("name", None)
    spec.setdefault("regions", [])
    return spec


def mesh_request(argv: Sequence[str], registry: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Path]] | None:
    """``rw_mpas_mesh`` argv (binary first) -> (key, material, outputs), or None to build without the store."""
    flags = _parse(argv[1:], _MESH_VALUED, _MESH_SWITCHES)
    if flags is None or "--out" not in flags:
        return None
    if ("--spec" in flags) == ("--background-km" in flags):
        return None
    if "--spec" in flags:
        spec = canonical_spec(json.loads(Path(flags["--spec"]).read_text(encoding="utf-8")))
    else:
        spec = canonical_spec({"background_km": float(flags["--background-km"])})
    options = {k: v for k, v in flags.items()
               if k not in ("--spec", "--background-km", "--out", "--receipt", "--clobber")
               and _MESH_DEFAULTS.get(k) != (repr(float(v)) if k in ("--tolerance", "--omega") else v)}
    material = {"schema": KEY_SCHEMA, "kind": "mesh", "engine": engine_identity(argv[0], registry)["class"],
                "spec": spec, "options": options}
    outputs = {"grid.nc": Path(flags["--out"])}
    if "--receipt" in flags:
        outputs["receipt.json"] = Path(flags["--receipt"])
    return key_of(material), material, outputs


def pinned_geog(stores: Sequence[Store] | None = None) -> str | None:
    """The fingerprint of a geography root staged from woof's pinned archives, as a store declares it."""
    for store in (stores if stores is not None else configured_stores()):
        value = store.pinned_geog()
        if value:
            return value
    return None


def static_request(argv: Sequence[str], registry: Mapping[str, Any],
                   pinned: str | None = None) -> tuple[str, dict[str, Any], dict[str, Path]] | None:
    flags = _parse(argv[1:], _STATIC_VALUED, _STATIC_SWITCHES)
    if flags is None or "--out" not in flags or "--grid" not in flags or "--geog" not in flags:
        return None  # without an explicit --geog the binary's own ladder picks the archive; build
    root = Path(flags["--geog"]).expanduser()
    if not root.exists() and pinned:
        # No archive on this box at all: the build would need `woof fetch-geog`
        # first, which stages exactly the pinned archives, so the static it would
        # then write is the one built from the pinned fingerprint.  A PARTIAL
        # root is never treated this way: the binary reads what is there.
        geog_sha = pinned
    else:
        geog = geog_fingerprint(root)
        if any(slot.get("missing") for slot in geog["slots"].values()):
            return None  # the binary refuses an incomplete archive by name; let it
        geog_sha = geog["sha256"]
    material = {"schema": KEY_SCHEMA, "kind": "static", "engine": engine_identity(argv[0], registry)["class"],
                "grid_sha256": sha256_file(Path(flags["--grid"])), "geog_sha256": geog_sha,
                "options": {"nominal_dx_m": flags.get("--nominal-dx-m"), "terrain_only": bool(flags.get("--terrain-only")),
                            "valid_time": flags.get("--valid-time")}}
    for flag in ("--native-init", "--native-gwd-static"):
        if flags.get(flag):  # only when given, so every key built without them is unchanged
            material["options"][flag.lstrip("-").replace("-", "_")] = True
    if flags.get("--supersample-30s") is not None:
        # Only when given, so every key built without it is unchanged.
        material["options"]["supersample_30s"] = flags["--supersample-30s"]
    outputs = {"static.nc": Path(flags["--out"])}
    if "--receipt" in flags:
        outputs["receipt.json"] = Path(flags["--receipt"])
    return key_of(material), material, outputs


def request_for(argv: Sequence[str], registry: Mapping[str, Any], pinned: str | None = None):
    name = Path(str(argv[0])).name.removesuffix(".exe")
    if name.startswith("rw_mpas_mesh"):
        return "mesh", mesh_request([str(a) for a in argv], registry)
    if name.startswith("rw_mpas_static"):
        return "static", static_request([str(a) for a in argv], registry, pinned)
    return None, None


def run_or_fetch(argv: Sequence[str | Path], *, kind: str | None = None,
                 runner: Callable[[list[str]], Any] | None = None) -> dict[str, Any]:
    """Fetch what this ``rw_mpas_mesh``/``rw_mpas_static`` command would write, else run it and stage the result.

    ``runner`` runs the argv (default: subprocess.run(check=True)); its return
    value is passed back under ``"completed"`` on a build.
    """
    argv = [str(a) for a in argv]
    runner = runner or (lambda a: subprocess.run(a, check=True))
    stores = configured_stores()
    box = outbox()
    if not stores and box is None:
        return {"source": "built", "completed": runner(argv)}
    registry = engine_registry(stores)
    found_kind, request = request_for(argv, registry, pinned_geog(stores))
    kind = kind or found_kind
    if request is None:
        return {"source": "built", "completed": runner(argv)}
    key, material, outputs = request
    clobber = "--clobber" in argv
    if clobber or not any(p.exists() for p in outputs.values()):
        hit = fetch_entry(kind, key, outputs, stores)
        if hit is not None:
            return {"source": "store", "kind": kind, "key": key, **hit}
    started = time.perf_counter()
    completed = runner(argv)
    seconds = time.perf_counter() - started
    succeeded = getattr(completed, "returncode", 0) == 0
    if succeeded and box is not None and all(p.exists() for p in outputs.values()):
        stage_entry(kind, key, material, outputs, root=box,
                    built={"seconds": round(seconds, 2), "engine": engine_identity(argv[0], registry)})
    return {"source": "built", "kind": kind, "key": key, "seconds": round(seconds, 2), "completed": completed}


def fetch_only(argv: Sequence[str | Path]) -> dict[str, Any] | None:
    """What :func:`run_or_fetch` would fetch for this command, or None (no store, an unkeyed command, a miss).

    Never builds: the zoom-globe door asks whether a library holds this exact request before it rotates a
    template instead of generating."""
    argv = [str(a) for a in argv]
    stores = configured_stores()
    if not stores:
        return None
    registry = engine_registry(stores)
    kind, request = request_for(argv, registry, pinned_geog(stores))
    if request is None:
        return None
    key, _material, outputs = request
    if any(p.exists() for p in outputs.values()):
        return None
    hit = fetch_entry(kind, key, outputs, stores)
    return None if hit is None else {"source": "store", "kind": kind, "key": key, **hit}


# ---------------------------------------------------------------------------
# cell ordering
# ---------------------------------------------------------------------------
ORDER_FILES = ("ordering.nc", "ordered.grid.nc", "ordered.static.nc")


def order_request(grid: Path, static: Path, hex_order: Path, registry: Mapping[str, Any]):
    material = {"schema": KEY_SCHEMA, "kind": "order", "engine": engine_identity(hex_order, registry)["class"],
                "grid_sha256": sha256_file(grid), "static_sha256": sha256_file(static),
                "steps": ["--make-map GRID ordering.nc", "--apply ordering.nc GRID ordered.grid.nc",
                          "--record-donor ordering.nc ordered.grid.nc",
                          "--apply ordering.nc STATIC ordered.static.nc",
                          "--record-donor ordering.nc ordered.static.nc"]}
    return key_of(material), material


def order(grid: Path, static: Path, out_dir: Path, hex_order: Path) -> dict[str, Any]:
    """Hilbert-order a grid and its static (hex_order), or fetch the same three files from the store."""
    grid, static, out_dir = Path(grid), Path(static), Path(out_dir)
    targets = {name: out_dir / name for name in ORDER_FILES}
    stores = configured_stores()
    box = outbox()
    key = material = None
    if stores or box is not None:
        registry = engine_registry(stores)
        key, material = order_request(grid, static, Path(hex_order), registry)
    if stores and not any(p.exists() for p in targets.values()):
        hit = fetch_entry("order", key, targets, stores)
        if hit is not None:
            return {"source": "store", "kind": "order", "key": key, **hit}
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run([str(hex_order), *map(str, a)], check=True)  # noqa: E731
    run("--make-map", grid, targets["ordering.nc"])
    for role, src in (("grid", grid), ("static", static)):
        run("--apply", targets["ordering.nc"], src, targets[f"ordered.{role}.nc"])
        run("--record-donor", targets["ordering.nc"], targets[f"ordered.{role}.nc"])
    seconds = time.perf_counter() - started
    if box is not None:
        stage_entry("order", key, material, targets, root=box,
                    built={"seconds": round(seconds, 2), "engine": engine_identity(hex_order, registry)})
    return {"source": "built", "kind": "order", "key": key, "seconds": round(seconds, 2)}


# ---------------------------------------------------------------------------
# GSL drag orographic statistics (<mesh>.ugwp_oro_data.nc)
# ---------------------------------------------------------------------------
def ugwp_oro_request(grid: Path, geog: Path, registry: Mapping[str, Any]):
    """Key: the generator's identity and method, the grid's SHA-256 and the 30" terrain it reads."""
    from . import ugwp_oro as _ugwp
    material = {"schema": KEY_SCHEMA, "kind": "ugwp_oro", "engine": engine_identity(_ugwp.binary(), registry)["class"],
                "method": _ugwp.METHOD, "grid_sha256": sha256_file(grid), "terrain": _ugwp.topo_fingerprint(geog)}
    return key_of(material), material


def ugwp_oro(grid: Path, geog: Path, out: Path, *, builder: Callable[[], Any],
             receipt: Path | None = None) -> dict[str, Any]:
    """The mesh's ugwp_oro_data.nc: fetched from the store when it holds this grid's, else built and staged.

    The file depends on the grid and the terrain alone, never the cycle, so it
    rides the store beside the mesh's static (THE BREAKAGE: every RRFS-shape
    run on a library mesh would otherwise rebuild it before its card starts).
    """
    targets = {"ugwp_oro_data.nc": Path(out)}
    if receipt is not None:
        targets["receipt.json"] = Path(receipt)
    stores = configured_stores()
    box = outbox()
    if not stores and box is None:
        builder()
        return {"source": "built"}
    registry = engine_registry(stores)
    key, material = ugwp_oro_request(Path(grid), Path(geog), registry)
    if stores and not Path(out).exists():
        hit = fetch_entry("ugwp_oro", key, {"ugwp_oro_data.nc": Path(out)}, stores)
        if hit is not None:
            return {"source": "store", "kind": "ugwp_oro", "key": key,
                    **{k: v for k, v in hit.items() if k != "entry"}}
    started = time.perf_counter()
    builder()
    seconds = time.perf_counter() - started
    if box is not None:
        from . import ugwp_oro as _ugwp
        stage_entry("ugwp_oro", key, material, {k: v for k, v in targets.items() if v.exists()}, root=box,
                    built={"seconds": round(seconds, 2), "engine": engine_identity(_ugwp.binary(), registry)})
    return {"source": "built", "kind": "ugwp_oro", "key": key, "seconds": round(seconds, 2)}


def mesh_set(*, out_dir: Path, name: str, mesh_exe: Path, static_exe: Path, geog: Path,
             hex_order: Path | None, spec: Path | None = None, background_km: float | None = None) -> dict[str, Any]:
    """Grid, static and (with ``hex_order``) the ordered pair for one mesh request, in one process.

    One process so a digest proved by a fetch keys the next artifact without
    re-reading the file (the 20 km grid and static are 5 GB).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid, static = out_dir / f"{name}.grid.nc", out_dir / f"{name}.static.nc"
    request = ["--spec", str(spec)] if spec is not None else ["--background-km", repr(float(background_km))]
    steps: dict[str, Any] = {}
    started = time.perf_counter()
    for step, argv in (
            ("mesh", [str(mesh_exe), *request, "--out", str(grid), "--receipt", str(out_dir / f"{name}.mesh.json")]),
            ("static", [str(static_exe), "--grid", str(grid), "--geog", str(geog), "--out", str(static),
                        "--receipt", str(out_dir / f"{name}.static.json")])):
        t0 = time.perf_counter()
        result = run_or_fetch(argv)
        steps[step] = {"source": result["source"], "key": result.get("key"), "seconds": round(time.perf_counter() - t0, 2)}
    if hex_order is not None:
        t0 = time.perf_counter()
        result = order(grid, static, out_dir / "order", Path(hex_order))
        steps["order"] = {"source": result["source"], "key": result.get("key"), "seconds": round(time.perf_counter() - t0, 2)}
    return {"out_dir": str(out_dir), "steps": steps, "seconds": round(time.perf_counter() - started, 2)}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _engines_arg(path: Path | None) -> dict[str, Any]:
    registry = engine_registry()
    if path is not None:
        registry.update(_load_registry_file(path))
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m woof.hex.prep_store", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="fetch what an rw_mpas_mesh / rw_mpas_static command would write, else run it")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    key = sub.add_parser("key", help="print the key and key material a command would use")
    key.add_argument("--engines", type=Path)
    key.add_argument("argv", nargs=argparse.REMAINDER)
    stage = sub.add_parser("stage", help="stage outputs an earlier run of this exact command already wrote")
    stage.add_argument("--into", type=Path, required=True, help="directory store to stage into")
    stage.add_argument("--engines", type=Path)
    stage.add_argument("argv", nargs=argparse.REMAINDER)
    ordr = sub.add_parser("order", help="hex_order --make-map/--apply/--record-donor for a grid and static, or fetch")
    for name in ("--grid", "--static", "--out-dir", "--hex-order"):
        ordr.add_argument(name, type=Path, required=True)
    ordr.add_argument("--stage-existing", type=Path, metavar="STORE",
                      help="stage the three files already in --out-dir into this directory store")
    ordr.add_argument("--engines", type=Path)
    geog = sub.add_parser("geog", help="print the geography fingerprint the static key uses")
    geog.add_argument("root", type=Path)
    mset = sub.add_parser("set", help="grid + static + ordered pair for one mesh request: fetch each, else build it")
    mset.add_argument("--out-dir", type=Path, required=True)
    mset.add_argument("--name", required=True)
    what = mset.add_mutually_exclusive_group(required=True)
    what.add_argument("--spec", type=Path)
    what.add_argument("--background-km", type=float)
    mset.add_argument("--geog", type=Path, required=True)
    mset.add_argument("--mesh-exe", type=Path)
    mset.add_argument("--static-exe", type=Path)
    mset.add_argument("--hex-order", type=Path, help="also order the pair (omit to skip)")
    args = parser.parse_args(argv)
    if args.command == "set":
        from .engines import MESH, STATIC, resolve
        result = mesh_set(out_dir=args.out_dir, name=args.name, spec=args.spec, background_km=args.background_km,
                          geog=args.geog, hex_order=args.hex_order,
                          mesh_exe=resolve(MESH, args.mesh_exe), static_exe=resolve(STATIC, args.static_exe))
        print(json.dumps(result))
        return 0

    def strip(rest: list[str]) -> list[str]:
        return rest[1:] if rest[:1] == ["--"] else rest

    if args.command == "geog":
        print(json.dumps(geog_fingerprint(args.root), indent=2))
        return 0
    if args.command == "run":
        result = run_or_fetch(strip(args.argv))
        result.pop("completed", None)
        result.pop("entry", None)
        print(json.dumps(result))
        return 0
    if args.command in ("key", "stage"):
        command = strip(args.argv)
        kind, request = request_for(command, _engines_arg(args.engines))
        if request is None:
            print("this command is built, never stored (unrecognised flags or no --geog)", file=sys.stderr)
            return 2
        key_, material, outputs = request
        if args.command == "key":
            print(json.dumps({"kind": kind, "key": key_, "material": material}, indent=2))
            return 0
        missing = [str(p) for p in outputs.values() if not p.is_file()]
        if missing:
            print(f"cannot stage: {', '.join(missing)} not written", file=sys.stderr)
            return 2
        stage_entry(kind, key_, material, outputs, root=args.into,
                    built={"staged_from": {n: str(p) for n, p in outputs.items()},
                           "engine": engine_identity(command[0], _engines_arg(args.engines))})
        print(json.dumps({"kind": kind, "key": key_}))
        return 0
    if args.command == "order":
        if args.stage_existing is not None:
            registry = _engines_arg(args.engines)
            key_, material = order_request(args.grid, args.static, args.hex_order, registry)
            stage_entry("order", key_, material, {n: args.out_dir / n for n in ORDER_FILES},
                        root=args.stage_existing,
                        built={"staged_from": str(args.out_dir), "engine": engine_identity(args.hex_order, registry)})
            print(json.dumps({"kind": "order", "key": key_}))
            return 0
        result = order(args.grid, args.static, args.out_dir, args.hex_order)
        result.pop("entry", None)
        print(json.dumps(result))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
