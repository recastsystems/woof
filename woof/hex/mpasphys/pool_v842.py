"""Native Registry pool ownership. MPAS Registry.xml:6-55,1360-4020.

Rust decodes/permutes file payloads. This module selects metadata, allocates the
same named pools and time levels, and copies ABI buffers; it does no field maths.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import numpy as np

from ..config import parse_fortran_namelist
from ..host_parallel import threads as host_copy_threads
from .driver_v842 import DriverConfig

_ENTITY = {"nCells", "nEdges", "nVertices"}

# Payload digests this process has computed, keyed by the file's stat identity (see NativePoolHydration).
_VERIFIED_DIGESTS: dict[tuple, str] = {}
_VERIFIED_LOCK = __import__("threading").Lock()
# Bytes of payload read ahead of the serial native copies in populate().
_PREFETCH_BYTES = 8 << 30


def remember_payload_digest(path, value: str) -> None:
    """Record the SHA-256 this process computed from the exact bytes it just wrote to ``path``, under the file's
    stat identity, as NativePoolHydration's own digest() records what it hashes (partition_v842.project_payloads:
    the rank payloads were hashed while written and then read back whole and hashed again, ~4 s on conus3km).
    Any later write changes the identity, so a changed file is hashed again and refused exactly as before."""
    import os
    info = os.stat(path)
    identity = (os.path.realpath(path), info.st_dev, info.st_ino, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)
    with _VERIFIED_LOCK:
        _VERIFIED_DIGESTS[identity] = value


def registry_text(path: Path) -> str:
    """Physics-enabled Registry preprocessing; Registry.xml:36-55,4016-4020."""
    defined = {"DO_PHYSICS"}
    stack = [True]
    result = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(("#ifdef ", "#ifndef ")):
            directive, name = stripped.split()
            enabled = name in defined
            if directive == "#ifndef":
                enabled = not enabled
            stack.append(stack[-1] and enabled)
        elif stripped == "#else":
            stack[-1] = stack[-2] and not stack[-1]
        elif stripped == "#endif":
            stack.pop()
        elif stripped.startswith("#include ") and stack[-1]:
            result.append(registry_text(path.parent / stripped.split('"')[1]))
        elif stripped.startswith("#"):
            if not stripped.startswith("#include "):
                raise ValueError(f"unsupported Registry directive {stripped}")
        elif stack[-1]:
            result.append(line)
    if len(stack) != 1:
        raise ValueError("unbalanced Registry conditional")
    return "\n".join(result)


def active_packages(config: DriverConfig, configs: Mapping[str, Any]) -> set[str]:
    """Physics packages; mpas_atmphys_packages.F:52-205; Registry.xml:466-491."""
    o = config.execution_options
    packages = {"no_invariant_stream"}
    mapping = {"mp_kessler": "mp_kessler_in", "mp_thompson": "mp_thompson_in",
               "mp_thompson_aerosols": "mp_thompson_aers_in", "mp_wsm6": "mp_wsm6_in",
               "cu_grell_freitas": "cu_grell_freitas_in", "cu_kain_fritsch": "cu_kain_fritsch_in",
               "cu_tiedtke": "cu_ntiedtke_in", "cu_ntiedtke": "cu_ntiedtke_in",
               "bl_mynn": "bl_mynn_in", "bl_ysu": "bl_ysu_in", "sf_noahmp": "sf_noahmp_in"}
    module = getattr(config,'driver_module',None)
    if module is None:
        for value in o.values():
            if isinstance(value, str) and value in mapping:
                packages.add(mapping[value])
    else:
        from importlib import import_module
        packages.update(name for name,enabled in import_module(module).package_selection(config).items()
                        if enabled)
    for flag, package in (("config_apply_lbcs", "limited_area"),
                          ("config_jedi_da", "jedi_da")):
        if configs.get(flag, False):
            packages.add(package)
    # mpas_atm_core_interface.F:143-155 uses the character option, not a bool.
    if configs.get("config_IAU_option", "off").strip() != "off":
        packages.add("iau")
    if configs.get("config_les_model", "none") != "none":
        packages.add("les")
    if o["config_sfclayer_scheme"] != "off":
        packages.add("sfclayer")
    if o["config_gwdo_scheme"] == "bl_ugwp_gwdo":
        packages.add("ugwp_orog_stream")
    # mpas_atm_core_interface.F:249-253 selects this output package from
    # its own flag, independently of either gravity-wave scheme.
    if configs.get("config_ugwp_diags", False):
        packages.add("ugwp_diags_stream")
    if o["config_ngw_scheme"]:
        packages.add("ugwp_ngw_stream")
    return packages


class RegistryLayout:
    """Complete active field metadata; Registry.xml:1360-4020."""

    # MPAS field metadata allocation: Registry.xml:1360-4020.
    def __init__(self, source: Path, config: DriverConfig, namelist: str,
                 dimensions: Mapping[str, int]):
        self.source = source
        self.root = ET.fromstring(registry_text(source / "src/core_atmosphere/Registry.xml"))
        prepare = getattr(config, 'native_registry_prepare', None)
        if prepare is not None:
            from importlib import import_module
            module, operation = prepare.split(':', 1)
            getattr(import_module(module), operation)(self.root)
        self.configs = {}
        self.config_kinds = {}
        for option in self.root.findall(".//nml_option"):
            kind, value = option.get("type"), option.get("default_value")
            if value is None:
                continue
            key = option.get("name")
            self.config_kinds[key] = kind
            self.configs[key] = (value if kind == "character" else
                                value.lower() == "true" if kind == "logical" else
                                int(value) if kind == "integer" else float(value.replace('D', 'e')))
        # Fortran names ignore case; retain Registry spelling for pool lookup.
        # Registry.xml:424 and mpas_atm_core_interface.F:143-155.
        canonical = {key.casefold(): key for key in self.config_kinds}
        for group, options in parse_fortran_namelist(namelist).items():
            if group != "hex_mpasphys":
                self.configs.update({canonical.get(key.casefold(), key): value
                                     for key, value in options.items()})
        self.configs.update({canonical.get(key.casefold(), key): value
                             for key, value in config.execution_options.items()})
        for key, kind in self.config_kinds.items():
            value = self.configs[key]
            if kind == "real":
                if type(value) not in (int, float):
                    raise ValueError(f"{key} requires Registry real type")
                self.configs[key] = float(value)
            elif kind == "integer" and type(value) is not int:
                raise ValueError(f"{key} requires Registry integer type")
            elif kind == "logical" and not isinstance(value, bool):
                raise ValueError(f"{key} requires Registry logical type")
            elif kind == "character" and not isinstance(value, str):
                raise ValueError(f"{key} requires Registry character type")
        self.packages = active_packages(config, self.configs)
        self.dimensions = dict(dimensions)
        pending = {d.get("name"): d.get("definition") for d in self.root.findall("./dims/dim")
                   if d.get("name") not in self.dimensions and d.get("definition")}
        while pending:
            progressed = False
            for name, definition in list(pending.items()):
                if definition.startswith("namelist:"):
                    value = self.configs[definition.split(":")[1]]
                elif definition.isdigit():
                    value = int(definition)
                elif re.fullmatch(r"\w+(?:\+\w+)+", definition):
                    terms = definition.split("+")
                    if any(not t.isdigit() and t not in self.dimensions for t in terms):
                        continue
                    value = sum(int(t) if t.isdigit() else self.dimensions[t] for t in terms)
                else:
                    raise ValueError(f"unsupported native dimension definition {definition}")
                self.dimensions[name] = int(value)
                del pending[name]
                progressed = True
            if not progressed:
                raise ValueError(f"unresolved Registry dimensions {pending}")
        for entity in _ENTITY:
            self.dimensions[entity + "Solve"] = self.dimensions[entity]
            self.dimensions[entity + "All"] = self.dimensions[entity]
        self.fields = []
        self.indices = {}
        self.pools = {s.get("name_in_code", s.get("name")) for s in self.root.findall("./var_struct")}
        for struct in self.root.findall("./var_struct"):
            pool = struct.get("name")
            if not self.enabled(struct):
                continue
            for field in struct:
                if field.tag not in ("var", "var_array") or not self.enabled(field):
                    continue
                row = dict(field.attrib)
                row["pool"] = pool
                row["pool_name"] = field.get("name_in_code", row["name"])
                row["kind"] = {"text": "character"}.get(row["type"], row["type"])
                row["nlevels"] = int(field.get("time_levs", struct.get("time_levs", "1")))
                dims = field.get("dimensions", "").split()
                dims = [d for d in dims if d != "Time"]
                components = []
                if field.tag == "var_array":
                    all_components = list(field.findall("var"))
                    selected = [c for c in all_components if self.enabled(c)]
                    components = [c.get("name") for c in selected]
                    count_name = "num_" + row["name"]
                    self.dimensions[count_name] = len(selected)
                    for c in all_components:
                        name = c.get("name")
                        code_name = c.get("name_in_code", name)
                        self.indices[pool, "index_" + code_name] = (components.index(name) + 1
                                                              if name in components else -1)
                    groups = {c.get("array_group") for c in all_components if c.get("array_group")}
                    for group in groups:
                        positions = [i + 1 for i, c in enumerate(selected) if c.get("array_group") == group]
                        self.indices[pool, "num_" + group] = len(positions)
                        self.indices[pool, group + "_start"] = min(positions, default=-1)
                        self.indices[pool, group + "_end"] = max(positions, default=-1)
                    if not selected:
                        continue
                    dims.insert(0, count_name)
                if row["kind"] == "character" and dims:
                    raise ValueError(f"native ABI lacks character array {pool}.{row['name']}")
                row.update(dimensions=dims, components=components)
                row["shape"] = [self.dimensions[d] + int(d in _ENTITY) for d in dims]
                # gen_inc.c:1286-1292,1709-1715 uses missing_value as the
                # allocation default when no default_value is specified.
                row["initial"] = row.get("default_value", row.get("missing_value", ""))
                row["required"] = False
                self.fields.append(row)

    def enabled(self, element: ET.Element) -> bool:
        """Package OR selection; Registry.xml:1705-1747."""
        packages = element.get("packages", "")
        return not packages or bool(set(packages.split(';')) & self.packages)

    def request(self, sources: list[Path], *, initial_state=None) -> dict[str, Any]:
        """Rust field transport request; mpas_io_streams.F:925-1023."""
        definitions = {s.get("name"): s for s in self.root.findall("./streams/stream")}
        readable = set()
        # Mutable stream declarations own their packages and members. A deck
        # can supply da_state without the Registry's default jedi_da package.
        # mpas_atm_core.F:139-150 reads that selected stream unconditionally.
        if initial_state is not None and initial_state[1] is not None:
            readable.update(initial_state[1])
        # MPAS input stream package expansion: Registry.xml:498-1351.
        def include(stream, seen):
            for item in stream:
                if not self.enabled(item):
                    continue
                name = item.get("name")
                if item.tag in ("var", "var_array"):
                    readable.add(name)
                elif item.tag == "var_struct":
                    readable.update(f["name"] for f in self.fields if f["pool"] == name)
                elif item.tag == "stream":
                    if name in seen:
                        raise ValueError(f"cyclic Registry stream {name}")
                    include(definitions[name], seen | {name})
        for name, stream in definitions.items():
            if "input" in stream.get("type", "").split(';') and self.enabled(stream):
                if initial_state is not None and name in ('input', 'restart', 'da_state'):
                    selected, members = initial_state
                    if name == selected:
                        if members is None:
                            include(stream, {name})
                        else:
                            readable.update(members)
                    continue
                if name == "restart" and not self.configs["config_do_restart"]:
                    continue
                # The fork's da_state stream (MPAS-Model-rrfs Registry.xml:1312-1618) is the initial state only
                # under DA cycling (mpas_atm_core.F:138-145 at 1d6e370e). On a cold start its initial_only input
                # re-reads, at the first timestep, the mpasout file the run itself wrote at start-up, so it adds
                # no value native did not already hold. Hydrating its members from the input files filled
                # fields native never reads at start-up (ter: native writes zeros, hex wrote the input terrain).
                if name == "da_state" and not (self.configs.get("config_do_DAcycling", False)
                                                 or self.configs["config_do_restart"]):
                    continue
                include(stream, {name})
        readable.update(getattr(self, "land_state_fields", ()))
        return {"schema": "mpas-native-pool-request-v1",
                "sources": [str(p.resolve()) for p in sources],
                "fields": [{**{k: f[k] for k in ("pool", "name", "kind", "dimensions", "shape",
                                                "components", "initial", "required")},
                            "read_input": f["name"] in readable} for f in self.fields]}


class NativePoolHydration:
    """Hydrate actual f3 pools with Rust-owned payloads; mpas_pool_routines.F:2110-3143."""

    # MPAS pool allocation: mpas_pool_routines.F:2110-3143.
    def __init__(self, native: Any, layout: RegistryLayout, payload: Path):
        self.native, self.layout, self.payload = native, layout, payload
        import sys
        resident = sys.modules.get('woof.hex.mpasphys.moving_pool_payload_v842')
        self.resident_payload = None if resident is None else resident.lookup(payload)
        manifest = (self.resident_payload.manifest if self.resident_payload is not None
                    else json.loads((payload / "manifest.json").read_text()))
        if manifest.get("schema") != "mpas-native-pool-payload-v1":
            raise ValueError("unsupported native pool transport")
        expected = {(r["pool"], r["name"]): r for r in layout.fields}
        actual = {(r["pool"], r["name"]): r for r in manifest["fields"]}
        if set(actual) != set(expected) or len(actual) != len(manifest["fields"]):
            raise ValueError("Rust payload fields differ from selected Registry")
        self.records = actual
        # THE COST THIS REMOVES (global 30 km, x1.655362, convection_permitting, 6000 box, 2026-10-16): every
        # payload file was read whole and hashed on the one Python thread, about 90 s for 51 GB before the
        # first step (4x that at 15 km) while 200+ cores sat idle.  The files are now hashed in streamed
        # pieces on worker threads (hashlib releases the GIL); the checks and their order are unchanged, so
        # the first failing field is still the one reported.
        import os
        from concurrent.futures import ThreadPoolExecutor

        def digest(path):
            # A file this process already hashed is not hashed again while its stat identity (device,
            # inode, size, mtime, ctime) is unchanged.  THE COST THIS REMOVES: one forecast builds this
            # hydration two to three times over the same payload (runtime_v842 load_native_dynamics_inputs,
            # the LES branch, session_v842), and each rehashed all 51 GB at 30 km.  Any write to a file
            # changes its mtime/ctime, so a changed file is hashed again and refused exactly as before.
            info = os.stat(path)
            identity = (os.path.realpath(path), info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns)
            with _VERIFIED_LOCK:
                known = _VERIFIED_DIGESTS.get(identity)
            if known is not None:
                return known
            with open(path, "rb") as handle:
                value = hashlib.file_digest(handle, "sha256").hexdigest()
            with _VERIFIED_LOCK:
                _VERIFIED_DIGESTS[identity] = value
            return value

        # The FIRST Registry-order refusal is the one reported, whichever check it is: a metadata or fill
        # refusal stops the walk, and only the fields before it are hashed, so an earlier digest failure still
        # wins over a later metadata one (tests/test_mpasphys_pool_v842.py, parallel validation).
        checked = []
        failure = None
        for key, row in expected.items():
            record = actual[key]
            failure = next((f"Rust metadata differs at {key}.{name}" for name in ("kind", "dimensions", "shape")
                            if record[name] != row[name]), None)
            if failure is None and row["kind"] != "character" and record.get("fill") is None:
                if self.resident_payload is not None:
                    # The source transport was validated before registration;
                    # field_buffer makes a private, shape-checked projection.
                    continue
                data = payload / record["payload"]
                if data.parent.resolve() != payload.resolve():
                    failure = "native payload escapes transport directory"
                else:
                    checked.append((key, row, record, data))
            elif failure is None and row["kind"] != "character" and record.get("fill") != 0:
                failure = f"native pool fill record at {key} is not the zero word"
            if failure is not None:
                break
        workers = max(1, min(32, host_copy_threads(), len(checked)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-pool-hash") as pool:
            digests = list(pool.map(digest, [data for _, _, _, data in checked]))
        for (key, row, record, data), value in zip(checked, digests):
            if value != record["sha256"]:
                raise ValueError(f"native payload digest differs at {key}")
            if data.stat().st_size != 4 * int(np.prod(row["shape"], dtype=np.int64)):
                raise ValueError(f"native payload byte count differs at {key}")
        if failure is not None:
            raise ValueError(failure)

    def field_buffer(self, pool: str, name: str):
        """Copy validated Rust words; mpas_io_streams.F:925-1023; Registry.xml:1701-1754."""
        record = self.records[pool, name]
        if record['kind'] == 'character':
            raise TypeError('numeric ABI buffer requested for a character field')
        dtype = '<f4' if record['kind'] == 'real' else '<i4'
        if record.get('fill') is not None:
            # A Registry-initial zero field (decode_registry_pools): the words the Rust payload carried.
            return np.zeros(record['shape'], dtype=dtype, order='F')
        if self.resident_payload is not None:
            values = self.resident_payload.field(pool, name)
            if values.shape != tuple(record['shape']) or values.dtype != np.dtype(dtype):
                raise ValueError(f'resident projected pool shape/dtype differs at {(pool, name)}')
            return values
        if getattr(self, 'copy_on_write', False) and all(record['shape']):
            # The transport remains immutable. The framework can adopt level
            # one's private mapping; writes never change the payload or a later
            # field_buffer read. Other time levels still own independent copies.
            return np.memmap(self.payload / record['payload'], dtype=dtype,
                             mode='c', shape=tuple(record['shape']), order='F')
        return np.fromfile(self.payload / record['payload'], dtype=dtype).reshape(
            record['shape'], order='F')

    def horizontal_length(self):
        """Resolve core's input/config assignment; mpas_atm_core.F:156-200."""
        nominal = self.field_buffer('mesh', 'nominalMinDc')
        configured = np.float32(self.layout.configs['config_len_disp'])
        if configured > np.float32(0):
            return np.asarray(configured)
        if nominal > np.float32(0):
            # Metadata selection and exact scalar transport, with no field maths.
            self.layout.configs['config_len_disp'] = float(nominal)
            return nominal
        raise ValueError('native core requires positive config_len_disp or nominalMinDc')

    def populate(self) -> None:
        """Allocate separate native time levels; mpas_pool_routines.F:2110-3143."""
        n, layout = self.native, self.layout
        for name, value in layout.configs.items():
            n.config(name, value)
        for pool in getattr(layout, "pools", {f["pool"] for f in layout.fields}) | {"mesh", "state"}:
            n.pool(pool)
            for name, value in layout.dimensions.items():
                n.dimension(pool, name, value)
        for (pool, name), value in layout.indices.items():
            n.dimension(pool, name, value)
        # THE COST THIS REMOVES (global 30 km, 588 payload files, 54.5 GB, 6000 box, 2026-10-16): every
        # file was read on the one Python thread between the serial native copies, about 0.6 s per GB.
        # The files are now read ahead on worker threads, at most _PREFETCH_BYTES in flight, and handed
        # to the native copies in the same Registry order with the same bytes; a read error surfaces at
        # the field it belongs to, as before.
        import os
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        deferred = (self.resident_payload is not None and callable(getattr(n, 'put_deferred_field', None))
                    and os.environ.get('WOOF_HEX_MOVING_DEFER_COLD_FIELDS', '1') != '0')
        def defer(row):
            return deferred and row['kind'] != 'character' and self.records[row['pool'], row['name']].get('fill') is None
        def zero(row):
            return (row['kind'] != 'character' and callable(getattr(n, 'put_zero_field', None))
                    and self.records[row['pool'], row['name']].get('fill') == 0)
        numeric = [row for row in layout.fields if row["kind"] != "character" and not defer(row) and not zero(row)]
        upcoming = iter(numeric)
        pending: deque = deque()
        in_flight = 0
        workers = max(1, min(16, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-pool-read") as reader:
            def fill():
                nonlocal in_flight
                while not pending or in_flight < _PREFETCH_BYTES:
                    row = next(upcoming, None)
                    if row is None:
                        return
                    size = 4 * int(np.prod(self.records[row["pool"], row["name"]]["shape"], dtype=np.int64))
                    pending.append(((row["pool"], row["name"]), size,
                                    reader.submit(self.field_buffer, row["pool"], row["name"])))
                    in_flight += size

            for row in layout.fields:
                pool, name = row["pool"], row["pool_name"]
                record = self.records[pool, row["name"]]
                if row["kind"] == "character":
                    for level in range(1, row["nlevels"] + 1):
                        n.put_text_field(pool, name, record["text"],
                                         level=level, nlevels=row["nlevels"])
                    continue
                if defer(row):
                    n.put_deferred_field(pool, name, record['shape'],
                        '<f4' if record['kind'] == 'real' else '<i4',
                        lambda row=row: self.field_buffer(row['pool'], row['name']), nlevels=row['nlevels'])
                    continue
                if zero(row):
                    # The validated transport declares the zero word. Hand its
                    # shape/dtype directly to native zero storage, rather than
                    # first materializing a second multi-GB zero buffer merely
                    # to inspect its metadata and discard it.
                    n.put_zero_field(pool, name, tuple(record['shape']),
                        np.dtype('<f4' if record['kind'] == 'real' else '<i4'), nlevels=row['nlevels'])
                    continue
                # Rust has already encoded Fortran ordering and native dummy storage.
                fill()
                key, size, future = pending.popleft()
                if key != (pool, row["name"]):
                    raise RuntimeError(f"pool read-ahead order differs at {key} vs {(pool, row['name'])}")
                buffer = future.result()
                in_flight -= size
                if record.get("fill") is not None and callable(getattr(n, "put_zero_field", None)):
                    # A Registry-initial zero field: the same words, allocated untouched (put_zero_field).
                    n.put_zero_field(pool, name, buffer.shape, buffer.dtype, nlevels=row["nlevels"])
                elif callable(getattr(n, "adopt_field", None)):
                    # The buffer was read for this call alone: the ported framework keeps it as level 1's
                    # storage instead of copying it into a fresh allocation on this thread (adopt_field).
                    n.adopt_field(pool, name, buffer, nlevels=row["nlevels"])
                else:
                    for level in range(1, row["nlevels"] + 1):
                        n.put_field(pool, name, buffer, level=level, nlevels=row["nlevels"])
                del buffer, future
        if ('mesh', 'nominalMinDc') in self.records:
            nominal = self.horizontal_length()
            n.config('config_len_disp', self.layout.configs['config_len_disp'])
            n.put_field('mesh', 'nominalMinDc', nominal)

    def register(self) -> None:
        """Native metadata and halo groups; mpas_dmpar.F:6853-7147."""
        n, layout = self.native, self.layout
        n.block()
        for entity, idname in (("nCells", "indexToCellID"), ("nEdges", "indexToEdgeID"),
                              ("nVertices", "indexToVertexID")):
            ids = n.get_field("mesh", idname)[:layout.dimensions[entity]]
            # A selected provider may retain native empty halo rings; RRFS fork
            # MPAS-Model-rrfs/src/framework/mpas_bootstrapping.F:144-156.
            ends = getattr(layout, 'halo_ends', {}).get(entity)
            options = {'halo_ends': ends} if ends is not None else {}
            n.decomposition(entity, global_ids=ids, owned=layout.dimensions[entity + "Solve"],
                            **options)
        for row in layout.fields:
            dims = row["dimensions"]
            parameters=inspect.signature(n.register_field).parameters
            if 'registry_name' not in parameters:
                raise RuntimeError('native field metadata ABI lacks distinct Registry/pool names; '
                                   'state.u and tend_u otherwise collide in allFields')
            n.register_field(row["pool"], row["pool_name"], dimensions=dims,
                             decomposed=bool(dims and dims[-1] in _ENTITY),
                             registry_name=row["name"],components=row["components"],
                             has_time='Time' in self.layout.root.find(
                                 f"./var_struct[@name='{row['pool']}']/*[@name='{row['name']}']"
                             ).get('dimensions','').split())
        for group, fields in (("physics:blten", ("rublten", "rvblten")),
                              ("physics:cuten", ("rucuten", "rvcuten"))):
            # Registry packages omit disabled tendencies; Registry.xml:3635-3697.
            active_fields = tuple(name for name in fields
                                  if any(row['name'] == name for row in layout.fields))
            if active_fields:
                n.exchange_group(group, active_fields)


#: Workers of the Rust pool transport; the reader takes at most this many (rw_mpas_physics_pool MAX_WORKERS).
POOL_TRANSPORT_MAX_THREADS = 16


def pool_transport_threads() -> int:
    """Workers rw_mpas_physics_pool runs: the CPUs this process may use, held to the CPU quota, at most 16.

    THE BREAKAGE THIS REMOVES (U-H3, hex 0.4.8, 30 km globe on 4x RTX PRO 6000, 2026-10-27): every
    MPAS-physics member start ran the transport on one thread for about 30 s while its card waited.
    ``RAYON_NUM_THREADS`` (hexcore sets it to the container quota) lowers the count; so does
    ``WOOF_HEX_HOST_COPY_THREADS``.  The count is handed to the reader explicitly, so the stage row
    and the process agree.
    """
    import os
    count = min(POOL_TRANSPORT_MAX_THREADS, host_copy_threads())
    text = os.environ.get("RAYON_NUM_THREADS", "").strip()
    if text.isdigit() and int(text) > 0:
        count = min(count, int(text))
    return max(1, count)


#: Reader processes one transport fans out to.  Each process opens the sources itself.
POOL_READER_MAX_PROCESSES = 8


def pool_reader_processes(workers: int, fields: int) -> int:
    """Reader processes for one transport: the worker count, at most POOL_READER_MAX_PROCESSES.

    ``WOOF_HEX_POOL_READER_PROCESSES`` sets the count; 1 runs the single reader exactly as before.
    """
    import os
    text = os.environ.get("WOOF_HEX_POOL_READER_PROCESSES", "").strip()
    count = int(text) if text.isdigit() and int(text) > 0 else min(POOL_READER_MAX_PROCESSES, workers)
    return max(1, min(count, fields))


def run_pool_reader(reader: Path, request: Path, output: Path, workers: int) -> None:
    """Run rw_mpas_physics_pool on ``request``, split over several reader processes.

    THE COST THIS REMOVES (V5090 box, 2026-10-27, x1.163842 GEFS init, 117 payload fields, 1.5 GB):
    the reader serialises every netCDF-C call inside one process behind a mutex (netCDF-C is not
    thread safe), so the decode took 7.4 s at 16 threads and 8.6 s at 1: the reads ran on one core.
    At 30 km (6 GB of payload) it was 39-52 s of every member start while the cards waited.
    Separate processes keep separate netCDF-C state, so their reads run side by side.

    Identity: every field is transported by the same binary from the same sources with the same
    request entry, and the reader's bytes per field do not depend on the other fields it carries
    (its own tests: one worker and many write the same bytes).  Each part's payload file is renamed
    to the name the single reader gives it (``field-<request index>.bin``) and the manifest lists
    the records in request order, as the single reader writes it.  If any part fails, the single
    reader runs the whole request, so a refusal is the one a single pass reports.
    """
    import os
    import shutil
    request = Path(request)
    output = Path(output)
    full = json.loads(request.read_text())
    fields = full["fields"]
    processes = pool_reader_processes(workers, len(fields))
    if processes <= 1:
        subprocess.run([str(reader), str(request), str(output)], check=True,
                       env={**os.environ, "RAYON_NUM_THREADS": str(workers)})
        return
    # Balance the parts by payload bytes, largest field first; each part keeps request order.
    sizes = [int(np.prod(f.get("shape") or [1], dtype=np.int64)) for f in fields]
    loads = [0] * processes
    parts: list[list[int]] = [[] for _ in range(processes)]
    for index in sorted(range(len(fields)), key=lambda i: (-sizes[i], i)):
        target = min(range(processes), key=lambda p: (loads[p], p))
        parts[target].append(index)
        loads[target] += sizes[index]
    parts = [sorted(part) for part in parts if part]
    output.mkdir(parents=True, exist_ok=True)
    each = max(1, -(-workers // len(parts)))
    directories = [output / f".reader-part-{k}" for k in range(len(parts))]
    launched = []
    try:
        for part, directory in zip(parts, directories):
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True)
            part_request = directory / "request.json"
            part_request.write_text(json.dumps({**full, "fields": [fields[i] for i in part]}))
            launched.append(subprocess.Popen(
                [str(reader), str(part_request), str(directory / "out")],
                env={**os.environ, "RAYON_NUM_THREADS": str(each)},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        failed = False
        for process in launched:
            failed = process.wait() != 0 or failed
        if failed:
            # The single reader reports the first refusal in request order, as before.
            for directory in directories:
                shutil.rmtree(directory, ignore_errors=True)
            subprocess.run([str(reader), str(request), str(output)], check=True,
                           env={**os.environ, "RAYON_NUM_THREADS": str(workers)})
            return
        records: list[Any] = [None] * len(fields)
        for part, directory in zip(parts, directories):
            manifest = json.loads((directory / "out" / "manifest.json").read_text())
            if manifest.get("schema") != "mpas-native-pool-payload-v1" or len(manifest["fields"]) != len(part):
                raise ValueError("a pool reader part returned an unexpected manifest")
            for index, record in zip(part, manifest["fields"]):
                if record.get("payload") is not None:
                    name = f"field-{index:04d}.bin"
                    os.replace(directory / "out" / record["payload"], output / name)
                    record["payload"] = name
                records[index] = record
        (output / "manifest.json").write_text(json.dumps(
            {"schema": "mpas-native-pool-payload-v1", "fields": records}, indent=2, ensure_ascii=False))
    finally:
        for process in launched:
            if process.poll() is None:
                process.kill()
                process.wait()
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)


def decode_registry_pools(reader: Path, source: Path, config: DriverConfig,
                          namelist: str, sources: list[Path], output: Path,
                          threads: int | None = None, initial_state=None, beside_transport=None) -> RegistryLayout:
    """Rust I/O followed by native allocation; mpas_io_streams.F:925-1023.

    ``beside_transport(layout)``, when given, is called with the finished layout just before the Rust transport
    starts and returns a callable that is called once the transport has ended (or failed).  It runs work that reads
    only the source files, never the payload, while the reader processes run (runtime_v842.start_regional_inputs).
    """
    import os
    abi = subprocess.check_output([str(reader), "--abi"], text=True).strip()
    if abi != "gpuwm-rw-mpas-physics-pool-v1":
        raise ValueError("unexpected Rust physics-pool transport ABI")
    dimensions = {}
    for path in sources:
        dimensions.update(json.loads(subprocess.check_output(
            [str(reader), "--inspect", str(path)], text=True)))
    layout = RegistryLayout(source, config, namelist, dimensions)
    from woof.hex.cycle.land_state import warm_land_fields, admit_warm_configuration
    layout.land_state_fields = warm_land_fields(
        sources, xice_threshold=.02 if layout.configs['config_frac_seaice'] else .5)
    if layout.land_state_fields:
        admit_warm_configuration(layout.configs)
    for row in layout.fields:
        if row["name"] in layout.land_state_fields:
            row["required"] = True
    output.mkdir(parents=True, exist_ok=True)
    selected = None
    if initial_state is not None:
        name, streams_path, list_directory = initial_state
        members = None
        if streams_path is not None:
            from .forecast_v842 import read_streams
            from ..physics_suites import load_table
            from importlib import import_module
            row = load_table()[getattr(config, 'table_key', 'mpasphys_v842')]
            parser = row.get('streams_parser')
            tree = None
            if parser:
                module, operation = parser.split(':', 1)
                tree = getattr(import_module(module), operation)(streams_path)
            streams = read_streams(streams_path, layout, list_directory, tree=tree)
            members = next(s.fields for s in streams if s.name == name)
        selected = name, members
    full = layout.request(sources, initial_state=selected)
    # THE COST THIS REMOVES (global 30 km, x1.655362, convection_permitting, 2026-10-16): of the 50.8 GiB the
    # Rust reader wrote as payload files, 44.8 GiB (473 of 591 fields: abstot, ozmixm, the diag and tend
    # arrays) were fields no input stream reads, every word of each the Registry initial value 0.0 (checked
    # word by word on that payload).  They were written, hashed, read back and copied on the host before the
    # first step, and at 15 km they did not fit the disk.  Such a field is now a fill record: native receives
    # the same zero words (field_buffer), with the origin the Rust reader records for it.  A field with a
    # non-zero Registry initial, a character field and every field an input stream reads still travel through
    # Rust exactly as before.
    fills = [f for f in full["fields"]
             if not f["read_input"] and f["kind"] != "character" and _zero_initial(f["initial"])]
    filled = {(f["pool"], f["name"]) for f in fills}
    request = output / "request.json"
    request.write_text(json.dumps({**full, "fields": [f for f in full["fields"]
                                                      if (f["pool"], f["name"]) not in filled]}, indent=2) + "\n")
    # Fields are transported on ``threads`` workers (U-H3); each payload is the same bytes at any count.
    workers = pool_transport_threads() if threads is None else max(1, int(threads))
    finish = beside_transport(layout) if beside_transport is not None else None
    try:
        run_pool_reader(reader, request, output, workers)
    finally:
        if finish is not None:
            finish()
    if fills:
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        records = {(r["pool"], r["name"]): r for r in manifest["fields"]}
        for f in fills:
            records[f["pool"], f["name"]] = {
                "pool": f["pool"], "name": f["name"], "kind": f["kind"], "dimensions": f["dimensions"],
                "shape": f["shape"],
                "origins": [f"registry-initial-value:{c}" for c in (f.get("components") or [f["name"]])],
                "payload": None,
                "sha256": None, "text": None, "fill": 0}
        manifest["fields"] = [records[f["pool"], f["name"]] for f in full["fields"]
                              if (f["pool"], f["name"]) in records]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return layout


def _zero_initial(value: Any) -> bool:
    """A Registry default_value that is the zero word (empty means zero, as the Rust reader writes it)."""
    text = str(value if value is not None else "").strip().lower().replace("d", "e")
    if text == "":
        return True
    try:
        number = float(text)
    except ValueError:
        return False
    return number == 0.0 and not text.startswith("-")
