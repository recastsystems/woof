"""Resident surface probabilities and Rust output for one ensemble frame."""
from __future__ import annotations

import hashlib
import json
from operator import index
from pathlib import Path
from types import SimpleNamespace
import time
import threading
import os
import sys
import tempfile
import numpy as np
from woof.certify.kernel_manifest import record_module
from woof.ensemble.member_weights import roster_weights, weight_receipt, unequal_weights

from woof.ensemble.batch_products import (FieldProducts, prepare_product_frame,
                                           write_product_frame, product_memory_plan,
                                           PRODUCT_CONTRACT, default_product_requests,
                                           DEFAULT_THRESHOLDS, ThresholdCondition,
                                           PreparedCompoundField, compound_memory_plan)
from woof.ensemble.physical_fields import sha256_of_file as _sha

REQUESTS = (FieldProducts("rain_total", "mm", (1.0, 5.0)),
            FieldProducts("wind10", "m s-1", (10.0, 15.0)),
            FieldProducts("temperature2", "K", (293.15, 303.15)))


def _member_order(members, values):
    order = tuple(range(members)) if values is None else tuple(index(value) for value in values)
    if (len(order) != members or len(set(order)) != members
            or any(value < 0 or value >= (1 << 64) for value in order)):
        raise ValueError("member order must retain distinct unsigned global IDs for the complete roster")
    return order

_SOURCE = r'''
extern "C" __global__ void ensemble_surface_products(
    const float *u, const float *v, const float *rainnc, const float *rainc,
    const float *rainsh, const float *initial_rain,
    float *wind, float *rain, unsigned long long count,
    int has_rainc, int has_rainsh, int initial) {
    unsigned long long point = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= count) return;
    wind[point] = __fsqrt_rn(__fadd_rn(__fmul_rn(u[point], u[point]), __fmul_rn(v[point], v[point])));
    float total = __fadd_rn(rainnc[point], has_rainc ? rainc[point] : 0.0f);
    total = __fadd_rn(total, has_rainsh ? rainsh[point] : 0.0f);
    rain[point] = initial ? total : __fsub_rn(total, initial_rain[point]);
}
'''


class PreparedEnsembleOutput:
    """All per-frame diagnostic storage is allocated before stepping.

    Supplied inputs are member-outermost contiguous float32 arrays, or
    equivalent packed column views reshaped by the caller without a copy.
    Forecast arrays are only read. The initial rain counter is retained so
    `rain_total` means precipitation since the forecast start, not source-cycle
    accumulation. Finished frame writes include only aggregate products.
    """

    def __init__(self, *, u10, v10, temperature2, rainnc, latitude, longitude,
                 members, available_bytes, rainc=None, rainsh=None,
                 requests=REQUESTS, array_module=None, member_weights=None):
        if array_module is None:
            import cupy as array_module
        self.xp = array_module
        self.members = members
        self.latitude, self.longitude = np.asarray(latitude), np.asarray(longitude)
        self.shape = (members,) + self.latitude.shape
        if self.latitude.ndim != 2 or self.longitude.shape != self.latitude.shape:
            raise ValueError("ensemble output needs matching two-dimensional static coordinates")
        self.inputs = dict(u10=u10, v10=v10, temperature2=temperature2,
                           rainnc=rainnc, rainc=rainc, rainsh=rainsh)
        for name, array in self.inputs.items():
            if array is not None and (array.shape != self.shape or array.dtype != np.dtype("float32") or not array.flags.c_contiguous):
                raise ValueError(f"{name} needs complete-roster contiguous float32 surface storage")
        diagnostic_bytes = 3 * ((int(np.prod(self.shape)) * 4 + 511) // 512) * 512
        if available_bytes < diagnostic_bytes:
            raise ValueError("ensemble diagnostic and initial-rain backings do not fit the supplied memory budget")
        self.wind = array_module.empty(self.shape, dtype=np.float32)
        self.rain = array_module.empty(self.shape, dtype=np.float32)
        self.initial_rain = array_module.empty(self.shape, dtype=np.float32)
        module = array_module.RawModule(code=_SOURCE, options=("--std=c++17",))
        self.kernel = module.get_function("ensemble_surface_products")
        record_module("woof.ensemble.batch_product_output:surface-products", source=_SOURCE,
                      options=("--std=c++17",), module=module)
        self.products = prepare_product_frame(
            dict(wind10=self.wind, rain_total=self.rain, temperature2=temperature2),
            requests, available_bytes=available_bytes - diagnostic_bytes,
            array_module=array_module, member_weights=member_weights)
        self.initialized = False
        self.frames = []

    def _diagnose(self, initial=False):
        count = int(np.prod(self.shape)); get = self.inputs.get
        # Unused precipitation pointers reuse an existing typed const input.
        self.kernel(((count + 255) // 256,), (256,),
                    (get("u10"), get("v10"), get("rainnc"),
                     get("rainc") if get("rainc") is not None else get("rainnc"),
                     get("rainsh") if get("rainsh") is not None else get("rainnc"),
                     self.initial_rain, self.wind, self.initial_rain if initial else self.rain,
                     np.uint64(count), np.int32(get("rainc") is not None),
                     np.int32(get("rainsh") is not None), np.int32(initial)))

    def capture_initial_rain(self):
        if self.initialized:
            raise ValueError("initial precipitation counter was already captured")
        self._diagnose(initial=True)
        self.initialized = True

    def __call__(self, *, path, valid_time, maps_directory, renderer, domain="d01", global_attrs=None):
        from woof import rustwx
        if not self.initialized:
            raise ValueError("forecast-start rain must be captured before an ensemble output")
        started = time.perf_counter(); self._diagnose(); self.products()
        self.xp.cuda.get_current_stream().synchronize()
        reduction_seconds = time.perf_counter() - started
        started = time.perf_counter()
        write_product_frame(path, self.products, valid_time=valid_time,
                            latitude=self.latitude, longitude=self.longitude,
                            global_attrs=global_attrs)
        write_seconds = time.perf_counter() - started
        started = time.perf_counter()
        written, transcript = rustwx.run_ensemble_product_renderer(
            renderer, path, out_dir=maps_directory, domain=domain,
            fields=tuple(request.field for request in self.products.requests))
        render_seconds = time.perf_counter() - started
        row = {"valid_time": str(valid_time), "product_path": str(path),
               "product_bytes": Path(path).stat().st_size,
               "reduction_seconds": reduction_seconds, "write_seconds": write_seconds,
               "render_seconds": render_seconds, "maps": [str(item) for item in written],
               "renderer_transcript": transcript}
        self.frames.append(row)
        return row

    def receipt(self, *, hash_artifacts=False):
        frames = [dict(row) for row in self.frames]
        if hash_artifacts:
            for row in frames:
                row["product_sha256"] = _sha(row["product_path"])
                row["map_sha256"] = {path: _sha(path) for path in row["maps"]}
        return {"contract": "gpuwm-ensemble-surface-output.v1",
                "member_history_count": 0, "probability_products": self.products.receipt(),
                "diagnostic_backings_bytes": sum(int(a.nbytes) for a in (self.wind, self.rain, self.initial_rain)),
                "diagnostic_pool_rounded_bytes": sum(((int(a.nbytes) + 511) // 512) * 512 for a in (self.wind, self.rain, self.initial_rain)),
                "rain_scope": "since forecast start, complete member roster",
                "frames": frames}


def replay_memory_plan_for_shape(requests, shape, *, members, tile_rows=32, member_weights=None):
    """Price complete-roster replay without files, a spool or a CUDA context."""
    from dataclasses import replace
    from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
    shape = tuple(index(size) for size in shape)
    members, tile_rows = index(members), index(tile_rows)
    if len(shape) != 2 or min(shape) < 1 or members < 1 or tile_rows < 1:
        raise ValueError("product replay needs positive member count, grid extents and tile rows")
    rows = min(tile_rows, shape[0])
    plans = []
    for request in requests:
        tile_request = replace(request, postage_stamp=False)
        plan = product_memory_plan((tile_request,), {request.field: (rows, shape[1])}, members=members, member_weights=member_weights)
        plans.append(BatchMemoryPlan((BatchArraySpec(f"{request.field}:replay", (rows, shape[1]), "member"),)
                                     + plan.arrays, reserved_bytes=0))
    if not plans:
        raise ValueError("product replay needs at least one requested field")
    return max(plans, key=lambda plan: plan.required_bytes(members))


class NativeDiagnosticSpool:
    """Bounded 2-D member replay across cards, packs and ordinary fallback.

    The spool contains requested diagnostics, never prognostic histories.
    Completed frames reduce in global member order on the GPU, with the same
    two-pass arithmetic as the all-resident reducer. A partial roster has no
    published probability. Only files created by this object are removed.
    """

    CONTRACT = "gpuwm-ensemble-output.v2"

    def __init__(self, root, *, members, requests, latitude, longitude,
                 renderer, domain="d01", keep_member_files=False,
                 unavailable_products=(), tile_rows=32, events=None,
                 member_order=None, member_metadata=(), export_coordinates=False,
                 gpu_replay=False, resume=False):
        from woof.ensemble.batch_products import _positive_int
        self.root = Path(root)
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.members = _positive_int(members, "members")
        self.member_order = _member_order(self.members, member_order)
        self.member_positions = {member: position for position, member in enumerate(self.member_order)}
        self.member_metadata = tuple(member_metadata)
        self.member_weights = roster_weights(self.member_order, self.member_metadata)
        self.gpu_replay = bool(gpu_replay)
        self.requests = tuple(requests)
        self.latitude, self.longitude = np.asarray(latitude), np.asarray(longitude)
        self.shape = self.latitude.shape
        if len(self.shape) != 2 or self.longitude.shape != self.shape or min(self.shape) < 1:
            raise ValueError("ensemble diagnostics need matching two-dimensional static coordinates")
        self.renderer, self.domain = Path(renderer), str(domain)
        if not self.domain or Path(self.domain).name != self.domain:
            raise ValueError("ensemble domain must be one safe path component")
        if not self.requests or len({r.field for r in self.requests}) != len(self.requests):
            raise ValueError("ensemble spool needs unique FieldProducts requests")
        if any(r.spaghetti for r in self.requests) and min(self.shape) < 2:
            raise ValueError("spaghetti diagnostics need at least two points per grid axis")
        self.tile_rows = max(2, _positive_int(tile_rows, "tile rows"))
        self.keep_member_files = bool(keep_member_files)
        self.unavailable_products = tuple(unavailable_products)
        self.events = dict(events or {})
        self.frames = {}
        self.deleted = []
        self.member_files = []
        self._processing = set()
        self.coordinate_file = None
        self._publish_coordinates = bool(export_coordinates)
        (self.root / self.domain).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / self.domain / "ensemble-manifest.json"
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError("ensemble manifest already exists; use a new run output directory")
            record = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (record.get("schema") != self.CONTRACT or record.get("member_order") != list(self.member_order)
                    or record.get("member_metadata") != list(self.member_metadata)
                    or record.get("products") != [request.describe() for request in self.requests]):
                raise ValueError("retained ensemble spool belongs to another member or product roster")
            self.deleted, self.member_files = record["deleted_scratch"], record["member_files"]
            from woof.ensemble.product_worker import _identity, _owned_path
            if record.get("coordinates_file") is not None:
                self.coordinate_file = dict(record["coordinates_file"])
                _identity(_owned_path(self.root, self.coordinate_file["path"]), expected=self.coordinate_file)
            for row in record["frames"]:
                row = dict(row)
                packs = []
                for pack in row.pop("diagnostic_files", ()):
                    if not {"path", "member_ids", "sha256", "bytes"} <= set(pack):
                        raise ValueError("retained ensemble diagnostic pack lacks its committed identity")
                    path = _owned_path(self.root, pack["path"])
                    _identity(path, expected=pack)
                    packs.append({**pack, "path": path, "member_ids": tuple(pack["member_ids"])})
                received = [member for pack in packs for member in pack["member_ids"]]
                if (row.get("domain") != self.domain or row.get("members_expected") != self.members
                        or any(type(member) is not int or member not in self.member_positions for member in received)
                        or len(set(received)) != len(received)
                        or (row["status"] == "pending" and sorted(received) != row["members_received"])):
                    raise ValueError("retained ensemble diagnostic packs changed their member or domain roster")
                row["packs"] = packs
                self.frames[row["valid_time"]] = row
            return
        if export_coordinates:
            self._ensure_coordinate_file()
        self._manifest()

    def _ensure_coordinate_file(self):
        """Persist static coordinates once without expanding a process request."""
        with self._lock:
            if self.coordinate_file is None:
                from woof.io.classic_product import ClassicProduct
                directory = self.root / ".ensemble-diagnostics" / self.domain
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "coordinates.nc"
                with ClassicProduct(path) as writer:
                    writer.createDimension("south_north", self.shape[0])
                    writer.createDimension("west_east", self.shape[1])
                    for name, values in (("XLAT", self.latitude), ("XLONG", self.longitude)):
                        writer.createVariable(name, "float32", ("south_north", "west_east"))[:] = values
                self.coordinate_file = {"path": self._relative(path), "sha256": _sha(path),
                    "bytes": path.stat().st_size}
            return dict(self.coordinate_file)

    def _relative(self, path):
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def _manifest(self):
        with self._lock:
            return self._manifest_locked()

    def _manifest_locked(self):
        frames = []
        for frame in self.frames.values():
            row = {k: v for k, v in frame.items() if k != "packs"}
            row.update(weight_receipt(self.member_order, self.member_weights, frame["members_received"]))
            if frame["status"] == "pending":
                row["diagnostic_files"] = [{"path": self._relative(pack["path"]),
                    "member_ids": list(pack["member_ids"]),
                    **{key: pack[key] for key in ("sha256", "bytes") if key in pack}}
                    for pack in frame["packs"]]
            frames.append(row)
        from woof.ensemble.ice_accretion import CONTRACT as ICE_CONTRACT, FIELDS as ICE_FIELDS
        record = {"schema": self.CONTRACT, "members_requested": self.members,
                  **weight_receipt(self.member_order, self.member_weights),
                  "gridded_aggregate_policy": "unavailable when member coordinates differ; member products retain native coordinates",
                  "member_order": list(self.member_order),
                  "member_metadata": list(self.member_metadata),
                  "probability_denominator": ("sum of present roster masses" if unequal_weights(self.member_weights) else self.members), "spread_ddof": 1,
                  "nonfinite_policy": "mask aggregate; retain finite_count",
                  "keep_member_files": self.keep_member_files,
                  "member_files": list(self.member_files),
                  "products": [r.describe() for r in self.requests],
                  "events": self.events,
                  "unavailable_products": list(self.unavailable_products),
                  "frames": frames,
                  "deleted_scratch": list(self.deleted)}
        if any(r.field in (*ICE_FIELDS, "freezing_rain_rate") for r in self.requests):
            record["ice_accretion_sampling"] = ICE_CONTRACT
        if self.coordinate_file is not None and self._publish_coordinates:
            record["coordinates_file"] = self.coordinate_file
        temporary = self.manifest_path.with_suffix(".json.pending")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)
        return record

    def add_member_file(self, path, *, member_id, domain=None, identity=None):
        """Register an explicitly retained ordinary member history."""
        with self._lock:
            return self._add_member_file(path, member_id=member_id, domain=domain, identity=identity)

    def _add_member_file(self, path, *, member_id, domain=None, identity=None):
        if not self.keep_member_files:
            raise ValueError("member histories require keep_member_files")
        member = index(member_id)
        if member not in self.member_positions:
            raise ValueError("member history index is outside the ensemble roster")
        path = Path(path)
        if identity is None and not path.is_file():
            raise ValueError("retained member history must exist before registration")
        record = {"member_id": member, "domain": domain or self.domain,
            "path": self._relative(path), "bytes": path.stat().st_size if identity is None else identity["bytes"]}
        previous = next((row for row in self.member_files if row["path"] == record["path"]), None)
        if previous is not None:
            if previous != record:
                raise ValueError("member history path changed its recorded identity")
            return previous
        self.member_files.append(record)
        self.member_files.sort(key=lambda row: (self.member_positions[row["member_id"]], row["path"]))
        self._manifest()

    def submit(self, fields, *, member_ids, valid_time, unavailable_fields=(), array_module=None):
        """Persist a resident diagnostic pack, publishing only roster progress."""
        with self._lock:
            return self._submit(fields, member_ids=member_ids, valid_time=valid_time,
                                unavailable_fields=unavailable_fields, array_module=array_module)

    def _submit(self, fields, *, member_ids, valid_time, unavailable_fields=(), array_module=None):
        from woof.io.classic_product import ClassicProduct
        if array_module is None:
            import cupy as array_module
        xp = array_module
        ids = tuple(index(member) for member in member_ids)
        if not ids or len(set(ids)) != len(ids) or any(member not in self.member_positions for member in ids):
            raise ValueError("diagnostic pack member indices must be unique and within the complete roster")
        names = {request.field for request in self.requests}
        unavailable = set(unavailable_fields)
        if set(fields) | unavailable != names or set(fields) & unavailable:
            raise ValueError("diagnostic fields and explicit unavailable fields must partition the requested output contract")
        host = xp is np
        device = None if host else int(xp.cuda.runtime.getDevice())
        for name, array in fields.items():
            if (not isinstance(array, xp.ndarray) or array.dtype != np.dtype("float32")
                    or array.shape != (len(ids),) + self.shape or not array.flags.c_contiguous
                    or (not host and int(array.device.id) != device)):
                raise ValueError(f"{name} needs resident contiguous float32 (pack_member,y,x) diagnostics")
        valid = str(valid_time)
        frame = self.frames.setdefault(valid, {"valid_time": valid, "domain": self.domain,
            "status": "pending", "members_received": [], "members_expected": self.members,
            "packs": [], "products": [], "maps": [], "available_fields": sorted(fields),
            "unavailable_fields": sorted(unavailable)})
        if frame["available_fields"] != sorted(fields) or frame["unavailable_fields"] != sorted(unavailable):
            # Products are drawn only from fields every member of this frame supplied.
            common = sorted(set(frame["available_fields"]) & set(fields))
            dropped = sorted(set(frame["available_fields"]) - set(common))
            frame["available_fields"] = common
            frame["unavailable_fields"] = sorted(names - set(common))
            if dropped:
                frame.setdefault("availability_reductions", []).append(
                    {"members": list(ids), "dropped": dropped,
                     "reason": "a member lacked these diagnostics at this clock (for example a moving nest's new placement)"})
        if frame["status"] != "pending" or set(ids) & set(frame["members_received"]):
            raise ValueError("diagnostic member/time was already submitted; a duplicate would bias ensemble products")
        slug = hashlib.sha256(valid.encode("utf-8")).hexdigest()[:24]
        directory = self.root / ".ensemble-diagnostics" / self.domain / slug
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"pack-{len(frame['packs']):06d}.nc"
        if path.exists():
            raise FileExistsError("diagnostic scratch path already exists; an unowned file cannot be replaced")
        pending = path.with_suffix(".nc.pending")
        if pending.exists():
            raise FileExistsError("diagnostic pending path already exists; an unowned file cannot be replaced")
        with ClassicProduct(pending) as writer:
            writer.createDimension("pack_member", len(ids))
            writer.createDimension("south_north", self.shape[0])
            writer.createDimension("west_east", self.shape[1])
            writer.setncattr("ensemble_diagnostic_contract", "gpuwm-ensemble-diagnostic-spool.v1")
            writer.setncattr("valid_time", valid)
            writer.setncattr("ensemble_members", self.members)
            writer.createVariable("member_ids", "uint64", ("pack_member",))[:] = np.asarray(ids, np.uint64)
            for request in self.requests:
                if request.field not in fields:
                    continue
                variable = writer.createVariable(request.field, "float32", ("pack_member", "south_north", "west_east"))
                variable.units = request.units
                variable[:] = fields[request.field] if host else xp.asnumpy(fields[request.field])
        pending.replace(path)
        frame["packs"].append({"path": path, "member_ids": ids,
            "sha256": _sha(path), "bytes": path.stat().st_size})
        frame["members_received"] = sorted(frame["members_received"] + list(ids))
        self._manifest()
        return len(frame["members_received"]) == self.members

    def adopt_committed_pack(self, path, *, member_ids, valid_time, available_fields,
                            unavailable_fields=(), sha256, bytes=None):
        """Adopt an immutable native spill from an independent member process."""
        import shutil
        ids = tuple(index(member) for member in member_ids)
        fields, missing = sorted(available_fields), sorted(unavailable_fields)
        requested = {request.field for request in self.requests}
        if (not ids or len(set(ids)) != len(ids) or any(member not in self.member_positions for member in ids)
                or set(fields) | set(missing) != requested or set(fields) & set(missing)):
            raise ValueError("adopted native diagnostic pack has an invalid roster or field contract")
        source = Path(path)
        if not source.is_file() or source.is_symlink() or _sha(source) != sha256:
            raise ValueError("adopted diagnostic pack does not match its committed SHA-256")
        if bytes is not None and source.stat().st_size != bytes:
            raise ValueError("adopted diagnostic pack changed its committed byte size")
        valid = str(valid_time)
        from woof import netcdf_bridge
        with netcdf_bridge.Dataset(source) as reader:
            observed_ids = tuple(int(member) for member in np.asarray(reader.variables["member_ids"][:]).ravel())
            if observed_ids != ids or reader.getncattr("valid_time") != valid:
                raise ValueError("adopted diagnostic words do not match their declared member/time identity")
            for field in fields:
                if reader.variables[field].shape != (len(ids),) + self.shape:
                    raise ValueError("adopted diagnostic words do not match the receiving coordinate grid")
        with self._lock:
            frame = self.frames.setdefault(valid, {"valid_time": valid, "domain": self.domain,
                "status": "pending", "members_received": [], "members_expected": self.members,
                "packs": [], "products": [], "maps": [], "available_fields": fields,
                "unavailable_fields": missing})
            if frame["available_fields"] != fields or frame["unavailable_fields"] != missing:
                raise ValueError("adopted member packs disagree on diagnostic availability")
            if frame["status"] != "pending" or set(ids) & set(frame["members_received"]):
                raise ValueError("adopted diagnostic member/time was already submitted")
            slug = hashlib.sha256(valid.encode("utf-8")).hexdigest()[:24]
            directory = self.root / ".ensemble-diagnostics" / self.domain / slug
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"pack-{len(frame['packs']):06d}.nc"
            pending = destination.with_suffix(".nc.pending")
            if destination.exists() or pending.exists():
                raise FileExistsError("adopted diagnostic destination is already owned")
            shutil.copyfile(source, pending)
            if _sha(pending) != sha256:
                raise ValueError("adopted native diagnostic copy changed its committed SHA-256")
            pending.replace(destination)
            frame["packs"].append({"path": destination, "member_ids": ids,
                "sha256": sha256, "bytes": destination.stat().st_size})
            frame["members_received"] = sorted(frame["members_received"] + list(ids))
            self._manifest()
            return len(frame["members_received"]) == self.members

    def replay_memory_plan(self):
        """Exact maximum replay tile plus GPU outputs; host caches are separate."""
        return replay_memory_plan_for_shape(self.requests, self.shape,
                                            members=self.members, tile_rows=self.tile_rows, member_weights=self.member_weights)

    def retire_private_coordinates(self):
        """Remove the worker-only coordinate input once no hour can replay.

        ``finish_in_subprocess`` writes coordinates.nc as private input for
        the isolated Rust worker. It is published only when the spool was
        built with export_coordinates (deferred replay hands it to a later
        consumer). Otherwise it is owned scratch, and leaving it behind put a
        diagnostic NetCDF in every finished output tree. A pending hour still
        needs it, so it stays until every frame is settled.
        """
        with self._lock:
            if (self.coordinate_file is None or self._publish_coordinates
                    or self._processing
                    or any(frame["status"] == "pending" for frame in self.frames.values())):
                return None
            path = self.root / self.coordinate_file["path"]
            record = None
            if path.is_file():
                record = {"path": self._relative(path), "bytes": path.stat().st_size}
                path.unlink()
                self.deleted.append(record)
            self.coordinate_file = None
            self._manifest()
            return record

    def retire_pending_frame(self, valid_time, *, reason, status="incomplete"):
        """Finish roster accounting without publishing a partial probability."""
        if status not in ("incomplete", "unavailable"):
            raise ValueError("retired diagnostic frame status must be incomplete or unavailable")
        with self._lock:
            frame = self.frames[str(valid_time)]
            if frame["status"] == "complete":
                return
            frame.update(status=status, reason=str(reason), products=[], maps=[],
                product_availability=[dict(field=r.field, product=kind, status=status, reason=str(reason))
                    for r in self.requests for kind in ("mean", "spread", "min", "max", "finite_count",
                        *(("probability",) if r.thresholds else ()),
                        *(("pmm",) if r.field in ("refl", "rain_total") else ()),
                        *(("paintball",) if r.paintball else ()),
                        *(("spaghetti",) if r.spaghetti else ()))])
            self._manifest()
            for pack in frame["packs"]:
                path = pack["path"]
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    self.deleted.append({"path": self._relative(path), "bytes": size})
            self._manifest()

    def finish(self, valid_time, *, available_bytes, array_module=None,
               render_products=("mean", "spread", "min", "max", "prob", "paintball", "postage", "pmm", "spaghetti")):
        """Reduce one complete-roster frame and publish its finished Rust maps."""
        valid = str(valid_time)
        with self._lock:
            if valid in self._processing:
                raise ValueError("ensemble frame already has an active replay")
            self._processing.add(valid)
        try:
            # The durable pack roster is immutable after completion. Replay
            # borrows it without holding the manifest lock over CUDA work or
            # Rust rendering, so the next hour can commit concurrently.
            return self._finish(valid, available_bytes=available_bytes,
                                array_module=array_module, render_products=render_products)
        finally:
            with self._lock:
                self._processing.remove(valid)

    def _finish(self, valid_time, *, available_bytes, array_module=None,
                render_products=("mean", "spread", "min", "max", "prob", "paintball", "postage", "pmm", "spaghetti"),
                retire_diagnostics=True):
        from dataclasses import replace
        from woof import rustwx
        if array_module is None and self.gpu_replay:
            import cupy as array_module
        xp = array_module
        valid = str(valid_time)
        frame = self.frames[valid]
        if frame["status"] != "pending" or frame["members_received"] != sorted(self.member_order):
            raise ValueError("probability frame requires exactly the complete requested member roster")
        if self.gpu_replay:
            self.replay_memory_plan().admit(self.members, available_bytes=available_bytes)
        ny, nx = self.shape
        requests = tuple(r for r in self.requests if r.field in frame["available_fields"])
        if not requests:
            frame.update(status="complete", products=[], maps=[])
            self._manifest()
            for pack in frame["packs"] if retire_diagnostics else ():
                path = pack["path"]
                size = path.stat().st_size
                path.unlink()
                self.deleted.append({"path": self._relative(path), "bytes": size})
            self._manifest()
            return {k: v for k, v in frame.items() if k != "packs"}
        started = time.perf_counter()
        plan = product_memory_plan(requests, {r.field: self.shape for r in requests},
                                   members=self.members, include_pmm=True)
        if self.gpu_replay:
            outputs = self._gpu_reference_replay(frame, requests, plan,
                available_bytes=available_bytes, array_module=xp)
        else:
            outputs, retired = rustwx.reduce_ensemble_diagnostics(self.renderer,
                packs=frame["packs"], member_order=self.member_order, shape=self.shape,
                valid_time=valid, requests=requests, inventory=plan.inventory(self.members),
                scratch_directory=frame["packs"][0]["path"].parent,
                **({"member_weights": self.member_weights} if self.member_weights is not None else {}))
            self.deleted.extend({"path": self._relative(row["path"]), "bytes": row["bytes"]}
                                for row in retired)
        reduction_seconds = time.perf_counter() - started
        metadata = {r.field: SimpleNamespace(shape=(self.members,) + self.shape) for r in requests}
        product = SimpleNamespace(calls=1, _xp=np, fields=metadata, requests=requests,
            members=self.members, outputs=outputs, member_order=self.member_order, member_weights=self.member_weights)
        token = valid.replace(":", "-").replace("/", "-").replace("\\", "-")
        directory = self.root / self.domain / "ensemble" / "products"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"ensemble_{token}.nc"
        pending = path.with_suffix(".nc.pending")
        started = time.perf_counter()
        # Postage source planes belong to the private render cache. The public
        # aggregate contains only statistics, probabilities and membership.
        public = SimpleNamespace(**vars(product))
        public.requests = tuple(replace(r, postage_stamp=False) for r in requests)
        event_attrs = {}
        from woof.ensemble.ice_accretion import CONTRACT as ICE_CONTRACT, FIELDS as ICE_FIELDS
        if any(r.field in (*ICE_FIELDS, "freezing_rain_rate") for r in requests):
            event_attrs["ice_accretion_sampling"] = json.dumps({**ICE_CONTRACT,
                "intervals_hours": frame.get("ice_sampling_intervals_hours", [])}, sort_keys=True)
        if self.member_order != tuple(range(self.members)) or self.member_metadata:
            event_attrs.update(ensemble_member_order=json.dumps(self.member_order),
                              ensemble_member_metadata=json.dumps(self.member_metadata, sort_keys=True))
        if self.events:
            event_attrs["ensemble_events"] = json.dumps(self.events, sort_keys=True)
        write_product_frame(pending, public, valid_time=valid, latitude=self.latitude, longitude=self.longitude,
                            global_attrs=event_attrs or None)
        pending.replace(path)
        render_source = None
        if any(r.postage_stamp for r in requests) and "postage" in render_products:
            scratch = frame["packs"][0]["path"].parent
            render_source = scratch / "aggregate-render-source.nc"
            if render_source.exists():
                raise FileExistsError("private render source already exists; an unowned cache cannot be replaced")
            write_product_frame(render_source, product, valid_time=valid,
                                latitude=self.latitude, longitude=self.longitude, global_attrs=event_attrs or None)
        write_seconds = time.perf_counter() - started
        started = time.perf_counter()
        maps, transcript = rustwx.run_ensemble_product_renderer(self.renderer, render_source or path,
            out_dir=self.root / "maps", fields=tuple(r.field for r in requests),
            domain=self.domain, products=render_products)
        frame.update(status="complete", products=[self._relative(path)], maps=[self._relative(p) for p in maps],
            reduction_seconds=reduction_seconds, write_seconds=write_seconds,
            render_seconds=time.perf_counter() - started, renderer_transcript=transcript,
            weighting=weight_receipt(self.member_order, self.member_weights, frame["members_received"]),
            product_availability=[dict(field=r.field, product=kind, status="complete",
                **weight_receipt(self.member_order, self.member_weights, frame["members_received"]))
                for r in requests for kind in ("mean", "spread", "min", "max", "probability", "pmm", "paintball", "spaghetti")
                if f"{r.field}:{kind}" in outputs],
            replay_backend="cuda_reference" if self.gpu_replay else "rust_cpu",
            gpu_replay_required_bytes=self.replay_memory_plan().required_bytes(self.members) if self.gpu_replay else 0,
            host_product_payload_bytes=sum(a.nbytes for a in outputs.values()),
            host_extra_replay_cache_bytes=0 if all(r.postage_stamp for r in requests) else self.members * ny * nx * 4)
        self._manifest()
        if render_source is not None:
            size = render_source.stat().st_size
            render_source.unlink()
            self.deleted.append({"path": self._relative(render_source), "bytes": size})
        for pack in frame["packs"] if retire_diagnostics else ():
            path = pack["path"]
            size = path.stat().st_size
            path.unlink()
            self.deleted.append({"path": self._relative(path), "bytes": size})
        self._manifest()
        return {k: v for k, v in frame.items() if k != "packs"}

    def finish_in_subprocess(self, valid_time, *, available_bytes, consumer,
                render_products=("mean", "spread", "min", "max", "prob", "paintball", "postage", "pmm", "spaghetti")):
        """Keep product writing, hashes and render orchestration off forecast GIL."""
        from copy import deepcopy
        from woof.ensemble.product_worker import REQUEST_SCHEMA, RESULT_SCHEMA, _owned_path, _identity, _signature
        valid = str(valid_time)
        coordinate = self._ensure_coordinate_file()
        with self._lock:
            if valid in self._processing:
                raise ValueError("ensemble frame already has an active replay")
            original = self.frames[valid]
            if original["status"] != "pending" or original["members_received"] != sorted(self.member_order):
                raise ValueError("probability frame requires exactly the complete requested member roster")
            frame = deepcopy(original)
            self._processing.add(valid)
        root = self.root.resolve()
        completed = False
        directory = None
        try:
            directory = Path(tempfile.mkdtemp(prefix="product-process-", dir=frame["packs"][0]["path"].parent))
            request_path, result_path, log_path = (directory / name for name in ("request.json", "result.json", "worker.log"))
            coordinate["path"] = str(root / coordinate["path"])
            for pack in frame["packs"]:
                pack["path"] = str(Path(pack["path"]).resolve())
            request = {"schema": REQUEST_SCHEMA, "parent_pid": os.getpid(), "root": str(root),
                "domain": self.domain, "valid_time": valid, "shape": list(self.shape),
                "member_order": list(self.member_order), "member_metadata": list(self.member_metadata),
                "requests": [row.describe() for row in self.requests], "events": self.events,
                "coordinate_file": coordinate, "frame": frame, "tile_rows": self.tile_rows,
                "available_bytes": int(available_bytes), "renderer": str(self.renderer.resolve()),
                "render_products": list(render_products)}
            request_path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1")
            consumer.run_owned([sys.executable, "-m", "woof.ensemble.product_worker",
                str(request_path), "--result", str(result_path)], environment=environment, log_path=log_path)
            if result_path.is_symlink():
                raise ValueError("isolated ensemble result cannot be a symlink")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row = result.get("row", {})
            if (result.get("schema") != RESULT_SCHEMA or result.get("domain") != self.domain
                    or result.get("valid_time") != valid or row.get("valid_time") != valid
                    or row.get("status") != "complete" or row.get("members_received") != sorted(self.member_order)):
                raise ValueError("isolated ensemble result changed its committed hour or roster")
            expected_paths = set(row.get("products", ())) | set(row.get("maps", ()))
            records = result.get("artifactRecords", ())
            if (len(records) != len(expected_paths)
                    or {record.get("path") for record in records} != expected_paths):
                raise ValueError("isolated ensemble result omitted product hashes")
            for record in records:
                _identity(_owned_path(root, record.get("path")), expected=record)
            signatures = {pack["path"]: _identity(pack["path"], expected=pack)[1]
                for pack in original["packs"]}
            with self._lock:
                if self.frames[valid] is not original or original["status"] != "pending":
                    raise ValueError("isolated ensemble hour authority changed during replay")
                original.update(row)
                self.deleted.extend(result.get("deletedScratch", ()))
                self._manifest()
                for pack in original["packs"]:
                    path = pack["path"]
                    if _signature(path.stat()) != signatures[path]:
                        raise ValueError("diagnostic spill changed before verified retirement")
                    size = path.stat().st_size
                    path.unlink()
                    self.deleted.append({"path": self._relative(path), "bytes": size})
                self._manifest()
            completed = True
            return row
        finally:
            try:
                if directory is not None:
                    paths = (request_path, result_path, result_path.with_suffix(".json.pending"))
                    for path in (*paths, *((log_path,) if completed else ())):
                        if path.is_file() and not path.is_symlink():
                            size = path.stat().st_size
                            path.unlink()
                            self.deleted.append({"path": self._relative(path), "bytes": size})
                    if completed:
                        directory.rmdir()
            finally:
                with self._lock:
                    self._processing.remove(valid)

    def _gpu_reference_replay(self, frame, requests, plan, *, available_bytes, array_module):
        """Keep the existing CUDA implementation only as an exact test oracle."""
        from dataclasses import replace
        from woof import rustwx
        xp = array_module
        ny, nx = self.shape
        outputs = {row["name"]: np.empty(row["shape"], dtype=row["dtype"])
                   for row in plan.inventory(self.members)}
        for request in requests:
            if request.thresholds:
                outputs[f"{request.field}:thresholds"][:] = request.thresholds
            rows = min(self.tile_rows, ny)
            values = xp.empty((self.members, rows, nx), dtype=np.float32)
            tile_request = replace(request, postage_stamp=False)
            tile = prepare_product_frame({request.field: values}, (tile_request,),
                available_bytes=available_bytes - ((values.nbytes + 511) // 512) * 512, array_module=xp,
                member_weights=self.member_weights)
            # One typed read per field/pack avoids launching a decoder for
            # every row tile. This cache is at most one complete 2-D field
            # when postage is disabled; with postage it reuses its ledgered
            # headline cache. Forecast volumes never enter this replay.
            member_cache = outputs.get(f"{request.field}:members")
            if member_cache is None:
                member_cache = np.empty((self.members, ny, nx), np.float32)
            for pack in frame["packs"]:
                words = rustwx.read_ensemble_diagnostic_rows(self.renderer, pack["path"], request.field, row=0, rows=ny)
                if words.shape != (len(pack["member_ids"]), ny, nx):
                    raise ValueError("native diagnostic replay does not match the submitted member pack")
                for local, member in enumerate(pack["member_ids"]):
                    member_cache[self.member_positions[member]] = words[local]
            if request.field in ("refl", "rain_total"):
                from woof.ensemble.batch_products import probability_matched_mean
                stack = xp.asarray(member_cache)
                outputs[f"{request.field}:pmm"][:] = xp.asnumpy(probability_matched_mean(
                    stack, (xp.mean(stack, axis=0) if self.member_weights is None or len(set(self.member_weights)) == 1 else
                        xp.sum(stack.astype(xp.float64) * xp.asarray([w/max(self.member_weights) for w in self.member_weights], dtype=xp.float64)[:,None,None], axis=0) / sum(w/max(self.member_weights) for w in self.member_weights)),
                    xp, self.member_weights))
            step = rows - 1 if request.spaghetti else rows
            for start in range(0, ny, step):
                count = min(rows, ny - start)
                if request.spaghetti and count == 1 and start > 0:
                    break
                values.fill(0)
                for member in range(self.members):
                    values[member, :count].set(member_cache[member, start:start + count])
                tile()
                for kind in ("mean", "spread", "min", "max", "finite_count", "probability", "paintball"):
                    name = f"{request.field}:{kind}"
                    if name in tile.outputs:
                        outputs[name][..., start:start + count, :] = xp.asnumpy(tile.outputs[name])[..., :count, :]
                if request.spaghetti:
                    outputs[f"{request.field}:spaghetti"][..., start:start + count - 1, :] = xp.asnumpy(tile.outputs[f"{request.field}:spaghetti"])[..., :count - 1, :]
            del tile, values, member_cache, words
        xp.cuda.get_current_stream().synchronize()
        return outputs


_HEADLINE_SOURCE = r'''
extern "C" __global__ void ensemble_headline_diagnostics(
    const float *u, const float *v, const float *t, const float *q, const float *p,
    const float *rainnc, const float *rainc, const float *rainsh,
    const float *initial, float *wind, float *cumulative, float *rain,
    float *dewpoint, float *humidity, unsigned long long cells,
    int has_wind, int has_humidity, int has_rainc, int has_rainsh, int subtract_initial) {
    unsigned long long cell = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells) return;
    if (has_wind) wind[cell] = __fsqrt_rn(__fadd_rn(__fmul_rn(u[cell], u[cell]), __fmul_rn(v[cell], v[cell])));
    float total = __fadd_rn(rainnc[cell], has_rainc ? rainc[cell] : 0.0f);
    total = __fadd_rn(total, has_rainsh ? rainsh[cell] : 0.0f);
    cumulative[cell] = total;
    rain[cell] = fmaxf(0.0f, subtract_initial ? __fsub_rn(total, initial[cell]) : total);
    if (has_humidity) {
        // The surface equations are the production Rust import's f64 DAG.
        // CUDA log/exp are library calls, not transcribed libm source.
        // Q2 is a mixing ratio: e = w p / (0.622 + w). The specific-humidity
        // form (0.622 + 0.378 w) read Td2 0.1-0.25 K moist
        // (lane/dewpoint-mixing-ratio).
        double moisture = (double)q[cell], pressure = (double)p[cell];
        double vapor = __ddiv_rn(__dmul_rn(moisture, pressure), __dadd_rn(0.622, moisture));
        float missing = __int_as_float(0x7fc00000);
        if (!isfinite(q[cell]) || !isfinite(p[cell]) || q[cell] <= 0.0f || p[cell] <= 0.0f)
            dewpoint[cell] = missing;
        else {
            double logarithm = log(__ddiv_rn(fmax(vapor, 1.0), 611.2));
            double td = __ddiv_rn(__dmul_rn(243.5, logarithm), __dsub_rn(17.67, logarithm));
            dewpoint[cell] = __double2float_rn(__dadd_rn(td, 273.15));
        }
        if (!isfinite(t[cell]) || !isfinite(q[cell]) || !isfinite(p[cell]) || t[cell] <= 0.0f)
            humidity[cell] = missing;
        else {
            double temperature = __dsub_rn((double)t[cell], 273.15);
            double saturation = __dmul_rn(611.2, exp(__ddiv_rn(__dmul_rn(17.67, temperature), __dadd_rn(temperature, 243.5))));
            double rh = __ddiv_rn(__dmul_rn(100.0, vapor), saturation);
            humidity[cell] = isnan(rh) ? missing : __double2float_rn(fmin(fmax(rh, 0.0), 100.0));
        }
    }
}
extern "C" __global__ void ensemble_headline_rain_difference(
    const float *current, const float *earlier, float *result, unsigned long long cells) {
    unsigned long long cell = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < cells) result[cell] = fmaxf(0.0f, __fsub_rn(current[cell], earlier[cell]));
}
extern "C" __global__ void ensemble_headline_composite(
    const float *volume, float *result, unsigned long long cells, int levels,
    unsigned long long level_stride) {
    unsigned long long cell = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cells) return;
    float maximum = __int_as_float(0x7fc00000);
    for (int level = 0; level < levels; ++level) {
        float value = volume[(unsigned long long)level * level_stride + cell];
        if (isfinite(value)) maximum = isnan(maximum) ? value : fmaxf(maximum, value);
    }
    result[cell] = maximum;
}
'''


def headline_diagnostic_memory_plan(shape, *, refl_levels=0, host_inputs=False):
    """Preallocated ordinary-member diagnostic buffers and host-frame uploads."""
    from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
    shape = tuple(shape)
    fields = ("wind10", "cumulative", "rain_total", "dewpoint2", "humidity2",
              "refl", "initial", "earlier", "qpf_1h", "qpf_3h", "qpf_6h", "visibility", "fog_flag")
    specs = [BatchArraySpec(f"headline:{name}", shape, "shared") for name in fields]
    from woof.ensemble.visibility import INPUT_NAMES
    specs.extend(BatchArraySpec(f"headline:visibility:{name}", (3,) + shape, "shared")
                 for name in INPUT_NAMES)
    if host_inputs:
        specs.extend(BatchArraySpec(f"headline:input:{name}", shape, "shared")
                     for name in ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "RAINC", "RAINSH"))
        if refl_levels:
            specs.append(BatchArraySpec("headline:input:REFL_10CM", (refl_levels,) + shape, "shared"))
    return BatchMemoryPlan(tuple(specs), reserved_bytes=0)


class HeadlineDiagnosticCollector:
    """Commit ordinary diagnostics and drain complete hours independently.

    The callback borrows only the real physics/output diagnostics. It never
    reads a stale resident state for a streamed domain or changes stepping.
    All derived member fields and ensemble reductions run on the GPU.
    """

    def __init__(self, root, *, members, renderer, start_time, requests=None,
                 thresholds=None, comparison=None, keep_member_files=False, available_bytes=None,
                 tile_rows=32, render_products=("mean", "spread", "min", "max", "prob", "paintball", "postage", "pmm", "spaghetti"),
                 events=None, array_module=None, member_order=None, member_metadata=(),
                 retain_member_diagnostics=False, defer_replay=False, gpu_replay=False, resume=False,
                 columns_end_time=None):
        if array_module is None:
            import cupy as array_module
        self.xp, self.root, self.members = array_module, Path(root), index(members)
        self.member_order = _member_order(self.members, member_order)
        self.member_metadata = tuple(member_metadata)
        self.member_weights = roster_weights(self.member_order, self.member_metadata)
        self.member_archive = None
        self.cyclone_request = None
        self.cyclone_captures = {}
        self.cyclone_swaths = {}
        self.renderer, self.start_time = Path(renderer), start_time
        # NumPy is the explicit host-only planning facade. Real collectors
        # require the native CPU contract before a member starts stepping.
        if not gpu_replay and array_module is not np:
            from woof import rustwx
            rustwx.require_ensemble_diagnostic_reducer(self.renderer)
        if retain_member_diagnostics:
            from woof.ensemble.member_diagnostics import MemberDiagnosticArchive
            self.member_archive = MemberDiagnosticArchive(self.root,
                member_order=self.member_order, member_metadata=self.member_metadata, start_time=start_time,
                **({"resume": True} if resume else {}))
        self.requests = tuple(requests) if requests is not None else default_product_requests(
            DEFAULT_THRESHOLDS, thresholds=thresholds, comparison=comparison)[0]
        self.events = {str(name): tuple(conditions) for name, conditions in (events or {}).items()}
        for name, conditions in self.events.items():
            if not conditions or any(not isinstance(c, ThresholdCondition) for c in conditions):
                raise ValueError(f"{name} compound event needs ThresholdCondition terms")
            for condition in conditions:
                if condition.field not in DEFAULT_THRESHOLDS or condition.units != DEFAULT_THRESHOLDS[condition.field][0]:
                    raise ValueError("compound conditions must use declared headline fields and their stored units")
            if name not in {r.field for r in self.requests}:
                self.requests += (FieldProducts(name, "1", (0.5,), paintball=True, postage_stamp=True),)
        self.keep_member_files = bool(keep_member_files)
        self.available_bytes = available_bytes
        self.tile_rows = tile_rows
        self.render_products = tuple(render_products)
        self.spools, self.workspaces, self.rain_history, self.cohorts = {}, {}, {}, {}
        from woof.ensemble.ice_accretion import SavedFrameIce
        self.ice = SavedFrameIce(self.xp)
        self.rain_initial_ticks = {}
        self.rain_relocations = []
        self.output_ticks = set()
        self.member_files = []
        self._lock = threading.RLock()
        self._kernels = {}
        self._events = {}
        self._resume_output_ticks, self._resume_rain_ticks = set(), {}
        from woof.ensemble.product_consumer import DiagnosticProductConsumer
        from woof.ensemble.history_ledger import EnsembleHistoryLedger
        self.gpu_replay = bool(gpu_replay)
        self.product_consumer = DiagnosticProductConsumer(self.xp, gpu_replay=self.gpu_replay)
        self.history_ledger = EnsembleHistoryLedger(self.root, member_order=self.member_order, resume=resume)
        self.defer_replay = bool(defer_replay)
        #: Each member's any-point column store (rw-store v2), written as its frames land
        #: (:class:`woof.live_columns.MemberColumns`); armed on the first member frame, off with WOOF_COLUMNS=off.
        self.member_columns = None
        self._columns_end_time = columns_end_time

    def _member_column_frame(self, *, state, streamed, fields, metadata, valid_time, domain, member_id):
        """Queue this member frame for its column store when it is due. Never raises into the member."""
        if self.xp is np:      # the host-only planning facade carries no model state
            return
        try:
            if self.member_columns is None:
                from woof.live_columns import MemberColumns, columns_enabled
                if not columns_enabled():
                    self.member_columns = False
                    return
                self.member_columns = MemberColumns(self.root, end_time=self._columns_end_time)
            if self.member_columns:
                self.member_columns.submit(member=f"member-{index(member_id):04d}", domain=domain,
                    valid_time=valid_time, state=None if streamed is not None else state,
                    streamed_fields=fields if streamed is not None else None, metadata=metadata)
        except Exception as error:  # noqa: BLE001 - a store never stops a member
            print(f"member columns: frame not queued: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    def stop_member_columns(self):
        columns, self.member_columns = self.member_columns, False
        if columns:
            self._member_columns_summary = columns.stop()
        return getattr(self, "_member_columns_summary", None)

    def save_resume(self, *, timeout=60):
        with self._lock:
            # Member callbacks cannot enqueue another frame while the existing
            # product owner drains. The consumer stays open for later hours.
            self.product_consumer.wait_idle(timeout=timeout)
            from woof.ensemble.collector_resume import save
            return save(self)

    def checkpoint_hold(self):
        """Hold member submissions and pack retirement while a roster seals.

        Every submit, frame completion and product scheduling takes this
        lock, so checkpointed members can run on several cards: no other
        card retires a pack between save_resume and the roster hash.
        """
        return self._lock

    def restore_resume(self):
        with self._lock:
            from woof.ensemble.collector_resume import restore
            return restore(self)

    def _schedule_product(self, spool, valid, *, device=0):
        budget = self._budget() if self.gpu_replay else 0
        self.product_consumer.submit((spool.domain, valid), device=device,
            replay=(lambda: spool.finish(valid, available_bytes=budget,
                array_module=self.xp, render_products=self.render_products)) if self.gpu_replay else
                (lambda: spool.finish_in_subprocess(valid, available_bytes=budget,
                    consumer=self.product_consumer, render_products=self.render_products)))

    def _device_kernels(self, device):
        if device not in self._kernels:
            module = self.xp.RawModule(code=_HEADLINE_SOURCE, options=("--std=c++17",))
            self._kernels[device] = (module.get_function("ensemble_headline_diagnostics"),
                module.get_function("ensemble_headline_rain_difference"),
                module.get_function("ensemble_headline_composite"))
            record_module("woof.ensemble.batch_product_output:headline-diagnostics", source=_HEADLINE_SOURCE,
                          options=("--std=c++17",), module=module)
        return self._kernels[device]

    def _budget(self):
        if self.available_bytes is None:
            return int(self.xp.cuda.runtime.memGetInfo()[0])
        return int(self.available_bytes() if callable(self.available_bytes) else self.available_bytes)

    def _workspace(self, shape, device):
        from woof.ensemble.batch_storage import BatchStorage
        key = (device, tuple(shape))
        if key not in self.workspaces:
            plan = headline_diagnostic_memory_plan(shape)
            self.workspaces[key] = BatchStorage(plan, 1, array_module=self.xp, available_bytes=self._budget())
        return self.workspaces[key].arrays

    def _tick(self, valid_time):
        tick = round((valid_time - self.start_time).total_seconds() * 1_000_000)
        if tick < 0:
            raise ValueError("ensemble diagnostic clock precedes the forecast start")
        return tick

    def _placement_key(self, grid, episode, member, geometry, *, advance):
        """Rain-counter key for one uninterrupted footprint placement of a grid.

        A following nest that returns to an earlier footprint is a new placement:
        its cells were re-initialised on entry, so counters cannot be compared
        with the earlier visit. A grid that never moves keeps its plain key.
        """
        placements = self.__dict__.setdefault("_placements", {})
        base = (grid, episode, member)
        last = placements.get(base)
        if last is None:
            sequence = 0
        elif last[0] == geometry:
            sequence = last[1]
        else:
            sequence = last[1] + 1
        if advance:
            placements[base] = (geometry, sequence)
        key = (grid, episode, member, geometry)
        return key if sequence == 0 else key + (sequence,)

    def _record_rain(self, key, tick, cumulative):
        """Record exact member words; repeated observation of a tick is allowed."""
        history = self.rain_history.setdefault(key, {})
        if tick <= self._resume_rain_ticks.get(key, -1):
            if tick in history and cumulative.tobytes() != history[tick].tobytes():
                raise ValueError("resumed member precipitation differs at a retained clock")
            return history
        if not np.isfinite(cumulative).all():
            raise ValueError("member cumulative precipitation is nonfinite; QPF would be invalid")
        if tick in history:
            if cumulative.tobytes() != history[tick].tobytes():
                raise ValueError("member precipitation changed at the same captured clock; QPF endpoints would disagree")
            return history
        numeric_ticks = [item for item in history if isinstance(item, int)]
        if numeric_ticks:
            latest = max(numeric_ticks)
            if tick < latest:
                raise ValueError("member precipitation counter snapshots must advance in ordinary clock order")
            # A reset is a decrease on the order of the accumulated rain. Float
            # rounding residue (one cell at -2^-32 mm after 0.0 refused a real
            # member on 2026-10-19) sits far below this floor.
            delta = cumulative - history[latest]
            floor = 1e-6 + 4 * np.spacing(np.maximum(np.abs(cumulative), np.abs(history[latest])))
            if np.any(delta < -floor):
                point = np.unravel_index(int(np.argmin(delta)), delta.shape)
                raise ValueError("member cumulative precipitation reset; QPF would be invalid: "
                    f"member/grid={key[:3]}, ticks={latest}->{tick}, "
                    f"cells={int(np.count_nonzero(delta < -floor))}, min_change_mm={float(delta[point])}, "
                    f"at={point}, previous_mm={float(history[latest][point])}, "
                    f"current_mm={float(cumulative[point])}")
        if "initial" not in history:
            history["initial"] = cumulative.copy()
            self.rain_initial_ticks[key] = tick
            if len(key) > 4:
                previous = next((old for old in reversed(self.rain_history)
                    if old != key and old[:3] == key[:3] and self.rain_history[old]), None)
                # The ordinary relocation preparer carries overlap accumulators
                # and seeds strips under its declared accumulation policy.
                # A proven zero forecast origin is coordinate independent. Keep
                # it: subtracting the first moved frame would erase prior rain.
                # Nonzero origins need a remap; never invent their new values.
                origin = next((old for old in self.rain_history
                    if old != key and old[:3] == key[:3]
                    and self.rain_initial_ticks.get(old) == 0
                    and np.all(self.rain_history[old]["initial"] == 0)), None)
                if origin is not None:
                    history["initial"] = np.zeros_like(cumulative)
                    self.rain_initial_ticks[key] = 0
                self.rain_relocations.append({
                    "grid_id": key[0], "episode": key[1], "member_id": key[2],
                    "ticks_us": tick, "placement_sequence": key[4],
                    "previous_geometry": None if previous is None else previous[3],
                    "geometry": key[3],
                    "reason": "grid coordinates changed; reset comparison and rolling-window endpoints to avoid comparing different ground by index",
                    "rain_total": ("preserved forecast-zero origin; engine supplies placement counters under its declared accumulation policy"
                        if origin is not None else "unavailable: nonzero or missing forecast origin needs remapping"),
                    "rolling_qpf": "unavailable across relocation; requires both endpoints in this uninterrupted placement"})
        history[tick] = cumulative
        for earlier in numeric_ticks:
            if earlier < tick - 6 * 3_600_000_000:
                del history[earlier]
        return history

    def capture_rain_counters(self, fields, *, valid_time, grid_id, episode,
                              member_id, latitude, longitude, absent_zero_fields=()):
        """Capture baseline or exact QPF endpoints without publishing a frame.

        This observer never clips a timestep or interpolates a counter.
        Initial capture occurs before stepping, independently of history_begin.
        """
        with self._lock:
            xp = self.xp
            fields = {str(name).upper(): value for name, value in fields.items()}
            absent_zero_fields = {str(name).upper() for name in absent_zero_fields}
            if not absent_zero_fields <= {"RAINNC", "RAINC", "RAINSH"}:
                raise ValueError("zero-counter declarations must name original precipitation providers")
            if fields.get("RAINNC") is None and "RAINNC" not in absent_zero_fields:
                return False
            latitude, longitude = np.asarray(latitude, np.float32), np.asarray(longitude, np.float32)
            if latitude.ndim == 3 and latitude.shape[0] == 1:
                latitude, longitude = latitude[0], longitude[0]
            if latitude.ndim != 2 or longitude.shape != latitude.shape:
                raise ValueError("rain counter coordinates need matching two-dimensional grids")
            device = int(xp.cuda.runtime.getDevice())
            diagnose, _, _ = self._device_kernels(device)
            buffers = self._workspace(latitude.shape, device)
            get = lambda name: buffers[f"headline:{name}"]
            zero = get("earlier")
            zero.fill(0)
            resident = {}
            for name in ("RAINNC", "RAINC", "RAINSH"):
                if fields.get(name) is not None:
                    array = xp.asarray(fields[name], dtype=np.float32)
                    if array.shape != latitude.shape:
                        raise ValueError(f"{name} counter grid differs from its coordinates")
                    resident[name] = xp.ascontiguousarray(array)
                elif name in absent_zero_fields:
                    resident[name] = zero
            cells = int(np.prod(latitude.shape))
            diagnose(((cells + 255) // 256,), (256,),
                (zero, zero, zero, zero, zero, resident["RAINNC"], resident.get("RAINC", zero), resident.get("RAINSH", zero),
                 get("initial"), get("wind10"), get("cumulative"), get("rain_total"), get("dewpoint2"), get("humidity2"),
                 np.uint64(cells), np.int32(0), np.int32(0), np.int32("RAINC" in resident),
                 np.int32("RAINSH" in resident), np.int32(0)))
            geometry = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
            key = self._placement_key(index(grid_id), index(episode), index(member_id), geometry, advance=True)
            self._record_rain(key, self._tick(valid_time), xp.asnumpy(get("cumulative")))
            return True

    def has_rain_counter(self, valid_time, grid_id, episode, member_id, latitude, longitude):
        """Report an exact cached endpoint without opening a CUDA device."""
        with self._lock:
            latitude, longitude = np.asarray(latitude, np.float32), np.asarray(longitude, np.float32)
            if latitude.ndim == 3 and latitude.shape[0] == 1:
                latitude, longitude = latitude[0], longitude[0]
            if latitude.ndim != 2 or longitude.shape != latitude.shape:
                raise ValueError("rain counter coordinates need matching two-dimensional grids")
            geometry = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
            key = self._placement_key(index(grid_id), index(episode), index(member_id), geometry, advance=False)
            return self._tick(valid_time) in self.rain_history.get(key, {})

    @property
    def manifest_paths(self):
        return tuple(spool.manifest_path for spool in self.spools.values())

    def capturework_bytes(self, metadata):
        """Bound the 2-D diagnostic snapshot payload for output progress."""
        latitude = np.asarray(metadata["XLAT"])
        shape = latitude.shape[-2:]
        return int(np.prod(shape)) * 4 * len(self.requests)

    def memory_plan(self, shape, *, refl_levels=0, host_inputs=False):
        """Per-card collector buffers; add replay_memory_plan for its roster."""
        from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
        plan = headline_diagnostic_memory_plan(shape, refl_levels=refl_levels, host_inputs=host_inputs)
        specs = list(plan.arrays)
        from woof.ensemble.ice_accretion import memory_plan as ice_memory_plan
        specs.extend(ice_memory_plan(shape).arrays)
        for name, conditions in self.events.items():
            event_plan = compound_memory_plan(shape, conditions)
            for spec in event_plan.arrays:
                specs.append(BatchArraySpec(name + ":" + spec.name, spec.shape, spec.ownership, spec.dtype))
        return BatchMemoryPlan(tuple(specs), reserved_bytes=0)

    def receipt(self):
        """Domain manifests and explicit product coverage for the run manifest."""
        from woof.ensemble.ice_accretion import CONTRACT as ICE_CONTRACT
        with self._lock:
            domains = []
            pending = []
            unavailable = []
            for spool in self.spools.values():
                domains.append({"domain": spool.domain,
                    "manifest": spool.manifest_path.resolve().relative_to(self.root.resolve()).as_posix()})
                for frame in spool.frames.values():
                    row = {"domain": spool.domain, "valid_time": frame["valid_time"],
                        "status": frame["status"], "members_received": frame["members_received"],
                        "members_expected": self.members, "reason": frame.get("reason"),
                        "product_availability": frame.get("product_availability", []),
                        **weight_receipt(self.member_order, self.member_weights, frame["members_received"])}
                    if frame["status"] in ("pending", "incomplete"):
                        pending.append(row)
                    elif frame["status"] == "unavailable":
                        unavailable.append(row)
            return {"schema": NativeDiagnosticSpool.CONTRACT, "members_requested": self.members,
                "member_order": list(self.member_order), "member_metadata": list(self.member_metadata),
                **weight_receipt(self.member_order, self.member_weights),
                "gridded_aggregate_policy": "unavailable when member coordinates differ; member products retain native coordinates",
                "native_product_consumer": self.product_consumer.receipt(),
                "deferred_replay": self.defer_replay,
                "cyclone_swaths": dict(self.cyclone_swaths),
                "history_identity_ledger": (self.history_ledger.path.relative_to(self.root.resolve()).as_posix()
                    if self.history_ledger.records else None),
                "member_diagnostics": (None if self.member_archive is None else self.member_archive.receipt()),
                "member_columns": (self.member_columns.summary() if self.member_columns
                                   else getattr(self, "_member_columns_summary", None)),
                "domain_manifests": domains, "pending_rosters": pending,
                "unavailable_products": unavailable, "member_files": list(self.member_files),
                "events": {name: [{"field": c.field, "units": c.units,
                    "threshold": c.threshold, "comparison": c.comparison} for c in conditions]
                    for name, conditions in self.events.items()},
                "aggregate_only": not self.keep_member_files,
                "rain_history_host_bytes": sum(int(array.nbytes) for history in self.rain_history.values() for array in history.values()),
                "rain_history_device_bytes": 0,
                "rain_relocations": list(self.rain_relocations),
                "ice_accretion_sampling": {**ICE_CONTRACT, "intervals_hours": sorted(self.ice.intervals)},
                "ice_history_host_bytes": sum(int(array.nbytes)
                    for history in self.ice.history.values() for name, array in history.items() if name != "tick"),
                "ice_diagnostic_buffers": [{"device": key[0], "shape": key[1],
                    "required_bytes": storage.plan.required_bytes(1)} for key, storage in self.ice.workspaces.items()],
                "rain_initial_ticks": [{"grid_id": key[0], "episode": key[1], "member_id": key[2],
                    "geometry": key[3], "ticks_us": tick} for key, tick in self.rain_initial_ticks.items()],
                "compound_event_buffers": [{"device": key[0], "shape": key[1], "event": key[2],
                    "required_bytes": event.storage.plan.required_bytes(1)} for key, event in self._events.items()],
                "diagnostic_buffers": [{"device": key[0], "shape": key[1],
                    "required_bytes": storage.plan.required_bytes(1)} for key, storage in self.workspaces.items()]}

    def check_products(self):
        self.product_consumer.check()

    def cancel_products(self):
        self.product_consumer.close(cancel=True)
        self.stop_member_columns()

    def require_complete(self):
        """Check aligned output rosters; differing member grids stay explicit.

        A complete roster's hour is replayed by the product consumer after the
        last member commits it, so it stays "pending" until that replay
        lands. Judging completeness before the consumer drains reported a
        whole roster as incomplete ("received [0, 1, 2, 3] of 4") at the end
        of every native pack. Wait for the scheduled hours first; a consumer
        failure raises as itself here.
        """
        self.product_consumer.wait_idle(timeout=None)
        if self.cyclone_request is not None and set(self.cyclone_captures) != set(self.member_order):
            raise RuntimeError("every-step cyclone swaths are missing member captures")
        missing = self.receipt()["pending_rosters"]
        if self.defer_replay:
            missing = [row for row in missing if row["members_received"] != sorted(self.member_order)]
        if missing:
            first = missing[0]
            raise RuntimeError(f"ensemble output roster incomplete for {first['domain']} at {first['valid_time']}: "
                f"received {first['members_received']} of {self.members}; publishing probability would bias its denominator")
        return self.receipt()

    def observe_initial_state(self, *, model):
        """Capture the restored analysis before the first integration step.

        The counter handoff runs after member perturbations and restoration.
        Without this sample an ordinary ensemble starts at its first step,
        shortening exposure by that step and omitting analysis extrema.
        """
        if self.cyclone_request is None:
            return
        from woof.ensemble.runtime_context import current_capture
        if current_capture().member_id in self.cyclone_captures:
            return
        for node in model.walk_parent_first():
            if getattr(node, '_started', True):
                self.committed_step(node=node, model=model)

    def committed_step(self, *, node, model):
        """Keep each original member's extrema before its nest can move."""
        request = self.cyclone_request
        if request is None or node.cfg.grid_id != request.get('grid_id', 2):
            return
        from woof.ensemble.runtime_context import current_capture
        member = current_capture().member_id
        capture = self.cyclone_captures.get(member)
        if capture is None:
            from woof.tc_accumulator import Capture, utc_datetime
            with self._lock:
                capture = self.cyclone_captures.get(member)
                if capture is None:
                    options = dict(request, grid_id=request.get('grid_id', 2),
                        analysis_time=utc_datetime(self.start_time).isoformat(),
                        products=request.get('products', ['max-wind', 'rain', 'track']),
                        experiment_label=f'original member {member}')
                    capture = Capture(options)
                    capture.root = self.root/'cyclone'/'members'/f'member-{member:04d}'/'swath'
                    self.cyclone_captures[member] = capture
        capture.committed_step(node=node, model=model)

    def finish_run(self):
        """Retire unpublishable products and release only owned diagnostics."""
        self.stop_member_columns()
        self.product_consumer.close()
        with self._lock:
            for member, capture in self.cyclone_captures.items():
                if member not in self.cyclone_swaths:
                    with capture.common.maximum.device:
                        capture.finish()
                    self.cyclone_swaths[member] = str((capture.root/'receipt.json').relative_to(self.root))
            if self.member_archive is not None:
                self.member_archive.finish()
            for (_, _, valid), entries in self.cohorts.items():
                union = set().union(*(entry["members"] for entry in entries.values()))
                geometry_mismatch = len(entries) > 1 and union == set(self.member_order)
                for entry in entries.values():
                    spool = entry["spool"]
                    frame = spool.frames[valid]
                    if frame["status"] == "pending":
                        if self.defer_replay and frame["members_received"] == sorted(self.member_order):
                            continue
                        reason = ("member coordinates differ at this valid time; a common-grid aggregate would misplace weather"
                                  if geometry_mismatch else "requested member output roster is incomplete")
                        spool.retire_pending_frame(valid, reason=reason,
                            status="unavailable" if geometry_mismatch else "incomplete")
            for spool in self.spools.values():
                spool.retire_private_coordinates()
            return self.receipt()

    def history_committed(self, proof, *, member_id, grid_id, episode=0, valid_time):
        """Ledger a writer-committed history only when this ensemble retains it.

        The ordinary writer reports every history it commits. Two kinds are
        not this collector's: a history written while the request does not
        retain member files (a member's simulated radar needs its histories
        and deletes them itself), and a history its caller wrote outside this
        output tree. Registering either made the writer fail ("member
        histories require keep_member_files" or "is not in the subpath"),
        and a ledgered radar history would later be demanded back without a
        retirement receipt. Only retained histories inside the tree are
        ledgered, which is every history the production session keeps.
        """
        if not self.keep_member_files:
            return None
        try:
            Path(proof.path).resolve().relative_to(self.root.resolve())
        except ValueError:
            return None
        return self.add_member_file(proof.path, member_id=member_id, grid_id=grid_id,
            episode=episode, valid_time=valid_time, proof=proof)

    def add_member_file(self, path, *, member_id, grid_id, episode=0, proof=None, valid_time=None):
        """Register the opt-in ordinary history without changing its bytes."""
        with self._lock:
            if not self.keep_member_files:
                raise ValueError("member histories require keep_member_files")
            domain = f"d{index(grid_id):02d}" + (f"-episode-{index(episode):03d}" if episode else "")
            path = Path(path)
            if proof is None and not path.is_file():
                raise ValueError("retained member history must exist before registration")
            if proof is not None:
                identity = self.history_ledger.register(proof, member_id=member_id,
                    grid_id=grid_id, episode=episode, valid_time=valid_time)
            else:
                identity = next((row for row in self.history_ledger.inventory()
                                 if row["path"] == path.resolve().relative_to(self.root.resolve()).as_posix()), None)
            record = {"member_id": index(member_id), "domain": domain,
                "path": path.resolve().relative_to(self.root.resolve()).as_posix(),
                "bytes": path.stat().st_size if identity is None else identity["bytes"]}
            previous = next((row for row in self.member_files if row["path"] == record["path"]), None)
            if previous is not None:
                if previous != record:
                    raise ValueError("retained history changed its registered bytes")
                return previous
            self.member_files.append(record)
            self.member_files.sort(key=lambda row: (self.member_order.index(row["member_id"]), row["path"]))
            for spool in self.spools.values():
                if spool.domain == domain or spool.domain.startswith(domain + "-grid-"):
                    spool.add_member_file(path, member_id=member_id, domain=domain, identity=identity)
            return record

    def submit(self, *, state, streamed, metadata, refl_field, valid_time,
               grid_id, episode, member_id):
        """Consume the ordinary writer callback's borrowed fields once."""
        with self._lock:
            self.product_consumer.check()
            return self._submit(state=state, streamed=streamed, metadata=metadata,
                refl_field=refl_field, valid_time=valid_time, grid_id=grid_id,
                episode=episode, member_id=member_id)

    def __call__(self, **kwargs):
        return self.submit(**kwargs)

    def _submit(self, *, state, streamed, metadata, refl_field, valid_time,
                grid_id, episode, member_id):
        xp = self.xp
        if streamed is not None:
            fields = streamed.history_fields()
            if getattr(fields, "deferred", False):
                fields = fields.materialize()
            fields = dict(fields)
        else:
            physics = getattr(state, "physics", None)
            fields = dict(physics.output_fields()) if callable(getattr(physics, "output_fields", None)) else {}
            if physics is not None and "psfc" in getattr(physics, "fields", {}):
                from woof.io.wrfout import _driver_refreshes_psfc
                if _driver_refreshes_psfc(state):
                    fields["PSFC"] = physics.fields["psfc"]
        if streamed is None:
            for source, target in (("qr", "QRAIN"), ("qs", "QSNOW"), ("qg", "QGRAUP")):
                value = getattr(state, source, None)
                if value is not None:
                    fields[target] = value
        fields = {str(name).upper(): value for name, value in fields.items()}
        if streamed is None:
            existing_scratch = getattr(state, "existing_scratch", None)
            up_heli_max = existing_scratch("up_heli_max") if callable(existing_scratch) else None
            if up_heli_max is not None:
                fields["UP_HELI_MAX"] = up_heli_max
            for suffix in ("01", "02", "03"):
                band = existing_scratch("up_heli_max" + suffix) if callable(existing_scratch) else None
                if band is not None:
                    fields["UP_HELI_MAX" + suffix] = band
        if "UP_HELI_MAX" in fields:
            fields["UH"] = fields["UP_HELI_MAX"]
        if refl_field is not None:
            fields["REFL_10CM"] = refl_field
        latitude, longitude = np.asarray(metadata["XLAT"], np.float32), np.asarray(metadata["XLONG"], np.float32)
        if latitude.ndim == 3 and latitude.shape[0] == 1:
            latitude, longitude = latitude[0], longitude[0]
        domain = f"d{index(grid_id):02d}" + (f"-episode-{index(episode):03d}" if episode else "")
        valid = valid_time.strftime("%Y-%m-%d_%H:%M:%S") if hasattr(valid_time, "strftime") else str(valid_time)
        geometry = hashlib.sha256(latitude.tobytes() + longitude.tobytes()).hexdigest()
        key = (domain, geometry)
        if any(old_domain == domain and old_geometry != geometry for old_domain, old_geometry in self.spools):
            # Each moving footprint has its own native coordinate truth. It
            # cannot be painted on the earlier frame's static mesh.
            domain += "-grid-" + geometry[:12]
            key = (domain, geometry)
        rain_key = self._placement_key(index(grid_id), index(episode), index(member_id), geometry, advance=True)
        tick = self._tick(valid_time)
        if (rain_key, tick) in self._resume_output_ticks:
            return None
        self._member_column_frame(state=state, streamed=streamed, fields=fields, metadata=metadata,
            valid_time=valid_time, domain=domain, member_id=member_id)
        if key not in self.spools:
            self.spools[key] = NativeDiagnosticSpool(self.root, members=self.members,
                requests=self.requests, latitude=latitude, longitude=longitude,
                renderer=self.renderer, domain=domain, keep_member_files=self.keep_member_files,
                member_order=self.member_order, member_metadata=self.member_metadata,
                export_coordinates=self.defer_replay,
                gpu_replay=self.gpu_replay,
                tile_rows=self.tile_rows, events={name: [{"field": c.field, "units": c.units,
                    "threshold": c.threshold, "comparison": c.comparison} for c in conditions]
                    for name, conditions in self.events.items()})
            for row in self.member_files:
                if row["domain"] == domain or domain.startswith(row["domain"] + "-grid-"):
                    self.spools[key].add_member_file(self.root / row["path"],
                        member_id=row["member_id"], domain=row["domain"], identity=row)
        spool = self.spools[key]
        device = int(xp.cuda.runtime.getDevice())
        diagnose, difference, composite = self._device_kernels(device)
        buffers = self._workspace(latitude.shape, device)
        cells = int(np.prod(latitude.shape))
        get = lambda name: buffers[f"headline:{name}"]
        zero = get("earlier")
        zero.fill(0)
        resident = {}
        for name in ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "RAINC", "RAINSH", "REFD_MAX", "GUST", "WIND_GUST", "UH",
                     "UP_HELI_MAX01", "UP_HELI_MAX02", "UP_HELI_MAX03"):
            if name in fields:
                array = xp.asarray(fields[name], dtype=np.float32)
                if array.shape != latitude.shape:
                    raise ValueError(f"{name} has a different surface grid than ordinary output metadata")
                resident[name] = xp.ascontiguousarray(array)
        rain_key = self._placement_key(index(grid_id), index(episode), index(member_id), geometry, advance=True)
        history = self.rain_history.setdefault(rain_key, {})
        tick = self._tick(valid_time)
        output_key = (rain_key, tick)
        if output_key in self.output_ticks:
            raise ValueError("ensemble member output clock is duplicated")
        prior_outputs = [old_tick for old_key, old_tick in self.output_ticks if old_key == rain_key]
        if prior_outputs and tick <= max(prior_outputs):
            raise ValueError("ensemble member output times must advance in ordinary clock order")
        if history:
            get("initial").set(history["initial"])
        has_rain = "RAINNC" in resident
        has_wind = "U10" in resident and "V10" in resident
        has_humidity = all(name in resident for name in ("T2", "Q2", "PSFC"))
        diagnose(((cells + 255) // 256,), (256,),
            tuple(resident.get(name, zero) for name in ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "RAINC", "RAINSH")) +
            (get("initial"), get("wind10"), get("cumulative"), get("rain_total"), get("dewpoint2"), get("humidity2"),
             np.uint64(cells), np.int32(has_wind), np.int32(has_humidity),
             np.int32("RAINC" in resident), np.int32("RAINSH" in resident), np.int32(bool(history))))
        diagnosed = {}
        if any(request.field == "visibility" for request in self.requests):
            from woof.ensemble.visibility import diagnose as diagnose_visibility
            if diagnose_visibility(state=state, fields=fields, streamed=streamed is not None,
                    visibility=get("visibility"), fog_flag=get("fog_flag"), workspace=buffers,
                    array_module=xp):
                diagnosed["visibility"] = get("visibility")
        if has_wind:
            diagnosed["wind10"] = get("wind10")
        if "T2" in resident:
            diagnosed["temperature2"] = resident["T2"]
            diagnosed["temperature2_below_freezing"] = resident["T2"]
        if has_humidity:
            diagnosed.update(dewpoint2=get("dewpoint2"), humidity2=get("humidity2"))
        if has_rain:
            cumulative = xp.asnumpy(get("cumulative"))
            history = self._record_rain(rain_key, tick, cumulative)
            get("initial").set(history["initial"])
            difference(((cells + 255) // 256,), (256,),
                       (get("cumulative"), get("initial"), get("rain_total"), np.uint64(cells)))
            if self.rain_initial_ticks[rain_key] == 0:
                diagnosed["rain_total"] = get("rain_total")
            for name, hours in (("qpf_1h", 1), ("qpf_3h", 3), ("qpf_6h", 6)):
                earlier = tick - hours * 3_600_000_000
                if earlier >= 0 and earlier in history:
                    get("earlier").set(history[earlier])
                    difference(((cells + 255) // 256,), (256,),
                                     (get("cumulative"), get("earlier"), get(name), np.uint64(cells)))
                    diagnosed[name] = get(name)
        elif tick in history:
            # The ordinary counter sampler can serve declared off-scheme
            # zeros without manufacturing a surface physics driver.
            get("cumulative").set(history[tick])
            difference(((cells + 255) // 256,), (256,),
                       (get("cumulative"), get("initial"), get("rain_total"), np.uint64(cells)))
            if self.rain_initial_ticks[rain_key] == 0:
                diagnosed["rain_total"] = get("rain_total")
            for name, hours in (("qpf_1h", 1), ("qpf_3h", 3), ("qpf_6h", 6)):
                earlier = tick - hours * 3_600_000_000
                if earlier >= 0 and earlier in history:
                    get("earlier").set(history[earlier])
                    difference(((cells + 255) // 256,), (256,),
                               (get("cumulative"), get("earlier"), get(name), np.uint64(cells)))
                    diagnosed[name] = get(name)
        if has_rain:
            diagnosed.update(self.ice.submit(rain_key, tick, fields, get("cumulative")))
        else:
            self.ice.invalidate(rain_key)
        if self.member_archive is not None:
            archive_fields = {name: resident[name] for name in ("T2", "U10", "V10") if name in resident}
            if "rain_total" in diagnosed:
                archive_fields["RAIN_TOTAL"] = diagnosed["rain_total"]
            self.member_archive.submit(archive_fields, member_id=member_id, valid_time=valid_time,
                grid_id=grid_id, episode=episode, latitude=latitude, longitude=longitude, array_module=xp)
        if "REFD_MAX" in resident:
            diagnosed["refl"] = resident["REFD_MAX"]
        elif "REFL_10CM" in fields:
            volume = xp.asarray(fields["REFL_10CM"], dtype=np.float32)
            if volume.ndim == 2 and volume.shape == latitude.shape:
                diagnosed["refl"] = volume
            elif volume.ndim == 3 and volume.shape[1:] == latitude.shape:
                if (volume.strides[-2:] != (latitude.shape[1] * 4, 4)
                        or volume.strides[0] < cells * 4 or volume.strides[0] % 4):
                    volume = xp.ascontiguousarray(volume)
                composite(((cells + 255) // 256,), (256,), (volume, get("refl"),
                    np.uint64(cells), np.int32(volume.shape[0]), np.uint64(volume.strides[0] // 4)))
                diagnosed["refl"] = get("refl")
            else:
                raise ValueError("reflectivity has a different grid than ordinary output metadata")
        for name in ("GUST", "WIND_GUST"):
            if name in resident:
                diagnosed["gust"] = resident[name]
                break
        if "UH" in resident:
            diagnosed["uh"] = resident["UH"]
        for suffix in ("01", "02", "03"):
            if "UP_HELI_MAX" + suffix in resident:
                diagnosed["uh" + suffix] = resident["UP_HELI_MAX" + suffix]
        for name, conditions in self.events.items():
            required = {condition.field for condition in conditions}
            if required <= set(diagnosed):
                event_key = (device, latitude.shape, name)
                inputs = {field: diagnosed[field][None] for field in required}
                if event_key not in self._events:
                    self._events[event_key] = PreparedCompoundField(inputs, conditions,
                        available_bytes=self._budget(), array_module=xp)
                event = self._events[event_key]
                event.rebind_fields(inputs)
                diagnosed[name] = event()[0]
        requested = {request.field for request in self.requests}
        supplied = {name: array[None] for name, array in diagnosed.items() if name in requested}
        complete = spool.submit(supplied, member_ids=(member_id,), valid_time=valid,
                                unavailable_fields=requested - set(supplied), array_module=xp)
        if any(name in supplied for name in ("ice_accretion", "freezing_rain_total")):
            with spool._lock:
                frame = spool.frames[valid]
                used = set(frame.get("ice_sampling_intervals_hours", []))
                if self.ice.last_interval is not None:
                    used.add(self.ice.last_interval)
                frame["ice_sampling_intervals_hours"] = sorted(used)
                spool._manifest_locked()
        self.output_ticks.add(output_key)
        for event in self._events.values():
            event.fields = {}
        cohort = self.cohorts.setdefault((index(grid_id), index(episode), valid), {})
        entry = cohort.setdefault(geometry, {"members": set(), "spool": spool})
        entry["members"].add(index(member_id))
        if complete and not self.defer_replay:
            self._schedule_product(spool, valid, device=device)
        return None
