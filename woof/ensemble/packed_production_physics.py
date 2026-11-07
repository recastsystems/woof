"""Original member physics drivers with coordinated native column leaves.

Each driver retains its real mass and staggered grids, cadence, carriers,
surface wrapper and tendency composition. Only the supported column leaves join
members. This component does not admit or qualify a complete forecast graph.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields as dataclass_fields
from hashlib import sha256
import inspect
from math import prod
import threading
from types import FunctionType, MethodType, SimpleNamespace

from woof.ensemble.batch_rrtmgp import _config_identity, _scalar


_MYNN_SELECTOR_DEFAULTS = {"bl_mynn_version": "wrf_461",
    "bl_mynn_gsd41_unsquared_qtke": False, "bl_mynn_cloud_tendency_form": "wrf_461"}


def _mynn_pbl_selectors(cfg):
    """Validate actual generation metadata before any packed member write."""
    from woof.ensemble.batch_mynn import _pbl_options
    values = {name: getattr(cfg, name) for name in _MYNN_SELECTOR_DEFAULTS}
    _pbl_options(values)
    if values["bl_mynn_version"] == "wrf_461":
        # The ordinary wrapper omits all generation kwargs in this form.
        values = dict(_MYNN_SELECTOR_DEFAULTS)
    return values


def validate_production_physics_group(drivers, configs):
    """Check actual current clocks and schedules before any member writes."""
    drivers, configs = tuple(drivers), tuple(configs)
    if not drivers or len(drivers) != len(configs):
        raise ValueError("packed production physics needs one live configuration per driver")
    if len({id(driver) for driver in drivers}) != len(drivers):
        raise ValueError("packed production physics members share a mutable driver owner")
    first_cfg = _config_identity(configs[0])
    clocks = ("elapsed_seconds", "domain_start_offset", "p_top")
    cadence = ("radt_minutes", "bldt_seconds", "stepbl", "radiation_due_override",
               "surface_pbl_due_override", "cumulus_due_override",
               "carriers_need_producer_refresh")
    first_clock = {name: _scalar(getattr(drivers[0].state, name, 0.0)) for name in clocks}
    first_cadence = {name: _scalar(getattr(drivers[0], name, None)) for name in cadence}
    for member, (driver, cfg) in enumerate(zip(drivers, configs)):
        if _config_identity(cfg) != first_cfg:
            raise ValueError(f"packed physics member {member} has a different current configuration; joined leaves would use another member's clock or policy")
        if {name: _scalar(getattr(driver.state, name, 0.0)) for name in clocks} != first_clock:
            raise ValueError(f"packed physics member {member} has a different live clock, activation or top; joined leaves would execute at the wrong member time")
        if {name: _scalar(getattr(driver, name, None)) for name in cadence} != first_cadence:
            raise ValueError(f"packed physics member {member} has a different due schedule; a leaf rendezvous would change its ordinary cadence")
        if (int(cfg.sf_sfclay_physics), int(cfg.sf_surface_physics), int(cfg.bl_pbl_physics)) != (5, 3, 5):
            raise ValueError("packed production leaves bind MYNN surface, RUC and MYNN PBL; another selector needs its own ordinary member driver")
        if int(getattr(cfg, "cu_physics", 0)) not in (0, 99):
            raise ValueError("packed production cumulus binds cu_physics=99; another scheme needs its own column producer")
        if getattr(driver, "cam_ozone", None) is not None:
            raise ValueError("packed production leaves do not bind external CAM ozone")
        if int(getattr(cfg, "cu_physics", 0)) == 99:
            from woof.core.kf import KainFritsch
            adapter = driver.cumulus_callable
            if type(adapter) is not KainFritsch or adapter.scheme != 99:
                raise ValueError("packed CU99 needs each original KainFritsch owner")
            if "_call_kfcps" in vars(adapter):
                raise ValueError("packed CU99 cannot replace a customized member column method")
            if any(adapter is other.cumulus_callable for other in drivers[:member]):
                raise ValueError("packed CU99 members share a mutable cumulus owner")
            # Driver initialization owns these arrays; sharing a view would
            # let another member's kernel overwrite the held trigger state.
            for name, array in (("cu_nca", driver.cu_nca), ("w0avg", adapter.w0avg)):
                if array is None:
                    raise ValueError(f"packed CU99 needs initialized private {name}")
                lo, hi = int(array.data.ptr), int(array.data.ptr) + int(array.nbytes)
                for other in drivers[:member]:
                    previous = other.cu_nca if name == "cu_nca" else other.cumulus_callable.w0avg
                    start = int(previous.data.ptr)
                    if lo < start + int(previous.nbytes) and start < hi:
                        raise ValueError(f"packed CU99 members share mutable {name} storage")
        if int(getattr(cfg, "spp_pbl", 0)) or int(getattr(cfg, "spp_lsm", 0)):
            raise ValueError("packed production leaves have no member-indexed SPP pattern arguments; ordinary members retain their seeded patterns")
        if int(getattr(cfg, "mosaic_lu", 0)) or int(getattr(cfg, "mosaic_soil", 0)):
            raise ValueError("packed RUC has no member-indexed mosaic-fraction arguments; ordinary members retain their category planes")
        if cfg.swint_opt:
            raise ValueError("packed production has no priced per-member shortwave interpolation carrier binding; "
                "joining its direct-beam outputs would omit the original between-call fit state")
        if cfg.aer_opt:
            raise ValueError("packed production radiation binds RTE-RRTMGP columns without the legacy RRTMG "
                "Thompson aerosol-band optics; ordinary members retain aer_opt and aerosol-number coupling")
        if cfg.alb_sol:
            raise ValueError("packed production land binds ALBEDO/ALBBCK and its radiation proxy lacks the "
                "original sun-angle geometry; alb_sol needs private ALBSOL/ALBBCKSOL aliases")
        selectors = _mynn_pbl_selectors(cfg)
        if selectors["bl_mynn_version"] == "gsd_41" and cfg.icloud_bl > 0:
            raise ValueError("packed production radiation has no legacy RRTMG in-cloud MYNN merge; "
                "GSD QC_BL cannot enter the packed RRTMGP grid-mean coupling")
        if type(cfg.icloud_bl) is not int or cfg.icloud_bl != 1:
            raise ValueError("packed MYNN retains the original numerical driver requirement icloud_bl=1; "
                "an inactive-cloud configuration has no original column-driver binding")
        if driver.carriers is None:
            raise ValueError("packed physics cannot consume unstamped radiation carriers")
        if bool(driver.carriers.unsourced_consumed(3)) != bool(drivers[0].carriers.unsourced_consumed(3)):
            raise ValueError("packed members need different producer refreshes; joining radiation would change the ordinary carrier order")
    if getattr(configs[0], "ruc_irrigation", "wrf_461") == "wrf_45":
        fractions = [getattr(driver, "fields", {}).get("landusef") for driver in drivers]
        if any(value is not None for value in fractions):
            if any(value is None for value in fractions):
                raise ValueError("packed RUC irrigation members have different landusef presence; each ordinary member retains its category authority")
            if any(value.shape != fractions[0].shape or value.dtype != fractions[0].dtype for value in fractions):
                raise ValueError("packed RUC irrigation members have different landusef category layouts; one bank cannot substitute another member's fractions")
    return {"members": len(drivers), "clock": first_clock, "cadence": first_cadence,
            "config": first_cfg}


def _rebind_function(function, replacements):
    """Retain the exact ordinary code object without changing module globals."""
    globals_ = dict(function.__globals__)
    globals_.update(replacements)
    result = FunctionType(function.__code__, globals_, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class _MemberDriver:
    """Method-only proxy: all original mutable metadata stays on its owner."""
    def __init__(self, owner, member):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_member", member)
        object.__setattr__(self, "_ordinal", 0)
        object.__setattr__(self, "_counts", {})

    def __getattr__(self, name):
        owner, member = self._owner, self._member
        driver = owner.drivers[member]
        if name == "radiation_callable":
            def radiation(**kwargs):
                return self._leaf("radiation", (), kwargs)
            return radiation
        value = getattr(driver, name)
        if isinstance(value, MethodType) and value.__self__ is driver:
            replacements = {
                "_run_sfclay": {"launch_mynn_surface_layer":
                    lambda *args, **kwargs: self._leaf("surface", args, kwargs)},
                "_run_ruc": {"ruc_lsm_step":
                    lambda *args, **kwargs: self._leaf("land", args, kwargs)},
                "_run_mynn_pbl": {"mynn_pbl_step":
                    lambda *args, **kwargs: self._leaf("pbl", args, kwargs)},
            }.get(name)
            function = value.__func__
            if replacements:
                function = _rebind_function(function, replacements)
            return MethodType(function, self)
        return value

    def __setattr__(self, name, value):
        if name.startswith("_") and name in ("_owner", "_member", "_ordinal", "_counts"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._owner.drivers[self._member], name, value)

    def _leaf(self, kind, args, kwargs):
        ordinal = self._ordinal
        self._ordinal += 1
        occurrence = self._counts.get(kind, 0)
        self._counts[kind] = occurrence + 1
        return self._owner._rendezvous(self._member, ordinal, kind,
                                      (kind, occurrence), args, kwargs)


class OriginalWorkspaceReuseScope:
    """Ownership proof only while every original member is at one leaf."""
    def __init__(self, owner):
        self.owner, self.kind = owner, None

    def require(self, kind, *, scratch_owner=None, adapters=None):
        owner = self.owner
        if (owner is None or owner._closed or not owner._active or self.kind != kind):
            raise RuntimeError("original workspace reuse requires the complete active member leaf rendezvous; an ordinary callback may own its scratch outside that interval")
        if scratch_owner is not None and scratch_owner is not owner.drivers[0].state:
            raise ValueError("MYNN workspace reuse cannot substitute another original member scratch owner")
        if adapters is not None and (len(adapters) != len(owner.adapters)
                or any(left is not right for left, right in zip(adapters, owner.adapters))):
            raise ValueError("radiation workspace reuse cannot substitute another original member adapter roster")


class PackedProductionPhysics:
    """Synchronize equal-clock original drivers at all-member column calls.

    ``compute`` owns private member streams and waits for their complete
    original compositions. ``member_compute`` instead accepts already
    concurrent calls on streams owned by a complete-step coordinator, after
    that coordinator calls ``begin`` and before it calls ``finish``.
    """
    def __init__(self, drivers, *, available_bytes, array_module=None, reuse_original_workspaces=False):
        if array_module is None:
            import cupy as array_module
        from woof.core.physics import PhysicsDriver
        from woof.core.rrtmgp import RRTMGPRadiation
        self.xp, self.drivers = array_module, tuple(drivers)
        if not self.drivers or any(type(driver) is not PhysicsDriver for driver in self.drivers):
            raise TypeError("packed production physics needs initialized ordinary PhysicsDriver owners")
        self.adapters = tuple(driver.radiation_callable for driver in self.drivers)
        if any(type(adapter) is not RRTMGPRadiation for adapter in self.adapters):
            raise TypeError("packed production radiation binds each member's native RRTMGPRadiation owner")
        self.members, self.available_bytes = len(self.drivers), int(available_bytes)
        if self.available_bytes <= 0:
            raise ValueError("packed production physics needs a positive separately admitted workspace budget")
        self.device = int(self.xp.cuda.runtime.getDevice())
        self.stream = self.xp.cuda.Stream(non_blocking=True)
        self.member_streams = tuple(self.xp.cuda.Stream(non_blocking=True) for _ in self.drivers)
        self.proxies = tuple(_MemberDriver(self, member) for member in range(self.members))
        self._condition, self._compute_lock = threading.Condition(), threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=self.members)
        self._banks, self._components, self._priced, self._active = {}, {}, 0, False
        self._closed, self._error = False, None
        if type(reuse_original_workspaces) is not bool:
            raise TypeError("original workspace reuse needs an explicit boolean ownership declaration")
        self.reuse_original_workspaces = reuse_original_workspaces
        self.workspace_reuse = OriginalWorkspaceReuseScope(self) if reuse_original_workspaces else None
        self.receipt = {"members": self.members, "complete_forecast_qualified": False,
            "driver_order": "unchanged ordinary compute and wrapper code objects",
            "grid_coupling": "ordinary member mass coupling, staggered interpolation and rings",
            "completed_compositions": 0, "leaf_calls": {}, "allocations": [],
            "ordinary_source_sha256": {name: sha256(inspect.getsource(getattr(PhysicsDriver, name)).encode()).hexdigest()
                for name in ("compute", "_run_sfclay", "_run_ruc", "_run_mynn_pbl", "_run_radiation", "_compose_tendencies")}}

    def begin(self, configs):
        if self._closed or self._active:
            raise RuntimeError("packed production owner is closed or a composition is already active")
        configs = tuple(configs)
        metadata = validate_production_physics_group(self.drivers, configs)
        if hasattr(self, "_bound_metadata"):
            # Only runtime adaptive-derived step words may change together.
            from woof.ensemble.batch_rrtmgp import validate_rrtmgp_binding_metadata
            first = {"adapter": {}, "config": metadata["config"], "p_top": float(self.drivers[0].state.p_top)}
            validate_rrtmgp_binding_metadata(first, self._bound_metadata)
        else:
            self._bound_metadata = {"adapter": {}, "config": metadata["config"],
                                    "p_top": float(self.drivers[0].state.p_top)}
        self.configs, self._rounds, self._finished, self._error = configs, {}, set(), None
        for proxy in self.proxies:
            proxy._ordinal = 0
            proxy._counts = {}
        self._active = True
        self.receipt["last_group"] = metadata

    def abort(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def member_compute(self, member, state, cfg):
        if not self._active or self._closed:
            raise RuntimeError("packed production member compute needs an active group")
        if not 0 <= member < self.members or state is not self.drivers[member].state:
            raise ValueError("packed production member compute received another member's state")
        if _config_identity(cfg) != _config_identity(self.configs[member]):
            raise ValueError("packed production member configuration changed after its current-clock rendezvous")
        adapter, previous = None, None
        had_override = False
        if int(getattr(cfg, "cu_physics", 0)) == 99:
            # Rebind the exact original adapter method on its private owner.
            # Its result provenance, diagnostics and fatal handling remain
            # the ordinary driver's; only the CUDA column call rendezvous
            # changes. Module globals and other members are never patched.
            adapter = self.drivers[member].cumulus_callable
            had_override = "_call_kfcps" in vars(adapter)
            previous = vars(adapter).get("_call_kfcps")
            function = _rebind_function(type(adapter)._call_kfcps,
                {"launch_kf_old": lambda *args, **kwargs:
                    self.proxies[member]._leaf("cumulus", args, kwargs)})
            adapter._call_kfcps = MethodType(function, adapter)
        try:
            return self.proxies[member].compute(state, cfg)
        except BaseException as error:
            self.abort(error)
            raise
        finally:
            if adapter is not None:
                if had_override:
                    adapter._call_kfcps = previous
                else:
                    del adapter._call_kfcps
            with self._condition:
                self._finished.add(member)
                self._condition.notify_all()

    def finish(self):
        if not self._active:
            raise RuntimeError("packed production group is not active")
        self.stream.synchronize()
        if len(self._finished) != self.members:
            raise RuntimeError("packed production group ended before every original member composition completed")
        self._active = False
        if self._error is not None:
            raise self._error
        self.receipt["completed_compositions"] += 1

    def compute(self, configs):
        with self._compute_lock:
            self.begin(configs)
            ready = self.xp.cuda.Event()
            ready.record(self.xp.cuda.get_current_stream())
            def work(member):
                with self.xp.cuda.Device(self.device), self.member_streams[member]:
                    self.member_streams[member].wait_event(ready)
                    try:
                        return self.member_compute(member, self.drivers[member].state, self.configs[member])
                    finally:
                        self.member_streams[member].synchronize()
            futures = [self._pool.submit(work, member) for member in range(self.members)]
            results, error = [], None
            for future in futures:
                try:
                    results.append(future.result())
                except BaseException as caught:
                    if error is None:
                        error = caught
            try:
                self.finish()
            finally:
                self._active = False
            if error is not None:
                raise error
            return tuple(results)

    def _rendezvous(self, member, ordinal, kind, binding, args, kwargs):
        current = self.xp.cuda.get_current_stream()
        ready = self.xp.cuda.Event()
        ready.record(current)
        with self._condition:
            row = self._rounds.setdefault(ordinal, {"kind": kind, "binding": binding, "requests": {}})
            if row["kind"] != kind or row["binding"] != binding or member in row["requests"]:
                error = ValueError("packed member leaf order differs from another member; joining would change ordinary physics ordering")
                self.abort(error)
                raise error
            row["requests"][member] = (args, kwargs, ready)
            if len(row["requests"]) == self.members:
                try:
                    with self.xp.cuda.Device(self.device), self.stream:
                        requests = tuple(row["requests"][index] for index in range(self.members))
                        for _, _, event in requests:
                            self.stream.wait_event(event)
                        if self.workspace_reuse is not None:
                            self.workspace_reuse.kind = kind
                        try:
                            row["results"] = getattr(self, "_" + kind)(binding, requests)
                        finally:
                            if self.workspace_reuse is not None:
                                self.workspace_reuse.kind = None
                        row["done"] = self.xp.cuda.Event()
                        row["done"].record(self.stream)
                    self.receipt["leaf_calls"][kind] = self.receipt["leaf_calls"].get(kind, 0) + 1
                except BaseException as error:
                    self.abort(error)
                self._condition.notify_all()
            while "done" not in row and self._error is None:
                if self._finished - row["requests"].keys():
                    self.abort(ValueError("a packed member completed before another member's leaf; its ordinary due schedule cannot join this call"))
                    break
                self._condition.wait()
            if self._error is not None:
                raise self._error
            current.wait_event(row["done"])
            return row["results"][member]

    def _reserve(self, name, amount):
        amount = int(amount)
        if self._priced + amount > self.available_bytes:
            raise MemoryError(f"packed production {name} would exhaust its admitted workspace before the all-member leaf")
        self._priced += amount
        self.receipt["allocations"].append({"name": name, "required_bytes": amount})

    def _gather(self, namespace, name, arrays, *, joined=True):
        arrays = tuple(arrays)
        first = arrays[0]
        if any(not isinstance(array, self.xp.ndarray) or array.shape != first.shape
               or array.dtype != first.dtype or int(array.device.id) != self.device for array in arrays):
            raise ValueError(f"packed production {namespace}:{name} needs equal-shaped resident member arrays on the owner device")
        key = namespace, name
        if joined and first.ndim == 3:
            shape = (first.shape[0], self.members * first.shape[1], first.shape[2])
            if key not in self._banks:
                size = prod(shape) * first.dtype.itemsize
                self._reserve(f"{namespace}:{name}", ((size + 511) // 512) * 512)
                self._banks[key] = self.xp.empty(shape, dtype=first.dtype)
            bank = self._banks[key]
            if bank.shape != shape or bank.dtype != first.dtype:
                raise ValueError(f"packed production {namespace}:{name} changed its bound column shape or dtype")
            ny = first.shape[1]
            for member, array in enumerate(arrays):
                self.xp.copyto(bank[:, member * ny:(member + 1) * ny], array)
            return bank
        if key not in self._banks:
            shape = (self.members,) + first.shape
            size = prod(shape) * first.dtype.itemsize
            self._reserve(f"{namespace}:{name}", ((size + 511) // 512) * 512)
            self._banks[key] = self.xp.empty(shape, dtype=first.dtype)
        bank = self._banks[key]
        if bank.shape != (self.members,) + first.shape or bank.dtype != first.dtype:
            raise ValueError(f"packed production {namespace}:{name} changed its bound member shape or dtype")
        for member, array in enumerate(arrays):
            self.xp.copyto(bank[member], array)
        if not joined:
            return bank
        if first.ndim == 2:
            return bank.reshape(self.members * first.shape[0], first.shape[1])
        raise ValueError("packed production column leaves bind two- or three-dimensional member fields")

    def _gather_maps(self, namespace, maps, *, joined=True, names=None):
        keys = tuple(maps[0]) if names is None else tuple(names)
        return {name: self._gather(namespace, name, [mapping[name] for mapping in maps], joined=joined)
                for name in keys if isinstance(maps[0][name], self.xp.ndarray)}

    def _member_view(self, array, member, *, joined=True):
        if not joined:
            return array[member]
        ny = self.configs[member].ny
        stripe = slice(member * ny, (member + 1) * ny)
        return array[:, stripe] if array.ndim == 3 else array[stripe]

    def _scatter(self, values, targets, *, joined=True):
        for member, target in enumerate(targets):
            for name, array in values.items():
                self.xp.copyto(target[name], self._member_view(array, member, joined=joined))

    @staticmethod
    def _common_kwargs(requests, names):
        result = {name: requests[0][1].get(name) for name in names}
        for _, kwargs, _ in requests[1:]:
            if any(_scalar(kwargs.get(name)) != _scalar(value) for name, value in result.items()):
                raise ValueError("packed production leaf scalar arguments differ; an all-member call would change an ordinary member's policy")
        return result

    def _surface(self, ordinal, requests):
        from woof.core.physics_inventory import MYNN_SURFACE_OUTPUTS
        from woof.ensemble.batch_mynn import prepare_mynn_surface_column_batch
        namespace = f"surface:{ordinal}"
        inputs = self._gather_maps(namespace + ":inputs", [args[0] for args, _, _ in requests])
        outputs = self._gather_maps(namespace + ":outputs", [vars(args[3]) for args, _, _ in requests], names=MYNN_SURFACE_OUTPUTS)
        mol = self._gather(namespace, "mol", [args[1] for args, _, _ in requests])
        ustm = self._gather(namespace, "ustm", [args[2] for args, _, _ in requests])
        knobs = self._common_kwargs(requests, ("dx", "itimestep", "isfflx", "variant"))
        if ordinal not in self._components:
            cfg = self.configs[0]
            self._components[ordinal] = prepare_mynn_surface_column_batch(inputs, mol=mol, ustm=ustm,
                outputs=outputs, members=self.members, ny=cfg.ny, nx=cfg.nx,
                available_bytes=self.available_bytes - self._priced, dx=knobs["dx"],
                isfflx=knobs["isfflx"], variant=knobs["variant"], array_module=self.xp)
        self._components[ordinal](itimestep=knobs["itimestep"])
        self._scatter(outputs, [vars(args[3]) for args, _, _ in requests])
        return (None,) * self.members

    def _land(self, ordinal, requests):
        from woof.ensemble.batch_ruc import PackedRucDriver
        from woof.core.surface_forcing import SurfacePrecipitationForcing, SURFACE_PRECIPITATION_FIELDS
        from woof.core.ruc_runtime import RUC_STATE_BINDING, RUC_PROFILE_BINDING
        from woof.core import ruc_memory as layout
        namespace = f"land:{ordinal}"
        field_maps, atmospheres = [args[0] for args, _, _ in requests], [args[1] for args, _, _ in requests]
        knobs = self._common_kwargs(requests, ("dt", "itimestep", "mosaic_lu", "mosaic_soil", "lakemodel", "ruc_soilprop",
            "ruc_irrigation", "ruc_qvg_cold_start", "ruc_2m_diagnostic", "ruc_snow", "flag_sm_adj", "spp_lsm"))
        supplied = set(RUC_STATE_BINDING.values()) | set(RUC_PROFILE_BINDING.values()) | {
            "z3d", "p8w", "t3d", "qv3d", "qc3d", "rho3d", "frzfrac", "tbot", "rainncv", "snowncv", "graupelncv"}
        needed = (set(RUC_STATE_BINDING) | set(RUC_PROFILE_BINDING) | set(layout.DRIVER_EXTRAS)
            | (set(layout.DRIVER_INPUT_NAMES) - supplied) | set(SURFACE_PRECIPITATION_FIELDS)
            | {"ivgtyp", "isltyp", "sr", "tmn", "psfc", "t2", "th2", "q2", "albbck", "chs", "flhc", "flqc"}
            | {"ruc_" + name for name in ("infiltr", "smelt", "runoff1", "runoff2")})
        if knobs["ruc_irrigation"] == "wrf_45" and any(value.get("landusef") is not None for value in field_maps):
            if any(value.get("landusef") is None for value in field_maps):
                raise ValueError("packed RUC irrigation member lacks its original landusef fractions")
            needed.add("landusef")
        fields = self._gather_maps(namespace + ":fields", field_maps, joined=False, names=sorted(needed))
        atmosphere = self._gather_maps(namespace + ":atmosphere", atmospheres, joined=False,
            names=("dz", "pressure", "temperature", "qv", "qc", "rho"))
        if knobs["flag_sm_adj"] or knobs["spp_lsm"]:
            raise ValueError("packed RUC lacks these runtime adjustment/pattern bindings; use each ordinary member")
        first_params = requests[0][1]["params"]
        def parameters_identity(params):
            return _scalar({"declared": params.restart_identity(), "iswater": params.iswater,
                            "isice": params.isice, "soil_half_depths": list(params.dzs)})
        params_id = parameters_identity(first_params)
        if any(parameters_identity(kwargs["params"]) != params_id for _, kwargs, _ in requests):
            raise ValueError("packed RUC members have different parameter authorities; one column launch would apply another member's tables")
        for member, (_, kwargs, _) in enumerate(requests):
            params = kwargs["params"]
            if params.rdlai2d and not getattr(params, "_seeded_lai_verified", False):
                if not bool(self.xp.isfinite(field_maps[member]["lai"]).all()):
                    raise ValueError(f"packed RUC member {member} has no finite monthly LAI seed; its land call would consume an uninitialized leaf area")
                params._seeded_lai_verified = True
        if ordinal not in self._components:
            cfg = self.configs[0]
            component = PackedRucDriver(self.members, (cfg.ny, cfg.nx), nzs=fields["smois"].shape[1],
                available_bytes=self.available_bytes - self._priced, array_module=self.xp)
            self._reserve(namespace + ":workspace", component.required_bytes)
            self._components[ordinal] = component
        component = self._components[ordinal]
        result = component.step(fields, atmosphere, params=first_params,
            precipitation=SurfacePrecipitationForcing.from_fields(fields),
            dt=knobs["dt"], itimestep=knobs["itimestep"], mosaic_lu=knobs["mosaic_lu"],
            mosaic_soil=knobs["mosaic_soil"], lakemodel=knobs["lakemodel"], soilprop=knobs["ruc_soilprop"],
            irrigation=knobs["ruc_irrigation"], qvg_cold_start=knobs["ruc_qvg_cold_start"],
            diagnostic_2m=knobs["ruc_2m_diagnostic"], snow=knobs["ruc_snow"])
        self._scatter(fields, field_maps, joined=False)
        return result

    def _pbl(self, ordinal, requests):
        from woof.ensemble.batch_mynn import prepare_mynn_pbl_column_batch
        from woof.core.mynn_pbl_runtime import (_ATMOSPHERE_LAYERS, _COLUMN_FIELDS, _STATE_FIELD,
            MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D)
        selectors = []
        for member, (_, kwargs, _) in enumerate(requests):
            actual = {name: kwargs.get(name, default) for name, default in _MYNN_SELECTOR_DEFAULTS.items()}
            expected = _mynn_pbl_selectors(self.configs[member])
            if any(type(actual[name]) is not type(expected[name]) or actual[name] != expected[name]
                   for name in actual):
                raise ValueError("packed MYNN PBL generation arguments differ from the actual member configuration; "
                    "substituting default options would change its ordinary kernel generation")
            selectors.append(actual)
        if any(value != selectors[0] for value in selectors[1:]):
            raise ValueError("packed MYNN PBL members have different generation arguments; one column binding cannot substitute another member's science")
        namespace = f"pbl:{ordinal}"
        needed = set(_STATE_FIELD.values()) | {name for _, name in _COLUMN_FIELDS} | {
            "exch_h", "exch_m", "rmol", "pblh", "kpbl", *MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D}
        fields = self._gather_maps(namespace + ":fields", [args[1] for args, _, _ in requests], names=sorted(needed))
        atmosphere = self._gather_maps(namespace + ":atmosphere", [args[0] for args, _, _ in requests],
            names=sorted({name for _, name in _ATMOSPHERE_LAYERS}))
        w = self._gather(namespace, "w", [kwargs["w"] for _, kwargs, _ in requests])
        names = ("dx", "delt", "itimestep", "mp_physics", "scalar_pblmix", "spp_pbl", "column_chunk",
            "closure", "bl_mynn_cloudpdf", "bl_mynn_mixlength", "bl_mynn_edmf", "bl_mynn_edmf_mom",
            "bl_mynn_edmf_tke", "bl_mynn_mixscalars", "bl_mynn_cloudmix", "bl_mynn_mixqt", "bl_mynn_output",
            "bl_mynn_tkeadvect", "icloud_bl")
        knobs = self._common_kwargs(requests, names)
        if knobs["scalar_pblmix"] or knobs["spp_pbl"]:
            raise ValueError("packed MYNN PBL lacks member scalar-pattern bindings; use ordinary members")
        if ordinal not in self._components:
            from woof.core.mynn_pbl_scratch import resolve_mynn_column_chunk
            cfg = self.configs[0]
            options = {name: knobs[name] for name in names[7:]} | selectors[0]
            component = prepare_mynn_pbl_column_batch(atmosphere, fields, w=w,
                members=self.members, ny=cfg.ny, nx=cfg.nx, dx=knobs["dx"], mp_physics=knobs["mp_physics"],
                column_chunk=resolve_mynn_column_chunk(cfg.nz) if knobs["column_chunk"] is None else knobs["column_chunk"],
                options=options, available_bytes=self.available_bytes - self._priced, array_module=self.xp,
                **({"scratch_owner": self.drivers[0].state, "reuse_scope": self.workspace_reuse}
                   if getattr(self, "reuse_original_workspaces", False) and cfg.ny * cfg.nx >= (
                       resolve_mynn_column_chunk(cfg.nz) if knobs["column_chunk"] is None else knobs["column_chunk"])
                   else {}))
            self._reserve(namespace + ":workspace", component.storage.plan.required_bytes(self.members))
            self._components[ordinal] = component
        result = self._components[ordinal](delt=knobs["delt"], itimestep=knobs["itimestep"])
        self._scatter(fields, [args[1] for args, _, _ in requests])
        # Member raw rates become contiguous before the original tendency
        # validator, mass coupling and A-grid-to-C-grid interpolation.
        returned = {name: self._gather(namespace + ":returned", name,
                    [self._member_view(array, member) for member in range(self.members)], joined=False)
                    for name, array in result.items()}
        return tuple({name: array[member] for name, array in returned.items()}
                     for member in range(self.members))

    def _cumulus(self, ordinal, requests):
        from dataclasses import asdict
        from woof.core.kf import launch_kf_old
        from woof.ensemble.batch_kf import KF_OLD_INPUTS, kf_old_call_reservation
        namespace = f"cumulus:{ordinal}"
        knobs = self._common_kwargs(requests,
            ("dx", "dt", "cudt_minutes", "adaptive", "stepcu", "kf_edrates"))
        species = requests[0][1]["species"]
        if any(asdict(kwargs["species"]) != asdict(species) for _, kwargs, _ in requests):
            raise ValueError("packed CU99 members have different native moist-package flags")
        inputs = {name: self._gather(namespace + ":inputs", name,
                      [args[index] for args, _, _ in requests])
                  for index, name in enumerate(KF_OLD_INPUTS)}
        if ordinal not in self._components:
            cfg = self.configs[0]
            amount = kf_old_call_reservation(cfg.nz, cfg.ny, cfg.nx,
                members=self.members, edrates=knobs["kf_edrates"],
                multiprocessors=self.xp.cuda.Device(self.device).attributes["MultiProcessorCount"])
            self._reserve(namespace + ":workspace_and_outputs", amount)
            self._components[ordinal] = {"required_call_bytes": amount}
        result = launch_kf_old(*(inputs[name] for name in KF_OLD_INPUTS),
                               species=species, **knobs)
        # KFCPS advances W0AVG even for held columns. Copy that in-place
        # mutation back before the original adapter resumes on its stream.
        self._scatter({"w0avg": inputs["w0avg"]},
            [{"w0avg": args[9]} for args, _, _ in requests])
        returned = {name: self._gather(namespace + ":returned", name,
                    [self._member_view(array, member) for member in range(self.members)], joined=False)
                    for name, array in result.items()}
        return tuple({name: array[member] for name, array in returned.items()}
                     for member in range(self.members))

    def _radiation(self, ordinal, requests):
        from woof.ensemble.batch_rrtmgp import prepare_rrtmgp_column_batch
        from woof.core.physics import RadiationResult
        namespace = f"radiation:{ordinal}"
        states = [kwargs["state"] for _, kwargs, _ in requests]
        atmosphere = self._gather_maps(namespace + ":atmosphere", [kwargs["atmosphere"] for _, kwargs, _ in requests],
            names=("pressure", "p_interface", "temperature", "exner", "qv", "qc", "qi"))
        field_maps = [kwargs["fields"] for _, kwargs, _ in requests]
        fields = self._gather_maps(namespace + ":fields", field_maps,
            names=tuple(name for name in ("tsk", "albedo", "emiss", "glw", "xland", "swddir", "qc_bl", "qi_bl", "cldfra_bl") if name in field_maps[0]))
        state = SimpleNamespace(elapsed_seconds=float(states[0].elapsed_seconds), p_top=float(states[0].p_top),
            physics=SimpleNamespace(microphysics_updates=int(states[0].physics.microphysics_updates)))
        for name in ("qc", "qr", "qi", "qs", "effc", "effr", "effi", "effs", "nc", "nr", "ni", "ns", "qnc", "qnr", "qni", "qns"):
            arrays = [getattr(member, name, None) for member in states]
            if arrays[0] is not None:
                setattr(state, name, self._gather(namespace + ":state", name, arrays))
        if ordinal not in self._components:
            cfg = self.configs[0]
            component = prepare_rrtmgp_column_batch(self.adapters, member_states=states, member_configs=self.configs,
                atmosphere=atmosphere, fields=fields, packed_state=state, members=self.members, ny=cfg.ny, nx=cfg.nx,
                available_bytes=self.available_bytes - self._priced, array_module=self.xp,
                **({"borrow_workspace": True, "reuse_scope": self.workspace_reuse}
                   if getattr(self, "reuse_original_workspaces", False) and self.adapters[0].chunk_workspace is not None else {}))
            self._reserve(namespace + ":workspace_and_transients", component.receipt["required_binding_bytes"])
            self._components[ordinal] = component
        component = self._components[ordinal]
        component.member_configs = self.configs
        component.state.elapsed_seconds, component.state.p_top = state.elapsed_seconds, state.p_top
        component.state.physics.microphysics_updates = state.physics.microphysics_updates
        result = component()

        def split(name, value):
            return self._gather(namespace + ":returned", name,
                [self._member_view(value, member) for member in range(self.members)], joined=False)

        # boundary_fluxes is a name -> (ny, nx) map; each plane splits by member.
        returned = {field.name: None if getattr(result, field.name) is None else
            ({key: split(f"{field.name}/{key}", value) for key, value in getattr(result, field.name).items()}
             if isinstance(getattr(result, field.name), dict) else split(field.name, getattr(result, field.name)))
            for field in dataclass_fields(result)}

        def member_value(value, member):
            if value is None:
                return None
            if isinstance(value, dict):
                return {key: array[member] for key, array in value.items()}
            return value[member]
        return tuple(RadiationResult(**{name: member_value(value, member)
                      for name, value in returned.items()}) for member in range(self.members))

    def close(self):
        if self._active:
            raise RuntimeError("packed production owner cannot release an active member composition")
        if self._closed:
            return
        self._pool.shutdown(wait=True)
        with self.xp.cuda.Device(self.device), self.stream:
            self.stream.synchronize()
            for component in self._components.values():
                if hasattr(component, "close"):
                    component.close()
            from woof.core.mynn_pbl_runtime import release_mynn_stream_scratch
            release_mynn_stream_scratch(device_id=self.device, stream=self.stream)
            self._components.clear()
            self._banks.clear()
        for stream in self.member_streams:
            with self.xp.cuda.Device(self.device), stream:
                stream.synchronize()
                release_mynn_stream_scratch(device_id=self.device, stream=stream)
        self._closed = True


__all__ = ["PackedProductionPhysics", "validate_production_physics_group"]
