"""Full-domain host views for cycle-boundary operations on streamed nests.

The slab template supplies metadata and array identities only. Horizontal
arrays must resolve through canonical store/geography keys. This object is
not a device DomainState and cannot run resident physics methods.
"""
from collections.abc import Mapping


class CanonicalStateRefused(RuntimeError):
    pass


class _CanonicalView:
    def __init__(self, template, arrays, cache):
        self._template_metadata = template
        self._canonical_arrays = arrays
        self._view_cache = cache

    def _view(self, value):
        replacement = self._canonical_arrays.get(id(value))
        if replacement is not None:
            return replacement
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            if value.ndim >= 2:
                raise CanonicalStateRefused("horizontal template array has no canonical store mapping")
            return value
        if isinstance(value, Mapping):
            return {key: self._view(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._view(item) for item in value)
        if callable(value):
            raise CanonicalStateRefused("resident methods cannot execute on a canonical host state")
        if hasattr(value, "__dict__"):
            cached = self._view_cache.get(id(value))
            if cached is None:
                cached = _CanonicalView(value, self._canonical_arrays, self._view_cache)
                self._view_cache[id(value)] = cached
            return cached
        return value

    def __getattr__(self, name):
        return self._view(getattr(self._template_metadata, name))


class CanonicalStoreState(_CanonicalView):
    """Borrow canonical full arrays while keeping scratch allocation explicit.

    ``inventory`` and ``geography_inventory`` are the inventories harvested
    from the same template that classified the store. An absent key is an
    error before publication. ``scratch_allocator`` is the route's bounded
    device allocator for coupler tables; no full state is allocated here.
    """
    _tslist_output = None
    _tslist_resume = None
    def __init__(self, template, cfg, *, store, geography, scalars,
                 inventory, geography_inventory, scratch_allocator=None):
        arrays = {}
        for names, canonical in ((inventory, store), (geography_inventory, geography)):
            for key, value in names.items():
                if key not in canonical:
                    raise CanonicalStateRefused(f"missing canonical array {key}")
                arrays.setdefault(id(value), canonical[key])
        super().__init__(template, arrays, {})
        self.cfg = cfg
        self.nx, self.ny, self.nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
        self._canonical_store = store
        self._canonical_geography = geography
        self._canonical_scalars = scalars
        self._scratch = {key[8:]: value for key, value in store.items() if key.startswith("scratch/")}
        self._scratch_allocator = scratch_allocator
        self.elapsed_seconds = float(scalars.get("elapsed_seconds", 0.))
        if ("wrf_mass_carriers_cold" in scalars
                or hasattr(template, "_wrf_mass_carriers_cold")):
            self._wrf_mass_carriers_cold = bool(scalars.get("wrf_mass_carriers_cold", False))

    def __getattr__(self, name):
        store = self.__dict__.get("_canonical_store", {})
        if name in store:
            return store[name]
        geography = self.__dict__.get("_canonical_geography", {})
        if "setup/"+name in geography:
            return geography["setup/"+name]
        return super().__getattr__(name)

    def existing_scratch(self, slot):
        return self._scratch.get(slot)

    def scratch(self, shape, slot, dtype=None):
        value = self._scratch.get(slot)
        if value is not None:
            if tuple(value.shape) != tuple(shape) or (dtype is not None and value.dtype != dtype):
                raise CanonicalStateRefused(f"canonical scratch shape/dtype mismatch: {slot}")
            return value
        if self._scratch_allocator is None:
            raise CanonicalStateRefused(f"scratch {slot} requires an explicit bounded allocator")
        value = self._scratch_allocator(shape, slot, dtype=dtype)
        self._scratch[slot] = value
        return value
