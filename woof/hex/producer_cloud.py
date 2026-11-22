"""Native parameterized cloud publication; never infer SGS from total cloud.

MPAS Registry.xml diag_physics gates these three fields on bl_mynn_in.
woof physics_inventory.MYNN_PBL_STATE_3D and mynn_pbl_runtime._STATE_FIELD
persist the corresponding WRF names. YSU has no such prognostic/held cloud
state. Structural absence describes its parameterization, not zero cloud.
"""
from dataclasses import replace


# public MPAS name, ArWen's native name, physical units
MYNN_CLOUD_FIELDS = (
    ("cldfrac_bl", "cldfra_bl", "unitless"),
    ("qc_bl", "qc_bl", "kg kg^{-1}"),
    ("qi_bl", "qi_bl", "kg kg^{-1}"),
)
PBL_CLOUD_AUTHORITY = {
    "mynn": "native_mynn",
    "mynnedmf": "native_mynn",
    "bl_mynn": "native_mynn",
    "bl_mynnedmf": "native_mynn",
    "ysu": "structural_absence",
    "bl_ysu": "structural_absence",
}


def cloud_authority(pbl_scheme):
    """Refuse unreviewed schemes rather than classify them from file absence."""
    try:
        return PBL_CLOUD_AUTHORITY[pbl_scheme]
    except KeyError:
        raise ValueError(f"no native SGS cloud authority for PBL {pbl_scheme!r}") from None


def arwen_cloud_fields(pbl_scheme):
    return (tuple(native for _, native, _ in MYNN_CLOUD_FIELDS)
            if cloud_authority(pbl_scheme) == "native_mynn" else ())


def normalize_cloud(name, array, n_levels, n_cells):
    """A shape-only view: no numerical conversion, clipping, or filling."""
    if array.shape == (n_levels, 1, n_cells):
        return array[:, 0, :]
    if array.shape == (n_levels, n_cells):
        return array
    raise ValueError(f"native SGS {name} has shape {array.shape}; "
                     f"expected {(n_levels, 1, n_cells)} or {(n_levels, n_cells)}")


def publish_native_cloud(streams, layout):
    """Publish allocated MYNN pool words on history's existing clock.

    Registry allocation is package-qualified. No YSU fields are allocated or
    inserted. A partial MYNN group is a broken Registry contract and refuses.
    The ordinary PhysicsStreams copier preserves the pool's original words.
    """
    available = {r['name'] for r in layout.fields if r['pool'] == 'diag_physics'}
    names = tuple(public for public, _, _ in MYNN_CLOUD_FIELDS)
    present = available.intersection(names)
    if present and present != set(names):
        raise ValueError(f"incomplete native MYNN cloud pool: {sorted(present)}")
    if not present:
        return streams
    return [replace(s, fields=tuple(dict.fromkeys((*s.fields, *names))))
            if s.name == 'output' and 'output' in s.directions else s for s in streams]


def mean_cloud_contribution(pbl_scheme):
    """Meaning of an absent parameterized-cloud contribution to a full mean.

    The engine's gsd_diagnostics.run cloud() reads absent physics fields from
    its gsd_zero workspace. For YSU this is zero parameterized SGS contribution,
    not a measured cloud fraction. Never allocate those fields in a YSU member.
    MYNN must supply all three actual native fields; absence is corruption.
    """
    return None if cloud_authority(pbl_scheme) == 'native_mynn' else 0.0
