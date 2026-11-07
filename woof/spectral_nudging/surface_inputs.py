"""Read native WRF auxinput9 endpoint words through the existing Rust bridge.

No interpolated fields or substitute analyses are manufactured here. WRF
stores OLD and NEW endpoints together in each auxinput9 record, including
already-staggered U10/V10. PX's soil nudging uses T2/Q2/SN independently of
the grid_sfdda selector. The forecast adapter owns device selection and
coupling; this reader preserves the native endpoint words for both consumers.
"""
from dataclasses import dataclass
from pathlib import Path

SURFACE_FIELDS = ("U10", "V10", "TH2", "Q2", "ODIS")
PX_FIELDS = ("T2", "Q2", "SN")


@dataclass(frozen=True)
class NativeSurfaceRecord:
    path: Path
    record: int
    fields: dict
    timestamp: str


def read_native_surface(path, *, record=0, surface=True, px=False):
    """Read one native record without promoting or rounding its float words.

    The result preserves each field's native NetCDF shape. In particular,
    U and V have different staggered shapes; conversion to the common-grid
    kernel ABI belongs to the GPU forecast adapter, not a CPU interpolation.
    Rust selects the numeric record before reading field data. Only the
    small character clock uses netcrust's complete-text metadata interface.
    """
    from woof.netcdf_bridge import Dataset
    if type(surface) is not bool or type(px) is not bool or not (surface or px):
        raise ValueError("request surface FDDA inputs, PX inputs, or both")
    if type(record) is not int or record < 0:
        raise ValueError("surface FDDA record must be a nonnegative integer")
    fields = tuple(dict.fromkeys((SURFACE_FIELDS if surface else ()) + (PX_FIELDS if px else ())))
    names = tuple(f"{field}_NDG_{endpoint}" for endpoint in ("OLD","NEW") for field in fields)
    path = Path(path)
    with Dataset(path) as dataset:
        required = ("Times",*names)
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            raise ValueError(f"surface FDDA {path} lacks native auxinput9 fields: {', '.join(missing)}")
        for name in names:
            var = dataset.variables[name]
            stagger = ("south_north","west_east_stag") if name.startswith("U10_") else (
                ("south_north_stag","west_east") if name.startswith("V10_") else
                ("south_north","west_east"))
            if (tuple(var.dimensions) != ("Time",*stagger) or var.dtype.kind != "f"
                    or var.dtype.itemsize != 4 or record >= var.shape[0]):
                raise ValueError(f"surface FDDA {name} must contain record {record} as native float32 ('Time', {stagger})")
        times = dataset.variables["Times"]
        if tuple(times.dimensions) != ("Time","DateStrLen") or record >= times.shape[0]:
            raise ValueError("surface FDDA Times must use native (Time,DateStrLen) dimensions")
        words = dataset.stored_words(required, record=record)
    timestamp = words.pop("Times")[0].tobytes().decode("ascii").rstrip("\x00 ")
    arrays = {name:value[0] for name,value in words.items()}
    # Views preserve native words and prevent an accidental edit to one endpoint
    # from changing the cached analysis while another consumer is using it.
    for value in arrays.values():
        value.flags.writeable = False
    return NativeSurfaceRecord(path,record,arrays,timestamp)
