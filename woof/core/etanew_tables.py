"""ETAMPNEW lookup tables for the Eta Ferrier scheme (``mp_physics=95``).

WRF v4.6.1 ``ETANEWinit`` (phys/module_mp_etanew.F:2266-2281) opens
``ETAMPNEW_DATA.expanded_rain`` (unformatted, sequential, big-endian under
WRF's ``-fconvert=big-endian``) and READs twelve records in this order:

====  =======  =======  ==========================================
rec   name     words    meaning
====  =======  =======  ==========================================
1     VENTR1   951      rain ventilation, capacitance term
2     VENTR2   951      rain ventilation, fall-speed term
3     ACCRR    951      cloud-water collection by rain
4     MASSR    951      rain mass
5     VRAIN    951      mass-weighted rain fall speed
6     RRATE    951      rain rate
7     VENTI1   951      ice ventilation, capacitance term
8     VENTI2   951      ice ventilation, fall-speed term
9     ACCRI    951      cloud-water collection by large ice
10    MASSI    951      large-ice mass
11    VSNOWI   951      large-ice mass-weighted fall speed
12    VEL_RF   328      rimed-ice fall-speed increase, (2:9, 0:40)
====  =======  =======  ==========================================

The rain arrays span mean diameters MDRmin:MDRmax = 50:1000 microns and
the ice arrays MDImin:MDImax = 50:1000 (:15-26).  The file carries a
thirteenth record (35 words) that ``ETANEWinit`` never reads.

The byte-identical file from WRF v4.6.1 ``run/`` is packaged under
``woof/data/ferrier/tables/`` and SHA-256-validated before parsing (the P3 and
Thompson table pattern).  The same file is the one WRF's Ferrier-Aligo
scheme (mp_physics=5, ``fer_hires_init``) reads, and this loader exposes the
matching twelve-record device-table layout for both ports.

Fortran record markers are 4-byte big-endian lengths before and after
each record (``-frecord-marker=4``, WRF's BYTESWAPIO flags); the parser
checks both markers and refuses a file whose record lengths are not the
ones above, naming the record.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Environment override for the table directory (byte-identical copies
#: only: a file with other bytes fails the digest check).
ETAMPNEW_TABLE_ROOT_ENV = "WOOF_ETAMPNEW_TABLE_ROOT"

TABLE_FILENAME = "ETAMPNEW_DATA.expanded_rain"
#: SHA-256 of WRF v4.6.1 run/ETAMPNEW_DATA.expanded_rain (commit d66e442).
TABLE_SHA256 = (
    "28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13")
TABLE_SIZE = 43400

#: The twelve records ETANEWinit reads, in file order, with their lengths.
RECORDS: tuple[tuple[str, int], ...] = (
    ("VENTR1", 951), ("VENTR2", 951), ("ACCRR", 951), ("MASSR", 951),
    ("VRAIN", 951), ("RRATE", 951), ("VENTI1", 951), ("VENTI2", 951),
    ("ACCRI", 951), ("MASSI", 951), ("VSNOWI", 951), ("VEL_RF", 328),
)
#: Words in the flat device table (the twelve records back to back, in
#: the order kernels/etanew.cu's ETA_* offsets name).
TABLE_WORDS = sum(n for _, n in RECORDS)


class EtampnewTableError(ValueError):
    """The table file is missing, has other bytes, or other records."""


def _packaged_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "ferrier" / "tables"


def table_root() -> Path:
    env = (os.environ.get(ETAMPNEW_TABLE_ROOT_ENV)
           or os.environ.get("WOOF_FERRIER_TABLE_ROOT"))
    return Path(env) if env else _packaged_root()


@dataclass(frozen=True)
class EtampnewTables:
    """The parsed arrays, by name, plus the flat device-order table."""

    arrays: dict
    flat: np.ndarray
    sha256: str

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


SEED_FILENAME = TABLE_FILENAME + ".f32le"
SEED_SHA256 = "658644591b89f602995b4a7407cda721f8786f3fa7c4d199bcec313a2e6c329c"


# Both public WRF variants use the same records. Original rain ends at
# MDR=450, while ETANEWinit selects expanded rain through MDR=1000.
_TABLE_VARIANTS = {
    "expanded_rain": (TABLE_FILENAME, TABLE_SHA256, SEED_FILENAME, SEED_SHA256, RECORDS),
    "original_rain": ("ETAMPNEW_DATA",
        "a913785f0bf5f75f8428ede706ce67ae804c3b285b4a34aeb533c4ad5cd9bea2",
        "ETAMPNEW_DATA.f32le",
        "abeb1c8339cc8525c7bb9a496a9bd639111ab3041c6943b3a9ea36c2fa9876ec",
        tuple((name, 401 if i < 6 else words) for i, (name, words) in enumerate(RECORDS))),
}


def _variant_spec(variant):
    try:
        return _TABLE_VARIANTS[variant]
    except KeyError:
        raise EtampnewTableError(f"unknown ETAMPNEW table variant {variant!r}") from None


def parse_table_bytes(raw: bytes, *, variant="expanded_rain") -> EtampnewTables:
    """Validate WRF source and select its Rust-seeded binary32 words.

    The Rust tool owns all Fortran-record decoding. The source and seed
    hashes pin the exact data, including the narrower original rain range.
    No interpolation extends the original variant into the expanded range.
    """
    filename, source_hash, seed_name, seed_hash, records = _variant_spec(variant)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source_hash:
        raise EtampnewTableError(f"{filename}: sha256 {digest} differs from the pinned WRF table")
    path = _packaged_root() / seed_name
    seed = path.read_bytes()
    words = sum(n for _, n in records)
    if hashlib.sha256(seed).hexdigest() != seed_hash or len(seed) != words * 4:
        raise EtampnewTableError(f"{path}: Rust-seeded table sha256 or length differs; regenerate with tools/etampnew_seed.rs")
    flat = np.frombuffer(seed, dtype="<f4").astype(np.float32)
    flat.setflags(write=False)
    arrays = {}
    off = 0
    for name, words in records:
        arrays[name] = flat[off:off + words]
        off += words
    return EtampnewTables(arrays=arrays, flat=flat, sha256=digest)


@lru_cache(maxsize=4)
def _load(path_str: str, variant: str) -> EtampnewTables:
    path = Path(path_str)
    filename, expected, _, _, _ = _variant_spec(variant)
    if not path.is_file():
        raise EtampnewTableError(
            f"{path}: the {filename} table is missing (packaged under "
            f"woof/data/ferrier/tables; {ETAMPNEW_TABLE_ROOT_ENV} overrides the directory)")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise EtampnewTableError(
            f"{path}: sha256 {digest} is not WRF v4.6.1's {expected}; "
            "different fall-speed and ventilation bytes cannot enter the column physics")
    return parse_table_bytes(raw, variant=variant)


def load_tables(root: Path | None = None, *, variant="expanded_rain") -> EtampnewTables:
    """Load the shared table variant; mp=95 always selects expanded rain."""
    base = table_root() if root is None else Path(root)
    filename, _, _, _, _ = _variant_spec(variant)
    return _load(str((base / filename).resolve()), variant)


__all__ = [
    "ETAMPNEW_TABLE_ROOT_ENV", "EtampnewTableError", "EtampnewTables",
    "RECORDS", "TABLE_FILENAME", "TABLE_SHA256", "TABLE_WORDS",
    "load_tables", "parse_table_bytes", "table_root",
]
