"""Register of words where native MPAS-A itself reads undefined memory.

Proof-bar ruling 1: such words are excluded from an MPAS-suite identity check
only with a full witness (the measured run-to-run variation of the pinned
strict native binary, plus a bounds-checked build naming the read) and the
recorded sign-off for that specific case.  ``native_undefined_exclusions.json``
beside this module is the only source of such an exclusion.  Each entry names
one unit, one case, one field and one exact word set; excluded words are a
DECLARED RESIDUE and are never called identical.  Every other word of the field
is still compared as a raw 32-bit word, with no tolerance.

Breakage this prevents: an exclusion widening silently -- reused for another
case or field, grown past its witnessed cell, or turned into a tolerance --
which would let a real port defect pass as "native garbage".  ``entry`` refuses
any (unit, case, field) the register does not name, ``excluded_mask`` refuses
any layout other than the registered one, and ``compare`` counts every word
outside the set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REGISTER = Path(__file__).with_name("native_undefined_exclusions.json")
SCHEMA = "hex.native-undefined-exclusions/v1"
LABEL = "DECLARED RESIDUE (native reads undefined memory); never called identical"

_ENTRY_KEYS = {
    "id", "unit", "case", "label", "native", "trigger", "field", "layout",
    "cell_index", "mpas_indices", "netcdf_c_words", "witness", "sign_off",
    "process_count", "consumers",
}
_OPTIONAL_KEYS = {"all_input_check", "notes"}
# A grade of this kind never counts as landing proof for the MPAS suites (ruling 1).
_FORBIDDEN = ("tolerance", "rms", "relative", "ulp", "atol", "rtol", "mean")


class ExclusionRefusal(ValueError):
    """The request is outside what the register witnesses and the owner signed."""


def _keys(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            out.add(str(key).lower())
            out |= _keys(item)
    elif isinstance(value, list):
        for item in value:
            out |= _keys(item)
    return out


def validate_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Refuse an entry that is not one witnessed, signed, single-cell word set."""
    keys = set(entry)
    missing = _ENTRY_KEYS - keys
    if missing:
        raise ExclusionRefusal(f"entry {entry.get('id')!r} lacks {sorted(missing)}")
    unknown = keys - _ENTRY_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise ExclusionRefusal(f"entry {entry['id']!r} has unknown keys {sorted(unknown)}")
    graded = sorted(k for k in _keys(entry) if any(k == f or k.startswith(f + "_") for f in _FORBIDDEN))
    if graded:
        raise ExclusionRefusal(f"entry {entry['id']!r} carries a graded allowance {graded}; ruling 1 admits word sets only")
    if entry["label"] != LABEL:
        raise ExclusionRefusal(f"entry {entry['id']!r} label must be {LABEL!r}")
    tail = entry["layout"].get("shape_tail")
    if not (isinstance(tail, list) and tail and all(isinstance(n, int) and n > 0 for n in tail)):
        raise ExclusionRefusal(f"entry {entry['id']!r} needs a positive integer layout.shape_tail")
    if entry["layout"].get("order") != "C":
        raise ExclusionRefusal(f"entry {entry['id']!r} word indices must be NetCDF C order")
    cell = entry["cell_index"]
    if not (isinstance(cell, int) and cell >= 0):
        raise ExclusionRefusal(f"entry {entry['id']!r} cell_index must be a zero-based cell")
    words = entry["netcdf_c_words"]
    per_cell = int(np.prod(tail))
    first, last, count = words.get("first"), words.get("last"), words.get("count")
    if not (count == per_cell and first == cell * per_cell and last == first + count - 1):
        raise ExclusionRefusal(
            f"entry {entry['id']!r} word set {first}..{last} ({count}) is not exactly the {per_cell} words of cell {cell}")
    witness = entry["witness"]
    varying = witness.get("varying_word_indices") or []
    if not varying:
        raise ExclusionRefusal(f"entry {entry['id']!r} shows no measured native run-to-run variation")
    if len(varying) != witness.get("varying_words") or sorted(set(varying)) != varying:
        raise ExclusionRefusal(f"entry {entry['id']!r} varying word list and count disagree")
    if not all(first <= i <= last for i in varying):
        raise ExclusionRefusal(f"entry {entry['id']!r} witnessed variation lies outside its own word set")
    if witness.get("hosts", 0) < 2 or len(witness.get("process_counts") or []) < 2:
        raise ExclusionRefusal(f"entry {entry['id']!r} witness needs two hosts and two process counts")
    if not witness.get("bounds_check", {}).get("line"):
        raise ExclusionRefusal(f"entry {entry['id']!r} witness needs a bounds-checked build naming the read")
    receipts = witness.get("receipts") or []
    if not any(r.get("role") == "primary" for r in receipts) or not all(
            len(str(r.get("sha256", ""))) == 64 for r in receipts):
        raise ExclusionRefusal(f"entry {entry['id']!r} witness receipts need sha256 digests and a primary set")
    sign = entry["sign_off"]
    if not (sign.get("date") and sign.get("verbatim") and sign.get("by")):
        raise ExclusionRefusal(f"entry {entry['id']!r} has no case-specific sign-off")
    if not (isinstance(entry["consumers"], list) and entry["consumers"]):
        raise ExclusionRefusal(f"entry {entry['id']!r} names no consumer")
    if not (isinstance(entry["process_count"], dict) and entry["process_count"].get("native")
            and entry["process_count"].get("hex")):
        raise ExclusionRefusal(f"entry {entry['id']!r} must state the fixed process counts of its claim")
    return entry


def load(path: Path = REGISTER) -> dict[str, Any]:
    register = json.loads(Path(path).read_text(encoding="utf-8"))
    if register.get("schema") != SCHEMA:
        raise ExclusionRefusal(f"{path}: schema must be {SCHEMA!r}")
    entries = register.get("entries")
    if not isinstance(entries, list):
        raise ExclusionRefusal(f"{path}: entries must be a list")
    ids = [e.get("id") for e in entries]
    if len(set(ids)) != len(ids):
        raise ExclusionRefusal(f"{path}: duplicate entry ids")
    triples = [(e.get("unit"), e.get("case"), e.get("field")) for e in entries]
    if len(set(triples)) != len(triples):
        raise ExclusionRefusal(f"{path}: two entries for one unit/case/field")
    for item in entries:
        validate_entry(item)
    return register


def entry(unit: str, case: str, field: str, path: Path = REGISTER) -> dict[str, Any]:
    """The one registered exclusion for exactly this unit, case and field."""
    found = [e for e in load(path)["entries"]
             if (e["unit"], e["case"], e["field"]) == (unit, case, field)]
    if not found:
        raise ExclusionRefusal(
            f"no registered native-undefined exclusion for {unit}/{case}/{field}: compare every word")
    return found[0]


def excluded_mask(item: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    """Flat C-order mask of the registered words for a field of ``shape``."""
    validate_entry(item)
    tail = tuple(item["layout"]["shape_tail"])
    shape = tuple(int(n) for n in shape)
    if len(shape) != len(tail) + 1 or shape[1:] != tail:
        raise ExclusionRefusal(f"{item['field']} shape {shape} is not the registered (nCells, *{tail})")
    if shape[0] <= item["cell_index"]:
        raise ExclusionRefusal(f"{item['field']} has no cell {item['cell_index']}")
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    words = item["netcdf_c_words"]
    mask[words["first"]:words["last"] + 1] = True
    return mask


def _words(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    if array.dtype not in (np.dtype("<f4"), np.dtype("<u4")):
        raise ExclusionRefusal(f"compare needs little-endian 32-bit words, got {array.dtype}")
    return array.view("<u4").reshape(-1)


def compare(native: np.ndarray, actual: np.ndarray, item: dict[str, Any]) -> dict[str, Any]:
    """Compare every raw word; only the registered words are left out."""
    if np.shape(native) != np.shape(actual):
        raise ExclusionRefusal(f"shape {np.shape(native)} != {np.shape(actual)}")
    mask = excluded_mask(item, np.shape(native))
    a, b = _words(native), _words(actual)
    differ = a != b
    outside = np.flatnonzero(differ & ~mask)
    return {
        "entry_id": item["id"], "field": item["field"], "shape": list(np.shape(native)),
        "words": int(a.size), "excluded_words": int(mask.sum()),
        "compared_words": int(a.size - mask.sum()),
        "differing_words_outside_set": int(outside.size),
        "first_differing_outside": [int(i) for i in outside[:8]],
        "differing_words_inside_set_diagnostic": int((differ & mask).sum()),
        "native_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "actual_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
        "native_outside_set_sha256": hashlib.sha256(a[~mask].tobytes()).hexdigest(),
        "actual_outside_set_sha256": hashlib.sha256(b[~mask].tobytes()).hexdigest(),
        "label": item["label"],
    }


def require_identical_outside(result: dict[str, Any]) -> None:
    if result["differing_words_outside_set"]:
        raise ExclusionRefusal(
            f"{result['field']}: {result['differing_words_outside_set']} words differ outside the registered set, "
            f"first {result['first_differing_outside']}")
