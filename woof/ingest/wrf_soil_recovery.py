"""Locate original soil inputs; numeric recovery belongs to the Rust reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from woof.filesystem_paths import canonical_path, io_path

AUTHORITY_SCHEMA = "gpuwm-wrf-soil-authority-v1"

#: The largest land soil moisture still read as a volume fraction. Layer
#: water mislabeled as a fraction (kg m-2 or metres of water) lands one to
#: three orders of magnitude above 1, which is the breakage the recovery
#: below exists for. A fraction itself can sit a little above 1: WRF writes
#: 1.0 into land-ice columns (share/module_soil_pre.F) and metgrid's
#: interpolation next to them overshoots it. MEASURED 2026-10-03: real.exe of
#: NOAA-EMC/HRRR v4.1.21 on a public RAP analysis wrote land soil moisture up
#: to 1.000479 in 2,597 cells of 651 columns, and a strict bound of 1 refused
#: that file as mislabeled. 1.005 is WRF's own bound for a soil moisture too
#: large to be data (the "bad soil moisture" branch of
#: dyn_em/module_initialize_real.F in WRF 3.9).
SOIL_FRACTION_CEILING = 1.005


def _sha(path: Path) -> str:
    with io_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authority_from_vtable(path: Path) -> dict:
    """Read declared WPS quantities and layer bounds, without decoding data.

    GRIB1 level type 112 specifies centimetre layer bounds. Temperature
    selectors and truncated field suffixes do not establish the metgrid
    interpolation coordinate. Rust reads the actual SOIL_LEVELS axis and
    verifies its sorted depths against these source layer bounds.
    """
    water = {}
    for line in io_path(path).read_text(encoding="utf-8").splitlines():
        columns = [item.strip() for item in line.split("|")]
        if len(columns) < 7:
            continue
        name = columns[4]
        if not re.fullmatch(r"SOILM\d+", name):
            continue
        try:
            level_type = int(columns[1])
            if level_type != 112:
                raise ValueError("soil water layer bounds must use level type 112")
            if name in water:
                raise ValueError(f"duplicate soil water {name}")
            water[name] = (float(columns[2]) / 100.0,
                           float(columns[3]) / 100.0, columns[5])
        except ValueError as error:
            raise ValueError(f"{path}: invalid {name} source authority: {error}") from error
    if not water:
        raise ValueError(f"{path}: soil recovery needs SOILM fields with source layer bounds")
    ordered = sorted(water, key=lambda key: water[key][0])
    units = {water[key][2] for key in ordered}
    if len(units) != 1:
        raise ValueError(f"{path}: source soil layers declare inconsistent water units")
    unit = units.pop()
    compact = re.sub(r"[\s^*()]", "", unit.lower())
    quantities = {"kgm-2": "layer_water_mass", "kg/m2": "layer_water_mass",
                  "m": "equivalent_water_depth", "mm": "equivalent_water_depth",
                  "m3m-3": "volume_fraction", "m3/m3": "volume_fraction"}
    if compact not in quantities:
        raise ValueError(f"{path}: unsupported declared soil-water units {unit!r}")
    return {
        "schema": AUTHORITY_SCHEMA,
        "source_variable": "SOILM", "source_depth_variable": "SOIL_LEVELS",
        "source_quantity": quantities[compact], "source_units": unit,
        "source_layer_bounds_m": [[water[key][0], water[key][1]] for key in ordered],
        "source_depths_from_metgrid": True,
    }


#: The reader's refusals about the DECLARED LAYER SET, by the words
#: only they carry.  Nothing about where the sources are is wrong when
#: one of these comes back, so the discovery remedy below would send a
#: user to do again the exact thing they have already done -- which is
#: what the field report did with it.
#:
#: These words are a hand-transcribed match on the OTHER side's sentences
#: (``tools/rustwx/crates/rw-netcdf/src/soil_recovery.rs``), so a
#: rewording on either side would silently send the user back to the
#: discovery remedy.  ``tests/test_wrf_soil_refusal_marks.py`` reads the
#: Rust source and holds the pairing: exactly the layer refusals carry a
#: mark, and no other refusal does.
_LAYER_REFUSAL_MARKS = ("the authority declares", "declared layer")

#: The conversion refusal, which is NOT about the declared layer set and
#: NOT about finding the files: both inputs are the right ones and the
#: arithmetic between them does not land in [0, 1].
_CONVERSION_REFUSAL_MARK = "invalid volume fraction"


def _conversion_remedy(authority_path, met_path, authority):
    """The way out when the numbers are read and the conversion is wrong.

    The reader names the fraction it produced, the declared layer, that
    layer's thickness and the source value it started from, because the
    conversion is ``value * scale / thickness`` and only those numbers
    separate its two causes: a source that is not the declared quantity
    (the scale is wrong, and the fraction is usually out by a round
    factor) or a declared layer whose bounds are the wrong thickness for
    the source it is being applied to (the fraction is out by the ratio
    of the two thicknesses).  This remedy points at both, in that order,
    rather than sending anyone to look for other files.
    """

    variable = authority.get("source_variable", "SOILM")
    quantity = authority.get("source_quantity", "unstated")
    units = authority.get("source_units", "unstated")
    return (
        f"Both inputs were found and read; the conversion between them is "
        f"what failed. {authority_path} declares {variable} as "
        f"{quantity!r} in units {units!r}, and the reader divides by the "
        f"declared layer thickness printed above to reach a volume "
        f"fraction. Two things produce a fraction outside [0, 1]. Either "
        f"{met_path} does not hold that quantity -- check the SOILM units "
        f"attribute with `ncdump -h {met_path}` against source_units, "
        f"since a layer water mass read as a volume fraction is out by "
        f"roughly the layer thickness in millimetres -- or the declared "
        f"bounds for that layer are not the bounds the source was "
        f"produced on, in which case the fraction is out by the ratio of "
        f"the declared thickness to the real one. Correct whichever of "
        f"source_quantity/source_units or source_layer_bounds_m the two "
        f"numbers above disagree with.")


def _layer_remedy(authority_path, met_path, authority, domain):
    """The way out of a layer-declaration refusal: two numbers, then what to change.

    A declaration is allowed to name MORE layers than a cycle stacked,
    and that case recovers.  What cannot be reconciled is a stacked
    depth that no declared layer contains, so the way out is to widen
    the declaration rather than to look for other files.
    """

    variable = authority.get("source_variable", "SOILM")
    axis = authority.get("source_depth_variable", "SOIL_LEVELS")
    declared = len(authority.get("source_layer_bounds_m", []))
    return (
        f"Compare two numbers. {authority_path} declares {declared} source soil "
        f"layer(s): the SOILM rows of a Vtable, or the source_layer_bounds_m "
        f"entries of a {AUTHORITY_SCHEMA} file. `ncdump -h {met_path}` gives the "
        f"layer dimension of {variable} and {axis}, which is how many layers that "
        "cycle stacked. The reader prints both lists above. A declaration naming "
        "more layers than the cycle stacked is fine and recovers; what it must do "
        "is name a layer whose bounds contain each stacked depth. So extend "
        f"{authority_path} to cover those depths, or put a "
        f"wrf-soil-authority.{domain}.json beside the met_em ({AUTHORITY_SCHEMA}, "
        "source_layer_bounds_m in metres) declaring the layers this source was "
        "produced on. Pointing --soil-source at another copy of the same "
        "directory returns this same answer.")


def recover_supplied_soil(path, raw, attributes, *, source_directory=None):
    """Recover impossible mislabeled total water using original source inputs.

    Already physical volume fractions do not inspect additional inputs or
    change. Missing original evidence produces an actionable input error,
    before the caller initializes physics or launches a forecast.
    """
    if "SMOIS" not in raw or "LANDMASK" not in raw:
        return {}, None
    moisture = np.asarray(raw["SMOIS"])
    land = np.asarray(raw["LANDMASK"]) > .5
    if moisture.ndim != 3 or land.shape != moisture.shape[-2:]:
        return {}, None  # The main reader owns malformed domain geometry.
    if not np.any(moisture[:, land] > SOIL_FRACTION_CEILING):
        return {}, None

    supplied_path = Path(path)
    directory = canonical_path(source_directory if source_directory is not None else supplied_path.parent)
    path = canonical_path(supplied_path)
    remedy = ("Keep the matching first met_em file and its producing Vtable beside the WRF inputs, "
              "or pass --soil-source DIR for their original directory. WOOF-ArWen will recover the "
              "source layers automatically. Alternatively initialize directly from the source "
              "model through native preparation. The final four WRF layers cannot identify "
              "the source layer amounts on their own.")
    prefix = f"{supplied_path.name}: land soil moisture exceeds a volume fraction of 1; an upstream layer-water conversion may be missing. "
    domain = re.fullmatch(r"wrfinput_(d\d{2})", supplied_path.name)
    start = str(attributes.get("START_DATE", ""))
    if domain is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", start):
        raise ValueError(prefix + "The source domain and initialization time are not available. " + remedy)
    met_paths = [directory / f"met_em.{domain[1]}.{clock}.nc"
                 for clock in (start, start.replace(":", "_"))]
    present = [item for item in met_paths if io_path(item).is_file()]
    if not present:
        raise ValueError(prefix + "The matching source met_em file is missing. " + remedy)
    if len(present) > 1 and len({_sha(item) for item in present}) != 1:
        raise ValueError(prefix + "Two matching met_em filenames contain different data. " + remedy)
    met_path = present[0]
    # A normal WRF run links met_em from its producing WPS directory.
    # Follow that exact source link to find its table, without searching
    # unrelated runs or assuming that all nearby Vtables are equivalent.
    authority_roots = tuple(dict.fromkeys((directory, canonical_path(met_path).parent)))
    authority_paths = [root / name for root in authority_roots for name in (
        f"wrf-soil-authority.{domain[1]}.json", "wrf-soil-authority.json", "Vtable")]
    authority_path = next((item for item in authority_paths if io_path(item).is_file()), None)
    if authority_path is None:
        raise ValueError(prefix + "The producing Vtable or source-layer authority is missing. " + remedy)
    input_paths = {"wrfinput": path, "met_em": met_path, "authority": authority_path}
    hashes = {name: _sha(item) for name, item in input_paths.items()}
    # Which two files this run read, and the rule that picked each one.
    # A refusal from the reader is about THESE bytes, and a user holding
    # a WPS directory with several cycles and several domains in it
    # cannot check the refusal against the right pair unless it is named.
    selection = {
        "met_em": f"{met_path}, the met_em for domain {domain[1]} at the "
                  f"initialization time {start} this wrfinput declares, "
                  f"found in {directory}",
        "authority": f"{authority_path}, the first of "
                     + ", ".join(dict.fromkeys(item.name for item in authority_paths))
                     + " present in " + " then ".join(str(root) for root in authority_roots),
    }
    if authority_path.name == "Vtable":
        authority = authority_from_vtable(authority_path)
    else:
        authority = json.loads(io_path(authority_path).read_text(encoding="utf-8"))
        if not isinstance(authority, dict) or authority.get("schema") != AUTHORITY_SCHEMA:
            raise ValueError(f"{authority_path}: expected {AUTHORITY_SCHEMA} source layer authority")
    from woof.netcdf_bridge import NetcdfDecodeError, recover_wrf_soil
    try:
        values, receipt = recover_wrf_soil(path, met_path, authority)
    except NetcdfDecodeError as error:
        # Which remedy: a refusal about the declared layers is answered
        # by the declaration, a refusal about the conversion by the two
        # numbers the reader printed, and everything else by finding the
        # right pair of source files.
        way_out = remedy
        if any(mark in str(error) for mark in _LAYER_REFUSAL_MARKS):
            way_out = _layer_remedy(authority_path, met_path, authority, domain[1])
        elif _CONVERSION_REFUSAL_MARK in str(error):
            way_out = _conversion_remedy(authority_path, met_path, authority)
        raise ValueError(
            f"{prefix}The source layers were read from {selection['met_em']}, "
            f"under the layer authority {selection['authority']}, and they "
            f"could not be used: {error}  " + way_out) from error
    if set(values) - {"SMOIS", "SH2O"} or "SMOIS" not in values:
        raise ValueError("soil recovery did not return the declared soil-water fields")
    for name, value in values.items():
        if name not in raw or value.shape != raw[name].shape:
            raise ValueError(f"soil recovery {name} shape does not match the WRF input")
    if any(_sha(item) != hashes[name] for name, item in input_paths.items()):
        raise ValueError("soil recovery inputs changed while being read; launch again with stable inputs")
    receipt = dict(receipt, input_files={
        name: {"path": str(item), "sha256": hashes[name]}
        for name, item in input_paths.items()}, authority=authority,
        source_selection=selection)
    return values, receipt
