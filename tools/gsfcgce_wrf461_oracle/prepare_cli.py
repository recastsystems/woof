"""Build a table-column CLI fixture through Rust NetCDF.
Missing package masses and lateral values start at zero. Source files
remain untouched. Every public-namelist adaptation is receipted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import numpy as np
from woof.netcdf_bridge import open_dataset
from woof.io.classic_tape import ClassicTape
from woof.fortran_namelist import parse_namelist_text
from woof.ingest.wrfinput import read_wrfinput_metadata
from woof.wrfinput_door import synthesize_wps_namelist
from woof.core.mp_column_schemes import MP_COLUMN_SCHEMES
from woof.wrf_physics_inventory import stock_wrf_physics_inventory

PUBLIC_URL = "https://raw.githubusercontent.com/wrf-model/WRF/d66e442fccc04111067e29274c9f9eaccc3cef28/test/em_real/namelist.input"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def copy_inputs(source, target, *, boundary, mp_physics, attribute_copier=None,
                omitted_fields=()):
    copier = Path(attribute_copier or os.environ.get(
        "GPUWM_GSFCGCE_ATTRIBUTE_COPIER", "restore_attributes"))
    if not copier.is_file():
        raise ValueError("build restore_attributes.rs with rustc and set "
                         "GPUWM_GSFCGCE_ATTRIBUTE_COPIER to its executable; "
                         "JSON metadata alone does not preserve native attribute types")
    copier = copier.resolve()
    additions = []
    inventory = stock_wrf_physics_inventory(mp_physics)
    names = tuple(f.netcdf_name for f in inventory.wrfinput_fields)
    with open_dataset(source) as src, ClassicTape(target, container="cdf2") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        dst.setncatts(dict(src.global_attributes))
        dst.setncattr("MP_PHYSICS", np.int32(mp_physics))
        declared = {}
        for name, var in src.variables.items():
            if name in omitted_fields:
                continue
            out = dst.createVariable(name, "S1" if var.is_character else var.dtype, var.dimensions)
            out.setncatts(dict(var.attributes))
            declared[name] = (out, var)
        new_fields = {}
        for mass in names:
            suffixes = ("BXS", "BXE", "BYS", "BYE", "BTXS", "BTXE", "BTYS", "BTYE") if boundary else (None,)
            for suffix in suffixes:
                name = mass + "_" + suffix if suffix else mass
                if name in declared:
                    continue
                template = src.variables["QVAPOR_" + suffix if suffix else "QVAPOR"]
                out = dst.createVariable(name, "f4", template.dimensions)
                attrs = dict(template.attributes)
                attrs["description"] = mass + (" coupled lateral value or tendency" if boundary else " mixing ratio")
                out.setncatts(attrs)
                new_fields[name] = (out, template)
                additions.append(name)
        for out, var in declared.values():
            values = np.asarray(var[...])
            if var.dimensions and src.dimensions[var.dimensions[0]].isunlimited():
                for record in range(values.shape[0]):
                    out[record] = values[record]
            else:
                out[0] = values
        for out, template in new_fields.values():
            values = np.zeros(template.shape[1:], np.float32)
            for record in range(template.shape[0]):
                out[record] = values
    subprocess.run([str(copier), str(source), str(target), str(mp_physics),
                    *omitted_fields], check=True)
    return additions

def literal(value):
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)

def adapt_sections(public, reference, mp_physics):
    sections = parse_namelist_text(public)
    original = parse_namelist_text(public)
    for section, entries in parse_namelist_text(reference).items():
        sections.setdefault(section, {}).update(entries)
    sections.setdefault("physics", {})["mp_physics"] = [mp_physics]
    sections["physics"].pop("physics_suite", None)
    sections["physics"].update({name: [0] for name in MP_COLUMN_SCHEMES[mp_physics].switches})
    sections["time_control"]["history_interval_s"] = [60]
    sections["time_control"].pop("history_interval", None)
    sections["time_control"].pop("iofields_filename", None)
    overrides = {}
    for section, entries in sections.items():
        for name, values in entries.items():
            before = original.get(section, {}).get(name)
            if before != values:
                overrides[section + "." + name] = {"public": before, "fixture": values}
    return sections, overrides

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("reference_namelist", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mp-physics", type=int, default=97, choices=tuple(MP_COLUMN_SCHEMES))
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    public = urllib.request.urlopen(PUBLIC_URL, timeout=60).read()
    (args.output / "public-em-real.namelist.input").write_bytes(public)
    sections, overrides = adapt_sections(public.decode(), args.reference_namelist.read_text(), args.mp_physics)
    text = (chr(10)*2).join("&" + section + chr(10) + chr(10).join(
        " " + name + " = " + ", ".join(literal(v) for v in values) + ","
        for name, values in entries.items()) + chr(10) + "/" for section, entries in sections.items()) + chr(10)
    (args.output / "namelist.input").write_text(text)
    added = {}
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        added[name] = copy_inputs(args.source / name, args.output / name,
                                 boundary=name.startswith("wrfbdy"), mp_physics=args.mp_physics)
    metadata = read_wrfinput_metadata(args.output / "wrfinput_d01")
    (args.output / "namelist.wps").write_text(synthesize_wps_namelist({1: metadata}))
    receipt = {"public_namelist_url": PUBLIC_URL, "public_sha256": hashlib.sha256(public).hexdigest(),
               "adaptations": overrides, "selected_microphysics": args.mp_physics,
               "added_zero_fields": added, "source_sha256": {n: sha(args.source / n) for n in added},
               "produced_sha256": {n: sha(args.output / n) for n in added},
               "reference_namelist_sha256": sha(args.reference_namelist),
               "scope": "adapted public em_real namelist and shared real initialization; integration check"}
    args.receipt.write_text(json.dumps(receipt, indent=1) + chr(10))
    print(json.dumps({"selected_microphysics": args.mp_physics, "added_zero_fields": added}))
if __name__ == "__main__":
    main()
