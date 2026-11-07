#!/usr/bin/env python3
"""Compare Rust OBS_DOMAIN decoding with WRF's exact Fortran formats.

This is a developer format check with constructed inputs. It is NOT the
required native-run module replay or a forecast qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
READER = TREE / "tools/rustwx/crates/rw-obs/src/fddaobs.rs"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wrf_source", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    wrf, out = args.wrf_source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pin = json.loads((HERE / "source-pin.json").read_text())
    for entry in pin["source_files"]:
        data = (wrf / entry["path"]).read_bytes()
        if sha(data) != entry["sha256"]:
            raise SystemExit(f"source pin mismatch: {entry['path']}")
    notice = "\n".join((wrf / "README").read_text().splitlines()[3:24]) + "\n"
    if (HERE / "WRF-NOTICE.txt").read_text() != notice:
        raise SystemExit("WRF-NOTICE.txt must retain the complete pinned public-domain notice")
    source = (wrf / "share/wrf_fddaobs_in.F").read_text()
    # Copy the FORMAT statements from the verified WRF source, retaining
    # their exact bytes. No substitute list-directed decoder is used.
    formats = []
    for label in [101, 102, 1021, 103, 104, 105]:
        match = re.search(rf"^\s*{label}\s+FORMAT\(.*$", source, re.M | re.I)
        if match is None:
            raise SystemExit(f"native FORMAT {label} missing")
        formats.append(match.group(0))
    driver = '''program native_format_probe
  use iso_fortran_env, only: int32, real32
  implicit none
  integer :: inunit, outunit, status, level, pairs, i, count
  real(real32) :: latitude, longitude, elevation, values(2,9)
  character(len=14) :: date_char
  character(len=40) :: id, namef
  character(len=16) :: platform, source
  character(len=4096) :: input_path, output_path
  logical :: is_sound, bogus
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  open(newunit=inunit,file=trim(input_path),status='old',form='formatted')
  open(newunit=outunit,file=trim(output_path),status='replace', &
       access='stream',form='unformatted')
  do
    read(inunit,101,iostat=status) date_char
    if (status < 0) exit
    if (status /= 0) stop 1
    read(inunit,102) latitude, longitude
    read(inunit,1021) id,namef
    read(inunit,103) platform,source,elevation,is_sound,bogus,count
    write(outunit) date_char,latitude,longitude,id,namef,platform,source,elevation
    write(outunit) merge(1_int32,0_int32,is_sound),merge(1_int32,0_int32,bogus),int(count,int32)
    do level=1,merge(count,1,is_sound)
      if (is_sound) then
        pairs=6
        read(inunit,104) values(:,1:pairs)
      else
        pairs=9
        read(inunit,105) values(:,1:pairs)
      end if
      write(outunit) values(:,1:pairs)
    end do
  end do
  close(inunit)
  close(outunit)
@FORMATS@
end program native_format_probe
'''.replace("@FORMATS@", "\n".join(formats))
    native_driver = out / "native_format_probe.f90"
    native_driver.write_text(driver)
    # Construct a single mixed surface/sounding file, including native
    # sentinel/QC words, signed zero, Fortran implicit decimal and exponent
    # input, PAD=YES short rows, CRLF, and no final newline.
    records, float_words = [], 0
    for report in range(12):
        sounding = report % 2 == 1
        count = 1 + report % 4 if sounding else 1
        header = (
            f" 20240229{report:02d}1530\n"
            f"  {39.125 + report / 4:9.4f} {-105.125 - report / 4:9.4f} \n"
            f"  {'station ' + str(report):40}   {'decoder fixture':40}\n"
            f"  {'FM-35 TEMP' if sounding else 'FM-12 SYNOP':16}  {'constructed':16}  "
            f"{1500.0 + report:8.0f}  {'T' if sounding else 'F':>4}  "
            f"{'T' if report == 3 else 'F':>4}  {count:5}\n"
        )
        rows = []
        for level in range(count):
            n = 6 if sounding else 9
            fields = []
            for i in range(n):
                value = (report * 17 + level * 3 + i) / 32.0
                qc = [0.0, 129.0, 256.0, -777777.0, -888888.0, 30000.0][i % 6]
                fields.extend([f"{value:11.3f}", f"{qc:11.3f}"])
            if report == 0:
                fields[:6] = ["     123456", "   12345E-2", " 1.250D+02 ",
                              "    1.25-02", " 1 2 3 . 4 ", " -000000000"]
            if report == 2:
                fields[0] = "-888888.000"
            row = " " + " ".join(fields) + " "
            # Fortran pads the final missing F11.3 with blanks => zero.
            if report == 4:
                row = row[:13]
            rows.append(row + "\n")
            float_words += 2 * n
        records.append(header + "".join(rows))
        float_words += 3
    fixture = out / "OBS_DOMAIN101.constructed"
    fixture.write_bytes("".join(records).rstrip("\n").replace("\n", "\r\n").encode("ascii"))
    commands = []

    def run(argv):
        argv = [str(x) for x in argv]
        commands.append(argv)
        result = subprocess.run(argv, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (out / "commands.log").open("a") as log:
            log.write(json.dumps(argv) + "\n" + result.stdout + f"exit={result.returncode}\n")
        if result.returncode:
            raise SystemExit(result.stdout)
        return result.stdout.strip()

    compiler_versions = {tool: run([tool, "--version"]) for tool in ["gfortran", "rustc"]}
    native = out / "native_format_probe"
    rust = out / "rust_format_probe"
    tests = out / "reader_tests"
    run(["gfortran", "-O0", "-ffp-contract=off", "-fno-unsafe-math-optimizations",
         "-fconvert=little-endian", native_driver, "-o", native])
    run(["rustc", "--edition=2024", "-C", "opt-level=2", HERE / "format_probe.rs", "-o", rust])
    run(["rustc", "--edition=2024", "--test", READER, "-o", tests])
    test_result = run([tests, "--test-threads=2"])
    a, b, c = out / "native.bin", out / "rust-one.bin", out / "rust-two.bin"
    run([native, fixture, a])
    run([rust, fixture, b, "1"])
    run([rust, fixture, c, "2"])
    ba, bb, bc = a.read_bytes(), b.read_bytes(), c.read_bytes()
    if len(ba) != len(bb) or len(ba) != len(bc):
        raise SystemExit(f"byte count mismatch: native={len(ba)} rust={len(bb)} parallel={len(bc)}")
    differences = sum(x != y for x, y in zip(ba, bb))
    parallel_differences = sum(x != y for x, y in zip(bb, bc))
    receipt = {
        "schema": "gpuwm.fddaobs-reader-format-check.v1",
        "kind": "constructed fixed-format input, not native-run physics replay",
        "wrf_tag": pin["wrf_tag"], "source_pin_sha256": sha((HERE / "source-pin.json").read_bytes()),
        "reader_sha256": sha(READER.read_bytes()), "reports": len(records),
        "float32_words": float_words, "int32_words": len(records) * 3,
        "compared_bytes_including_text": len(ba), "differing_bytes": differences,
        "parallel_differing_bytes": parallel_differences,
        "native_sha256": sha(ba), "rust_sha256": sha(bb), "fixture_sha256": sha(fixture.read_bytes()),
        "workers": [1, 2], "compiler_versions": compiler_versions,
        "unit_tests": test_result, "commands": commands,
        "physics_replay_words": 0, "forecast_run": False,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ["reports", "float32_words", "differing_bytes", "parallel_differing_bytes"]}))
    if differences or parallel_differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
