"""Full MPAS driver replay; run GPU tests only inside the GPU queue.

MPAS cu_ntiedtke.F90:1-3594, module_cu_ntiedtke.F:20-351.
Synthetic input construction originates in the existing WRF column oracle;
expected words are regenerated from pristine MPAS under strict GNU flags.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof.hex.mpasphys.ntiedtke_v842 import (
    Constants, CUDA_OPTIONS, LEVEL_INPUTS, LEVEL_OUTPUTS,
    MPAS_ROUTINES, NewTiedtkeV842, cuda_source, source_receipt,
    ntiedtkeinit,
)


def test_complete_module_authority_and_strict_math():
    assert len(MPAS_ROUTINES) == 20
    source = cuda_source()
    assert "const float als = xls;" in source
    assert "float xlv, float xls, float xlf" in source
    assert "--fmad=false" in CUDA_OPTIONS
    assert "--ftz=false" in CUDA_OPTIONS
    assert "gfk_pow(meshDensity[j], 0.25f)" in source


def test_gcov_zero_counts_do_not_claim_branch_coverage(tmp_path):
    import runpy
    inventory = runpy.run_path(str(Path(__file__).parent / "ntiedtke_native/coverage.py"))["inventory"]
    (tmp_path / "cu_ntiedtke.F90.gcov").write_text(
        "1:81:subroutine cu_ntiedtke_init\n"
        "branch 0 taken 0 (fallthrough)\nbranch 1 taken 4\nbranch 2 never executed\n")
    (tmp_path / "cu_ntiedtke_post.F.gcov").write_text(
        "1:113:errmsg = 'cu_ntiedtke_post_run OK'\n"
        "branch 0 taken 0%\nbranch 1 taken 100%\n")
    receipt = tmp_path / "coverage.json"
    inventory(tmp_path, receipt)
    files = json.loads(receipt.read_text())["files"]
    assert files["cu_ntiedtke.F90.gcov"]["executed"] == 2
    assert files["cu_ntiedtke.F90.gcov"]["taken_at_least_once"] == 1
    assert [r["executed"] for r in files["cu_ntiedtke.F90.gcov"]["uncovered"]] == [True, False]
    assert files["cu_ntiedtke_post.F.gcov"]["taken_at_least_once"] == 1


def test_stored_word_proof_retains_original_and_replaced_output_inodes(tmp_path):
    import shutil
    import time
    from ntiedtke_native.preserve_outputs import OutputArchive
    source, target = tmp_path/'fresh-output', tmp_path/'retained-output'
    source.mkdir()
    output = source/'output.nc'
    output.write_bytes(b'opaque-original-words')
    with OutputArchive(source, target, interval=0.001):
        retained = target/output.name
        assert retained.stat().st_ino == output.stat().st_ino
        replacement = source/'replacement.tmp'
        replacement.write_bytes(b'opaque-completed-words')
        replacement.replace(output)
        deadline = time.monotonic()+5
        while retained.stat().st_ino != output.stat().st_ino:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        shutil.rmtree(source)
    assert retained.read_bytes() == b'opaque-completed-words'


@pytest.mark.parametrize('tamper', [None, 'native-words', 'receipt', 'restart'])
def test_full_replay_seal_rejects_mismatched_evidence(tmp_path, tamper):
    import hashlib
    from ntiedtke_native.summarize_replay import summarize
    def write(path, value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    settings = [{'variant': 0, 'hfx_word': '42c80000'}]
    settings_hash = write(tmp_path/'settings.json', {'complete': True, 'settings': settings})
    for mode, execution in (('host', 'host-glibc-witness'), ('cuda', 'hex-cuda'),
                            ('native-repeat', 'strict-native-repeat')):
        directory = tmp_path/mode
        directory.mkdir()
        row = dict(index=0, setting=settings[0], native_csv_sha256={
            name: '1'*64 for name in ('nt-levels.csv', 'nt-surface.csv', 'nt-isolation.csv')})
        if mode == 'native-repeat':
            row['identical'] = True
            if tamper == 'native-words':
                row['native_csv_sha256']['nt-levels.csv'] = '2'*64
        else:
            receipt = dict(execution=execution, differing_words=0,
                           repeat_differing_words=0, restart_counter_one_checked=True,
                           native_spread_or_forecast_proof=False,
                           results=[dict(arrays={'rainc': dict(words=18, differing_words=0)})])
            if tamper == 'restart' and mode == 'cuda':
                receipt['restart_counter_one_checked'] = False
            receipt_hash = write(directory/'0.json', receipt)
            row.update(sha256=receipt_hash, compared_words=18, differing_words=0)
            if tamper == 'receipt' and mode == 'cuda':
                (directory/'0.json').write_text('{}')
        write(directory/'matrix.json', dict(complete=True, passed=True,
              native_process_count=1, execution=execution, settings_sha256=settings_hash,
              requested=1, runs=[row], native_binary_sha256='3'*64))
    if tamper:
        with pytest.raises(ValueError):
            summarize(tmp_path)
    else:
        result = summarize(tmp_path)
        assert result['cuda_words'] == result['host_words'] == 18
        assert result['passed'] and result['complete']
        assert not result['acceptance'] and not result['forecast_proof']
        assert not result['every_branch_covered']


def lifecycle(directory: Path, receipt: Path, *, host_library: Path | None = None):
    """Compare native initializer and driver arithmetic with real compiled code."""
    if host_library is None:
        import cupy as cp
        from woof.hex.mpasphys.ntiedtke_v842 import convection_from_MPAS
    else:
        cp = np
        from ntiedtke_native.host import HostPipeline
    names = LEVEL_OUTPUTS+("rthften","rqvften")
    mismatches = 0
    rows = list(csv.DictReader((directory/"lifecycle.csv").open()))
    for mode in (0,1,2,3):
        target = {n:cp.full((3,2),0.125,dtype=cp.float32) for n in names}
        ntiedtkeinit(target,restart=mode==1,p_qc=0 if mode==2 else 2,p_qi=0 if mode==3 else 2,p_first_scalar=1)
        for row in rows:
            if int(row['mode'])!=mode: continue
            a = target[names[int(row['field'])-1]]
            a = cp.asnumpy(a) if host_library is None else a
            mismatches += int(a.view(np.uint32)[int(row['k'])-1,int(row['i'])-1]!=int(row['word'],16))
    rows = list(csv.DictReader((directory/"driver-slice.csv").open()))
    def field(n): return cp.asarray(np.array([_word(r[n]) for r in rows],dtype=np.float32))
    forcing = dict(zip(("rthdynten","rthblten","rthratenlw","rthratensw","exner","rqvdynten","rqvblten"),
                      (field(n).reshape(1,-1) for n in ('rt','rb','lw','sw','pi','qd','qb'))))
    if host_library is None:
        fields = {n:cp.zeros((1,7),dtype=cp.float32) for n in LEVEL_INPUTS}
        fields.update({n:cp.zeros((2,7),dtype=cp.float32) for n in ("p8w","w")})
        fields.update({n:cp.zeros(7,dtype=cp.float32) for n in ('hfx','qfx','xland')})
        fields.update(forcing,meshDensity=field('density'))
        out = convection_from_MPAS(fields,len_disp=120000.)
    else:
        p = HostPipeline(host_library,7,4,60.,1,0,Constants())
        out = {n:np.empty(7,dtype=np.float32) for n in ('thften','qvften','dx')}
        p._call('mpas_ntiedtke_forcing',list(forcing.values())+[out['thften'],out['qvften'],7])
        p._call('mpas_ntiedtke_dx',[field('density'),120000.,out['dx'],7])
    arrays = {}
    for name,native in (('thften','th'),('qvften','qv'),('dx','dx')):
        a = cp.asnumpy(out[name]) if host_library is None else out[name]
        w = np.array([int(r[native],16) for r in rows],dtype=np.uint32)
        bad = int(np.count_nonzero(a.view(np.uint32).reshape(-1)!=w))
        arrays[name] = bad
        mismatches += bad
    result = dict(differing_words=mismatches,driver_arrays=arrays,execution='hex-cuda' if host_library is None else 'host-glibc-witness')
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    return result


def _word(value):
    return np.array(int(value, 16), dtype=np.uint32).view(np.float32).item()


def load_native(directory: Path):
    """Load synthetic test inputs and raw oracle words, never model data."""
    columns = {}
    with (directory / "nt-levels.csv").open() as f:
        for row in csv.DictReader(f):
            key = (int(row["case"]), float(row["dx"]))
            columns.setdefault(key, []).append(row)
    surface = {}
    with (directory / "nt-surface.csv").open() as f:
        for row in csv.DictReader(f):
            surface[(int(row["case"]), float(row["dx"]))] = row
    return columns, surface


def replay(directory: Path, receipt_path: Path, *, host_library: Path | None = None,
           isolated: bool = False, launcher=None):
    """Compare every output array of the full public MPAS driver."""
    if host_library is None:
        import cupy as cp
    else:
        cp = np
        from ntiedtke_native.host import HostTiedtke
    columns, surface = load_native(directory)
    settings_path = directory / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {"grav": Constants().grav, "initflag": 0, "stepcu": 1, "flag_mode": 0}
    results = []
    groups = [[key] for key in sorted(columns)] if isolated else [
        sorted(key for key in columns if key[1] == dx)
        for dx in sorted({key[1] for key in columns})]
    if isolated:
        with (directory / "nt-isolation.csv").open() as f:
            assert all(int(r["differing_words"]) == 0 for r in csv.DictReader(f)), "native isolation differs; capture isolated expected words"
    constants = Constants(grav=settings["grav"])
    port = launcher or (NewTiedtkeV842(constants=constants) if host_library is None else HostTiedtke(host_library, constants=constants))
    repeated_words = 0
    for keys in groups:
        dx = keys[0][1]
        nz = len(columns[keys[0]]) - 1
        fields = {}
        for name in LEVEL_INPUTS + ("p8w", "w"):
            source_name = "pi3d" if name == "exner" else name
            rows = nz + 1 if name in ("p8w", "w") else nz
            a = np.array([[_word(columns[key][k][source_name]) for key in keys]
                          for k in range(rows)], dtype=np.float32)
            fields[name] = cp.asarray(a)
        for name in ("xland", "hfx", "qfx"):
            fields[name] = cp.asarray(np.array([_word(surface[key][name]) for key in keys], dtype=np.float32))
        fields["dx"] = cp.full(len(keys), dx, dtype=cp.float32)
        def invoke(restart=False):
            return port(fields, dt=_word(settings["dt_word"]) if "dt_word" in settings else 60.0,
            itimestep=1 if restart or settings["initflag"] == 1 else 2,
            restart=restart,
            stepcu=settings["stepcu"], f_qc=settings["flag_mode"] not in (1,3),
            f_qi=settings["flag_mode"] not in (2,3),
            previous={"rqccuten": cp.full((nz, len(keys)), 0.0003, dtype=cp.float32),
                      "rqicuten": cp.full((nz, len(keys)), -0.0002, dtype=cp.float32)})
        got = invoke()
        repeated = invoke()
        # A restarted model may begin with counter 1, while native must
        # retain the supplied forcing (driver_convection.F:476-478).
        restarted = invoke(restart=True) if settings["initflag"] == 0 else None
        for name in got:
            first = cp.asnumpy(got[name]) if host_library is None else got[name]
            second = cp.asnumpy(repeated[name]) if host_library is None else repeated[name]
            if first.dtype == np.float32:
                first, second = first.view(np.uint32), second.view(np.uint32)
            repeated_words += int(np.count_nonzero(first != second))
            if restarted is not None:
                restart_value = cp.asnumpy(restarted[name]) if host_library is None else restarted[name]
                if restart_value.dtype == np.float32:
                    restart_value = restart_value.view(np.uint32)
                repeated_words += int(np.count_nonzero(first != restart_value))
        arrays = {}
        for name in LEVEL_OUTPUTS + ("raincv", "pratec", "cu_act_flag"):
            if name in LEVEL_OUTPUTS:
                want = np.array([[_word(columns[key][k][name]) for key in keys]
                                 for k in range(nz)], dtype=np.float32)
            elif name == "cu_act_flag":
                want = np.array([surface[key][name].strip() in ("T", "1") for key in keys])
            else:
                want = np.array([_word(surface[key][name]) for key in keys], dtype=np.float32)
            actual = cp.asnumpy(got[name]) if host_library is None else got[name]
            a = actual.view(np.uint32) if actual.dtype == np.float32 else actual
            w = want.view(np.uint32) if want.dtype == np.float32 else want
            mismatch = np.argwhere(a != w)
            samples = [{"index": ix.tolist(), "got": int(a[tuple(ix)]),
                        "native": int(w[tuple(ix)])} for ix in mismatch[:5]]
            arrays[name] = {"words": int(a.size), "differing_words": int(len(mismatch)), "samples": samples}
        results.append({"dx": dx, "cases": [key[0] for key in keys],
                        "columns": len(keys), "levels": nz, "arrays": arrays})
        if host_library is None:
            for _, pipeline in port._cached.values():
                pipeline.stages.check_geometry()
    receipt = {"schema": "mpas-ntiedtke-standalone-replay/v1", "source": source_receipt(),
               "native_host": settings.get("native_host", "reference-host"), "execution": "hex-cuda" if host_library is None else "host-glibc-witness",
               "isolated": isolated, "repeat_differing_words": repeated_words,
               "repeat_invocations_per_group": 2,
               "restart_counter_one_checked": settings["initflag"] == 0,
               "native_spread_or_forecast_proof": False,
               "results": results, "differing_words": repeated_words + sum(a["differing_words"] for r in results for a in r["arrays"].values())}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


@pytest.mark.skipif(os.environ.get("MPAS_NTIEDTKE_GPU") != "1", reason="requires the GPU queue and standalone native fixture")
def test_every_native_output_word():
    fixture = Path(os.environ["MPAS_NTIEDTKE_ORACLE"])
    result = replay(fixture, Path(os.environ["MPAS_NTIEDTKE_RECEIPT"]))
    assert result["differing_words"] == 0


@pytest.mark.skipif(os.environ.get("MPAS_NTIEDTKE_GPU") != "1", reason="requires the GPU queue")
def test_exceptional_timestep_words_across_cached_calls(tmp_path):
    """cu_ntiedtke_pre.F:105; all strict timestep fixtures, one launcher.

    In particular, signed zero inputs must never share cached scalar words.
    Every expected array comes from the independent strict Fortran driver.
    """
    directory = os.environ.get("MPAS_NT_TIMESTEP_ORACLE")
    if not directory:
        pytest.skip("requires complete strict native timestep fixture matrix")
    root = Path(directory)
    fixtures = sorted((p for p in root.iterdir() if p.name.isdecimal()),
                      key=lambda p: int(p.name))
    assert len(fixtures) == 50
    assert {json.loads((p / "settings.json").read_text())["dt_word"]
            for p in fixtures} == {
                "42700000", "c2700000", "00000000", "80000000",
                "00000001", "80000001", "7f800000", "ff800000",
                "7fc00000", "ffc00000"}
    launcher = NewTiedtkeV842()
    for fixture in fixtures:
        result = replay(fixture, tmp_path / (fixture.name + ".json"),
                        launcher=launcher)
        assert result["differing_words"] == 0, fixture.name


@pytest.mark.skipif(os.environ.get("MPAS_NTIEDTKE_GPU") != "1", reason="requires the GPU queue")
def test_driver_preserves_subnormal_input_words():
    """mpas_atmphys_driver_convection.F:827-850; detect forced FTZ."""
    import cupy as cp
    from woof.hex.mpasphys.ntiedtke_v842 import _module
    words = np.array([1, 2, 0x007fffff, 0x00800000,
                      0x80000001, 0x807fffff], dtype=np.uint32)
    thermal = cp.asarray(words.view(np.float32))
    vapor_words = words ^ np.uint32(0x80000000)
    vapor = cp.asarray(vapor_words.view(np.float32))
    zero = cp.asarray(np.zeros(words.size, dtype=np.float32))
    one = cp.asarray(np.ones(words.size, dtype=np.float32))
    th, qv = cp.empty_like(thermal), cp.empty_like(thermal)
    _module(cp.cuda.Device().id).get_function('mpas_ntiedtke_forcing')(
        (1,), (128,), (thermal, zero, zero, zero, one, vapor, zero,
                      th, qv, np.int32(words.size)))
    actual = (th.get().view(np.uint32), qv.get().view(np.uint32))
    differing = sum(int(np.count_nonzero(a != b)) for a, b in
                    zip(actual, (words, vapor_words)))
    result = dict(schema='mpas-ntiedtke-subnormal-input/v1',
                  source=source_receipt(), words=int(2*words.size),
                  differing_words=differing,
                  input_words=words.tolist(),
                  thermal_words=actual[0].tolist(), vapor_words=actual[1].tolist())
    if os.environ.get('MPAS_NT_SUBNORMAL_RECEIPT'):
        Path(os.environ['MPAS_NT_SUBNORMAL_RECEIPT']).write_text(json.dumps(result,indent=2)+'\n')
    assert differing == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--host-library", type=Path)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    if args.batch:
        args.receipt.mkdir(parents=True, exist_ok=True)
        runs = {}
        if args.host_library is None:
            os.environ['MPAS_NT_SUBNORMAL_RECEIPT'] = str(args.receipt / 'subnormal-input.json')
            test_driver_preserves_subnormal_input_words()
            runs['subnormal-input'] = 0
            launcher = NewTiedtkeV842()
        else:
            from ntiedtke_native.host import HostTiedtke
            launcher = HostTiedtke(args.host_library)
        fixtures = sorted(args.oracle.glob("oracle-*")) + sorted(args.oracle.glob("stress-*"))
        for fixture in fixtures:
            if not fixture.is_dir():
                continue
            result = replay(fixture, args.receipt / (fixture.name + ".json"),
                            host_library=args.host_library, launcher=launcher)
            runs[fixture.name] = result["differing_words"]
        for name, result in (
            ("isolated", replay(args.oracle / "oracle-0-1-0", args.receipt / "isolated.json",
                                host_library=args.host_library, isolated=True, launcher=launcher)),
            ("lifecycle", lifecycle(args.oracle, args.receipt / "lifecycle.json",
                                    host_library=args.host_library)),
        ):
            runs[name] = result["differing_words"]
        if (args.oracle / "branch-flux.csv").exists():
            from ntiedtke_native.branch_fixture import compare
            result = compare(args.oracle / "branch-flux.csv",
                             args.receipt / "branch-flux.json", args.host_library)
            runs["auxiliary-flux-branches"] = result["differing_words"]
        (args.receipt / "batch.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(json.dumps(runs))
        raise SystemExit(0 if runs and not any(runs.values()) else 1)
    result = lifecycle(args.oracle,args.receipt,host_library=args.host_library) if args.lifecycle else replay(args.oracle, args.receipt, host_library=args.host_library, isolated=args.isolated)
    print(json.dumps({"differing_words": result["differing_words"]}))
    raise SystemExit(0 if result["differing_words"] == 0 else 1)
