"""Replay the complete MPAS call, comparing every output word without tolerance.

Synthetic inputs are test fixtures, not forecast data processing. The pristine
Fortran module is the oracle; the mechanically translated host-glibc module is
a separate witness. This test cannot substitute for CUDA or forecast proof.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare import DIAGNOSTICS, ENVIRONMENT, STATE, SURFACE, THIRD_MOMENTS, load_target
import translate

F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
U64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def fixture(nc=12, nz=55, moments=2):
    """Cold/warm, saturation, zero mass, mixed phase and variable densities."""
    state = np.zeros((20 if moments==3 else len(STATE), nz, nc), dtype=np.float32)
    env = np.zeros((len(ENVIRONMENT), nz, nc), dtype=np.float32)
    surface = np.zeros((len(SURFACE), nc), dtype=np.float32)
    diagnostics = np.full((len(DIAGNOSTICS), nz, nc), -777, dtype=np.float32)
    k = np.arange(nz, dtype=np.float32)[:, None]
    c = np.arange(nc, dtype=np.float32)[None, :]
    temperature = np.clip(305 - k*2.5 - (c % 4)*8, 180, 315).astype(np.float32)
    env[1] = np.broadcast_to(95000*np.exp(-k/11), (nz, nc))
    env[2] = (env[1]/100000)**np.float32(0.2857143)
    env[0] = env[1]/(np.float32(287.0)*temperature)
    env[3] = 150 + k*45
    env[4] = np.sin(k/4)*((c%3)-1)*12
    state[0] = temperature/env[2]
    # The fixture straddles both saturation surfaces and reaches cold nuclei.
    es = np.float32(610.78)*np.exp(np.float32(17.269388)*(temperature-273.15)/(temperature-35.86))
    state[1] = np.float32(0.622)*es/env[1] * (np.float32(0.6) + (c%4)*np.float32(0.3))
    for species in range(6):
        field = 2+species
        mass = np.float32(0.00015)*(1+((k+species)%7))
        state[field] = np.where(((c+species)%4)!=0, mass, 0)
        # Mixing ratio number moments, kg^-1. Cloud droplets are smaller.
        mean_mass = np.float32(4e-12 if species==0 else 2e-8 if species==1 else 1e-9 if species==2 else 2e-7)
        state[8+species] = state[field]/mean_mass
    state[14] = np.broadcast_to((c%3)*1e8, (nz,nc))
    state[15] = state[6]/(np.float32(200)+(c%4)*np.float32(200))
    state[16] = state[7]/np.float32(700)
    if moments == 3:
        for third, mass, number, density in ((17,3,9,1000),(18,6,12,500),(19,7,13,900)):
            volume = state[mass]*np.float32(6/(np.pi*density))
            state[third] = volume*volume/np.maximum(state[number],np.float32(1e-20))
    # A completely dry column checks preservation of zeros and initialization.
    state[1:, :, 0] = 0
    return dict(state=state, environment=env, surface=surface,
                diagnostics=diagnostics, refl1km=np.full(nc,-999,np.float32),
                rainprod=np.zeros((nz,nc),np.float32),
                evapprod=np.zeros((nz,nc),np.float32))


def r1_fixture(name):
    """Literal r1 synthetic profiles; no native dump or weather-field decoder.

    These match r1's ipctmp=8, 1x55 fixtures, including zero third moments.
    The first call matches its dtp=10/itimestep=1 profile. Later calls evolve
    that complete state rather than resetting it.
    """
    values = dict(th=290, qv=0.001, cn=1e8, dn=1.2, p=100000, pii=1, dz=200)
    if name == "warm_rain":
        values.update(qv=0.015, qc=0.001, qr=0.002, ccw=1e8, crw=1e6)
    elif name == "mixed_ice":
        values.update(th=253, qc=0.0001, qi=0.0001, qs=0.001, qh=0.002,
                      qhl=0.001, ccw=1e8, cci=1e6, csw=1e5, chw=1e4,
                      chl=1e4, vhw=4.8e-6, vhl=2.4e-6)
    elif name != "dry":
        raise ValueError(name)
    data = dict(state=np.zeros((20,55,1),np.float32),
                environment=np.zeros((5,55,1),np.float32),
                surface=np.zeros((7,1),np.float32),
                diagnostics=np.zeros((4,55,1),np.float32),
                refl1km=np.zeros(1,np.float32), rainprod=np.zeros((55,1),np.float32),
                evapprod=np.zeros((55,1),np.float32))
    for buffer, names in (("state", STATE+THIRD_MOMENTS), ("environment", ENVIRONMENT)):
        for index, field in enumerate(names):
            data[buffer][index].fill(np.float32(values.get(field,0)))
    return data


class Native:
    def __init__(self, path, moments=2):
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.rrfs_nssl2m_initialize.argtypes = [ctypes.c_int]
        self.lib.rrfs_nssl2m_initialize.restype = None
        self.lib.rrfs_nssl2m_step.argtypes = [ctypes.c_int]*9 + [ctypes.c_float] + [F32]*7
        self.lib.rrfs_nssl2m_step.restype = None
        self.lib.rrfs_nssl2m_initialize(moments)

    def run(self, data, *, nt, nmicro, diagnose, dbz_due, reqc, reqi, reqs, dt):
        nc, nz = data["state"].shape[2], data["state"].shape[1]
        self.lib.rrfs_nssl2m_step(nc,nz,nt,nmicro,diagnose,dbz_due,reqc,reqi,reqs,dt,
            *(data[n] for n in ("state","environment","surface","diagnostics","refl1km","rainprod","evapprod")))


class Host:
    def __init__(self, path, overrides=None, moments=2):
        self.lib = ctypes.CDLL(str(path.resolve()))
        self.lib.nssl_host_parameter_bytes.restype = ctypes.c_ulonglong
        self.parameters = np.empty(self.lib.nssl_host_parameter_bytes(), np.uint8)
        self.storage = np.empty(128*1024*1024, np.uint8)
        self.used = np.zeros(1,np.uint64)
        self.errors = np.zeros(1,np.int32)
        fields = load_target().GLOBAL_SCALARS
        overrides = overrides or {}
        indices = np.array([fields.index(n) for n in overrides],np.int32)
        values = np.array(list(overrides.values()),np.float64)
        self.lib.nssl_initialize_body.argtypes = [U8,U8,U64,F64,I32,ctypes.c_int,ctypes.c_int,I32]
        self.lib.nssl_initialize_body.restype = None
        self.lib.nssl_initialize_body(self.parameters,self.storage,self.used,values,indices,moments,len(values),self.errors)
        assert not self.errors.any(), self.errors
        self.lib.nssl_step_body.argtypes = [ctypes.c_int]*11 + [ctypes.c_float] + [F32]*7 + [U8,U8,ctypes.c_ulonglong,I32]
        self.lib.nssl_step_body.restype = None

    def run(self, data, *, nt, nmicro, diagnose, dbz_due, reqc, reqi, reqs, dt):
        nc, nz = data["state"].shape[2], data["state"].shape[1]
        # All thread-local automatic arrays are allocated through an arena.
        # One host column at a time needs just one allocation, reused below.
        stride = 65536 + 10240*nz
        workspace = np.empty(32*stride,np.uint8)
        errors = np.zeros(nc,np.int32)
        for active in range(nc):
            # Column zero reuses one warp arena; retain its physical stride
            # so the narrow and double bank boundaries match the GPU.
            self.lib.nssl_step_body(active,0,nc,nz,nt,nmicro,diagnose,dbz_due,reqc,reqi,reqs,dt,
                *(data[n] for n in ("state","environment","surface","diagnostics","refl1km","rainprod","evapprod")),
                self.parameters,workspace,stride,errors)
        assert not errors.any(), errors


def compare(reference, candidate):
    result = {}
    for name, expected in reference.items():
        actual = candidate[name]
        words = expected.view(np.uint32)
        differing = words != actual.view(np.uint32)
        indices = np.argwhere(differing)
        witnesses = []
        for index in indices[:8]:
            ix = tuple(int(i) for i in index)
            witnesses.append(dict(index=ix, native_word=f"{int(words[ix]):08x}",
                                  candidate_word=f"{int(actual.view(np.uint32)[ix]):08x}"))
        result[name] = dict(words=int(words.size), differing_words=int(differing.sum()),
                            witnesses=witnesses,
                            native_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                            candidate_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    return result


def compare_initialization(native, host, native_path, source):
    """Inspect compiler-owned initialized storage, including private tables.

    ELF names, offsets and sizes are symbol metadata. No pristine source is
    patched or private checkpoint substituted. This is an oracle diagnostic,
    and the resulting word counts are distinct from complete-call proof.
    """
    symbols = {}
    prefix = "__module_mp_nssl_2mom_MOD_"
    listing = subprocess.check_output(["nm", "-S", "--defined-only", str(native_path)], text=True)
    for line in listing.splitlines():
        row = line.split()
        if len(row) == 4 and row[3].startswith(prefix):
            symbols[row[3][len(prefix):]] = (int(row[0], 16), int(row[1], 16))
    anchor = ctypes.cast(getattr(native.lib, prefix + "nssl_2mom_init"), ctypes.c_void_p).value
    base = anchor - symbols["nssl_2mom_init"][0]
    get = host.lib.nssl_host_variable
    get.argtypes = [U8, ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int)]
    get.restype = ctypes.c_void_p
    variables, _ = translate.parse_modules([source])
    result = {}
    for index, var in enumerate(variables.values()):
        if var.parameter or var.kind not in ("float", "double", "int"):
            continue
        count, width = ctypes.c_ulonglong(), ctypes.c_int()
        pointer = get(host.parameters, index, ctypes.byref(count), ctypes.byref(width))
        offset, size = symbols[var.name]
        assert size == count.value * width.value, (var.name, size, count.value, width.value)
        dtype = np.uint64 if width.value == 8 else np.uint32
        expected = np.frombuffer((ctypes.c_ubyte * size).from_address(base + offset), dtype=dtype)
        actual = np.frombuffer((ctypes.c_ubyte * size).from_address(pointer), dtype=dtype)
        differing = np.flatnonzero(expected != actual)
        result[var.name] = dict(words=int(count.value), differing_words=int(differing.size),
                               witnesses=[dict(index=int(i), native_word=f"{int(expected[i]):0{width.value*2}x}",
                                               candidate_word=f"{int(actual[i]):0{width.value*2}x}") for i in differing[:8]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--moments", type=int, choices=(2,3), default=2)
    parser.add_argument("--overrides", type=Path, help="Native nssl_mp_params configuration JSON")
    parser.add_argument("--options", type=Path, help="MPAS call flags and time-step configuration JSON")
    parser.add_argument("--r1-profile", choices=("dry","warm_rain","mixed_ice"))
    args = parser.parse_args()
    if args.r1_profile and args.moments != 3:
        parser.error("r1 NSSL synthetic profiles initialize ipctmp=8, requiring --moments 3")
    native_path, host_path = args.native.resolve(), args.host.resolve()
    source = args.source.resolve() if args.source else None
    overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    call_options = json.loads(args.options.read_text()) if args.options else {}
    args.scratch.mkdir(parents=True,exist_ok=True)
    receipt_path = args.receipt.resolve()
    nml = []
    for name,value in overrides.items():
        assert name in load_target().NSSL_NAMELIST_FIELDS
        nml.append(name + " = " + ((".true." if value else ".false.") if type(value) is bool else repr(value)))
    (args.scratch/"namelist.atmosphere").write_text("&nssl_mp_params\n" + "\n".join(nml) + "\n/\n")
    os.chdir(args.scratch)
    native, host = Native(native_path,args.moments), Host(host_path,overrides,args.moments)
    initialized = compare_initialization(native, host, native_path, source) if source else None
    current = r1_fixture(args.r1_profile) if args.r1_profile else fixture(moments=args.moments)
    candidate = {name: array.copy() for name,array in current.items()}
    attempts = []
    for step in (1,2,3):
        options = dict(nt=step,nmicro=1,diagnose=1,dbz_due=1,reqc=1,reqi=1,reqs=1,dt=6.0)
        if args.r1_profile:
            options["dt"] = 10.0
        options.update(call_options)
        native.run(current,**options)
        host.run(candidate,**options)
        attempts.append(dict(options=options, arrays=compare(current,candidate)))
    result = dict(scope="synthetic whole-module host-glibc witness; no CUDA or forecast claim",
                  libc=platform.libc_ver(),source_sha256=load_target().SOURCE_SHA256,
                  native_library_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                  host_library_sha256=hashlib.sha256(host_path.read_bytes()).hexdigest(),
                  moments=args.moments, r1_profile=args.r1_profile, nssl_mp_params=overrides,
                  r1_profile_provenance=(json.loads((ROOT/"r1-profile-provenance.json").read_text())[args.r1_profile] if args.r1_profile else None),
                  initialization=initialized, attempts=attempts)
    receipt_path.write_text(json.dumps(result,indent=2)+"\n")
    failures = sum(a["differing_words"] for r in attempts for a in r["arrays"].values())
    if initialized:
        failures += sum(row["differing_words"] for row in initialized.values())
    print(f"Host witness: {failures} differing words in {len(attempts)} complete MPAS calls")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
