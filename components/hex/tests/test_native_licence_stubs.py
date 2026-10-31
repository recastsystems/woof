"""The default native library contains no code that cannot be sold; the opt-in builds it, by name.

THE BREAKAGE THIS PREVENTS (owner ruling 2026-10-14: no default routes into code that cannot be sold).
MPAS-A v8.4.2's own sources USE Noah-MP v5.0 (UCAR terms: no sale of it or of any work that contains it):
mpas_atmphys_vars.F:12 and the Noah-MP driver, init and finalize files.  So `woof hex physics-resources
build` compiled it into every install's native framework library, for runs that never select it.  The
default build now compiles a licence stub in its place.  (UGWP and the other NOAA-GSL physics stay compiled:
owner ruling 2026-10-14, US Government code, not held.)  These checks hold that:

1. the default build stubs Noah-MP v5.0 and compiles UGWP, and only ``--licence-opt-in ucar-noahmp``
   compiles Noah-MP;
2. a stubbed component is neither fetched nor compiled, and the opted-in build hands the builder exactly the
   arguments it had before stubs existed (measured with the strict reference toolchain, the opted-in build
   compiled a library byte-identical to the pre-stub builder's, with every generated file identical);
3. the stubs carry only the interface MPAS's own sources USE: every routine stops the run naming the opt-in
   and none returns;
4. the build flavour reaches the build receipt and the installed record;
5. an opted-in run on a stubbed library is refused by name before any work.

CPU only: no compiler, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest

from woof.hex import licence_gate
from woof.hex.mpasphys.native_library_build import referee_builder

nlb = referee_builder()
from woof.hex.mpasphys import resources_build as rb
from woof.hex.mpasphys.driver_v842 import resolve_namelist

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "tools/native_physics_reference/backend"
NOAHMP_DIR = "src/core_atmosphere/physics/physics_noahmp"
UGWP_DIR = "src/core_atmosphere/physics/physics_noaa/UGWP"
# The interface MPAS-A v8.4.2's own sources USE (make_licence_stubs.py derives it from those sources).
INTERFACE = {
    "noahmp": {"NoahmpIOVarType": [], "NoahmpDriverMainMod": ["NoahmpDriverMain"],
               "NoahmpInitMainMod": ["NoahmpInitMain"], "NoahmpIOVarFinalizeMod": ["NoahmpIOVarFinalizeDefault"],
               "NoahmpIOVarInitMod": ["NoahmpIOVarInitDefault"], "NoahmpReadNamelistMod": [],
               "NoahmpReadTableMod": ["NoahmpReadTable"]},
}
TERMS = {"noahmp": "ucar-noahmp"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _builder():
    spec = importlib.util.spec_from_file_location("native_backend_build", BACKEND / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 1. the default stubs Noah-MP
def test_the_default_build_stubs_noahmp_and_only_its_named_opt_in_compiles_it():
    assert nlb.stubbed_components() == ["noahmp"]
    assert nlb.stubbed_components(["ucar-noahmp"]) == []
    # another licence term does not touch the library: UGWP is compiled in every flavour
    assert nlb.stubbed_components(["noaa-gsl-unlicensed"]) == ["noahmp"]
    assert nlb.stubbed_components(["ugwp-held"]) == ["noahmp"]
    assert set(nlb.LICENCE_COMPONENTS) == {"ucar-noahmp"}


def test_the_builder_and_the_python_layer_name_the_same_components_and_terms():
    builder = _builder()
    assert {name: row["terms"] for name, row in builder.LICENCE_STUBS.items()} == TERMS
    assert {component: directory for component, directory in nlb.LICENCE_COMPONENTS.values()} == {
        name: row["replaces"] for name, row in builder.LICENCE_STUBS.items()}
    for name, row in builder.LICENCE_STUBS.items():
        assert row["file"] in nlb.BACKEND and (BACKEND / row["file"]).is_file()
    # The licence gate's terms are the opt-ins the build accepts.
    assert set(nlb.LICENCE_COMPONENTS) <= set(licence_gate.known_terms())


# ---------------------------------------------------------------- 2. what the builder is handed
@pytest.fixture
def fake_stack(tmp_path, monkeypatch):
    """build_native_library with every host step stood in; records what build.py would be run with."""
    mpich = tmp_path / "mpich"
    (mpich / "bin").mkdir(parents=True)
    for tool in ("mpif90", "mpicc"):
        (mpich / "bin" / tool).write_text("#!/bin/sh\n")
    pnetcdf = tmp_path / "pnetcdf"
    (pnetcdf / "lib").mkdir(parents=True)
    (pnetcdf / "lib/libpnetcdf.a").write_bytes(b"archive")
    calls = {"run": [], "skip": []}
    monkeypatch.setattr(nlb, "check_prerequisites", lambda: None)
    monkeypatch.setattr(nlb, "toolchain", lambda: {"gfortran": "GNU Fortran 15.2", "gcc": "gcc 15.2"})
    monkeypatch.setattr(nlb, "build_pnetcdf", lambda *a, **k: pnetcdf)

    def fetch_tree(cache, fetch_file, *, mpas_source, offline, log, skip=(), pin_set=None):
        calls["skip"].append(sorted(skip))
        return tmp_path / "tree"

    def run(command, *, cwd, log, env=None):
        calls["run"].append(command)
        work = Path(command[command.index("--work") + 1])
        stubs = [command[i + 1] for i, part in enumerate(command) if part == "--licence-stub"]
        library = work / nlb.LIBRARY
        library.write_bytes(b"\x7fELF " + ",".join(stubs).encode())
        (work / "abi.json").write_text("{}")
        rows = {name: {"terms": TERMS[name], "stub_source_sha256": "0" * 64} for name in stubs}
        (work / "build-receipt.json").write_text(json.dumps({
            "status": "built", "library_sha256": _sha(library), "abi_sha256": _sha(work / "abi.json"),
            "licence": {"stubbed": rows, "compiled": {n: {"terms": t} for n, t in TERMS.items() if n not in stubs}}}))

    monkeypatch.setattr(nlb, "fetch_mpas_tree", fetch_tree)
    monkeypatch.setattr(nlb, "_run", run)
    return tmp_path, mpich, calls


def _build(tmp_path, mpich, opt_in=()):
    return nlb.build_native_library(tmp_path / "cache", fetch=None, fetch_file=None, mpich=mpich, jobs=3,
                                    licence_opt_in=opt_in, log=lambda message: None)


def test_the_default_build_hands_the_builder_the_noahmp_stub_and_does_not_fetch_it(fake_stack):
    tmp_path, mpich, calls = fake_stack
    row = _build(tmp_path, mpich)
    command = calls["run"][-1]
    assert [command[i + 1] for i, part in enumerate(command) if part == "--licence-stub"] == ["noahmp"]
    assert calls["skip"][-1] == [NOAHMP_DIR]
    assert row["licence"]["flavour"] == "stubbed" and set(row["licence"]["stubbed"]) == {"noahmp"}
    assert row["licence"]["opt_in"] == []


def test_the_opted_in_build_runs_the_builder_exactly_as_before_stubs(fake_stack):
    tmp_path, mpich, calls = fake_stack
    stubbed = Path(_build(tmp_path, mpich)["path"])
    row = _build(tmp_path, mpich, ["ucar-noahmp"])
    command = calls["run"][-1]
    # The pre-stub argument list, part for part: no --licence-stub, nothing added.
    assert command[2:] == ["--source", str(tmp_path / "tree"), "--work", command[5],
                           "--pnetcdf", str(tmp_path / "pnetcdf"), "--source-manifest", str(nlb.source_manifest_path()),
                           "--no-oracle-transport", "--jobs", "3"]
    assert calls["skip"][-1] == []
    assert row["licence"]["flavour"] == "opted-in"
    assert row["licence"]["opt_in"] == ["ucar-noahmp"]
    # Each flavour builds in its own work tree and stages its own bytes.
    assert calls["run"][0][5] != command[5] and Path(row["path"]) != stubbed


def test_an_unknown_native_opt_in_is_refused(fake_stack):
    tmp_path, mpich, _ = fake_stack
    with pytest.raises(nlb.NativeLibraryError, match="unknown licence opt-in mynn"):
        _build(tmp_path, mpich, ["mynn"])


def test_a_stubbed_component_is_never_fetched(tmp_path):
    requested = []

    def fetch_file(relative, origin, path, sha256, target, *, mpas_source, offline):
        requested.append(relative)
        return target

    nlb.fetch_mpas_tree(tmp_path, fetch_file, mpas_source=None, offline=True, log=lambda m: None,
                        skip=[NOAHMP_DIR])
    assert requested and not [r for r in requested if r.startswith(NOAHMP_DIR + "/")]
    # MPAS's own Noah-MP driver files, which USE the stubbed modules, are still fetched and compiled, and so
    # is UGWP (compiled in every flavour).
    assert "src/core_atmosphere/physics/mpas_atmphys_driver_lsm_noahmp.F" in requested
    assert any(r.startswith(UGWP_DIR + "/") for r in requested)
    requested.clear()
    nlb.fetch_mpas_tree(tmp_path, fetch_file, mpas_source=None, offline=True, log=lambda m: None)
    assert any(r.startswith(NOAHMP_DIR + "/") for r in requested)


# ---------------------------------------------------------------- 3. the stubs are interface only
def _stub_modules(text):
    return {m.group(1): m.group(2) for m in
            re.finditer(r"^module\s+(\w+)\s*$(.*?)^end module\s+\1\s*$", text, re.M | re.S)}


@pytest.mark.parametrize("component", sorted(INTERFACE))
def test_each_stub_is_the_interface_and_every_routine_stops_naming_the_opt_in(component):
    text = (BACKEND / f"licence_stub_{component}.F90").read_text()
    assert "SPDX-License-Identifier: Apache-2.0" in text
    modules = _stub_modules(text)
    stop = f"hex_licence_stub_{component}"
    assert set(modules) == set(INTERFACE[component]) | {stop}
    helper = modules.pop(stop)
    assert "error stop" in helper and f"--licence-opt-in {TERMS[component]}" in helper
    for module, body in modules.items():
        routines = re.findall(r"^ subroutine\s+(\w+)\((.*?)\)\s*$(.*?)^ end subroutine\s+\1\s*$",
                              body.replace("&\n", " "), re.M | re.S)
        assert [name for name, _, _ in routines] == INTERFACE[component][module]
        for name, _, statements in routines:
            executable = [line.strip() for line in statements.splitlines() if line.strip() and not
                          re.match(r"(integer|real|logical|character|type)\b.*::", line.strip())]
            # Declarations, then exactly one statement: the stop.  No routine returns.
            assert executable == [f"call {stop}_stop('{module}::{name}')"], (module, name)


def test_the_stubs_are_what_the_generator_writes_and_the_builder_checks_them():
    generator = (BACKEND / "make_licence_stubs.py").read_text()
    assert "No executable upstream" in generator and "Do not hand edit" in generator
    for component in INTERFACE:
        assert "Generated by make_licence_stubs.py" in (BACKEND / f"licence_stub_{component}.F90").read_text()


# ---------------------------------------------------------------- 4. the flavour is recorded
def test_the_library_licence_is_read_from_its_build_receipt(tmp_path):
    library = tmp_path / nlb.LIBRARY
    library.write_bytes(b"\x7fELF")
    assert nlb.library_licence(library)["flavour"] == "unrecorded"   # a lab build named by path
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text(json.dumps({"licence": {"stubbed": {"noahmp": {}}, "compiled": {}}}))
    assert nlb.library_licence(library)["flavour"] == "stubbed"
    receipt.write_text(json.dumps({"licence": {"stubbed": {}, "compiled": {"noahmp": {}}}}))
    assert nlb.library_licence(library)["flavour"] == "opted-in"


def _cli(*argv):
    parser = argparse.ArgumentParser()
    rb.add_physics_resources_parser(parser.add_subparsers())
    return parser.parse_args(["physics-resources", "build", *argv])


def test_the_build_command_passes_the_named_opt_ins(monkeypatch):
    seen = {}
    monkeypatch.setattr(rb, "build", lambda *a, **k: seen.update(k) or {})
    monkeypatch.delenv(licence_gate.ENVIRONMENT, raising=False)
    assert rb._build_command(_cli()) == 0 and seen["licence_opt_in"] == {}
    rb._build_command(_cli("--licence-opt-in", "ucar-noahmp,ugwp-held,noaa-gsl-unlicensed"))
    # ugwp-held and noaa-gsl-unlicensed are retired (owner ruling 2026-10-14, US Government code): the gate
    # announces them and drops them, so only the UCAR term reaches the builder.
    assert seen["licence_opt_in"] == {"ucar-noahmp": "--licence-opt-in"}
    monkeypatch.setenv(licence_gate.ENVIRONMENT, "ucar-noahmp")
    rb._build_command(_cli())
    assert seen["licence_opt_in"] == {"ucar-noahmp": licence_gate.ENVIRONMENT}


@pytest.mark.parametrize("term", ["no-such-terms"])
def test_the_build_command_refuses_unknown_terms(monkeypatch, term):
    monkeypatch.setattr(rb, "build", lambda *a, **k: pytest.fail("built"))
    monkeypatch.delenv(licence_gate.ENVIRONMENT, raising=False)
    with pytest.raises(rb.PhysicsResourceError, match="unknown opt-in"):
        rb._build_command(_cli("--licence-opt-in", term))


def test_the_library_build_takes_only_the_terms_it_compiles(tmp_path, monkeypatch):
    """WOOF_HEX_LICENCE_OPT_IN=ugwp-held (a retired name some RRFS recipes still set) changes nothing in the library."""
    seen = {}
    monkeypatch.setattr(rb, "build_packet", lambda cache, **k: {
        "cache": str(tmp_path), "packet": str(tmp_path / "packet.json"), "packet_sha256": "0" * 64,
        "thompson_tables": str(tmp_path), "thompson_tables_sha256": "1" * 64})
    monkeypatch.setattr(rb, "stage_pool_reader", lambda cache, log: {"path": str(tmp_path), "sha256": "2" * 64})
    monkeypatch.setattr(rb, "assemble_mpas_source", lambda cache, **k: {"path": str(tmp_path), "sha256": "3" * 64})
    monkeypatch.setattr(nlb, "toolchain", lambda: {})

    def native(cache, **k):
        seen.update(k)
        return {"path": str(tmp_path), "sha256": "4" * 64, "licence": {
            "flavour": "opted-in", "stubbed": {}, "compiled": {"noahmp": {"terms": "ucar-noahmp"}},
            "opt_in": ["ucar-noahmp"]}}

    monkeypatch.setattr(nlb, "build_native_library", native)
    receipt = rb.build(tmp_path, licence_opt_in={"ucar-noahmp": "--licence-opt-in",
                                                 "ugwp-held": licence_gate.ENVIRONMENT},
                       log=lambda message: None)
    assert seen["licence_opt_in"] == ["ucar-noahmp"]
    assert receipt["native_library_licence"]["named_by"] == {"ucar-noahmp": "--licence-opt-in"}
    rb.build(tmp_path, licence_opt_in={"ugwp-held": licence_gate.ENVIRONMENT}, log=lambda message: None)
    assert seen["licence_opt_in"] == []


# ---------------------------------------------------------------- 5. the route refuses before any work
def _config(physics=""):
    return resolve_namelist(f"&physics\nconfig_physics_suite='mesoscale_reference'\n{physics}/\n")


def _library(tmp_path, stubbed=("noahmp",)):
    library = tmp_path / nlb.LIBRARY
    library.write_bytes(b"\x7fELF")
    (tmp_path / "build-receipt.json").write_text(json.dumps({"licence": {
        "stubbed": {name: {"terms": TERMS[name]} for name in stubbed},
        "compiled": {name: {"terms": TERMS[name]} for name in TERMS if name not in stubbed}}}))
    return library


def test_an_opted_in_selection_on_a_stubbed_library_is_refused_with_the_rebuild(tmp_path):
    refusal = licence_gate.native_library_refusal(_config("config_lsm_scheme='sf_noahmp'\n"), _library(tmp_path))
    text = str(refusal)
    assert "land_surface=sf_noahmp" in text and "licence stubs" in text
    assert "woof hex physics-resources build --licence-opt-in ucar-noahmp" in text
    # UGWP is compiled in the stubbed library too: an opted-in UGWP run is not refused here.
    ugwp = _config("config_gwdo_scheme='bl_ugwp_gwdo'\n")
    assert licence_gate.native_library_refusal(ugwp, _library(tmp_path)) is None


def test_a_library_that_carries_the_code_or_a_run_without_it_is_not_refused(tmp_path):
    noahmp = _config("config_lsm_scheme='sf_noahmp'\n")
    assert licence_gate.native_library_refusal(noahmp, _library(tmp_path, stubbed=())) is None
    assert licence_gate.native_library_refusal(_config(), _library(tmp_path)) is None
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / nlb.LIBRARY).write_bytes(b"\x7fELF")   # no receipt beside it: a lab build compiled everything
    assert licence_gate.native_library_refusal(noahmp, lab / nlb.LIBRARY) is None
