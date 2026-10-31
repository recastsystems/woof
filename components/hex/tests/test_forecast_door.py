"""The forecast door: argument grammar, refusals, admission, receipt shape.

Everything here runs on a CPU-only box.  The door's device contact is
confined to three functions -- :func:`measure_device_memory`,
:func:`read_device_compute` and :func:`read_card_profile` -- so that the
DECISION they feed can be tested at every interesting free-memory value
without owning a card at that value, which is the only way to test the
refusal that fires on a card too small for the request.  All three are
substituted for every test in this file by the autouse fixtures below.

THE BREAKAGE THE THIRD ONE PREVENTS, measured on both CI matrix legs and
reproduced on the Windows desktop (``evidence/xmachine-20260827/`` §4):
``read_card_profile`` is a SECOND point of device contact that arrived after
this file was written, and it refuses outright when ``WOOF_HEX_NO_LOCAL_GPU``
is set.  ``ci.yml``'s tier-1 step sets exactly that variable, so this
distribution's own CI environment made this distribution's own door refuse,
and ``test_the_door_leaves_destination_creation_to_the_driver`` -- a test
about directory creation, with no interest in any card -- went red on the
release commit.  The refusal itself is correct and stays: a box that has
declared no GPU work cannot run a forecast.  What was wrong is a test file
claiming its device contact was confined to one function when it was not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from woof.hex import forecast_door as door
from woof.hex.cli import build_parser
from woof.hex.errors import ConfigurationRefusal


X1_CELLS = 40_962


def test_turbines_do_not_shift_existing_positional_request_arguments():
    import inspect
    signature = inspect.signature(door.ForecastRequest)
    assert signature.parameters['wind_farms'].kind is inspect.Parameter.KEYWORD_ONLY
    positional = [name for name, parameter in signature.parameters.items()
                  if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[positional.index('chemistry_config') + 1] == 'input_problems'


@pytest.mark.parametrize('selection', [None, 'off', 'uswtdb', '/data/turbines/windturbines.txt'])
def test_turbine_selection_reaches_the_driver_unchanged(tmp_path, selection):
    request = door.resolve_request(_namespace(tmp_path, wind_farms=selection), registry=_registry())
    argv = door.build_driver_argv(request)
    assert request.wind_farms == selection
    if selection is None:
        assert '--wind-farms' not in argv
    else:
        assert argv[argv.index('--wind-farms') + 1] == selection


@pytest.mark.parametrize('prefix', ['output', 'history', 'custom'])
def test_native_success_products_use_mpas_frames_and_real_start(tmp_path, monkeypatch, prefix):
    frame = tmp_path / f"{prefix}.2023-03-31_12.00.00.nc"
    frame.touch()
    (tmp_path / "native-physics-request.json").write_text(json.dumps(
        {"grid": "/grid.nc", "init": "/init.nc", "start": "2023-03-31_12:00:00",
         "streams": [{"name": "output", "directions": ["output"],
                      "filename": prefix + ".$Y-$M-$D_$h.$m.$s.nc"}]}))
    seen = []
    monkeypatch.setattr(door, "run_default_products", lambda request, history, args:
                        seen.append((door.products_command(request, history), history)))
    args = argparse.Namespace(out=tmp_path, preflight=False, no_products=False)
    door.native_default_products(args)
    assert seen[0][1] == [frame]
    assert seen[0][0][-2:] == ["--simulation-start", "2023-03-31_12:00:00"]
    args.no_products = True
    door.native_default_products(args)
    assert len(seen) == 1


def test_fullstep_shadow_is_separate_and_forwarded(tmp_path):
    args = _namespace(tmp_path)
    args.fullstep_local_timestep = True
    request = door.resolve_request(args,registry=_registry())
    argv = door.build_driver_argv(request)
    assert '--fullstep-local-timestep' in argv
    assert '--local-timestep' not in argv
    assert argv[argv.index('--fullstep-lts-mode')+1] == 'shadow'
    assert request.time_integration_order == 2


def test_fullstep_disabled_receipt_keeps_shape(tmp_path):
    args = _namespace(tmp_path)
    args.fullstep_local_timestep = False
    request = door.resolve_request(args,registry=_registry())
    assert '--fullstep-local-timestep' not in door.build_driver_argv(request)
    receipt = door.build_receipt(request=request,admission=None,bind_receipt=None,
                                driver_receipt=None,history=[],driver_argv=[],seconds=0,status='test')
    assert 'fullstep_lts' not in receipt


def test_fullstep_only_knob_disabled_is_refused_even_at_default(tmp_path):
    args = _namespace(tmp_path)
    args.fullstep_lts_max_rate = 16
    with pytest.raises(door.ForecastDoorRefusal,match='ignored'):
        door.resolve_request(args,registry=_registry())


def test_fullstep_active_is_forwarded_with_the_default_top_rate(tmp_path):
    # Retired refusal: active mode used to stop here for want of a per-class
    # physics clock.  It now resolves and reaches the driver, which owns the
    # refusals that remain.
    args = _namespace(tmp_path)
    args.fullstep_local_timestep = True
    args.fullstep_lts_mode = 'active'
    request = door.resolve_request(args,registry=_registry())
    argv = door.build_driver_argv(request)
    assert argv[argv.index('--fullstep-lts-mode')+1] == 'active'
    # 8 guards the full-physics top-rate-16 stop at 10.3 h (drivers/fullstep_lts_forecast_v841.py)
    assert argv[argv.index('--fullstep-lts-max-rate')+1] == '8'


def test_fullstep_active_resolves_on_several_cards(tmp_path):
    # Retired refusal (multi-card stage): active LTS now cuts every rate class
    # into blocks across the same-host cards, so the request resolves and is
    # forwarded; the same-host executor routes it to the class executor.
    args = _namespace(tmp_path)
    args.fullstep_local_timestep = True
    args.fullstep_lts_mode = 'active'
    args.samehost_devices = '0,1'
    request = door.resolve_request(args,registry=_registry())
    argv = door.build_driver_argv(request)
    assert argv[argv.index('--fullstep-lts-mode')+1] == 'active'


def test_fullstep_shadow_still_refuses_more_than_one_card(tmp_path):
    args = _namespace(tmp_path)
    args.fullstep_local_timestep = True
    args.samehost_devices = '0,1'
    with pytest.raises(door.ForecastDoorRefusal,match='halo duplicates'):
        door.resolve_request(args,registry=_registry())


def test_samehost_ranks_route_active_lts_to_the_class_executor(monkeypatch):
    from types import SimpleNamespace
    from woof.hex.drivers import fullstep_lts_forecast_v841 as fullstep
    from woof.hex.drivers import samehost_forecast_v841 as samehost

    seen = {}

    def fake(forecast, *, native_execute, policy, devices, memory_floor_by_device=None, **kwargs):
        seen.update(policy=policy, devices=devices, native=native_execute, keys=sorted(kwargs))
        return {"routed": True}

    monkeypatch.setattr(fullstep, "execute_fullstep_lts", fake)
    forecast = SimpleNamespace(execute_forecast=None)
    with samehost.installed(forecast, devices=(0, 1), layouts_path=None, part_path=None):
        result = forecast.execute_forecast(host={}, fullstep_lts_options={
            "fullstep_local_timestep": True, "fullstep_lts_mode": "active", "fullstep_lts_max_rate": 8})
        assert result == {"routed": True}
        assert seen["devices"] == (0, 1) and seen["policy"]["fullstep_lts_mode"] == "active"
        assert "fullstep_lts_options" not in seen["keys"]
        with pytest.raises(ValueError, match="halo duplicates"):
            forecast.execute_forecast(host={}, fullstep_lts_options={
                "fullstep_local_timestep": True, "fullstep_lts_mode": "shadow"})
X4_CELLS = 163_842


@pytest.mark.parametrize("execution", ["single", "samehost", "lts"])
def test_cycling_flags_reach_each_forecast_executor(tmp_path, execution):
    args = _namespace(tmp_path, checkpoint_hours="3,6", restart_from=tmp_path / "restart.pkl",
                      jedi_background=True, da_analysis=tmp_path / "analysis.nc")
    if execution == "samehost":
        args.samehost_devices = "0,1"
    if execution == "lts":
        args.fullstep_local_timestep = True
        args.fullstep_lts_mode = "active"
    argv = door.build_driver_argv(door.resolve_request(args, registry=_registry()))
    assert argv[argv.index("--checkpoint-hours") + 1] == "3,6"
    assert argv[argv.index("--restart-from") + 1] == str(args.restart_from.absolute())
    assert argv[argv.index("--da-analysis") + 1] == str(args.da_analysis.absolute())
    assert "--jedi-background" in argv
V15_CELLS = 38_857
MIB = 1024**2
GIB = 1024**3


@pytest.fixture(autouse=True)
def _proven_architecture(monkeypatch):
    """Pin the architecture seam to the proven floor for every door test.

    The architecture half of admission has its own tests below; everything
    else in this file is about the door's other decisions and must not
    depend on which card (if any) the test box carries.
    """

    monkeypatch.setattr(door, "read_device_compute", lambda: (12, 0))
    monkeypatch.setattr(door, "read_nvrtc_targets", lambda: NVRTC_TARGETS)
    monkeypatch.setattr(door, "probe_numeric_route", _no_probe)
    from woof.hex.drivers import samehost_forecast_v841
    monkeypatch.setattr(samehost_forecast_v841, "visible_card_memory", lambda: (32 * GIB,))


#: The targets CUDA 13's NVRTC lists.
NVRTC_TARGETS = (75, 80, 86, 89, 90, 100, 120, 121)


def _no_probe():
    raise AssertionError("the numeric route is probed only on an unanchored card")


def _anchored_route():
    """What the probe reports on a card whose route is the anchored one."""

    from woof.hex.cuda_backend import arch_admission

    expected = arch_admission._expected_bits()
    sub = expected["subnormal"]
    record = arch_admission.classify_numeric_route(
        (0, 0, sub, sub, expected["separately_rounded"],
         expected["x_times_y"], expected["x_plus_y"]),
        (expected["dx_times_dy"], expected["dx_plus_dy"], expected["subnormal_in_fp64"]),
    )
    record.update(schema=arch_admission.NUMERIC_ROUTE_SCHEMA, dual_run_identical=True)
    return record


@pytest.fixture(autouse=True)
def _reference_card(monkeypatch):
    """Pin the card SHAPE seam for every door test.

    The footprint row is priced from the card's multiprocessor count, and
    ``read_card_profile`` reads that from the driver -- a device query, and
    a refusal when the box has banned GPU work.  Every test in this file
    decides something else, so the shape is supplied here rather than asked
    of whatever card (or ban) the runner happens to have.  The tests that
    are ABOUT the card's shape substitute their own row against the same
    seam, in tests/test_device_admission.py.
    """

    from woof.hex.device_admission import REFERENCE_CARD

    monkeypatch.setattr(door, "read_card_profile", lambda: REFERENCE_CARD)


@pytest.fixture(autouse=True)
def _satisfied_seam_pin(monkeypatch):
    """Pin the woof SEAM-PIN seam for every door test.

    ``--gpuwm-checkout`` in this file is an empty directory under
    ``tmp_path``: these tests are about the argument grammar, the schedule,
    the receipt and the driver vector, and none of them can carry a real
    woof checkout at the pinned commit.  The door's byte check over that
    checkout is therefore supplied here, exactly as the card shape and the
    architecture are.

    The check ITSELF -- that it names the version found, the version wanted
    and the remedy, and that it stays silent on a checkout whose bytes match
    -- is exercised against real bytes in tests/test_engine_pin.py, where
    this substitution is not in force.
    """

    monkeypatch.setattr(door, "seam_pin_problem", lambda checkout: None)


def _registry() -> dict[str, door.MeshRow]:
    """A stand-in for the checkout registry, with the real three rows."""

    return {
        "x4.163842": door.MeshRow("x4.163842", X4_CELLS, 120.0),
        "x1.40962": door.MeshRow("x1.40962", X1_CELLS, 120.0),
        "v15.150.38857": door.MeshRow("v15.150.38857", V15_CELLS, 60.0),
    }


def _register_timestep_anchor(monkeypatch, dt_seconds: float) -> None:
    """Register one timestep anchor for the duration of a test.

    The same gesture a ruling would make in
    ``woof.hex.dt_admission.ADMITTED_TIMESTEPS`` -- one row naming its
    evidence -- so a test can exercise behaviour past the gate without the
    gate being weakened for anyone else.
    """

    from woof.hex import dt_admission

    anchor = dt_admission.DtAnchor(
        dt_seconds=float(dt_seconds),
        radiation_seconds=600.0,
        surface_pbl_seconds=float(dt_seconds),
        cumulus_seconds=float(dt_seconds),
        cumulus_scheme="gf",
        meshes=(),
        card="test-registered anchor",
        admitted_on="2026-08-26",
        schedule_receipt="evidence/dt-admission-20260826/",
        integration_anchor="evidence/dt-admission-20260826/",
        native_reference=None,
        basis="test-registered anchor",
        physics_health="TRACKS test-registered anchor",
    )
    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {
            **dt_admission.ADMITTED_TIMESTEPS,
            dt_admission.dt_key(dt_seconds): anchor,
        },
    )


def _namespace(tmp_path: Path, **overrides) -> argparse.Namespace:
    """A complete, valid argument vector, so each test perturbs exactly one thing."""

    grid = tmp_path / "x1.40962.grid.nc"
    static = tmp_path / "x1.40962.static.nc"
    init = tmp_path / "init.nc"
    checkout = tmp_path / "woof"
    for path in (grid, static):
        path.write_bytes(b"not really netcdf")
    from netCDF4 import Dataset
    with Dataset(init, "w") as dataset:
        dataset.createDimension("nVertLevels", 55)
        dataset.createDimension("nVertLevelsP1", 56)
    checkout.mkdir()
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args(
        [
            "--mesh", "x1.40962",
            "--grid", str(grid),
            "--static", str(static),
            "--init", str(init),
            "--init-source", "GFS 2026-08-24 00Z",
            "--hours", "1.0",
            "--history-every-minutes", "30",
            "--out", str(tmp_path / "out"),
            "--gpuwm-checkout", str(checkout),
        ]
    )
    for key, value in overrides.items():
        setattr(arguments, key, value)
    return arguments


# ---------------------------------------------------------------------------
# the door exists, on the console script, with the doors' grammar
# ---------------------------------------------------------------------------
def test_forecast_is_a_subcommand_of_the_console_script() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "forecast",
            "--mesh", "x1.40962",
            "--grid", "g.nc",
            "--static", "s.nc",
            "--init", "i.nc",
            "--init-source", "GFS",
            "--hours", "1",
            "--history-every-minutes", "30",
            "--out", "out",
        ]
    )
    assert arguments.command == "forecast"
    assert arguments.handler is not None


def test_defaults_are_the_pinned_configuration() -> None:
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args([])
    # Fixed means default: the proven lane is what a bare run gets.
    assert arguments.horiz_mixing == "2d_smagorinsky"
    assert arguments.local_timestep is False
    # None means "the model's own margin", which is the default now: the
    # margin is priced from the card, so a flat default would be one card's
    # number handed to every other card -- the shape of the borrowed-row defect.
    assert arguments.headroom_mib is None
    assert arguments.device_fixed_mib is None
    assert arguments.device_bytes_per_cell is None
    assert arguments.scratch is None


def test_help_names_every_flag_the_manual_documents() -> None:
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    text = parser.format_help()
    for flag in (
        "--mesh", "--grid", "--static", "--init", "--init-source",
        "--hours", "--history-every-minutes", "--out", "--scratch",
        "--gpuwm-checkout", "--repo", "--case-label", "--horiz-mixing",
        "--local-timestep", "--headroom-mib", "--device-fixed-mib",
        "--device-bytes-per-cell", "--stop-on-refusal", "--preflight",
        "--receipt",
    ):
        assert flag in text, flag


# ---------------------------------------------------------------------------
# refusals: each names the breakage and the remedy
# ---------------------------------------------------------------------------
def _refusal(arguments: argparse.Namespace, **kwargs) -> str:
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_request(arguments, registry=_registry(), **kwargs)
    return str(caught.value)


def _admission_message(verdict: door.AdmissionVerdict) -> str:
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.require_admitted(verdict)
    return str(caught.value)


@pytest.mark.parametrize(
    "flag,attribute",
    [
        ("--mesh", "mesh"),
        ("--grid", "grid"),
        ("--static", "static"),
        ("--init", "init"),
        ("--init-source", "init_source"),
        ("--hours", "hours"),
        ("--history-every-minutes", "history_every_minutes"),
        ("--out", "out"),
    ],
)
def test_a_missing_required_flag_is_refused_by_name(
    tmp_path: Path, flag: str, attribute: str
) -> None:
    message = _refusal(_namespace(tmp_path, **{attribute: None}))
    assert flag in message
    assert len(message) > 80, "a refusal states the breakage, not just the flag"


def test_an_unregistered_mesh_names_the_registered_ones(tmp_path: Path) -> None:
    message = _refusal(_namespace(tmp_path, mesh="x9.999999"))
    assert "x9.999999" in message
    for name in _registry():
        assert name in message


def test_a_missing_mesh_file_is_refused_before_anything_expensive(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.grid.nc"
    message = _refusal(_namespace(tmp_path, grid=missing))
    assert str(missing) in message
    assert "--grid" in message


def test_a_missing_gpuwm_checkout_names_the_pin_it_satisfies(
    tmp_path: Path,
) -> None:
    """The refusal must name the breakage that is LIVE, not the one that closed.

    This gate's own table was the stale side and moved 2026-08-28.  It used to
    demand the words "source checkout" and "sixteen", which the refusal
    supplied by saying one of the sixteen pinned paths reaches no wheel.  At
    engine 2.5.8 that is false -- all sixteen resolve from site-packages,
    measured against the published wheels -- so a refusal giving that reason
    names a breakage that no longer exists, which the refusal law forbids.
    What survives is git provenance, so that is what is demanded here.
    """

    message = _refusal(_namespace(tmp_path, gpuwm_checkout=tmp_path / "nope"))
    assert "--gpuwm-checkout" in message
    assert "git clone" in message.lower(), (
        "a named tree is a git clone; naming only 'a checkout' leaves a user "
        "free to unpack a tarball, which names no commit"
    )
    assert "sixteen" in message, "the refusal names what the tree supplies"
    assert "installed engine" in message, (
        "since 0.3.2 the flag is optional: the remedy names the default"
    )
    assert "no wheel" not in message, "the retired reason must not come back"


def test_an_existing_output_directory_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "already"
    out.mkdir()
    message = _refusal(_namespace(tmp_path, out=out))
    assert str(out) in message
    assert "exists" in message


def test_an_output_whose_parent_is_absent_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "no" / "such" / "parent" / "out"
    message = _refusal(_namespace(tmp_path, out=out))
    assert str(out.parent) in message


def test_scratch_inside_the_output_tree_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    message = _refusal(_namespace(tmp_path, out=out, scratch=out / "inside"))
    assert "--scratch" in message
    assert str(out) in message


def test_the_default_scratch_is_a_sibling_of_the_output(tmp_path: Path) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    assert request.scratch.parent == request.out.parent
    assert request.scratch != request.out
    assert request.out not in request.scratch.parents


def test_a_forecast_length_that_is_not_whole_steps_is_refused(
    tmp_path: Path,
) -> None:
    message = _refusal(_namespace(tmp_path, hours=0.51))
    assert "--hours" in message
    assert "120" in message, "the refusal names the mesh's admitted timestep"


def test_a_history_cadence_that_does_not_divide_the_run_is_refused(
    tmp_path: Path,
) -> None:
    message = _refusal(_namespace(tmp_path, hours=1.0, history_every_minutes=45))
    assert "--history-every-minutes" in message


def test_the_generated_mesh_row_validates_the_schedule_at_its_own_timestep(
    tmp_path: Path, monkeypatch
) -> None:
    # v15.150.38857 declares 60 s, not x4's 120 s; a schedule check that used a
    # single module constant would admit a half-step run on this row.
    #
    # 60 s holds no timestep anchor, so the door refuses this row outright
    # (see the dt-admission test below).  The property under test here is the
    # SCHEDULE arithmetic, so this test registers a 60 s anchor the way a
    # ruling would -- one row -- and then checks the schedule.  That the whole
    # difference between "refused" and "runs" is a table row is itself the
    # point of the registry.
    _register_timestep_anchor(monkeypatch, 60.0)
    arguments = _namespace(
        tmp_path, mesh="v15.150.38857", hours=0.05, history_every_minutes=1
    )
    request = door.resolve_request(arguments, registry=_registry())
    assert request.dt_seconds == 60.0
    assert request.steps == 3


def test_command_line_problems_are_refused_before_filesystem_ones(
    tmp_path: Path,
) -> None:
    """A property of the request outranks a property of the disk.

    The order this pins replaced one where --init's existence was checked
    first: a user with a mistyped --hours and a not-yet-built init was told
    about the init, built it, and only then learned the schedule was wrong.
    """

    message = _refusal(
        _namespace(tmp_path, hours=0.51, init=tmp_path / "absent.init.nc")
    )
    assert "--hours" in message
    assert "--init" not in message


def test_a_half_model_row_outranks_a_missing_file_too(tmp_path: Path) -> None:
    message = _refusal(
        _namespace(
            tmp_path, device_fixed_mib=4000.0, init=tmp_path / "absent.init.nc"
        )
    )
    assert "--device-bytes-per-cell" in message


def test_a_local_timestep_ladder_that_does_not_start_at_one_is_refused(
    tmp_path: Path,
) -> None:
    message = _refusal(
        _namespace(tmp_path, local_timestep=True, local_timestep_rates="2,3")
    )
    assert "--local-timestep-rates" in message


def test_half_a_measured_footprint_row_is_refused(tmp_path: Path) -> None:
    message = _refusal(_namespace(tmp_path, device_fixed_mib=4000.0))
    assert "--device-bytes-per-cell" in message
    assert "--device-fixed-mib" in message


# ---------------------------------------------------------------------------
# admission: the measured model, decided against memory measured NOW

@pytest.mark.parametrize('options', [
    '{"sppt":true,"degree":48}',
    '{"skeb":true,"degree":24}',
    '{"sppt":true,"skeb":true,"sppt_scales":[[0.3,500000,600,91]]}',
])
def test_stochastic_array_threshold_agrees_with_driver_floor(tmp_path, options):
    from dataclasses import replace
    from woof.hex.device_admission import required_free_bytes, harmonic_basis_bytes
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    request = replace(request, stochastic=options)
    from woof.hex.stochastic import StochasticConfig
    basis = harmonic_basis_bytes(request.cells, StochasticConfig.parse(options).degree)
    arrays = door.stochastic_array_ledger(request.cells, options)
    array_bytes = sum(arrays.values())
    model = door._driver_floor_model(request)
    required = required_free_bytes(request.cells, model, request.headroom_bytes) + array_bytes
    args = dict(mesh=request.mesh, cells=request.cells, total_bytes=required*2,
                headroom_bytes=request.headroom_bytes, registry={}, model=model,
                stochastic=request.stochastic)
    rejected = door.admission_verdict(**args, free_bytes=required-1)
    accepted = door.admission_verdict(**args, free_bytes=required)
    assert not rejected.admitted and accepted.admitted
    assert rejected.fitted_cells < request.cells <= accepted.fitted_cells
    assert accepted.as_dict()['stochastic_basis_bytes'] == basis
    assert accepted.as_dict()['stochastic_array_bytes'] == array_bytes > basis
    assert accepted.as_dict()['stochastic_arrays'] == arrays
    argv = door.build_driver_argv(request)
    assert int(argv[argv.index('--required-free-bytes')+1]) == required


def test_disabled_stochastic_arrays_do_not_change_admission():
    assert door.stochastic_array_ledger(40962, None) == {}
    assert door.stochastic_array_bytes(40962, '{"sppt":false,"skeb":false}') == 0
    # Inactive SPPT scales are neither constructed nor charged for SKEB-only.
    assert door.stochastic_array_bytes(40962, '{"skeb":true}') == door.stochastic_array_bytes(
        40962, '{"skeb":true,"sppt_scales":[[0.3,500000,600,91]]}')
# ---------------------------------------------------------------------------
def test_the_shipped_model_is_the_measured_row() -> None:
    # The merged-tip row of record: the deeper pin against the raw evidence
    # ledgers lives in test_device_admission.py.  The row is SHAPED as of
    # 2026-08-27 -- a card core plus tiled physics workspaces plus a per-cell
    # term -- so what must reproduce the measured peaks is predict_bytes, not
    # any one coefficient.
    assert not hasattr(door.FOOTPRINT_MODEL, "fixed_bytes"), (
        "an affine fixed term must not survive as an attribute: reading one "
        "off the shaped row is how a caller silently prices a mesh without "
        "the workspace terms"
    )
    # Today's row is measured on today's build (the hex memory census cuts,
    # device_admission.CURRENT_GLOBAL_ROW): nothing restated, and it
    # reproduces its two measured peaks, 5,726 MiB at x1.40962 and 12,244 MiB
    # at x1.163842, within rounding.
    from woof.hex.device_admission import STACK_RESTATEMENT_BYTES_PER_CELL, _row_model

    model = door.FOOTPRINT_MODEL
    assert model.restated_bytes_per_cell == 0.0
    assert model.predict_bytes(X1_CELLS) / MIB == pytest.approx(5726.0, abs=1.0)
    assert model.predict_bytes(X4_CELLS) / MIB == pytest.approx(12244.0, abs=1.0)
    # The 2026-08 proof row it replaced, without its restatement, is still the
    # 8,874 and 20,446 MiB it was fitted from.
    proof = _row_model("global/170sm")
    assert proof.restated_bytes_per_cell == STACK_RESTATEMENT_BYTES_PER_CELL

    def measured_part(cells: int) -> float:
        return (proof.predict_bytes(cells) - proof.restated_bytes_per_cell * cells) / MIB

    assert measured_part(X1_CELLS) == pytest.approx(8874.0, abs=1.0)
    assert measured_part(X4_CELLS) == pytest.approx(20446.0, abs=1.0)


def test_a_twelve_gib_card_admits_the_published_global_mesh() -> None:
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=int(11.2 * GIB),
        total_bytes=12 * GIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.admitted is True
    assert verdict.shortfall_bytes == 0


def test_a_ten_gib_card_in_use_refuses_and_names_the_shortfall() -> None:
    free = 5900 * MIB
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=free,
        total_bytes=10240 * MIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.admitted is False
    assert verdict.shortfall_bytes > 0
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.require_admitted(verdict)
    message = str(caught.value)
    assert "x1.40962" in message
    # Today's row reproduces the measured 5,726 MiB x1 peak of the build it
    # prices (device_admission.CURRENT_GLOBAL_ROW).
    assert "5,726" in message or "5726" in message
    assert "MiB" in message
    # The remedy is not "try again": it is a fitted alternative, by name.
    assert "fits" in message or "fitted" in message
    assert "hex_ledger_probe" in message, (
        "the remedy for a card whose fixed term is smaller is to MEASURE it"
    )


def test_the_refusal_names_a_smaller_registered_mesh_when_one_fits() -> None:
    # A budget that holds the 38,857-cell row but not the 40,962-cell one.
    fitted = door.FOOTPRINT_MODEL.predict_bytes(V15_CELLS)
    free = int(fitted + door.DEFAULT_HEADROOM_BYTES + 8 * MIB)
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=free,
        total_bytes=12 * GIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.admitted is False
    assert "v15.150.38857" in verdict.alternatives
    assert "x4.163842" not in verdict.alternatives
    assert "v15.150.38857" in _admission_message(verdict)


def test_when_no_registered_mesh_fits_the_refusal_says_so_with_the_number() -> None:
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=2 * GIB,
        total_bytes=4 * GIB,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert verdict.alternatives == ()
    message = _admission_message(verdict)
    assert "no registered mesh" in message
    # The refusal must name THIS card's core, not "the fixed term": a card
    # is refused by its own arithmetic now, and a sentence that quotes a
    # shared constant is how a 32 GiB card's number ends up explaining a
    # 10 GiB card's refusal (the borrowed-row defect).
    assert "this card's own core" in message


def test_a_supplied_measured_row_replaces_the_shipped_one() -> None:
    model = door.ShapedFootprintModel(
        core_bytes=3000.0 * MIB,
        bytes_per_cell=93_474,
        card=door.CardProfile("a 68 SM part", 68),
        configuration="global",
        provenance="measured on this card",
    )
    verdict = door.admission_verdict(
        mesh="x1.40962",
        cells=X1_CELLS,
        free_bytes=9500 * MIB,
        total_bytes=10240 * MIB,
        headroom_bytes=None,
        registry=_registry(),
        model=model,
    )
    assert verdict.admitted is True
    assert verdict.model_provenance == "measured on this card"


def test_the_headroom_is_part_of_the_decision() -> None:
    exact = int(round(door.FOOTPRINT_MODEL.predict_bytes(X1_CELLS)))
    tight = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=exact,
        total_bytes=12 * GIB, headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    assert tight.admitted is False
    zero = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=exact,
        total_bytes=12 * GIB, headroom_bytes=0, registry=_registry(),
    )
    assert zero.admitted is True


def test_admission_measures_the_card_at_decision_time(monkeypatch) -> None:
    """Never a frozen budget: the free-memory read happens per decision."""

    calls: list[int] = []

    def _measure() -> tuple[int, int]:
        calls.append(1)
        return 11 * GIB, 12 * GIB

    monkeypatch.setattr(door, "measure_device_memory", _measure)
    first = door.admit_device(
        mesh="x1.40962", cells=X1_CELLS,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=_registry(),
    )
    second = door.admit_device(
        mesh="x1.40962", cells=X1_CELLS,
        headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=_registry(),
    )
    assert len(calls) == 2
    assert first.admitted and second.admitted


def test_a_card_that_cannot_be_measured_refuses_rather_than_guesses(
    monkeypatch,
) -> None:
    def _measure() -> tuple[int, int]:
        raise door.ForecastDoorRefusal(
            "cupy is not importable, so free device memory cannot be measured"
        )

    monkeypatch.setattr(door, "measure_device_memory", _measure)
    with pytest.raises(door.ForecastDoorRefusal):
        door.admit_device(
            mesh="x1.40962", cells=X1_CELLS,
            headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=_registry(),
        )


# ---------------------------------------------------------------------------
# the run receipt, and the hand-off to the render door
# ---------------------------------------------------------------------------
def test_the_receipt_carries_what_ran_and_serialises(tmp_path: Path) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    verdict = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=11 * GIB,
        total_bytes=12 * GIB, headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    receipt = door.build_receipt(
        request=request,
        admission=verdict,
        bind_receipt={"mesh": "x1.40962", "rebound": True},
        driver_receipt={"status": "passed"},
        history=[tmp_path / "out" / "cuda-history.2026-08-24_01.00.00.nc"],
        driver_argv=["--hours", "1.0"],
        seconds=12.5,
        status="passed",
    )
    assert receipt["schema"] == door.RECEIPT_SCHEMA
    assert receipt["status"] == "passed"
    assert receipt["admission"]["admitted"] is True
    assert receipt["mesh"]["name"] == "x1.40962"
    assert receipt["render_command"][0] == "woof hex"
    assert receipt["render_command"][1] == "render"
    assert receipt["history"]
    json.dumps(receipt, sort_keys=True, allow_nan=False)


def test_the_render_command_names_the_files_the_render_door_takes(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    history = [tmp_path / "out" / "cuda-history.2026-08-24_01.00.00.nc"]
    command = door.render_command(request, history)
    assert "--history" in command
    assert str(history[0]) in command
    assert "--mesh" in command
    assert str(request.grid) in command
    assert "--out" in command


def test_the_receipt_records_a_refusal_without_claiming_a_forecast(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    verdict = door.admission_verdict(
        mesh="x1.40962", cells=X1_CELLS, free_bytes=2 * GIB,
        total_bytes=4 * GIB, headroom_bytes=door.DEFAULT_HEADROOM_BYTES,
        registry=_registry(),
    )
    receipt = door.build_receipt(
        request=request, admission=verdict, bind_receipt=None,
        driver_receipt=None, history=[], driver_argv=[], seconds=0.4,
        status="refused_by_admission",
    )
    assert receipt["status"] == "refused_by_admission"
    assert receipt["admission"]["admitted"] is False
    assert receipt["history"] == []
    assert receipt["render_command"] is None


# ---------------------------------------------------------------------------
# what the door hands the driver
# ---------------------------------------------------------------------------
def test_the_driver_vector_carries_the_run_and_maps_the_checkout_flag(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    argv = door.build_driver_argv(request)
    assert "--cache-root" in argv
    assert argv[argv.index("--cache-root") + 1] == str(request.scratch)
    assert argv[argv.index("--output") + 1] == str(request.out)
    # The door's spelling is the brand's; the driver's is the engine's.
    assert argv[argv.index("--arwen-checkout") + 1] == str(request.gpuwm_checkout)
    assert "--preflight-only" not in argv
    assert "--local-timestep" not in argv
    # The schedule always reaches the driver explicitly: MPAS-A's Registry
    # defaults (woof.hex.mpas_defaults) unless the request names others.
    assert argv[argv.index("--time-integration-order") + 1] == "2"
    assert argv[argv.index("--acoustic-substeps") + 1] == "4"
    assert argv[argv.index("--radiation-seconds") + 1] == "1800.0"
    assert "--monotonic" in argv and "--no-monotonic" not in argv


@pytest.mark.parametrize("count,radiation", [(4, 900.0), (2, 1800.0)])
def test_the_requested_schedule_reaches_the_internal_parser(tmp_path, count, radiation):
    from woof.hex.drivers.run_cuda_v841_forecast import parse_args
    # dt 60 divides both radiation intervals; 900 s does not divide the
    # row's 120 s, which the door now refuses by name before the card.
    request = door.resolve_request(_namespace(tmp_path, time_integration_order=2,
        acoustic_substeps=count, radiation_seconds=radiation, monotonic=True, dt="60"),
        registry=_registry())
    parsed = parse_args(door.build_driver_argv(request))
    assert (parsed.time_integration_order, parsed.acoustic_substeps, parsed.radiation_seconds,
            parsed.monotonic) == (2, count, radiation, True)


def test_the_opt_in_lane_reaches_the_driver_only_when_asked(tmp_path: Path) -> None:
    arguments = _namespace(
        tmp_path, local_timestep=True, local_timestep_rates="1,3,6"
    )
    request = door.resolve_request(arguments, registry=_registry())
    argv = door.build_driver_argv(request)
    assert "--local-timestep" in argv
    assert argv[argv.index("--local-timestep-rates") + 1] == "1,3,6"


def test_boundary_fingerprint_cadence_preserves_default_and_rejects_negative(tmp_path):
    args = _namespace(tmp_path)
    request = door.resolve_request(args, registry=_registry())
    assert "--fingerprint-every" not in door.build_driver_argv(request)
    args.boundary_fingerprint_every = 36
    request = door.resolve_request(args, registry=_registry())
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--fingerprint-every") + 1] == "36"
    args.boundary_fingerprint_every = -1
    assert "nonnegative" in _refusal(args)


def test_the_door_leaves_destination_creation_to_the_driver(
    tmp_path: Path, monkeypatch
) -> None:
    """The driver's validate_destination requires --out and --scratch ABSENT
    and creates both itself.  A door that pre-creates them kills every
    admitted run with FileExistsError immediately after admission -- measured
    on the RTX 3080, 2026-08-24, where the door's success leg had never run.
    This test drives run_forecast with a driver stub enforcing the real
    driver's absence contract; it fails if the door ever creates either root
    before the hand-off.
    """

    seen: dict[str, bool] = {}
    from woof.hex import kernel_prewarm
    # This test supplies a CPU driver. Compile replay has its own tests.
    monkeypatch.setattr(kernel_prewarm, "install_recorder", lambda: None)
    monkeypatch.setattr(kernel_prewarm, "start_replay", lambda: None)

    class _Driver:
        @staticmethod
        def main(argv: list[str]) -> int:
            out = Path(argv[argv.index("--output") + 1])
            cache = Path(argv[argv.index("--cache-root") + 1])
            # the real driver's contract, verbatim in spirit:
            for label, path in (("output root", out), ("cache root", cache)):
                if path.exists():
                    raise FileExistsError(f"{label} must be absent: {path}")
            out.mkdir(parents=False)
            cache.mkdir(parents=False)
            seen["created_by_driver"] = True
            return 0

    monkeypatch.setattr(door, "load_registry", lambda repo: _registry())
    monkeypatch.setattr(
        door, "_load_drivers", lambda repo: (object(), _Driver())
    )
    monkeypatch.setattr(
        door, "_bind", lambda binding, driver, request: {"rebound": True}
    )
    # Free memory ample for x1.40962 on today's restated row; this test is
    # about destination creation, not the admission number.
    monkeypatch.setattr(
        door, "measure_device_memory", lambda: (15 * GIB, 16 * GIB)
    )
    # The ArWen driver's plumbing: name the ArWen suite (MPAS physics is the hex default since 0.4.6).
    arguments = _namespace(tmp_path, physics_suite="arwen_mpas_column")
    rc = door.run_forecast(arguments)
    assert seen.get("created_by_driver") is True
    # rc 1 is the door's own "driver exited 0 but wrote no history frame"
    # verdict from the stub; the point of this test is that no
    # ForecastDoorRefusal wrapping FileExistsError was raised above.
    assert rc == 1
    import importlib.util
    if importlib.util.find_spec("woof") is None or importlib.util.find_spec("woof.timeline") is None:
        # No engine ledger (woof 2.8.7 and earlier): nothing is recorded.
        assert not (arguments.out / "timeline.jsonl").exists()
        return
    events = [json.loads(line) for line in (arguments.out / "timeline.jsonl").read_text().splitlines()]
    assert [(event["stage"], event["phase"]) for event in events] == [
        ("prep.door", "begin"), ("forecast.construct", "begin"),
        ("forecast.construct", "mark"), ("prep.door", "end"), ("finish", "mark")]
    assert all(event["product"] == "hex" for event in events)


def test_samehost_two_device_run_finishes_the_door_receipt(tmp_path, monkeypatch):
    """A successful N-rank driver must also return a successful CLI receipt."""
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace
    from woof.hex.drivers import samehost_forecast_v841 as samehost
    from woof.hex import engine_per_card, kernel_prewarm
    from woof.hex.cuda_backend import runtime

    # This receipt fixture supplies CPU admission and a CPU driver. Do not
    # open a real CUDA context or replay the machine's kernel manifest before
    # invoking that stub (the same-host prewarm path does both on GPU hosts).
    monkeypatch.setattr(runtime, "require_cuda", lambda **kwargs: None)
    monkeypatch.setattr(kernel_prewarm, "install_recorder", lambda: None)
    monkeypatch.setattr(kernel_prewarm, "start_replay", lambda **kwargs: None)
    # The forward-merged door also binds engine modules per card before the
    # CPU driver stub. Supply a fake provider and record that binding without
    # importing CuPy or creating a device context in this receipt-only test.
    fake_cupy = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    bound = []
    monkeypatch.setattr(engine_per_card, "bind_engine_modules_per_card", bound.append)

    selected = []
    options = {"devices": (0, 1), "admission_records": [
        {"device": 0, "compute_capability": [12, 0]},
        {"device": 1, "compute_capability": [12, 0]},
    ]}

    @contextmanager
    def installed(driver, **received):
        # A rank process rebuilds the forecast's module state from the door's own request and driver argv.
        replay = received.pop("door_replay")
        assert set(replay) == {"repo", "request", "driver_argv"} and "--output" in replay["driver_argv"]
        assert received == options
        selected.append(True)
        try:
            yield
        finally:
            selected.pop()

    def admit(request, registry, received, module):
        assert received is options
        return door.admission_verdict(mesh=request.mesh, cells=request.cells,
            free_bytes=96 * GIB, total_bytes=96 * GIB,
            headroom_bytes=door.DEFAULT_HEADROOM_BYTES, registry=registry)

    def driver_main(argv):
        assert selected == [True]
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        (output / "cuda-history.2026-08-24_01.00.00.nc").write_bytes(b"driver-owned history")
        (output / door.DRIVER_RECEIPT_NAME).write_text(json.dumps({
            "status": "passed", "forecast": {"admission": options["admission_records"]}}))
        return 0

    monkeypatch.setattr(door, "load_registry", lambda repo: _registry())
    monkeypatch.setattr(door, "_load_drivers", lambda repo: (object(), SimpleNamespace(main=driver_main)))
    monkeypatch.setattr(door, "_bind", lambda binding, driver, request: {"rebound": True})
    monkeypatch.setattr(samehost, "request_options", lambda arguments, request: options)
    monkeypatch.setattr(samehost, "admit_request", admit)
    monkeypatch.setattr(samehost, "installed", installed)
    monkeypatch.setattr(door, "admit_architecture", lambda: pytest.fail("same-host already admits every card"))
    receipt_path = tmp_path / "complete-door.json"
    # The ArWen driver's plumbing: name the ArWen suite (MPAS physics is the hex default since 0.4.6).
    arguments = _namespace(tmp_path, samehost_devices="0,1", receipt=str(receipt_path),
                           physics_suite="arwen_mpas_column")
    assert door.run_forecast(arguments) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "passed"
    assert receipt["architecture"] is None
    assert receipt["driver_receipt"]["forecast"]["admission"] == options["admission_records"]
    assert selected == []
    assert bound == [fake_cupy]


def test_preflight_asks_the_driver_for_a_preflight_and_no_destination(
    tmp_path: Path,
) -> None:
    request = door.resolve_request(
        _namespace(tmp_path, preflight=True), registry=_registry()
    )
    argv = door.build_driver_argv(request)
    assert "--preflight-only" in argv
    assert "--cache-root" not in argv
    assert "--output" not in argv


def test_preflight_collects_missing_inputs_instead_of_stopping_at_the_first(
    tmp_path: Path,
) -> None:
    # A user asking "will this run on my card?" has often not built the init
    # yet.  If a missing init stopped the answer, the card question -- the
    # one no file fixes -- could not be asked until every file existed.
    from woof.hex import engine_pin

    arguments = _namespace(
        tmp_path,
        preflight=True,
        init=tmp_path / "absent.init.nc",
        gpuwm_checkout=None,
    )
    original = engine_pin.installed_root
    engine_pin.installed_root = lambda: None  # no engine in this interpreter
    try:
        request = door.resolve_request(arguments, registry=_registry())
    finally:
        engine_pin.installed_root = original
    assert request.inputs_present is False
    joined = " | ".join(request.input_problems)
    assert "--init" in joined
    assert "no woof is installed" in joined
    # and the request is still complete enough to decide the card question
    assert request.cells == X1_CELLS


def test_a_real_run_refuses_the_same_missing_input(tmp_path: Path) -> None:
    message = _refusal(_namespace(tmp_path, init=tmp_path / "absent.init.nc"))
    assert "--init" in message


def test_preflight_does_not_demand_a_fresh_output_directory(tmp_path: Path) -> None:
    # Preflight writes nothing into --out, so refusing an existing one would
    # make the answer "will this run?" unaskable a second time.
    out = tmp_path / "out"
    out.mkdir()
    request = door.resolve_request(
        _namespace(tmp_path, out=out, preflight=True), registry=_registry()
    )
    assert request.preflight is True
    assert request.out == out


# ---------------------------------------------------------------------------
# the registry the door resolves through is the checkout's own
# ---------------------------------------------------------------------------
def test_the_registry_comes_from_the_mesh_binding_module() -> None:
    registry = door.load_registry()
    # The oracle is the binding module itself, not a restated list: a frozen
    # set here would refuse every mesh the registry legitimately gains.
    binding = door._load_module(
        "mpas_mesh_binding_oracle", door.DRIVERS_DIR / "mpas_mesh_binding.py"
    )
    assert set(registry) == set(binding.MESH_BINDINGS)
    for name, row in binding.MESH_BINDINGS.items():
        assert registry[name].cells == int(row.n_cells)
        assert registry[name].dt_seconds == float(row.dt_seconds)
    assert registry["x1.40962"].cells == X1_CELLS


def test_the_drivers_ship_in_the_package_and_need_no_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """THE BREAKAGE THIS PREVENTS: through 0.3.1 the drivers lived in the
    repository's ``tools/``, which the wheel does not carry, so on every
    install the door refused with "needs the woof hex SOURCE CHECKOUT".
    The drivers now load from the package that is executing, wherever the
    process stands, and ``--repo`` is accepted, reported and not used."""

    import woof.hex

    package = Path(woof.hex.__file__).resolve().parent
    assert door.DRIVERS_DIR == package / "drivers"
    for name in ("run_cuda_v841_full_physics_x4.py", "run_cuda_v841_forecast.py",
                 "mpas_mesh_binding.py"):
        assert (door.DRIVERS_DIR / name).is_file(), name

    monkeypatch.setattr(door, "PROJECT_ROOT", None)  # the wheel shape
    monkeypatch.chdir(tmp_path)  # standing nowhere near a checkout
    assert door.resolve_repo(None) == door.DRIVERS_DIR
    assert capsys.readouterr().err == ""

    assert door.resolve_repo(tmp_path) == door.DRIVERS_DIR
    note = capsys.readouterr().err
    assert f"--repo {tmp_path} is not used" in note


def test_a_damaged_install_names_the_missing_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(door, "DRIVERS_DIR", tmp_path)
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_repo(None)
    message = str(caught.value)
    assert "run_cuda_v841_forecast.py" in message
    assert "Reinstall recast-woof" in message


def test_the_seam_tree_defaults_to_the_installed_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--gpuwm-checkout`` is optional: with none given the door checks the
    installed engine's sixteen seam files, and with no engine installed it
    refuses naming the pip line rather than a clone."""

    from woof.hex import engine_pin

    (tmp_path / "a").mkdir()
    arguments = _namespace(tmp_path / "a", gpuwm_checkout=None)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: None)
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.resolve_request(arguments, registry=_registry())
    message = str(caught.value)
    assert "no woof is installed" in message
    assert "pip install" in message

    seen: list[Path] = []
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(engine_pin, "installed_root", lambda: tmp_path / "b" / "woof")
    monkeypatch.setattr(
        door, "backend_pin_problem",
        lambda row, checkout: seen.append(checkout) or None,
    )
    arguments = _namespace(tmp_path / "b", gpuwm_checkout=None)
    try:
        door.resolve_request(arguments, registry=_registry())
    except door.ForecastDoorRefusal:
        pass  # later checks on the stand-in files may refuse; the seam tree is what is asserted
    assert seen and seen[0] == tmp_path / "b" / "woof"


# ---------------------------------------------------------------------------
# the architecture half of admission
# ---------------------------------------------------------------------------


def test_architecture_at_the_proven_floor_is_admitted(monkeypatch):
    monkeypatch.setattr(door, "read_device_compute", lambda: (12, 0))
    verdict = door.admit_architecture()
    assert verdict["admitted"] is True
    assert verdict["sm"] == "sm_120"
    assert verdict["anchor_status"] == "anchored"
    assert "proven contract floor" in verdict["basis"]
    assert "numeric_route" not in verdict


def test_unanchored_architecture_is_admitted_and_says_so(monkeypatch):
    """Fixed means default: a card of an architecture holding no anchor runs
    with no flag, and the record says it is unanchored rather than claiming
    an anchor.  The numeric route is measured on it and recorded."""

    from woof.hex.cuda_backend import arch_admission

    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    monkeypatch.setattr(door, "read_device_compute", lambda: (8, 6))
    monkeypatch.setattr(door, "probe_numeric_route", _anchored_route)
    verdict = door.admit_architecture()
    assert verdict["admitted"] is True
    assert verdict["sm"] == "sm_86"
    assert verdict["anchor_status"] == "unanchored"
    assert verdict["anchor"] is None and verdict["evidence_sha256"] is None
    assert verdict["basis"].startswith("unanchored:")
    assert "the same as on the anchored architectures" in verdict["basis"]
    assert verdict["numeric_route"]["matches_anchored_route"] is True


def test_anchored_architecture_is_admitted_with_its_pin_named(monkeypatch):
    from woof.hex.cuda_backend import arch_admission

    anchor = arch_admission.ArchAnchor(
        compute=(8, 6),
        card="RTX 3080 (test double)",
        admitted_on="2026-08-25",
        contract_receipt="evidence/sm86-tier-20260825/RECEIPT.md",
        authority_anchor="evidence/sm86-tier-20260825/authority",
        basis="test-registered anchor",
        evidence_sha256="75437c663c9894dc38e6a1b1517709906e08a14c9a56468016aec09a8fa93155",
    )
    monkeypatch.setattr(
        arch_admission, "ADMITTED_BELOW_FLOOR", {(8, 6): anchor}
    )
    monkeypatch.setattr(door, "read_device_compute", lambda: (8, 6))
    verdict = door.admit_architecture()
    assert verdict["admitted"] is True
    assert verdict["sm"] == "sm_86"
    assert verdict["anchor_status"] == "anchored"
    assert verdict["evidence_sha256"] == anchor.evidence_sha256
    assert "sm86-tier-20260825" in verdict["basis"]
    assert "75437c663c9894dc" in verdict["basis"]


def test_a_card_nvrtc_cannot_compile_for_is_refused_naming_nvrtc(monkeypatch):
    monkeypatch.setattr(door, "read_device_compute", lambda: (6, 1))
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.admit_architecture()
    message = str(caught.value)
    assert "sm_61" in message and "compute_75" in message
    assert "none of them could be built" in message
    assert "anchor" not in message


def test_ordinary_arithmetic_that_is_not_ieee_is_refused_naming_it(monkeypatch):
    from woof.hex.cuda_backend import arch_admission

    def _broken():
        record = _anchored_route()
        record["normal_range"] = "differs"
        return record

    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    monkeypatch.setattr(door, "read_device_compute", lambda: (8, 6))
    monkeypatch.setattr(door, "probe_numeric_route", _broken)
    with pytest.raises(door.ForecastDoorRefusal) as caught:
        door.admit_architecture()
    assert "IEEE 754" in str(caught.value)


def test_preflight_admits_an_unanchored_card_and_its_receipt_says_so(
    monkeypatch, tmp_path
):
    """Rung 4 of the small-card walk stays closed: preflight and the run
    answer the architecture question the same way.  An unanchored card now
    passes the architecture half, and the preflight receipt records its
    compute capability, that it is unanchored and the route measured on it."""

    from woof.hex.cuda_backend import arch_admission

    monkeypatch.setattr(arch_admission, "ADMITTED_BELOW_FLOOR", {})
    monkeypatch.setattr(door, "read_device_compute", lambda: (7, 5))
    monkeypatch.setattr(door, "probe_numeric_route", _anchored_route)
    monkeypatch.setattr(
        door, "measure_device_memory", lambda: (11 * GIB, 12 * GIB)
    )
    arguments = _namespace(
        tmp_path, preflight=True, receipt=tmp_path / "receipt.json"
    )
    registry = _registry()
    request = door.resolve_request(arguments, registry=registry)
    door._run_preflight(request, registry, started=0.0)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert not any("sm_75" in problem for problem in receipt["preflight_problems"])
    assert receipt["architecture"]["compute_capability"] == "7.5"
    assert receipt["architecture"]["anchor_status"] == "unanchored"
    assert receipt["architecture"]["evidence_sha256"] is None
    assert receipt["architecture"]["numeric_route"]["normal_range"] == "ieee"


def test_preflight_reports_the_nvrtc_refusal(monkeypatch, tmp_path):
    monkeypatch.setattr(door, "read_device_compute", lambda: (6, 1))
    monkeypatch.setattr(
        door, "measure_device_memory", lambda: (11 * GIB, 12 * GIB)
    )
    arguments = _namespace(
        tmp_path, preflight=True, receipt=tmp_path / "receipt.json"
    )
    registry = _registry()
    request = door.resolve_request(arguments, registry=registry)
    rc = door._run_preflight(request, registry, started=0.0)
    assert rc == 1
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    problems = receipt["preflight_problems"]
    assert any("sm_61" in problem and "NVRTC" in problem for problem in problems)
    assert receipt["architecture"] is None


# ---------------------------------------------------------------------------
# the physics backend row and the point-source table
# ---------------------------------------------------------------------------
def test_help_names_the_backend_row_and_the_source_table() -> None:
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    text = parser.format_help()
    for flag in ("--physics-backend", "--source-table"):
        assert flag in text, flag


def test_an_unregistered_physics_backend_refuses_naming_the_rows(tmp_path):
    message = _refusal(_namespace(tmp_path, physics_backend="nope"))
    assert "--physics-backend nope" in message
    assert "wsm6_column" in message


def test_a_source_table_on_the_frozen_row_refuses_as_never_read(tmp_path):
    table = tmp_path / "sources.txt"
    table.write_text("# epoch lat lon alt on rate src\n")
    message = _refusal(_namespace(tmp_path, source_table=table))
    assert "carries no point source" in message
    assert "never read" in message


def _provider_row_name() -> str:
    """A PROVIDER row's name, read off the registry rather than spelled here.

    This file ships in the public tree and no provider does, so a
    provider's own vocabulary must not be a byte of this file: the row is
    whichever non-default row the installed providers registered.  Skips by
    name where no provider is installed.
    """

    from woof.hex import physics_backend_admission as admission

    names = sorted(
        name
        for name in admission.backend_rows()
        if name not in admission.IN_TREE_ROWS
    )
    if not names:
        pytest.skip(
            "no row provider is installed (entry-point group "
            f"{admission.ROW_ENTRY_POINT_GROUP!r}); the public install resolves "
            "the frozen row only"
        )
    return names[0]


def test_a_provider_row_is_pinned_by_its_own_adapter_at_the_door(tmp_path):
    """The frozen manifest can never verify a provider's engine, so the
    door asks the row's adapter; the refusal is the second manifest's
    own text, at the door, before a card is touched."""
    provider = _provider_row_name()
    message = _refusal(_namespace(tmp_path, physics_backend=provider))
    assert "pinned sibling source is missing" in message
    assert "re-pin" in message or "REMEDY" in message


def test_preflight_carries_the_row_and_the_table_into_the_driver_argv(tmp_path):
    provider = _provider_row_name()
    table = tmp_path / "sources.txt"
    table.write_text("# epoch lat lon alt on rate src\n")
    arguments = _namespace(
        tmp_path, physics_backend=provider, source_table=table, preflight=True
    )
    request = door.resolve_request(arguments, registry=_registry())
    assert request.physics_backend == provider
    assert request.source_table == table.absolute()
    # The pin problem is COLLECTED in preflight, and it is the row's own.
    assert any("pinned sibling source is missing" in p for p in request.input_problems)
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--physics-backend") + 1] == provider
    assert argv[argv.index("--source-table") + 1] == str(table.absolute())


def test_a_default_run_carries_no_backend_token(tmp_path):
    """Byte-stable: the frozen row's argv and receipt shape do not move."""
    request = door.resolve_request(
        _namespace(tmp_path, preflight=True), registry=_registry()
    )
    assert request.physics_backend == "wsm6_column"
    assert request.source_table is None
    argv = door.build_driver_argv(request)
    assert "--physics-backend" not in argv
    assert "--source-table" not in argv


def test_boundary_fingerprint_interval_reaches_the_driver(tmp_path: Path) -> None:
    request = door.resolve_request(
        _namespace(tmp_path, fingerprint_every=720), registry=_registry()
    )
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--fingerprint-every") + 1] == "720"


def test_negative_boundary_fingerprint_interval_refuses(tmp_path: Path) -> None:
    with pytest.raises(door.ForecastDoorRefusal, match="nonnegative"):
        door.resolve_request(
            _namespace(tmp_path, fingerprint_every=-1), registry=_registry()
        )


@pytest.mark.parametrize("first,alias", [(720, 0), (0, 720), (720, 720)])
def test_boundary_fingerprint_alias_has_one_effective_schedule(tmp_path, first, alias):
    request = door.resolve_request(
        _namespace(tmp_path, fingerprint_every=first, boundary_fingerprint_every=alias),
        registry=_registry(),
    )
    argv = door.build_driver_argv(request)
    assert argv.count("--fingerprint-every") == 1
    assert argv[argv.index("--fingerprint-every") + 1] == "720"


def test_conflicting_boundary_fingerprint_aliases_refuse(tmp_path):
    with pytest.raises(door.ForecastDoorRefusal, match="two different identity schedules"):
        door.resolve_request(
            _namespace(tmp_path, fingerprint_every=720, boundary_fingerprint_every=360),
            registry=_registry(),
        )


def test_resident_member_publishes_standard_forecast_discovery_receipt(tmp_path):
    from woof.hex import multi_member, products_door
    request = door.resolve_request(_namespace(tmp_path), registry=_registry())
    request.out.mkdir()
    history = request.out / 'cuda-history.2026-08-24_00.00.00.nc'
    history.write_bytes(b'frame discovery fixture')
    multi_member.publish_forecast_receipt(door, request, {'steps': request.steps}, {'mesh': request.mesh},
                                         {'sm': '120'}, 7.25, 'passed')
    receipt = products_door.forecast_receipt(request.out)
    assert receipt['mesh']['grid'] == str(request.grid)
    assert receipt['init']['path'] == str(request.init)
    assert receipt['history'] == [str(history)]
    assert receipt['schedule']['start_time'] == request.start_time
    assert receipt['door_seconds'] == 7.25 and receipt['status'] == 'passed'
