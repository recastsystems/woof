"""The region-global engine: the selector, the handshake, the refusal.

The solver itself is the project's vendored crate and is tested by its own Rust
suite (``cargo test`` in ``tools/region_global_dealias``) and, on real
sweeps, against Py-ART itself -- see
``evidence/dealias-region/pyart-parity.json``.  What is tested here is the
part this tree owns: that ``engine`` selects, that the ABI is checked
before a solve rather than after, that a missing library is a named
refusal rather than a traceback, and that the result honours the same
three-state contract the VAD engine does.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

from woof.obs import dealias_region
from woof.obs.dealias import (ENGINE_REGION_GLOBAL, ENGINE_VAD_REGION,
                               ENGINES, REASON_NO_NYQUIST, REASON_NONE,
                               REASON_NONFINITE, REASON_SPEED, STATE_REJECTED,
                               STATE_UNCHANGED, STATE_UNFOLDED,
                               DealiasParams, DealiasParamsError,
                               dealias_sweep, engine_unavailable_reason)

#: Every gate one real 14-cut volume's region-global solve put past the
#: physical bound, with the couplet's own two gates beside them.  Written
#: by ``tools/dealias_engine_compare.py --speed-bound-fixture``.
SPEED_BOUND_FIXTURE = (Path(__file__).resolve().parent / "data"
                       / "dealias_speed_bound_gates.json")


def _engine_or_skip():
    try:
        return dealias_region.load_region_dealiaser()
    except FileNotFoundError as error:                     # pragma: no cover
        pytest.skip(f"the region-global engine is not built: {error}")


def _folded_sweep(rows: int = 72, gates: int = 200, nyquist: float = 10.0):
    """A sweep with one real fold in it, and the truth it came from.

    A solid-body-plus-uniform-flow field: the uniform part is what a
    region method resolves against, and the amplitude is chosen so a
    contiguous arc exceeds Nyquist and wraps.
    """

    azimuth = np.linspace(0.0, 360.0, rows, endpoint=False)
    radians = np.radians(azimuth)[:, None]
    ramp = np.linspace(0.4, 1.0, gates)[None, :]
    truth = 16.0 * np.cos(radians) * ramp
    interval = 2.0 * nyquist
    folded = truth - interval * np.rint(truth / interval)
    return azimuth, truth, folded.astype(np.float64), nyquist


#: A WSR-88D super-resolution velocity cut's range geometry.  The default
#: engine's refinement pass fits a wrapped vortex in real space, so every
#: caller of the default path supplies these -- ``woof.obs.superob`` from
#: the sweep's own moment, and these tests from the same numbers a real
#: cut carries.
FIRST_GATE_M = 2125.0
GATE_SPACING_M = 250.0


def _region_sweep(folded, azimuth, nyquist, params):
    """The default path: the engine, its refinement, and the geometry."""

    return dealias_sweep(folded, azimuth, nyquist, params,
                         first_gate_m=FIRST_GATE_M,
                         gate_spacing_m=GATE_SPACING_M)


class TestTheSelector:
    def test_the_default_engine_is_region_global_and_refines(self):
        """The owner's call, arriving as a diff in the test that pinned it.

        The previous pin here said the default must not move without one:
        this is that change.  Ruling of 2026-08-12, on the measurement in
        ``evidence/dealias-region/`` -- region-global keeps 3,894 more
        velocity cells per volume than the abstainer and recovers the
        couplet the abstainer rejects, and refinement rides with it
        because it self-declines.
        """

        params = DealiasParams()
        assert params.engine == ENGINE_REGION_GLOBAL
        assert params.refinement is True
        assert ENGINES == (ENGINE_VAD_REGION, ENGINE_REGION_GLOBAL)

    def test_the_legacy_engine_stays_selectable_without_a_second_flag(self):
        """It is a supported engine, not a deprecated one.

        Refinement is meaningless there and refused when asked for, so an
        unqualified default of True would make ``vad-region`` unusable
        unless the caller also turned a switch off for an engine that has
        no such pass.  The default is the ENGINE's, which is what the
        tri-state exists for.
        """

        params = DealiasParams(engine=ENGINE_VAD_REGION)
        assert params.engine == ENGINE_VAD_REGION
        assert params.refinement is False
        assert params.to_payload()["refinement"] is False

    def test_an_unknown_engine_is_refused_rather_than_defaulted(self):
        with pytest.raises(DealiasParamsError) as error:
            DealiasParams(engine="region_global")
        assert "known engines" in str(error.value)

    def test_refinement_without_the_engine_that_has_it_is_refused(self):
        """A switch that is accepted and then ignored is a false treatment."""

        with pytest.raises(DealiasParamsError) as error:
            DealiasParams(engine=ENGINE_VAD_REGION, refinement=True)
        assert "no refinement pass" in str(error.value)
        # And it is legal on the engine that does have one.
        assert DealiasParams(engine=ENGINE_REGION_GLOBAL,
                             refinement=True).refinement is True

    def test_the_engine_reaches_provenance(self):
        payload = DealiasParams(engine=ENGINE_REGION_GLOBAL,
                                refinement=False).to_payload()
        assert payload["engine"] == ENGINE_REGION_GLOBAL
        # A resolved bool, never the word "default": provenance states
        # what ran.
        assert payload["refinement"] is False
        assert DealiasParams().to_payload()["refinement"] is True
        # Still JSON-clean: the writer refuses anything that is not.
        import json
        assert json.loads(json.dumps(payload))["engine"] \
            == ENGINE_REGION_GLOBAL

    def test_dealias_sweep_dispatches_on_the_engine(self, monkeypatch):
        """The selector is what routes, not the caller's import."""

        seen = {}

        def fake(velocity, azimuth, nyquist, params, **kwargs):
            seen["called"] = True
            seen["kwargs"] = kwargs
            return "region-result"

        monkeypatch.setattr(dealias_region, "dealias_sweep_region", fake)
        out = dealias_sweep(np.zeros((4, 4)), np.zeros(4), 10.0,
                            DealiasParams(engine=ENGINE_REGION_GLOBAL),
                            first_gate_m=2125.0, gate_spacing_m=250.0)
        assert out == "region-result"
        assert seen["called"]
        # The geometry the refinement pass needs is forwarded, not dropped.
        assert seen["kwargs"]["first_gate_m"] == 2125.0
        assert seen["kwargs"]["gate_spacing_m"] == 250.0


class TestTheAbiHandshake:
    def test_the_wrapper_and_the_library_agree(self):
        engine = _engine_or_skip()
        assert engine.abi_version == dealias_region.REGION_DEALIAS_ABI
        assert engine.rift_api_version \
            == dealias_region.REGION_DEALIAS_RIFT_API
        assert engine.refinement_available

    def test_a_library_reporting_another_abi_is_refused_before_any_solve(
            self, monkeypatch):
        """The handshake runs at load, not at the first bad answer.

        A library built from a different contract does not fail loudly on
        its own -- it returns numbers.  This pins that the version is
        asked for first and that the refusal names the crate to rebuild.
        """

        engine = _engine_or_skip()
        real = ctypes.CDLL

        class WrongAbi:
            def __init__(self, path):
                self._real = real(path)

            def __getattr__(self, name):
                if name == "bw_abi_version":
                    stub = lambda: 99                      # noqa: E731
                    stub.argtypes = []
                    stub.restype = None
                    return stub
                return getattr(self._real, name)

        monkeypatch.setattr(ctypes, "CDLL", WrongAbi)
        with pytest.raises(dealias_region.RegionDealiasError) as error:
            dealias_region.RegionDealiaser(engine.path)
        message = str(error.value)
        assert "ABI 99" in message
        assert dealias_region.CRATE_RELATIVE in message

    def test_the_structure_layouts_match_the_published_header(self):
        """20/48/64 bytes, asserted by the crate's own header too.

        A ctypes structure that disagrees with the C one reads garbage
        out of a valid call, which is the failure mode no return code
        catches.
        """

        assert ctypes.sizeof(dealias_region.BwStats) == 20
        assert ctypes.sizeof(dealias_region.BwRiftOptionsV1) == 48
        assert ctypes.sizeof(dealias_region.BwRiftStatsV1) == 64
        header = (Path(dealias_region.crate_dir())
                  / "region_global_dealias.h").read_text(encoding="utf-8")
        for size in ("sizeof(BwStats) == 20",
                     "sizeof(BwRiftOptionsV1) == 48",
                     "sizeof(BwRiftStatsV1) == 64"):
            assert size in header

    def test_the_recorded_upstream_commit_is_a_full_sha(self):
        """Provenance of the algorithm, not just of the wrapper."""

        commit = dealias_region.UPSTREAM_COMMIT
        assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")


class TestTheRefusalWhenTheLibraryIsMissing:
    def test_an_absent_library_names_every_place_it_looked(self, monkeypatch,
                                                           tmp_path):
        monkeypatch.delenv(dealias_region.REGION_DEALIAS_ENV, raising=False)
        monkeypatch.setattr(dealias_region, "region_bridge_candidates",
                            lambda: (tmp_path / "nowhere.dll",))
        monkeypatch.setattr(dealias_region, "find_region_bridge",
                            lambda: None)
        with pytest.raises(FileNotFoundError) as error:
            dealias_region.resolve_region_bridge()
        message = str(error.value)
        assert "nowhere.dll" in message
        # The remedy names the crate to build.  Its path is spelled for the
        # shell the reader will paste into -- backslashes on Windows -- so
        # the directory name is what is pinned, not the separator.
        assert "region_global_dealias" in message
        assert "cargo build" in message

    def test_an_override_naming_a_missing_file_is_a_hard_error(
            self, monkeypatch, tmp_path):
        """Explicit configuration must never fall through to something else.

        The same contract :func:`woof.bridges.find_artifact` holds: a
        typo'd override that silently used a different library is a run
        that reports the wrong engine's provenance.
        """

        missing = tmp_path / "not-here.dll"
        monkeypatch.setenv(dealias_region.REGION_DEALIAS_ENV, str(missing))
        with pytest.raises(FileNotFoundError) as error:
            dealias_region.find_region_bridge()
        assert dealias_region.REGION_DEALIAS_ENV in str(error.value)
        assert "not-here.dll" in str(error.value)

    def test_the_front_door_reports_it_before_a_run_starts(self, monkeypatch):
        """Same reason scipy is checked at the door for the other engine."""

        monkeypatch.setattr(dealias_region, "region_engine_available",
                            lambda: False)
        monkeypatch.setattr(dealias_region, "region_bridge_remedy",
                            lambda: "REMEDY-TEXT")
        reason = engine_unavailable_reason(ENGINE_REGION_GLOBAL)
        assert reason is not None and "REMEDY-TEXT" in reason

    def test_an_unknown_engine_has_a_reason_too(self):
        assert "unknown dealiasing engine" \
            in engine_unavailable_reason("nope")


class TestTheThreeStateContract:
    def test_a_real_fold_is_unfolded_and_the_states_account_for_every_gate(
            self):
        _engine_or_skip()
        azimuth, truth, folded, nyquist = _folded_sweep()
        params = DealiasParams(engine=ENGINE_REGION_GLOBAL)
        result = _region_sweep(folded, azimuth, nyquist, params)

        # The sweep really was folded, or this test proves nothing.
        assert result.stats["gates_unfolded"] > 0
        assert np.abs(folded - truth).max() > nyquist

        # Every gate lands in exactly one state, and they add up.
        total = (result.stats["gates_unchanged"]
                 + result.stats["gates_unfolded"]
                 + result.stats["gates_rejected"])
        assert total == result.stats["gates_finite"] == folded.size
        assert set(np.unique(result.state)) <= {STATE_REJECTED,
                                                STATE_UNCHANGED,
                                                STATE_UNFOLDED}
        # Recovered to the truth up to the whole-sweep constant a
        # single-sweep solver cannot know (Py-ART centres the sweep the
        # same way), so the SHAPE is what is checked.
        residual = result.velocity - truth
        assert np.allclose(residual, np.median(residual), atol=1e-3)

    def test_the_account_balances_when_most_of_the_sweep_is_no_data(self):
        """The identity is over gates OFFERED, not over the plane.

        A radar sweep is mostly empty -- the volume this lane measured on
        is 1.36 M finite gates in a 4.1 M gate rectangle -- and
        STATE_REJECTED is zero, so a count taken over the whole plane
        reports every no-data gate as a refusal.  Measured on that volume
        before this was fixed: 4,073,183 "rejected" gates against
        1,355,617 offered, and the volume totals' own balance assertion
        went false.
        """

        _engine_or_skip()
        azimuth, _truth, folded, nyquist = _folded_sweep()
        folded = folded.copy()
        folded[:, 120:] = np.nan
        result = _region_sweep(folded, azimuth, nyquist,
                               DealiasParams(engine=ENGINE_REGION_GLOBAL))
        offered = int(np.isfinite(folded).sum())
        assert result.stats["gates_finite"] == offered < folded.size
        assert (result.stats["gates_unchanged"]
                + result.stats["gates_unfolded"]
                + result.stats["gates_rejected"]) == offered
        assert result.stats["gates_rejected"] == 0

    def test_an_unmoved_gate_comes_back_exactly(self):
        """The crate's own claim, checked here because woof relies on it.

        ``dealias_sweep_region`` recovers the integer fold by dividing the
        difference; that is only sound because the engine moves gates by
        whole intervals and returns the rest untouched.
        """

        _engine_or_skip()
        azimuth, _truth, folded, nyquist = _folded_sweep()
        result = _region_sweep(folded, azimuth, nyquist,
                               DealiasParams(engine=ENGINE_REGION_GLOBAL))
        still = result.state == STATE_UNCHANGED
        assert still.any()
        assert np.array_equal(result.velocity[still].astype(np.float32),
                              folded[still].astype(np.float32))
        moved = result.state == STATE_UNFOLDED
        applied = (result.velocity[moved] - folded[moved]) / (2.0 * nyquist)
        assert np.allclose(applied, np.rint(applied), atol=1e-3)
        assert np.array_equal(np.rint(applied).astype(np.int16),
                              result.fold[moved])

    def test_no_data_gates_stay_no_data_with_their_reason(self):
        _engine_or_skip()
        azimuth, _truth, folded, nyquist = _folded_sweep()
        folded = folded.copy()
        folded[3:7, 10:40] = np.nan
        result = _region_sweep(folded, azimuth, nyquist,
                               DealiasParams(engine=ENGINE_REGION_GLOBAL))
        hole = ~np.isfinite(folded)
        assert np.all(result.state[hole] == STATE_REJECTED)
        assert np.all(result.reason[hole] == REASON_NONFINITE)
        assert np.all(~np.isfinite(result.velocity[hole]))
        assert np.all(result.reason[~hole] == REASON_NONE)

    def test_an_unbelievable_nyquist_is_refused_by_BOTH_engines(self):
        """A gate whose Nyquist is unknown has an unknown fold state.

        The native solver would fall back to the sweep median and pass the
        velocities through, which is right for a display and wrong for a
        filter.  woof makes the same refusal whichever engine is chosen,
        so an operator cannot change that answer by changing solver.
        """

        azimuth, _truth, folded, _nyquist = _folded_sweep()
        for engine in ENGINES:
            result = _region_sweep(folded, azimuth, None,
                                   DealiasParams(engine=engine))
            assert result.stats["gates_rejected"] == folded.size
            assert result.stats["rejected"]["no_nyquist"] == folded.size
            assert np.all(result.reason == REASON_NO_NYQUIST)
            assert not np.isfinite(result.velocity).any()

    def test_this_engine_carries_no_environmental_reference(self):
        """All NaN, rather than a plane that states a fit nobody made."""

        _engine_or_skip()
        azimuth, _truth, folded, nyquist = _folded_sweep()
        result = _region_sweep(folded, azimuth, nyquist,
                               DealiasParams(engine=ENGINE_REGION_GLOBAL))
        assert not np.isfinite(result.reference).any()
        assert result.stats["reference"]["external_supplied"] is False
        assert result.stats["engine"] == ENGINE_REGION_GLOBAL


class TestTheSuperobSeam:
    def test_a_background_wind_beside_this_engine_is_refused(self, tmp_path):
        """Not ignored: ignored is how a run claims a treatment it lacked.

        The VAD engine anchors against a supplied ``WindProfile``; the
        region-global engine has no reference input at all. Handing one to
        the second would leave provenance saying a background wind was
        supplied over velocities that never saw it.
        """

        from woof.obs.dealias import WindProfile
        from woof.obs.superob import SuperobParams, superob_volume

        profile = WindProfile(height_m=np.array([0.0, 1000.0]),
                              u_ms=np.array([5.0, 10.0]),
                              v_ms=np.array([1.0, 2.0]))
        params = SuperobParams(
            dealias=DealiasParams(engine=ENGINE_REGION_GLOBAL))
        with pytest.raises(ValueError) as error:
            superob_volume(object(), object(), params=params,
                           velocity_reference=profile)
        assert "carries no environmental reference" in str(error.value)


class TestTheDoctorLine:
    """``woof doctor`` reports the engine, and now BLOCKS on it.

    The status is the finding, and it changed with the default: while
    ``vad-region`` shipped, an absent library cost an option and the line
    was ``info``.  ``region-global`` is the default engine now, so an
    absent library costs every dealiased velocity in the nowcast -- on a
    path a run reaches an hour in -- and a doctor that called that
    ``info`` would be reporting a green estate that cannot run the
    default configuration.
    """

    @staticmethod
    def _check(monkeypatch, *, found):
        if isinstance(found, BaseException):
            def _find():
                raise found
        else:
            def _find():
                return found
        monkeypatch.setattr(dealias_region, "find_region_bridge", _find)
        from woof import doctor
        return doctor._region_dealias_check()

    def test_the_estate_asks_about_the_engine_at_all(self):
        """Read off the assembling function: collect_checks() probes CUDA.

        ``_collect_checks`` is that function; the public
        ``collect_checks`` is a thin wrapper holding the reader scope
        every resolution in the report runs inside
        (``woof.bridges.inspection_only``).  Read the assembler
        wherever it lives, falling back to the public name so this keeps
        asking its question if the two are ever merged.
        """

        import inspect

        from woof import doctor

        assembler = getattr(doctor, "_collect_checks", doctor.collect_checks)
        source = inspect.getsource(assembler)
        assert "_region_dealias_check()" in source

    def test_an_absent_library_BLOCKS_and_the_remedy_is_the_build(
            self, monkeypatch):
        check = self._check(monkeypatch, found=None)
        assert check.status == "missing"
        assert "blocks the default velocity dealiasing path" in check.detail
        remedy = check.remedy or ""
        # Both routes to the library, and the other engine named as the
        # different solver it is rather than as a way around the gap.
        assert "region_global_dealias" in remedy
        assert "woof fetch-bridges" in remedy
        assert "DIFFERENT solver" in remedy

    def test_an_override_naming_a_missing_file_is_reported_missing(
            self, monkeypatch):
        check = self._check(
            monkeypatch,
            found=FileNotFoundError(
                f"{dealias_region.REGION_DEALIAS_ENV} names a missing file: "
                "C:/nope.dll"))
        assert check.status == "missing"
        assert dealias_region.REGION_DEALIAS_ENV in check.detail

    def test_a_built_library_is_verified_with_its_abi_and_upstream(
            self, monkeypatch):
        engine = _engine_or_skip()
        check = self._check(monkeypatch, found=engine.path)
        assert check.status == "verified"
        assert f"ABI {engine.abi_version}" in check.detail
        assert dealias_region.UPSTREAM_COMMIT[:12] in check.detail


class TestAnInstallThatDoesNotHaveIt:
    """A wheel install with no library refuses, with the remedy, at the door.

    This is the failure the default flip creates, so it is the failure
    that gets a test: `pip install recast-woof` ships no compiled Rust, and the
    engine `--dealias` now runs by default is compiled Rust.  What must
    happen is a named refusal naming both routes to the file -- never a
    traceback from inside a cycle, and never a quiet fall back to the
    other solver, which would report an engine in provenance that is not
    the one the operator asked for.
    """

    @staticmethod
    def _absent(monkeypatch):
        monkeypatch.setattr(dealias_region, "find_region_bridge",
                            lambda: None)
        monkeypatch.setattr(dealias_region, "region_engine_available",
                            lambda: False)

    def test_the_observation_front_door_refuses_before_a_byte_is_fetched(
            self, monkeypatch):
        import argparse

        from tools.obs_radar_grid_build import dealias_params_from_args

        self._absent(monkeypatch)
        args = argparse.Namespace(dealias=True,
                                  dealias_engine=ENGINE_REGION_GLOBAL,
                                  dealias_refinement=None)
        with pytest.raises(SystemExit) as error:
            dealias_params_from_args(args, DealiasParams,
                                     engine_unavailable_reason)
        message = str(error.value)
        assert "region-global" in message
        assert "region_global_dealias" in message
        # A remedy, not an apology: the crate to build is named, and so
        # is the command that stages it prebuilt.
        assert "cargo build" in message or "fetch-bridges" in message

    def test_the_legacy_engine_still_runs_without_it(self, monkeypatch):
        """Selectable means selectable, including on a box without a build."""

        self._absent(monkeypatch)
        assert engine_unavailable_reason(ENGINE_VAD_REGION) is None
        azimuth, _truth, folded, nyquist = _folded_sweep()
        result = dealias_sweep(folded, azimuth, nyquist,
                               DealiasParams(engine=ENGINE_VAD_REGION))
        assert result.stats["gates_finite"] == folded.size


class TestTheFoldIslandInstrument:
    """The neighbour-consistency check the couplet ruling data rests on.

    It is a measuring instrument, so it is tested against known answers in
    BOTH directions before any number it produced is quoted: it must find
    a fold that is there, and must not find one that is not.
    """

    @staticmethod
    def _smooth(rows: int = 40, gates: int = 60):
        azimuth = np.radians(np.linspace(0.0, 360.0, rows, endpoint=False))
        ramp = np.linspace(0.5, 1.0, gates)
        return 18.0 * np.cos(azimuth)[:, None] * ramp[None, :]

    def test_it_finds_a_gate_displaced_by_exactly_one_interval(self):
        from tools.dealias_engine_compare import fold_islands

        nyquist = 26.0
        field = self._smooth()
        field[17, 33] += 2.0 * nyquist
        count, mask = fold_islands(field, nyquist)
        assert count == 1
        assert mask[17, 33]

    def test_it_finds_nothing_in_a_field_with_no_fold_error(self):
        from tools.dealias_engine_compare import fold_islands

        count, _mask = fold_islands(self._smooth(), 26.0)
        assert count == 0

    def test_a_steep_but_real_gradient_is_not_a_fold(self):
        """A couplet is a large gate-to-gate jump and must survive this."""

        from tools.dealias_engine_compare import fold_islands

        nyquist = 26.0
        field = self._smooth()
        field[:, 30:] += 0.5 * nyquist
        count, _mask = fold_islands(field, nyquist)
        assert count == 0

    def test_a_velocity_EXTREMUM_is_not_a_fold_however_large(self):
        """The false positive the band exists to kill.

        A tornado's peak gate is genuinely isolated -- that is what a
        tornado is at this range -- and a plain large-departure test calls
        it an error.  Measured on the real couplet in the Moore volume, a
        0.6-interval threshold flagged the -58.7 m/s peak itself.  A fold
        error sits at a WHOLE interval; a wind sits wherever the wind is,
        so the peak here is placed at 0.75 intervals: unmistakably large,
        and unmistakably not a fold.
        """

        from tools.dealias_engine_compare import fold_islands

        nyquist = 26.0
        field = self._smooth()
        field[17, 33] = field[17, 33] - 0.75 * 2.0 * nyquist
        count, _mask = fold_islands(field, nyquist)
        assert count == 0

    def test_a_double_fold_is_found_too(self):
        from tools.dealias_engine_compare import fold_islands

        nyquist = 26.0
        field = self._smooth()
        field[17, 33] += 2.0 * (2.0 * nyquist)
        count, mask = fold_islands(field, nyquist)
        assert count == 1 and mask[17, 33]

    def test_it_cannot_see_a_whole_region_on_the_wrong_branch(self):
        """Its documented blind spot, pinned so the limit stays stated."""

        from tools.dealias_engine_compare import fold_islands

        nyquist = 26.0
        field = self._smooth()
        field[10:20, 20:40] += 2.0 * nyquist
        count, _mask = fold_islands(field, nyquist)
        # Only the perimeter can possibly show; the interior is continuous
        # with itself.  The point is that the count is nothing like the
        # 200 gates that are actually wrong.
        assert count < 40

    def test_a_gate_with_too_few_neighbours_is_not_judged(self):
        from tools.dealias_engine_compare import fold_islands

        nyquist = 26.0
        field = np.full((40, 60), np.nan)
        field[17, 33] = 1000.0
        field[17, 34] = 0.0
        field[18, 33] = 0.0
        count, _mask = fold_islands(field, nyquist)
        assert count == 0


class _PreparedSolver:
    """A library that returns an answer prepared by the caller.

    The layer under test is woof's -- the bound, the reason, the account
    -- so the solver is stubbed rather than run: that is what lets the
    fixture's real gates be asserted on a box with no library built and no
    53 MB radar volume in the tree.  The prepared plane still honours the
    contract the real crate does (whole Nyquist intervals only, untouched
    gates returned bit-identically), because
    ``dealias_sweep_region`` checks that before anything else and a stub
    that failed it would prove nothing about the real path.
    """

    def __init__(self, out: np.ndarray, path: str = "<prepared>"):
        self.out = np.asarray(out, dtype=np.float32)
        self.path = path

    def _stats(self, velocity) -> dict:
        finite = np.isfinite(np.asarray(velocity))
        return {"gates_total": int(np.asarray(velocity).size),
                "gates_finite": int(finite.sum()), "gates_modified": 0,
                "max_abs_fold": 0, "wraps": False}

    def dealias(self, velocity, azimuth_deg, nyquist_ms):
        return self.out, self._stats(velocity)

    def dealias_rift(self, velocity, azimuth_deg, nyquist_ms, *,
                     first_gate_m, gate_spacing_m):
        stats = self._stats(velocity)
        stats.update({"rois_detected": 0, "rois_solved": 0,
                      "rois_accepted": 0, "gates_refined": 0,
                      "gates_ambiguous": 0, "budget_aborts": 0,
                      "reasons": [], "abstained": []})
        return self.out, stats


class TestThePhysicalSpeedBound:
    """The obs-QC layer on an engine that abstains at nothing else.

    The region-global solver assigns a fold to every region it resolved
    and refuses nothing, so a region placed on the wrong branch arrives as
    a finite, confident 80-115 m/s wind.  The regression fixture is that
    failure measured on one real volume: 115 gates of 1,355,617, 91 of
    them on two mid-level cuts, against a violent low-level couplet in the
    same volume whose gates must not be touched.
    """

    @staticmethod
    def _fixture() -> dict:
        return json.loads(SPEED_BOUND_FIXTURE.read_text(encoding="utf-8"))

    @staticmethod
    def _plane(rows: list[dict], nyquist: float):
        """One synthetic sweep carrying these gates, raw in and solved out.

        The gates are re-laid onto a compact plane rather than their
        original (ray, gate) coordinates: the bound is a per-gate test, so
        what has to be preserved is each gate's raw velocity, its fold and
        the Nyquist it was measured at -- and a plane the size of the real
        cut would be 850k mostly-empty gates per assertion.  The solved
        plane is built as ``raw + fold * 2 * Vn`` rather than from the
        recorded speed, which makes the recorded speed an independent
        check of the fixture rather than an input to itself.
        """

        width = 16
        height = max(2, -(-len(rows) // width))
        raw = np.full((height, width), np.nan)
        solved = np.full((height, width), np.nan)
        expected = np.full((height, width), np.nan)
        interval = 2.0 * nyquist
        for index, row in enumerate(rows):
            here = (index // width, index % width)
            raw[here] = row["raw_ms"]
            solved[here] = row["raw_ms"] + row["fold"] * interval
            expected[here] = row["unfolded_ms"]
        assert np.allclose(solved[np.isfinite(solved)],
                           expected[np.isfinite(expected)], atol=2e-3), (
            "the fixture's recorded speeds disagree with raw + fold * 2Vn; "
            "it was written by a different solver than it says")
        azimuth = np.linspace(0.0, 360.0, height, endpoint=False)
        return raw, solved, azimuth

    def _run(self, monkeypatch, rows, nyquist, *, max_speed_ms):
        raw, solved, azimuth = self._plane(rows, nyquist)
        # A prepared solver can only be injected into the Python reference
        # (the Rust route links the real one), so these bound tests state
        # that route.  The Rust route's bound is held to the reference by
        # tests/test_obs_superob_rust_parity.py.
        monkeypatch.setenv("WOOF_SUPEROB_PYTHON", "1")
        monkeypatch.setattr(dealias_region, "load_region_dealiaser",
                            lambda path=None: _PreparedSolver(solved))
        params = DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=False,
                               max_speed_ms=max_speed_ms)
        return raw, dealias_sweep(raw, azimuth, nyquist, params)

    @staticmethod
    def _by_nyquist(rows: list[dict]) -> dict:
        grouped: dict[float, list[dict]] = {}
        for row in rows:
            grouped.setdefault(float(row["nyquist_ms"]), []).append(row)
        return grouped

    def test_the_fixture_is_the_volume_it_says_it_is(self):
        """A fixture nobody checks is a number nobody measured."""

        fixture = self._fixture()
        rows = fixture["gates_beyond_bound"]
        assert fixture["bound_ms"] == 75.0 == DealiasParams().max_speed_ms
        assert len(rows) == 115
        assert fixture["volume"]["gates_offered"] == 1_355_617
        assert round(max(abs(row["unfolded_ms"]) for row in rows), 1) == 114.5
        # Concentrated, not scattered: two mid-level cuts carry 91 of them.
        per_cut: dict[float, int] = {}
        for row in rows:
            per_cut[row["elevation_angle_deg"]] = \
                per_cut.get(row["elevation_angle_deg"], 0) + 1
        assert per_cut[3.91] == 29 and per_cut[5.08] == 62
        # And the lowest cut -- the one the couplet lives on -- carries none.
        assert 0.53 not in per_cut

    def test_every_gate_past_the_bound_is_rejected_and_counted(
            self, monkeypatch):
        rejected = 0
        for nyquist, rows in self._by_nyquist(
                self._fixture()["gates_beyond_bound"]).items():
            raw, result = self._run(monkeypatch, rows, nyquist,
                                    max_speed_ms=75.0)
            offered = np.isfinite(raw)
            assert np.all(result.state[offered] == STATE_REJECTED)
            assert np.all(result.reason[offered] == REASON_SPEED)
            # Never clamped and never passed: the velocity is gone.
            assert not np.isfinite(result.velocity[offered]).any()
            assert np.all(result.fold[offered] == 0)
            assert result.stats["rejected"]["speed_out_of_range"] == len(rows)
            assert result.stats["gates_rejected"] == len(rows)
            # The account still balances over the gates it was offered.
            assert (result.stats["gates_unchanged"]
                    + result.stats["gates_unfolded"]
                    + result.stats["gates_rejected"]
                    == result.stats["gates_finite"] == int(offered.sum()))
            assert result.stats["max_speed_ms"] == 75.0
            rejected += len(rows)
        assert rejected == 115

    def test_raising_the_bound_admits_them(self, monkeypatch):
        """The A/B that proves the treatment ran at all.

        Same gates, same solver, one parameter moved: every gate the
        default refuses comes back, at the speed the fixture recorded.
        """

        admitted = 0
        for nyquist, rows in self._by_nyquist(
                self._fixture()["gates_beyond_bound"]).items():
            raw, result = self._run(monkeypatch, rows, nyquist,
                                    max_speed_ms=200.0)
            offered = np.isfinite(raw)
            assert result.stats["rejected"]["speed_out_of_range"] == 0
            assert result.stats["gates_rejected"] == 0
            assert np.all(result.state[offered] != STATE_REJECTED)
            recorded = np.array([row["unfolded_ms"] for row in rows])
            assert np.allclose(np.sort(result.velocity[offered]),
                               np.sort(recorded), atol=2e-3)
            admitted += len(rows)
        assert admitted == 115

    def test_the_couplet_is_not_touched_at_the_default_bound(
            self, monkeypatch):
        """The bound must cost the storm nothing.

        The strongest couplet in the same volume is 98 m/s across two
        adjacent beams 197 m apart, and both of its gates sit inside the
        bound -- which is the whole reason the bound is a speed and not a
        departure from a fit.
        """

        rows = self._fixture()["couplet_gates"]
        assert len(rows) == 2
        nyquist = rows[0]["nyquist_ms"]
        raw, result = self._run(monkeypatch, rows, nyquist,
                                max_speed_ms=75.0)
        offered = np.isfinite(raw)
        assert np.all(result.state[offered] == STATE_UNFOLDED)
        assert np.all(result.reason[offered] == REASON_NONE)
        assert result.stats["rejected"]["speed_out_of_range"] == 0
        pair = sorted(float(v) for v in result.velocity[offered])
        assert pair == pytest.approx([-58.74, 39.24], abs=2e-3)
        assert abs(pair[1] - pair[0]) == pytest.approx(97.98, abs=2e-3)

    def test_the_bound_moves_no_gate_that_is_inside_it(self, monkeypatch):
        """Parity: below the bound, the two runs are the same run.

        The bound is an abstention layer, not a filter with an opinion
        about the field: every gate it does not refuse must carry the same
        velocity, the same fold and the same state as it does with the
        bound lifted.
        """

        fixture = self._fixture()
        rows = (fixture["gates_beyond_bound"] + fixture["couplet_gates"])
        for nyquist, group in self._by_nyquist(rows).items():
            raw, bounded = self._run(monkeypatch, group, nyquist,
                                     max_speed_ms=75.0)
            _raw, lifted = self._run(monkeypatch, group, nyquist,
                                     max_speed_ms=1.0e6)
            kept = bounded.state != STATE_REJECTED
            assert np.array_equal(bounded.fold[kept], lifted.fold[kept])
            assert np.array_equal(bounded.state[kept], lifted.state[kept])
            assert np.allclose(bounded.velocity[kept], lifted.velocity[kept],
                               equal_nan=True)
            # And the only gates that differ are the ones it refused.
            refused = np.isfinite(raw) & ~kept
            assert int(refused.sum()) == sum(
                1 for row in group if abs(row["unfolded_ms"]) > 75.0)

    def test_both_engines_spell_the_refusal_the_same_way(self):
        """One reason name, one bound, whichever solver ran."""

        azimuth, _truth, folded, nyquist = _folded_sweep()
        params = DealiasParams(engine=ENGINE_VAD_REGION, max_speed_ms=1.0)
        result = dealias_sweep(folded, azimuth, nyquist, params)
        assert result.stats["rejected"]["speed_out_of_range"] > 0
        assert np.any(result.reason == REASON_SPEED)


class TestTheRefinementPass:
    def test_it_needs_the_physical_gate_geometry(self):
        _engine_or_skip()
        azimuth, _truth, folded, nyquist = _folded_sweep()
        params = DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=True)
        with pytest.raises(ValueError) as error:
            dealias_sweep(folded, azimuth, nyquist, params)
        assert "gate geometry" in str(error.value)

    def test_it_is_additive_and_says_which_pass_ran(self):
        """Refinement returns the region-global result where it abstains.

        Pinned because the whole safety argument for the pass is that it
        can only ever move gates it accepted a proposal for.
        """

        _engine_or_skip()
        azimuth, _truth, folded, nyquist = _folded_sweep()
        plain = _region_sweep(
            folded, azimuth, nyquist,
            DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=False))
        refined = _region_sweep(
            folded, azimuth, nyquist,
            DealiasParams(engine=ENGINE_REGION_GLOBAL, refinement=True))
        assert plain.stats["native"]["refinement"] is False
        assert refined.stats["native"]["refinement"] is True
        # Every gate the refinement did not accept is the plain answer.
        accepted = refined.stats["native"]["rois_accepted"]
        if accepted == 0:
            assert np.array_equal(refined.fold, plain.fold)
        assert "rois_detected" in refined.stats["native"]
