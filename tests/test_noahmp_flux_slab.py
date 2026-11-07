"""The gate on :mod:`woof.core.noahmp_flux_slab`.

The slab packers exist to delete the per-column CPython loop from Noah-MP's two
surface-flux leaves.  A repacking has exactly one interesting failure mode --
it puts a value in the wrong place, quietly -- so the bar here is bitwise
identity with the shipped per-column path, on the packed bytes *and* on the
evaluated outputs, plus negative controls that show the comparison can fail.

What is compared, and why that choice:

* **The bytes the shipped evaluator actually launches.**  The per-column
  packers are not fully public -- ``noahmp_vegeflux_gpu`` packs its six device
  arrays inline inside ``evaluate_vege_flux_calls`` -- so rather than write
  that layout out a third time (which would test a transcription on itself)
  the reference is captured at the launch site: the kernel entry point is
  wrapped, the real evaluator is run, and the six / two arguments it handed the
  kernel are kept.  That is the real bytes, including every ``cp.asarray``
  conversion on the way.  For BARE_FLUX the public ``pack_bare_flux_calls`` is
  compared as well, so both spellings of the reference agree.
* **Heterogeneous columns.**  The oracle decks hold 10 VEGE_FLUX and 27
  BARE_FLUX physical columns.  Both gates need >= 32, so the columns are tiled
  in a stride coprime with the deck length: every block of ``len(deck)`` covers
  the whole deck and *no two adjacent columns are the same case*, which is what
  makes the one-column roll below observable.  No physics value is invented.
* **Varying VEGE_FLUX parameters.**  Every ``P_*`` in the VEGE_FLUX oracle deck
  is constant -- the deck is one vegetation type -- so a parameter slab built
  from it alone is uniform and a transposition inside it would be invisible.
  The three vegetated MODIS classes in ``noahmp-parameters.csv`` (the WRF
  parameter oracle, same provenance) are cycled alongside the deck's own set so
  that HVT, VCMX25, MP, BP, QE25 and FOLNMX actually differ between columns.
  That changes the physics away from any oracle-verified combination, which is
  fine and deliberate: this file compares two *implementations* against each
  other on identical inputs, not either one against WRF -- ``max_ulp 0``
  against WRF is what ``test_noahmp_vegeflux_cuda.py`` and
  ``test_noahmp_bareflux_cuda.py`` hold.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np
import pytest

from conftest import requires_gpu

import test_noahmp_vegeflux as vege_oracle

from woof.core import noahmp_bareflux_gpu as bare_gpu
from woof.core import noahmp_flux_slab as slab
from woof.core import noahmp_vegeflux_gpu as vege_gpu
from woof.core.noahmp_vegeflux import VegeFluxParameters

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools" / "noahmp_wrf461_oracle"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from gen_bareflux_cases import (                                # noqa: E402
    ARRAY_FIELDS, INT_FIELDS, NSNOW, NSOIL, REAL_FIELDS, unhexf,
)

_ORACLE = _ROOT / "tests" / "data" / "oracles" / "noahmp"
_BARE_FIXTURE = _ORACLE / "noahmp-bareflux.csv"
_PARAM_FIXTURE = _ORACLE / "noahmp-parameters.csv"

#: Column counts and the tiling strides.  ``gcd(stride, len(deck)) == 1`` in
#: both cases, so adjacent columns are always different oracle cases.
_N_VEGE, _VEGE_STRIDE = 40, 3          # 10-case deck, four times over
_N_BARE, _BARE_STRIDE = 54, 5          # 27-case deck, twice over

#: The MODIS classes VEGE_FLUX runs on, as ``noahmp-parameters.csv`` names
#: them.
_VEGETATED = ("evergreen_needleleaf", "grassland", "cropland")


# --------------------------------------------------------------------------
# bit-level comparison
# --------------------------------------------------------------------------
def _cp():
    import cupy as cp

    return cp


def _host(array) -> np.ndarray:
    return np.ascontiguousarray(_cp().asnumpy(array))


def _words(array) -> np.ndarray:
    """The raw 32-bit words of ``array``; NaN compares equal to itself here."""
    host = _host(array)
    if host.dtype not in (np.dtype(np.float32), np.dtype(np.int32)):
        raise AssertionError(f"unexpected packed dtype {host.dtype}")
    return host.view(np.uint32).ravel()


def _mismatched(got, want) -> int:
    """How many 32-bit words differ, counting a shape/dtype change as all."""
    a, b = _host(got), _host(want)
    if a.dtype != b.dtype or a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(_words(got) != _words(want)))


def _assert_identical(got, want, what: str) -> None:
    a, b = _host(got), _host(want)
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    ga, gb = _words(got), _words(want)
    bad = np.flatnonzero(ga != gb)
    assert bad.size == 0, (
        f"{what}: {bad.size}/{ga.size} words differ; first at flat index "
        f"{int(bad[0])}: got 0x{ga[int(bad[0])]:08X} "
        f"want 0x{gb[int(bad[0])]:08X}")


# --------------------------------------------------------------------------
# VEGE_FLUX columns
# --------------------------------------------------------------------------
def _vege_deck() -> list:
    """The ten physical VEGE_FLUX calls, lifted out of the oracle harness.

    ``VegeFluxBatch.capture`` is the collector the CUDA gate already uses: it
    pauses each oracle row at the leaf and keeps its argument list, so the deck
    is the physical Python call, not a fixture-shaped reconstruction.
    """
    batch = vege_gpu.VegeFluxBatch()
    saved = vege_oracle.vege_flux
    vege_oracle.vege_flux = batch.capture
    try:
        for tag in sorted(vege_oracle.TABLE["vegeflux"]):
            try:
                vege_oracle.call_vege_flux(vege_oracle.TABLE["vegeflux"][tag])
            except vege_gpu.VegeFluxBatchPending:
                pass
            else:
                raise AssertionError(
                    "the VEGE_FLUX capture seam did not pause")
    finally:
        vege_oracle.vege_flux = saved
    assert len(batch.calls) == 10
    return batch.calls


def _parameter_sets(deck_parameters) -> list:
    """The deck's own parameter object plus three real MODIS classes."""
    wanted = set(vege_gpu.PARAMETER_NAMES)
    table: dict[str, dict[str, float]] = {}
    with _PARAM_FIXTURE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case"] in _VEGETATED and row["field"] in wanted \
                    and row["index"] == "0":
                table.setdefault(row["case"], {})[row["field"]] = float(
                    row["value"])
    sets = [deck_parameters]
    for case in _VEGETATED:
        values = table[case]
        assert set(values) == wanted, (
            f"{case} is missing VEGE_FLUX parameters "
            f"{sorted(wanted - set(values))}")
        sets.append(VegeFluxParameters(**values))
    return sets


def _vege_calls() -> list:
    deck = _vege_deck()
    parameters = _parameter_sets(deck[0][0][0])
    calls = []
    for index in range(_N_VEGE):
        args, kwargs = deck[(index * _VEGE_STRIDE) % len(deck)]
        assert not kwargs, "the oracle deck is positional; keep it that way"
        parameter = parameters[index % len(parameters)]
        calls.append(((parameter,) + tuple(args[1:]), {}))
    return calls


def _vege_fields(calls) -> dict:
    """The slab spelling of ``calls``: one CuPy column per bound name."""
    cp = _cp()
    bound = [vege_gpu._bind_call(args, kwargs) for args, kwargs in calls]
    fields = {name: cp.asarray([float(call[name]) for call in bound],
                               dtype=cp.float32)
              for name in vege_gpu.INPUT_NAMES}
    for name in vege_gpu.PARAMETER_NAMES:
        fields[name] = cp.asarray(
            [float(getattr(call["p"], name)) for call in bound],
            dtype=cp.float32)
    fields["isnow"] = cp.asarray([int(call["isnow"]) for call in bound],
                                 dtype=cp.int32)
    # Layer axis last, Fortran order -NSNOW+1 .. NSOIL -- the order the
    # per-column packer's ``slot = layer + LAYER_OFFSET`` produces.
    layers = range(-NSNOW + 1, NSOIL + 1)
    for name in ("dzsnso", "stc", "df"):
        fields[name] = cp.asarray(
            [[float(call[name][layer]) for layer in layers] for call in bound],
            dtype=cp.float32)
    return fields


def _vege_launch(calls, use_device_libm: bool = False):
    """Run the shipped per-column evaluator; keep what it handed the kernel."""
    recorded: dict = {}
    real_module = vege_gpu._module

    class _Proxy:
        def __init__(self, module):
            self._module = module

        def get_function(self, name):
            function = self._module.get_function(name)

            def spy(grid, block, args):
                recorded["args"] = args
                return function(grid, block, args)

            return spy

    vege_gpu._module = lambda flag=False, **options: _Proxy(real_module(flag, **options))
    try:
        states = vege_gpu.evaluate_vege_flux_calls(
            calls, use_device_libm=use_device_libm)
    finally:
        vege_gpu._module = real_module
    assert "args" in recorded, "the VEGE_FLUX launch was never observed"
    return states, recorded["args"]


def _vege_reference_outputs(states) -> dict:
    return {name: np.array([np.float32(getattr(state, name))
                            for state in states], dtype=np.float32)
            for name in vege_gpu.OUTPUT_NAMES}


# --------------------------------------------------------------------------
# BARE_FLUX columns
# --------------------------------------------------------------------------
def _bare_rows() -> list:
    with _BARE_FIXTURE.open(newline="") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert len(rows) == 27
    return rows


def _bare_kwargs(row: dict) -> dict:
    """The physical BARE_FLUX keyword call for one fixture row."""
    kwargs = {name: int(row[name]) for name in INT_FIELDS if name != "iurban"}
    kwargs["urban_flag"] = bool(int(row["iurban"]))
    for name in REAL_FIELDS:
        kwargs[name] = unhexf(row[name])
    for name in ARRAY_FIELDS:
        kwargs[name] = [unhexf(row[f"{name}{layer}"])
                        for layer in range(-NSNOW + 1, NSOIL + 1)]
    kwargs["nsnow"] = NSNOW
    kwargs["nsoil"] = NSOIL
    kwargs["opt_sfc"] = int(row["opt_sfc"])
    kwargs["opt_stc"] = int(row["opt_stc"])
    return kwargs


def _bare_calls() -> list:
    rows = _bare_rows()
    return [((), _bare_kwargs(rows[(index * _BARE_STRIDE) % len(rows)]))
            for index in range(_N_BARE)]


def _bare_fields(calls) -> dict:
    cp = _cp()
    bound = [bare_gpu._bind_call(args, kwargs) for args, kwargs in calls]
    fields = {name: cp.asarray([float(call[name]) for call in bound],
                               dtype=cp.float32)
              for name in bare_gpu.SCALAR_NAMES}
    for name in bare_gpu.ARRAY_NAMES:
        fields[name] = cp.asarray([[float(v) for v in call[name]]
                                   for call in bound], dtype=cp.float32)
    for name in ("isnow", "ivgtyp", "iloc", "jloc"):
        fields[name] = cp.asarray([int(call[name]) for call in bound],
                                  dtype=cp.int32)
    # Deliberately a bool column: the per-column packer writes
    # ``1 if call["urban_flag"] else 0`` and the slab must agree on the cast.
    fields["urban_flag"] = cp.asarray([bool(call["urban_flag"])
                                       for call in bound])
    return fields


def _bare_launch(calls):
    """Run the shipped per-column evaluator; keep what it handed the kernel."""
    import woof.core.kernels as kernels

    recorded: dict = {}
    real_get_kernel = kernels.get_kernel

    def spy_get_kernel(name, func):
        function = real_get_kernel(name, func)

        def spy(grid, block, args):
            recorded["args"] = args
            return function(grid, block, args)

        return spy

    kernels.get_kernel = spy_get_kernel
    try:
        results = bare_gpu.evaluate_bare_flux_calls(calls)
    finally:
        kernels.get_kernel = real_get_kernel
    assert "args" in recorded, "the BARE_FLUX launch was never observed"
    return results, recorded["args"]


def _bare_reference_outputs(results) -> dict:
    return {name: np.array([np.float32(getattr(result, name))
                            for result in results], dtype=np.float32)
            for name in bare_gpu.OUTPUT_NAMES}


# --------------------------------------------------------------------------
# gate 1: the packed bytes
# --------------------------------------------------------------------------
@requires_gpu
def test_vege_flux_slab_packs_the_bytes_the_per_column_path_launches():
    calls = _vege_calls()
    assert len(calls) >= 32
    fields = _vege_fields(calls)

    _, launched = _vege_launch(calls)
    packed = slab.pack_vege_flux_slab(fields, len(calls))
    assert len(packed) == 6

    for index, name in enumerate(packed._fields):
        _assert_identical(packed[index], launched[index], f"vege.{name}")
    assert int(launched[7]) == len(calls)

    # Shape and dtype are part of the layout, not incidental.
    assert packed.inputs.shape == (len(calls), vege_gpu.N_INPUT)
    assert packed.params.shape == (len(calls), len(vege_gpu.PARAMETER_NAMES))
    assert packed.isnow.dtype == np.int32
    for block in (packed.dzsnso, packed.stc, packed.df):
        assert block.shape == (len(calls), slab.NLAYER)
        assert block.dtype == np.float32


@requires_gpu
def test_bare_flux_slab_packs_the_bytes_the_per_column_path_launches():
    calls = _bare_calls()
    assert len(calls) >= 32
    fields = _bare_fields(calls)

    inputs, ints = slab.pack_bare_flux_slab(fields, len(calls))
    assert inputs.shape == (len(calls), bare_gpu.N_INPUT)
    assert ints.shape == (len(calls), bare_gpu.N_INT)

    # Reference one: the public per-column packer.
    want_inputs, want_ints = bare_gpu.pack_bare_flux_calls(calls)
    _assert_identical(inputs, want_inputs, "bare.inputs")
    _assert_identical(ints, want_ints, "bare.ints")

    # Reference two: what the shipped evaluator really launches.
    _, launched = _bare_launch(calls)
    _assert_identical(inputs.reshape(-1), launched[0], "bare.inputs.launched")
    _assert_identical(ints.reshape(-1), launched[1], "bare.ints.launched")
    assert int(launched[3]) == len(calls)


# --------------------------------------------------------------------------
# gate 2: the evaluated outputs
# --------------------------------------------------------------------------
@requires_gpu
def test_vege_flux_slab_evaluation_is_bitwise():
    calls = _vege_calls()
    fields = _vege_fields(calls)
    want = _vege_reference_outputs(vege_gpu.evaluate_vege_flux_calls(calls))
    got = slab.evaluate_vege_flux_slab(fields, len(calls))

    assert sorted(got) == sorted(vege_gpu.OUTPUT_NAMES)
    assert len(got) == 30
    for name in vege_gpu.OUTPUT_NAMES:
        _assert_identical(got[name], want[name], f"vege_flux.{name}")


@requires_gpu
def test_bare_flux_slab_evaluation_is_bitwise():
    calls = _bare_calls()
    fields = _bare_fields(calls)
    want = _bare_reference_outputs(bare_gpu.evaluate_bare_flux_calls(calls))
    got = slab.evaluate_bare_flux_slab(fields, len(calls))

    assert sorted(got) == sorted(bare_gpu.OUTPUT_NAMES)
    assert len(got) == 13
    for name in bare_gpu.OUTPUT_NAMES:
        _assert_identical(got[name], want[name], f"bare_flux.{name}")


# --------------------------------------------------------------------------
# the negative controls: both gates must be able to fail
# --------------------------------------------------------------------------
def _vege_rejects(corrupt) -> tuple[int, list]:
    """Corrupt the slab fields; return (packed words, outputs) that moved."""
    calls = _vege_calls()
    fields = _vege_fields(calls)
    corrupt(fields)

    states, launched = _vege_launch(calls)
    packed = slab.pack_vege_flux_slab(fields, len(calls))
    words = sum(_mismatched(packed[index], launched[index])
                for index in range(len(packed)))

    want = _vege_reference_outputs(states)
    got = slab.evaluate_vege_flux_slab(fields, len(calls))
    moved = [name for name in vege_gpu.OUTPUT_NAMES
             if _mismatched(got[name], want[name])]
    return words, moved


def _bare_rejects(corrupt) -> tuple[int, list]:
    calls = _bare_calls()
    fields = _bare_fields(calls)
    corrupt(fields)

    want_inputs, want_ints = bare_gpu.pack_bare_flux_calls(calls)
    inputs, ints = slab.pack_bare_flux_slab(fields, len(calls))
    words = _mismatched(inputs, want_inputs) + _mismatched(ints, want_ints)

    want = _bare_reference_outputs(bare_gpu.evaluate_bare_flux_calls(calls))
    got = slab.evaluate_bare_flux_slab(fields, len(calls))
    moved = [name for name in bare_gpu.OUTPUT_NAMES
             if _mismatched(got[name], want[name])]
    return words, moved


def _transpose(first: str, second: str):
    def corrupt(fields):
        fields[first], fields[second] = fields[second], fields[first]
    return corrupt


def _roll(name: str):
    def corrupt(fields):
        fields[name] = _cp().roll(fields[name], 1)
    return corrupt


@requires_gpu
def test_vege_flux_transposed_input_slots_are_rejected():
    """Slots 1 and 2 -- ``sav`` and ``sag`` -- swapped.

    Adjacent, both live, and both echoed straight back as outputs, so a gate
    that cannot see this cannot see anything.
    """
    assert vege_gpu.INPUT_NAMES[1:3] == ("sav", "sag")
    words, moved = _vege_rejects(_transpose("sav", "sag"))
    assert words > 0, "a transposed input slot left the packed bytes unchanged"
    assert moved, "a transposed input slot left every output unchanged"


@requires_gpu
def test_vege_flux_transposed_parameter_slots_are_rejected():
    """Parameter slots 1 and 2 -- ``HVT`` and ``CBIOM`` -- swapped.

    This is the control the uniform oracle parameter set could not support and
    the reason the three MODIS classes are cycled in: with one vegetation type
    the parameter slab is constant and this transposition is a no-op.
    """
    assert vege_gpu.PARAMETER_NAMES[1:3] == ("HVT", "CBIOM")
    words, moved = _vege_rejects(_transpose("HVT", "CBIOM"))
    assert words > 0, (
        "a transposed parameter slot left the packed bytes unchanged")
    assert moved, "a transposed parameter slot left every output unchanged"


@requires_gpu
def test_vege_flux_rolled_column_is_rejected():
    """``sfctmp`` shifted one column onto its neighbour.

    Every value is still present and every slot still holds a physically
    plausible temperature; only the column each belongs to is wrong.  That is
    the way a slab repacking fails in practice.
    """
    words, moved = _vege_rejects(_roll("sfctmp"))
    assert words > 0, "a one-column roll left the packed bytes unchanged"
    assert moved, "a one-column roll left every output unchanged"


@requires_gpu
def test_bare_flux_transposed_input_slots_are_rejected():
    """Slots 4 and 5 -- ``uu`` and ``vv`` -- swapped.

    ``ur`` is an input rather than a recomputed speed, so the swap leaves the
    wind speed alone and exchanges ``tauxb`` with ``tauyb``.
    """
    assert bare_gpu.SCALAR_NAMES[4:6] == ("uu", "vv")
    words, moved = _bare_rejects(_transpose("uu", "vv"))
    assert words > 0, "a transposed input slot left the packed bytes unchanged"
    assert moved, "a transposed input slot left every output unchanged"


@requires_gpu
def test_bare_flux_rolled_column_is_rejected():
    """``sfctmp`` shifted one column onto its neighbour.

    Measured elsewhere (``test_noahmp_bareflux_cuda._ONE_ULP_REACH``) to reach
    ten of the thirteen outputs, which is why it is the field rolled here.
    """
    words, moved = _bare_rejects(_roll("sfctmp"))
    assert words > 0, "a one-column roll left the packed bytes unchanged"
    assert moved, "a one-column roll left every output unchanged"


@requires_gpu
def test_a_wrong_layer_axis_is_rejected_by_both_gates(monkeypatch):
    """The layer axis is the part of this repacking most likely to be wrong.

    ``dzsnso``/``stc``/``df`` reach the per-column packer as dicts keyed by the
    Fortran index ``-nsnow+1 .. nsoil`` and reach the slab as ``(n, NLAYER)``;
    the origin conversion (``slot = layer + NSNOW - 1``) is exactly the kind of
    off-by-one that produces plausible numbers.  Both mutants of that axis --
    reversed, and shifted one layer -- must be rejected, by both gates, for
    both leaves.

    This is also what shows the two gates are not redundant.  The evaluation
    gate is much the blunter of the pair here: most oracle columns carry
    ``isnow = 0``, so their snow layers are inert and a scrambled snow layer
    changes nothing they can observe.
    """
    real_layers = slab._layers

    def reversed_axis(cp, fields, name, n, leaf):
        block = real_layers(cp, fields, name, n, leaf)
        return cp.ascontiguousarray(block[:, ::-1])

    def shifted_origin(cp, fields, name, n, leaf):
        block = real_layers(cp, fields, name, n, leaf)
        return cp.ascontiguousarray(cp.roll(block, 1, axis=1))

    gates = (
        test_vege_flux_slab_packs_the_bytes_the_per_column_path_launches,
        test_vege_flux_slab_evaluation_is_bitwise,
        test_bare_flux_slab_packs_the_bytes_the_per_column_path_launches,
        test_bare_flux_slab_evaluation_is_bitwise,
    )
    for mutant in (reversed_axis, shifted_origin):
        monkeypatch.setattr(slab, "_layers", mutant)
        for gate in gates:
            with pytest.raises(AssertionError):
                gate()
        monkeypatch.undo()


# --------------------------------------------------------------------------
# the trap: a strided view is a bare pointer once it reaches a raw kernel
# --------------------------------------------------------------------------
@requires_gpu
def test_strided_field_views_are_made_contiguous_before_launch():
    """VEGE_FLUX hands ``dzsnso``/``stc``/``df``/``isnow`` to a kernel as-is.

    A raw-kernel argument carries a pointer and no stride, so passing one of
    those through as a CuPy view makes the kernel read the interleaved poison
    below instead of the layer values.  Measured, not assumed: with the
    ``ascontiguousarray`` discipline removed from ``noahmp_flux_slab`` this
    fails at 36 of 40 columns on ``TG`` alone.

    The BARE_FLUX half cannot currently fail, and that is worth saying rather
    than implying otherwise: every caller-supplied BARE_FLUX array reaches the
    kernel only after being copied into a slab this module allocates, so a
    stride cannot leak through.  It is kept as the regression guard for the day
    someone forwards a caller array to that launch directly.
    """
    cp = _cp()

    def stripe(array):
        """The same values, in a view whose stride is 2."""
        if array.ndim == 1:
            padded = cp.full(array.size * 2, -999, dtype=array.dtype)
            padded[::2] = array
            return padded[::2]
        padded = cp.full((array.shape[0], array.shape[1] * 2), -999,
                         dtype=array.dtype)
        padded[:, ::2] = array
        return padded[:, ::2]

    vege_calls = _vege_calls()
    fields = _vege_fields(vege_calls)
    strided = dict(fields)
    for name in ("dzsnso", "stc", "df", "isnow"):
        strided[name] = stripe(fields[name])
        assert not strided[name].flags.c_contiguous
    want = slab.evaluate_vege_flux_slab(fields, len(vege_calls))
    got = slab.evaluate_vege_flux_slab(strided, len(vege_calls))
    for name in vege_gpu.OUTPUT_NAMES:
        _assert_identical(got[name], want[name], f"strided vege_flux.{name}")

    bare_calls = _bare_calls()
    fields = _bare_fields(bare_calls)
    strided = dict(fields)
    for name in ("dzsnso", "stc", "df", "isnow"):
        strided[name] = stripe(fields[name])
    want = slab.evaluate_bare_flux_slab(fields, len(bare_calls))
    got = slab.evaluate_bare_flux_slab(strided, len(bare_calls))
    for name in bare_gpu.OUTPUT_NAMES:
        _assert_identical(got[name], want[name], f"strided bare_flux.{name}")


@requires_gpu
def test_an_empty_slab_launches_nothing_and_returns_empty_columns():
    """The vegetated subset is empty on a domain with no vegetated tile.

    A zero-column launch is an invalid grid, so this is a real path, not a
    curiosity.
    """
    for evaluate, names in ((slab.evaluate_vege_flux_slab,
                             vege_gpu.OUTPUT_NAMES),
                            (slab.evaluate_bare_flux_slab,
                             bare_gpu.OUTPUT_NAMES)):
        got = evaluate({}, 0)
        assert sorted(got) == sorted(names)
        for name in names:
            assert got[name].shape == (0,)
            assert got[name].dtype == np.float32


@requires_gpu
def test_the_slab_refuses_an_option_identity_it_does_not_transcribe():
    """The per-column guards are essential; the slab must carry them too."""
    calls = _vege_calls()
    fields = _vege_fields(calls)
    with pytest.raises(ValueError, match="opt_sfc"):
        slab.pack_vege_flux_slab({**fields, "opt_sfc": 3}, len(calls))
    with pytest.raises(ValueError):
        slab.pack_vege_flux_slab({**fields, "nsnow": 2}, len(calls))

    calls = _bare_calls()
    fields = _bare_fields(calls)
    with pytest.raises(ValueError, match="opt_sfc"):
        slab.pack_bare_flux_slab({**fields, "opt_sfc": 3}, len(calls))
