"""The v8.4.1 lanes read a legacy ``config_epssm`` as upstream does: warn, ignore.

THE BREAKAGE THESE PREVENT: (1) a namelist written for an 8.3.1-era MPAS,
which sets ``config_epssm = 0.1``, refused before anything ran although
native v8.4.1 reads the same namelist with one warning, and any other real
value with the same warning; (2) the opposite fault, a nonzero value reaching
the v8.2.3 scalar coefficients on a v8.4.1 run, which would change the
answer while the receipt said v8.4.1; (3) a non-real value raising a bare
TypeError instead of a refusal that names the knob.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import math
import pickle
import re
import pathlib
import sys
import warnings

import numpy as np
import pytest

from woof.hex.config_v841 import (
    LegacyEpssmIgnored,
    V841DryDycoreConfig,
    V841MpasColumnPhysicsConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)
from woof.hex.driver import DryDycoreConfig
from woof.hex.errors import ConfigurationRefusal

TREE = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = TREE / ".." / ".." / "woof" / "hex"
RULER_TOOL = TREE / "tools" / "run_cuda_v841_numeric_ruler.py"

V841_LANES = (
    V841DryDycoreConfig,
    V841MpasColumnPhysicsConfig,
    V841MpasColumnPhysicsSmagorinskyGwdoConfig,
)


def _validate_collecting(config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.validate()
    return [w for w in caught if issubclass(w.category, LegacyEpssmIgnored)]


def test_the_pinned_default_is_silent():
    assert V841DryDycoreConfig().config_epssm == 0.0
    for lane in V841_LANES:
        assert _validate_collecting(lane()) == []


@pytest.mark.parametrize("value", [0.1, 0.5, 0.0999])
def test_a_legacy_value_is_read_and_ignored_with_upstreams_warning(value):
    caught = _validate_collecting(V841DryDycoreConfig(config_epssm=value))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert caught[0].message.value == value
    assert "no longer supported" in message
    assert "config_epssm_transition_bottom_z" in message


@pytest.mark.parametrize("value", [0.1, -0.5, math.nan, math.inf, -math.inf])
def test_legacy_warning_roundtrip_preserves_value_and_text(value):
    warning = LegacyEpssmIgnored(value)
    # xdist also reconstructs a warning with its type and args. A formatted
    # message in args instead of the numeric value crashed its controller.
    for restored in (type(warning)(*warning.args), pickle.loads(pickle.dumps(warning))):
        assert type(restored) is LegacyEpssmIgnored
        assert str(restored) == str(warning)
        if math.isnan(value):
            assert math.isnan(restored.value)
        else:
            assert restored.value == value


@pytest.mark.parametrize("lane", V841_LANES, ids=lambda lane: lane.__name__)
@pytest.mark.parametrize(
    "value", [0.1, 1.0, 1.5, -0.5, math.nan, math.inf, -math.inf, 2]
)
def test_every_real_value_is_ignored_on_every_v841_lane(lane, value):
    # Upstream v8.4.1 tests ``config_epssm /= 0.0`` and runs with any real
    # (mpas_atm_time_integration.F:174-197), so hex refuses none of them:
    # a refusal of a value native v8.4.1 runs with names no breakage.
    caught = _validate_collecting(lane(config_epssm=value))
    assert len(caught) == 1


@pytest.mark.parametrize("lane", V841_LANES, ids=lambda lane: lane.__name__)
@pytest.mark.parametrize("value", ["0.1", True, None, np.bool_(False)])
def test_a_non_real_value_is_refused_by_name_on_every_v841_lane(lane, value):
    with pytest.raises(ConfigurationRefusal) as raised:
        lane(config_epssm=value).validate()
    assert raised.value.knob == "config_epssm"
    assert "Registry type is real" in raised.value.reason


@pytest.mark.parametrize("value", ["0.1", True, None])
def test_the_v823_lane_refuses_a_non_real_value_by_name(value):
    with pytest.raises(ConfigurationRefusal) as raised:
        DryDycoreConfig(config_epssm=value).validate()
    assert raised.value.knob == "config_epssm"
    assert "Registry type is real" in raised.value.reason


@pytest.mark.parametrize("value", [1.0, 1.5, -0.5, math.nan])
def test_the_v823_lane_still_refuses_what_its_coefficients_cannot_take(value):
    # v8.2.3 consumes the knob as a constant; its range refusal stands.
    with pytest.raises(ConfigurationRefusal) as raised:
        DryDycoreConfig(config_epssm=value).validate()
    assert raised.value.knob == "config_epssm"


def test_no_v841_dycore_module_reads_the_legacy_knob():
    # The v8.4.1 acoustic and coefficient paths take their off-centering from
    # woof.hex.offcentering_v841 (the four &damping profile knobs).
    bare = re.compile(r"config_epssm(?!_)")
    offenders = []
    for path in sorted(PACKAGE.glob("*_v841.py")):
        if path.name == "config_v841.py":
            continue
        if bare.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == []


# Every numeric read of the legacy knob in the package, by file and enclosing
# function.  Each one feeds a v8.2.3 routine, and each must sit behind a
# v8.4.1 dispatch earlier in the same function.  A new read site fails here
# until it is reviewed and added.
_NUMERIC_READS = {
    ("driver.py", "DryDycoreDriver._vertical_coefficients"),
    ("driver.py", "DryDycoreDriver._advance_dynamics_subcycle"),
    ("cuda_driver.py", "CudaDryDycoreDriver._vertical_coefficients"),
    ("cuda_driver.py", "CudaDryDycoreDriver.step_device"),
}
# The separately selected NOAA core consumes scalar epssm natively:
# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2366
# at 1d6e370e. Its constructor requires the fork configuration; it cannot
# accept the default v841 configuration. Register only these reviewed sites,
# preserving the dispatch requirement for every legacy/default read above.
_RRFS_NUMERIC_READS = {
    ("cuda_driver_rrfs.py", "CudaRRFSDycoreDriver._vertical_coefficients"),
    ("cuda_driver_rrfs.py", "CudaRRFSDycoreDriver._advance_dynamics_subcycle_v841"),
}
# The table's dycore variant resolver seeds config_epssm from the SELECTED VARIANT CLASS's own default
# (``kind().config_epssm``: 0.1 for the 1d6e370e fork core, its own class default for the 44c0849 NOAA v8.4.1
# core), replacing whatever the incoming v8.4.1 configuration carried; it never reads the incoming knob.
# Registered as that exact expression (guide/repin-fold-0410, the 44c0849 re-pin), so a read of the
# incoming configuration's knob at this site still fails here.
_VARIANT_DEFAULT_READS = {
    ("dycore_variants.py", "resolve_dycore_variant"),
}
_V841_DISPATCH_MARKERS = ("V841_SOURCE_RELEASE", "'v8.4.1'", '"v8.4.1"', "v841_context")


def _epssm_reads(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []

    def visit(node, names, function):
        for child in ast.iter_child_nodes(node):
            here_names, here_function = names, function
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                here_names = names + (child.name,)
                if not isinstance(child, ast.ClassDef):
                    here_function = child
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "config_epssm"
                and isinstance(child.ctx, ast.Load)
                # ``self.config_epssm`` inside a config's own validate is
                # validation, not arithmetic.
                and not (isinstance(child.value, ast.Name) and child.value.id == "self")
            ):
                found.append((".".join(names), here_function, child))
            visit(child, here_names, here_function)

    visit(tree, (), None)
    return found


def _behind_a_v841_dispatch(function: ast.AST, read: ast.Attribute) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or node.lineno >= read.lineno:
            continue
        if not any(marker in ast.unparse(node.test) for marker in _V841_DISPATCH_MARKERS):
            continue
        in_else = any(
            child.lineno <= read.lineno <= child.end_lineno for child in node.orelse
        )
        returns_first = (
            isinstance(node.body[-1], ast.Return)
            and not node.orelse
            and read.lineno > node.end_lineno
        )
        if in_else or returns_first:
            return True
    return False


def test_every_numeric_read_of_the_legacy_knob_sits_behind_a_v841_dispatch():
    fork_tree = ast.parse((PACKAGE / "cuda_driver_rrfs.py").read_text(encoding="utf-8"))
    fork_driver = next(
        node for node in fork_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CudaRRFSDycoreDriver"
    )
    constructor = next(
        node for node in fork_driver.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    # The registration is conditional on the existing fail-closed type guard,
    # not merely on a filename. Removing it would admit an ignored v841 epssm
    # into fork arithmetic and must make this audit fail again.
    assert any(
        isinstance(node, ast.If)
        and ast.unparse(node.test) == "not isinstance(self.config, RRFSDryDycoreConfig)"
        and not node.orelse
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Raise)
        and isinstance(node.body[0].exc, ast.Call)
        and isinstance(node.body[0].exc.func, ast.Name)
        and node.body[0].exc.func.id == "TypeError"
        for node in constructor.body
    ), "the registered fork reads require a constructor that refuses v841 configurations"
    seen = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        for qualname, function, read in _epssm_reads(path):
            site = (str(path.relative_to(PACKAGE)).replace("\\", "/"), qualname)
            seen.add(site)
            if site in _RRFS_NUMERIC_READS:
                assert function is not None
                assert ast.unparse(read.value) == "self.config"
                continue
            if site in _VARIANT_DEFAULT_READS:
                assert function is not None
                assert ast.unparse(read.value) == "kind()"
                continue
            assert function is not None and _behind_a_v841_dispatch(function, read), (
                f"{site[0]}:{read.lineno} ({qualname}) reads config_epssm without "
                "a v8.4.1 dispatch before it; on a v8.4.1 run the ignored legacy "
                "value would reach v8.2.3 arithmetic"
            )
    assert seen == _NUMERIC_READS | _RRFS_NUMERIC_READS | _VARIANT_DEFAULT_READS


# --- the runtime identity, on the mesh the v8.4.1 numeric ruler uses -------
#
# A source hold says where the knob is read; this says what a run does.  The
# deck is the numeric ruler's (tools/run_cuda_v841_numeric_ruler.py): x1.2562,
# 3 levels, a nonconstant off-centering profile (transition 3 to 20 km), an
# active w damping, nonzero tracers and reference winds.  The mesh pair is an
# external asset at the CLI's default location; without it these skip.


def _ruler():
    if "_epssm_ruler_tool" in sys.modules:
        return sys.modules["_epssm_ruler_tool"]
    spec = importlib.util.spec_from_file_location("_epssm_ruler_tool", RULER_TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_epssm_ruler_tool"] = module
    spec.loader.exec_module(module)
    return module


def _ruler_case_or_skip():
    ruler = _ruler()
    missing = [str(path) for path in (ruler.DEFAULT_GRID, ruler.DEFAULT_STATIC) if not path.is_file()]
    if missing:
        pytest.skip(
            "the x1.2562 mesh pair is an external asset and is absent here "
            f"({', '.join(missing)})"
        )
    return ruler, ruler.prepare_synthetic_case(config=ruler.ruler_config())


def _state_digest(state) -> str:
    digest = hashlib.sha256()
    for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
        array = np.ascontiguousarray(np.asarray(getattr(state, name)))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cpu_step(case, config):
    from woof.hex.driver import DryDycoreDriver

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyEpssmIgnored)
        driver = DryDycoreDriver(
            case.mesh,
            case.vertical,
            case.reference,
            config,
            terrain_metrics=case.terrain,
            advection_coefficients=case.coefficients,
            reference_wind_profiles=case.profiles,
        )
    saved = driver._rebuild_saved_diagnostics(case.state)
    return _state_digest(driver.step(case.state.copy(), saved_diagnostics=saved).state)


def test_the_cpu_v841_step_is_bitwise_identical_for_any_legacy_value():
    _, case = _ruler_case_or_skip()
    base = case.config
    reference = _cpu_step(case, base)
    for value in (0.5, math.nan):
        assert _cpu_step(case, dataclasses.replace(base, config_epssm=value)) == reference
    # Non-vacuous: the profile knob v8.4.1 DOES read moves the answer.
    moved = dataclasses.replace(base, config_epssm_maximum=0.5)
    assert _cpu_step(case, moved) != reference
