"""The mp=50 restart-identity contract, as gates rather than as a document.

WHY THIS MODULE EXISTS
----------------------
The concrete breakage, and it has happened twice: a scheme-specific knob
lands on :class:`woof.config.RunConfig` WITHOUT a row in
:data:`woof.core.model.SCHEME_SCOPED_RUN_FIELDS`, so it enters
``restart_identity_payload`` for EVERY domain of EVERY experiment,
including runs that never select that scheme.  ``experiment_fingerprint``
is that payload plus the input catalogue, so every previously written
checkpoint refuses to resume -- for a field nothing in its trajectory
ever read.  The NSSL five did it at the 1.9 assembly; the mp=28 aerosol
pair did it at the 2.5.8 integration (fixed in ``62ce7c3f6``).

MEASURED 2026-08-28, when these tests were written:
``SCHEME_SCOPED_RUN_FIELDS`` carried schemes 16, 18 and 28 and no others,
and ``RunConfig`` held ZERO P3 fields.  The tests were set as a trap ahead
of the first one rather than after a fingerprint had already moved.

THE TRAP HAS SINCE SPRUNG, correctly, which is the useful part of the
record: ``p3_backend`` is on ``RunConfig`` now and
``SCHEME_SCOPED_RUN_FIELDS[50]`` carries it, so the scoped row exists and
the off-scheme refusal below covers it.  Read the paragraph above as the
state these gates were built against, not as the current inventory.

WHAT EACH TEST GUARDS
---------------------
* :func:`test_no_scheme_prefixed_run_field_is_unscoped` fires the moment a
  ``p3_*`` (or ``nssl_*``, ``wdm6_*``, ``mp28_*``) field is added to
  ``RunConfig`` without its row.  It has a mutation control beside it.
* :func:`test_every_scheme_scoped_field_is_refused_off_scheme` is what
  makes the scoping PROVABLY LOSSLESS: dropping a field from the identity
  discards no information only if that field can hold nothing but its
  default under any other scheme.  It runs over the table, so a scheme-50
  row inherits the requirement automatically the day it is added.
* :func:`test_the_p3_restart_contract_string_is_the_measured_configuration`
  pins the mp=50 algorithm-identity string against the Fortran facts each
  of its tokens claims.
"""

from __future__ import annotations

import dataclasses

import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.model import SCHEME_SCOPED_RUN_FIELDS
from woof.io.restart import MICROPHYSICS_ALGORITHM_IDENTITIES

#: Scheme selectors are named after their scheme in this project -- the
#: nssl_/wdm6_/mp28_ precedent -- so the prefix is a reliable detector.
#: ``p3_`` is here BEFORE any p3_ field exists: that is the point.
SCHEME_FIELD_PREFIXES = {
    "nssl_": 18,
    "wdm6_": 16,
    "mp28_": 28,
    "p3_": 50,
    "mp50_": 50,
}


#: Fields whose non-default value has to be chosen rather than derived,
#: because a mechanically perturbed value would be rejected by a DIFFERENT
#: validation before the off-scheme refusal is reached.  A new row here is
#: a statement that the field is enumerated, not that the test is lenient.
LEGAL_NON_DEFAULT = {
    "mp28_aerosol_source": "climatology",
    # p3_backend is an enum over ('cuda', 'fused', 'reference'), so the
    # mechanical "default + '-off-scheme'" perturbation produces
    # 'cuda-off-scheme' and trips the enum refusal in woof.config first.
    # That refusal is correct and is not the one under test here: this test
    # is about whether a P3 knob set on a NON-P3 run is refused for being
    # off-scheme, so it needs a value that is legal for the field and still
    # not the default.  'reference' is that value.
    "p3_backend": "reference",
}


def _run_config_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(RunConfig))


def _correct_mode_scoped(name, scheme) -> bool:
    """A correct-mode key (woof/correct_mode.py) carries its own scope:
    ``KEY_SCOPE`` drops it from the identity of every domain whose schemes do
    not read it (woof/core/model.py restart_identity_payload, the checkpoint
    echo and the public documents), which is this gate's requirement.  The
    off-scheme refusal does not apply: the mode authors EVERY key in both
    modes, so an exact run carries ``False`` on keys no selected scheme
    reads, and refusing that would refuse exact mode itself."""
    from woof.correct_mode import KEY_SCOPE
    scope = KEY_SCOPE.get(name)
    return (scope is not None and scope[0] == "mp_physics"
            and scheme in scope[1])


def _unscoped(field_names) -> list[tuple[str, int]]:
    """Scheme-prefixed fields missing from their scheme's scoped row."""
    offenders = []
    for name in field_names:
        for prefix, scheme in SCHEME_FIELD_PREFIXES.items():
            if not name.startswith(prefix):
                continue
            if (name not in SCHEME_SCOPED_RUN_FIELDS.get(scheme, ())
                    and not _correct_mode_scoped(name, scheme)):
                offenders.append((name, scheme))
    return offenders


def test_no_scheme_prefixed_run_field_is_unscoped():
    """A scheme knob outside its scoped row binds every run's identity.

    BREAKAGE PREVENTED: the exact 2.5.8 regression, ahead of time.  The
    aerosol pair landed unscoped, moved both frozen fingerprint anchors,
    and would have made every 2.5.7 checkpoint refuse to resume under
    2.5.8 for fields only mp=28 reads.  A P3 knob added the same way costs
    the same thing, and by then the fingerprints have already moved.
    """
    offenders = _unscoped(_run_config_field_names())
    assert not offenders, (
        "RunConfig fields are scheme-scoped by name but missing from "
        f"SCHEME_SCOPED_RUN_FIELDS: {offenders}. Add the row in "
        "woof/core/model.py AND the off-scheme refusal in "
        "validate_run_config, in the same commit that adds the field.")


def test_the_unscoped_detector_actually_detects():
    """MUTATION CONTROL: today's answer is empty, so prove it can be full.

    ``RunConfig`` carries no P3 field yet, so the test above passes
    vacuously.  Without this, it would keep passing if the detector were
    broken -- and the day it mattered would be the day it was silent.
    """
    offenders = _unscoped(_run_config_field_names() + ("p3_nccnst",))
    assert ("p3_nccnst", 50) in offenders


@pytest.mark.parametrize(
    "scheme,field",
    [(scheme, field)
     for scheme, fields in sorted(SCHEME_SCOPED_RUN_FIELDS.items())
     for field in fields],
)
def test_every_scheme_scoped_field_is_refused_off_scheme(scheme, field):
    """Scoping is lossless only if the field cannot hold a value off-scheme.

    BREAKAGE PREVENTED: a silently DISCARDED setting.  ``restart_identity
    _payload`` pops a scoped field from every domain that did not select
    its scheme.  If such a domain could legally carry a non-default value,
    the pop would drop a real request from the identity -- two experiments
    that differ in a knob the operator set would fingerprint the same, and
    each would resume the other's checkpoints.  The refusal is what turns
    "dropped" into "impossible", and it is why the mp=16 and mp=28 rows
    each shipped with one.

    Parametrized over the table so a scheme-50 row inherits the
    requirement on the day it is added, with no test to remember to write.
    """
    other = 8 if scheme != 8 else 6
    base = dict(nx=4, ny=3, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
                dt=10.0, run_seconds=10.0, moist=True, mp_physics=other,
                sf_sfclay_physics=91, sf_surface_physics=2,
                bl_pbl_physics=1, num_soil_layers=4, cu_physics=0)
    assert validate_run_config(RunConfig(**base)) is not None

    default = getattr(RunConfig(**base), field)
    if field in LEGAL_NON_DEFAULT:
        stray = LEGAL_NON_DEFAULT[field]
    elif isinstance(default, bool):
        stray = not default
    elif isinstance(default, int):
        stray = int(default) + 1
    elif isinstance(default, float):
        stray = float(default) * 2.0 + 1.0
    elif isinstance(default, str):
        stray = (default or "x") + "-off-scheme"
    else:  # pragma: no cover - a new field type needs a deliberate rule
        pytest.fail(f"{field} has unhandled default type {type(default)}")
    assert stray != default

    with pytest.raises(ValueError) as caught:
        validate_run_config(RunConfig(**dict(base, **{field: stray})))
    message = str(caught.value)
    # The SHAPE of the refusal, not merely that something was refused.
    # Asserting only "the field name appears" was a false green: a stray
    # string value tripped mp28_aerosol_source's ENUM check instead, whose
    # message names the field and contains "28" inside the field name
    # itself.  Requiring "requires mp_physics=<scheme>" is what separates
    # the off-scheme refusal from every other validation in the function.
    assert field in message
    owners = tuple(owner for owner, fields in SCHEME_SCOPED_RUN_FIELDS.items()
                   if field in fields)
    # The refusal lists every scheme that reads the field ("requires
    # mp_physics=14 or mp_physics=16 or ..."), so the owners are named
    # inside that list rather than each right after "requires".
    assert "requires mp_physics=" in message
    assert all(f"mp_physics={owner}" in message for owner in owners)


def test_the_off_scheme_assertion_rejects_a_different_validations_message():
    """MUTATION CONTROL for the parametrized test's assertion shape.

    The first draft asserted only that the message named the field and
    contained the scheme number.  ``mp28_aerosol_source`` satisfied both
    from a VALUE-domain rejection that says nothing about scheme scoping,
    with "28" supplied by the field name itself -- a green that proved
    nothing.  This pins that the stronger assertion actually separates the
    two, so the strengthening cannot be quietly undone.
    """
    # On its own scheme, so the scope check passes and only the VALUE-domain
    # check can refuse (on mp=8 the off-scheme refusal now fires first).
    base = dict(nx=4, ny=3, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
                dt=10.0, run_seconds=10.0, moist=True, mp_physics=28,
                sf_sfclay_physics=91, sf_surface_physics=2,
                bl_pbl_physics=1, num_soil_layers=4, cu_physics=0)
    with pytest.raises(ValueError) as caught:
        validate_run_config(
            RunConfig(**dict(base, mp28_aerosol_source="not-a-source")))
    message = str(caught.value)
    assert "mp28_aerosol_source" in message      # the weak assertion holds
    assert "28" in message                       # and so does this one
    assert "requires mp_physics=" not in message   # the strong one does not


def test_the_p3_restart_contract_string_is_the_measured_configuration():
    """Every token in the mp=50 identity string is a Fortran fact.

    Pinned because the string IS the refusal: it is what makes a resume
    across a physics change fail BEFORE restore rather than integrate a
    different trajectory from a restored state.  Each claim, verified in
    ``phys/module_mp_p3.F`` (P3 v4.5.2, byte-identical across WRF v4.6.1,
    v4.7.1 and v4.8.0):

    * ``1cat`` / ``2mom-ice`` -- ``p3_init(nCat=1, trplMomI=.false.)``.
      Measured: at nCat=1 the multi-category table 2 is never opened (the
      read is inside ``if (nCat>1)``, :475), and the 2-moment arm reads
      table 1 version ``5.4_2momI`` (:138, :161).
    * ``specified-nc`` -- ``log_predictNc = present(nc_3d)`` (:1064-1065),
      and WRF's mp=50 arm passes no nc_3d.  This is the ONLY difference
      between this row and the unported mp=51.
    * ``diagnosed-ssat`` -- ``log_predictSsat = .false.`` is hard-coded at
      :2252, so ssat/sup/supi are diagnosed from ``qv_old`` and ``qvs``
      each step (:2333-2336), which is what makes ``th_old``/``qv_old``
      the cross-step carriers this restart serializes (written at the end
      of every call under model='WRF', :5018-5021).
    * ``rime-mass-volume-transported`` -- ``qir``/``qib`` advect with
      ``qi`` (woof/core/moist.py::P3_SPECIES).
    * ``real-clamp-product`` -- ``moist_physics_finish_em`` clamps the
      theta increment at ``mp_tend_lim*dt``, two REALs multiplied
      (dyn_em/module_big_step_utilities_em.F:5706-5707), and P3 finishes
      through woof.core.microphysics.moist_physics_finish, which forms
      that float32 product since 2.8.8 (v2).

    KNOWN IMPRECISION, recorded rather than fixed: the string says
    ``wrf-v4.6.1``, which names a WRF RELEASE, not the P3 version (4.5.2)
    and not the file digest.  It is accurate -- the file is byte-identical
    through v4.8.0 -- but it would not move if a future WRF changed the
    file while keeping the release name in our comment.  Correcting it
    means moving the string, which moves the algorithm identity, which
    makes every existing mp=50 checkpoint refuse to resume.  That is a
    deliberate decision for a release boundary, not a tidy-up.
    """
    identity = MICROPHYSICS_ALGORITHM_IDENTITIES[50]
    assert identity == (
        "p3-one-category-wrf-v4.6.1-v2-2mom-ice-specified-nc-"
        "diagnosed-ssat-rime-mass-volume-transported-real-clamp-product")
    for token in ("2mom-ice", "specified-nc", "diagnosed-ssat",
                  "rime-mass-volume-transported", "real-clamp-product"):
        assert token in identity
    # The unported siblings must NOT be describable by this row.
    assert "3mom" not in identity
    assert "prognostic-nc" not in identity
