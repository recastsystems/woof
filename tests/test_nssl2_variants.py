"""The NSSL option-18 variant family: selector resolution and refusals.

WRF v4.6.1 has exactly one NSSL scheme.  The former ``mp_physics`` values
17/19/21/22 are compatibility spellings that
``share/module_check_a_mundo.F:3382-3421`` rewrites onto ``mp_physics=18``
plus ``nssl_2moment_on`` / ``nssl_hail_on`` / ``nssl_ccn_on`` /
``nssl_density_on``.  These pins cover the resolution of those flags, the
refusals that keep woof off WRF's undefined paths and off unported
branches, and -- on a card -- the shipped ``run_nssl2_production_step`` seam
driven once per ported variant.

The column smoke that drives the shipped seam on a card lives in
``tests/test_nssl2_variants_gpu.py``; this file is CPU-only so the
resolution and refusal contract still runs on a machine with no device.

EVIDENCE SCOPE.  Nothing in this port is checked against WRF Fortran.
There is NO oracle comparison for any variant path: no WRF run, no
matched trajectory, no ULP measurement.  The existing oracle gates cover
the default lane only.
"""

from __future__ import annotations

import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.nssl2_contract import (
    DEFAULT_MODE,
    NSSL2Mode,
    PORTED_VARIANT_MODES,
    canonicalize_deprecated_mp_physics,
    nssl2_contract_receipt,
    pinned_zero_fields,
    require_ported_nssl2_mode,
    resolve_nssl2_mode,
    resolve_nssl2_mode_for_config,
)


# --------------------------------------------------------------------------
# CPU: WRF's consistency pass, transcribed
# --------------------------------------------------------------------------

def test_unset_selectors_resolve_to_wrfs_option_18_default():
    """module_check_a_mundo.F:3433-3455 with everything left at -1."""
    mode = resolve_nssl2_mode()
    assert mode == NSSL2Mode(
        two_moment=True, hail=True, predicted_ccn=True,
        density_moments=2, sixth_moments=0)
    assert mode == DEFAULT_MODE
    assert "qh" in mode.transported_fields
    assert "qnh" in mode.transported_fields
    assert "qnn" in mode.transported_fields
    assert "qvolh" in mode.transported_fields


def test_hail_default_keys_on_two_moment_and_density_keys_on_hail():
    """The two defaults WRF derives rather than fixes (:3441-3455)."""
    # Hail off makes the density default 1 (graupel volume only), because
    # :3450 tests nssl_hail_on == 1 exactly.
    no_hail = resolve_nssl2_mode(nssl_hail_on=0)
    assert no_hail.hail is False
    assert no_hail.density_moments == 1
    assert "qvolh" not in no_hail.transported_fields
    assert "qvolg" in no_hail.transported_fields
    # Hail on keeps the graupel+hail volume default.
    assert resolve_nssl2_mode(nssl_hail_on=1).density_moments == 2


def test_deprecated_scheme_ids_transcribe_wrfs_rewrite():
    """A literal transcription of module_check_a_mundo.F:3382-3421."""
    assert canonicalize_deprecated_mp_physics(17) == (18, {
        "nssl_2moment_on": 1, "nssl_hail_on": 1,
        "nssl_ccn_on": 0, "nssl_density_on": 2})
    assert canonicalize_deprecated_mp_physics(22) == (18, {
        "nssl_2moment_on": 1, "nssl_hail_on": 0,
        "nssl_ccn_on": 0, "nssl_density_on": 1})
    assert canonicalize_deprecated_mp_physics(19) == (18, {
        "nssl_2moment_on": 0, "nssl_hail_on": 2, "nssl_density_on": 1})
    assert canonicalize_deprecated_mp_physics(21) == (18, {
        "nssl_2moment_on": 0, "nssl_hail_on": 0, "nssl_density_on": 0})
    # WRF's 19/21 blocks do NOT touch nssl_ccn_on despite what their own
    # deprecation message suggests, so the consistency pass still defaults
    # it to 1.  The transcription follows the code, not the message.
    assert resolve_nssl2_mode(
        **canonicalize_deprecated_mp_physics(19)[1]).predicted_ccn is True
    # A live scheme ID passes through untouched.
    assert canonicalize_deprecated_mp_physics(18) == (18, {})
    assert canonicalize_deprecated_mp_physics(8) == (8, {})


def test_two_and_one_moment_modes_are_ported():
    assert len(PORTED_VARIANT_MODES) == 8
    for selectors in ({}, {"nssl_ccn_on": 0}, {"nssl_hail_on": 0},
                      {"nssl_hail_on": 0, "nssl_ccn_on": 0}):
        require_ported_nssl2_mode(resolve_nssl2_mode(**selectors))


@pytest.mark.parametrize("selectors,message", [
    ({"nssl_3moment": 1}, "three-moment"),
])
def test_unported_families_refuse_rather_than_substitute(selectors, message):
    with pytest.raises(ValueError, match=message):
        require_ported_nssl2_mode(resolve_nssl2_mode(**selectors))


def test_wrfs_own_undefined_pairings_refuse_at_resolution():
    """woof refuses where WRF reads a field its packages never allocated.

    Both of these are refusals of WRF rather than of woof's coverage, and
    both are the standing rule in action: implement the defined behaviour
    and document the divergence, never reproduce an undefined read.
    """
    # nssl_hail_on=2 selects the nssl_hail1m package, which allocates qh
    # and NOT qnh (Registry.EM_COMMON:3053); a two-moment run still reads
    # the hail-number pointer at module_mp_nssl_2mom.F:2775.
    with pytest.raises(ValueError, match="unallocated hail-number"):
        resolve_nssl2_mode(nssl_2moment_on=1, nssl_hail_on=2)
    # nssl_density_on=1 allocates qvolg only (nssl_graupelvol,
    # Registry.EM_COMMON:3055), but with hail on the module sets lvhl>0 at
    # :1674-1679 and then loads qvolh at :2778 and stores it at :3336.
    with pytest.raises(ValueError, match="unallocated qvolh"):
        resolve_nssl2_mode(nssl_hail_on=1, nssl_density_on=1)
    # Native one-moment hail does not read an unallocated hail number.
    one = resolve_nssl2_mode(nssl_2moment_on=0, nssl_hail_on=2)
    assert require_ported_nssl2_mode(one) is one
    assert "qh" in one.transported_fields and "qnh" not in one.transported_fields
    assert "qvolg" in one.transported_fields and "qvolh" not in one.transported_fields
    plain = resolve_nssl2_mode(nssl_2moment_on=0,nssl_hail_on=0,nssl_density_on=0)
    assert require_ported_nssl2_mode(plain) is plain
    assert all(n not in plain.transported_fields for n in ("qh","qnr","qng","qvolg"))


def test_pinned_zero_fields_name_what_the_registry_would_not_allocate():
    assert pinned_zero_fields(DEFAULT_MODE) == ()
    assert pinned_zero_fields(resolve_nssl2_mode(nssl_ccn_on=0)) == ("qnn",)
    assert pinned_zero_fields(resolve_nssl2_mode(nssl_hail_on=0)) == (
        "qh", "qnh", "qvolh")
    assert pinned_zero_fields(
        resolve_nssl2_mode(nssl_hail_on=0, nssl_ccn_on=0)) == (
            "qh", "qnh", "qnn", "qvolh")


# --------------------------------------------------------------------------
# CPU: the config seam
# --------------------------------------------------------------------------

def _cfg(*, mp_physics: int = 18, **kwargs) -> RunConfig:
    return RunConfig(
        nx=4, ny=4, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0,
        dt=2.0, run_seconds=4.0, moist=True, mp_physics=mp_physics,
        **kwargs)


def test_validate_admits_every_ported_variant():
    for selectors in ({}, {"nssl_ccn_on": 0}, {"nssl_hail_on": 0},
                      {"nssl_hail_on": 0, "nssl_ccn_on": 0}):
        validate_run_config(_cfg(**selectors))


def test_validate_refuses_an_unported_variant_before_state_allocation():
    validate_run_config(_cfg(nssl_2moment_on=0))
    with pytest.raises(ValueError, match="NSSL variant selectors"):
        validate_run_config(_cfg(nssl_3moment=1))


def test_nssl_selectors_require_the_nssl_scheme():
    """A stray NSSL flag under another scheme refuses instead of vanishing."""
    with pytest.raises(ValueError, match="requires mp_physics=18"):
        validate_run_config(_cfg(mp_physics=6, nssl_hail_on=0))


# --------------------------------------------------------------------------
# CPU: the receipts have to describe the run that produced them
# --------------------------------------------------------------------------

def test_receipt_describes_the_resolved_variant_not_the_default_lane():
    """A variant's receipt names its own mode, fields and absences.

    Regression pin for a receipt built from the hardcoded DEFAULT_MODE: a
    hail-off run whose receipt advertises hail, qh and qnh is a receipt
    that misdescribes the run.
    """
    default = nssl2_contract_receipt(resolve_nssl2_mode_for_config(_cfg()))
    assert default["is_default_lane"] is True
    assert default["absent_fields"] == []
    assert "qh" in default["transported_fields"]

    hail_off = nssl2_contract_receipt(
        resolve_nssl2_mode_for_config(_cfg(nssl_hail_on=0)))
    assert hail_off["is_default_lane"] is False
    assert hail_off["resolved_mode"]["hail"] is False
    assert hail_off["absent_fields"] == ["qh", "qnh", "qvolh"]
    assert "qh" not in hail_off["transported_fields"]
    assert "qnh" not in hail_off["transported_fields"]

    ccn_off = nssl2_contract_receipt(
        resolve_nssl2_mode_for_config(_cfg(nssl_ccn_on=0)))
    assert ccn_off["is_default_lane"] is False
    assert ccn_off["resolved_mode"]["predicted_ccn"] is False
    assert ccn_off["absent_fields"] == ["qnn"]

    # Three configurations, three distinct receipts.
    assert len({repr(sorted(receipt.items())) for receipt in
                (default, hail_off, ccn_off)}) == 3


def test_restart_contract_identity_carries_the_run_s_own_mode():
    """woof/io/restart.py's MP18 nested contract, per variant."""
    from woof.io import restart

    default = restart._nssl2_restart_contract_identity(_cfg())
    hail_off = restart._nssl2_restart_contract_identity(
        _cfg(nssl_hail_on=0))

    assert default["resolved_mode"]["hail"] is True
    assert default["absent_fields"] == []
    assert hail_off["resolved_mode"]["hail"] is False
    assert hail_off["absent_fields"] == ["qh", "qnh", "qvolh"]
    assert "qh" not in hail_off["transported_fields"]
    # The archive still carries every allocated array -- the variant's
    # absent fields are present in the file AS ZEROS, and the contract
    # says so rather than pretending the members are gone.
    assert "state/qh" in hail_off["state_members"]
    assert default != hail_off
