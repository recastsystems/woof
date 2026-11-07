"""Exact initial values for prognostic species absent from analyzed input.

These are state-allocation contracts, independent of the source name and of
radiation, turbulence, surface or cumulus presets. WRF v4.6.1 Registry.EM_COMMON
lines 3025/3031 bind Milbrandt/WDM6 qnc to QNCLOUD (line 542); NSSL
uses the distinct qndrop/QNDROP declaration at lines 521-522.
An analyzed mass inventory is validated separately by initialize_real and its correspondence receipt.
"""
from __future__ import annotations

from types import SimpleNamespace
import struct


def source_absent_microphysics(cfg):
    """Return (WRF field names, native FP32 initial values) for active extras."""
    mp = int(cfg.mp_physics)
    from woof.core.column_schemes import COLUMN_SCHEMES
    if mp in COLUMN_SCHEMES:
        return (), {}  # Single-moment column rows have no number tracers.
    from woof.core.wsm_family_constants import (
        selectors, source_absent_initialization,
    )
    if mp in selectors():
        return source_absent_initialization(mp)
    if mp in (0, 1, 2, 3, 6, 106):
        return (), {}
    if mp == 30:
        from woof.core.fast_sbm_contract import BIN_SPECIES,NUMBER_SPECIES
        return (BIN_SPECIES+("QNCCN","QNCLOUD","QNRAIN","QNICE","QNSNOW","QNGRAUPEL"),
                dict.fromkeys(BIN_SPECIES+NUMBER_SPECIES,0.0))
    if mp == 8:
        return ("QNICE", "QNRAIN"), {"ni": 0.0, "nr": 0.0}
    if mp == 9:
        return (("QHAIL", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW",
                 "QNGRAUPEL", "QNHAIL"),
                dict.fromkeys(("qh", "nc", "nr", "ni", "ns", "ng", "nh"), 0.0))
    if mp == 10:
        return (("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"),
                dict.fromkeys(("nc", "nr", "ni", "ns", "ng"), 0.0))
    if mp in (14, 16):
        return (("QNCLOUD", "QNRAIN", "QNCCN"),
                {"nc": 0.0, "nr": 0.0, "nn": float(cfg.wdm6_ccn_conc)})
    if mp in (26, 27):
        # WDM7 (Registry.EM_COMMON:3035): WDM6's three plus the hail mass,
        # which an analysis without hail starts at zero.
        return (("QHAIL", "QNCLOUD", "QNRAIN", "QNCCN"),
                {"qh": 0.0, "nc": 0.0, "nr": 0.0,
                 "nn": float(cfg.wdm6_ccn_conc)})
    if mp == 18:
        from woof.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
        values = dict.fromkeys(("qh", "qndrop", "qnr", "qni", "qns", "qng",
                                "qnh", "qnn", "qvolg", "qvolh"), 0.0)
        values["qnn"] = NSSL2_BACKGROUND_CCN_PER_KG
        return (("QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
                 "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL"), values)
    if mp == 15:
        return ("QRIMEF",), {"qrimef": 0.0}
    if mp == 5:
        # Ferrier-Aligo: the scalar qt (WRF output name CWM) has no analysed
        # source and real.exe never computes it; FER_HIRES rebuilds it from
        # qc+qr+qi on the first call (module_mp_fer_hires.F:388).  The
        # f_* state starts where fer_hires_init puts it (woof/core/state.py).
        return ("CWM",), {"qt": 0.0}
    if mp == 52:
        return (("QNCLOUD", "QNICE", "QNRAIN", "QIR", "QIB", "QICE2", "QNICE2", "QIR2", "QIB2"),
                dict.fromkeys(("nc", "ni", "nr", "qir", "qib", "qi2", "ni2", "qir2", "qib2"), 0.0))
    if mp == 50:
        return (("QNICE", "QNRAIN", "QIR", "QIB"),
                dict.fromkeys(("ni", "nr", "qir", "qib"), 0.0))
    if mp == 53:
        # P3 three-moment ice (Registry.EM_COMMON:3041): mp=50's four
        # scalars plus the droplet number and the advected ice moment, all
        # at the exact zero WRF's allocator and real.exe leave them at when
        # the analysis carries none.  P3 itself floors what it needs: the
        # wrapper sets Z = 0 where Ni < qsmall (module_mp_p3.F:830-834) and
        # p3_main floors Z at zsmall and clamps it into the mu_i bounds
        # wherever ice exists (:2612, :2654-2660).
        return (("QNICE", "QNRAIN", "QIR", "QIB", "QNCLOUD", "QZI"),
                dict.fromkeys(("ni", "nr", "qir", "qib", "nc", "qzi"), 0.0))
    if mp in THOMPSON_AEROSOL_FAMILY:
        # Aerosol-aware Thompson (Registry.EM_COMMON:3036).  The three
        # number species a cold start owns are nc, nr and ni, and every
        # one of them is allocated at exact zero -- the value
        # woof/core/state.py allocates and the value real.exe keeps where
        # the paired analysed mass is zero. For mp=28, real.exe seeds the
        # number where mass is positive (cold_start_seeded_numbers below).
        # WRF v4.7.1 module_initialize_real.F:4801-4804 excludes mp=38
        # from that block, so its initial numbers remain zero everywhere.
        #
        # nwfa/nifa (and the two 2-D emission fields) are deliberately NOT
        # here.  Their initial condition is not "absent from the analysed
        # input, therefore zero": it is DECIDED by mp28_aerosol_source --
        # the WIF climatology ingest fills them
        # (woof/ingest/real.py, real.exe's aer_init_opt=1 path) and the
        # synthetic source leaves the exact zeros thompson_init's MAXVAL
        # test reads to install its own profile
        # (module_mp_thompson.F:493/:531).  That authority publishes its
        # own receipt and refuses a nonzero field it did not write, so a
        # second contract over the same fields here would be a second
        # spelling -- and it would be WRONG for the climatology run, whose
        # nwfa is nonzero by the time any consumer of this contract looks.
        #
        # This function used to RAISE for mp=28, on the premise that no
        # source carried the aerosol boundary species.  Audit R-044
        # retired that premise (the WIF ingest is the source, on every
        # route), and the raise outlived it: it was reached from
        # tools/prepare_hrrr_wrf.py after the wizard had already reported
        # PASS -- a refusal after step 0, which the gate law forbids.  The
        # precondition that survives is the DATASET one, measured by
        # woof.config.mp28_aerosol_lateral_forcing_precondition: reported
        # at plan review and raised at the run door by
        # woof.config.validate_run_preparation.
        extra = THOMPSON_AEROSOL_FAMILY[mp].graupel_species
        fields = ("QNGRAUPEL", "QVGRAUPEL") if extra else ()
        return (("QNCLOUD", "QNRAIN", "QNICE") + fields,
                dict.fromkeys(("nc", "nr", "ni") + extra, 0.0))
    from woof.microphysics_schemes import scheme as _named_scheme
    named = _named_scheme(mp)
    if named is not None:
        # A named scheme: every moment species starts at exact zero (the
        # allocation value); the scheme's own entry block and its adapter's
        # first-call graupel seeding fill them from the analysed masses.
        # The aerosol tracers are owned by the aerosol initialisation, as
        # for mp=28.
        wrf_names = {"nc": "QNCLOUD", "nr": "QNRAIN", "ni": "QNICE",
                     "ng": "QNGRAUPEL", "ns": "QNSNOW", "volg": "QVGRAUPEL"}
        names = tuple(n for n in named.moment_species if n in wrf_names)
        return (tuple(wrf_names[n] for n in names), dict.fromkeys(names, 0.0))
    raise ValueError(f"no native prognostic-species initialization for mp_physics={mp}")


def cold_start_seeded_numbers(cfg):
    """Number moments a Thompson cold start seeds from their analysed mass.

    Returns ``{state number field: (paired state mass field, seed receipt
    key)}``.  :func:`source_absent_microphysics` gives the ALLOCATION value
    of every source-absent field; for the fields named here that value is
    exact only where the paired mass is zero.  WRF v4.7.1
    dyn_em/module_initialize_real.F:4829-4852 fills each of them where the
    mass is above zero and the number is at or below zero, with
    make_DropletNumber (QNCLOUD, mp=28 only: mp=8's cloud number is the
    constant Nt_c), make_RainNumber and make_IceNumber, and
    woof.ingest.real's cold-start closure does the same and records each
    fill under the seed receipt key of its ``cold_start_moment_closure``.
    Every other microphysics option returns an empty map: its source-absent
    numbers stay at the allocation value everywhere.
    """
    cfg = SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg
    mp = int(cfg.mp_physics)
    seeded = {}
    if mp == 28:
        seeded["nc"] = ("qc", "droplet_number_seed")
    if mp in (8, 28):
        seeded["nr"] = ("qr", "rain_number_seed")
        seeded["ni"] = ("qi", "ice_number_seed")
    if mp == 10:
        # Morrison: no real.exe seed; the handover closure
        # (woof.ingest.handover_numbers) seeds all four from their masses
        # where correct-mode key hydrometeor_handover_numbers is on (off in
        # exact mode and in a strict WRF process), under the same receipt keys.
        from woof.ingest.handover_numbers import closure_active
        if closure_active(cfg):
            seeded["nr"] = ("qr", "rain_number_seed")
            seeded["ni"] = ("qi", "ice_number_seed")
            seeded["ns"] = ("qs", "snow_number_seed")
            seeded["ng"] = ("qg", "graupel_number_seed")
    # Named schemes seed nothing here: their own entry blocks set every
    # number moment from its mass under the scheme's assumed distribution
    # and their adapters seed graupel
    # number and volume from graupel mass on the first call.
    return seeded


def cold_start_ndcnst_numbers(cfg):
    """Number moments a cold start sets to the scheme's NDCNST diagnostic.

    Morrison (mp=10) under correct-mode ``hydrometeor_handover_numbers``
    (:func:`woof.ingest.handover_numbers.closure_active`) starts its droplet
    number ``nc`` at NDCNST*1e6/rho everywhere (morrison.cu), not at the
    allocation zero :func:`source_absent_microphysics` lists, so its receipt
    states "finite and above zero" instead of an exact value
    (tools/hrrr_single_domain_benchmark._initial_hrrr_microphysics_receipt).
    Every other selection returns an empty set.
    """
    cfg = SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg
    if int(cfg.mp_physics) != 10:
        return frozenset()
    from woof.ingest.handover_numbers import closure_active
    return frozenset({"nc"}) if closure_active(cfg) else frozenset()


def hrrr_cold_start_preflight(selection):
    """Refuse, before any download, a microphysics an HRRR start cannot prepare.

    The HRRR preparation (tools/prepare_hrrr_wrf._validated_physics_receipt)
    checks a cold-start receipt for every source-absent prognostic field only
    after the whole forcing window is fetched.  This computes the same
    contract from the run's selection first, so a selection without one is
    refused by name while nothing has been spent.  Returns the field names.
    """
    cfg = SimpleNamespace(**selection) if isinstance(selection, dict) else selection
    try:
        fields, expected = cold_start_contract(cfg)
        special = set(cold_start_seeded_numbers(cfg)) | set(cold_start_ndcnst_numbers(cfg))
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(
            "an HRRR start cannot prepare this microphysics "
            f"(mp_physics={getattr(cfg, 'mp_physics', None)!r}): its cold-start "
            f"contract is undefined ({error}); refused before the download, so "
            "nothing was spent") from error
    unknown = special - set(expected)
    if unknown:
        raise ValueError(
            "an HRRR start cannot prepare this microphysics "
            f"(mp_physics={int(cfg.mp_physics)}): number moment(s) {sorted(unknown)} "
            "have no source-absent contract; refused before the download, so "
            "nothing was spent")
    return fields


def cold_start_contract(selection):
    """Wire-level expected FP32 values and bits, computed from active settings."""
    cfg = SimpleNamespace(**selection) if isinstance(selection, dict) else selection
    fields, values = source_absent_microphysics(cfg)
    expected = {}
    for name, value in values.items():
        packed = struct.pack("<f", value)
        expected[name] = (struct.unpack("<f", packed)[0], struct.unpack("<I", packed)[0])
    return fields, expected

from woof.config import THOMPSON_AEROSOL_FAMILY
