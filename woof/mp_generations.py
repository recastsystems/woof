"""WRF source generations of the NTU (mp_physics=56) and UDM (27) schemes.

THIRD-PARTY NOTICE: the statements below follow WRF v4.8.0
phys/module_mp_ntu.F and phys/module_mp_udm.F (public domain; see
licenses/LICENSE-WRF-public-domain.txt and the root NOTICE).

WRF v4.8.0 (06d4240a) changes three statements in these two schemes against
v4.7.1 (f52c197e), the release the shipping kernels transcribe:

* NTU: ``RATIO = (RHOGW/DNGRM)**THRD`` in the rain-graupel collection block
  is evaluated after DNGRM is set and clamped (v4.8.0 :5694-5707).  At
  v4.7.1 (:5695) it read DNGRM before the block set it: the value left by
  an earlier block or level, or undefined (the step-60 signalling-NaN
  witness in tools/ntu_wrf471_oracle/README.md).
* UDM: the ice-supersaturation budget ahead of ice nucleation subtracts hail
  deposition too, ``supice = satrdt-prevp-pidep-psdep-pgdep-phdep``
  (v4.8.0 :1959); and the graupel and hail fall densities ``denqg``/``denqh``
  are zeroed with the others at each substep (v4.8.0 :947), so the
  semi-Lagrangian fall no longer reads levels above the species top that
  were never written.

module_mp_radar.F's v4.8.0 ``xam_h``/``xbm_h`` guard changes no word here:
UDM is the only scheme that reads ``xoamh``/``xobmh``/``xocmh``, and it
sets ``xam_h``/``xbm_h`` > 0 before ``radar_init`` (module_mp_udm.F
:3479-3483), so the guard passes and the values are those already in
udm_generated.cuh.

``wrf_480`` is the default (a fix ships default-on).  ``wrf_471`` compiles
today's frozen sources unchanged.  The v4.8.0 kernels are those frozen
sources with the edits below applied at composition; each edit must match
exactly once, so a moved anchor refuses to compile rather than run the old
arithmetic under the new name.
"""
from __future__ import annotations

DEFAULT_GENERATION = "wrf_480"
#: What a run that names no generation ran before this selector existed.
PREVIOUS_GENERATION = "wrf_471"

#: Selector -> the mp_physics option it belongs to.
MP_GENERATION_SELECTORS = {"ntu_version": 56, "udm_version": 27}

_WRF_471 = "f52c197ed39d12e087d02c50f412d90d418f6186"
_WRF_480 = "06d4240ae989cc3e50af412bb472df3d9048783c"

GENERATIONS = {
    "ntu_version": {
        "wrf_471": {
            "source_commit": _WRF_471,
            "source_sha256": "c864483fa572d42ab1dc51cc538f8062dc2f582369f597261e8c850c49ce34dd",
            # None: the checkpoint table's mp_physics row (checkpoint_identity).
            "restart_algorithm_identity": None,
        },
        "wrf_480": {
            "source_commit": _WRF_480,
            "source_sha256": "e94adbcb0a42c74c51674d47fa6b7d73697b5ec89dcd31c1f877ab00e29af7de",
            "restart_algorithm_identity":
                "ntu-native-wrf-v4.8.0-v1-log10-native-aero-moments-set-rain-graupel-ratio",
        },
    },
    "udm_version": {
        "wrf_471": {
            "source_commit": _WRF_471,
            "source_sha256": "9e0feb16cfa964bad85f7c8fca95309eebc09efbeaee0fca5c664208e44bb82e",
            "restart_algorithm_identity": None,
        },
        "wrf_480": {
            "source_commit": _WRF_480,
            "source_sha256": "c76e37d6ad81dd0fd86064857fc8a352bc9075b1370262713a6b4abb53ef50b4",
            "restart_algorithm_identity":
                "udm-wrf480-v1-native-kinds-frozen-tables-native-radius-flags-zero-"
                "hail-deposition-supice-zeroed-fall-densities",
        },
    },
}


def generation(selector: str, value) -> dict:
    """The row for ``selector = value``; a name it does not have is refused."""
    rows = GENERATIONS[selector]
    try:
        return rows[value]
    except (KeyError, TypeError):
        raise ValueError(
            f"{selector} must be one of {tuple(rows)}, got {value!r}") from None


def selected(cfg_or_values, selector: str) -> str:
    """The generation a configuration (or its value mapping) runs."""
    if isinstance(cfg_or_values, dict):
        value = cfg_or_values.get(selector, DEFAULT_GENERATION)
    else:
        value = getattr(cfg_or_values, selector, DEFAULT_GENERATION)
    generation(selector, value)
    return value


def drop_inactive(values: dict) -> None:
    """Omit each selector while it reaches nothing or names the previous pin.

    Shared by the checkpoint echo and digest and the run and experiment
    documents.  With its scheme off the key reaches no arithmetic, and at
    ``wrf_471`` it is what every earlier header (which has no such key) ran,
    so those runs keep the bytes they had; the v4.8.0 selection binds.
    """
    scheme = int(values.get("mp_physics", 0) or 0)
    for selector, mp_physics in MP_GENERATION_SELECTORS.items():
        if (scheme != mp_physics
                or values.get(selector, PREVIOUS_GENERATION) == PREVIOUS_GENERATION):
            values.pop(selector, None)


def microphysics_algorithm_identity(cfg, table_identity: str) -> str:
    """The restart identity of the selected scheme's generation."""
    scheme = int(getattr(cfg, "mp_physics", 0) or 0)
    for selector, mp_physics in MP_GENERATION_SELECTORS.items():
        if scheme == mp_physics:
            row = generation(selector, selected(cfg, selector))
            return row["restart_algorithm_identity"] or table_identity
    return table_identity


def _once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"WRF v4.8.0 source anchor {label} moved; refusing to compile the "
            "v4.7.1 arithmetic under the v4.8.0 selection")
    return source.replace(old, new, 1)


_NTU_RHOGW = ("            RHOGW = ((QG1D + QR1D) / (((QG1D / RHOG) + (QR1D / NTU_RHOW))"
              " + NTU_ISMALL));\n")
_NTU_RATIO = "            RATIO = NTU_POWF((RHOGW / DNGRM), NTU_THRD);\n"
_NTU_DNGRM = "            DNGRM = ntu_fmin(ntu_fmax(DNGRM, NTU_RHOIMIN), NTU_RHOG0);\n"
_NTU_RIMING = "            if (((MVDG >= 2.0E-4f) && (MVDG >= MVDR))) {\n"


def ntu_source(source: str, version: str) -> str:
    """NTU's composed CUDA source for ``ntu_version``.

    The result's transcription block is f2cu.py's v4.8.0 transcription byte
    for byte (tools/ntu_wrf471_oracle/generate.py EXPECTED_BY_WRF).
    """
    generation("ntu_version", version)
    if version == "wrf_471":
        return source
    # module_mp_ntu.F v4.8.0:5694-5707: RATIO after the riming density.
    source = _once(source, _NTU_RHOGW + _NTU_RATIO, _NTU_RHOGW,
                   "NTU rain-graupel RATIO")
    return _once(source, _NTU_DNGRM + _NTU_RIMING,
                 _NTU_DNGRM + _NTU_RATIO + _NTU_RIMING,
                 "NTU rain-graupel riming density")


def _udm_index(name: str) -> str:
    return f"{name}[0+(k-(kts))*(1)]"


def _udm_zero(name: str) -> str:
    return (f"for(int udmslice0=kts;udmslice0<=kte;++udmslice0) "
            f"{name}[0+(udmslice0-(kts))*(1)] = 0.f;\n")


_UDM_SUPICE = ("satrdt", "prevp", "pidep", "psdep", "pgdep")


def _udm_difference(names) -> str:
    text = _udm_index(names[0])
    for name in names[1:]:
        text = f"({text}-{_udm_index(name)})"
    return text


def udm_header(header: str, version: str) -> str:
    """udm_generated.cuh for ``udm_version``.

    ``// source:N`` comments keep their v4.7.1 line numbers; the added
    statements are marked with their v4.8.0 lines.
    """
    generation("udm_version", version)
    if version == "wrf_471":
        return header
    # module_mp_udm.F v4.8.0:947.
    anchor = "// source:946\n" + _udm_zero("denqi")
    header = _once(header, anchor,
                   anchor + "// source:947 (v4.8.0)\n" + _udm_zero("denqh")
                   + "// source:947 (v4.8.0)\n" + _udm_zero("denqg"),
                   "UDM fall-density initialisation")
    # module_mp_udm.F v4.8.0:1959.
    supice = _udm_index("supice")
    return _once(
        header,
        f"// source:1958\n{supice} = {_udm_difference(_UDM_SUPICE)};\n",
        f"// source:1958 (v4.8.0:1959)\n{supice} = "
        f"{_udm_difference(_UDM_SUPICE + ('phdep',))};\n",
        "UDM ice-nucleation supersaturation budget")
