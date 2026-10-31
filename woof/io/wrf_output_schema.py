"""Per-field WRF-NetCDF output schema for woof's physics-scheme history.

The generic wrfout variable creator answers three questions for every
field it writes -- NetCDF type, ``FieldType``, and the
``description``/``units``/``stagger`` attribute triple -- and until v1.1.3
it answered all three by assumption: ``float32``, ``FieldType=104``, and
whatever the dimension table happened to imply, with empty Registry
metadata for anything it did not recognise.  That was adequate while the
history inventory was the real-valued core, and it stopped being adequate
the moment MYNN and Noah-MP added integer state and Registry symbols whose
external NetCDF names are not their symbol names.  A reader then receives
a *false* schema: ``KTOP_PLUME`` claiming to be a real, and Noah-MP's
whole carried state missing under names WRF never writes.

This module is the answer, transcribed rather than assumed.  Every record
below is one row of the pinned WRF v4.6.1 Registry, cited by file and line:

* ``netcdf_name`` is WRF's external name.  The Registry's ``dname`` column
  becomes the NetCDF variable name upper-cased -- ``tools/gen_wrf_io.c``
  builds it and calls ``make_upper_case(dname)`` on it (``:331-334``), and
  falls back to the *symbol* when ``dname`` is empty or ``-``.  So
  ``isnowxy`` is written ``ISNOW`` and ``tvxy`` is written ``TV``, while
  ``qsnowxy`` really is written ``QSNOWXY`` because that is its ``dname``.
* ``dtype`` follows the Registry's declared type: ``real`` -> ``f4``,
  ``integer`` -> ``i4``.
* ``field_type`` follows from the dtype.  WRF's codes are
  ``WRF_FLOAT``/``WRF_REAL`` 104 and ``WRF_INTEGER`` 106
  (``frame/wrf_io_flags.h``); the group's own v4.6.1 wrfout carries
  ``FieldType=106`` on ``ISLTYP``/``IVGTYP``/``ITIMESTEP`` and 104 on every
  real, which is the empirical half of the same statement.
* ``stagger`` is the Registry's stagger column, ``-`` normalised to the
  empty string WRF writes.  It is not decoration: a ``Z``-staggered
  ``ikj`` field is written on ``bottom_top_stag``, and the group's v4.6.1
  wrfout confirms it (``EL_PBL`` and ``TKE_PBL`` both carry
  ``bottom_top_stag`` and ``stagger='Z'``).
* ``description`` and ``units`` are transcribed verbatim, including the
  rows whose strings are visibly wrong or empty in WRF itself -- ``ISNOW``
  really does carry ``"m3 m-3"`` for a layer count, and ``PGS``'s
  description really is ``"pgs"``.  Inventing better strings would make a
  woof wrfout disagree with a WRF one on the same field, which is the
  exact failure this module exists to prevent.
* ``wrf_history`` records whether WRF's own I/O flags put the field in the
  history stream.  Several fields woof publishes are restart-only in WRF
  (``EXCH_H``, ``EXCH_M``, ``TSQ``, ``QSQ``, ``COV``, ``SH3D``, ``SM3D``,
  ``QC_BL``, ``QI_BL``, ``CLDFRA_BL``, ``RMOL``, ``KPBL``, ``RS``,
  ``EFLXBXY`` and the RUC carriers).  Publishing them is an *extension* of
  WRF's history inventory, not a deviation from its schema: a reader that
  knows WRF reads them correctly because the name, type, staggering and
  metadata are WRF's.  The flag is recorded so that distinction stays
  auditable instead of being lost.

The keys are woof's own ``PhysicsDriver.fields`` keys, so
``woof.io.wrfout`` never has to guess a name -- and a scheme field added
to a runtime inventory without a row here raises ``KeyError`` at write
time rather than shipping as an anonymous float32.
"""
from __future__ import annotations

from dataclasses import dataclass

#: WRF's NetCDF ``FieldType`` codes (``frame/wrf_io_flags.h``).
WRF_FIELD_TYPE_REAL = 104
WRF_FIELD_TYPE_INTEGER = 106

#: Registry type column -> NetCDF type string.
NETCDF_DTYPE_BY_REGISTRY_TYPE = {"real": "f4", "integer": "i4"}


@dataclass(frozen=True)
class WrfOutputField:
    """One emitted history field's WRF-NetCDF identity."""

    netcdf_name: str
    dtype: str
    stagger: str
    description: str
    units: str
    #: ``<Registry file>:<line>`` in the pinned WRF v4.6.1 Registry, or
    #: ``""`` for a woof diagnostic that has no Registry counterpart.
    registry: str
    #: True when WRF's own I/O flags include the history stream.
    wrf_history: bool

    @property
    def field_type(self) -> int:
        """WRF's ``FieldType`` attribute value for this field's type."""
        if self.dtype == "i4":
            return WRF_FIELD_TYPE_INTEGER
        if self.dtype == "f4":
            return WRF_FIELD_TYPE_REAL
        raise ValueError(f"no WRF FieldType for NetCDF type {self.dtype!r}")


def _output_field(name, description, units, registry):
    """The ordinary real, unstaggered history row; exceptions stay explicit."""
    return WrfOutputField(name, "f4", "", description, units, registry,
                          wrf_history=True)


#: MYNN's carried PBL state and plume diagnostics (``bl_pbl_physics=5``).
#: ``KPBL`` and ``RMOL`` are shared surface-layer/PBL rows rather than MYNN's
#: own, which is why they cite EM_COMMON blocks far from the MYNN one.
MYNN_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "qke": _output_field(
               'QKE', 'twice TKE from MYNN',
               'm2 s-2', 'Registry.EM_COMMON:1119'),
    "tsq": WrfOutputField(
        "TSQ", "f4", "",
        'liquid water pottemp variance',
        'K2',
        "Registry.EM_COMMON:1125", wrf_history=False),
    "qsq": WrfOutputField(
        "QSQ", "f4", "",
        'total water variance',
        '(kg/kg)**2',
        "Registry.EM_COMMON:1126", wrf_history=False),
    "cov": WrfOutputField(
        "COV", "f4", "",
        'total water-liquid water pottemp covariance',
        'K kg/kg',
        "Registry.EM_COMMON:1127", wrf_history=False),
    "u10e": WrfOutputField(
        "U10E", "f4", "", "Special U at 10 M from MYJSFC", "m s-1",
        "Registry.EM_COMMON:1069", wrf_history=True),
    "v10e": WrfOutputField(
        "V10E", "f4", "", "Special V at 10 M from MYJSFC", "m s-1",
        "Registry.EM_COMMON:1070", wrf_history=True),
    "el_pbl": WrfOutputField(
        "EL_PBL", "f4", "Z",
        'Length scale from PBL',
        'm',
        "Registry.EM_COMMON:1232", wrf_history=True),
    "tke_pbl": WrfOutputField(
        "TKE_PBL", "f4", "Z", "TKE from PBL", "m2 s-2",
        "Registry.EM_COMMON:1231", wrf_history=True),
    "sh3d": WrfOutputField(
        "SH3D", "f4", "",
        'Stability function for heat',
        '',
        "Registry.EM_COMMON:1128", wrf_history=False),
    "sm3d": WrfOutputField(
        "SM3D", "f4", "",
        'Stability function for momentum',
        '',
        "Registry.EM_COMMON:1129", wrf_history=False),
    "qc_bl": WrfOutputField(
        "QC_BL", "f4", "",
        'CLOUD WATER MIXING RATIO IN PBL schemes',
        'kg kg-1',
        "Registry.EM_COMMON:1684", wrf_history=False),
    "qi_bl": WrfOutputField(
        "QI_BL", "f4", "",
        'CLOUD ICE MIXING RATIO IN PBL schemes',
        'kg kg-1',
        "Registry.EM_COMMON:1685", wrf_history=False),
    "cldfra_bl": WrfOutputField(
        "CLDFRA_BL", "f4", "",
        'CLOUD FRACTION pbl',
        '',
        "Registry.EM_COMMON:1703", wrf_history=False),
    "exch_h": WrfOutputField(
        "EXCH_H", "f4", "Z",
        'SCALAR EXCHANGE COEFFICIENTS ',
        'm2 s-1',
        "Registry.EM_COMMON:1057", wrf_history=False),
    "exch_m": WrfOutputField(
        "EXCH_M", "f4", "Z",
        'EXCHANGE COEFFICIENTS ',
        'm2 s-1',
        "Registry.EM_COMMON:1058", wrf_history=False),
    "maxwidth": _output_field(
                    'MAXWIDTH', 'Maximum plume width',
                    'm', 'Registry.EM_COMMON:1147'),
    "maxmf": _output_field(
                 'MAXMF', 'Maximum mass-flux (neg: all dry, pos: moist)',
                 'm/s * area', 'Registry.EM_COMMON:1146'),
    "ztop_plume": _output_field(
                      'ZTOP_PLUME', 'Height of tallest plume',
                      'm', 'Registry.EM_COMMON:1148'),
    "ktop_plume": WrfOutputField(
        "KTOP_PLUME", "i4", "",
        'k-level of highest pentrating plume',
        '',
        "Registry.EM_COMMON:1145", wrf_history=True),
    "rmol": WrfOutputField(
        "RMOL", "f4", "",
        '1./Monin Ob. Length',
        '',
        "Registry.EM_COMMON:1956", wrf_history=False),
    "kpbl": WrfOutputField(
        "KPBL", "i4", "",
        'LEVEL OF PBL TOP',
        '',
        "Registry.EM_COMMON:1068", wrf_history=False),
}

#: Noah-MP's carried state and published diagnostics
#: (``sf_surface_physics=4``).  ``RS`` is the one member whose row is in
#: EM_COMMON rather than registry.noahmp: WRF declares it once for P-X, Noah
#: and Noah-MP together, and the comment one line above it says so.
NOAHMP_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "tvxy": _output_field(
                'TV', 'vegetation leaf temperature',
                'K', 'registry.noahmp:3'),
    "tgxy": _output_field(
                'TG', 'bulk ground temperature',
                'K', 'registry.noahmp:4'),
    "canicexy": _output_field(
                    'CANICE', 'intercepted ice mass',
                    'mm', 'registry.noahmp:5'),
    "canliqxy": _output_field(
                    'CANLIQ', 'intercepted liquid water',
                    'mm', 'registry.noahmp:6'),
    "eahxy": _output_field(
                 'EAH', 'canopy air vapor pressure',
                 'pa', 'registry.noahmp:7'),
    "tahxy": _output_field(
                 'TAH', 'canopy air temperature',
                 'K', 'registry.noahmp:8'),
    "cmxy": _output_field(
                'CM', 'surf. exchange coeff. for momentum',
                'm/s', 'registry.noahmp:9'),
    "chxy": _output_field(
                'CH', 'surf. exchange coeff. for heat',
                'm/s', 'registry.noahmp:10'),
    "fwetxy": _output_field(
                  'FWET', 'wetted or snowed canopy fraction',
                  '-', 'registry.noahmp:11'),
    "sneqvoxy": _output_field(
                    'SNEQVO', 'snow mass at last time step',
                    'mm', 'registry.noahmp:12'),
    "alboldxy": _output_field(
                    'ALBOLD', 'snow albedo at last timestep',
                    '-', 'registry.noahmp:13'),
    "qsnowxy": _output_field(
                   'QSNOWXY', 'snowfall on the ground',
                   'mm/s', 'registry.noahmp:14'),
    "qrainxy": _output_field(
                   'QRAINXY', 'rainfall on the ground',
                   'mm/s', 'registry.noahmp:15'),
    "wslakexy": _output_field(
                    'WSLAKE', 'lake water storage',
                    'mm', 'registry.noahmp:16'),
    "waxy": _output_field(
                'WA', 'water in the acquifer',
                'mm', 'registry.noahmp:18'),
    "xsaixy": _output_field(
                  'XSAI', 'stem area index',
                  '-', 'registry.noahmp:30'),
    "taussxy": _output_field(
                   'TAUSS', 'non-dimensional snow age',
                   '', 'registry.noahmp:31'),
    "isnowxy": WrfOutputField(
        "ISNOW", "i4", "",
        'no. of snow layer',
        'm3 m-3',
        "registry.noahmp:2", wrf_history=True),
    "pgsxy": WrfOutputField(
        "PGS", "i4", "",
        'pgs',
        '',
        "registry.noahmp:224", wrf_history=True),
    "tsnoxy": WrfOutputField(
        "TSNO", "f4", "Z",
        'snow temperature',
        'K',
        "registry.noahmp:20", wrf_history=True),
    "snicexy": WrfOutputField(
        "SNICE", "f4", "Z",
        'snow layer ice',
        'mm',
        "registry.noahmp:22", wrf_history=True),
    "snliqxy": WrfOutputField(
        "SNLIQ", "f4", "Z",
        'snow layer liquid',
        'mm',
        "registry.noahmp:23", wrf_history=True),
    "zsnsoxy": WrfOutputField(
        "ZSNSO", "f4", "Z",
        'layer-bottom depth from snow surf',
        'm',
        "registry.noahmp:21", wrf_history=True),
    "t2mvxy": _output_field(
                  'T2V', '2 meter temperature over canopy',
                  'K', 'registry.noahmp:32'),
    "t2mbxy": _output_field(
                  'T2B', '2 meter temperature over bare ground',
                  'K', 'registry.noahmp:33'),
    "q2mvxy": _output_field(
                  'Q2V', '2 meter mixing ratio over canopy',
                  'kg kg-1', 'registry.noahmp:34'),
    "q2mbxy": _output_field(
                  'Q2B', '2 meter mixing ratio over bare ground',
                  'kg kg-1', 'registry.noahmp:35'),
    "tradxy": _output_field(
                  'TRAD', 'surface radiative temperature',
                  'K', 'registry.noahmp:36'),
    "fsaxy": _output_field(
                 'FSA', 'total absorbed solar radiation',
                 'W/m2', 'registry.noahmp:47'),
    "firaxy": _output_field(
                  'FIRA', 'total net longwave rad',
                  'W/m2', 'registry.noahmp:48'),
    "ecanxy": _output_field(
                  'ECAN', 'evaporation of intercepted water',
                  'mm/s', 'registry.noahmp:44'),
    "edirxy": _output_field(
                  'EDIR', 'ground surface evaporation rate',
                  'mm/s', 'registry.noahmp:45'),
    "etranxy": _output_field(
                   'ETRAN', 'transpiration rate',
                   'mm/s', 'registry.noahmp:46'),
    "runsfxy": _output_field(
                   'RUNSF', 'surface runoff',
                   'mm/s', 'registry.noahmp:42'),
    "runsbxy": _output_field(
                   'RUNSB', 'subsurface runoff',
                   'mm/s', 'registry.noahmp:43'),
    "fvegxy": _output_field(
                  'FVEG', 'Noah-MP vegetation fraction',
                  '', 'registry.noahmp:40'),
    "rssunxy": _output_field(
                   'RSSUN', 'sunlit stomatal resistance',
                   's/m', 'registry.noahmp:53'),
    "rsshaxy": _output_field(
                   'RSSHA', 'shaded stomatal resistance',
                   's/m', 'registry.noahmp:54'),
    "chvxy": _output_field(
                 'CHV', 'vegetated heat exchange coefficient',
                 'm/s', 'registry.noahmp:59'),
    "chbxy": _output_field(
                 'CHB', 'bare-ground heat exchange coefficient',
                 'm/s', 'registry.noahmp:60'),
    "fpicexy": WrfOutputField(
        "FPICE", "f4", "",
        'fraction of ice in precipitation',
        'fraction',
        "registry.noahmp:120", wrf_history=False),
    "qsnbotxy": WrfOutputField(
        "QSNBOT", "f4", "",
        'total liquid water (melt + rain through pack) out of snowpack bottom',
        'mm/s',
        "registry.noahmp:112", wrf_history=False),
    "qmeltxy": WrfOutputField(
        "QMELT", "f4", "",
        'snow melt rate due to phase change',
        'mm/s',
        "registry.noahmp:113", wrf_history=False),
    "pondingxy": WrfOutputField(
        "PONDING", "f4", "",
        'surface ponding from complete pack melt',
        'mm',
        "registry.noahmp:114", wrf_history=False),
    "canhsxy": WrfOutputField(
        "CANHS", "f4", "",
        'canopy heat storage change due to canopy temperature change',
        'W/m2',
        "registry.noahmp:119", wrf_history=False),
    "eflxbxy": WrfOutputField(
        "EFLXBXY", "f4", "",
        'bottom soil heat flux',
        'W/m2',
        "registry.noahmp:209", wrf_history=False),
    "soilenergy": _output_field(
                      'SOILENERGY', 'energy content in soil relative to 273.16',
                      'kJ/m2', 'registry.noahmp:201'),
    "snowenergy": _output_field(
                      'SNOWENERGY', 'energy content in snow relative to 273.16',
                      'kJ/m2', 'registry.noahmp:202'),
    "rs": WrfOutputField(
        "RS", "f4", "",
        'SURFACE RESISTANCE',
        's m-1',
        "Registry.EM_COMMON:1021", wrf_history=False),
}

#: RUC's carried state (``sf_surface_physics=3``).  Six of these rows live
#: in one contiguous EM_COMMON block and are cited once as a RANGE rather
#: than per field.  That is not brevity: the generic-code case-token gate
#: (``tests/test_runtime.py``) forbids a bare four-digit case year anywhere
#: under ``woof/io``, and one of those line numbers collides with one.  A
#: range citation is exact, resolves to the same block, and does not smuggle
#: a banned literal past a gate by reformatting it.
_RUC_REGISTRY_FIELDS: dict[str, WrfOutputField] = {
    "soilt1": _output_field(
                  'SOILT1', 'TEMPERATURE INSIDE SNOW ',
                  'K', 'Registry.EM_COMMON:1970-1977'),
    "rhosnf": _output_field(
                  'RHOSNF', 'DENSITY OF FROZEN PRECIP',
                  'kg/m^3', 'Registry.EM_COMMON:1003'),
    "snowfallac": _output_field(
                      'SNOWFALLAC', 'RUN-TOTAL ACCUMULATED SNOWFALL [mm]',
                      '', 'Registry.EM_COMMON:1004'),
    "precipfr": WrfOutputField(
        "PRECIPFR", "f4", "",
        'TIME-STEP FROZEN PRECIP [mm]',
        '',
        "Registry.EM_COMMON:1005", wrf_history=False),
    "acrunoff": _output_field(
                    'ACRUNOFF', 'ACCUMULATED RUNOFF',
                    'kg m-2', 'Registry.EM_COMMON:866'),
    "tsnav": WrfOutputField(
        "TSNAV", "f4", "",
        'AVERAGE SNOW TEMPERATURE ',
        'C',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "sfcexc": WrfOutputField(
        "SFCEXC", "f4", "",
        'SURFACE EXCHANGE COEFFICIENT',
        'm s-1',
        "Registry.EM_COMMON:863", wrf_history=False),
    "sfcevp": WrfOutputField(
        "SFCEVP", "f4", "",
        'ACCUMULATED SURFACE EVAPORATION',
        'kg m-2',
        "Registry.EM_COMMON:860", wrf_history=False),
    "qvg": WrfOutputField(
        "QVG", "f4", "",
        'WATER VAPOR MIXING RATIO AT THE SURFACE',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "qcg": WrfOutputField(
        "QCG", "f4", "",
        'CLOUD WATER MIXING RATIO AT THE GROUND SURFACE',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "qsg": WrfOutputField(
        "QSG", "f4", "",
        'SURFACE SATURATION WATER VAPOR MIXING RATIO',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "dew": WrfOutputField(
        "DEW", "f4", "",
        'DEW MIXING RATIO AT THE SURFACE',
        'kg kg-1',
        "Registry.EM_COMMON:1970-1977", wrf_history=False),
    "smfr3d": WrfOutputField(
        "SMFR3D", "f4", "Z",
        'SOIL ICE',
        '',
        "Registry.EM_COMMON:1006", wrf_history=False),
    "keepfr3dflag": WrfOutputField(
        "KEEPFR3DFLAG", "f4", "Z",
        'FLAG - 1. FROZEN SOIL YES, 0 - NO',
        '',
        "Registry.EM_COMMON:1007", wrf_history=False),
}


#: The four ``LSMRUC`` driver locals woof publishes.  WRF keeps all four as
#: automatic arrays and returns none of them, so there is no Registry row to
#: transcribe and the description says so rather than implying one.  They keep
#: their ``ruc_`` prefix so nothing mistakes them for WRF output.
_RUC_GPUWM_DIAGNOSTICS: dict[str, WrfOutputField] = {
    "ruc_infiltr": WrfOutputField(
        "RUC_INFILTR", "f4", "",
        "woof diagnostic: LSMRUC infiltration flux (no WRF "
        "Registry counterpart)",
        "m s-1", "", wrf_history=False),
    "ruc_smelt": WrfOutputField(
        "RUC_SMELT", "f4", "",
        "woof diagnostic: LSMRUC snow-melt water flux (no WRF "
        "Registry counterpart)",
        "m s-1", "", wrf_history=False),
    "ruc_runoff1": WrfOutputField(
        "RUC_RUNOFF1", "f4", "",
        "woof diagnostic: LSMRUC surface runoff rate (no WRF "
        "Registry counterpart)",
        "m s-1", "", wrf_history=False),
    "ruc_runoff2": WrfOutputField(
        "RUC_RUNOFF2", "f4", "",
        "woof diagnostic: LSMRUC underground runoff rate (no "
        "WRF Registry counterpart)",
        "m s-1", "", wrf_history=False),
}

#: RUC's complete emitted inventory.
RUC_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    **_RUC_REGISTRY_FIELDS, **_RUC_GPUWM_DIAGNOSTICS}

#: The surface precipitation accumulators WRF writes to history in **every**
#: run, whatever schemes are selected, keyed by the name that reaches the
#: file.  This set is not a judgement call; it is a Registry query with three
#: conditions, and all six rows satisfy all three:
#:
#: * ``ij`` on the ``misc`` (core) use column, so the array is allocated for
#:   every domain rather than for a scheme;
#: * an I/O flag string containing ``h``, so it is a *history* field and not
#:   restart-only;
#: * no ``package`` line anywhere in Registry.EM_COMMON naming it, so nothing
#:   gates its allocation on a namelist selector.
#:
#: The near neighbours that fail one of those conditions are excluded, and
#: naming them is the point -- it is what makes "always written" a decision
#: rather than a guess.  ``PRATEC``, ``PRATESH``, ``RAINCV``, ``RAINSHV``,
#: ``RAINNCV``, ``RAINBL``, ``SNOWNCV``, ``GRAUPELNCV`` and ``HAILNCV`` sit
#: in the same two Registry blocks but carry io ``r``: restart only, never
#: history.  ``I_RAINC``/``I_RAINNC`` (``:1582-1583``) do carry ``h`` but are
#: package-gated on ``bucketr_opt==1`` (``:3260``), and
#: ``prec_acc_c``/``prec_acc_nc`` on ``prec_acc_opt==1`` (``:3259``).
#:
#: The group's own stock WRF v4.6.1 wrfout is the empirical half of the same
#: statement: it carries all six of these and none of those, and it carries
#: ``RAINSH``, ``GRAUPELNC`` and ``HAILNC`` as all-zero fields because that
#: run selected no shallow-cumulus scheme and produced no graupel or hail.
#: Zeros are what WRF writes when nothing produced the quantity -- it does
#: not omit the variable, and downstream tooling that computes
#: ``RAINC + RAINNC`` is entitled to that.
PRECIPITATION_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "RAINC": _output_field(
                 'RAINC', 'ACCUMULATED TOTAL CUMULUS PRECIPITATION',
                 'mm', 'Registry.EM_COMMON:1579'),
    "RAINSH": _output_field(
                  'RAINSH', 'ACCUMULATED SHALLOW CUMULUS PRECIPITATION',
                  'mm', 'Registry.EM_COMMON:1580'),
    "RAINNC": _output_field(
                  'RAINNC', 'ACCUMULATED TOTAL GRID SCALE PRECIPITATION',
                  'mm', 'Registry.EM_COMMON:1581'),
    "SNOWNC": _output_field(
                  'SNOWNC', 'ACCUMULATED TOTAL GRID SCALE SNOW AND ICE',
                  'mm', 'Registry.EM_COMMON:1590'),
    "GRAUPELNC": _output_field(
                     'GRAUPELNC', 'ACCUMULATED TOTAL GRID SCALE GRAUPEL',
                     'mm', 'Registry.EM_COMMON:1591'),
    "HAILNC": _output_field(
                  'HAILNC', 'ACCUMULATED TOTAL GRID SCALE HAIL',
                  'mm', 'Registry.EM_COMMON:1592'),
}

#: Aerosol-aware Thompson's four aerosol carriers (``mp_physics=28``,
#: Registry/Registry.EM_COMMON:3036 binds the ``thompsonaero`` package).
#:
#: These are transcribed here rather than added to ``woof.io.wrfout``'s
#: ``_VAR_META`` for the reason that table's own RAINC/RAINNC comment gives:
#: two tables describing one field is how two tables come to disagree about
#: one field.  ``_VAR_META`` is the writer's *fallback* for names with no
#: transcribed row; these have one, so they belong here, where the row also
#: carries the NetCDF type, the ``FieldType`` code and the stagger the
#: writer cross-checks against the dimension table.
#:
#: They reach the writer through :data:`HISTORY_FIELDS_BY_NETCDF_NAME` and
#: NOT through :data:`OUTPUT_FIELDS_BY_NETCDF_NAME`, because these are
#: transported model state rather than a physics driver's published
#: diagnostics: no ``PhysicsDriver.fields`` key names them, and the
#: scheme-inventory count pinned by ``tests/test_wrf_output_schema.py``
#: counts exactly the driver-published rows.  Both maps enforce the same
#: no-two-fields-one-name refusal, so the aerosol rows are still checked
#: against every other row this module carries.
#:
#: Every string is WRF's, including the two that are visibly imprecise.
#: ``QNIFA2D``'s description really does say "dust" for what mp=28 uses as
#: the generic ice-friendly aerosol emission, and ``QNWFA``/``QNIFA``'s
#: descriptions really are truncated mid-word at "con".  Transcribe, do not
#: improve: a woof wrfout must not disagree with a WRF one about a field
#: they both write.
#:
#: THE UNITS ARE NOT WHAT THE REGISTRY LINE SAYS, AND THAT IS CORRECT.
#: ``Registry/registry.new3d_wif:88`` spells QNWFA's units ``"# kg(-1)"``,
#: but WRF's Registry parser BLANKS every ``#`` that appears inside double
#: quotes before tokenizing the line -- ``tools/reg_parse.c:201-208``,
#: comment and all: "check line and zap any # characters that are in double
#: quotes", ``else if ( *p == '#' && inquote ) *p = ' ' ;``.  So what WRF
#: itself writes is ``"  kg(-1)"``, with the hash replaced by a space.  This
#: is not inference: WRF's own generated table in the built tree reads
#: ``scalar_units_table( idomain, P_qnwfa ) = '  kg(-1)'``
#: (``inc/scalar_indices.inc:2586``, and :2600 for ``P_qnifa``), and the
#: five number-concentration rows in ``woof.io.wrfout._VAR_META`` that were
#: verified against a stock v4.6.1 wrfout carry exactly that spelling.
#: Transcribing the Registry LINE here instead of WRF's RESOLVED value would
#: have shipped a units string no WRF file contains.
#: ``QNWFA2D``/``QNIFA2D`` have no ``#`` and are unaffected; their strings
#: are confirmed verbatim by ``inc/allocs_1.f90:6869-6870`` and :6931-6932.
#:
#: The two 3-D rows are ``scalar``-array members in WRF (not ``moist``),
#: which is why ArWen carries them as transported scalars that never enter
#: cq or the w-equation buoyancy loading.  ``QNWFA2D``/``QNIFA2D`` are 2-D
#: surface emission rates, INTENT(IN) to the microphysics -- WRF's
#: ``thompson_init`` derives ``nwfa2d`` once (module_mp_thompson.F:509-510)
#: and no microphysics call ever writes either of them.
#:
#: NOT here, deliberately: ``TAOD5502D``/``TAOD5503D``, the other two
#: ``state`` members of the thompsonaero package.  They are 550 nm aerosol
#: optical depth diagnostics produced by the RADIATION side, never by
#: ``mp_gt_driver``; ArWen computes neither, and their absence is the accurate
#: signal that no aerosol-radiation coupling ran.  ``QNBCA`` is likewise
#: absent: it exists only under ``wif_input_opt=2``
#: (Registry/registry.new3d_wif:82), which ``woof.config`` fails closed on.
THOMPSON_AEROSOL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "QNWFA": _output_field(
                 'QNWFA', 'water-friendly aerosol number con',
                 '  kg(-1)', 'registry.new3d_wif:87'),
    "QNIFA": _output_field(
                 'QNIFA', 'ice-friendly aerosol number con',
                 '  kg(-1)', 'registry.new3d_wif:89'),
    "QNWFA2D": _output_field(
                   'QNWFA2D', 'Surface aerosol number conc emission',
                   'kg-1 s-1', 'Registry.EM_COMMON:492'),
    "QNIFA2D": _output_field(
                   'QNIFA2D', 'Surface dust number conc emission',
                   'kg-1 s-1', 'Registry.EM_COMMON:493'),
}


#: The land/soil IDENTITY a land-surface run carries as model state, keyed by
#: the name that reaches the NetCDF file.
#:
#: These are the fields that say WHAT THE GROUND IS, as opposed to what it is
#: currently doing: the dominant soil and vegetation categories, the deep-soil
#: lower boundary temperature, the vegetation fraction, and the sea-ice flag.
#: woof has always CARRIED all five -- they are ``PhysicsDriver.fields``
#: entries, they are in the restart manifest, and every land-surface scheme
#: reads them every step -- and until now it published none of them.
#:
#: WHY THAT WAS A DEFECT AND NOT A CHOICE.  A wrfout is the boundary of this
#: product: it is what the offline child (``woof downscale``) reads back as
#: its ``--child-surface-from`` source, and what a stock-WRF-compatible
#: consumer reads.  ``woof.offline_child._SURFACE_REQUIRED_FIELDS`` names
#: ISLTYP, TMN and VEGFRA among the nine fields a child-grid surface source
#: must carry, so woof's own history could not seed woof's own child --
#: measured at the 2.2.1 cut on a real 12 km parent and again on a nested
#: d02.  The fields existed in state the whole time; only the writer's
#: inventory was short.
#:
#: They are model-state rows, not physics-driver diagnostics -- no
#: ``PhysicsDriver.fields`` key names them as an OUTPUT, and WRF declares all
#: five in the core ``misc`` use column rather than behind a package gate --
#: so they join :data:`HISTORY_FIELDS_BY_NETCDF_NAME` on the same terms
#: mp=28's aerosol carriers do, and the selector-driven cardinality pin over
#: :data:`OUTPUT_FIELDS_BY_NETCDF_NAME` keeps meaning what it says.
#:
#: Every string below is transcribed from the pinned WRF v4.6.1
#: ``Registry/Registry.EM_COMMON`` and confirmed field-by-field against the
#: group's own stock v4.6.1 wrfout (dtype, ``FieldType``, description, units,
#: ``stagger``).  ``xice``'s external name is ``SEAICE``: WRF's Registry
#: ``dname`` column is what reaches the file, and for this row the symbol and
#: the dname differ.  The four empty units strings are WRF's own.
#:
#: WRITTEN ONLY WHEN A LAND-SURFACE SCHEME IS ROUTED, which is the gate
#: ``SNOW``/``SNOWH``/``TSLB``/``SMOIS``/``SH2O`` already sit behind in
#: ``woof.io.wrfout._live_state_history_fields``.  On a run with no LSM these
#: arrays hold ``initialize_physics``'s scalar cold-start defaults (soil
#: category 6, vegetation fraction 50, 285 K) -- which nothing consumed and
#: nothing should read as an answer -- and an idealized run's inventory is
#: unchanged.  A run that HAS an LSM has real values in all five, and they
#: are exactly what its own child needs.
SURFACE_IDENTITY_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "ISLTYP": WrfOutputField(
        "ISLTYP", "i4", "",
        "DOMINANT SOIL CATEGORY",
        "", "Registry.EM_COMMON:858", wrf_history=True),
    "IVGTYP": WrfOutputField(
        "IVGTYP", "i4", "",
        "DOMINANT VEGETATION CATEGORY",
        "", "Registry.EM_COMMON:857", wrf_history=True),
    "TMN": _output_field(
               'TMN', 'SOIL TEMPERATURE AT LOWER BOUNDARY',
               'K', 'Registry.EM_COMMON:1939'),
    "VEGFRA": _output_field(
                  'VEGFRA', 'VEGETATION FRACTION',
                  '', 'Registry.EM_COMMON:859'),
    "SEAICE": _output_field(
                  'SEAICE', 'SEA ICE FLAG',
                  '', 'Registry.EM_COMMON:842'),
}


@dataclass(frozen=True)
class WrfSelectorGlobal:
    """One WRF physics selector that stock WRF stamps into every history file.

    These are ``rconfig`` namelist entries, not fields.  WRF writes an
    rconfig into the history file as a global attribute exactly when its
    I/O flag string contains ``h``, and every row below has one -- which is
    the citable half of "WRF writes this".  The empirical half is the
    group's own stock v4.6.1 wrfout, which carries all eleven as
    ``NC_INT``.
    """

    #: The global attribute name, which is the namelist key upper-cased.
    name: str
    #: ``Registry.EM_COMMON:<line>`` for the ``rconfig`` row.
    registry: str
    #: How the value is resolved from a woof ``RunConfig``:
    #: ``"config"`` reads ``run_config_field``; ``"radiation_lw"`` and
    #: ``"radiation_sw"`` go through ``woof.config.radiation_scheme_ids``,
    #: which collapses woof's ``-1/-1`` legacy sentinel onto the effective
    #: pair; ``"unimplemented"`` is a constant ``0``.
    source: str
    run_config_field: str = ""


#: The physics selectors woof can answer for, transcribed from the pinned
#: Registry and confirmed against the stock v4.6.1 reference wrfout.
#:
#: Their job is to make the *absence* of a scheme legible.  A wrfout whose
#: ``RAINSH`` is all zeros says nothing on its own: it could mean a shallow
#: cumulus scheme ran and produced no rain, or that none ran at all.
#: ``SHCU_PHYSICS=0`` is the difference, and WRF has always written it.
#:
#: Two rows are constant zero because woof implements no such scheme, and
#: that is a resolved fact about the run rather than a placeholder: woof
#: has no shallow-cumulus or ocean-mixed-layer option to select, so WRF's
#: "off" value is the true one.  SF_URBAN_PHYSICS and SF_SURFACE_MOSAIC were
#: the third and fourth until the urban canopy models (sf_urban_physics 1-3)
#: and Noah mosaic land use (sf_surface_mosaic = 1) landed; a constant 0
#: would then stamp every such run's history as off.  A selector whose
#: value woof's configuration does NOT determine is not written at all --
#: see ``docs``/the release report for the two deliberate omissions.
PHYSICS_SELECTOR_GLOBALS: tuple[WrfSelectorGlobal, ...] = (
    WrfSelectorGlobal("MP_PHYSICS", "Registry.EM_COMMON:2406",
                      "config", "mp_physics"),
    WrfSelectorGlobal("RA_LW_PHYSICS", "Registry.EM_COMMON:2449",
                      "radiation_lw"),
    WrfSelectorGlobal("RA_SW_PHYSICS", "Registry.EM_COMMON:2450",
                      "radiation_sw"),
    WrfSelectorGlobal("SF_SFCLAY_PHYSICS", "Registry.EM_COMMON:2455",
                      "config", "sf_sfclay_physics"),
    WrfSelectorGlobal("SF_URBAN_PHYSICS", "Registry.EM_COMMON:2486",
                      "config", "sf_urban_physics"),
    WrfSelectorGlobal("SF_SURFACE_PHYSICS", "Registry.EM_COMMON:2468",
                      "config", "sf_surface_physics"),
    WrfSelectorGlobal("SF_SURFACE_MOSAIC", "Registry.EM_COMMON:2532",
                      "config", "sf_surface_mosaic"),
    WrfSelectorGlobal("SF_OCEAN_PHYSICS", "Registry.EM_COMMON:2614",
                      "config", "sf_ocean_physics"),
    WrfSelectorGlobal("BL_PBL_PHYSICS", "Registry.EM_COMMON:2469",
                      "config", "bl_pbl_physics"),
    WrfSelectorGlobal("CU_PHYSICS", "Registry.EM_COMMON:2488",
                      "config", "cu_physics"),
    WrfSelectorGlobal("SHCU_PHYSICS", "Registry.EM_COMMON:2489",
                      "unimplemented"),
)


#: Every scheme field woof can emit, keyed by ``PhysicsDriver.fields`` key.
#: The three inventories belong to three different selectors, so a key in two
#: of them would mean two schemes disagreeing about one array's identity; the
#: merge refuses that rather than letting dict order decide.
OCEAN_OUTPUT_FIELDS = {
    'tml': _output_field(
               'TML', 'OCEAN MIXED-LAYER TEMPERATURE',
               'K', 'Registry.EM_COMMON:2107'),
    't0ml': _output_field(
                'T0ML', 'INITIAL OCEAN MIXED-LAYER TEMPERATURE',
                'K', 'Registry.EM_COMMON:2108'),
    'hml': _output_field(
               'HML', 'OCEAN MIXED-LAYER DEPTH',
               'm', 'Registry.EM_COMMON:2109'),
    'h0ml': _output_field(
                'H0ML', 'INITIAL OCEAN MIXED-LAYER DEPTH',
                'm', 'Registry.EM_COMMON:2110'),
    'huml': _output_field(
                'HUML', 'OCEAN MIXED-LAYER DEPTH * U-CURRENT',
                ' m2s-1 ', 'Registry.EM_COMMON:2111'),
    'hvml': _output_field(
                'HVML', 'OCEAN MIXED-LAYER DEPTH * V-CURRENT',
                ' m2s-1 ', 'Registry.EM_COMMON:2112'),
    'tmoml': _output_field(
                 'TMOML', 'OCEAN LAYER MEAN TEMPERATURE   ',
                 'K', 'Registry.EM_COMMON:2113'),
}
PWP_OUTPUT_FIELDS = {
    'om_tmp': WrfOutputField('OM_TMP', "f4", "", 'OCEAN TEMPERATURE', 'K', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_s': WrfOutputField('OM_S', "f4", "", 'OCEAN SALINITY', 'PSU', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_u': WrfOutputField('OM_U', "f4", "", 'OCEAN X CURRENT', 'm s-1', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_v': WrfOutputField('OM_V', "f4", "", 'OCEAN Y CURRENT', 'm s-1', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_depth': WrfOutputField('OM_DEPTH', "f4", "", 'OCEAN LAYER BOTTOM DEPTH', 'm', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_density': WrfOutputField('OM_DENSITY', "f4", "", 'OCEAN DENSITY', 'kg m-3', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_tini': WrfOutputField('OM_TINI', "f4", "", 'INITIAL OCEAN TEMPERATURE', 'K', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_sini': WrfOutputField('OM_SINI', "f4", "", 'INITIAL OCEAN SALINITY', 'PSU', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
    'om_ml': WrfOutputField('OM_ML', "f4", "", 'OCEAN MIXED LAYER DEPTH', 'm', "WRF v4.6.1 module_sf_3dpwp.F", wrf_history=True),
}
OCEAN_DIAGNOSTIC_FIELDS = {
    'ocean_sst': WrfOutputField('OCEAN_SST', "f4", "", 'PROGNOSTIC OCEAN SURFACE TEMPERATURE; LAND AND LAKES MISSING', 'K', "coupled ocean diagnostics", wrf_history=True),
    'ocean_sst_change': WrfOutputField('OCEAN_SST_CHANGE', "f4", "", 'OCEAN SURFACE TEMPERATURE CHANGE FROM COUPLED INITIAL STATE', 'K', "coupled ocean diagnostics", wrf_history=True),
}
SCHEME_OUTPUT_FIELDS: dict[str, WrfOutputField] = {}
from woof.io.wrf_input_history_schema import (WRF_INPUT_HISTORY_ROWS,
                                              WRF_INITIALIZED_HISTORY_ROWS)
WRF_INPUT_ECHO_OUTPUT_FIELDS = {
    name: WrfOutputField(name, row["dtype"], row["stagger"], row["description"],
                         row["units"], row["registry"], wrf_history=True)
    for name, row in (WRF_INPUT_HISTORY_ROWS | WRF_INITIALIZED_HISTORY_ROWS).items()
}
CUMULUS_RATE_OUTPUT_FIELDS = {}
for _offset, (_key, _description, _units) in enumerate((
        ("rucuten", "X WIND", "m s-2"), ("rvcuten", "Y WIND", "m s-2"),
        ("rthcuten", "THETA", "K s-1"), ("rqvcuten", "Q_V", "kg kg-1 s-1"),
        ("rqrcuten", "Q_R", "kg kg-1 s-1"), ("rqccuten", "Q_C", "kg kg-1 s-1"),
        ("rqscuten", "Q_S", "kg kg-1 s-1"), ("rqicuten", "Q_I", "kg kg-1 s-1"))):
    CUMULUS_RATE_OUTPUT_FIELDS[_key] = WrfOutputField(
        _key.upper(), "f4", "", _description + " TENDENCY DUE TO CUMULUS PARAMETERIZATION",
        _units, f"Registry.EM_COMMON:{1565 + _offset}", wrf_history=False)
RADIATION_BOUNDARY_OUTPUT_FIELDS = {}
for _side,_wave,_first in (("sw","SHORTWAVE",1813),("lw","LONGWAVE",1825)):
    for _offset,(_suffix,_direction,_edge) in enumerate((
            ("upt","UPWELLING","TOP"),("uptc","UPWELLING CLEAR SKY","TOP"),
            ("dnt","DOWNWELLING","TOP"),("dntc","DOWNWELLING CLEAR SKY","TOP"),
            ("upb","UPWELLING","BOTTOM"),("upbc","UPWELLING CLEAR SKY","BOTTOM"),
            ("dnb","DOWNWELLING","BOTTOM"),("dnbc","DOWNWELLING CLEAR SKY","BOTTOM"))):
        _key=_side+_suffix
        _line=_first+_offset+_offset//2
        RADIATION_BOUNDARY_OUTPUT_FIELDS[_key]=WrfOutputField(
            _key.upper(),"f4","",f"INSTANTANEOUS {_direction} {_wave} FLUX AT {_edge}",
            "W m-2",f"Registry.EM_COMMON:{_line}",wrf_history=True)
        _acline=(1781 if _side=="sw" else 1789)+_offset
        RADIATION_BOUNDARY_OUTPUT_FIELDS["ac"+_key]=WrfOutputField(
            "AC"+_key.upper(),"f4","",f"ACCUMULATED {_direction} {_wave} FLUX AT {_edge}",
            "J m-2",f"Registry.EM_COMMON:{_acline}",wrf_history=True)

PHYSICS_PRESSURE_OUTPUT_FIELDS = {
    "p_hyd": _output_field(
                 'P_HYD', 'hydrostatic pressure',
                 'Pa', 'Registry.EM_COMMON:376'),
}
SURFACE_ENERGY_OUTPUT_FIELDS = {
    "achfx": _output_field(
                 'ACHFX', 'ACCUMULATED UPWARD HEAT FLUX AT THE SURFACE',
                 'J m-2', 'Registry.EM_COMMON:1964'),
    "aclhf": _output_field(
                 'ACLHF', 'ACCUMULATED UPWARD LATENT HEAT FLUX AT THE SURFACE',
                 'J m-2', 'Registry.EM_COMMON:1967'),
    "acgrdflx": _output_field(
                    'ACGRDFLX', 'ACCUMULATED GROUND HEAT FLUX',
                    'J m-2', 'Registry.EM_COMMON:862'),
}
PBL_RATE_OUTPUT_FIELDS = {
    "rublten": WrfOutputField("RUBLTEN","f4","",
        "X WIND TENDENCY DUE TO PBL PARAMETERIZATION","m s-2","Registry.EM_COMMON:1855",False),
    "rvblten": WrfOutputField("RVBLTEN","f4","",
        "Y WIND TENDENCY DUE TO PBL PARAMETERIZATION","m s-2","Registry.EM_COMMON:1856",False),
    "rthblten": WrfOutputField("RTHBLTEN","f4","",
        "THETA TENDENCY DUE TO PBL PARAMETERIZATION","K s-1","Registry.EM_COMMON:1857",False),
    "rqvblten": WrfOutputField("RQVBLTEN","f4","",
        "Q_V TENDENCY DUE TO PBL PARAMETERIZATION","kg kg-1 s-1","Registry.EM_COMMON:1858",False),
    "rqcblten": WrfOutputField("RQCBLTEN","f4","",
        "Q_C TENDENCY DUE TO PBL PARAMETERIZATION","kg kg-1 s-1","Registry.EM_COMMON:1859",False),
    "rqiblten": WrfOutputField("RQIBLTEN","f4","",
        "Q_I TENDENCY DUE TO PBL PARAMETERIZATION","kg kg-1 s-1","Registry.EM_COMMON:1860",False),
}
SCM_FORCING_OUTPUT_FIELDS = {}
for _offset, _stem in enumerate(("hfx","lh","tsk")):
    _label={"hfx":"sensible heat flux","lh":"latent heat flux",
            "tsk":"skin temperature"}[_stem]
    for _tendency in (False,True):
        _key=_stem+"_force"+("_tend" if _tendency else "")
        SCM_FORCING_OUTPUT_FIELDS[_key]=WrfOutputField(
            _key.upper(),"f4","","SCM ideal surface "+_label+(" tendency" if _tendency else ""),
            "W m-2 s-1" if _tendency else "W m-2",
            f"Registry.EM_COMMON:{278+_offset+(3 if _tendency else 0)}",True)
INACTIVE_DIAGNOSTIC_OUTPUT_FIELDS = {
    "sstsk": _output_field(
                 'SSTSK', 'SKIN SEA SURFACE TEMPERATURE',
                 'K', 'Registry.EM_COMMON:880'),
    "refd_max": _output_field(
                    'REFD_MAX', 'MAX DERIVED RADAR REFL',
                    'dbZ', 'Registry.EM_COMMON:2082'),
}
NOAH_OUTPUT_FIELDS = {
    "smcrel": WrfOutputField(
        "SMCREL", "f4", "Z", "RELATIVE SOIL MOISTURE", "",
        "Registry.EM_COMMON:841", wrf_history=True),
    "sfcrunoff": _output_field(
                     'SFROFF', 'SURFACE RUNOFF',
                     'mm', 'Registry.EM_COMMON:855'),
    "udrunoff": _output_field(
                    'UDROFF', 'UNDERGROUND RUNOFF',
                    'mm', 'Registry.EM_COMMON:856'),
    "acsnom": _output_field(
                  'ACSNOM', 'ACCUMULATED MELTED SNOW',
                  'kg m-2', 'Registry.EM_COMMON:867'),
    "noahres": _output_field(
                   'NOAHRES', 'RESIDUAL OF THE NOAH SURFACE ENERGY BUDGET',
                   'W m{-2}', 'Registry.EM_COMMON:1851'),
}
CARRIED_SURFACE_OUTPUT_FIELDS = {
    "sst": _output_field(
               'SST', 'SEA SURFACE TEMPERATURE',
               'K', 'Registry.EM:15'),
    "xicem": _output_field(
                 'XICEM', 'SEA ICE FLAG (PREVIOUS STEP)',
                 '', 'Registry.EM_COMMON:844'),
    "coszen": _output_field(
                  'COSZEN', 'COS of SOLAR ZENITH ANGLE',
                  'dimensionless', 'Registry.EM_COMMON:997'),
    "lai": _output_field(
               'LAI', 'LEAF AREA INDEX',
               'm-2/m-2', 'Registry.EM_COMMON:1022'),
    "albedo": _output_field(
                  'ALBEDO', 'ALBEDO',
                  '', 'Registry.EM_COMMON:1845'),
    "albbck": _output_field(
                  'ALBBCK', 'BACKGROUND ALBEDO',
                  '', 'Registry.EM_COMMON:1847'),
    "emiss": _output_field(
                 'EMISS', 'SURFACE EMISSIVITY',
                 '', 'Registry.EM_COMMON:1849'),
    "xland": _output_field(
                 'XLAND', 'LAND MASK (1 FOR LAND, 2 FOR WATER)',
                 '', 'Registry.EM_COMMON:1946'),
}
SOLAR_ALBEDO_OUTPUT_FIELDS = {
    "albsol": _output_field(
                  'ALBSOL', 'albedo corrected for solar angle',
                  '', 'Registry.EM_COMMON:1586 (HRRR v4.1.21)'),
    "albbcksol": _output_field(
                     'ALBBCKSOL', 'BACKGROUND ALBEDO corrected for solar angle',
                     '', 'Registry.EM_COMMON:1587 (HRRR v4.1.21)'),
}
for _group in (CARRIED_SURFACE_OUTPUT_FIELDS, NOAH_OUTPUT_FIELDS, RADIATION_BOUNDARY_OUTPUT_FIELDS,
               CUMULUS_RATE_OUTPUT_FIELDS,
               PHYSICS_PRESSURE_OUTPUT_FIELDS,
               SURFACE_ENERGY_OUTPUT_FIELDS,
               PBL_RATE_OUTPUT_FIELDS,
               SCM_FORCING_OUTPUT_FIELDS,
               INACTIVE_DIAGNOSTIC_OUTPUT_FIELDS,
               MYNN_PBL_OUTPUT_FIELDS, NOAHMP_OUTPUT_FIELDS,
               RUC_OUTPUT_FIELDS, SOLAR_ALBEDO_OUTPUT_FIELDS,
               OCEAN_OUTPUT_FIELDS, PWP_OUTPUT_FIELDS, OCEAN_DIAGNOSTIC_FIELDS):
    for _key, _field in _group.items():
        _clash = SCHEME_OUTPUT_FIELDS.get(_key)
        if _clash is not None and _clash != _field:
            raise ValueError(
                "two schemes declare different output schemas for field "
                f"{_key!r}: {_clash} and {_field}")
        SCHEME_OUTPUT_FIELDS[_key] = _field

#: Every SCHEME-SELECTOR record this module carries, keyed by the name that
#: reaches the NetCDF file: the scheme fields plus the always-written
#: accumulators.  Two distinct woof fields sharing one external name would
#: make the wrfout inventory depend on dict order, so the reverse map
#: refuses it.
#:
#: This map is deliberately the SELECTOR-DRIVEN inventory only, and its
#: cardinality is pinned to exactly ``len(SCHEME_OUTPUT_FIELDS) +
#: len(PRECIPITATION_OUTPUT_FIELDS)`` by
#: ``tests/test_wrf_output_schema.py::
#: test_no_two_gpuwm_fields_claim_one_netcdf_name``.  Model-state rows --
#: which is what mp=28's four aerosol carriers are -- go in
#: :data:`HISTORY_FIELDS_BY_NETCDF_NAME` below instead of here, so that pin
#: keeps meaning what it says.  The writer consults the wider map.
# WRF restart-only diagnostics are published as an explicit history extension.
for _key, _description, _units, _line in (
    ("gd_cloud", "CLOUD WATER/ICE MIXING RAIO IN GD CLOUD", "kg kg-1", 1653),
    ("gd_cloud2", "TEST for GD CLOUD", "kg kg-1", 1654),
    ("gd_cldfr", "GD CLOUD Fraction", " ? ", 1655),
    ("raincv_a", "taveragd TIME-STEP CUMULUS PRECIPITATION", "mm", 1656),
    ("raincv_b", "taveragd TIME-STEP CUMULUS PRECIPITATION", "mm", 1657),
    ("gd_cloud_a", "taveragd CLOUD WATER MIXING RAIO IN GD CLOUD", "kg kg-1", 1658),
    ("gd_cloud2_a", "taveragd cloud ice mix ratio in GD", "kg kg-1", 1659),
    ("qc_cu", "CLOUD WATER MIXING RATIO FROM A CU SCHEME", "kg kg-1", 1660),
    ("qi_cu", "CLOUD ICE MIXUNG RATIO FROM A CU SCHEME", "kg kg-1", 1661),
):
    SCHEME_OUTPUT_FIELDS[_key] = WrfOutputField(_key.upper(), "f4", "", _description,
        _units, f"Registry.EM_COMMON:{_line}", False)


OUTPUT_FIELDS_BY_NETCDF_NAME: dict[str, WrfOutputField] = {}
for _group in (SCHEME_OUTPUT_FIELDS, PRECIPITATION_OUTPUT_FIELDS):
    for _key, _field in _group.items():
        _clash = OUTPUT_FIELDS_BY_NETCDF_NAME.get(_field.netcdf_name)
        if _clash is not None and _clash != _field:
            raise ValueError(
                "two woof fields both write NetCDF variable "
                f"{_field.netcdf_name!r}: {_clash} and {_field}")
        OUTPUT_FIELDS_BY_NETCDF_NAME[_field.netcdf_name] = _field

#: EVERY record this module carries, keyed by its NetCDF name: the
#: selector-driven inventory above plus the transcribed model-state rows
#: that belong to no physics-driver ``fields`` dict.  THIS is what
#: ``woof.io.wrfout`` consults, so a name described here is described the
#: same way whichever table it came from, and the same clash refusal covers
#: the union -- an aerosol row that collided with a scheme diagnostic would
#: raise at import, not silently win or lose a dict-order race.
#:
#: Why two maps instead of one: ``OUTPUT_FIELDS_BY_NETCDF_NAME`` answers
#: "which fields does a selected SCHEME publish", which is the question the
#: driver-inventory tests ask and count.  ``HISTORY_FIELDS_BY_NETCDF_NAME``
#: answers "what does the writer know about this variable name".  mp=28's
#: aerosol carriers are transported model state (WRF ``scalar``-array
#: members, plus two ``misc`` 2-D surface fields), not scheme diagnostics,
#: so only the second question includes them.
HISTORY_FIELDS_BY_NETCDF_NAME: dict[str, WrfOutputField] = {}
from woof.io.sfire_schema import SFIRE_REGISTRY_FIELDS
SFIRE_OUTPUT_FIELDS = {
    name: WrfOutputField(name, NETCDF_DTYPE_BY_REGISTRY_TYPE[row[0]], row[1],
                         row[2], row[3], row[4], wrf_history=row[5])
    for name, row in SFIRE_REGISTRY_FIELDS.items()
}
for _group in (OUTPUT_FIELDS_BY_NETCDF_NAME, THOMPSON_AEROSOL_OUTPUT_FIELDS,
               WRF_INPUT_ECHO_OUTPUT_FIELDS,
               SURFACE_IDENTITY_OUTPUT_FIELDS, SFIRE_OUTPUT_FIELDS):
    for _key, _field in _group.items():
        _clash = HISTORY_FIELDS_BY_NETCDF_NAME.get(_field.netcdf_name)
        if _clash is not None and _clash != _field:
            raise ValueError(
                "two woof fields both write NetCDF variable "
                f"{_field.netcdf_name!r}: {_clash} and {_field}")
        HISTORY_FIELDS_BY_NETCDF_NAME[_field.netcdf_name] = _field

del _group, _key, _field, _clash


def netcdf_names(keys) -> tuple[str, ...]:
    """External WRF names for ``keys``, in order; unknown keys fail loud."""
    return tuple(SCHEME_OUTPUT_FIELDS[key].netcdf_name for key in keys)


#: WRF Registry metadata ``name -> (description, units)`` for the
#: variables woof writes; unknown names get empty strings.
#:
#: This table lives HERE, in the data module, rather than beside the
#: writer that reads it.  ``woof.io.history_selection`` needs the NAMES
#: at config-resolution time to build ``HISTORY_VOCABULARY``, and while
#: the table sat in ``woof.io.wrfout`` it got them by importing the
#: writer -- which imports ``woof.supervisor``, ``netCDF4`` and the
#: runtime behind it.  MEASURED: that made ``import woof.era5_direct``
#: pull the whole forecast output side in through
#: case_data -> experiment -> history_selection -> wrfout, and the
#: RW-WPS preprocessing wheel's no-CuPy/no-executor import proof
#: (tests/test_native_wrf_distribution.py) went red on it.  A metadata
#: table is data; the module that carries it must not drag a writer
#: behind it.
#: Extra attributes on the history CLDFRA variable.  The field is the
#: cloud fraction the model's radiation radiated through, computed on
#: radiation's cadence (radt), which is coarser than the history cadence:
#: a frame between radiation calls carries the most recent call's field,
#: and the initial frame, written before the first radiation call,
#: carries zeros (as WRF's own t=0 wrfout does).  A reader that needs the
#: fraction AT the initial time diagnoses it instead.
RADIATION_CLDFRA_ATTRIBUTES: dict[str, str] = {
    "source": ("radiation cloud fraction, 0 to 1: icloud=1 cal_cldfra1, "
               "with the MYNN subgrid cloud merged where icloud_bl=1"),
    "sampling": ("held between radiation calls: each frame carries the "
                 "field from the most recent radiation call at or before "
                 "its valid time (radiation runs every radt minutes)"),
    "initial_frame": ("zero before the first radiation call; the "
                      "initial-time frame is written before it"),
}

REGISTRY_VAR_META: dict[str, tuple[str, str]] = {
    "POWER": ("WIND TURBINE POWER PER GRID CELL", "W"),
    "WIND_TKE_SOURCE": ("TURBINE ADDED TKE RATE", "m2 s-3"),
    "WIND_TKE_SOURCE_MAX": ("VERTICAL MAXIMUM TURBINE ADDED TKE RATE", "m2 s-3"),
    "TURBINE_POWER": ("POWER PER TURBINE; ordered by wind-farm receipt turbine_ids", "W"),
    "TURBINE_HUB_WIND": ("HUB HEIGHT WIND SPEED PER TURBINE", "m s-1"),
    "TURBINE_FARM_INDEX": ("ONE BASED FARM INDEX; ordered by wind-farm receipt farm_ids", "1"),
    "TURBINE_HUB_HEIGHT": ("TURBINE HUB HEIGHT ABOVE GROUND", "m"),
    "FARM_POWER": ("SUM OF TURBINE POWER PER FARM", "W"),
    "FARM_HUB_WIND": ("ARITHMETIC MEAN HUB HEIGHT WIND PER FARM", "m s-1"),
    "IC_FLASHCOUNT": ("Accumulated IC flash count", "#"),
    "CG_FLASHCOUNT": ("Accumulated CG flash count", "#"),
    "IC_FLASHRATE": ("IC flash rate", "#/s"),
    "CG_FLASHRATE": ("CG flash rate", "#/s"),
    "LPI": ("Lightning Potential Index", "m^2 s-2"),
    # Correct mode, finding WB1 (woof/core/thompson_cloud_booking.py): no
    # WRF Registry row (WRF drops the flux); the MPAS-A port of the same fix
    # books it as cloudnc/cloudncv.  Present only in a correct-mode
    # Thompson run.
    "CLOUDNC": ("ACCUMULATED CLOUD WATER SETTLED TO THE SURFACE", "mm"),
    "CLOUDNCV": ("CLOUD WATER SETTLED TO THE SURFACE IN THE LAST CALL", "mm"),
    "U": ("x-wind component", "m s-1"),
    "V": ("y-wind component", "m s-1"),
    "W": ("z-wind component", "m s-1"),
    "T": ("perturbation potential temperature theta-t0", "K"),
    "P": ("perturbation pressure", "Pa"),
    "PB": ("BASE STATE PRESSURE", "Pa"),
    "PH": ("perturbation geopotential", "m2 s-2"),
    "PHB": ("base-state geopotential", "m2 s-2"),
    "MU": ("perturbation dry air mass in column", "Pa"),
    "MUB": ("base state dry air mass in column", "Pa"),
    "QVAPOR": ("Water vapor mixing ratio", "kg kg-1"),
    "QCLOUD": ("Cloud water mixing ratio", "kg kg-1"),
    "QRAIN": ("Rain water mixing ratio", "kg kg-1"),
    "QICE": ("Ice mixing ratio", "kg kg-1"),
    "QSNOW": ("Snow mixing ratio", "kg kg-1"),
    "QGRAUP": ("Graupel mixing ratio", "kg kg-1"),
    "QHAIL": ("Hail mixing ratio", "kg kg-1"),
    "QNDROP": ("Droplet number mixing ratio", "# kg-1"),
    "QNCLOUD": ("cloud water Number concentration", "  kg(-1)"),
    "QNRAIN": ("Rain Number concentration", "  kg(-1)"),
    "QNICE": ("Ice Number concentration", "  kg-1"),
    "QNSNOW": ("Snow Number concentration", "  kg(-1)"),
    "QNGRAUPEL": ("Graupel Number concentration", "  kg(-1)"),
    "QNHAIL": ("Hail Number concentration", "# kg(-1)"),
    "QNCCN": ("CCN Number concentration", "# kg(-1)"),
    # QNWFA/QNIFA/QNWFA2D/QNIFA2D (mp_physics=28) are deliberately NOT here.
    # They have transcribed rows -- with NetCDF type, FieldType and the
    # stagger the writer cross-checks -- in
    # woof.io.wrf_output_schema.THOMPSON_AEROSOL_OUTPUT_FIELDS, which
    # _create_variable consults BEFORE this fallback table.  Same reason the
    # precipitation accumulators moved out below: two tables describing one
    # field is how two tables come to disagree about one field.
    "QVGRAUPEL": ("Graupel Particle Volume", "m(3) kg(-1)"),
    "QVHAIL": ("Hail Particle Volume", "m(3) kg(-1)"),
    # P3 one-category (mp_physics=50).  Transcribed verbatim from
    # Registry.EM_COMMON:555-558, whose IO string ``i0rhusdf`` carries the
    # ``h`` that puts both in WRF's history stream.  They are what makes an
    # mp=50 wrfout more than a one-moment ice field: without the rime pair a
    # reader cannot recover rime fraction qir/qi or rime density qir/qib,
    # which are the two indices P3's whole ice inventory is built on.
    "QIR": ("Rime ice mass-1 mixing ratio", "kg kg(-1)"),
    "QIB": ("Rime ice volume-1 mixing ratio", "m(3) kg(-1)"),
    # Registry.EM_COMMON:1596 (verified against the reference wrfout's
    # REFL_10CM attributes).  Opt-in: cases merge the field into their
    # frame dict (woof.core.refl.compute_refl_10cm supplies the values).
    "REFL_10CM": ("Radar reflectivity (lamda = 10 cm)", "dBZ"),
    # Registry.EM_COMMON:2083 (nwp_output package, IO "rh02").  Present
    # exactly when the run carries the accumulator (nwp_diagnostics = 1;
    # woof.core.uh_diag owns the per-step update and the post-write reset).
    "UP_HELI_MAX": ("MAX UPDRAFT HELICITY", "m2 s-2"),
    "HGT": ("Terrain Height", "m"),
    "PSFC": ("SFC PRESSURE", "Pa"),
    # RAINC/RAINSH/RAINNC/SNOWNC/GRAUPELNC/HAILNC used to live here.  They
    # are now in woof.io.wrf_output_schema.PRECIPITATION_OUTPUT_FIELDS,
    # because they are no longer merely *described* by this table -- they are
    # emitted unconditionally, and the thing that decides that has to be the
    # same thing that carries their Registry metadata.  Two tables describing
    # one field is how two tables come to disagree about one field.
    "SNOW": ("SNOW WATER EQUIVALENT", "kg m-2"),
    "SNOWH": ("PHYSICAL SNOW DEPTH", "m"),
    "SNOWC": ("FLAG INDICATING SNOW COVERAGE (1 FOR SNOW COVER)", ""),
    "TSLB": ("SOIL TEMPERATURE", "K"),
    "SMOIS": ("SOIL MOISTURE", "m3 m-3"),
    "SH2O": ("SOIL LIQUID WATER", "m3 m-3"),
    # WRF v4.7.1 Registry.EM_COMMON CANWAT has the history flag h.
    "CANWAT": ("CANOPY WATER", "kg m-2"),
    "SWDOWN": ("DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE", "W m-2"),
    "ECOBSC": ("SOLAR ECLIPSE OBSCURATION FRACTION", "1"),
    "ECMASK": ("ECLIPSE MASK: 0 NONE, 1 PARTIAL, 2 TOTAL, 3 ANNULAR", "1"),
    "GLW": ("DOWNWARD LONG WAVE FLUX AT GROUND SURFACE", "W m-2"),
    # Registry.EM_COMMON:1839, transcribed verbatim beside its two
    # surface-flux neighbours.  ``ij`` mass grid, no stagger, real -- so
    # the dimension table and the FieldType default already describe it
    # correctly and it needs no row of its own.
    "OLR": ("TOA OUTGOING LONG WAVE", "W m-2"),
    'SWDDNI': ('Shortwave surface downward direct normal irradiance', 'W m-2'),
    'SWDDIR': ('Shortwave surface downward direct irradiance', 'W m-2'),
    'SWDDIF': ('Shortwave surface downward diffuse irradiance', 'W m-2'),
    'TLWDN': ('TOA LW downwelling flux', 'W m-2'),
    'TLWUP': ('TOA LW upwelling flux', 'W m-2'),
    'SLWDN': ('Surface LW downwelling flux', 'W m-2'),
    'SLWUP': ('Surface LW upwelling flux', 'W m-2'),
    'TSWDN': ('TOA SW downwelling flux', 'W m-2'),
    'TSWUP': ('TOA SW upwelling flux', 'W m-2'),
    'SSWDN': ('Surface SW downwelling flux', 'W m-2'),
    'SSWUP': ('Surface SW upwelling flux', 'W m-2'),
    "RTHRATEN": ("THETA TENDENCY DUE TO RADIATION", "K s-1"),
    "SR": ("fraction of frozen precipitation", ""),
    "THM": ("perturbation dry potential temperature", "K"),
    "PC": ("perturbation dry air mass in column", "Pa"),
    "O3_GFS_DU": ("Total ozone from GFS", "Dobson Units"),
    # Registry.EM_COMMON:1699 (``ikj``, ``irh``), verbatim: mass grid, no
    # stagger, real, unitless.  Written where a radiation scheme that
    # computes a cloud fraction runs; see RADIATION_CLDFRA_ATTRIBUTES.
    "CLDFRA": ("CLOUD FRACTION", ""),
    "AOD5502D": ("Column aerosol optical depth at 550 nm", ""),
    "AOD5503D": ("Aerosol optical depth in RRTMG band 10", ""),
    "ANGEXP2D": ("Aerosol Angstrom exponent", ""),
    "AERSSA2D": ("Aerosol single scattering albedo", ""),
    "AERASY2D": ("Aerosol asymmetry parameter", ""),
    # Registry.EM_COMMON:1715, written where slope_rad = 1 runs.
    "SWNORM": ("NORMAL SHORT WAVE FLUX AT GROUND SURFACE (SLOPE-DEPENDENT)",
               "W m-2"),
    "TSK": ("SURFACE SKIN TEMPERATURE", "K"),
    "T2": ("TEMP at 2 M", "K"),
    "TH2": ("POT TEMP at 2 M", "K"),
    "Q2": ("QV at 2 M", "kg kg-1"),
    "U10": ("U at 10 M", "m s-1"),
    "V10": ("V at 10 M", "m s-1"),
    "UST": ("U* IN SIMILARITY THEORY", "m s-1"),
    "HFX": ("UPWARD HEAT FLUX AT THE SURFACE", "W m-2"),
    "QFX": ("UPWARD MOISTURE FLUX AT THE SURFACE", "kg m-2 s-1"),
    "LH": ("LATENT HEAT FLUX AT THE SURFACE", "W m-2"),
    "ALBEDO": ("ALBEDO", ""),
    "EMISS": ("SURFACE EMISSIVITY", ""),
    "GSW": ("NET SHORT WAVE FLUX AT GROUND SURFACE", "W m-2"),
    "AODTOT": ("TOTAL AEROSOL OPTICAL DEPTH AT 550 NM", "1"),
    "FRC_URB2D": ("URBAN FRACTION", "dimensionless"),
    "SH_URB": ("SENSIBLE HEAT FLUX FROM URBAN SFC", "W m{-2}"),
    "LH_URB": ("LATENT HEAT FLUX FROM URBAN SFC", "W m{-2}"),
    "G_URB": ("GROUND HEAT FLUX INTO URBAN", "W m{-2}"),
    "RN_URB": ("NET RADIATION ON URBAN SFC", "W m{-2}"),
    "TS_URB": ("SKIN TEMPERATURE", "K"),
    "TR_URB": ("ROOF SURFACE TEMPERATURE", "K"),
    "TB_URB": ("WALL SURFACE TEMPERATURE", "K"),
    "TG_URB": ("ROAD SURFACE TEMPERATURE", "K"),
    "TC_URB": ("CANOPY AIR TEMPERATURE", "K"),
    "TVG_URB": ("VEGETATED GROUND SURFACE TEMPERATURE", "K"),
    "TT_URB": ("TREE TEMPERATURE", "K"),
    "XXXVG_URB": ("VEGETATED GROUND STABILITY", "dimensionless"),
    "TVGL_URB": ("VEGETATED GROUND LAYER TEMPERATURE", "K"),
    "SMG_URB": ("VEGETATED GROUND SOIL MOISTURE", "dimensionless"),
    "CMCG_URB": ("VEGETATED GROUND CANOPY INTERCEPTION", "m"),
    "FLXHUMVG_URB": ("VEGETATED GROUND HUMIDITY FLUX", "m s-1"),
    "FLXHUMT_URB": ("TREE HUMIDITY FLUX", "m s-1"),
    "TKE_SASE": ("SASE prognostic subgrid turbulence kinetic energy",
                 "m2 s-2"),
    "TKE_SHINHONG": ("Shin-Hong published subgrid turbulence kinetic "
                     "energy diagnostic", "m2 s-2"),
    # The SPLIT SUBGRID-FLUX DIAGNOSTIC (RunConfig sase_flux_diag, off by
    # default; per domain).  Four z-FACE fields on the mass column --
    # (nz+1, ny, nx), stagger "Z", registered exactly like W -- carrying
    # the closure's own vertical subgrid fluxes with the conditional-
    # venting channel separated from the K_v implicit vertical diffusion
    # channel.  BOTH CHANNELS ARE POSITIVE UPWARD and both divide by the
    # SAME lowest-level moist density the venting deposit divides by, so
    # VENT + DIFF is the closure's total subgrid flux and its
    # convergence is the model's own scalar increment.  Face 0 is +0.0
    # by the interface contract: the surface flux is owned by HFX/QFX
    # and is not double-counted here.
    "SASE_FQV_VENT": ("SASE vent subgrid water vapor flux, positive up",
                      "kg m-2 s-1"),
    "SASE_FQV_DIFF": ("SASE Kv subgrid water vapor flux, positive up",
                      "kg m-2 s-1"),
    "SASE_FTH_VENT": ("SASE vent subgrid heat flux, positive up",
                      "W m-2"),
    "SASE_FTH_DIFF": ("SASE Kv subgrid heat flux, positive up",
                      "W m-2"),
    # The HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (RunConfig hmix_k_diag,
    # off by default; per domain).  Two (nz, ny, nx) mass-grid fields
    # carrying the horizontal mixing coefficients the run's own producer
    # used, under that producer's name: XKMH/XKHH are WRF's Registry
    # names for the km_opt = 4 2-D Smagorinsky viscosities, SASE_KMH/
    # SASE_KHH the SASE closure's governed horizontal diffusivity and
    # the K_h = KMH/Pr_t(f) its scalar channel rides.  Same units, same
    # grid, same meaning, so the two are directly comparable -- which is
    # the point: SASE's claim to replace the km_opt operator is a claim
    # about this number.  A run with NO horizontal mixing producer (the
    # acknowledged km_opt = 0 control) publishes NEITHER pair, so an
    # absent variable says "no operator ran" and can never be misread as
    # a measured zero.  Instantaneous: the value the last completed
    # model step used, not a history-interval mean.
    "XKMH": ("HORIZONTAL MOMENTUM EDDY VISCOSITY", "m2 s-1"),
    "XKHH": ("HORIZONTAL HEAT EDDY VISCOSITY", "m2 s-1"),
    "SASE_KMH": ("SASE governed horizontal momentum diffusivity",
                 "m2 s-1"),
    "SASE_KHH": ("SASE governed horizontal scalar diffusivity, KMH/Pr_t",
                 "m2 s-1"),
    "PBLH": ("PBL HEIGHT", "m"),
    "GRDFLX": ("GROUND HEAT FLUX", "W m-2"),
    "GSW": ("NET SHORTWAVE FLUX AT GROUND", "W m-2"),
    "PSIM": ("SIMILARITY STABILITY FUNCTION FOR MOMENTUM", ""),
    "PSIH": ("SIMILARITY STABILITY FUNCTION FOR HEAT", ""),
    "XLAT": ("LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
    "XLONG": ("LONGITUDE, WEST IS NEGATIVE", "degree_east"),
    "XLAT_U": ("LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
    "XLONG_U": ("LONGITUDE, WEST IS NEGATIVE", "degree_east"),
    "XLAT_V": ("LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
    "XLONG_V": ("LONGITUDE, WEST IS NEGATIVE", "degree_east"),
    "MAPFAC_M": ("Map scale factor on mass grid", ""),
    "MAPFAC_U": ("Map scale factor on u-grid", ""),
    "MAPFAC_V": ("Map scale factor on v-grid", ""),
    "F": ("Coriolis sine latitude term", "s-1"),
    "E": ("Coriolis cosine latitude term", "s-1"),
    "LU_INDEX": ("LAND USE CATEGORY", ""),
    "LANDMASK": ("LAND MASK (1 FOR LAND, 0 FOR WATER)", ""),
    "SINALPHA": ("Local sine of map rotation", ""),
    "COSALPHA": ("Local cosine of map rotation", ""),
    "P_TOP": ("PRESSURE TOP OF THE MODEL", "Pa"),
    "ZNU": ("eta values on half (mass) levels", ""),
    "ZNW": ("eta values on full (w) levels", ""),
    # Static grid, coordinate and surface words wrf.exe writes to every
    # history frame (WRF v4.6.1 wrfout attributes, verbatim).  The WRF-file
    # doors write the input file's own words for the static rows
    # (woof.io.history_layout.WRF_FILE_STATIC_HISTORY_FIELDS); XLAND and
    # LAKEMASK are the live surface masks.
    "ZS": ('DEPTHS OF CENTERS OF SOIL LAYERS', 'm'),
    "DZS": ('THICKNESSES OF SOIL LAYERS', 'm'),
    "FNM": ('upper weight for vertical stretching', ''),
    "FNP": ('lower weight for vertical stretching', ''),
    "RDNW": ('inverse d(eta) values between full (w) levels', ''),
    "RDN": ('inverse d(eta) values between half (mass) levels', ''),
    "DNW": ('d(eta) values between full (w) levels', ''),
    "DN": ('d(eta) values between half (mass) levels', ''),
    "CFN": ('extrapolation constant', ''),
    "CFN1": ('extrapolation constant', ''),
    "CF1": ('2nd order extrapolation constant', ''),
    "CF2": ('2nd order extrapolation constant', ''),
    "CF3": ('2nd order extrapolation constant', ''),
    "C1H": ('half levels, c1h = d bf / d eta, using znw', 'Dimensionless'),
    "C2H": ('half levels, c2h = (1-c1h)*(p0-pt)', 'Pa'),
    "C1F": ('full levels, c1f = d bf / d eta, using znu', 'Dimensionless'),
    "C2F": ('full levels, c2f = (1-c1f)*(p0-pt)', 'Pa'),
    "C3H": ('half levels, c3h = bh', 'Dimensionless'),
    "C4H": ('half levels, c4h = (eta-bh)*(p0-pt), using znu', 'Pa'),
    "C3F": ('full levels, c3f = bf', 'Dimensionless'),
    "C4F": ('full levels, c4f = (eta-bf)*(p0-pt), using znw', 'Pa'),
    "RDX": ('INVERSE X GRID LENGTH', 'm-1'),
    "RDY": ('INVERSE Y GRID LENGTH', 'm-1'),
    "RESM": ('TIME WEIGHT CONSTANT FOR SMALL STEPS', ''),
    "ZETATOP": ('ZETA AT MODEL TOP', ''),
    "T00": ('BASE STATE TEMPERATURE', 'K'),
    "P00": ('BASE STATE PRESSURE', 'Pa'),
    "TLP": ('BASE STATE LAPSE RATE', ''),
    "TISO": ('TEMP AT WHICH THE BASE T TURNS CONST', 'K'),
    "TLP_STRAT": ('BASE STATE LAPSE RATE (DT/D(LN(P)) IN STRATOSPHERE', 'K'),
    "P_STRAT": ('BASE STATE PRESSURE AT BOTTOM OF STRATOSPHERE', 'Pa'),
    "MAPFAC_MX": ('Map scale factor on mass grid, x direction', ''),
    "MAPFAC_MY": ('Map scale factor on mass grid, y direction', ''),
    "MAPFAC_UX": ('Map scale factor on u-grid, x direction', ''),
    "MAPFAC_UY": ('Map scale factor on u-grid, y direction', ''),
    "MAPFAC_VX": ('Map scale factor on v-grid, x direction', ''),
    "MAPFAC_VY": ('Map scale factor on v-grid, y direction', ''),
    "MF_VX_INV": ('Inverse map scale factor on v-grid, x direction', ''),
    "CLAT": ('COMPUTATIONAL GRID LATITUDE, SOUTH IS NEGATIVE', 'degree_north'),
    "VAR": ('STANDARD DEVIATION OF SUBGRID-SCALE OROGRAPHY', 'm'),
    "VAR_SSO": ('variance of subgrid-scale orography', 'm2'),
    "SHDMAX": ('ANNUAL MAX VEG FRACTION', ''),
    "SHDMIN": ('ANNUAL MIN VEG FRACTION', ''),
    "PCB": ('base state dry air mass in column', 'Pa'),
    "XLAND": ('LAND MASK (1 FOR LAND, 2 FOR WATER)', ''),
    "LAKEMASK": ('LAKE MASK (1 FOR LAKE, 0 FOR NON-LAKE)', ''),
}


from woof.core.output_diagnostics import HISTORY as _OUTPUT_HISTORY
REGISTRY_VAR_META.update({symbol: (description, units)
    for symbol, units, description in _OUTPUT_HISTORY.values()})
from woof.core.hail import HISTORY as _HAIL_HISTORY
REGISTRY_VAR_META.update({symbol: (description, units)
    for symbol, units, description in _HAIL_HISTORY.values()})

__all__ = [
    "HISTORY_FIELDS_BY_NETCDF_NAME",
    "MYNN_PBL_OUTPUT_FIELDS",
    "NETCDF_DTYPE_BY_REGISTRY_TYPE",
    "NOAHMP_OUTPUT_FIELDS",
    "OUTPUT_FIELDS_BY_NETCDF_NAME",
    "PHYSICS_SELECTOR_GLOBALS",
    "PRECIPITATION_OUTPUT_FIELDS",
    "RADIATION_CLDFRA_ATTRIBUTES",
    "REGISTRY_VAR_META",
    "RUC_OUTPUT_FIELDS",
    "SCHEME_OUTPUT_FIELDS",
    "SURFACE_IDENTITY_OUTPUT_FIELDS",
    "THOMPSON_AEROSOL_OUTPUT_FIELDS",
    "WRF_FIELD_TYPE_INTEGER",
    "WRF_FIELD_TYPE_REAL",
    "WrfOutputField",
    "WrfSelectorGlobal",
    "netcdf_names",
]

# Ocean rows append native history metadata without changing atmosphere fields.
from woof.ocean.door import rows as _ocean_rows
REGISTRY_VAR_META.update({row["history_name"]: (row["title"], row.get("storage_units", row["units"]))
                          for row in _ocean_rows()["products"].values()})
# Prognostic sea-ice extensions, no WRF Registry identity is claimed.
from woof.ocean.seaice import OUTPUT as _SEAICE_OUTPUT
for _key, (_external, _units) in _SEAICE_OUTPUT.items():
    _schema = WrfOutputField(_external, "f4", "", _external, _units, "", False)
    SCHEME_OUTPUT_FIELDS[_key] = _schema
    OUTPUT_FIELDS_BY_NETCDF_NAME[_external] = _schema
    HISTORY_FIELDS_BY_NETCDF_NAME[_external] = _schema
