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


EEPS_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "pek_pbl": _output_field("PEK_PBL", "TKE from EEPS", "m2 s-2",
                              "Registry.EM_COMMON:1051"),
    "pep_pbl": _output_field("PEP_PBL", "TKE dissipation rate from EEPS", "m2 s-3",
                              "Registry.EM_COMMON:1052"),
}


#: Eta similarity diagnostics, owned by sf_sfclay_physics=2.
MYJ_SURFACE_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "u10e": WrfOutputField(
        "U10E", "f4", "", "Special U at 10 M from MYJSFC", "m s-1",
        "Registry.EM_COMMON:1069", wrf_history=True),
    "v10e": WrfOutputField(
        "V10E", "f4", "", "Special V at 10 M from MYJSFC", "m s-1",
        "Registry.EM_COMMON:1070", wrf_history=True),
}


#: MYNN's carried PBL state and plume diagnostics (``bl_pbl_physics=5``).
#: ``KPBL`` and ``RMOL`` are shared surface-layer/PBL rows rather than MYNN's
#: own, which is why they cite EM_COMMON blocks far from the MYNN one.
MYNN_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    'qshear': WrfOutputField('QSHEAR', "f4", 'Z', 'TKE Production - shear', 'm2 s-2', "WRF 4.7.1 Registry.EM_COMMON:1120", wrf_history=True),
    'qbuoy': WrfOutputField('QBUOY', "f4", 'Z', 'TKE Production - buoyancy', 'm2 s-2', "WRF 4.7.1 Registry.EM_COMMON:1121", wrf_history=True),
    'qdiss': WrfOutputField('QDISS', "f4", 'Z', 'TKE dissipation', 'm2 s-2', "WRF 4.7.1 Registry.EM_COMMON:1122", wrf_history=True),
    'qwt': WrfOutputField('QWT', "f4", 'Z', 'TKE vertical transport', 'm2 s-2', "WRF 4.7.1 Registry.EM_COMMON:1123", wrf_history=True),
    'dqke': WrfOutputField('DTKE', "f4", '', 'TKE change', 'm2 s-2', "WRF 4.7.1 Registry.EM_COMMON:1124", wrf_history=True),
    'edmf_a': WrfOutputField('EDMF_A', "f4", '', 'EDMF relative updraft  area - moist updrafts', '-', "WRF 4.7.1 Registry.EM_COMMON:1135", wrf_history=True),
    'edmf_w': WrfOutputField('EDMF_W', "f4", '', 'EDMF vertical velocity  - mean moist updrafts', 'm s-1', "WRF 4.7.1 Registry.EM_COMMON:1136", wrf_history=True),
    'edmf_thl': WrfOutputField('EDMF_THL', "f4", '', 'EDMF thetaL  - mean moist updrafts', 'K', "WRF 4.7.1 Registry.EM_COMMON:1137", wrf_history=True),
    'edmf_qt': WrfOutputField('EDMF_QT', "f4", '', 'EDMF qt  - mean moist updrafts', 'kg kg-1', "WRF 4.7.1 Registry.EM_COMMON:1138", wrf_history=True),
    'edmf_ent': WrfOutputField('EDMF_ENT', "f4", '', 'EDMF entrainment  - mean moist updrafts', 'm-1', "WRF 4.7.1 Registry.EM_COMMON:1139", wrf_history=True),
    'edmf_qc': WrfOutputField('EDMF_QC', "f4", '', 'EDMF qc - mean moist updrafts', 'kg kg-1', "WRF 4.7.1 Registry.EM_COMMON:1140", wrf_history=True),
    'sub_thl3d': WrfOutputField('SUB_THL3D', "f4", '', 'thetaL subsidence tendency from EDMF', 'K s-1', "WRF 4.7.1 Registry.EM_COMMON:1141", wrf_history=True),
    'sub_sqv3d': WrfOutputField('SUB_SQV3D', "f4", '', 'qv subsidence tendency from EDMF', 'kg kg-1 s-1', "WRF 4.7.1 Registry.EM_COMMON:1142", wrf_history=True),
    'det_thl3d': WrfOutputField('DET_THL3D', "f4", '', 'thetaL detrainment tendency from EDMF', 'K s-1', "WRF 4.7.1 Registry.EM_COMMON:1143", wrf_history=True),
    'det_sqv3d': WrfOutputField('DET_SQV3D', "f4", '', 'qv detrainment tendency from EDMF', 'kg kg-1 s-1', "WRF 4.7.1 Registry.EM_COMMON:1144", wrf_history=True),
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

MRF_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "kpbl": MYNN_PBL_OUTPUT_FIELDS["kpbl"],
    "hol": WrfOutputField(
        "HOL", "f4", "", "PBL HEIGHT OVER MONIN-OBUKHOV LENGTH", "",
        "Registry.EM_COMMON:1957", wrf_history=False),
}

#: BouLac's carried TKE and the shared PBL rows its history block publishes
#: (``bl_pbl_physics=8``).  ``TKE_PBL`` is the Registry row BouLac's package
#: line allocates (Registry.EM_COMMON:3170); EL_PBL, EXCH_H, EXCH_M and KPBL
#: are the same shared rows MYNN's block publishes, so the records are
#: identical and the merge below keeps one. Native restart-only turbulent
#: fluxes are additional history products; wrf_history records their flags.
BOULAC_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "tke_pbl": WrfOutputField(
        "TKE_PBL", "f4", "Z",
        'TKE from PBL',
        'm2 s-2',
        "Registry.EM_COMMON:1231", wrf_history=True),
    "el_pbl": MYNN_PBL_OUTPUT_FIELDS["el_pbl"],
    "exch_h": MYNN_PBL_OUTPUT_FIELDS["exch_h"],
    "exch_m": MYNN_PBL_OUTPUT_FIELDS["exch_m"],
    "kpbl": MYNN_PBL_OUTPUT_FIELDS["kpbl"],
    "wu_tur": WrfOutputField(
        "WU_TUR", "f4", "", "Turbulent flux of momentum(x)", "m2 s-2",
        "Registry.EM_COMMON:1240", wrf_history=False),
    "wv_tur": WrfOutputField(
        "WV_TUR", "f4", "", "Turbulent flux of momentum(y)", "m2 s-2",
        "Registry.EM_COMMON:1241", wrf_history=False),
    "wt_tur": WrfOutputField(
        "WT_TUR", "f4", "", "Turbulent flux of temperature", "K m s-1",
        "Registry.EM_COMMON:1242", wrf_history=False),
    "wq_tur": WrfOutputField(
        "WQ_TUR", "f4", "", "Turbulent flux of water vapor", "- m s-1",
        "Registry.EM_COMMON:1243", wrf_history=False),
}

#: ACM2's additional carried surface regime. The exchange coefficients,
#: inverse Monin-Obukhov length and PBL-top index reuse the shared rows above.
ACM2_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "regime": WrfOutputField(
        "REGIME", "f4", "",
        'FLAGS: 1=Night/Stable, 2=Mechanical Turbulent, 3=Forced Conv, 4=Free Conv',
        '', "Registry.EM_COMMON:1979", wrf_history=False),
}
# PX's additional surface state. Registry restart-only and scratch fields
# are explicit history extensions, emitted only when surface layer 7 runs.
PX_SFCLAY_OUTPUT_FIELDS = {
    "rmol": MYNN_PBL_OUTPUT_FIELDS["rmol"],
}
for _key, _description, _units, _line in (
    ("wspd", "Wind Speed At Lowest Model Level (may contain vconv)", "m s-1", 1041),
    ("gz1oz0", "LOG OF Z1 over Z0", "", 1042),
    ("br", "Bulk Richardson", "", 1043),
    ("zol", "z/L", "", 1044),
    ("qsfc", "SPECIFIC HUMIDITY AT LOWER BOUNDARY", "kg kg-1", 1065),
    ("znt", "TIME-VARYING ROUGHNESS LENGTH", "m", 1948),
    ("mol", "T* IN SIMILARITY THEORY", "K", 1957),
    ("flhc", "SURFACE EXCHANGE COEFFICIENT FOR HEAT", "", 1968),
    ("flqc", "SURFACE EXCHANGE COEFFICIENT FOR MOISTURE", "", 1969),
    ("regime", "FLAGS: 1=Night/Stable, 2=Mechanical Turbulent, 3=Forced Conv, 4=Free Conv", "", 1979),
):
    PX_SFCLAY_OUTPUT_FIELDS[_key] = WrfOutputField(
        _key.upper(), "f4", "", _description, _units,
        f"Registry.EM_COMMON:{_line}", wrf_history=False)
# These five are surface-driver scratch rather than Registry state. Their
# descriptions and units follow module_sf_pxsfclay.F:71-74,91.
for _key, _description, _units, _line in (
    ("chs", "exchange coefficient for heat", "m s-1", 71),
    ("chs2", "exchange coefficient for heat at 2 m", "m s-1", 72),
    ("cqs2", "exchange coefficient for moisture at 2 m", "m s-1", 73),
    ("cpm", "heat capacity at constant pressure for moist air", "J kg-1 K-1", 74),
    ("qgh", "lowest-level saturated mixing ratio", "kg kg-1", 91),
):
    PX_SFCLAY_OUTPUT_FIELDS[_key] = WrfOutputField(
        _key.upper(), "f4", "", _description, _units,
        f"phys/module_sf_pxsfclay.F:{_line}", wrf_history=False)


#: Noah-MP's carried state and published diagnostics
#: (``sf_surface_physics=4``).  ``RS`` is the one member whose row is in
#: EM_COMMON rather than registry.noahmp: WRF declares it once for P-X, Noah
#: and Noah-MP together, and the comment one line above it says so.
NOAHMP_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    # ArWen exposes the sixty native vector slots as named scalar planes.
    # This records the complete opt-in state without inventing slot units
    # or changing the history schema of any option-off run.
    **{f"gecros{k:02d}xy": WrfOutputField(
        f"GECROS{k:02d}", "f4", "", f"GECROS native state slot {k}", "",
        "registry.noahmp:227 (WRF v4.7.1 vector slot; WOOF scalar export)", False)
       for k in range(1, 61)},
    **{name:WrfOutputField(external.strip().upper(),"f4","",description,units,f"registry.noahmp:{line}",False)
       for name,external,description,units,line in (('tradxy', 'trad', 'surface radiative temperature', 'K', 36), ('fvegxy', 'fveg', 'Noah-MP vegetation fraction', '', 40), ('runsfxy', 'runsf', 'surface runoff', 'mm/s', 42), ('runsbxy', 'runsb', 'subsurface runoff', 'mm/s', 43), ('ecanxy', 'ecan', 'evaporation of intercepted water', 'mm/s', 44), ('edirxy', 'edir', 'ground surface evaporation rate', 'mm/s', 45), ('etranxy', 'etran', 'transpiration rate', 'mm/s', 46), ('savxy', 'sav', 'solar rad absorbed by veg', 'W/m2', 51), ('sagxy', 'sag', 'solar rad absorbed by ground', 'W/m2', 52), ('shgxy', 'shg', 'sensible heat flux: ground to canopy', 'W/m2', 61), ('shcxy', 'shc', 'sensible heat flux: leaf to canopy', 'W/m2', 62), ('shbxy', 'shb', 'sensible heat flux: bare grnd to atmo', 'W/m2', 63), ('evgxy', 'evg', 'latent heat flux: ground to canopy', 'W/m2', 64), ('evbxy', 'evb', 'latent heat flux: bare grnd to atmo', 'W/m2', 65), ('ghvxy', 'ghv', 'heat flux into soil: under canopy', 'W/m2', 66), ('ghbxy', 'ghb', 'heat flux into soil: bare fraction', 'W/m2', 67), ('irgxy', 'irg', 'net longwave at below canopy surface', 'W/m2', 68), ('ircxy', 'irc', 'net longwave in canopy', 'W/m2', 69), ('irbxy', 'irb', 'net longwave at bare fraction surface', 'W/m2', 70), ('trxy', 'tr', 'transpiration', 'W/m2', 71), ('evcxy', 'evc', 'canopy evaporation', 'W/m2', 72), ('deeprechxy', 'deeprech', 'deep water table recharge', 'mm', 81), ('acrech', 'acrech', 'accumulated recharge', 'mm', 82), ('qrfxy', 'qrf', 'baseflow ', 'm', 84), ('qspringxy', 'qspring', 'seeping water ', 'm', 86), ('acqspring', 'acqspring', 'accumulated seeping water', 'mm', 88), ('qlatxy', 'qlat', 'instantaneous lateral flow', 'mm', 90), ('qintsxy', 'qints', 'canopy snow interception rate', 'mm/s', 98), ('qintrxy', 'qintr', 'canopy rain interception rate', 'mm/s', 99), ('qdripsxy', 'qdrips', 'canopy snow drip rate', 'mm/s', 100), ('qdriprxy', 'qdripr', 'canopy rain drip rate', 'mm/s', 101), ('qthrosxy', 'qthros', 'snow throughfall rate', 'mm/s', 102), ('qthrorxy', 'qthror', 'rain throughfall rate', 'mm/s', 103), ('qsnsubxy', 'qsnsub', 'snow pack sublimation rate', 'mm/s', 104), ('qsnfroxy', 'qsnfro', 'snow pack frost rate', 'mm/s', 105), ('qsubcxy', 'qsubc', 'canopy sublimation rate', 'mm/s', 106), ('qfrocxy', 'qfroc', 'canopy frost rate', 'mm/s', 107), ('qevacxy', 'qevac', 'canopy evaporation rate', 'mm/s', 108), ('qdewcxy', 'qdewc', 'canopy dew rate', 'mm/s', 109), ('qfrzcxy', 'qfrzc', 'canopy liquid freeze rate', 'mm/s', 110), ('qmeltcxy', 'qmeltc', 'canopy snow melt rate', 'mm/s', 111), ('qsnbotxy', 'qsnbot', 'total liquid water (melt + rain through pack) out of snowpack bottom', 'mm/s', 112), ('qmeltxy', 'qmelt', 'snow melt rate due to phase change', 'mm/s', 113), ('pondingxy', 'ponding', 'surface ponding from complete pack melt', 'mm', 114), ('pahgxy', 'pahg', 'precipitation advected energy - to below canopy', 'W/m2', 116), ('pahvxy', 'pahv', 'precipitation advected energy - to vegetation', 'W/m2', 117), ('pahbxy', 'pahb', 'precipitation advected energy - to bare ground', 'W/m2', 118), ('canhsxy', 'canhs', 'canopy heat storage change due to canopy temperature change', 'W/m2', 119), ('fpicexy', 'fpice', 'fraction of ice in precipitation', 'fraction', 120), ('rainlsm', 'rainlsm', 'liquid precipitation into LSM', 'mm/s', 121), ('snowlsm', 'snowlsm', 'frozen precipitation into LSM', 'mm/s', 122), ('acints', 'acints', 'Accumulated canopy snow interception', 'mm', 130), ('acintr', 'acintr', 'Accumulated canopy rain interception', 'mm', 131), ('acdripr', 'acdripr', 'Accumulated canopy rain drip', 'mm', 132), ('acthror', 'acthror', 'Accumulated rain throughfall', 'mm', 133), ('acevac', 'acevac', 'Accumulated canopy evaporation', 'mm', 134), ('acdewc', 'acdewc', 'Accumulated canopy dew', 'mm', 135), ('acrainlsm', 'acrainlsm', 'Accumulated liquid precipitation into LSM', 'mm', 145), ('acrunsb', 'acrunsb', 'Accumulated subsurface runoff', 'mm', 146), ('acrunsf', 'acrunsf', 'Accumulated surface runoff', 'mm', 147), ('acecan', 'acecan', 'Accumulated evaporation of intercepted water', 'mm', 148), ('acetran', 'acetran', 'Accumulated transpiration', 'mm', 149), ('acedir', 'acedir', 'Accumulated ground surface evaporation', 'mm', 150), ('acqlat', 'acqlat', 'accumulated lateral flow', 'mm', 151), ('acqrf', 'acqrf', 'accumulated baseflow', 'mm', 152), ('acetlsm', 'acetlsm', 'Accumulated evaporation', 'mm', 153), ('acsnowlsm', 'acsnowlsm', 'Accumulated frozen precipitation into LSM', 'mm', 156), ('acsubc', 'acsubc', 'Accumulated canopy sublimation', 'mm', 157), ('acfroc', 'acfroc', 'Accumulated canopy frost', 'mm', 158), ('acfrzc', 'acfrzc', 'Accumulated canopy liquid freeze', 'mm', 159), ('acmeltc', 'acmeltc', 'Accumulated canopy snow melt', 'mm', 160), ('acsnbot', 'acsnbot', 'Accumulated total liquid water (melt + rain through pack) out of snowpack bottom', 'mm', 163), ('acsnmelt', 'acsnmelt', 'Accumulated snowmelt due to phase change', 'mm', 164), ('acponding', 'acponding', 'Accumulated surface ponding from complete pack melt', 'mm', 165), ('acsnsub', 'acsnsub', 'Accumulated snow pack sublimation', 'mm', 166), ('acsnfro', 'acsnfro', 'Accumulated snow pack frost', 'mm', 167), ('acrainsnow', 'acrainsnow', 'accumulated rain on snow pack', 'mm', 168), ('acdrips', 'acdrips', 'Accumulated canopy snow drip', 'mm', 169), ('acthros', 'acthros', 'Accumulated snow throughfall', 'mm', 170), ('acsagb', 'acsagb', 'Accumulated solar radiation absorbed at bare fraction', 'kJ/m2', 173), ('acirb', 'acirb', 'Accumulated net longwave at bare fraction ', 'kJ/m2', 174), ('acshb', 'acshb', 'Accumulated sensible heat flux at bare fraction', 'kJ/m2', 175), ('acevb', 'acevb', 'Accumulated latent heat flux at bare fraction', 'kJ/m2', 176), ('acghb', 'acghb', 'Accumulated heat flux into soil at bare fraction', 'kJ/m2', 177), ('acpahb', 'acpahb', 'Accumulated precipitation advected energy to bare ground', 'kJ/m2', 178), ('acsagv', 'acsagv', 'Accumulated solar radiation absorbed at vegetated fraction', 'kJ/m2', 179), ('acirg', 'acirg', 'Accumulated net longwave below canopy surface', 'kJ/m2', 180), ('acshg', 'acshg', 'Accumulated sensible heat flux: ground to canopy', 'kJ/m2', 181), ('acevg', 'acevg', 'Accumulated latent heat flux: ground to canopy', 'kJ/m2', 182), ('acghv', 'acghv', 'Accumulated heat flux into soil under canopy', 'kJ/m2', 183), ('acpahg', 'acpahg', 'Accumulated precipitation advected energy to below canopy', 'kJ/m2', 184), ('acsav', 'acsav', 'Accumulated solar rad absorbed by vegetated fraction', 'kJ/m2', 185), ('acirc', 'acirc', 'Accumulated net longwave in canopy', 'kJ/m2', 186), ('acshc', 'acshc', 'Accumulated sensible heat flux: canopy to atmosphere', 'kJ/m2', 187), ('acevc', 'acevc', 'Accumulated canopy evaporation', 'kJ/m2', 188), ('actr', 'actr', 'Accumulated transpiration', 'kJ/m2', 189), ('acpahv', 'acpahv', 'Accumulated precipitation advected energy to vegetation', 'kJ/m2', 190), ('acswdnlsm', 'acswdnlsm', 'Accumulated shortwave down at LSM', 'kJ/m2', 191), ('acswuplsm', 'acswuplsm', 'Accumulated shortwave up at LSM', 'kJ/m2', 192), ('aclwdnlsm', 'aclwdnlsm', 'Accumulated longwave down at LSM', 'kJ/m2', 193), ('aclwuplsm', 'aclwuplsm', 'Accumulated longwave up at LSM', 'kJ/m2', 194), ('acshflsm', 'acshflsm', 'accumulated total sensible heat flux', 'kJ/m2', 195), ('aclhflsm', 'aclhflsm', 'accumulated total latent heat flux', 'kJ/m2', 196), ('acghflsm', 'acghflsm', 'accumulated total ground heat flux', 'kJ/m2', 197), ('acpahlsm', 'acpahlsm', 'accumulated total precip heat flux', 'kJ/m2', 198), ('accanhs', 'accanhs', 'accumulated canopy storage change', 'KJ/m2', 199), ('aceflxb', 'aceflxb ', 'accumulated bottom soil heat flux', 'kJ/m2', 208), ('eflxbxy', 'eflxbxy ', 'bottom soil heat flux', 'W/m2', 209))},
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
    **{f"soilcl{k}":WrfOutputField(f"SOILCL{k}","f4","",
        f"soil texture class - level {k}","cat",f"registry.noahmp:{123+k}",False)
        for k in range(1,5)},
    **{name+"_3d":WrfOutputField(name.upper()+"_3D","f4","Z",
        "direct Noah-MP soil "+name,units,"WRF v4.7.1 noahmpdrv:933-941 (completed direct-input arm)",False)
       for name,units in (("smcwlt","m3 m-3"),("smcref","m3 m-3"),("smcmax","m3 m-3"),
                          ("bexp",""),("psisat","m"),("dksat","m s-1"),("dwsat","m2 s-1"),
                          ("smcdry","m3 m-3"),("quartz",""))},
    "soilcomp":WrfOutputField("SOILCOMP","f4","Z","soil composition","%",
                               "registry.noahmp:123",False),
    **{name:WrfOutputField(external.strip().upper(),"f4","Z" if name=="acc_etrani" else "",description,units,f"registry.noahmp:{line}",False)
       for name,external,description,units,line in (
           ("acc_ssoil","ACC_SSOIL","accumulated SSOIL between soil updates","W/m2",204),
           ("acc_qinsur","ACC_QINSUR","accumulated QINSUR between soil updates","m/s*dt_soil/dt_noahmp",205),
           ("acc_qseva","ACC_QSEVA","accumulated QSEVA between soil updates","m/s*dt_soil/dt_noahmp",206),
           ("acc_etrani","ACC_ETRANI","accumulated ETRANI between soil updates","m/s*dt_soil/dt_noahmp",207),
           ("acc_dwaterxy","ACC_DWATER","accumulated canopy snow soil water change","mm",210),
           ("acc_prcpxy","ACC_PRCP","accumulated precipitation","mm",211),
           ("acc_ecanxy","ACC_ECAN","accumulated net canopy evaporation","mm",212),
           ("acc_etranxy","ACC_ETRAN","accumulated transpiration","mm",213),
           ("acc_edirxy","ACC_EDIR","accumulated net soil evaporation","mm",214))},
    "td_fraction": WrfOutputField("TD_FRACTION","f4","","Tile drainage fraction","","registry.noahmp:230",False),
    "planting": WrfOutputField("PLANTING","f4","","planting date","julian day","registry.noahmp:220",False),
    "harvest": WrfOutputField("HARVEST","f4","","harvest date","julian day","registry.noahmp:221",False),
    # Complete native crop carriers and statics.
    'grainxy': WrfOutputField('GRAIN','f4',"Z" if 'grainxy'=="croptype" else "",'mass of grain','g/m2',"registry.noahmp:217",True),
    'gddxy': WrfOutputField('GDD','f4',"Z" if 'gddxy'=="croptype" else "",'growing degree days','',"registry.noahmp:218",True),
    'croptype': WrfOutputField('CROPTYPE','f4',"Z" if 'croptype'=="croptype" else "",'crop type','fraction',"registry.noahmp:219",False),
    'season_gdd': WrfOutputField('SEASON_GDD','f4',"Z" if 'season_gdd'=="croptype" else "",'growing season GDD','C',"registry.noahmp:222",False),
    'cropcat': WrfOutputField('CROPCAT','i4',"Z" if 'cropcat'=="croptype" else "",'dominant crop category','category',"registry.noahmp:223",True),
    # Native irrigation records, Registry/registry.noahmp:234-248.
    'irfract': WrfOutputField('IRFRACT','f4',"",'Irrigation fraction','',"registry.noahmp:234",False),
    'sifract': WrfOutputField('SIFRACT','f4',"",'Sprinkler Irrigation fraction','',"registry.noahmp:235",False),
    'mifract': WrfOutputField('MIFRACT','f4',"",'Micro Irrigation fraction','',"registry.noahmp:236",False),
    'fifract': WrfOutputField('FIFRACT','f4',"",'Flood Irrigation fraction','',"registry.noahmp:237",False),
    'irnumsi': WrfOutputField('IRNUMSI','i4',"",'Sprinkler Irrigation Event Count','',"registry.noahmp:238",True),
    'irnummi': WrfOutputField('IRNUMMI','i4',"",'Micro Irrigation Event Count','',"registry.noahmp:239",True),
    'irnumfi': WrfOutputField('IRNUMFI','i4',"",'Flood Irrigation Event Count','',"registry.noahmp:240",True),
    'irwatsi': WrfOutputField('IRWATSI','f4',"",'Sprinkler Irrigation Amount for the Event','m',"registry.noahmp:241",False),
    'irwatmi': WrfOutputField('IRWATMI','f4',"",'Micro Irrigation Amount for the Event','m',"registry.noahmp:242",False),
    'irwatfi': WrfOutputField('IRWATFI','f4',"",'Flood Irrigation Amount for the Event','m',"registry.noahmp:243",False),
    'irsivol': WrfOutputField('IRSIVOL','f4',"",'Sprinkler Irrigation Water Accumulated','mm',"registry.noahmp:244",True),
    'irmivol': WrfOutputField('IRMIVOL','f4',"",'Micro Irrigation Water Accumulated','mm',"registry.noahmp:245",True),
    'irfivol': WrfOutputField('IRFIVOL','f4',"",'Flood Irrigation Water Accumulated','mm',"registry.noahmp:246",True),
    'ireloss': WrfOutputField('IRELOSS','f4',"",'Sprinkler Evaporation Loss Accumulated','mm',"registry.noahmp:247",True),
    'irrsplh': WrfOutputField('IRRSPLH','f4',"",'Sprinkler Evaporation Loss Accumulated','Joules m^-2',"registry.noahmp:248",True),
    "qtdrain": _output_field("QTDRAIN","Accumulated tile drainage flux","mm","registry.noahmp:231"),
    'smcwtdxy':WrfOutputField('smcwtd',"f4","",'deep soil moisture ','m3 m-3',"registry.noahmp:79",False),
    'deeprechxy':WrfOutputField('deeprech',"f4","",'deep water table recharge','mm',"registry.noahmp:81",True),
    'rechxy':WrfOutputField('rech',"f4","",'water table recharge','mm',"registry.noahmp:80",True),
    'qslatxy':WrfOutputField('qslat',"f4","",'sum lateral flow ','mm',"registry.noahmp:89",False),
    'qrfsxy':WrfOutputField('qrfs',"f4","",'sum baseflow ','mm',"registry.noahmp:85",False),
    'qspringsxy':WrfOutputField('qsprings',"f4","",'sum seeping water ','mm',"registry.noahmp:87",False),
    'qlatxy':WrfOutputField('qlat',"f4","",'instantaneous lateral flow','mm',"registry.noahmp:90",False),
    'qrfxy':WrfOutputField('qrf',"f4","",'baseflow ','m',"registry.noahmp:84",False),
    'qspringxy':WrfOutputField('qspring',"f4","",'seeping water ','m',"registry.noahmp:86",False),
    'fdepthxy':WrfOutputField('FDEPTH',"f4","",'e-folding depth for transmissivity ','m',"registry.noahmp:94",False),
    'areaxy':WrfOutputField('area',"f4","",'area of grid boxes','m2',"registry.noahmp:83",False),
    'ht':_output_field('TOPO','terrain height','m',"registry.noahmp:107"),
    'eqzwt':WrfOutputField('EQZWT',"f4","",'equilibrium water table depth ','m',"registry.noahmp:95",False),
    'pexpxy':WrfOutputField('pexp',"f4","",'exponent for river conductance','Kg m s-1',"registry.noahmp:92",False),
    'riverbedxy':WrfOutputField('RIVERBED',"f4","",'river bed depth','m',"registry.noahmp:97",False),
    'rivercondxy':WrfOutputField('rivercond',"f4","",'river conductance','Kg m s-1',"registry.noahmp:93",False),
    'rechclim':WrfOutputField('RECHCLIM',"f4","",'equilibrium recharge rate ','mm',"registry.noahmp:96",False),
    'smoiseq':WrfOutputField('SMOISEQ',"f4","Z",'EQ. SOIL MOISTURE','m3 m-3',"registry.noahmp:78",False),
    "zwtxy": _output_field("ZWT","depth to water table","m","registry.noahmp:17"),
    "wtxy": _output_field("WT","water in aquifer and saturated soil","mm","registry.noahmp:19"),
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
    "lfmassxy": _output_field("LFMASS", "leaf mass", "g/m2", "registry.noahmp:24"),
    "rtmassxy": _output_field("RTMASS", "mass of fine roots", "g/m2", "registry.noahmp:25"),
    "stmassxy": _output_field("STMASS", "stem mass", "g/m2", "registry.noahmp:26"),
    "woodxy": _output_field("WOOD", "mass of wood", "g/m2", "registry.noahmp:27"),
    "stblcpxy": _output_field("STBLCP", "stable carbon pool", "g/m2", "registry.noahmp:28"),
    "fastcpxy": _output_field("FASTCP", "short-lived carbon", "g/m2", "registry.noahmp:29"),
    "neexy": _output_field("NEE", "net ecosystem exchange", "g/m2/s CO2", "registry.noahmp:37"),
    "gppxy": _output_field("GPP", "gross primary productivity", "g/m2/s C", "registry.noahmp:38"),
    "nppxy": _output_field("NPP", "net primary productivity", "g/m2/s C", "registry.noahmp:39"),
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
    "SST": WrfOutputField(
        "SST", "f4", "", "SEA SURFACE TEMPERATURE", "K",
        "Registry.EM:15", wrf_history=True),
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
#: The GBM moist TKE PBL's own rows (``bl_pbl_physics=12``).  EL_PBL,
#: EXCH_H, EXCH_M and KPBL are the shared rows above.
GBM_PBL_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    # GBMPBL retains the surface-layer REGIME carrier. ACM2/PX publish the
    # same Registry.EM_COMMON:1979 flag, including its native units and text.
    "regime": ACM2_OUTPUT_FIELDS["regime"],
    "tke_pbl": WrfOutputField(
        "TKE_PBL", "f4", "Z",
        'TKE from PBL',
        'm2 s-2',
        "Registry.EM_COMMON:1231", wrf_history=True),
    "exch_tke": WrfOutputField(
        "EXCH_TKE", "f4", "",
        'Exchange coefficient TKE enhanced',
        'm2 s-1',
        "Registry.EM_COMMON:1156", wrf_history=True),
}

GSFCGCE4ICE_OUTPUT_FIELDS = {
    're_cloud_gsfc': WrfOutputField('RE_CLOUD_GSFC', "f4", "", 'Cloud Water effective radius', 'micron', "Registry.EM_COMMON:500", wrf_history=True),
    're_rain_gsfc': WrfOutputField('RE_RAIN_GSFC', "f4", "", 'Rain Water effective radius', 'micron', "Registry.EM_COMMON:501", wrf_history=True),
    're_ice_gsfc': WrfOutputField('RE_ICE_GSFC', "f4", "", 'Cloud Ice effective radius', 'micron', "Registry.EM_COMMON:502", wrf_history=True),
    're_snow_gsfc': WrfOutputField('RE_SNOW_GSFC', "f4", "", 'Snow effective radius', 'micron', "Registry.EM_COMMON:503", wrf_history=True),
    're_graupel_gsfc': WrfOutputField('RE_GRAUPEL_GSFC', "f4", "", 'Graupel Water effective radius', 'micron', "Registry.EM_COMMON:504", wrf_history=True),
    're_hail_gsfc': WrfOutputField('RE_HAIL_GSFC', "f4", "", 'Hail Water effective radius', 'micron', "Registry.EM_COMMON:505", wrf_history=True),
    'phys_tot': WrfOutputField('PHYS_TOT', "f4", "", 'TOTAL LATENT HEATING RATE', 'K s-1', "Registry.EM_COMMON:1487", wrf_history=True),
    'physc': WrfOutputField('PHYSC', "f4", "", 'LATENT HEATING RATE DUE TO CONDENSATION', 'K s-1', "Registry.EM_COMMON:1488", wrf_history=False),
    'physe': WrfOutputField('PHYSE', "f4", "", 'LATENT HEATING RATE DUE TO EVAPORATION', 'K s-1', "Registry.EM_COMMON:1489", wrf_history=False),
    'physd': WrfOutputField('PHYSD', "f4", "", 'LATENT HEATING RATE DUE TO DEPOSITION', 'K s-1', "Registry.EM_COMMON:1490", wrf_history=False),
    'physs': WrfOutputField('PHYSS', "f4", "", 'LATENT HEATING RATE DUE TO SUBLIMATION', 'K s-1', "Registry.EM_COMMON:1491", wrf_history=False),
    'physm': WrfOutputField('PHYSM', "f4", "", 'LATENT HEATING RATE DUE TO MELTING', 'K s-1', "Registry.EM_COMMON:1492", wrf_history=False),
    'physf': WrfOutputField('PHYSF', "f4", "", 'LATENT HEATING RATE DUE TO FREEZING', 'K s-1', "Registry.EM_COMMON:1493", wrf_history=False),
    'acphys_tot': WrfOutputField('ACPHYS_TOT', "f4", "", 'ACCUMULATED TOTAL LATENT HEATING', 'K', "Registry.EM_COMMON:1495", wrf_history=False),
    'acphysc': WrfOutputField('ACPHYSC', "f4", "", 'ACCUMULATED LATENT HEATING DUE TO CONDENSATION', 'K', "Registry.EM_COMMON:1496", wrf_history=False),
    'acphyse': WrfOutputField('ACPHYSE', "f4", "", 'ACCUMULATED LATENT HEATING DUE TO EVAPORATION', 'K', "Registry.EM_COMMON:1497", wrf_history=False),
    'acphysd': WrfOutputField('ACPHYSD', "f4", "", 'ACCUMULATED LATENT HEATING DUE TO DEPOSITION', 'K', "Registry.EM_COMMON:1498", wrf_history=False),
    'acphyss': WrfOutputField('ACPHYSS', "f4", "", 'ACCUMULATED LATENT HEATING DUE TO SUBLIMATION', 'K', "Registry.EM_COMMON:1499", wrf_history=False),
    'acphysm': WrfOutputField('ACPHYSM', "f4", "", 'ACCUMULATED LATENT HEATING DUE TO MELTING', 'K', "Registry.EM_COMMON:1500", wrf_history=False),
    'acphysf': WrfOutputField('ACPHYSF', "f4", "", 'ACCUMULATED LATENT HEATING DUE TO FREEZING', 'K', "Registry.EM_COMMON:1501", wrf_history=False),
    'preci3d': WrfOutputField('PRECI3D', "f4", "", 'Sedimentation Fluxes for cloud ice', 'KG/M^2/S', "Registry.EM_COMMON:1505", wrf_history=False),
    'precs3d': WrfOutputField('PRECS3D', "f4", "", 'Sedimentation Fluxes for snow', 'KG/M^2/S', "Registry.EM_COMMON:1506", wrf_history=False),
    'precg3d': WrfOutputField('PRECG3D', "f4", "", 'Sedimentation Fluxes for graupel', 'KG/M^2/S', "Registry.EM_COMMON:1507", wrf_history=False),
    'prech3d': WrfOutputField('PRECH3D', "f4", "", 'Sedimentation Fluxes for hail', 'KG/M^2/S', "Registry.EM_COMMON:1508", wrf_history=False),
    'precr3d': WrfOutputField('PRECR3D', "f4", "", 'Sedimentation Fluxes for rain', 'KG/M^2/S', "Registry.EM_COMMON:1509", wrf_history=False),
    "hailncv": WrfOutputField("HAILNCV", "f4", "", "Hail precipitation in the current microphysics call", "mm", "module_microphysics_driver.F:2095", wrf_history=False),
}


KF_DIAGNOSTIC_OUTPUT_FIELDS = {
    "udr_kf": WrfOutputField("UDR_KF", "f4", "", "UPDRAFT DETRAINMENT RATE FROM KF", "kg s-1", "Registry.EM_COMMON:1625", True),
    "ddr_kf": WrfOutputField("DDR_KF", "f4", "", "DOWNDRAFT DETRAINMENT RATE FROM KF", "kg s-1", "Registry.EM_COMMON:1626", True),
    "uer_kf": WrfOutputField("UER_KF", "f4", "", "UPDRAFT ENTRAINMENT RATE FROM KF", "kg s-1", "Registry.EM_COMMON:1627", True),
    "der_kf": WrfOutputField("DER_KF", "f4", "", "DOWNDRAFT ENTRAINMENT RATE FROM KF", "kg s-1", "Registry.EM_COMMON:1628", True),
    "timec_kf": WrfOutputField("TIMEC_KF", "f4", "", "CONVECTIVE TIMESCALE FROM MSKF", "s", "Registry.EM_COMMON:1629", True),
}
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
# Native WRF v4.6.1 Registry.EM_COMMON:1339-1348.
PWP_OUTPUT_FIELDS = {
    'om_tmp': WrfOutputField('OM_TMP', "f4", 'Z', 'temperature', 'k', "Registry.EM_COMMON:1339", wrf_history=True),
    'om_s': WrfOutputField('OM_S', "f4", 'Z', 'salinity', 'degree', "Registry.EM_COMMON:1340", wrf_history=True),
    'om_u': WrfOutputField('OM_U', "f4", 'Z', 'u current', 'm/s', "Registry.EM_COMMON:1342", wrf_history=True),
    'om_v': WrfOutputField('OM_V', "f4", 'Z', 'v current', 'm/s', "Registry.EM_COMMON:1343", wrf_history=True),
    'om_depth': WrfOutputField('OM_DEPTH', "f4", 'Z', 'depth', 'm', "Registry.EM_COMMON:1341", wrf_history=True),
    'om_tini': WrfOutputField('OM_TINI', "f4", 'Z', 'temperature init', 'k', "Registry.EM_COMMON:1347", wrf_history=False),
    'om_sini': WrfOutputField('OM_SINI', "f4", 'Z', 'salinity init', 'degree', "Registry.EM_COMMON:1348", wrf_history=False),
    'om_ml': WrfOutputField('OM_ML', "f4", '', 'om ml', 'm', "Registry.EM_COMMON:1346", wrf_history=True),
    "om_density": WrfOutputField("OM_DENSITY", "f4", "Z", "Ocean density; no WRF Registry counterpart", "kg m-3", "", wrf_history=False),
    "om_active": WrfOutputField("OM_ACTIVE", "f4", "", "Complete ocean analysis coverage; one active zero prescribed", "1", "TC ocean coverage", wrf_history=False),
}
OCEAN_DIAGNOSTIC_FIELDS = {
    "ocean_sst": WrfOutputField("OCEAN_SST", "f4", "", "Prognostic ocean surface temperature; land and lakes missing; no WRF Registry counterpart", "K", "", wrf_history=False),
    "ocean_sst_change": WrfOutputField("OCEAN_SST_CHANGE", "f4", "", "Ocean surface temperature change from coupled initial state; no WRF Registry counterpart", "K", "", wrf_history=False),
    'ocean_ohc26': WrfOutputField('OCEAN_OHC26', "f4", "", 'HEAT ABOVE 26 C IN SURFACE CONNECTED WARM LAYER; LOWER BOUND IF D26 CENSORED', 'kJ cm-2', "TC ocean diagnostics", wrf_history=False),
    'ocean_d26': WrfOutputField('OCEAN_D26', "f4", "", 'FIRST DOWNWARD 26 C ISOTHERM; MISSING IF BELOW OCEAN COLUMN', 'm', "TC ocean diagnostics", wrf_history=False),
    'ocean_ild': WrfOutputField('OCEAN_ILD', "f4", "", 'ISOTHERMAL LAYER DEPTH FROM 10 M AND 0.2 K THRESHOLD', 'm', "TC ocean diagnostics", wrf_history=False),
    'ocean_mld': WrfOutputField('OCEAN_MLD', "f4", "", 'LAYER DEPTH FROM 10 M AND 0.03 KG M-3 MODEL DENSITY INCREASE', 'm', "TC ocean diagnostics", wrf_history=False),
    'ocean_barrier_layer': WrfOutputField('OCEAN_BARRIER_LAYER', "f4", "", 'MAXIMUM OF ZERO AND ISOTHERMAL MINUS DENSITY LAYER DEPTH', 'm', "TC ocean diagnostics", wrf_history=False),
    'ocean_d26_censored': WrfOutputField('OCEAN_D26_CENSORED', "f4", "", 'ONE IF 26 C ISOTHERM IS BELOW COLUMN; ZERO OTHERWISE', '1', "TC ocean diagnostics", wrf_history=False),
}

TEMF_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    'te_temf': WrfOutputField('TE_TEMF', "f4", "", 'Total energy from TEMF PBL scheme', 'm2 s-2', "Registry.EM_COMMON:1098", wrf_history=True),
    'kh_temf': WrfOutputField('KH_TEMF', "f4", "", 'Diffusion coefficient for heat from TEMF PBL', '', "Registry.EM_COMMON:1099", wrf_history=True),
    'km_temf': WrfOutputField('KM_TEMF', "f4", "", 'Diffusion coefficient for momentum from TEMF PBL', '', "Registry.EM_COMMON:1100", wrf_history=True),
    'shf_temf': WrfOutputField('SHF_TEMF', "f4", "", 'Sensible heat flux from TEMF PBL', 'K m s-1', "Registry.EM_COMMON:1101", wrf_history=True),
    'qf_temf': WrfOutputField('QF_TEMF', "f4", "", 'Sensible heat flux from TEMF PBL', 'kg/kg m s-1', "Registry.EM_COMMON:1102", wrf_history=True),
    'uw_temf': WrfOutputField('UW_TEMF', "f4", "", 'U momentum flux from TEMF PBL', 'm2 s-2', "Registry.EM_COMMON:1103", wrf_history=True),
    'vw_temf': WrfOutputField('VW_TEMF', "f4", "", 'V momentum flux from TEMF PBL', 'm2 s-2', "Registry.EM_COMMON:1104", wrf_history=True),
    'wupd_temf': WrfOutputField('WUPD_TEMF', "f4", "", 'Updraft velocity from TEMF PBL', 'm s-1', "Registry.EM_COMMON:1105", wrf_history=True),
    'mf_temf': WrfOutputField('MF_TEMF', "f4", "", 'Mass flux from TEMF PBL', 'm s-1', "Registry.EM_COMMON:1106", wrf_history=True),
    'thup_temf': WrfOutputField('THUP_TEMF', "f4", "", 'Updraft thetal from TEMF PBL', 'K', "Registry.EM_COMMON:1107", wrf_history=True),
    'qtup_temf': WrfOutputField('QTUP_TEMF', "f4", "", 'Updraft qt from TEMF PBL', '1', "Registry.EM_COMMON:1108", wrf_history=True),
    'qlup_temf': WrfOutputField('QLUP_TEMF', "f4", "", 'Updraft ql (liquid water) from TEMF PBL', '1', "Registry.EM_COMMON:1109", wrf_history=True),
    'cf3d_temf': WrfOutputField('CF3D_TEMF', "f4", "", '3D Cloud fraction from TEMF PBL', '1', "Registry.EM_COMMON:1110", wrf_history=True),
    'hd_temf': WrfOutputField('HD_TEMF', "f4", "", 'Dry thermal top height from TEMF PBL', 'm', "Registry.EM_COMMON:1111", wrf_history=True),
    'lcl_temf': WrfOutputField('LCL_TEMF', "f4", "", 'Lifting condensation level from TEMF PBL', 'm', "Registry.EM_COMMON:1112", wrf_history=True),
    'hct_temf': WrfOutputField('HCT_TEMF', "f4", "", 'Cloud top height from TEMF PBL', 'm', "Registry.EM_COMMON:1113", wrf_history=True),
    'cfm_temf': WrfOutputField('CFM_TEMF', "f4", "", 'Column cloud fraction from TEMF PBL', '1', "Registry.EM_COMMON:1114", wrf_history=True),
    'wm_temf': WrfOutputField('WM_TEMF', "f4", "", 'Velocity scale in TEMF surface layer scheme', 'm s-1', "Registry.EM_COMMON:1115", wrf_history=True),
}



#: The K-epsilon PBL's (bl_pbl_physics=17) Registry rows, keyed by
#: ``PhysicsDriver.fields`` key (woof.core.physics_inventory.KEPS_FIELDS).
#: Registry.EM_COMMON:1231-1243 gives TKE/DISS/TPE/PR the history ``h``;
#: the advected copies and the four turbulent fluxes are restart-only.
KEPS_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    "keps_tke": WrfOutputField(
        "TKE_PBL", "f4", "Z", 'TKE from PBL', 'm2 s-2',
        "Registry.EM_COMMON:1231", wrf_history=True),
    "keps_diss": WrfOutputField(
        "DISS_PBL", "f4", "", 'Dissipation Rrate from PBL - keps', 'm2 s-3',
        "Registry.EM_COMMON:1233", wrf_history=True),
    "keps_tpe": WrfOutputField(
        "TPE_PBL", "f4", "", 'TPE from PBL- keps', 'K2',
        "Registry.EM_COMMON:1234", wrf_history=True),
    "keps_pr": WrfOutputField(
        "PR_PBL", "f4", "", 'prandtl number from pbl - keps', '-',
        "Registry.EM_COMMON:1238", wrf_history=True),
    "wu_tur": WrfOutputField(
        "WU_TUR", "f4", "", 'Turbulent flux of momentum(x)', 'm2 s-2',
        "Registry.EM_COMMON:1240", wrf_history=False),
    "wv_tur": WrfOutputField(
        "WV_TUR", "f4", "", 'Turbulent flux of momentum(y)', 'm2 s-2',
        "Registry.EM_COMMON:1241", wrf_history=False),
    "wt_tur": WrfOutputField(
        "WT_TUR", "f4", "", 'Turbulent flux of temperature', 'K m s-1',
        "Registry.EM_COMMON:1242", wrf_history=False),
}

# GRELLDRV per-call diagnostics. Published as a history extension: these
# pinned WRF Registry fields do not have the history flag in WRF 4.6.1.
GD_OUTPUT_FIELDS = {
    'gd_cloud': WrfOutputField('GD_CLOUD', 'f4', "", 'CLOUD WATER/ICE MIXING RAIO IN GD CLOUD', 'kg kg-1', "Registry.EM_COMMON:1653", wrf_history=False),
    'gd_cloud2': WrfOutputField('GD_CLOUD2', 'f4', "", 'TEST for GD CLOUD', 'kg kg-1', "Registry.EM_COMMON:1654", wrf_history=False),
    'htop': WrfOutputField('HTOP', 'f4', "", 'TOP OF CONVECTION LEVEL', '', "Registry.EM_COMMON:1246", wrf_history=False),
    'hbot': WrfOutputField('HBOT', 'f4', "", 'BOT OF CONVECTION LEVEL', '', "Registry.EM_COMMON:1247", wrf_history=False),
    'ktop_deep': WrfOutputField('KTOP_DEEP', 'i4', "", 'KTOP LEVEL FROM DEEP CONVECTION (G3 and GF only)', '', "Registry.EM_COMMON:1645", wrf_history=False),
    'apr_gr': WrfOutputField('APR_GR', 'f4', "", 'PRECIP FROM CLOSURE OLD_GRELL', 'mm hour-1', "Registry.EM_COMMON:1630", wrf_history=False),
    'apr_w': WrfOutputField('APR_W', 'f4', "", 'PRECIP FROM CLOSURE W', 'mm hour-1', "Registry.EM_COMMON:1631", wrf_history=False),
    'apr_mc': WrfOutputField('APR_MC', 'f4', "", 'PRECIP FROM CLOSURE KRISH MV', 'mm hour-1', "Registry.EM_COMMON:1632", wrf_history=False),
    'apr_st': WrfOutputField('APR_ST', 'f4', "", 'PRECIP FROM CLOSURE STABILITY', 'mm hour-1', "Registry.EM_COMMON:1633", wrf_history=False),
    'apr_as': WrfOutputField('APR_AS', 'f4', "", 'PRECIP FROM CLOSURE AS-TYPE', 'mm hour-1', "Registry.EM_COMMON:1634", wrf_history=False),
    'apr_capma': WrfOutputField('APR_CAPMA', 'f4', "", 'PRECIP FROM MAX CAP', 'mm hour-1', "Registry.EM_COMMON:1635", wrf_history=False),
    'apr_capme': WrfOutputField('APR_CAPME', 'f4', "", 'PRECIP FROM MEAN CAP', 'mm hour-1', "Registry.EM_COMMON:1636", wrf_history=False),
    'apr_capmi': WrfOutputField('APR_CAPMI', 'f4', "", 'PRECIP FROM MIN CAP', 'mm hour-1', "Registry.EM_COMMON:1637", wrf_history=False),
}

SCHEME_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    'm11': WrfOutputField('m11', "f4", '', '11 component of NBA subgrid stress tensor', 'm2 s-2', "registry.les:2", wrf_history=True),
    'm22': WrfOutputField('m22', "f4", '', '22 component of NBA subgrid stress tensor', 'm2 s-2', "registry.les:3", wrf_history=True),
    'm33': WrfOutputField('m33', "f4", '', '33 component of NBA subgrid stress tensor', 'm2 s-2', "registry.les:4", wrf_history=True),
    'm12': WrfOutputField('m12', "f4", '', '12 component of NBA subgrid stress tensor', 'm2 s-2', "registry.les:5", wrf_history=True),
    'm13': WrfOutputField('m13', "f4", '', '13 component of NBA subgrid stress tensor', 'm2 s-2', "registry.les:6", wrf_history=True),
    'm23': WrfOutputField('m23', "f4", '', '23 component of NBA subgrid stress tensor', 'm2 s-2', "registry.les:7", wrf_history=True),
    'massflux_edkf': WrfOutputField('MASS_FLUX', "f4", '', 'MASS FLUX FROM EDKF', 'Kg m s-1', "Registry.EM_COMMON:1084", wrf_history=False),
    'entr_edkf': WrfOutputField('ENTR', "f4", '', 'ENTRAINMENT FROM EDKF', 'm-1', "Registry.EM_COMMON:1085", wrf_history=False),
    'detr_edkf': WrfOutputField('DETR', "f4", '', 'ENTRAINMENT FROM EDKF', 'm-1', "Registry.EM_COMMON:1086", wrf_history=False),
    'thl_up': WrfOutputField('THL_UP', "f4", '', 'THL OF UPDRAFT FROM EDKF', 'K', "Registry.EM_COMMON:1087", wrf_history=False),
    'thv_up': WrfOutputField('THV_UP', "f4", '', 'THL OF UPDRAFT FROM EDKF', 'K', "Registry.EM_COMMON:1088", wrf_history=False),
    'rv_up': WrfOutputField('RV_UP', "f4", '', 'RV OF UPDRAFT FROM EDKF', 'kq/kg', "Registry.EM_COMMON:1089", wrf_history=False),
    'rt_up': WrfOutputField('RT_UP', "f4", '', 'RT OF UPDRAFT FROM EDKF', 'kq/kg', "Registry.EM_COMMON:1090", wrf_history=False),
    'rc_up': WrfOutputField('RC_UP', "f4", '', 'RC OF UPDRAFT FROM EDKF', 'kq/kg', "Registry.EM_COMMON:1091", wrf_history=False),
    'u_up': WrfOutputField('U_UP', "f4", '', 'U OF UPDRAFT FROM EDKF', 'm/s', "Registry.EM_COMMON:1092", wrf_history=False),
    'v_up': WrfOutputField('V_UP', "f4", '', 'U OF UPDRAFT FROM EDKF', 'm/s', "Registry.EM_COMMON:1093", wrf_history=False),
    'frac_up': WrfOutputField('FRAC_UP', "f4", '', 'FRACTION OF UPDRAFT FROM EDKF', '', "Registry.EM_COMMON:1094", wrf_history=False),
    'rc_mf': WrfOutputField('RC_MF', "f4", '', 'RC IN THE GRID COMPUTED BY EDKF', 'kg/kg', "Registry.EM_COMMON:1095", wrf_history=False),
}
# These are an explicit history extension for GFDL runs. Stock WRF keeps
# the cloud continuation in restart and leaves the remaining diagnostics
# out of its default streams. Blank metadata below is blank in WRF too.
GFDLETA_OUTPUT_FIELDS = {
    name: WrfOutputField(name.upper(), dtype, '', description, units,
                         f'Registry.EM_COMMON:{line}', wrf_history=False)
    for name, dtype, description, units, line in (
        ('htop', 'f4', 'TOP OF CONVECTION LEVEL', '', 1246),
        ('hbot', 'f4', 'BOT OF CONVECTION LEVEL', '', 1247),
        ('htopr', 'f4', 'TOP OF CONVECTION LEVEL FOR RADIATION', '', 1248),
        ('hbotr', 'f4', 'BOT OF CONVECTION LEVEL FOR RADIATION', '', 1249),
        ('cuppt', 'f4', 'ACCUMULATED CONVECTIVE RAIN SINC LAST CALL TO THE RADIATION', '', 1252),
        ('rswtoa', 'f4', '', '', 1253), ('rlwtoa', 'f4', '', '', 1254),
        ('czmean', 'f4', '', '', 1255), ('cfracl', 'f4', '', '', 1256),
        ('cfracm', 'f4', '', '', 1257), ('cfrach', 'f4', '', '', 1258),
        ('acfrst', 'f4', '', '', 1259), ('ncfrst', 'i4', '', '', 1260),
        ('acfrcv', 'f4', '', '', 1261), ('ncfrcv', 'i4', '', '', 1262),
        ('swdownc', 'f4', 'DOWNWARD CLEAR-SKY SHORT WAVE FLUX AT GROUND SURFACE', 'W m-2', 1709),
    )
}
SSIB_OUTPUT_FIELDS = {k: WrfOutputField(*v) for k, v in {'ssib_fm': ['SSIB_FM', 'f4', '', 'FM coeficient', '', 'Registry/registry.ssib:2', False], 'ssib_fh': ['SSIB_FH', 'f4', '', 'FH coeficient', '', 'Registry/registry.ssib:3', False], 'ssib_cm': ['SSIB_CM', 'f4', '', 'CM coeficient', '', 'Registry/registry.ssib:4', False], 'ssibxdd': ['SSIBXDD', 'f4', '', 'ZERO PLANE DISPLACEMENT', '', 'Registry/registry.ssib:5', False], 'ssib_br': ['SIBBR', 'f4', '', 'SSiB Bulk Richardson Number', '', 'Registry/registry.ssib:6', True], 'ssib_lhf': ['SIBLHF', 'f4', '', 'SSiB latent heat flx', '', 'Registry/registry.ssib:9', False], 'ssib_shf': ['SIBSHF', 'f4', '', 'SSiB sensible heat flx', '', 'Registry/registry.ssib:10', False], 'ssib_ghf': ['SIBGHF', 'f4', '', 'SSiB ground heat flx', '', 'Registry/registry.ssib:11', False], 'ssib_egs': ['SIBEGS', 'f4', '', 'SSiB evaporation from soil', '', 'Registry/registry.ssib:12', True], 'ssib_eci': ['SIBECI', 'f4', '', 'SSiB evaporation from interception', '', 'Registry/registry.ssib:13', True], 'ssib_ect': ['SIBECT', 'f4', '', 'SSiB evaporation from transpiration', '', 'Registry/registry.ssib:14', True], 'ssib_egi': ['SIBEGI', 'f4', '', 'SSiB evaporation from xxx', '', 'Registry/registry.ssib:15', True], 'ssib_egt': ['SIBEGT', 'f4', '', 'SSiB evaporation from snow', '', 'Registry/registry.ssib:16', True], 'ssib_sdn': ['SIBSDN', 'f4', '', 'SSiB short wave rad. down', '', 'Registry/registry.ssib:17', False], 'ssib_sup': ['SIBSUP', 'f4', '', 'SSiB short wave rad. up ', '', 'Registry/registry.ssib:18', True], 'ssib_ldn': ['SIBLDN', 'f4', '', 'SSiB long wave rad. down ', '', 'Registry/registry.ssib:19', False], 'ssib_lup': ['SIBLUP', 'f4', '', 'SSiB long wave rad. up ', '', 'Registry/registry.ssib:20', True], 'ssib_wat': ['SIBWAT', 'f4', '', 'SSiB total soil moisture content', '', 'Registry/registry.ssib:21', True], 'ssib_shc': ['SIBSHC', 'f4', '', 'SSiB sensible heat from canopy', '', 'Registry/registry.ssib:22', True], 'ssib_shg': ['SIBSHG', 'f4', '', 'SSiB sensible heat from ground', '', 'Registry/registry.ssib:23', True], 'ssib_lai': ['SIBLAI', 'f4', '', 'SSiB leaf area index', '', 'Registry/registry.ssib:24', True], 'ssib_vcf': ['SIBVCF', 'f4', '', 'SSiB vegetation cover', '', 'Registry/registry.ssib:25', True], 'ssib_z00': ['SIBZ00', 'f4', '', 'SSiB surface roughness length', '', 'Registry/registry.ssib:26', True], 'ssib_veg': ['SIBVEG', 'f4', '', 'SSiB vegetation map', '', 'Registry/registry.ssib:27', True], 'swe': ['SWE', 'f4', '', 'ssib-snow SWE', '', 'Registry/registry.ssib:31', False], 'snowden': ['SNOWDEN', 'f4', '', 'ssib-snow SNOWDEN', '', 'Registry/registry.ssib:32', False], 'snowdepth': ['SNOWDEPTH', 'f4', '', 'ssib-snow SNOWDEPTH', '', 'Registry/registry.ssib:33', False], 'tkair': ['TKAIR', 'f4', '', 'ssib-snow TKAIR', '', 'Registry/registry.ssib:34', False], 'dzo1': ['DZO1', 'f4', '', 'ssib-snow DZO1', '', 'Registry/registry.ssib:35', False], 'wo1': ['WO1', 'f4', '', 'ssib-snow WO1', '', 'Registry/registry.ssib:36', False], 'tssn1': ['TSSN1', 'f4', '', 'ssib-snow TSSN1', '', 'Registry/registry.ssib:37', False], 'tssno1': ['TSSNO1', 'f4', '', 'ssib-snow TSSNO1', '', 'Registry/registry.ssib:38', False], 'bwo1': ['BWO1', 'f4', '', 'ssib-snow BWO1', '', 'Registry/registry.ssib:39', False], 'bto1': ['BTO1', 'f4', '', 'ssib-snow BTO1', '', 'Registry/registry.ssib:40', False], 'cto1': ['CTO1', 'f4', '', 'ssib-snow CTO1', '', 'Registry/registry.ssib:41', False], 'fio1': ['FIO1', 'f4', '', 'ssib-snow FIO1', '', 'Registry/registry.ssib:42', False], 'flo1': ['FLO1', 'f4', '', 'ssib-snow FLO1', '', 'Registry/registry.ssib:43', False], 'bio1': ['BIO1', 'f4', '', 'ssib-snow BIO1', '', 'Registry/registry.ssib:44', False], 'blo1': ['BLO1', 'f4', '', 'ssib-snow BLO1', '', 'Registry/registry.ssib:45', False], 'ho1': ['HO1', 'f4', '', 'ssib-snow HO1', '', 'Registry/registry.ssib:46', False], 'dzo2': ['DZO2', 'f4', '', 'ssib-snow DZO2', '', 'Registry/registry.ssib:47', False], 'wo2': ['WO2', 'f4', '', 'ssib-snow WO2', '', 'Registry/registry.ssib:48', False], 'tssn2': ['TSSN2', 'f4', '', 'ssib-snow TSSN2', '', 'Registry/registry.ssib:49', False], 'tssno2': ['TSSNO2', 'f4', '', 'ssib-snow TSSNO2', '', 'Registry/registry.ssib:50', False], 'bwo2': ['BWO2', 'f4', '', 'ssib-snow BWO2', '', 'Registry/registry.ssib:51', False], 'bto2': ['BTO2', 'f4', '', 'ssib-snow BTO2', '', 'Registry/registry.ssib:52', False], 'cto2': ['CTO2', 'f4', '', 'ssib-snow CTO2', '', 'Registry/registry.ssib:53', False], 'fio2': ['FIO2', 'f4', '', 'ssib-snow FIO2', '', 'Registry/registry.ssib:54', False], 'flo2': ['FLO2', 'f4', '', 'ssib-snow FLO2', '', 'Registry/registry.ssib:55', False], 'bio2': ['BIO2', 'f4', '', 'ssib-snow BIO2', '', 'Registry/registry.ssib:56', False], 'blo2': ['BLO2', 'f4', '', 'ssib-snow BLO2', '', 'Registry/registry.ssib:57', False], 'ho2': ['HO2', 'f4', '', 'ssib-snow HO2', '', 'Registry/registry.ssib:58', False], 'dzo3': ['DZO3', 'f4', '', 'ssib-snow DZO3', '', 'Registry/registry.ssib:59', False], 'wo3': ['WO3', 'f4', '', 'ssib-snow WO3', '', 'Registry/registry.ssib:60', False], 'tssn3': ['TSSN3', 'f4', '', 'ssib-snow TSSN3', '', 'Registry/registry.ssib:61', False], 'tssno3': ['TSSNO3', 'f4', '', 'ssib-snow TSSNO3', '', 'Registry/registry.ssib:62', False], 'bwo3': ['BWO3', 'f4', '', 'ssib-snow BWO3', '', 'Registry/registry.ssib:63', False], 'bto3': ['BTO3', 'f4', '', 'ssib-snow BTO3', '', 'Registry/registry.ssib:64', False], 'cto3': ['CTO3', 'f4', '', 'ssib-snow CTO3', '', 'Registry/registry.ssib:65', False], 'fio3': ['FIO3', 'f4', '', 'ssib-snow FIO3', '', 'Registry/registry.ssib:66', False], 'flo3': ['FLO3', 'f4', '', 'ssib-snow FLO3', '', 'Registry/registry.ssib:67', False], 'bio3': ['BIO3', 'f4', '', 'ssib-snow BIO3', '', 'Registry/registry.ssib:68', False], 'blo3': ['BLO3', 'f4', '', 'ssib-snow BLO3', '', 'Registry/registry.ssib:69', False], 'ho3': ['HO3', 'f4', '', 'ssib-snow HO3', '', 'Registry/registry.ssib:70', False], 'dzo4': ['DZO4', 'f4', '', 'ssib-snow DZO4', '', 'Registry/registry.ssib:71', False], 'wo4': ['WO4', 'f4', '', 'ssib-snow WO4', '', 'Registry/registry.ssib:72', False], 'tssn4': ['TSSN4', 'f4', '', 'ssib-snow TSSN4', '', 'Registry/registry.ssib:73', False], 'tssno4': ['TSSNO4', 'f4', '', 'ssib-snow TSSNO4', '', 'Registry/registry.ssib:74', False], 'bwo4': ['BWO4', 'f4', '', 'ssib-snow BWO4', '', 'Registry/registry.ssib:75', False], 'bto4': ['BTO4', 'f4', '', 'ssib-snow BTO4', '', 'Registry/registry.ssib:76', False], 'cto4': ['CTO4', 'f4', '', 'ssib-snow CTO4', '', 'Registry/registry.ssib:77', False], 'fio4': ['FIO4', 'f4', '', 'ssib-snow FIO4', '', 'Registry/registry.ssib:78', False], 'flo4': ['FLO4', 'f4', '', 'ssib-snow FLO4', '', 'Registry/registry.ssib:79', False], 'bio4': ['BIO4', 'f4', '', 'ssib-snow BIO4', '', 'Registry/registry.ssib:80', False], 'blo4': ['BLO4', 'f4', '', 'ssib-snow BLO4', '', 'Registry/registry.ssib:81', False], 'ho4': ['HO4', 'f4', '', 'ssib-snow HO4', '', 'Registry/registry.ssib:82', False]}.items()}
# Retain the existing selector rows while adding SSiB and input echoes.
from woof.io.wrf_input_history_schema import (WRF_INPUT_HISTORY_ROWS,
                                              WRF_INITIALIZED_HISTORY_ROWS)
WRF_INPUT_ECHO_OUTPUT_FIELDS = {
    name: WrfOutputField(name, row["dtype"], row["stagger"], row["description"],
                         row["units"], row["registry"], wrf_history=True)
    for name, row in (WRF_INPUT_HISTORY_ROWS | WRF_INITIALIZED_HISTORY_ROWS).items()
}
# BMJ restart fields are also exposed as history diagnostics. WRF marks
# these rows restart-only; wrf_history=False preserves that provenance.
BMJ_OUTPUT_FIELDS = {
    key: WrfOutputField(key.upper(), "f4", "", description, units,
                        f"Registry.EM_COMMON:{line}", wrf_history=False)
    for key, description, units, line in (
        ("cldefi", "precipitation efficiency in BMJ SCHEME", "", 1852),
        ("qcconv", "CLOUD MIXING RATIO FOR BMJ CUMULUS SCHEME", "kg kg-1", 1577),
        ("qiconv", "ICE MIXING RATIO FOR BMJ CUMULUS SCHEME", "kg kg-1", 1578),
        ("convcld", "BMJ CONVECTIVE CLOUD", "kg m-2", 1700),
        ("ccldfra", "CONVECTIVE CLOUD FRACTION", "", 1701),
        ("htop", "TOP OF CONVECTION LEVEL", "", 1246),
        ("hbot", "BOT OF CONVECTION LEVEL", "", 1247),
        ("pratec", "PRECIP RATE FROM CUMULUS SCHEME", "mm s-1", 1584),
        ("raincv", "TIME-STEP CUMULUS PRECIPITATION", "mm", 1586),
    )
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
CAM_OUTPUT_FIELDS = {
    'swcf': WrfOutputField('SWCF','f4','','SHORT WAVE CLOUD FORCING AT TOA','W m-2','Registry.EM_COMMON:1837',False),
    'lwcf': WrfOutputField('LWCF','f4','','LONG WAVE CLOUD FORCING AT TOA','W m-2','Registry.EM_COMMON:1838',False),
    'coszr': WrfOutputField('COSZR','f4','','COSINE OF SOLAR ZENITH ANGLE','','phys/module_ra_cam.F:360',False),
    'cemiss': WrfOutputField('CEMISS','f4','','cloud emissivity for isccp','','phys/module_ra_cam.F:362',False),
    'taucldc': WrfOutputField('TAUCLDC','f4','','CLOUD OPTICAL THICKNESS FOR WATER','','Registry.EM_COMMON:1992',False),
    'taucldi': WrfOutputField('TAUCLDI','f4','','CLOUD OPTICAL THICKNESS FOR ICE','','Registry.EM_COMMON:1991',False),
    'rthratenlwc': WrfOutputField('RTHRATLWC','f4','','UNCOUPLED THETA TEND DUE TO CLEAR SKY LONG WAVE RAD','K s-1','Registry.EM_COMMON:1696',False),
    'rthratenswc': WrfOutputField('RTHRATSWC','f4','','UNCOUPLED THETA TEND DUE TO CLEAR SKY SHORT WAVE RAD','K s-1','Registry.EM_COMMON:1698',False),
}

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

from woof.core.misc_buckets import HISTORY as _BUCKET_HISTORY
BUCKET_OUTPUT_FIELDS = {key:WrfOutputField(name,'i4','',desc,units,
    'WRF v4.7.1 Registry.EM_COMMON:1581-1582,1797-1812',wrf_history=True)
    for key,(name,units,desc) in _BUCKET_HISTORY.items()}

from woof.core.physics_tendency_output import HISTORY as _TENDENCY_HISTORY
TENDENCY_OUTPUT_FIELDS = {key:WrfOutputField(name,'f4','',desc,units,
    f'WRF v4.7.1 Registry.EM_COMMON:{2142+offset}',wrf_history=True)
    for offset,(key,(name,units,desc)) in enumerate(_TENDENCY_HISTORY.items())}

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
SSTSKIN_OUTPUT_FIELDS = {
    "dtw": WrfOutputField("DTW", "f4", "", "WARM LAYER TEMP DIFF",
                          "C", "Registry.EM_COMMON:883", wrf_history=False),
}
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
PXLSM_OUTPUT_FIELDS = {
    key: _output_field(key.upper(), description, units,
                       f"Registry.EM_COMMON:{line}")
    for key, description, units, line in (
        ("t2obs", "2 M ANALYSIS TEMPERATURE USED BY PX SOIL NUDGING", "K", 1024),
        ("q2obs", "2 M ANALYSIS WATER VAPOR USED BY PX SOIL NUDGING", "kg kg-1", 1025),
        ("vegf_px", "PLEIM-XIU VEGETATION FRACTION", "", 1023),
        ("lai_px", "PLEIM-XIU LEAF AREA INDEX", "m-2/m-2", 1028),
        ("wwlt_px", "PLEIM-XIU WILTING SOIL MOISTURE", "m3 m-3", 1029),
        ("wfc_px", "PLEIM-XIU FIELD CAPACITY SOIL MOISTURE", "m3 m-3", 1030),
        ("wsat_px", "PLEIM-XIU SATURATED SOIL MOISTURE", "m3 m-3", 1031),
        ("clay_px", "PLEIM-XIU CLAY FRACTION", "", 1032),
        ("csand_px", "PLEIM-XIU COARSE SAND FRACTION", "", 1033),
        ("fmsand_px", "PLEIM-XIU FINE AND MEDIUM SAND FRACTION", "", 1034),
    )
}
from woof.io.cam5_schema import CAM_OUTPUT_ROWS
CAM5_OUTPUT_FIELDS = {
    key: WrfOutputField(name, "i4" if kind == "integer" else "f4",
        "" if stagger == "-" else stagger, description, units,
        f"Registry/{file}:{line}", wrf_history=("h" in flags))
    for key,(name,kind,stagger,description,units,file,line,flags) in CAM_OUTPUT_ROWS.items()
    if key not in CUMULUS_RATE_OUTPUT_FIELDS and key not in SCHEME_OUTPUT_FIELDS
    and key not in PRECIPITATION_OUTPUT_FIELDS
}
# CAM's source revision numbers these two native cumulus carriers one line
# earlier. Reuse the existing BMJ/OSAS writer rows for the shared variables;
# retain CAM's complete field set and the strict union check below.
for _key in ("pratec", "raincv"):
    if _key in CAM5_OUTPUT_FIELDS:
        CAM5_OUTPUT_FIELDS[_key] = BMJ_OUTPUT_FIELDS[_key]
for _group in (PXLSM_OUTPUT_FIELDS, CAM5_OUTPUT_FIELDS, BUCKET_OUTPUT_FIELDS, TENDENCY_OUTPUT_FIELDS, GD_OUTPUT_FIELDS, GSFCGCE4ICE_OUTPUT_FIELDS, SSTSKIN_OUTPUT_FIELDS, TEMF_OUTPUT_FIELDS, CARRIED_SURFACE_OUTPUT_FIELDS, NOAH_OUTPUT_FIELDS, RADIATION_BOUNDARY_OUTPUT_FIELDS, GFDLETA_OUTPUT_FIELDS, CAM_OUTPUT_FIELDS,
               CUMULUS_RATE_OUTPUT_FIELDS, BMJ_OUTPUT_FIELDS,
               PHYSICS_PRESSURE_OUTPUT_FIELDS,
               SURFACE_ENERGY_OUTPUT_FIELDS,
               PBL_RATE_OUTPUT_FIELDS,
               SCM_FORCING_OUTPUT_FIELDS,
               INACTIVE_DIAGNOSTIC_OUTPUT_FIELDS,
               MYJ_SURFACE_OUTPUT_FIELDS, MYNN_PBL_OUTPUT_FIELDS, BOULAC_PBL_OUTPUT_FIELDS, GBM_PBL_OUTPUT_FIELDS, EEPS_PBL_OUTPUT_FIELDS, MRF_PBL_OUTPUT_FIELDS, NOAHMP_OUTPUT_FIELDS, RUC_OUTPUT_FIELDS, SOLAR_ALBEDO_OUTPUT_FIELDS, OCEAN_OUTPUT_FIELDS, PWP_OUTPUT_FIELDS, OCEAN_DIAGNOSTIC_FIELDS, KEPS_OUTPUT_FIELDS, ACM2_OUTPUT_FIELDS, PX_SFCLAY_OUTPUT_FIELDS, KF_DIAGNOSTIC_OUTPUT_FIELDS, SSIB_OUTPUT_FIELDS):
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
    ("htop", "TOP OF CONVECTION LEVEL", "", 1246),
    ("hbot", "BOT OF CONVECTION LEVEL", "", 1247),
    ("gd_cloud", "CLOUD WATER/ICE MIXING RAIO IN GD CLOUD", "kg kg-1", 1653),
    ("gd_cloud2", "TEST for GD CLOUD", "kg kg-1", 1654),
    ("gd_cldfr", "GD CLOUD Fraction", " ? ", 1655),
    ("raincv_a", "taveragd TIME-STEP CUMULUS PRECIPITATION", "mm", 1656),
    ("raincv_b", "taveragd TIME-STEP CUMULUS PRECIPITATION", "mm", 1657),
    ("gd_cloud_a", "taveragd CLOUD WATER MIXING RAIO IN GD CLOUD", "kg kg-1", 1658),
    ("gd_cloud2_a", "taveragd cloud ice mix ratio in GD", "kg kg-1", 1659),
    ("qc_cu", "CLOUD WATER MIXING RATIO FROM A CU SCHEME", "kg kg-1", 1660),
    ("qi_cu", "CLOUD ICE MIXUNG RATIO FROM A CU SCHEME", "kg kg-1", 1661),
    ("cldfra_dp", "DEEP CONVECTIVE CLOUD FRACTION FROM KF", "", 1623),
    ("cldfra_sh", "SHALLOW CONVECTIVE CLOUD FRACTION FROM KF", "", 1624),
    ("cldfra_cu", "TOTAL CONVECTIVE CLOUD FRACTION; no WRF Registry counterpart", "1", 0),
):
    SCHEME_OUTPUT_FIELDS[_key] = WrfOutputField(_key.upper(), "f4", "", _description,
        _units, f"Registry.EM_COMMON:{_line}" if _line else "", False)


# KF-CuP retained diagnostics, published as restart-only history extensions.
# Logical scratch is encoded as int32 0/1; it has no WRF history row.
for _key, _dtype, _name, _description, _units, _line in (
    ('cldfra_cup', 'f4', 'CLDFRA_CUP', 'Cloud fraction due to CuP Scheme', 'fraction', 1367),
    ('wup_cup', 'f4', 'WUP_CUP', 'Updraft vertical velocity', 'm s-1', 1372),
    ('qc_ic_cup', 'f4', 'QC_IC_CUP', 'Cloud water due to CuP Scheme (in cumulus cloud)', 'kg kg-1', 1370),
    ('qndrop_ic_cup', 'f4', 'QNDROP_IC_CUP', 'Cloud droplet number due to CuP Scheme (in cumulus cloud)', '# kg-1', 1371),
    ('qc_iu_cup', 'f4', 'QC_IU_CUP', 'Cloud water due to CuP Scheme (in cumulus updraft)', 'kg kg-1', 1369),
    ('fcvt_qc_to_pr_cup', 'f4', 'FCVT_QC_TO_PR_CUP', 'Fraction of cloudwater converted to precip. as air rises thru an updraft layer', '--', 1379),
    ('fcvt_qc_to_qi_cup', 'f4', 'FCVT_QC_TO_QI_CUP', 'Fraction of cloudwater converted to cloudice as air rises thru an updraft layer', '--', 1380),
    ('fcvt_qi_to_pr_cup', 'f4', 'FCVT_QI_TO_PR_CUP', 'Fraction of cloudice converted to precip. as air rises thru an updraft layer', '--', 1381),
    ('mfup_cup', 'f4', 'MFUP_CUP', 'Updraft mass flux for shallow-cu in CuP Scheme', 'kg m-2 s-1', 1375),
    ('mfup_ent_cup', 'f4', 'MFUP_ENT_CUP', 'Updraft mass flux entrainment (across layer) for shallow-cu in CuP Scheme', 'kg m-2 s-1', 1376),
    ('mfdn_cup', 'f4', 'MFDN_CUP', 'Downdraft mass flux for shallow-cu in CuP Scheme', 'kg m-2 s-1', 1377),
    ('mfdn_ent_cup', 'f4', 'MFDN_ENT_CUP', 'Downdraft mass flux entrainment (across layer) for shallow-cu in CuP Scheme', 'kg m-2 s-1', 1378),
    ('updfra_cup', 'f4', 'UPDFRA_CUP', 'Updraft fractional area due to CuP Scheme', 'fraction', 1368),
    ('cldfratend_cup', 'f4', 'CLDFRATEND_CUP', 'Cloud fraction tendency due to CuP Scheme', 'fraction', 1366),
    ('lnterms', 'f4', 'LNTERMS', 'Liquid+ice+1 ln term for cloud fraction', '', 1383),
    ('cutop', 'f4', 'CUTOP', 'TOP OF CONVECTION LEVEL FROM CUMULUS PAR', '', 1250),
    ('cubot', 'f4', 'CUBOT', 'BOT OF CONVECTION LEVEL FROM CUMULUS PAR', '', 1251),
    ('shall', 'f4', 'SHALL', 'Cumulus type, 0=deep, 1=shallow, 2=none', '', 1360),
    ('slopesfc', 'f4', 'SLOPESFC', 'Slope of surface layer dTheta/dMixRatio', 'K', 1356),
    ('slopeez', 'f4', 'SLOPEEZ', 'Slope of entrainment layer dTheta/dMixRatio', 'K', 1357),
    ('sigmasfc', 'f4', 'SIGMASFC', 'Std. dev. of surface layer dTheta/dMixRatio', 'K kg kg-1', 1358),
    ('sigmaez', 'f4', 'SIGMAEZ', 'Std. dev. of entrainment layer dTheta/dMixRatio', 'K kg kg-1', 1359),
    ('taucloud', 'f4', 'TAUCLOUD', 'CuP cloud time scale for lifetime', '', 1361),
    ('tactive', 'f4', 'TACTIVE', 'CuP cloud formation time scale', '', 1362),
    ('tstar', 'f4', 'TSTAR', 'Boundary layer time scale', 's', 1382),
    ('lnint', 'f4', 'LNINT', 'Integrated ln term for cloud fraction', '', 1384),
    ('activefrac', 'f4', 'ACTIVEFRAC', 'Fraction of PDF the forms clouds', 'fraction', 1365),
    ('wcloudbase', 'f4', 'WCLOUDBASE', 'CuP cloud base vertical velocity', '', 1364),
    ('wact_cup', 'f4', 'WACT_CUP', 'CuP cloud base vertical velocity for activation', 'm s-1', 1373),
    ('wulcl_cup', 'f4', 'WULCL_CUP', 'CuP updraft vertical velocity at top of first cloud layer', 'm s-1', 1374),
    ('tcloud_cup', 'f4', 'TCLOUD_CUP', 'CuP cloud duration for modifying T,QV,chem,etc', '', 1363),
    ('cupflag', 'i4', 'CUPFLAG', 'CuP scheme activavted, T=yes, F=no', '', 1355),
    ('cu_act_flag', 'i4', 'CU_ACT_FLAG', 'CuP activation scratch flag', '', 1936),

):
    SCHEME_OUTPUT_FIELDS[_key] = WrfOutputField(_name, _dtype, "", _description,
        _units, f"Registry.EM_COMMON:{_line}", False)

# SAS cloud bounds are restart fields in stock WRF. Expose the retained
# values as history extensions as well, from the same persistent arrays.
# Alternative SAS schemes write the same WRF cloud-bound variables.
for _key in ("hbot", "htop"):
    SCHEME_OUTPUT_FIELDS["sas_" + _key] = SCHEME_OUTPUT_FIELDS[_key]
# OSAS native restart diagnostics, exposed as an explicit history extension.
for _key, _description, _units, _line in (
    ("pratec", "PRECIP RATE FROM CUMULUS SCHEME", "mm s-1", 1584),
    ("raincv", "TIME-STEP CUMULUS PRECIPITATION", "mm", 1586),
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

# MSKF module INOUT carriers. Registry r-only rows are history extensions.
# CU_UAF is passed as AINC_FRAC by module_cumulus_driver.F:1084.
MSKF_OUTPUT_FIELDS = {
    'cutop': WrfOutputField('CUTOP', 'f4', "", 'TOP OF CONVECTION LEVEL FROM CUMULUS PAR', '', 'Registry.EM_COMMON:1250', False),
    'cubot': WrfOutputField('CUBOT', 'f4', "", 'BOT OF CONVECTION LEVEL FROM CUMULUS PAR', '', 'Registry.EM_COMMON:1251', False),
    'rucuten': WrfOutputField('RUCUTEN', 'f4', "", 'X WIND TENDENCY DUE TO CUMULUS PARAMETERIZATION', 'm s-2', 'Registry.EM_COMMON:1565', False),
    'rvcuten': WrfOutputField('RVCUTEN', 'f4', "", 'Y WIND TENDENCY DUE TO CUMULUS PARAMETERIZATION', 'm s-2', 'Registry.EM_COMMON:1566', False),
    'rthcuten': WrfOutputField('RTHCUTEN', 'f4', "", 'THETA TENDENCY DUE TO CUMULUS SCHEME', 'K s-1', 'Registry.EM_COMMON:1567', False),
    'rqvcuten': WrfOutputField('RQVCUTEN', 'f4', "", 'Q_V TENDENCY DUE TO CUMULUS SCHEME', 'kg kg-1 s-1', 'Registry.EM_COMMON:1568', False),
    'rqrcuten': WrfOutputField('RQRCUTEN', 'f4', "", 'Q_R TENDENCY DUE TO CUMULUS SCHEME', 'kg kg-1 s-1', 'Registry.EM_COMMON:1569', False),
    'rqccuten': WrfOutputField('RQCCUTEN', 'f4', "", 'Q_C TENDENCY DUE TO CUMULUS SCHEME', 'kg kg-1 s-1', 'Registry.EM_COMMON:1570', False),
    'rqscuten': WrfOutputField('RQSCUTEN', 'f4', "", 'Q_S TENDENCY DUE TO CUMULUS SCHEME', 'kg kg-1 s-1', 'Registry.EM_COMMON:1571', False),
    'rqicuten': WrfOutputField('RQICUTEN', 'f4', "", 'Q_I TENDENCY DUE TO CUMULUS SCHEME', 'kg kg-1 s-1', 'Registry.EM_COMMON:1572', False),
    'w0avg': WrfOutputField('W0AVG', 'f4', "", 'AVERAGE VERTICAL VELOCITY FOR KF CUMULUS SCHEME', 'm s-1', 'Registry.EM_COMMON:1575', False),
    'pratec': WrfOutputField('PRATEC', 'f4', "", 'PRECIP RATE FROM CUMULUS SCHEME', 'mm s-1', 'Registry.EM_COMMON:1584', False),
    'raincv': WrfOutputField('RAINCV', 'f4', "", 'TIME-STEP CUMULUS PRECIPITATION', 'mm', 'Registry.EM_COMMON:1586', False),
    'nca': WrfOutputField('NCA', 'f4', "", 'COUNTER OF THE CLOUD RELAXATION TIME IN KF CUMULUS SCHEME', '', 'Registry.EM_COMMON:1619', False),
    'cldfra_dp': WrfOutputField('CLDFRA_DP', 'f4', "", 'DEEP CONVECTIVE CLOUD FRACTION FROM KF', '', 'Registry.EM_COMMON:1623', False),
    'cldfra_sh': WrfOutputField('CLDFRA_SH', 'f4', "", 'SHALLOW CONVECTIVE CLOUD FRACTION FROM KF', '', 'Registry.EM_COMMON:1624', False),
    'udr_kf': WrfOutputField('UDR_KF', 'f4', "", 'UPDRAFT DETRAINMENT RATE FROM KF', 'kg s-1', 'Registry.EM_COMMON:1625', True),
    'ddr_kf': WrfOutputField('DDR_KF', 'f4', "", 'DOWNDRAFT DETRAINMENT RATE FROM KF', 'kg s-1', 'Registry.EM_COMMON:1626', True),
    'uer_kf': WrfOutputField('UER_KF', 'f4', "", 'UPDRAFT ENTRAINMENT RATE FROM KF', 'kg s-1', 'Registry.EM_COMMON:1627', True),
    'der_kf': WrfOutputField('DER_KF', 'f4', "", 'DOWNDRAFT ENTRAINMENT RATE FROM KF', 'kg s-1', 'Registry.EM_COMMON:1628', True),
    'timec_kf': WrfOutputField('TIMEC_KF', 'f4', "", 'CONVECTIVE TIMESCALE FROM MSKF', 's', 'Registry.EM_COMMON:1629', True),
    'qc_cu': WrfOutputField('QC_CU', 'f4', "", 'CLOUD WATER MIXING RATIO FROM A CU SCHEME', 'kg kg-1', 'Registry.EM_COMMON:1660', False),
    'qi_cu': WrfOutputField('QI_CU', 'f4', "", 'CLOUD ICE MIXUNG RATIO FROM A CU SCHEME', 'kg kg-1', 'Registry.EM_COMMON:1661', False),
    'qr_cu': WrfOutputField('QR_CU', 'f4', "", 'RAIN MIXING RATIO FROM A CU SCHEME', 'kg kg-1', 'Registry.EM_COMMON:1662', True),
    'qs_cu': WrfOutputField('QS_CU', 'f4', "", 'SNOW MIXING RATIO FROM A CU SCHEME', 'kg kg-1', 'Registry.EM_COMMON:1663', True),
    'nc_cu': WrfOutputField('NC_CU', 'f4', "", 'CLOUD WATER NUMBER CONCENTRATION FROM A CU SCHEME', 'kg-1', 'Registry.EM_COMMON:1664', True),
    'ni_cu': WrfOutputField('NI_CU', 'f4', "", 'CLOUD ICE NUMBER CONCENTRATION FROM A CU SCHEME', 'kg-1', 'Registry.EM_COMMON:1665', True),
    'nr_cu': WrfOutputField('NR_CU', 'f4', "", 'RAIN NUMBER CONCENTRATION FROM A CU SCHEME', 'kg-1', 'Registry.EM_COMMON:1666', True),
    'ns_cu': WrfOutputField('NS_CU', 'f4', "", 'SNOW NUMBER CONCENTRATION FROM A CU SCHEME', 'kg-1', 'Registry.EM_COMMON:1667', True),
    'ccn_cu': WrfOutputField('CCN_CU', 'f4', "", 'CLOUD CONDENSATION NUCLEI CONCENTRATION FROM A CU SCHEME', 'kg-1', 'Registry.EM_COMMON:1668', True),
    'efcs': WrfOutputField('EFCS', 'f4', "", 'Sub-grid Scale Cloud Effective Radius', 'um', 'Registry.EM_COMMON:1670', True),
    'efis': WrfOutputField('EFIS', 'f4', "", 'Sub-grid Scale Ice Effective Radius', 'um', 'Registry.EM_COMMON:1671', True),
    'efss': WrfOutputField('EFSS', 'f4', "", 'Subgrid Scale Snow Effective Radius', 'um', 'Registry.EM_COMMON:1675', True),
    'ainc_frac': WrfOutputField('CU_UAF', 'f4', "", 'CU Updraft Area Fraction', '', 'Registry.EM_COMMON:1669', True),
    'cu_act': WrfOutputField('MSKF_CU_ACT', 'i4', "", 'MSKF active convective column flag', '', '', False),
}
# Shared native names keep the existing writer metadata.
for _key, _field in tuple(MSKF_OUTPUT_FIELDS.items()):
    MSKF_OUTPUT_FIELDS[_key] = OUTPUT_FIELDS_BY_NETCDF_NAME.get(_field.netcdf_name, _field)

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
NTU_OUTPUT_FIELDS: dict[str, WrfOutputField] = {
    'qdcn': WrfOutputField('QDCN', "f4", "", 'Mass mixing ratio for dry CN ', 'kg kg-1', "Registry/Registry.EM_COMMON:636", True),
    'qtcn': WrfOutputField('QTCN', "f4", "", 'Mass mixing ratio for total CN', 'kg kg-1', "Registry/Registry.EM_COMMON:638", True),
    'qccn': WrfOutputField('QCCN', "f4", "", 'Mass mixing ratio for cloud CN', 'kg kg-1', "Registry/Registry.EM_COMMON:640", True),
    'qrcn': WrfOutputField('QRCN', "f4", "", 'Mass mixing ratio for rain CN', 'kg kg-1', "Registry/Registry.EM_COMMON:642", True),
    'qnin': WrfOutputField('QNIN', "f4", "", 'Potential ice nuclei number mixing ratio', '  kg-1', "Registry/Registry.EM_COMMON:644", True),
    'fi': WrfOutputField('FI', "f4", "", 'Shape moment mixing ratio for pristine ice', 'm^3 kg-1', "Registry/Registry.EM_COMMON:646", True),
    'fs': WrfOutputField('FS', "f4", "", 'Shape moment mixing ratio for snow-aggregates', 'm^3 kg-1', "Registry/Registry.EM_COMMON:648", True),
    'vi': WrfOutputField('VI', "f4", "", 'Volume moment mixing ratio for pristine ice', 'm^3 kg-1', "Registry/Registry.EM_COMMON:650", True),
    'vs': WrfOutputField('VS', "f4", "", 'Volume moment mixing ratio for snow-aggragates', 'm^3 kg-1', "Registry/Registry.EM_COMMON:652", True),
    'vg': WrfOutputField('VG', "f4", "", 'Volume moment mixing ratio for rimed ice', 'm^3 kg-1', "Registry/Registry.EM_COMMON:654", True),
    'ai': WrfOutputField('AI', "f4", "", 'Second moment mixing ratio for pristine ice', 'm^2 kg-1', "Registry/Registry.EM_COMMON:656", True),
    'as_': WrfOutputField('AS', "f4", "", 'Second moment mixing ratio for snow-aggregates', 'm^2 kg-1', "Registry/Registry.EM_COMMON:658", True),
    'ag': WrfOutputField('AG', "f4", "", 'Second moment mixing ratio for rimed ice', 'm^2 kg-1', "Registry/Registry.EM_COMMON:660", True),
    'ah': WrfOutputField('AH', "f4", "", 'Second moment mixing ratio for hail', 'm^2 kg-1', "Registry/Registry.EM_COMMON:662", True),
    'i3m': WrfOutputField('I3M', "f4", "", 'Third moment mixing ratio for pristine ice', 'm^3 kg-1', "Registry/Registry.EM_COMMON:664", True),
}

HISTORY_FIELDS_BY_NETCDF_NAME: dict[str, WrfOutputField] = {}
from woof.io.sfire_schema import SFIRE_REGISTRY_FIELDS
SFIRE_OUTPUT_FIELDS = {
    name: WrfOutputField(name, NETCDF_DTYPE_BY_REGISTRY_TYPE[row[0]], row[1],
                         row[2], row[3], row[4], wrf_history=row[5])
    for name, row in SFIRE_REGISTRY_FIELDS.items()
}
for _group in (OUTPUT_FIELDS_BY_NETCDF_NAME, THOMPSON_AEROSOL_OUTPUT_FIELDS, NTU_OUTPUT_FIELDS,
               WRF_INPUT_ECHO_OUTPUT_FIELDS,
               SURFACE_IDENTITY_OUTPUT_FIELDS, SFIRE_OUTPUT_FIELDS, MSKF_OUTPUT_FIELDS):
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
    "CWM": ("Total condensate mixing ratio", "kg kg-1"),
    "F_ICE_PHY": ("FRACTION OF ICE", ""),
    "F_RAIN_PHY": ("FRACTION OF RAIN ", ""),
    "F_RIMEF_PHY": ("MASS RATIO OF RIMED ICE ", ""),
    "TKE": ("TURBULENCE KINETIC ENERGY", "m2 s-2"),
    "RUNDGDTEN": ("X WIND TENDENCY DUE TO FDDA GRID NUDGING", "m s-2"),
    "RVNDGDTEN": ("Y WIND TENDENCY DUE TO FDDA GRID NUDGING", "m s-2"),
    "RTHNDGDTEN": ("THETA TENDENCY DUE TO FDDA GRID NUDGING", "K s-1"),
    "RQVNDGDTEN": ("Q_V TENDENCY DUE TO FDDA GRID NUDGING", "kg kg-1 s-1"),
    "SDA_HFX": ("THETA TENDENCY AT THE FIRST MODEL LAYER", "K s-1"),
    "SDA_QFX": ("QV TENDENCY AT THE FIRST MODEL LAYER", "kg kg-1 s-1"),
    "QNORM": ("NORMALIZED QV FACTOR", "dimless"),
    "HFX_BOTH": ("UPWARD HEAT FLUX AT THE SURFACE", "W m-2"),
    "QFX_BOTH": ("UPWARD MOISTURE FLUX AT THE SURFACE", "kg m-2 s-1"),
    "HFX_FDDA": ("PSEUDO RADIATIVE HEAT FLUX FROM TEMPERATURE FDDA", "W m-2"),
    **{f"tr17_{index}": (f"tr17_{index}", "Dimensionless")
       for index in range(1, 9)},
    "THRUST": ("WIND TURBINE THRUST PER GRID CELL", "N"),
    "POWER": ("WIND TURBINE POWER PER GRID CELL", "W"),
    "TKE_WINDFARM": ("TURBINE ADDED TKE", "m2 s-2"),
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
    "FLX4": ("sensible heat from canopy", "W m-2"),
    "FVB": ("fraction of vegetation with snow below", ""),
    "FBUR": ("fraction of vegetation covered by snow", ""),
    "FGSN": ("fraction of ground covered by snow", ""),
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
    "GRAUPEL_MVD_MAX": ("Column maximum of graupel-hail median volume diameter", "mm"),
    "GRAUPEL_MVD": ("Median volume diameter of graupel-hail", "mm"),
    "GRAUPEL_DENSITY": ("Bulk density of graupel-hail", "kg m-3"),
    "QVGRAUPEL": ("Graupel Particle Volume", "m(3) kg(-1)"),
    "QVHAIL": ("Hail Particle Volume", "m(3) kg(-1)"),
    # P3 one-category (mp_physics=50).  Transcribed verbatim from
    # Registry.EM_COMMON:555-558, whose IO string ``i0rhusdf`` carries the
    # ``h`` that puts both in WRF's history stream.  They are what makes an
    # mp=50 wrfout more than a one-moment ice field: without the rime pair a
    # reader cannot recover rime fraction qir/qi or rime density qir/qib,
    # which are the two indices P3's whole ice inventory is built on.
    "v_ice": ("Mass-weighted ice fallspeed cat 1", "m s-1"),
    "d_ice": ("Mass-weighted mean ice size cat 1", "m"),
    "rho_ice": ("Mass-weighted mean ice density cat 1", "kg m-3"),
    "v_ice2": ("Mass-weighted ice fallspeed cat 2", "m s-1"),
    "d_ice2": ("Mass-weighted mean ice size cat 2", "m"),
    "rho_ice2": ("Mass-weighted mean ice density cat 2", "kg m-3"),
    "QICE2": ("Ice mixing ratio cat 2", "kg kg-1"),
    "QNICE2": ("Ice Number concentration cat 2", "  kg-1"),
    "QIR2": ("Rime ice mass-2 mixing ratio", "kg kg(-1)"),
    "QIB2": ("Rime ice volume-2 mixing ratio", "m(3) kg(-1)"),
    "QRIMEF": ("rime factor * qi", "kg kg-1"),
    "QIR": ("Rime ice mass-1 mixing ratio", "kg kg(-1)"),
    "QIB": ("Rime ice volume-1 mixing ratio", "m(3) kg(-1)"),
    # SBU-YLin (mp_physics=13), Registry.EM_COMMON:491 verbatim.
    "RIMI": ("riming intensity", "fraction"),
    "C1H": ("Hybrid dry-mass weight", ""),
    "C2H": ("Hybrid dry-mass offset", "Pa"),
    "QICE2": ("Adaptive-habit ice qi2", "kg kg-1"),
    "QICE3": ("Adaptive-habit ice qi3", "kg kg-1"),
    "QNICE2": ("Adaptive-habit ice ni2", "# kg-1"),
    "QNICE3": ("Adaptive-habit ice ni3", "# kg-1"),
    "QVOLI": ("Adaptive-habit ice qvoli", "m3 kg-1"),
    "QAOLI": ("Adaptive-habit ice qaoli", "m3 kg-1"),
    "QVOLI2": ("Adaptive-habit ice qvoli2", "m3 kg-1"),
    "QAOLI2": ("Adaptive-habit ice qaoli2", "m3 kg-1"),
    "QVOLI3": ("Adaptive-habit ice qvoli3", "m3 kg-1"),
    "QAOLI3": ("Adaptive-habit ice qaoli3", "m3 kg-1"),
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
    # Registry.EM_COMMON:68-69, verbatim.  Written by the slab scheme
    # (woof.io.history_layout), whose five layers are init_soil_depth_1's.
    "ZS": ("DEPTHS OF CENTERS OF SOIL LAYERS", "m"),
    "DZS": ("THICKNESSES OF SOIL LAYERS", "m"),
    "SMOIS": ("SOIL MOISTURE", "m3 m-3"),
    "SH2O": ("SOIL LIQUID WATER", "m3 m-3"),
    # WRF v4.7.1 Registry.EM_COMMON CANWAT has the history flag h.
    "CANWAT": ("CANOPY WATER", "kg m-2"),
    "SWDOWN": ("DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE", "W m-2"),
    "ECOBSC": ("SOLAR ECLIPSE OBSCURATION FRACTION", "1"),
    "ELAT_TRACK": ("ECLIPSE LAT", "degrees"),
    "ELON_TRACK": ("ECLIPSE LON", "degrees"),
    "ECMASK": ("ECLIPSE MASK: 0 NONE, 1 PARTIAL, 2 TOTAL, 3 ANNULAR", "1"),
    "GLW": ("DOWNWARD LONG WAVE FLUX AT GROUND SURFACE", "W m-2"),
    # Registry.EM_COMMON:1839, transcribed verbatim beside its two
    # surface-flux neighbours.  ``ij`` mass grid, no stagger, real -- so
    # the dimension table and the FieldType default already describe it
    # correctly and it needs no row of its own.
    "OLR": ("TOA OUTGOING LONG WAVE", "W m-2"),
    'SWDOWN2': ('FARMS global horizontal surface irradiance', 'W m-2'),
    'SWDDIR2': ('FARMS direct horizontal surface irradiance', 'W m-2'),
    'SWDDNI2': ('FARMS direct normal surface irradiance', 'W m-2'),
    'SWDDIF2': ('FARMS diffuse horizontal surface irradiance', 'W m-2'),
    'SWDOWNC2': ('FARMS clear sky global horizontal surface irradiance', 'W m-2'),
    'SWDDNIC2': ('FARMS clear sky direct normal surface irradiance', 'W m-2'),
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
    "TAUCLDC": ("CLOUD WATER OPTICAL DEPTH", ""),
    "TAUCLDI": ("CLOUD ICE OPTICAL DEPTH", ""),
    "RSWTOA": ("SHORTWAVE FLUX AT TOP OF ATMOSPHERE", "W m-2"),
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


from woof.avgflx_contract import HISTORY as _AVGFLX_HISTORY
AVGFLX_OUTPUT_FIELDS = {}
for (_slot, (_symbol, _units, _description)), _stagger in zip(_AVGFLX_HISTORY.items(), "XYZ"):
    _field = WrfOutputField(_symbol, "f4", _stagger, _description, _units,
                            "WRF v4.6.1 Registry/registry.avgflx", wrf_history=True)
    HISTORY_FIELDS_BY_NETCDF_NAME[_symbol] = _field
    AVGFLX_OUTPUT_FIELDS[_symbol] = _field
    REGISTRY_VAR_META[_symbol] = (_description, _units)

from woof.core.output_diagnostics import HISTORY as _OUTPUT_HISTORY
REGISTRY_VAR_META.update({symbol: (description, units)
    for symbol, units, description in _OUTPUT_HISTORY.values()})
from woof.core.hail import HISTORY as _HAIL_HISTORY
REGISTRY_VAR_META.update({symbol: (description, units)
    for symbol, units, description in _HAIL_HISTORY.values()})

__all__ = [
    "AVGFLX_OUTPUT_FIELDS",

    "ACM2_OUTPUT_FIELDS",
    "HISTORY_FIELDS_BY_NETCDF_NAME",
    "MYNN_PBL_OUTPUT_FIELDS", "GBM_PBL_OUTPUT_FIELDS",
    "NETCDF_DTYPE_BY_REGISTRY_TYPE",
    "NOAHMP_OUTPUT_FIELDS",
    "OUTPUT_FIELDS_BY_NETCDF_NAME",
    "PHYSICS_SELECTOR_GLOBALS",
    "PRECIPITATION_OUTPUT_FIELDS",
    "RADIATION_CLDFRA_ATTRIBUTES",
    "REGISTRY_VAR_META",
    "RUC_OUTPUT_FIELDS",
    "SCHEME_OUTPUT_FIELDS",
    "TEMF_OUTPUT_FIELDS",
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
    _schema = WrfOutputField(_external, "f4", "",
                            f"{_external}; no WRF Registry counterpart", _units, "", False)
    SCHEME_OUTPUT_FIELDS[_key] = _schema
    OUTPUT_FIELDS_BY_NETCDF_NAME[_external] = _schema
    HISTORY_FIELDS_BY_NETCDF_NAME[_external] = _schema

# WRF Registry.EM_COMMON:1672-1683.
for _n, _sat in enumerate(("0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0"), 1):
    REGISTRY_VAR_META[f"CCN{_n}_GS"] = (f"Grid Scale Cloud Condensation Nuclei at S={_sat}%", "#/cm-3")
REGISTRY_VAR_META.update({"EFCG": ("Grid Scale Cloud Effective Radius", "um"), "EFIG": ("Grid Scale Ice Effective Radius", "um"), "EFSG": ("Grid Scale Snow Effective Radius", "um"), "WACT": ("Aerosol Activation Updraft", "m s-1")})

def history_field_schema(name, *, sf_surface_physics=0):
    """Resolve schema aliases through the selected column metadata."""
    from woof.physics_registry import column_surface_contract
    contract = column_surface_contract(sf_surface_physics)
    override = (contract or {}).get('history_schema_overrides', {}).get(name)
    if override is None:
        return HISTORY_FIELDS_BY_NETCDF_NAME.get(name)
    row = WrfOutputField(*override)
    if row.netcdf_name != name:
        raise ValueError('history schema override names a different variable')
    return row
