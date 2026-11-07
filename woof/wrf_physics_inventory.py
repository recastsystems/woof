"""Declarative stock-WRF initialization inventories for RW-WPS.

This module deliberately answers a narrower question than woof's forecast
physics dispatcher: which package fields must exist when RW-WPS writes a
``wrfinput_dNN`` for an unchanged WRF v4.6.1 executable?  The inventory is
derived from ``Registry/Registry.EM_COMMON`` package declarations and field
I/O flags.  It must not be inferred from the schemes implemented by woof.

WRF Registry dimensions ``ikjftb`` become the NetCDF dimensions below.  WRF
``real.exe`` initializes package hydrometeors/moments to zero when the source
analysis does not provide them; water vapour is populated from the source.
Package ``state:`` auxiliaries whose Registry flags contain ``r`` but not
``i`` are runtime/restart state and are intentionally not invented as
``wrfinput`` variables.  The converse half of that rule first bites at
mp_physics=28: ``qnwfa2d``/``qnifa2d`` are ``state:`` members whose flags DO
carry ``i0``, so they are inventoried as ``wrfinput`` fields (with 2-D
dimensions) rather than as runtime state.  See :func:`_aerosol_emission`.

Units are the value WRF's Registry parser RESOLVES, not the text of the
Registry line; see :data:`WRF_RESOLVED_UNITS_NUMBER_PAREN`, which also
records the one place this module is knowingly inconsistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEMA = "rw-wps.stock-wrf-physics-inventory.v1"

#: The thermodynamic representation every stock-WRF export declares and
#: holds: dry potential temperature, WRF's ``use_theta_m = 0``.  The engine
#: integrates dry theta, so the prepared cache the exporter reads holds dry
#: theta in its state and mass-coupled dry theta in its lateral-boundary
#: tables (woof/ingest/lateral_bc.py, ``_coupled_device_fields``).  The
#: export (woof/wrf_direct.py) writes exactly that: ``USE_THETA_M = 0`` on
#: both files, ``THM`` equal to ``T`` (what real.exe writes under
#: ``use_theta_m = 0``, tests/data/real_em_461_theta_seam_west_strip.
#: README.md), and ``T_B*`` straight from the tables.
#:
#: Until this constant existed the pair was mixed: the header said
#: ``USE_THETA_M = 1`` and ``THM`` held moist theta, while ``T_B*`` held
#: the same dry-coupled tables.  WRF couples its prognostic into ``T_B*``
#: (main/real_em.F:872), so a ``use_theta_m = 1`` wrf.exe read those rows
#: as moist theta and its specified boundary came out colder than the
#: analysis by theta * Rv/Rd * qv / (1 + Rv/Rd * qv) from the first step
#: (measured with stock WRF V4.6.1: 2.887 K at qv = 4 g/kg, the prediction
#: to 4e-5 K; the same pair written dry holds the boundary to 3e-5 K).
#:
#: It lives here, beside the other facts a stock wrfinput must satisfy,
#: because this module imports nothing: every stock-WRF namelist the tree
#: renders for an export reads its ``use_theta_m`` from this name, and the
#: routes refuse a stock namelist that says otherwise.  WRF's input gate
#: (share/input_wrf.F) stops with "use_theta_m values must be consistent"
#: when the namelist and the file disagree.
EXPORT_USE_THETA_M = 0

WRFINPUT_3D_DIMS = (
    "Time",
    "bottom_top",
    "south_north",
    "west_east",
)
#: Registry dimension spec ``ij``.  Needed by mp_physics=28, the first
#: inventoried package with a 2-D member that carries an input-stream flag.
WRFINPUT_2D_DIMS = (
    "Time",
    "south_north",
    "west_east",
)

#: What WRF actually stores as a field's ``units``, which is NOT the text of
#: the Registry line.  ``tools/reg_parse.c:203-208`` walks every raw Registry
#: line before tokenizing and, for a ``#`` INSIDE double quotes, replaces it
#: with a blank (``:206``, ``else if ( *p == '#' && inquote ) *p = ' ' ;``);
#: a ``#`` outside quotes starts a comment (``:207``).  So the Registry text
#: ``"# kg(-1)"`` becomes ``'  kg(-1)'`` -- two leading blanks, one from the
#: blanked ``#`` and one that was already there -- and that string is what
#: WRF's generated tables carry and what its own writer emits as the NetCDF
#: ``units`` attribute.  Inside the generated ``mp_physics(idomain)==28``
#: block (``inc/scalar_indices.inc:2449-2618``) the six scalar members are
#: spelled ``'  kg-1'`` for ``P_qni`` (:2544) and ``'  kg(-1)'`` for
#: ``P_qnr`` (:2558), ``P_qnc`` (:2572), ``P_qnwfa`` (:2586), ``P_qnifa``
#: (:2600) and ``P_qnbca`` (:2614); the table is consumed as "! Units" at
#: ``inc/wrf_bdyout.inc:1449``.  woof's own WRF-output authority already
#: agrees: ``woof/io/wrfout.py::_VAR_META`` was verified against a stock
#: v4.6.1 wrfout and spells QNCLOUD/QNRAIN ``"  kg(-1)"`` and QNICE
#: ``"  kg-1"``.
#:
#: KNOWN DIVERGENCE, stated rather than hidden: the mp=6/8/10/18 rows below
#: predate this finding and carry the pre-parse Registry spelling
#: (``"# kg(-1)"``).  They are left byte-identical here on purpose -- they
#: are consumed by a shipped compatibility report and are not this change's
#: subject -- so the same NetCDF variable (QNRAIN) is spelled two ways in
#: this module depending on the package.  Only the mp=28 row is
#: authoritative on units.
WRF_RESOLVED_UNITS_NUMBER_PAREN = "  kg(-1)"
WRF_RESOLVED_UNITS_NUMBER_PLAIN = "  kg-1"


#: Registry members stock WRF allocates for a package that woof has no
#: species for.  They are still WRITTEN -- real.exe writes them too, at
#: zero, and a stock file missing a registered member is not a stock file --
#: but there is no prepared state to look for, so the exporter must not go
#: looking and must not report the absence as a gap.
_NO_GPUWM_SPECIES = frozenset({"qnbca"})


@dataclass(frozen=True)
class WrfInputField:
    """One package member written to stock-WRF initialization files."""

    registry_name: str
    netcdf_name: str
    collection: str
    dtype: str = "float32"
    dimensions: tuple[str, ...] = WRFINPUT_3D_DIMS
    units: str = "kg kg-1"
    initialization: str = "zero_if_source_absent"
    #: The woof state attribute holding this member's prepared values,
    #: when it differs from WRF's Registry name, and ``None`` when woof
    #: has no such species at all (the member is still written, at zero,
    #: because stock WRF registers it for the package).
    #:
    #: It exists because the exporter used to read
    #: ``state/{registry_name}`` and woof drops WRF's leading ``q`` on
    #: the number moments of Thompson (mp=8), Morrison (mp=10),
    #: aerosol-aware Thompson (mp=28) and P3 (mp=50): their qnc/qnr/qni/
    #: qns/qng/qnwfa/qnifa/qnwfa2d/qnifa2d are nc/nr/ni/ns/ng/nwfa/nifa/
    #: nwfa2d/nifa2d in the prepared cache, so every one of them missed
    #: and exported zeros, even where the cold start had seeded them
    #: with real.exe's make_RainNumber and make_IceNumber or the WIF
    #: climatology lane had filled them.  A transform ("strip a q") would
    #: be wrong for NSSL's qnr/qni/qns/qng/qnh/qndrop/qnn/qvolg/qvolh and
    #: P3's qir/qib, which DO exist under their Registry names, so the
    #: exception is declared per row rather than computed.
    state_name: str | None = None

    @property
    def state_key(self) -> str | None:
        """Prepared-cache key for this member, or ``None`` when woof has none."""

        if self.state_name is None and self.registry_name in _NO_GPUWM_SPECIES:
            return None
        return self.state_name or self.registry_name


@dataclass(frozen=True)
class RuntimeStateField:
    """Scheme state allocated by WRF but not required in wrfinput."""

    registry_name: str
    netcdf_name: str
    dtype: str = "float32"
    dimensions: tuple[str, ...] = WRFINPUT_3D_DIMS
    initialization: str = "wrf_runtime"


@dataclass(frozen=True)
class StockWrfPhysicsInventory:
    mp_physics: int
    scheme: str
    registry_package: str
    wrfinput_fields: tuple[WrfInputField, ...]
    runtime_state_not_wrfinput: tuple[RuntimeStateField, ...]
    registry_authority: str = "WRF-v4.6.1 Registry/Registry.EM_COMMON"

    def as_report(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "target": "stock_wrf_v4.6.1",
            "mp_physics": self.mp_physics,
            "scheme": self.scheme,
            "registry_package": self.registry_package,
            "registry_authority": self.registry_authority,
            "wrfinput_fields": [asdict(field) for field in self.wrfinput_fields],
            "runtime_state_not_wrfinput": [
                asdict(field) for field in self.runtime_state_not_wrfinput
            ],
        }


def _moist(name: str, netcdf_name: str) -> WrfInputField:
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="moist",
        initialization=(
            "source_specific_humidity" if name == "qv" else "zero_if_source_absent"
        ),
    )


def _scalar(
        name: str, netcdf_name: str, *, units: str = "# kg-1",
        state_name: str | None = None,
) -> WrfInputField:
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="scalar",
        units=units,
        state_name=state_name,
    )


_ICE_MASS = (
    _moist("qv", "QVAPOR"),
    _moist("qc", "QCLOUD"),
    _moist("qr", "QRAIN"),
    _moist("qi", "QICE"),
    _moist("qs", "QSNOW"),
    _moist("qg", "QGRAUP"),
)

_EFFECTIVE_RADII = (
    RuntimeStateField("re_cloud", "RE_CLOUD"),
    RuntimeStateField("re_ice", "RE_ICE"),
    RuntimeStateField("re_snow", "RE_SNOW"),
)


def _aerosol_emission(name: str, netcdf_name: str, *,
                      state_name: str | None = None) -> WrfInputField:
    """One of mp=28's two 2-D surface aerosol emission members.

    These are ``state:`` members of the ``thompsonaero`` package rather than
    ``scalar:`` members, and the module's own rule ("Package ``state:``
    auxiliaries whose Registry flags contain ``r`` but not ``i`` are
    runtime/restart state") therefore has to be applied and not assumed.
    ``Registry/Registry.EM_COMMON:492-493`` gives them the I/O string
    ``i01{17}rhdu`` -- an ``i`` list that BEGINS with stream 0, the same
    ``i0`` prefix carried by unambiguous wrfinput members such as HGT
    (:1407 ``i012rh056dus``) and TSK (:1417).  So unlike ``re_cloud`` /
    ``re_ice`` / ``re_snow`` (:497-499, bare ``r``) and ``taod5503d`` /
    ``taod5502d`` (:1738-1739, ``r`` and ``rh``), these two ARE
    initialization-file variables and belong in ``wrfinput_fields``.

    ``real.exe`` writes them: ``dyn_em/module_initialize_real.F:4496-4653``
    is the 2-D counterpart of the 3-D aerosol block, and its
    ``aer_init_opt = 0`` arm sets both to exactly ``0.0`` (:4501-4510) with
    the comment "Surface emissions of QNWFA will be computed in
    microphysics" -- which is ``thompson_init`` deriving ``nwfa2d`` from the
    synthetic CCN profile at ``phys/module_mp_thompson.F:509-510``.
    ``nifa2d`` has no such derivation anywhere in that file and stays zero.
    """
    return WrfInputField(
        registry_name=name,
        netcdf_name=netcdf_name,
        collection="state",
        dimensions=WRFINPUT_2D_DIMS,
        units="kg-1 s-1",
        state_name=state_name,
    )


#: P3's moist list, and it is a SUBSET of :data:`_ICE_MASS` rather than an
#: extension of it.  ``Registry.EM_COMMON:3038`` declares
#: ``moist:qv,qc,qr,qi`` for ``p3_1category`` -- there is no ``qs`` and no
#: ``qg``, because P3 carries ONE ice category whose rime mass and rime
#: volume (``qir``/``qib``) span the graupel-to-snow continuum instead of
#: splitting it into species.  Every other inventoried package so far has
#: been WSM6's six plus additions, so this is the first row where the
#: package declares FEWER moist members than the frozen WSM6 contract
#: carries; see ``woof/wrf_direct.py::_physics_contract_bundle``, which
#: prunes what the package does not declare.
_P3_MASS = (
    _moist("qv", "QVAPOR"),
    _moist("qc", "QCLOUD"),
    _moist("qr", "QRAIN"),
    _moist("qi", "QICE"),
)


from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES

_INVENTORIES = {
    96: StockWrfPhysicsInventory(
        mp_physics=96, scheme="MAD-WRF", registry_package="madwrf_mp",
        wrfinput_fields=(_moist("qv", "QVAPOR"), _moist("qc", "QCLOUD"),
                         _moist("qi", "QICE"), _moist("qs", "QSNOW")),
        runtime_state_not_wrfinput=()),
    # WRF v4.6.1 Registry.EM_COMMON:3016, no scalar or radius fields.
    2: StockWrfPhysicsInventory(mp_physics=2,scheme="Purdue Lin",
        registry_package="linscheme",wrfinput_fields=_ICE_MASS,
        runtime_state_not_wrfinput=()),
    # WRF Registry/registry.sbm:318; the bin members use lowercase NetCDF names.
    30: StockWrfPhysicsInventory(
        mp_physics=30,scheme="FAST_SBM",registry_package="fast_khain_lynn_shpund",
        wrfinput_fields=_ICE_MASS+tuple(
            _scalar(name,name,units="  kg-1" if name.startswith("ff8") else "  kg kg-1",state_name=name)
            for name in _FSBM_BIN_SPECIES)+tuple(
            _scalar(name,netcdf,units="Microns",state_name=target)
            for name,netcdf,target in (("effr","EFF_RADIUS","effc"),("ice_effr","ICE_EFF_RADIUS","effi"),("tot_effr","TOT_EFF_RADIUS","effs")))+tuple(
            _scalar("q"+name,netcdf,units="# kg-1" if name=="ni" else "# kg(-1)",state_name=name)
            for name,netcdf in (("nn","QNCCN"),("nc","QNCLOUD"),("nr","QNRAIN"),
                                ("ni","QNICE"),("ns","QNSNOW"),("ng","QNGRAUPEL"))),
        runtime_state_not_wrfinput=(RuntimeStateField("th_old","TH_OLD"),
                                   RuntimeStateField("qv_old","QV_OLD"),
                                   RuntimeStateField("tempc","TEMPC"),
                                   RuntimeStateField("height","HEIGHT")),
        registry_authority="WRF-v4.6.1 Registry/registry.sbm:318",
    ),
    # WRF 4.6.1 Registry.EM_COMMON:3019. qt is the scalar condensate.
    5: StockWrfPhysicsInventory(
        mp_physics=5, scheme="Ferrier-Aligo", registry_package="fer_mp_hires",
        wrfinput_fields=_P3_MASS + (_scalar("qt", "CWM"),),
        runtime_state_not_wrfinput=(
            RuntimeStateField("f_ice_phy", "F_ICE_PHY"),
            RuntimeStateField("f_rain_phy", "F_RAIN_PHY"),
            RuntimeStateField("f_rimef_phy", "F_RIMEF_PHY"),
        ),
    ),
    # Registry.EM_COMMON:3018, WSM5 has no graupel.
    4: StockWrfPhysicsInventory(
        mp_physics=4, scheme="WSM5", registry_package="wsm5scheme",
        wrfinput_fields=tuple(f for f in _ICE_MASS if f.registry_name != "qg"),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # WRF v4.6.1 Registry.EM_COMMON:3034, wsm7scheme:
    # moist:qv,qc,qr,qi,qs,qg,qh; state:re_cloud,re_ice,re_snow.
    24: StockWrfPhysicsInventory(
        mp_physics=24, scheme="WSM7", registry_package="wsm7scheme",
        wrfinput_fields=_ICE_MASS + (_moist("qh", "QHAIL"),),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    13: StockWrfPhysicsInventory(
        mp_physics=13,
        scheme="MP13 five-class",
        registry_package="sbu_ylinscheme",
        wrfinput_fields=tuple(f for f in _ICE_MASS if f.registry_name != "qg") + (
            WrfInputField(registry_name="rimi", netcdf_name="RIMI",
                          collection="state", units="fraction"),),
        runtime_state_not_wrfinput=(),
    ),
    # WRF v4.6.1 Registry.EM_COMMON:3035. Hail is mass-only; WDM's
    # CCN/cloud/rain numbers use the existing nn/nc/nr state names.
    26: StockWrfPhysicsInventory(
        mp_physics=26, scheme="WDM7", registry_package="wdm7scheme",
        wrfinput_fields=_ICE_MASS + (_moist("qh", "QHAIL"),
            _scalar("qnn", "QNCCN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nn"),
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nc"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr")),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII),
    27: StockWrfPhysicsInventory(
        mp_physics=27, scheme="UDM", registry_package="udmscheme",
        wrfinput_fields=_ICE_MASS + (_moist("qh", "QHAIL"),
            _scalar("qnn", "QNCCN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nn"),
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nc"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr")),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII),

    7: StockWrfPhysicsInventory(
        mp_physics=7, scheme='Goddard four-ice', registry_package='nuwrf4icescheme',
        wrfinput_fields=_ICE_MASS + (_moist('qh', 'QHAIL'),),
        runtime_state_not_wrfinput=tuple(RuntimeStateField(name, name.upper())
            for name in ('phys_tot', 'physc', 'physe', 'physd', 'physs', 'physm', 'physf',
                         'acphys_tot', 'acphysc', 'acphyse', 'acphysd', 'acphyss', 'acphysm', 'acphysf',
                         'preci3d', 'precs3d', 'precg3d', 'precr3d', 'prech3d',
                         're_cloud_gsfc', 're_rain_gsfc', 're_ice_gsfc', 're_snow_gsfc',
                         're_graupel_gsfc', 're_hail_gsfc', 're_cloud', 're_ice', 're_snow')),
    ),
    15: StockWrfPhysicsInventory(
        mp_physics=15, scheme="Ferrier-Aligo advected rime",
        registry_package="fer_mp_hires_advect",
        wrfinput_fields=_P3_MASS + (_scalar("qrimef", "QRIMEF", units="kg kg-1"),),
        runtime_state_not_wrfinput=()),
    # Registry.EM_COMMON:529-530 and :3045, with runtime-only phase state.
    95: StockWrfPhysicsInventory(mp_physics=95, scheme="Eta Ferrier",
        registry_package="etampnew",
        wrfinput_fields=(_moist("qv", "QVAPOR"), _moist("qc", "QCLOUD"),
                         _moist("qr", "QRAIN"), _moist("qs", "QSNOW"),
                         _scalar("qt", "CWM", units="kg kg-1")),
        runtime_state_not_wrfinput=(RuntimeStateField("f_ice_phy", "F_ICE_PHY"),
            RuntimeStateField("f_rain_phy", "F_RAIN_PHY"),
            RuntimeStateField("f_rimef_phy", "F_RIMEF_PHY"))),

    # Registry.EM_COMMON:3021
    6: StockWrfPhysicsInventory(
        mp_physics=6,
        scheme="WSM6",
        registry_package="wsm6scheme",
        wrfinput_fields=_ICE_MASS,
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # Registry.EM_COMMON:3022.  wsm6rscheme declares moist:qv,qc,qr,qi,qs,qg
    # and NOTHING else: no re_cloud/re_ice/re_snow state, because WRF's
    # wsm6r computes no radius (woof's radii for it are its own diagnosis).
    106: StockWrfPhysicsInventory(
        mp_physics=106,
        scheme="WSM6R",
        registry_package="wsm6rscheme",
        wrfinput_fields=_ICE_MASS,
        runtime_state_not_wrfinput=(),
    ),
    # Registry.EM_COMMON:3024.  Thompson's package includes both qni and qnr.
    8: StockWrfPhysicsInventory(
        mp_physics=8,
        scheme="Thompson",
        registry_package="thompson",
        wrfinput_fields=_ICE_MASS + (
            _scalar("qni", "QNICE", state_name="ni"),
            _scalar("qnr", "QNRAIN", units="# kg(-1)", state_name="nr"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # Registry.EM_COMMON:3026.  Do not collapse Morrison moments into mass.
    # Public WRF v4.7.1 Registry.EM_COMMON:3031; no graupel category.
    11: StockWrfPhysicsInventory(
        mp_physics=11, scheme="CAM Morrison-Gettelman", registry_package="cammgmpscheme",
        wrfinput_fields=tuple(f for f in _ICE_MASS if f.registry_name != "qg") + (
            _scalar("qnc", "QNCLOUD", state_name="nc"),
            _scalar("qni", "QNICE", state_name="ni"),
            _scalar("qnr", "QNRAIN", state_name="nr"),
            _scalar("qns", "QNSNOW", state_name="ns")),
        runtime_state_not_wrfinput=tuple(RuntimeStateField(n, n.upper()) for n in (
            "rh_old_mp","lcd_old_mp","cldfra_old_mp","cldfra_mp","cldfra_mp_all",
            "cldfra_conv","cldfrai","cldfral","wsedl3d","lradius","iradius")),
    ),
    10: StockWrfPhysicsInventory(
        mp_physics=10,
        scheme="Morrison two-moment",
        registry_package="morr_two_moment",
        wrfinput_fields=_ICE_MASS + (
            _scalar("qni", "QNICE", state_name="ni"),
            _scalar("qns", "QNSNOW", units="# kg(-1)", state_name="ns"),
            _scalar("qnr", "QNRAIN", units="# kg(-1)", state_name="nr"),
            _scalar("qng", "QNGRAUPEL", units="# kg(-1)", state_name="ng"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("rqrcuten", "RQRCUTEN"),
            RuntimeStateField("rqscuten", "RQSCUTEN"),
            RuntimeStateField("rqicuten", "RQICUTEN"),
        ),
    ),
    # Registry.EM_COMMON:3030, five masses and three transported numbers.
    14: StockWrfPhysicsInventory(
        mp_physics=14, scheme="WDM5", registry_package="wdm5scheme",
        wrfinput_fields=tuple(f for f in _ICE_MASS if f.registry_name != "qg") + (
            _scalar("qnn", "QNCCN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nn"),
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nc"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nr"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # Registry.EM_COMMON:3033,3049,3052,3054,3056 after WRF's option-18
    # selector defaults resolve to two moments, hail, predicted CCN, and
    # predicted graupel/hail volume (density).
    18: StockWrfPhysicsInventory(
        mp_physics=18,
        scheme="NSSL-2",
        registry_package=(
            "nssl_2mom+nssl2mconc+nssl_hail+nssl_ccn_opt+nssl_hailvol"
        ),
        wrfinput_fields=_ICE_MASS + (
            _moist("qh", "QHAIL"),
            _scalar("qndrop", "QNDROP"),
            _scalar("qnr", "QNRAIN", units="# kg(-1)"),
            _scalar("qni", "QNICE"),
            _scalar("qns", "QNSNOW", units="# kg(-1)"),
            _scalar("qng", "QNGRAUPEL", units="# kg(-1)"),
            _scalar("qnh", "QNHAIL", units="# kg(-1)"),
            _scalar("qnn", "QNCCN", units="# kg(-1)"),
            _scalar("qvolg", "QVGRAUPEL", units="m(3) kg(-1)"),
            _scalar("qvolh", "QVHAIL", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII,
    ),
    # Registry.EM_COMMON:3036.  Aerosol-aware Thompson.  Its moist list is
    # character for character mp=8's (:3024); everything new is a scalar or
    # state member.
    #
    # SIX scalar members, not five.  ``qnbca`` is declared by the
    # ``thompsonaero`` package itself and is therefore registered for EVERY
    # mp_physics=28 run, not only for wif_input_opt=2 -- WRF's generated
    # code proves it directly: the ``mp_physics(idomain)==28`` block at
    # inc/scalar_indices.inc:2449-2618 ends with
    # ``scalar_dname_table( idomain, P_qnbca ) = 'QNBCA'`` (:2612) and
    # ``F_qnbca = .TRUE.`` (:2617), with the wif_input_opt==2 block at
    # :18973-18988 registering it a SECOND, idempotent time.  It is listed
    # here because this module answers what an unchanged WRF v4.6.1
    # executable expects in wrfinput, not what woof implements; woof has
    # no nbca species anywhere (see woof.config.MP28_AEROSOL_SOURCE_OPTIONS
    # for wif_input_opt), and that is a woof scope statement, not a claim
    # about WRF's package.
    #
    # THAT GAP IS CLOSED (audit R-054).  The block that stood here recorded
    # a downstream defect and filed a request instead of fixing it: for
    # want of six rows in woof/wrf_direct.py's ``_PACKAGE_FIELD_METADATA``
    # a stock export of an mp=28 domain wrote 8 of these 14 variables and
    # dropped 6 with no error and no receipt line, and the two 2-D members
    # would have been written with the 3-D QCLOUD prototype's rank if the
    # rows had simply been added.  wrf_direct.py now carries all six rows,
    # a 2-D prototype branch, and the per-row ``state_name`` above; the
    # agreement check at its import holds this inventory and that table
    # equal, so a future package member cannot be dropped silently either.
    28: StockWrfPhysicsInventory(
        mp_physics=28,
        scheme="Thompson aerosol-aware",
        registry_package="thompsonaero",
        wrfinput_fields=_ICE_MASS + (
            # Units are WRF's post-reg_parse values; see
            # WRF_RESOLVED_UNITS_NUMBER_PAREN above for why they carry a
            # blank where the Registry line shows '#'.
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nc"),
            _scalar("qnwfa", "QNWFA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nwfa"),
            _scalar("qnifa", "QNIFA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nifa"),
            # No woof species: written at zero, which IS stock behaviour
            # (real.exe's aer_init_opt=0 arm writes exact 0.0 for it).
            _scalar("qnbca", "QNBCA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN),
            _aerosol_emission("qnwfa2d", "QNWFA2D", state_name="nwfa2d"),
            _aerosol_emission("qnifa2d", "QNIFA2D", state_name="nifa2d"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII + (
            # Registry.EM_COMMON:1738-1739.  taod5503d is bare ``r`` and
            # taod5502d is ``rh``: restart and history, never input.  They
            # are the 550 nm aerosol optical depth diagnostics WRF's
            # aerosol-aware Thompson publishes for radiation; woof does
            # not compute them, which is a woof gap and not a reason to
            # misreport WRF's package.
            RuntimeStateField("taod5503d", "TAOD5503D"),
            RuntimeStateField(
                "taod5502d", "TAOD5502D", dimensions=WRFINPUT_2D_DIMS),
        ),
    ),
    # WRF 4.7.1 Registry.EM_COMMON:3053 has the same package fields.
    29: StockWrfPhysicsInventory(
        mp_physics=29,
        scheme="RCON",
        registry_package="rcon_mp_scheme",
        wrfinput_fields=_ICE_MASS + (
            # Units are WRF's post-reg_parse values; see
            # WRF_RESOLVED_UNITS_NUMBER_PAREN above for why they carry a
            # blank where the Registry line shows '#'.
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nc"),
            _scalar("qnwfa", "QNWFA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nwfa"),
            _scalar("qnifa", "QNIFA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nifa"),
            # No woof species: written at zero, which IS stock behaviour
            # (real.exe's aer_init_opt=0 arm writes exact 0.0 for it).
            _scalar("qnbca", "QNBCA", units=WRF_RESOLVED_UNITS_NUMBER_PAREN),
            _aerosol_emission("qnwfa2d", "QNWFA2D", state_name="nwfa2d"),
            _aerosol_emission("qnifa2d", "QNIFA2D", state_name="nifa2d"),
        ),
        runtime_state_not_wrfinput=_EFFECTIVE_RADII + (
            # Registry.EM_COMMON:1738-1739.  taod5503d is bare ``r`` and
            # taod5502d is ``rh``: restart and history, never input.  They
            # are the 550 nm aerosol optical depth diagnostics WRF's
            # aerosol-aware Thompson publishes for radiation; woof does
            # not compute them, which is a woof gap and not a reason to
            # misreport WRF's package.
            RuntimeStateField("taod5503d", "TAOD5503D"),
            RuntimeStateField(
                "taod5502d", "TAOD5502D", dimensions=WRFINPUT_2D_DIMS),
        ),
    ),
    # Registry.EM_COMMON:3038.  P3 one-category, two-moment ice:
    #
    #   package p3_1category mp_physics==50 -
    #     moist:qv,qc,qr,qi;
    #     scalar:qni,qnr,qir,qib;
    #     state:re_cloud,re_ice,vmi3d,rhopo3d,di3d,refl_10cm,th_old,qv_old
    #
    # THE STATE HALF IS ENTIRELY RUNTIME, and that is measured from the I/O
    # flags rather than assumed, because this module's own rule ("Package
    # ``state:`` auxiliaries whose Registry flags contain ``r`` but not
    # ``i`` are runtime/restart state") is what mp=28's qnwfa2d/qnifa2d
    # already proved has to be applied case by case.  Not one of P3's eight
    # state members carries an ``i``: re_cloud (:497) and re_ice (:498) are
    # bare ``r``; vmi3d (:1600), di3d (:1601), rhopo3d (:1602) and
    # refl_10cm (:1596) are ``hdu``; th_old (:1598) and qv_old (:1599) are
    # ``rusd``.  So mp=50 adds NO state member to wrfinput, unlike mp=28.
    #
    # re_snow is deliberately absent from the runtime list.  P3's package
    # does not declare it, which is the same fact WRF acts on when
    # module_physics_init.F:1027-1033 sets has_reqs=0 for P3 -- and that is
    # in turn why the RTE+RRTMGP coupling for mp=50 (rrtmgp's ``50: "p3"``
    # row) remaps the single ice category onto the snow species at P3's own
    # ice radius instead of consuming a snow radius.  Listing RE_SNOW here
    # would invent the field that remap exists to do without.
    #
    # Units are WRF's post-reg_parse RESOLVED values, the mp=28 convention
    # rather than the pre-parse spelling the 6/8/10/18 rows carry: qni is
    # Registry ``"# kg-1"`` (:523-524) and qnr ``"# kg(-1)"`` (:533-534).
    # qir (:555-556) and qib (:557-558) carry NO ``#`` at all -- their
    # Registry text is ``"kg kg(-1)"`` and ``"m(3) kg(-1)"`` -- so for those
    # two the resolved value and the Registry text are the same string and
    # there is nothing to resolve.
    #
    # The two-category package (52, :3040) is inventoried separately below.
    # Other P3 variants retain their own selector admission and inventory.
    55: StockWrfPhysicsInventory(
        mp_physics=55,
        scheme="ISHMAEL adaptive-habit ice",
        registry_package="jensen_ishmael",
        wrfinput_fields=_P3_MASS + (
            _moist("qi2", "QICE2"), _moist("qi3", "QICE3"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qvoli", "QVOLI", units="m(3) kg(-1)"),
            _scalar("qaoli", "QAOLI", units="m(3) kg(-1)"),
            _scalar("qni2", "QNICE2", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni2"),
            _scalar("qvoli2", "QVOLI2", units="m(3) kg(-1)"),
            _scalar("qaoli2", "QAOLI2", units="m(3) kg(-1)"),
            _scalar("qni3", "QNICE3", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni3"),
            _scalar("qvoli3", "QVOLI3", units="m(3) kg(-1)"),
            _scalar("qaoli3", "QAOLI3", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("re_cloud", "RE_CLOUD"),
            RuntimeStateField("re_ice", "RE_ICE"),
            *(RuntimeStateField(reg + suffix, nc + number)
              for suffix, number in (("", ""), ("_2", "2"), ("_3", "3"))
              for reg, nc in (("vmi3d", "v_ice"), ("rhopo3d", "rho_ice"),
                              ("di3d", "d_ice"), ("phii3d", "phi_ice"),
                              ("itype", "ice_type"))),
            RuntimeStateField("refl_10cm", "refl_10cm"),
        ),
    ),

    50: StockWrfPhysicsInventory(
        mp_physics=50,
        scheme="P3 one-category two-moment ice",
        registry_package="p3_1category",
        wrfinput_fields=_P3_MASS + (
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qir", "QIR", units="kg kg(-1)"),
            _scalar("qib", "QIB", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("re_cloud", "RE_CLOUD"),
            RuntimeStateField("re_ice", "RE_ICE"),
            RuntimeStateField("vmi3d", "v_ice"),
            RuntimeStateField("di3d", "d_ice"),
            RuntimeStateField("rhopo3d", "rho_ice"),
            RuntimeStateField("refl_10cm", "refl_10cm"),
            RuntimeStateField("th_old", "TH_OLD"),
            RuntimeStateField("qv_old", "QV_OLD"),
        ),
    ),
    # Registry.EM_COMMON:3039.  P3 one-category with predicted droplet
    # number:
    #
    #   package p3_1category_nc mp_physics==51 -
    #     moist:qv,qc,qr,qi;
    #     scalar:qnc,qni,qnr,qir,qib;
    #     state:re_cloud,re_ice,vmi3d,rhopo3d,di3d,refl_10cm,th_old,qv_old
    #
    # mp=50's row with qnc first in the scalar list (WRF's package order,
    # the order its scalar indices are assigned in, inc/scalar_indices.inc).
    # qnc is Registry :541-542, ``i0rhusdf`` and ``"# kg(-1)"``, so it is a
    # wrfinput member with the mp=28 qnc's resolved unit; woof's state name
    # is ``nc``.  The state half is mp=50's, member for member.
    52: StockWrfPhysicsInventory(
        mp_physics=52,
        scheme="P3 two-category two-moment ice",
        registry_package="p3_2category",
        wrfinput_fields=_P3_MASS + (_moist("qi2", "QICE2"),) + (
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="nc"),
            _scalar("qni2", "QNICE2", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN, state_name="ni2"),
            _scalar("qir2", "QIR2", units="kg kg(-1)"),
            _scalar("qib2", "QIB2", units="m(3) kg(-1)"),
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qir", "QIR", units="kg kg(-1)"),
            _scalar("qib", "QIB", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("re_cloud", "RE_CLOUD"),
            RuntimeStateField("re_ice", "RE_ICE"),
            RuntimeStateField("vmi3d", "v_ice"),
            RuntimeStateField("di3d", "d_ice"),
            RuntimeStateField("vmi3d_2", "v_ice2"),
            RuntimeStateField("di3d_2", "d_ice2"),
            RuntimeStateField("rhopo3d_2", "rho_ice2"),
            RuntimeStateField("rhopo3d", "rho_ice"),
            RuntimeStateField("refl_10cm", "refl_10cm"),
            RuntimeStateField("th_old", "TH_OLD"),
            RuntimeStateField("qv_old", "QV_OLD"),
        ),
    ),
    53: StockWrfPhysicsInventory(
        mp_physics=53,
        scheme="P3 one-category three-moment ice",
        registry_package="p3_1cat_3mom",
        wrfinput_fields=_P3_MASS + (
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nc"),
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qir", "QIR", units="kg kg(-1)"),
            _scalar("qib", "QIB", units="m(3) kg(-1)"),
            _scalar("qzi", "QZI", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("re_cloud", "RE_CLOUD"),
            RuntimeStateField("re_ice", "RE_ICE"),
            RuntimeStateField("vmi3d", "v_ice"),
            RuntimeStateField("di3d", "d_ice"),
            RuntimeStateField("rhopo3d", "rho_ice"),
            RuntimeStateField("refl_10cm", "refl_10cm"),
            RuntimeStateField("th_old", "TH_OLD"),
            RuntimeStateField("qv_old", "QV_OLD"),
        ),
    ),
    51: StockWrfPhysicsInventory(
        mp_physics=51,
        scheme="P3 one-category two-moment ice, predicted droplet number",
        registry_package="p3_1category_nc",
        wrfinput_fields=_P3_MASS + (
            _scalar("qnc", "QNCLOUD", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nc"),
            _scalar("qni", "QNICE", units=WRF_RESOLVED_UNITS_NUMBER_PLAIN,
                    state_name="ni"),
            _scalar("qnr", "QNRAIN", units=WRF_RESOLVED_UNITS_NUMBER_PAREN,
                    state_name="nr"),
            _scalar("qir", "QIR", units="kg kg(-1)"),
            _scalar("qib", "QIB", units="m(3) kg(-1)"),
        ),
        runtime_state_not_wrfinput=(
            RuntimeStateField("re_cloud", "RE_CLOUD"),
            RuntimeStateField("re_ice", "RE_ICE"),
            RuntimeStateField("vmi3d", "v_ice"),
            RuntimeStateField("di3d", "d_ice"),
            RuntimeStateField("rhopo3d", "rho_ice"),
            RuntimeStateField("refl_10cm", "refl_10cm"),
            RuntimeStateField("th_old", "TH_OLD"),
            RuntimeStateField("qv_old", "QV_OLD"),
        ),
    ),
}


# Table-column packages declare their moist species in the owning row.
from woof.core.mp_column_schemes import MP_COLUMN_SCHEMES as _TABLE_ROWS
_mass_by_name = {field.registry_name: field for field in _ICE_MASS}
_INVENTORIES.update({
    mp: StockWrfPhysicsInventory(
        mp_physics=mp, scheme=row.name, registry_package=row.registry_package,
        wrfinput_fields=tuple(_mass_by_name[name] for name in row.species),
        runtime_state_not_wrfinput=())
    for mp, row in _TABLE_ROWS.items()
})
del _mass_by_name

from dataclasses import replace as _replace_inventory
_INVENTORIES[38] = _replace_inventory(
    _INVENTORIES[28], mp_physics=38, scheme="Thompson graupel-hail",
    registry_package="thompsongh",
    wrfinput_fields=_INVENTORIES[28].wrfinput_fields + (
        _scalar("qng", "QNGRAUPEL", units=WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name="ng"),
        _scalar("qvolg", "QVGRAUPEL", units="m(3) kg(-1)", state_name="qvolg")))
_INVENTORIES[40] = StockWrfPhysicsInventory(
    mp_physics=40, scheme="Morrison aerosol", registry_package="morr_tm_aero",
    wrfinput_fields=_INVENTORIES[10].wrfinput_fields + (_scalar("qnc","QNCLOUD",units="# kg(-1)",state_name="nc"),),
    runtime_state_not_wrfinput=_INVENTORIES[10].runtime_state_not_wrfinput + _EFFECTIVE_RADII + tuple(RuntimeStateField(name,name) for name in ("EFCG","EFIG","EFSG","WACT","CCN1_GS","CCN2_GS","CCN3_GS","CCN4_GS","CCN5_GS","CCN6_GS","CCN7_GS","mskf_refl_10cm")),
)
# NTU package, Registry.EM_COMMON:3044; every scalar carries input and restart.
from woof.core.ntu import NTU_SCALARS
from woof.io.wrf_output_schema import NTU_OUTPUT_FIELDS
_NTU_NUMBER_NAMES = dict(qnc="QNCLOUD", qnr="QNRAIN", qni="QNICE", qns="QNSNOW", qng="QNGRAUPEL", qnh="QNHAIL")
_INVENTORIES[56] = StockWrfPhysicsInventory(
    mp_physics=56, scheme="NTU multi-moment", registry_package="ntu3m",
    wrfinput_fields=tuple(_moist(n, w) for n, w in (("qv", "QVAPOR"), ("qc", "QCLOUD"), ("qr", "QRAIN"), ("qi", "QICE"), ("qs", "QSNOW"), ("qg", "QGRAUP"), ("qh", "QHAIL"))) + tuple(
        _scalar(wrf, _NTU_NUMBER_NAMES[wrf], units=WRF_RESOLVED_UNITS_NUMBER_PLAIN if wrf=="qni" else WRF_RESOLVED_UNITS_NUMBER_PAREN, state_name=attr) if wrf in _NTU_NUMBER_NAMES else _scalar(wrf, NTU_OUTPUT_FIELDS[attr].netcdf_name, units=NTU_OUTPUT_FIELDS[attr].units.replace('#',' '), state_name=attr)
        for wrf, attr in NTU_SCALARS), runtime_state_not_wrfinput=())


def supported_stock_wrf_mp_physics() -> tuple[int, ...]:
    """Return the declaratively inventoried stock-WRF microphysics ids."""

    return tuple(sorted(_INVENTORIES))


def _require_agreement_with_the_registry() -> None:
    """The registry's ``consumers.stock_wrf_export`` rows ARE this table.

    Generated from ``_INVENTORIES`` by tools/build_registry.py and held to
    it here, so a package row added without a rebuilt registry -- or a
    registry claiming an inventory this module does not carry -- fails this
    import.  The five uninventoried schemes are an export-only scope
    decision the row itself states (``inventoried: false`` with its
    reason); they are not absences to cite, because the registry publishes
    a row for them too.
    """

    from woof.physics_registry import require_consumer_rows_agreement

    require_consumer_rows_agreement(
        "woof.wrf_physics_inventory._INVENTORIES",
        "microphysics", "stock_wrf_export",
        {mp: {"inventoried": True,
              "netcdf_names": [field.netcdf_name
                               for field in inventory.wrfinput_fields]}
         for mp, inventory in _INVENTORIES.items()},
        project=lambda row: (
            {"inventoried": True,
             "netcdf_names": [field["netcdf_name"]
                              for field in row["wrfinput_fields"]]}
            if row.get("inventoried") is True else None),
        cited_absences={
            0: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            1: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            3: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014); WSM3 joined the runtime with its port, the stock-WRF export contract did not",
            9: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            16: "export-only scope: no evidenced Registry.EM_COMMON package contract packaged (audit R-014)",
            900: "TEMPO is a WOOF named scheme with no WRF Registry.EM_COMMON package; stock-WRF export does not apply",
        })


_require_agreement_with_the_registry()


def stock_wrf_physics_inventory(mp_physics: int) -> StockWrfPhysicsInventory:
    """Return an exact v4.6.1 package inventory or fail closed."""

    if isinstance(mp_physics, bool) or not isinstance(mp_physics, int):
        raise TypeError(
            f"mp_physics must be a WRF integer, got {mp_physics!r}"
        )
    try:
        return _INVENTORIES[mp_physics]
    except KeyError:
        supported = ", ".join(str(value) for value in sorted(_INVENTORIES))
        raise ValueError(
            f"stock-WRF wrfinput export for mp_physics={mp_physics} is not "
            "available on this route: it writes initialization files for an "
            "UNCHANGED WRF v4.6.1 executable, and no evidenced "
            "Registry.EM_COMMON package contract (member list plus "
            "real.exe initialized-state policy) is packaged for that "
            "selector, so the exported wrfinput would under-declare its own "
            "package's hydrometeors and moments. This says NOTHING about "
            "running the scheme in WOOF -- WOOF's own forecast route runs "
            "every scheme the physics registry publishes as implemented, "
            "and the runtime verdict is answered separately. Evidenced "
            f"package contracts: {supported}."
        ) from None


__all__ = [
    "EXPORT_USE_THETA_M",
    "SCHEMA",
    "StockWrfPhysicsInventory",
    "RuntimeStateField",
    "WRFINPUT_2D_DIMS",
    "WRFINPUT_3D_DIMS",
    "WRF_RESOLVED_UNITS_NUMBER_PAREN",
    "WRF_RESOLVED_UNITS_NUMBER_PLAIN",
    "WrfInputField",
    "stock_wrf_physics_inventory",
    "supported_stock_wrf_mp_physics",
]
