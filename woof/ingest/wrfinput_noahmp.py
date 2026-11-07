"""Disposition of the additional cold-start Noah-MP input records.

Registry/registry.noahmp in WRF v4.6.1 declares these records on input
stream zero. They are not additional atmospheric tracers. NOAHMP_INIT
(phys/module_sf_noahmpdrv.F:2009-2335) initializes the prognostic surface
carriers on a non-restart launch from the common TSK, soil, snow, canopy
water and land-category fields. Copying the placeholders in real.exe's
output over that initialized state would restore, for example, TV=0 K.

The selected configuration supplies the supported Noah-MP options. Crop,
irrigation, tile drainage and alternative soil/runoff inputs have no
consumer under that configuration. Flux diagnostics are calculated by
the first surface step. Engine checkpoints restore their own state and
do not reinterpret these cold-start records as a restart.
"""

from types import MappingProxyType

import numpy as np


def require_cold_start(dataset) -> None:
    """Reject explicitly stepped surface state before cold initialization.

    Standard real.exe inputs need not carry a provenance flag. A declared
    restart or elapsed-step marker is different: its surface carriers are
    live state, and the cold-start initializer cannot discard them safely.
    """
    attributes = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
    indicators = []
    restart = attributes.get("RESTART", 0)
    if isinstance(restart, str):
        restart = restart.strip().lower()
        restart = {"true": 1, ".true.": 1, "t": 1,
                   "false": 0, ".false.": 0, "f": 0}.get(restart, restart)
    for name in ("RESTART", "ITIMESTEP", "XTIME"):
        value = restart if name == "RESTART" else attributes.get(name, 0)
        if name in dataset.variables:
            value = dataset.variables[name][...]
        try:
            values = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(
                f"Noah-MP input has an invalid {name} state marker. "
                "Regenerate the original real.exe inputs.") from None
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(
                f"Noah-MP input has an invalid {name} state marker. "
                "Regenerate the original real.exe inputs.")
        if np.any(values > 0):
            indicators.append(name)
    title = " ".join(str(attributes.get("TITLE", "")).upper().split())
    # med_filter_out writes a nested input after start_domain/phy_init at
    # the simulation's initial instant. Reading it on WRF's cold branch
    # calls NOAHMP_INIT again (.NOT.restart, noahmpdrv.F:2008), so those
    # initialized records are reset from the common soil/snow inputs.
    # A model TITLE alone cannot distinguish this from stepped history.
    initial_child = False
    if (int(attributes.get("GRID_ID", 1)) > 1 and "Times" in dataset.variables
            and "OUTPUT FROM WRF" in title and "MODEL" in title):
        chars = np.asarray(dataset.variables["Times"][...]).reshape(-1)
        record = b"".join(bytes(item) for item in chars).decode("ascii").strip("\x00 ")
        initial_child = (record == str(attributes.get("START_DATE", "")).strip()
                         and not indicators)
    if "OUTPUT FROM WRF" in title and "MODEL" in title and not initial_child:
        indicators.append("model-output TITLE")
    # real_em.F:99 declares its producer name; output_wrf.F:336-337
    # records it in TITLE. That producer has not run NOAHMP_INIT, whose
    # actual state is constructed by start_em.F's phy_init call. Without
    # that producer declaration, zero/masked placeholders are still safe
    # to initialize, but meaningful carried state cannot be thrown away.
    real_input = title.startswith("OUTPUT FROM REAL_EM ") and "PREPROCESSOR" in title
    if not real_input and not initial_child and not indicators:
        for name in _COLD_START:
            if name not in dataset.variables:
                continue
            value = np.ma.asarray(dataset.variables[name][...], dtype=np.float64)
            known = np.asarray(value.filled(np.nan))
            if np.any(np.isfinite(known) & (known != 0.0)):
                indicators.append(f"non-placeholder {name}")
                break
    if indicators:
        raise ValueError(
            "This file carries Noah-MP state that cannot be discarded "
            f"({', '.join(indicators)}), but this input route does not restore "
            "the full WRF restart inventory, including snow-age state. "
            "Use the original real.exe inputs with a WOOF --restart checkpoint, "
            "or continue the WRF restart with WRF.")


# registry.noahmp:2-30 and module_sf_noahmpdrv.F:NOAHMP_INIT/SNOW_INIT.
_COLD_START = (
    "ISNOW", "TV", "TG", "CANICE", "CANLIQ", "EAH", "TAH", "CM", "CH",
    "FWET", "SNEQVO", "ALBOLD", "QSNOWXY", "QRAINXY", "WSLAKE", "WA",
    "TSNO", "ZSNSO", "SNICE", "SNLIQ", "XSAI",
)

# registry.noahmp:32-76,93-122,141-145. The column step writes these.
_DIAGNOSTICS = (
    "T2V", "T2B", "Q2V", "Q2B", "TRAD", "NEE", "GPP", "NPP", "FVEG",
    "QIN", "RUNSF", "RUNSB", "ECAN", "EDIR", "ETRAN", "FSA", "FIRA",
    "APAR", "PSN", "SAV", "SAG", "RSSUN", "RSSHA", "BGAP", "WGAP",
    "TGV", "TGB", "CHV", "CHB", "SHG", "SHC", "SHB", "EVG", "EVB",
    "GHV", "GHB", "IRG", "IRC", "IRB", "TR", "EVC", "CHLEAF", "CHUC",
    "CHV2", "CHB2", "CHSTAR", "QINTS", "QINTR", "QDRIPS", "QDRIPR",
    "QTHROS", "QTHROR", "QSNSUB", "QSNFRO", "QSUBC", "QFROC", "QEVAC",
    "QDEWC", "QFRZC", "QMELTC", "QSNBOT", "QMELT", "PONDING", "PAH",
    "PAHG", "PAHV", "PAHB", "CANHS", "FPICE", "RAINLSM", "SNOWLSM",
    "FORCTLSM", "FORCQLSM", "FORCPLSM", "FORCZLSM", "FORCWLSM",
)

# registry.noahmp:17-29,89-92,123-127,217-255. The option identity is
# woof/core/noahmp_mynn_contract.py:NOAHMP_NAMELIST_DEFAULTS: dveg=4,
# opt_run=3, opt_soil=1, opt_crop=0, opt_irr=0 and opt_tdrn=0.
_INACTIVE_OPTIONS = (
    "ZWT", "WT", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP",
    "FDEPTH", "EQZWT", "RECHCLIM", "RIVERBED", "SOILCOMP", "SOILCL1",
    "SOILCL2", "SOILCL3", "SOILCL4", "GRAIN", "GDD", "CROPTYPE",
    "PLANTING", "HARVEST", "SEASON_GDD", "TD_FRACTION", "QTDRAIN",
    "IRFRACT", "SIFRACT", "MIFRACT", "FIFRACT", "IRNUMSI", "IRNUMMI",
    "IRNUMFI", "IRSIVOL", "IRMIVOL", "IRFIVOL", "IRELOSS", "IRRSPLH",
)

NOAHMP_INPUT_DISPOSITIONS = MappingProxyType({
    **dict.fromkeys(_COLD_START, "initialized from common cold-start fields"),
    **dict.fromkeys(_DIAGNOSTICS, "recomputed by surface physics"),
    **dict.fromkeys(_INACTIVE_OPTIONS, "inactive under selected surface options"),
})

# COLD_START_WRITES in noahmp_runtime also names these shared carriers.
# Their initialized values must survive the generic supplied-field copy.
from woof.core.noahmp_soil_cadence import ACCUMULATORS
from woof.core.noahmp_output import ACCUMULATORS as OUTPUT_ACCUMULATORS,CARRIERS as OUTPUT_CARRIERS
NOAHMP_INITIALIZED_SURFACE_FIELDS = frozenset({
    *(OUTPUT_CARRIERS[n] for n in OUTPUT_ACCUMULATORS),
    *ACCUMULATORS.values(), "qtdrain",
    "irnumsi","irnummi","irnumfi","irwatsi","irwatmi","irwatfi","irsivol","irmivol","irfivol","ireloss","irrsplh",
    "snow", "snowh", "canwat", "tslb", "smois", "sh2o", "lai", "zwtxy", "wtxy",
    "lfmassxy","rtmassxy","stmassxy","woodxy","stblcpxy","fastcpxy","grainxy","gddxy","cropcat",
})


def input_dispositions(cfg) -> MappingProxyType:
    """Return only the records justified by the selected surface scheme."""
    if int(getattr(cfg, "sf_surface_physics", 0)) != 4:
        return MappingProxyType({})
    # Configuration admission checks this identity too. Keep the decision
    # valid for direct reader callers instead of admitting inactive inputs
    # when their operation was selected without a corresponding consumer.
    from woof.config import NOAHMP_OPTION_IDENTITY
    for name in ("dveg", "opt_run", "opt_soil", "opt_crop", "opt_irr", "opt_tdrn"):
        wanted = NOAHMP_OPTION_IDENTITY[name]
        # Prescribed vegetation 1/3/4 shares NOAHMP_INIT and has no carbon pools.
        allowed = ((1, 2, 3, 4, 5, 6, 7, 8, 9) if name == "dveg" else
                   (1, 2, 3, 4, 5, 6, 7, 8) if name == "opt_run" else
                   (1,2,3,4) if name == "opt_soil" else
                   (0,1,2) if name=="opt_crop" else
                   (0,1,2) if name=="opt_tdrn" else (0,1,2,3) if name=="opt_irr" else (wanted,))
        if getattr(cfg, name, wanted) not in allowed:
            raise ValueError(
                f"Noah-MP input {name}={getattr(cfg, name)} selects a surface "
                f"operation these cold-start records do not initialize. "
                f"Use the supported {name}={wanted} option or provide its state consumer.")
    result=dict(NOAHMP_INPUT_DISPOSITIONS)
    if int(getattr(cfg,"opt_run",3)) in (1,2):
        result.update({name:"initialized from common cold-start fields" for name in ("ZWT","WT")})
    if int(getattr(cfg,"opt_run",3))==5:
        for name in ("FDEPTH","EQZWT","RIVERBED","RECHCLIM"):result.pop(name,None)
        for name in ("ZWT","SMCWTD","DEEPRECH","RECH","SMOISEQ"):result[name]="initialized from common cold-start fields"
    if int(getattr(cfg,"dveg",4)) in (2,5,6):
        result.update({name:"initialized from common cold-start fields" for name in ("LFMASS","RTMASS","STMASS","WOOD","STBLCP","FASTCP")})
    if int(getattr(cfg,"opt_tdrn",0)):
        result.pop("TD_FRACTION",None)
        result["ZWT"]="initialized from common cold-start fields"
    if int(getattr(cfg,"opt_crop",0)):
        for name in ("CROPTYPE","PLANTING","HARVEST","SEASON_GDD"):result.pop(name,None)
        for name in ("GRAIN","GDD","LFMASS","RTMASS","STMASS","WOOD","STBLCP","FASTCP"):result[name]="initialized from common cold-start fields"
    if int(getattr(cfg,"opt_irr",0)):
        from woof.core.noahmp_irrigation_slab import MAPS,COUNTERS,STATE
        for name in MAPS:result.pop(name.upper(),None)
        if int(cfg.opt_irr)==2:
            for name in ("PLANTING","HARVEST"):result.pop(name,None)
        for name in (*COUNTERS,*STATE):result[name.upper()]="initialized from common cold-start fields"
    return MappingProxyType(result)
