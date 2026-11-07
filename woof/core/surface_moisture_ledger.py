"""Per-column accounting for where the published 2 m moisture came from.

WHY THIS EXISTS.  A 2 m dewpoint that collapses overnight can be blamed on
four different things, and the shipped record of the v1.6.2 Gulf-coast
report named one of them without ever separating it from the other three:

1. the lowest model level dried out (``QV1`` moved),
2. the surface endpoint or its exchange moved (``QSFC``, ``CQS2``, ``QFX``),
3. the 2 m diagnostic was written by a different provider than the one the
   causal story assumed, or written twice with the second writer silently
   winning, or
4. the mixing-ratio-to-dewpoint conversion in the product chain is wrong.

Each of those leaves a different signature, and this module records all four
at once so a run can be read rather than argued about.  Every row carries the
inputs, the provider's OWN formula re-evaluated from the published inputs
(``q2_expected``), and the difference (``q2_residual``).  A residual inside
the FP32 budget means the named provider really is the writer and its inputs
really are the ones recorded, so any moisture change must live in those
inputs.  A residual outside the budget means the accounting itself is wrong,
and the run's causal story cannot be trusted until that is explained.

THE LEDGER IS OPT-IN AND OFF BY DEFAULT.  It allocates nothing and costs
nothing until a caller attaches one, and it never writes to a physics field.

FP32 BUDGET.  The identities below are re-evaluated in FP64 from FP32
published inputs, so they cannot close better than the rounding of the
original FP32 arithmetic.  :data:`Q2_RESIDUAL_BUDGET_KG_KG` is the floor and
:data:`Q2_RESIDUAL_BUDGET_ULP` the relative term; a residual passes when it
is inside the LARGER of the two, which keeps the test meaningful at both
Antarctic (1e-5 kg/kg) and tropical (2e-2 kg/kg) mixing ratios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Sequence

import numpy as np

from woof.core import constants as c

#: The FINAL writer of Q2, by name.  These are the only four that exist.
PROVIDER_SFCLAY = "SFCLAY"
PROVIDER_NOAH_SFCDIAGS = "NOAH_SFCDIAGS"
PROVIDER_RUC_SFCDIAGS = "RUC_SFCDIAGS"
PROVIDER_NOAHMP_DIAGNOSTIC = "NOAHMP_DIAGNOSTIC"
PROVIDER_NONE = "NONE"

#: Absolute floor of the residual budget, in kg/kg.
Q2_RESIDUAL_BUDGET_KG_KG = 2.0e-6
#: Relative term, in units of FP32 ULP of the published Q2.
Q2_RESIDUAL_BUDGET_ULP = 8.0

#: Land-surface selectors whose scheme runs WRF's SFCDIAGS after the LSM and
#: therefore overwrites SFCLAY's Q2.  Mirrors
#: :data:`woof.core.physics.LAND_SURFACE_SFCDIAGS_SCHEMES`; asserted equal by
#: the covering test so the two can never drift apart.
#: The slab scheme (1) joined with its port: WRF CASE (SLABSCHEME) runs the
#: same SFCDIAGS after SLAB (module_surface_driver.F:2543-2583), so its Q2
#: is the SFCDIAGS form too.
SFCDIAGS_SCHEMES = frozenset({1, 2})


def q2_residual_budget(q2: float) -> float:
    """Largest residual that is still only FP32 rounding, in kg/kg."""
    if not math.isfinite(q2):
        return math.inf
    ulp = float(np.spacing(np.float32(abs(q2)))) if q2 else 0.0
    return max(Q2_RESIDUAL_BUDGET_KG_KG, Q2_RESIDUAL_BUDGET_ULP * ulp)


def resolve_q2_provider(*, sf_sfclay_physics: int, sf_surface_physics: int,
                        ) -> str:
    """Name the LAST scheme to write Q2 for this configuration.

    This is deliberately the same dispatch rule the driver itself runs, not
    an independent guess: the surface-layer scheme writes Q2 for every
    column, and a land-surface scheme in :data:`SFCDIAGS_SCHEMES` then
    overwrites it -- over WATER as well as land, because WRF's SFCDIAGS
    carries no land mask (phys/module_sf_sfcdiags.F:45-72) and neither does
    woof's transcription of it.  RUC and Noah-MP are NOT in that set, so
    under those two the surface-layer value survives to wrfout.
    """
    sfclay = int(sf_sfclay_physics)
    land = int(sf_surface_physics)
    if land in SFCDIAGS_SCHEMES:
        return PROVIDER_NOAH_SFCDIAGS
    if land == 3:
        # RUC has its own 2 m diagnostic in WRF
        # (module_sf_sfcdiags_ruclsm.F).  woof does not route it, so the
        # surface-layer value is what reaches wrfout; the ledger names the
        # scheme that WOULD own the row so a future wiring change shows up
        # here as a provider change rather than as a silent residual.
        return PROVIDER_RUC_SFCDIAGS
    if land == 4:
        return PROVIDER_NOAHMP_DIAGNOSTIC
    if sfclay:
        return PROVIDER_SFCLAY
    return PROVIDER_NONE


def dewpoint_k(q: float, p_pa: float) -> float:
    """2 m dewpoint in kelvin from mixing ratio and pressure.

    This is the conversion the rendered product chain applies to Q2/PSFC,
    transcribed here so ``td2_residual`` measures the CHAIN against the
    ledger rather than the ledger against itself.  Bolton's form over water,
    the same one WRF's own post-processing uses:

        e  = p_hPa * q / (EPS + q)          with EPS = 0.622
        Td = 243.5 / (17.67/ln(e/6.112) - 1) + 273.15

    A run whose product TD2 disagrees with this is either using a different
    conversion or was handed a different Q2, and ``td2_residual`` separates
    those two cases when read beside ``q2_residual``.
    """
    if not (math.isfinite(q) and math.isfinite(p_pa)) or q <= 0.0 or p_pa <= 0.0:
        return math.nan
    e = (p_pa / 100.0) * q / (0.622 + q)
    if e <= 0.0:
        return math.nan
    lg = math.log(e / 6.112)
    if lg >= 17.67:
        return math.nan
    return 243.5 / (17.67 / lg - 1.0) + 273.15


def expected_q2(provider: str, column: Mapping[str, float]) -> float:
    """Re-evaluate the named provider's OWN Q2 formula in FP64.

    SFCLAY (``module_sf_sfclay.F``, woof ``kernels/sfclay.cu``)::

        q2 = qs + (qv1 - qs) * psiq2/psiq

    ``psiq`` and ``psiq2`` are kernel-internal, but the kernel publishes
    ``chs = ust*k/psiq`` and ``cqs2 = ust*k/psiq2`` from the same ``ust``,
    so ``psiq2/psiq == chs/cqs2`` exactly and the identity closes from
    published fields alone.

    Noah SFCDIAGS (``module_sf_sfcdiags.F:56``, woof
    ``PhysicsDriver._refresh_surface_diagnostics``)::

        q2 = qsfc - qfx/(rho*cqs2)          rho = psfc/(RD*tsk)

    with woof's documented lower-bound divergence: where that leaves the
    physical range the published value is ``qv1`` instead.  Both branches
    are reproduced, because a ledger that only knew the unbounded form would
    report a huge false residual on exactly the cold columns the bound was
    written for.

    NOTE the two forms are NOT the same function.  Substituting SFCLAY's own
    ``qfx = rho*mavail*chs*(qs-qv1)`` into the SFCDIAGS form gives
    ``qs + mavail*(chs/cqs2)*(qv1-qs)`` -- SFCLAY's expression with an extra
    ``mavail``.  Over water ``mavail == 1`` and they agree exactly; over land
    they do not, which is why the provider column is essential and not
    decoration.
    """
    qsfc = float(column["qsfc"])
    qv1 = float(column["qv1"])
    if provider == PROVIDER_SFCLAY or provider == PROVIDER_RUC_SFCDIAGS \
            or provider == PROVIDER_NOAHMP_DIAGNOSTIC:
        chs = float(column["chs"])
        cqs2 = float(column["cqs2"])
        if cqs2 == 0.0:
            # isfflx=0 leaves every exchange coefficient at zero and the
            # ratio is 0/0.  The kernel still evaluates psiq2/psiq from the
            # psi terms, so the identity is unrecoverable from published
            # fields here and the ledger says so rather than inventing one.
            return math.nan
        return qsfc + (qv1 - qsfc) * (chs / cqs2)
    if provider == PROVIDER_NOAH_SFCDIAGS:
        cqs2 = float(column["cqs2"])
        psfc = float(column["psfc"])
        tsk = float(column["tsk"])
        qfx = float(column["qfx"])
        rho = psfc / (float(c.RD) * tsk)
        if cqs2 < 1.0e-5:
            # q_active false: woof publishes qsfc unchanged.
            return qsfc
        diagnosed = qsfc - qfx / (rho * cqs2)
        return diagnosed if diagnosed > 0.0 else qv1
    return math.nan


@dataclass
class CarrierRecord:
    """One radiative carrier's provenance as of a surface call."""

    name: str
    value: float = math.nan
    source: str = "absent"
    last_update_model_time: float = math.nan

    def age_seconds(self, now: float) -> float:
        if not math.isfinite(self.last_update_model_time):
            return math.nan
        return float(now) - float(self.last_update_model_time)


@dataclass
class LedgerRow:
    """Everything needed to attribute one column's published 2 m moisture."""

    model_time: float
    i: int
    j: int
    # -- surface classification
    xland: float = math.nan
    landmask: float = math.nan
    lakemask: float = math.nan
    distance_to_land: float = math.nan
    # -- provider
    q2_provider: str = PROVIDER_NONE
    # -- state the provider read
    tsk: float = math.nan
    psfc: float = math.nan
    t1: float = math.nan
    qv1: float = math.nan
    qsfc: float = math.nan
    qfx: float = math.nan
    cqs2: float = math.nan
    chs: float = math.nan
    chs2: float = math.nan
    mavail: float = math.nan
    # -- psi_q terms, recovered from the published exchange pair
    psiq: float = math.nan
    psiq2: float = math.nan
    ust: float = math.nan
    # -- what was published, what the formula says, and the gap
    q2: float = math.nan
    q2_expected: float = math.nan
    q2_residual: float = math.nan
    q2_residual_budget: float = math.nan
    q2_within_budget: bool = False
    # -- the product chain's dewpoint against the ledger's own conversion
    td2_product: float = math.nan
    td2_from_q2: float = math.nan
    td2_residual: float = math.nan
    # -- radiative carriers
    carriers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SurfaceMoistureLedger:
    """Records one row per selected column per surface-physics call.

    Attach to a driver and it captures; leave it off and the driver is
    byte-for-byte the run it was without it.  The ledger never writes a
    physics field and never reads a device array outside the columns asked
    for, so a whole-domain run can be instrumented on a handful of points.
    """

    def __init__(self, columns: Sequence[tuple[int, int]], *,
                 distance_to_land: Mapping[tuple[int, int], float] | None = None,
                 td2_product: Mapping[tuple[int, int], float] | None = None):
        self.columns = [(int(j), int(i)) for j, i in columns]
        self.rows: list[LedgerRow] = []
        self.carriers: dict[str, CarrierRecord] = {}
        self._distance_to_land = dict(distance_to_land or {})
        self._td2_product = dict(td2_product or {})

    # -- carrier provenance -------------------------------------------------
    def note_carrier(self, name: str, *, source: str, model_time: float,
                     value: float = math.nan) -> None:
        """Stamp a radiative carrier as written by ``source`` at ``model_time``."""
        self.carriers[name] = CarrierRecord(
            name=name, value=float(value), source=str(source),
            last_update_model_time=float(model_time))

    # -- capture ------------------------------------------------------------
    def capture(self, *, fields: Mapping[str, Any], atmosphere: Mapping[str, Any],
                model_time: float, sf_sfclay_physics: int,
                sf_surface_physics: int) -> list[LedgerRow]:
        """Read the selected columns and append one row each."""
        provider = resolve_q2_provider(
            sf_sfclay_physics=sf_sfclay_physics,
            sf_surface_physics=sf_surface_physics)
        host = _HostView(fields, atmosphere, self.columns)
        made = []
        for (j, i) in self.columns:
            col = host.column(j, i)
            row = LedgerRow(model_time=float(model_time), i=int(i), j=int(j))
            row.xland = col.get("xland", math.nan)
            row.landmask = col.get("landmask", math.nan)
            row.lakemask = col.get("lakemask", math.nan)
            row.distance_to_land = float(
                self._distance_to_land.get((j, i), math.nan))
            row.q2_provider = provider
            for name in ("tsk", "psfc", "qsfc", "qfx", "cqs2", "chs",
                         "chs2", "mavail", "ust", "q2"):
                setattr(row, name, col.get(name, math.nan))
            row.t1 = col.get("t1", math.nan)
            row.qv1 = col.get("qv1", math.nan)
            # psi_q terms back out of the published exchange pair; both use
            # the SAME ust the kernel used, so the division is exact.
            if row.chs:
                row.psiq = row.ust * 0.4 / row.chs
            if row.cqs2:
                row.psiq2 = row.ust * 0.4 / row.cqs2
            row.q2_expected = expected_q2(provider, {
                "qsfc": row.qsfc, "qv1": row.qv1, "chs": row.chs,
                "cqs2": row.cqs2, "psfc": row.psfc, "tsk": row.tsk,
                "qfx": row.qfx})
            if math.isfinite(row.q2) and math.isfinite(row.q2_expected):
                row.q2_residual = row.q2 - row.q2_expected
            row.q2_residual_budget = q2_residual_budget(row.q2)
            row.q2_within_budget = bool(
                math.isfinite(row.q2_residual)
                and abs(row.q2_residual) <= row.q2_residual_budget)
            row.td2_from_q2 = dewpoint_k(row.q2, row.psfc)
            product = self._td2_product.get((j, i))
            if product is not None:
                row.td2_product = float(product)
                row.td2_residual = row.td2_product - row.td2_from_q2
            row.carriers = {
                name: {"value": rec.value, "source": rec.source,
                       "last_update_model_time": rec.last_update_model_time,
                       "age_seconds": rec.age_seconds(model_time)}
                for name, rec in sorted(self.carriers.items())}
            self.rows.append(row)
            made.append(row)
        return made

    # -- reporting ----------------------------------------------------------
    def to_records(self) -> list[dict[str, Any]]:
        return [row.as_dict() for row in self.rows]

    def breaches(self) -> list[LedgerRow]:
        """Rows whose provider identity did not close inside the budget."""
        return [r for r in self.rows if not r.q2_within_budget]


class _HostView:
    """Pulls the named columns off the device exactly once per capture."""

    #: field name -> (container, key).  ``a`` is the atmosphere mapping and
    #: needs its lowest level taken; ``f`` is the driver's surface fields.
    _SURFACE = ("tsk", "psfc", "qsfc", "qfx", "cqs2", "chs", "chs2",
                "mavail", "ust", "q2", "xland", "lakemask", "landmask")

    def __init__(self, fields, atmosphere, columns):
        js = [j for j, _ in columns]
        ids = [i for _, i in columns]
        self._data: dict[str, list[float]] = {}
        for name in self._SURFACE:
            arr = fields.get(name) if hasattr(fields, "get") else None
            if arr is None:
                continue
            self._data[name] = _gather(arr, js, ids)
        for name, key in (("t1", "temperature"), ("qv1", "qv")):
            arr = atmosphere.get(key) if hasattr(atmosphere, "get") else None
            if arr is None:
                continue
            self._data[name] = _gather(arr[0], js, ids)
        self._index = {(j, i): n for n, (j, i) in enumerate(columns)}

    def column(self, j, i) -> dict[str, float]:
        n = self._index[(j, i)]
        return {name: values[n] for name, values in self._data.items()}


def _gather(arr, js, ids) -> list[float]:
    """Host-side values at the requested columns, CuPy or NumPy alike."""
    get = getattr(arr, "get", None)
    host = arr.get() if callable(get) else np.asarray(arr)
    host = np.asarray(host)
    return [float(host[j, i]) for j, i in zip(js, ids)]


__all__ = [
    "PROVIDER_SFCLAY", "PROVIDER_NOAH_SFCDIAGS", "PROVIDER_RUC_SFCDIAGS",
    "PROVIDER_NOAHMP_DIAGNOSTIC", "PROVIDER_NONE",
    "Q2_RESIDUAL_BUDGET_KG_KG", "Q2_RESIDUAL_BUDGET_ULP", "SFCDIAGS_SCHEMES",
    "CarrierRecord", "LedgerRow", "SurfaceMoistureLedger",
    "dewpoint_k", "expected_q2", "q2_residual_budget", "resolve_q2_provider",
]
