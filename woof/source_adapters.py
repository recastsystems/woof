"""Native meteorological-source adapter inventory for WRF initialization.

This registry is deliberately stricter than a decoder capability list.
Runnable means a strict implementation route exists; certification separately
requires that its exact field, vertical-level, cadence, missing-state, and
domain envelope has been accepted by unchanged stock ``wrf.exe``. Merely being
readable by the Rust GRIB decoder does not make a product a scientifically
complete limited-area initial state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from woof.source_authorities import (packaged_authority_sha256,
                                      packaged_member_grammar_ids,
                                      packaged_member_grammar_sha256,
                                      packaged_profile_ids)
from woof.source_coverage import (COVERAGE_WINDOW_TYPES, CoverageWindow,
                                   LambertGridWindow, RegularLatLonWindow)
from woof.source_cycles import CycleGrid
from woof.source_availability import ArchiveWindow
from woof import fetch_endpoints
from woof.grid_requirements import NATIVE_TARGET_INTERIOR_AXIS
from woof.source_credentials import (CredentialLocation, SourceCredential,
                                      credential_declaration)


RUSTY_WEATHER_HEAD = "d3264700c125449a50ac0080be42796177691e1d"
RUSTY_WEATHER_INVENTORY_HASHES = {
    "crates/rustwx-core/src/lib.rs": (
        "cb5a6fbf87f0d0f5cac35f4735bfc0053c7f4dd79985a64afc1b920c7b20a2eb"
    ),
    "crates/rustwx-models/src/lib.rs": (
        "2b7c449a4f9e90ba242856c9955a8d1c9d813d7c6328943033074e7b058b2340"
    ),
    "README.md": (
        "6eb3ea080d78a634fea1fc74e32795fde14af30ad48032efe62bcd90e98db6f3"
    ),
}


class SourceKind(str, Enum):
    """Scientific role of a source product, not merely its file format."""

    DETERMINISTIC_STATE = "deterministic_state"
    ENSEMBLE_MEMBERS = "ensemble_members"
    ENSEMBLE_STATISTIC = "ensemble_statistic"
    SURFACE_ANALYSIS = "surface_analysis"
    POSTPROCESSED_GUIDANCE = "postprocessed_guidance"
    WRF_ARCHIVE = "wrf_archive"
    HYDRO_FORCING = "hydro_forcing"


@dataclass(frozen=True)
class SourceTopEra:
    """One era of a model-level source's native top, measured from its files.

    ``start`` is the first cycle (``YYYY-MM-DDTHH`` UTC) the era covers, or
    ``""`` for the source's whole earlier record.  ``top_pa`` is the model
    top (Pa) the source model ran with: on the source's own eta ladder a
    target with this ``p_top`` co-locates its top level with the source's,
    and any higher target would be extrapolated past the analysis.
    """

    start: str
    top_pa: float
    basis: str

    def __post_init__(self) -> None:
        if self.start:
            datetime.strptime(self.start, "%Y-%m-%dT%H")
        if not math.isfinite(float(self.top_pa)) or float(self.top_pa) <= 0.0:
            raise ValueError("SourceTopEra.top_pa must be a positive finite pressure (Pa)")
        if not self.basis.strip():
            raise ValueError("SourceTopEra.basis must say how the top was established")


class AdapterStatus(str, Enum):
    """Readiness for direct native production of stock-WRF inputs."""

    CERTIFIED = "certified_stock_wrf"
    MAPPING_REQUIRED = "adapter_mapping_required"
    MEMBER_SELECTION_REQUIRED = "member_selection_and_mapping_required"
    COMPOSITION_REQUIRED = "explicit_composition_required"
    ARCHIVE_MAPPING_REQUIRED = "wrf_archive_mapping_required"
    RUNNABLE_NOT_CERTIFIED = "runnable_mapping_not_stock_wrf_certified"


@dataclass(frozen=True)
class SourceAdapter:
    """A fail-closed source-to-WRF adapter declaration."""

    source_id: str
    aliases: tuple[str, ...]
    upstream_model_id: str | None
    source_kind: SourceKind
    file_family: str
    decoder: str
    default_product: str
    required_products: tuple[str, ...]
    max_forecast_hour: int
    upstream_ingest: str
    status: AdapterStatus
    field_mapping: str
    level_mapping: str
    cadence_mapping: str
    stock_wrf_gate: str
    parent_family: str = "physics"
    upstream_model_version: str | None = None
    lineage_donor_source: str | None = None
    lineage_receipts: tuple[str, ...] = ()
    lineage_root_keys: tuple[str, ...] = ()
    lineage_runtime_keys: tuple[str, ...] = ()
    runnable: bool = False
    runner: str | None = None
    #: The packaged profile (mapping + composition + provenance, shipped in
    #: the wheel and pinned by SHA-256 in :mod:`woof.source_authorities`)
    #: this source is decoded through.  A row that names one needs no
    #: per-source runner function: the front door reads the profile.  This
    #: is the arbitrary-acceptance seam -- adding a model whose mapping can
    #: be written is three JSON documents plus this row.
    packaged_profile: str | None = None
    # Canonical preparation row for transforms that require a complete state.
    canonical_preparation_source: str | None = None
    #: Row in data/hydro/forcing-sources.v1.json. This declares catchment
    #: forcing independently of atmospheric initialization capability.
    hydro_forcing_profile: str | None = None
    # A recommendation only; runtime capability never depends on this name.
    default_physics_profile: str | None = None
    #: The recommendation for a run whose finest grid is convection-permitting
    #: (finer than :data:`woof.physics_menu.CONVECTION_PERMITTING_BELOW_M`),
    #: binding ahead of ``default_physics_profile`` there.  A row, not a code
    #: path: a source whose coarse-grid default is wrong at storm scale names
    #: the suite here.
    convection_permitting_physics_profile: str | None = None
    #: Published static source selected by this source's configuration.
    #: An absent declaration leaves existing configuration bytes unchanged.
    static_source: str | None = None
    #: Runtime surface records appended to a native source's soil stream.
    #: Tuples are (engine name, product, index selector, units, numeric selector).
    runtime_surface_fields: tuple[tuple, ...] = ()
    #: Optional exact-lattice pairing for this source's own target grid.
    #: Undeclared sources retain projected indices and coverage checks.
    same_grid_pairing: str | None = None
    composition_requirement: str | None = None
    #: The packaged ``rw-wps.members.v1`` document (shipped in the wheel
    #: and pinned by SHA-256 in :mod:`woof.source_authorities`) this
    #: ensemble's member addressing/verification runs through.  The same
    #: seam as ``packaged_profile``, for the member axis: a row that
    #: names one gets `woof-member-prep` member selection with no
    #: per-source code -- the declared member set, filename patterns,
    #: verification triple and statistic guard are all table data.
    member_set: str | None = None
    #: Operational ensemble associated with this source. Selection is table
    #: data so deterministic and ensemble doors share one provider path.
    ensemble_source: str | None = None
    #: A separately published control trajectory, when the member grammar
    #: contains perturbed members only. It is not assigned invented GRIB
    #: ensemble octets; its own deterministic source verifies its inputs.
    ensemble_control_source: str | None = None
    #: A multi-member run started from this source needs a policy fitted
    #: against observations before automatic selection may run it: the
    #: reference model-error fallback is not offered for it.  The built-in
    #: unfitted policy (:mod:`woof.ensemble.automatic_sources`) reads this
    #: column, so a source that needs the same treatment is one row here.
    requires_ensemble_calibration: bool = False
    #: The native cadence of this source's model state, in seconds -- the
    #: spacing between the valid times a preparation gets lateral boundary
    #: conditions from, and therefore ``&share/interval_seconds`` in an
    #: emitted ``namelist.wps``.  THE reason `woof domain` could not emit a
    #: runnable config for these sources: the wizard held a three-entry dict
    #: and every other model's battery TOML was hand-assembled around it.
    #:
    #: For a row that names a ``packaged_profile`` this repeats the profile
    #: mapping's ``target.boundary_interval_seconds``, the same way
    #: :mod:`woof.source_authorities` repeats the composition's roles: the
    #: wizard has to know the cadence before anything opens a JSON
    #: authority, and a repeat guarded by a test is not a second definition.
    #: :func:`tests.test_source_wizard_facts` fails the moment the two
    #: disagree.  ``None`` means the cadence is not a property of the source
    #: (the generic ``mapped`` route reads it from the caller's own mapping),
    #: and the wizard refuses to plan such a source by name.
    forcing_interval_seconds: float | None = None
    #: The acquisition consumes every native frame; a cadence cannot subsample it.
    fetch_entire_window: bool = False
    #: How a run on the same machine publishes this source's leads as it
    #: writes them (``woof.local_posting``), for a source that is produced,
    #: not downloaded.  Set, ``woof prep --as-posted`` prepares it lead by
    #: lead with no network posting row; ``None`` keeps the posting-row rule.
    local_posting: str | None = None
    #: Hours of this source's 2 m air temperature that inland water beyond
    #: the source's reach averages (WRF use_tavg_for_tsk, WPS avg_tsfc).
    #: ``None`` keeps the WPS day (24 h); ``0`` takes the start time's own
    #: 2 m air temperature, so the initial state needs no later lead.
    inland_air_mean_hours: float | None = None
    #: The default fetch transport writes a template unless retrieval is requested.
    fetch_requires_retrieve: bool = False
    #: The smallest pressure (Pa) this source's CERTIFIED inventory serves
    #: -- the top of the ladder its route decodes, a published fact of the
    #: source like its cadence.  ``None`` means the column reaches at
    #: least the wizard's default model top
    #: (:data:`woof.domain_wizard.DEFAULT_MODEL_TOP_PA`, 5000 Pa) and the
    #: default is emitted unchanged; a declared value floors the emitted
    #: ``p_top`` so a bare ``woof domain --source X`` config never asks
    #: for a model top its own source cannot cover and refuses at
    #: preparation after the acquisition was already paid for.  Gated by
    #: tests/test_ptop_default.py against the packaged mapping's own
    #: level table.
    certified_source_top_pa: float | None = None
    #: The smallest pressure (Pa) this source's FETCH reaches when a run
    #: asks for a model top above ``certified_source_top_pa``: the fetch
    #: extends the certified ladder upward along the levels the product
    #: publishes (``woof fetch --p-top-pa``).  ``None`` means the fetch
    #: takes no model top, so the certified ladder is all a run gets.
    #: Every door that downloads for a configuration reads this through
    #: :func:`fetch_model_top_pa`, so a run whose own ladder tops out
    #: above the certified one fetches the levels it needs with no flag.
    extendable_source_top_pa: float | None = None
    #: The native model top (Pa) a model-level source's column reaches,
    #: by the cycle era it was produced in, oldest first: a published fact
    #: of the source's own vertical grid that changed with an upgrade
    #: (HRRR raised its top from 20 to 15 hPa with v4).  Read through
    #: :func:`source_top_pa_for_cycle`, so an emitted configuration whose
    #: requested top lies above what its cycle's era reaches is served at
    #: the era's top before anything is downloaded.  Breakage it prevents:
    #: the hrrr_v4_gsd41 recipe (p_top 1500 Pa) on a pre-v4 HRRR cycle
    #: fetched 26 GB and then refused at root_initialize ("target pressure
    #: lies above source top").  Empty means one top for every cycle.
    source_top_eras: tuple["SourceTopEra", ...] = ()
    #: Where this source's native grid reaches, or ``None`` for a global
    #: product.  A declared window (see :mod:`woof.source_coverage`) is what
    #: lets `woof domain` refuse an out-of-coverage plan AT PLAN TIME with
    #: the offending corner named, instead of letting the user pay for the
    #: whole acquisition and read the refusal out of a preparation
    #: traceback -- measured on ICON-EU over a central-US domain,
    #: 2026-08-17.  Rows with no route to run (``runnable`` false) declare
    #: nothing: a window nobody measured would be a number invented to fill
    #: a field.
    coverage_window: CoverageWindow | None = None
    #: WHEN this source initializes, and when its bytes land -- the
    #: declaration ``--cycle latest`` resolves against (see
    #: :mod:`woof.source_cycles`).  The resolver used to branch on
    #: gfs/gdas/hrrr by name and refuse every other source with a
    #: sentence about ERA5's latency, so `--cycle latest --source rap`
    #: was told about a reanalysis it had not asked for and a reanalysis
    #: with a KNOWN delay was told it had no latest at all.
    #:
    #: ``None`` is the common case and not a gap: a source whose fetch
    #: route declares ``cycle_hours`` has its grid DERIVED from that
    #: measured table, which is what makes a new model's ``latest`` a
    #: route row rather than a code change.  This column carries the
    #: schedules no route table can state -- the legacy transports, and
    #: a keyed job API like the CDS, which publishes no object to probe
    #: and so has to write its publication delay down.
    cycle_grid: CycleGrid | None = None
    #: Documented starts of the CURRENT acquisition layouts, per transport.
    #: An omitted window means unknown, never unlimited historical coverage.
    archive_windows: tuple[ArchiveWindow, ...] = ()
    #: What a PERSON calls this source.  A column, because every consumer
    #: that wanted a human name had to keep its own id-to-name lookup --
    #: a per-model table, and adding a model then meant editing a front
    #: end as well as this file.  ``None`` reads back as the id through
    #: :attr:`display_title`, so a row that omits it still renders.
    display_name: str | None = None
    #: What must be CONFIGURED before this source's bytes can be
    #: acquired, as declared :class:`~woof.source_credentials.
    #: SourceCredential` rows.  Empty means the row declares no
    #: prerequisite -- which is a statement about this table, not a
    #: promise that a provider asks for nothing.  The same seam as every
    #: other column: a source that needs an account key is one row, and
    #: no front end carries an exception for it.
    credentials: tuple[SourceCredential, ...] = ()
    # Internal preparation capability: the root target constructor's required
    # interior between specified edges. Not a public source identity or a
    # requirement for children initialized by a different target operation.
    root_target_interior_axis: int | None = field(
        default=None, compare=False, repr=False, kw_only=True)
    #: What the source's bytes ARE, for a person choosing between them:
    #: ``"forecast"`` (a model run forward from a start), ``"analysis"``
    #: (the forecast system's own estimate of the weather at each cycle)
    #: or ``"reanalysis"`` (one fixed system re-run over the past).  A
    #: column because the horizon cannot say it: an analysis cycle also
    #: publishes short forecasts, and a front end that inferred the kind
    #: from ``max_forecast_hour == 0`` called GDAS a global forecast model.
    record_kind: str = field(default="forecast", kw_only=True)
    #: Canonical analysis fields this row's own route serves, for a
    #: consumer that locates something in them -- today the cyclone
    #: seeder.  A ROW'S OWN STATEMENT, never a consumer's table: a row
    #: that names a ``packaged_profile`` states its inventory in that
    #: profile's mapping and declares nothing here, and
    #: :func:`woof.cyclone_seed.source_inventory` reads the two as one
    #: set.  So this column carries the NATIVE routes, whose inventory is
    #: in their runner rather than in a JSON mapping, and adding a model
    #: still touches no consumer.
    #:
    #: A name here is a claim about what the route serves and is only
    #: written from what this tree already records that the product
    #: publishes.  A field the mapping deliberately does not consume is
    #: not declared: it would read as available to the seeder and fail at
    #: the one place a seed is supposed to be dependable.
    seed_fields: tuple[str, ...] = ()
    #: The analysed hydrometeor masses (state names: qc, qr, qi, qs, qg)
    #: this row's own route decodes on EVERY forcing frame, which the
    #: root's specified lateral boundary therefore carries
    #: (:mod:`woof.boundary_fields`).  The same seam as ``seed_fields``: a
    #: row that names a ``packaged_profile`` states it in that profile's
    #: mapping (its declared hydrometeor fields) and declares nothing
    #: here, and :func:`woof.boundary_fields.source_boundary_species`
    #: reads the two as one set.  So this column carries the NATIVE
    #: routes, whose inventory is in their decoder rather than in a JSON
    #: mapping.  Empty keeps the boundary at water vapour only.
    boundary_species: tuple[str, ...] = ()
    # Selection metadata for regional initial and boundary conditions. The
    # source registry owns product/time semantics independently of transport.
    time_axis: str | None = None
    selection_owner: str | None = None
    selection_provider_hint: str | None = None
    ensemble_physics_arms: tuple = ()
    forecast_time_owner: str | None = None
    forecast_time_cycle_argument: bool = False
    #: The module whose ``fetch(args)`` produces this source's bytes when
    #: they are not files at a URL: a model the fetch door RUNS (the
    #: ``earth2`` row runs an NVIDIA Earth-2 model through earth2studio,
    #: table-driven by ``woof.earth2_run``).  The fetch door asks the row
    #: instead of comparing a source id.
    acquisition_owner: str | None = None
    # Parameterized sources share one route and retain a model identity.
    model_table: str | None = None
    model_key: str | None = None
    model_row: dict | None = field(default=None, compare=False)
    acquisition_area: bool = False
    ensemble_preparation_owner: str | None = None
    acquisition_prep_handoff: bool = False
    acquisition_hints: tuple = ()
    case_data_file: str | None = None
    case_data_vtable: str | None = None
    notes: str = ""

    @property
    def display_title(self) -> str:
        """The declared display name, or the id when none is declared."""

        return (self.display_name or "").strip() or self.source_id

    def __post_init__(self) -> None:
        if self.parent_family not in ("ai", "physics"):
            raise ValueError("parent_family must be ai or physics")
        if self.same_grid_pairing not in (None, "identity"):
            raise ValueError(
                "same_grid_pairing must be 'identity' or None; an unknown "
                "pairing cannot preserve the declared source indices")
        if type(self.forecast_time_cycle_argument) is not bool:
            raise TypeError("forecast_time_cycle_argument must be a boolean")
        for name in ("selection_owner", "forecast_time_owner", "acquisition_owner"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TypeError(f"{name} must name an owner or be None")
        if self.time_axis not in (None, "forecast_leads", "analysis_times", "supplied_times"):
            raise ValueError("time_axis must name forecast leads, analysis times or supplied times")
        for credential in self.credentials:
            if not isinstance(credential, SourceCredential):
                raise TypeError(
                    f"{self.source_id}: credentials must be declared "
                    "SourceCredential rows, so a consumer can read where "
                    "the credential lives instead of parsing prose")
        if self.coverage_window is not None and not isinstance(
                self.coverage_window, COVERAGE_WINDOW_TYPES):
            raise TypeError(
                f"{self.source_id}: coverage_window must be one of "
                + ", ".join(t.__name__ for t in COVERAGE_WINDOW_TYPES))
        if (self.forcing_interval_seconds is not None
                and not self.forcing_interval_seconds > 0.0):
            raise ValueError(
                f"{self.source_id}: forcing_interval_seconds must be a "
                "positive number of seconds between source valid times")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("root_target_interior_axis")
        if value["static_source"] is None:
            value.pop("static_source")
        if not value["runtime_surface_fields"]:
            value.pop("runtime_surface_fields")
        if value["same_grid_pairing"] is None:
            value.pop("same_grid_pairing")
        value["source_kind"] = self.source_kind.value
        value["status"] = self.status.value
        # The name a consumer shows, resolved: a row that declares none
        # still answers, and nobody has to reimplement the fallback.
        value["display_name"] = self.display_title
        # DECLARATION only.  The manifest is provenance -- two boxes
        # compare theirs -- so it must not carry a verdict about which
        # keys happen to sit on the box that generated it.
        value["credentials"] = [credential_declaration(credential)
                                for credential in self.credentials]
        value["coverage_envelope"] = (
            None if self.coverage_window is None
            else list(self.coverage_window.envelope()))
        # asdict() recurses into the grid and leaves a datetime in
        # record_end, which no JSON writer takes.  DECLARED form only:
        # what the row says, never what `latest` resolves to today.
        value["cycle_grid"] = (None if self.cycle_grid is None
                               else self.cycle_grid.declaration())
        return value


def _adapter(
    source_id: str,
    *,
    name: str | None = None,
    credentials: Iterable[SourceCredential] = (),
    aliases: Iterable[str] = (),
    upstream_model_id: str | None = "__same_as_source__",
    kind: SourceKind = SourceKind.DETERMINISTIC_STATE,
    file_family: str = "GRIB2",
    default_product: str,
    required_products: Iterable[str] = (),
    max_hour: int,
    upstream_ingest: str = "registry_only",
    decoder: str | None = None,
    status: AdapterStatus = AdapterStatus.MAPPING_REQUIRED,
    field_mapping: str = "pending",
    level_mapping: str = "pending",
    cadence_mapping: str = "pending",
    stock_wrf_gate: str = "not_run",
    parent_family: str = "physics",
    upstream_model_version: str | None = None,
    lineage_donor_source: str | None = None,
    lineage_receipts: tuple[str, ...] = (),
    lineage_root_keys: tuple[str, ...] = (),
    lineage_runtime_keys: tuple[str, ...] = (),
    runnable: bool = False,
    runner: str | None = None,
    packaged_profile: str | None = None,
    canonical_preparation_source: str | None = None,
    hydro_forcing_profile: str | None = None,
    default_physics_profile: str | None = None,
    convection_permitting_physics_profile: str | None = None,
    static_source: str | None = None,
    runtime_surface_fields: tuple[tuple, ...] = (),
    same_grid_pairing: str | None = None,
    composition: str | None = None,
    member_set: str | None = None,
    ensemble_source: str | None = None,
    ensemble_control_source: str | None = None,
    requires_ensemble_calibration: bool = False,
    forcing_interval_seconds: float | None = None,
    fetch_entire_window: bool = False,
    local_posting: str | None = None,
    inland_air_mean_hours: float | None = None,
    fetch_requires_retrieve: bool = False,
    certified_source_top_pa: float | None = None,
    extendable_source_top_pa: float | None = None,
    source_top_eras: tuple[SourceTopEra, ...] = (),
    coverage: CoverageWindow | None = None,
    cycles: CycleGrid | None = None,
    archives: tuple[ArchiveWindow, ...] = (),
    root_target_interior_axis: int | None = None,
    record: str = "forecast",
    notes: str = "",
    seed_fields: tuple[str, ...] = (),
    boundary_species: tuple[str, ...] = (),
    time_axis: str | None = None,
    selection_owner: str | None = None,
    selection_provider_hint: str | None = None,
    ensemble_physics_arms: tuple = (),
    acquisition_owner: str | None = None,
    forecast_time_owner: str | None = None,
    forecast_time_cycle_argument: bool = False,
    model_table: str | None = None,
    acquisition_area: bool = False,
    ensemble_preparation_owner: str | None = None,
    acquisition_prep_handoff: bool = False,
    acquisition_hints: tuple = (),
    case_data_file: str | None = None,
    case_data_vtable: str | None = None,
) -> SourceAdapter:
    return SourceAdapter(
        source_id=source_id,
        aliases=tuple(aliases),
        upstream_model_id=(
            source_id if upstream_model_id == "__same_as_source__" else upstream_model_id
        ),
        source_kind=kind,
        file_family=file_family,
        decoder=decoder or (
            "rusty-weather/grib-core"
            if file_family in {"GRIB1", "GRIB2"}
            else "rusty-weather/rustwx-io"
        ),
        default_product=default_product,
        required_products=tuple(required_products) or (default_product,),
        max_forecast_hour=max_hour,
        upstream_ingest=upstream_ingest,
        status=status,
        field_mapping=field_mapping,
        level_mapping=level_mapping,
        cadence_mapping=cadence_mapping,
        stock_wrf_gate=stock_wrf_gate,
        parent_family=parent_family, upstream_model_version=upstream_model_version,
        lineage_donor_source=lineage_donor_source,
        lineage_receipts=lineage_receipts, lineage_root_keys=lineage_root_keys,
        lineage_runtime_keys=lineage_runtime_keys,
        runnable=runnable,
        runner=runner,
        packaged_profile=packaged_profile,
        canonical_preparation_source=canonical_preparation_source,
        hydro_forcing_profile=hydro_forcing_profile,
        default_physics_profile=default_physics_profile,
        convection_permitting_physics_profile=(
            convection_permitting_physics_profile),
        static_source=static_source,
        runtime_surface_fields=runtime_surface_fields,
        same_grid_pairing=same_grid_pairing,
        composition_requirement=composition,
        member_set=member_set,
        ensemble_source=ensemble_source,
        ensemble_control_source=ensemble_control_source,
        requires_ensemble_calibration=requires_ensemble_calibration,
        forcing_interval_seconds=forcing_interval_seconds,
        fetch_entire_window=fetch_entire_window,
        local_posting=local_posting,
        inland_air_mean_hours=inland_air_mean_hours,
        fetch_requires_retrieve=fetch_requires_retrieve,
        certified_source_top_pa=certified_source_top_pa,
        extendable_source_top_pa=extendable_source_top_pa,
        source_top_eras=tuple(source_top_eras),
        coverage_window=coverage,
        cycle_grid=cycles,
        archive_windows=archives,
        root_target_interior_axis=root_target_interior_axis,
        record_kind=record,
        display_name=name,
        credentials=tuple(credentials),
        notes=notes,
        seed_fields=tuple(seed_fields),
        boundary_species=tuple(boundary_species),
        time_axis=time_axis, selection_owner=selection_owner,
        selection_provider_hint=selection_provider_hint,
        ensemble_physics_arms=ensemble_physics_arms,
        acquisition_owner=acquisition_owner, model_table=model_table,
        acquisition_area=acquisition_area,
        ensemble_preparation_owner=ensemble_preparation_owner,
        acquisition_prep_handoff=acquisition_prep_handoff,
        acquisition_hints=acquisition_hints,
        forecast_time_owner=forecast_time_owner,
        forecast_time_cycle_argument=forecast_time_cycle_argument,
        case_data_file=case_data_file, case_data_vtable=case_data_vtable,
    )


#: The 3 km CONUS Lambert, as the grid-definition section of both HRRR's
#: and RRFS's bytes declares it.  ONE constant for both rows because the
#: two grids were measured byte-identical in every geolocating octet
#: (gauntlet staging, rrfs/DIFF-MEASURED-rrfs-vs-hrrr.txt): RRFS is HRRR's
#: operational successor on HRRR's grid, and writing the numbers twice
#: would invite them to drift.  Reproduces
#: :func:`woof.ingest.hrrr_target.hrrr_coverage_envelope` exactly, which
#: is gated in tests/test_source_wizard_facts.py.
_CONUS_3KM_LAMBERT = LambertGridWindow(
    nx=1799, ny=1059,
    lat1=21.138123, lon1=-122.719528,
    dx_m=3000.0, truelat1=38.5, truelat2=38.5, stand_lon=-97.5)

#: AWIPS grid 221, RAP's awip32 product: 32 km Lambert over North America.
#: Measured from the real f00 bytes (gauntlet staging, rap/facts.md:
#: GDT 30, nx=349 ny=277, lat1=1.0 lon1=214.5, dx=dy=32463 m,
#: latin1=latin2=50.0, lov=253.0).
_NORTH_AMERICA_32KM_LAMBERT = LambertGridWindow(
    nx=349, ny=277,
    lat1=1.0, lon1=-145.5,
    dx_m=32463.0, truelat1=50.0, truelat2=50.0, stand_lon=-107.0)

#: AWIPS grid 130. Section 3 of the 2026-10-02 18Z native-level object.
_CONUS_13KM_LAMBERT = LambertGridWindow(
    nx=451, ny=337, lat1=16.281, lon1=-126.138,
    dx_m=13545.0, truelat1=25.0, truelat2=25.0, stand_lon=-95.0)

#: DWD's ICON-EU regular lat-lon window, exactly as DWD publishes it and as
#: the preparation stage's own out-of-grid refusal reports it: 1377 x 657
#: points at 0.0625 degrees, lon -23.5..62.5, lat 29.5..70.5.  This window
#: is why the wizard can now refuse a central-US ICON-EU plan in one
#: sentence instead of after the whole acquisition.
_ICON_EU_WINDOW = RegularLatLonWindow(
    south=29.5, west=-23.5, north=70.5, east=62.5, nx=1377, ny=657)

#: Where ICON-D2 publishes values, as a window a whole domain can sit in.
#: The model domain is a tilted quadrilateral over Germany and its
#: neighbours, and DWD masks its 13 km lateral boundary strip in every
#: record, so the published cells do not fill any lat/lon box.  The largest
#: box they fill is lat 43.66..57.70, lon -0.42..17.64 (measured from the
#: 2026-09-27 12Z 0.02-degree bitmap); this window is that box shrunk by the
#: normalization halo (0.25 degrees) and one more grid step, so a domain
#: inside it never asks the remapper for a cell DWD did not publish.  The
#: counts are the window's points at the 0.02-degree step the preparation
#: remaps onto, so a refusal names the index the preparation would read.
_ICON_D2_WINDOW = RegularLatLonWindow(
    south=44.0, west=0.0, north=57.3, east=17.3, nx=866, ny=666)


#: The personal Copernicus CDS API key, declared as the row fact it is.
#:
#: The leaf filename repeats :data:`woof.fetch.CDSAPIRC_NAME` rather
#: than importing it, because :mod:`woof.fetch` reaches this registry
#: and the acquisition layer must not become an import prerequisite of
#: the table that describes it.  The same rule the cadence columns
#: follow: a repeat guarded by a test is not a second definition, and
#: ``tests/test_source_registry_columns.py`` fails the moment the
#: declared location stops naming the file the fetch route stats.
_COPERNICUS_CDS_KEY = SourceCredential(
    credential_id="copernicus-cds-api-key",
    display_name="Copernicus CDS key",
    location_kind=CredentialLocation.HOME_FILE,
    location=".cdsapirc",
    needed_for="acquisition",
    breakage=(
        "the retrieval cannot be requested at all -- this source is not "
        "downloaded by woof but by the provider's own client, which "
        "reads a personal account key from that file and fails several "
        "commands later with its own exception if it is absent"
    ),
    obtain_url="https://cds.climate.copernicus.eu",
)

#: The analysis-ready ERA5 store: the same reanalysis on its 0.25 degree
#: grid, hourly, 37 pressure levels, read without an account key through
#: woof.era5_arco.  Its ``.zattrs`` carries the store's own bounds,
#: ``valid_time_stop_era5t`` naming the last preliminary day it holds.
_ERA5_ARCO_ARCHIVE = ArchiveWindow(
    "arco", "1940-01-01T00",
    "Google's analysis-ready ERA5 store holds the reanalysis from 1940, "
    "extended with preliminary ERA5T days as they are published; the "
    "store publishes its own last day.",
    ("https://cloud.google.com/storage/docs/public-datasets/era5",
     "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
     "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs"),
    checked_at="2026-09-25",
    user_note="Recent dates are preliminary ERA5T; their publication time can vary.",
    bounds_url=("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
                "full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs"),
    bounds_stop_keys=("valid_time_stop_era5t", "valid_time_stop"),
    # MEASURED 2026-09-30: the store's valid_time_stop_era5t read
    # 2026-09-23 at 01Z and 2026-09-24 at 04Z, so its last hour trailed
    # real time by 125 to 146 h across one daily refresh, a day behind the
    # CDS's five.  Seven days also covers one missed refresh.  Used only
    # when the bounds document above cannot be read; when it can, its
    # own last hour is the answer.
    publication_lag_hours=168.0,
)

_ERA5_ARCHIVE = ArchiveWindow(
    "cds", "1940-01-01T00",
    "ERA5 record begins in 1940. Recent dates are preliminary ERA5T; "
    "the approximately five-day publication delay is not guaranteed. "
    "Model levels require the same-hour surface donor as well.",
    ("https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview",
     "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview",
     "https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation"),
    user_note="Recent dates are preliminary ERA5T; their publication time can vary.",
)
_GLOBAL_PGRB2_ARCHIVE = ArchiveWindow(
    "s3", "2021-03-22T12",
    "The current /atmos/ object layout starts at the 2021-03-22 12Z "
    "implementation. Older archive objects use different paths; this is "
    "a transport-layout boundary, not the start of the scientific record.",
    ("https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs/implementations.php",),
)
_HRRR_NATIVE_ARCHIVE = ArchiveWindow(
    "s3", "2014-07-30T18",
    "The archive's first day has the native and pressure companion f00 "
    "objects at 18Z. Historical product versions still need the current "
    "source compatibility checks; every cycle is not guaranteed.",
    ("https://registry.opendata.aws/noaa-hrrr-pds/",
     "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/?list-type=2&max-keys=2&prefix=hrrr.20140730/conus/hrrr.t18z.wrfnatf00",
     "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/?list-type=2&max-keys=2&prefix=hrrr.20140730/conus/hrrr.t18z.wrfprsf00"),
)

#: HRRR's native model top by era, MEASURED 2026-10-27 from the archive's
#: own wrfnat f00 files (PRES on hybrid level 50, the top mass level, read
#: by byte range on noaa-hrrr-bdp-pds): 2230.3 Pa on every cycle checked
#: from the v3 start (2018-07-13 12Z) through 2020-12-01 23Z, and
#: 2139-2238 Pa on the v1/v2 sigma grid (2015-06-01, 2017-06-01 12Z); then
#: 1731.5 Pa from the 2020-12-02 00Z cycle on (2020-12-02 00/06/09-12Z,
#: 2024-06-01 12Z).  On HRRR's own eta ladder (the hrrr_v4_gsd41 recipe's)
#: those are the top levels of a 2000 Pa and a 1500 Pa model top.  The
#: same switch measured on the preparation (updraft-factory, 2026-10-27):
#: 2020-05-22, 2020-08-10, 2019-06-20 and 2019-03-24 12Z refused p_top
#: 1500 Pa at root_initialize, 2020-05-22 prepared at 2000 Pa, and
#: 2022-06-11/15 prepared at 1500 Pa.
_HRRR_NATIVE_TOP_ERAS = (
    SourceTopEra(
        "", 2000.0,
        "HRRR v1-v3 (to the 2020-12-01 23Z cycle) ran with a 20 hPa model "
        "top: wrfnat hybrid level 50 reads 2230.3 Pa (v3) on the "
        "archive's own files"),
    SourceTopEra(
        "2020-12-02T00", 1500.0,
        "HRRR v4 raised the model top to 15 hPa: wrfnat hybrid level 50 "
        "reads 1731.5 Pa from the 2020-12-02 00Z cycle on"),
)


#: The next six bounds were MEASURED on 2026-09-25 by asking each archive
#: for the exact object this program's route reads (the f000 of the
#: route's own path template) and bisecting on the day: the first day
#: named holds it, the day before does not.  The listing URL is the
#: evidence and reproduces the answer.  Each is where the CURRENT layout
#: begins, so a date before it answers no in plain words at the date
#: picker instead of failing at the download.
_GEFS_ARCHIVE = ArchiveWindow(
    "aws", "2020-09-23T12",
    "The GEFS v12 atmos/pgrb2ap5 layout this route reads starts with the "
    "2020-09-23 implementation; older objects in the bucket are GEFS v11 "
    "under other paths. Not every cycle of the first days is present.",
    ("https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&max-keys=5&prefix=gefs.20200923/12/atmos/pgrb2ap5/gec00",
     "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php"),
    checked_at="2026-10-20",
)
_AIGEFS_ARCHIVE = ArchiveWindow(
    "aws", "2025-06-01T00",
    "The ensemble's own bucket begins on 2025-06-01; the product did not "
    "exist before it.",
    ("https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=3&prefix=EAGLE_ensemble/",),
    checked_at="2026-09-25",
)
_ECMWF_ENS_ARCHIVE = ArchiveWindow(
    "aws", "2024-03-18T12",
    "Archived 0.25-degree ENS members with nine pressure levels and four "
    "soil layers are read from the open-data AWS replica. Its date listing "
    "begins 2023-01-18, but earlier soil inventories do not satisfy this "
    "preparation. CC-BY-4.0 and ECMWF Terms of Use. TIGGE is separate, "
    "CC-BY-NC-4.0 research-only data, never an automatic fallback.",
    ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20240318/12z/ifs/0p25/enfo/20240318120000-0h-enfo-ef.index",
     "https://www.ecmwf.int/en/forecasts/datasets/open-data",
     "https://apps.ecmwf.int/datasets/licences/tigge/"),
    checked_at="2026-10-20",
)
_AIFS_OPEN_ARCHIVE = ArchiveWindow(
    "aws", "2025-02-25T06",
    "AIFS single entered operations on 2025-02-25; the mirror's "
    "aifs-single objects begin at its 06Z cycle.",
    ("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/?list-type=2&max-keys=5&prefix=20250225/06z/aifs-single/0p25/oper/",
     "https://www.ecmwf.int/en/forecasts/datasets/open-data"),
    checked_at="2026-09-25",
)
_RAP_ARCHIVE = ArchiveWindow(
    "aws", "2021-04-26T00",
    "The mirror's rap.YYYYMMDD awip32 objects begin on 2021-02-22 with the "
    "00 to 09 UTC runs only (their whole forecasts, f00 on); every hour of "
    "the day begins on 2021-04-26 00 UTC. 2021-02-21 holds none. Older RAP "
    "runs are kept only by NCEI, which this route does not read.",
    ("https://noaa-rap-pds.s3.amazonaws.com/?list-type=2&delimiter=/&prefix=rap.2021022",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210222/rap.t00z.awip32f00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210425/rap.t09z.awip32f00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210425/rap.t10z.awip32f00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210426/rap.t12z.awip32f00.grib2",
     "https://registry.opendata.aws/noaa-rap/"),
    checked_at="2026-09-25",
    early_start="2021-02-22T00", early_hours=tuple(range(10)),
)
#: MEASURED 2026-10-10 the same way, on the native-level product this
#: route reads (rap.tHHz.awp130bgrbf00.grib2): none on 2021-02-21; from
#: 2021-02-22 the 00 to 08 UTC runs every day and 09 UTC on some days only
#: (absent 02-22, 02-23 and 02-26, present 02-24, 02-25 and 03-01), so
#: the early hours named are the ones every day holds; every hour from
#: 2021-04-26 00 UTC (2021-04-25 holds 00 to 09 only).
_RAP_NATIVE_ARCHIVE = ArchiveWindow(
    "aws", "2021-04-26T00",
    "The mirror's rap.YYYYMMDD awp130bgrb objects begin on 2021-02-22 with "
    "the 00 to 08 UTC runs (09 UTC on some days only); every hour of the "
    "day begins on 2021-04-26 00 UTC. 2021-02-21 holds none. Older RAP "
    "runs are kept only by NCEI, which this route does not read.",
    ("https://noaa-rap-pds.s3.amazonaws.com/rap.20210221/rap.t00z.awp130bgrbf00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210222/rap.t08z.awp130bgrbf00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210222/rap.t09z.awp130bgrbf00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210425/rap.t10z.awp130bgrbf00.grib2",
     "https://noaa-rap-pds.s3.amazonaws.com/rap.20210426/rap.t10z.awp130bgrbf00.grib2",
     "https://registry.opendata.aws/noaa-rap/"),
    checked_at="2026-10-10",
    early_start="2021-02-22T00", early_hours=tuple(range(9)),
)
_RRFS_OPS_ARCHIVE = ArchiveWindow(
    "aws", "2026-08-12T00",
    "The operational RRFS bucket begins on 2026-08-12; the old rrfs_a/ "
    "prefix is gone and the prototype bucket holds retro output and samples.",
    ("https://noaa-rrfs-ops-pds.s3.amazonaws.com/?list-type=2&max-keys=5&prefix=rrfs.20260812/00/rrfs.t00z.prslev",),
    checked_at="2026-09-25",
)


#: Publisher doors that keep a rolling window with no archive behind them,
#: measured 2026-09-25 by listing each door.  The fetch route table
#: declares the same hours on the same host rows (held equal by
#: tests/test_fetch_passed_cycle.py), so the date guidance, the fetch
#: ladder and the ``--cycle latest`` walk-back read one measurement.
_ECMWF_OPEN_DOOR = ArchiveWindow(
    "ecmwf", "",
    "ECMWF's own open-data door keeps its last four days of cycles; older "
    "cycles are read from the mirror.",
    ("https://data.ecmwf.int/forecasts/",),
    checked_at="2026-09-25", retention_hours=72.0,
)
_ICON_EU_DOOR = ArchiveWindow(
    "dwd", "",
    "DWD's open-data door keeps about one day of ICON-EU cycles and there "
    "is no archive behind it.",
    ("https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t/",),
    checked_at="2026-09-25", retention_hours=24.0,
)
_ICON_D2_DOOR = ArchiveWindow(
    "dwd", "",
    "DWD's open-data door keeps the newest run of each of ICON-D2's eight "
    "daily cycles, replacing a run when the same hour's next run starts "
    "to publish, and there is no archive behind it.",
    ("https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/t_2m/",),
    checked_at="2026-09-27", retention_hours=24.0,
)
_MSC_DATAMART_DOOR = ArchiveWindow(
    "msc", "",
    "MSC Datamart keeps 30 day directories with the oldest partly pruned, "
    "so 29 days are counted; there is no archive behind it.",
    ("https://dd.weather.gc.ca/",),
    checked_at="2026-09-25", retention_hours=696.0,
)


# The order is canonical: the 23 rusty-weather ModelId values, the ERA5 GRIB1
# source already decoded by woof's native ingest path, the packaged 20CRv3
# member profile, its complementary metadata-driven NetCDF-CF primitives, then
# the declarative format-level adapter exposed by RW-WPS.


_ADAPTERS = (
    _adapter(
        "aorc", name="NOAA AORC catchment forcing",
        upstream_model_id=None, kind=SourceKind.HYDRO_FORCING,
        file_family="NetCDF", default_product="forcing", max_hour=0,
        hydro_forcing_profile="aorc-grid", forcing_interval_seconds=3600.0,
        notes="Hydro forcing declaration; local NetCDF input, area weights and time origin required.",
    ),
    _adapter(
        "era5-land", name="ECMWF ERA5-Land hourly catchment forcing",
        upstream_model_id=None, kind=SourceKind.HYDRO_FORCING,
        file_family="NetCDF", default_product="forcing", max_hour=0,
        hydro_forcing_profile="era5-land-hourly", forcing_interval_seconds=3600.0,
        notes="CDS reanalysis-era5-land netCDF4 (CC BY 4.0, engine CDS credential); valid-cell area weights and time origin required.",
    ),
    _adapter(
        "ngen-csv", name="NextGen catchment CSV forcing", aliases=("aorc-csv",),
        upstream_model_id=None, kind=SourceKind.HYDRO_FORCING,
        file_family="CSV", default_product="forcing", max_hour=0,
        hydro_forcing_profile="ngen-catchment-csv", forcing_interval_seconds=3600.0,
        notes="Ordered per-catchment CSV inputs; native ngen specific humidity and precip_rate units.",
    ),
    _adapter(
        "nwm-retrospective-forcing", name="NOAA NWM retrospective forcing",
        aliases=("nwm-forcing",), upstream_model_id=None,
        kind=SourceKind.HYDRO_FORCING, file_family="NetCDF",
        default_product="forcing", max_hour=0,
        hydro_forcing_profile="decoded-canonical-grid", forcing_interval_seconds=3600.0,
        notes="RAINRATE is kg m-2 s-1 (numerically mm/s); native grid area weights required.",
    ),
    _adapter(
        "hrrr",
        hydro_forcing_profile="decoded-canonical-grid",
        static_source="hrrr-conus-v4",
        same_grid_pairing="identity",
        runtime_surface_fields=(("VEGFRA", "sfc", "VEG:surface", "percent", (
            ("discipline", 2), ("category", 0), ("parameter", 4),
            ("level_type", 1), ("level_value", 0), ("pdt", 0))),),
        requires_ensemble_calibration=True,
        forecast_time_owner="hrrr_forecast_hours",
        forecast_time_cycle_argument=True,
        fetch_entire_window=True,
        # The native route serves the sea-level field too: `sfc` carries
        # MSLMA, which is the sea-level pressure this product publishes
        # (wx-core's HRRR model config names it as such), so the seeder's
        # first rung is reachable here exactly as it is on the other two
        # native rows.  Without it a supplied HRRR analysis fell through
        # to the 850 hPa vorticity rung with "MSLP is not declared".
        seed_fields=('air_pressure', 'air_temperature', 'eastward_wind',
                     'northward_wind', 'mean_sea_level_pressure'),
        # The native decoder reads all five masses from every wrfnat
        # frame (woof/ingest/hrrr.py _ATMOSPHERE_3D), so the root's
        # boundary carries them as the start state does.
        boundary_species=("qc", "qr", "qi", "qs", "qg"),
        archives=(_HRRR_NATIVE_ARCHIVE,),
        source_top_eras=_HRRR_NATIVE_TOP_ERAS,
        name="HRRR (native hybrid levels)",
        default_product="sfc",
        required_products=("sfc", "nat"),
        max_hour=48,
        upstream_ingest="full",
        status=AdapterStatus.CERTIFIED,
        field_mapping="hrrr-native-state-v1",
        level_mapping="native-input-to-explicit-wrf-eta-v2",
        cadence_mapping="consecutive-hourly-f00-f12-v1",
        stock_wrf_gate=(
            "wrf-v4.6.1-pass-full-f00-f12-dynamic-lambert-geometry"
        ),
        runnable=True,
        runner="hrrr_f00_f12_v1",
        root_target_interior_axis=NATIVE_TARGET_INTERIOR_AXIS,
        # Held equal to woof.hrrr_route_inputs.
        # ROUTE_DEFAULT_PHYSICS_PROFILE by test, and spelled out here
        # rather than imported because this table is read while the
        # route module is not: the menu asks the adapter what it
        # recommends (woof.physics_menu._recommended_profile).
        default_physics_profile="thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1",
        forcing_interval_seconds=3600.0,
        # Hourly, and the walk-back is short on purpose: the
        # operational directories turn over quickly, and a cycle half
        # a day old is not an initialization anyone wants for a
        # convection-permitting run.
        cycles=CycleGrid(
            hours=tuple(range(24)), search_hours=12,
            # The synoptic cycles run out to f048 and the rest stop at
            # f018, so a window longer than 18 h cannot be served by an
            # off-synoptic cycle at all.  Same shape the fetch route
            # table states a horizon in; held in step with
            # woof.hrrr_forecast's constants by a test.
            horizons=(((0, 6, 12, 18), 48), (None, 18)),
            # The measured lead-by-lead times are the hrrr row of the
            # route table's legacy_posting; this is past that row's
            # latest posting seen of every cycle's last lead
            # (tests/test_source_posting_rows.py holds it there).
            usual_delay_hours=3.0,
            basis="HRRR initializes every hour; publication is "
                  "decided by the per-object completeness probe, "
                  "not by a declared delay; a whole run is usually "
                  "published within 3 hours of its start (measured "
                  "2026-09-26 on the public mirror)"),
        coverage=_CONUS_3KM_LAMBERT,
        notes=(
            "Certified slice: one CONUS Lambert specified domain, WSM6, YSU, "
            "classic MM5 surface layer (option 91), Noah, and an explicit "
            "validated shared eta grid; "
            "the accepted 49-level profile remains the stock-WRF authority "
            "while unlike counts are covered by structural gates; "
            "stock-WRF proofs cover 192x160 at 3 km in Oklahoma and Ohio and "
            "1000x1000 at 1 km in Oklahoma."
        ),
    ),
    _adapter(
        "hrrr-prs",
        static_source="hrrr-conus-v4",
        same_grid_pairing="identity",
        default_physics_profile=(
            "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
        ),
        archives=(ArchiveWindow("aws", _HRRR_NATIVE_ARCHIVE.start,
                                _HRRR_NATIVE_ARCHIVE.note,
                                _HRRR_NATIVE_ARCHIVE.documentation),),
        name="HRRR (pressure levels)",
        aliases=("hrrr-pressure", "hrrr-wrfprs"),
        upstream_model_id="hrrr",
        file_family="GRIB2",
        decoder=(
            "packaged HRRR pressure-level profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="wrfprs",
        required_products=("wrfprs",),
        max_hour=48,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-hrrr-prs-grib2-v1",
        level_mapping="39-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="hrrr-prs-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_CONUS_3KM_LAMBERT,
        notes=(
            "HRRR's public wrfprs product through the GENERIC mapped route: "
            "one wrfprs file per valid time carries the 39-level pressure "
            "state with its five hydrometeor masses (cloud water, cloud "
            "ice, rain, snow, graupel), the surface/2m/10m fields, in-band "
            "terrain and the nine-node RUC soil column.  The Lambert CONUS "
            "grid, the "
            "grid-relative wind rotation and the node soil geometry are "
            "declared in the packaged mapping/composition documents -- "
            "there is no HRRR decode code on this route.  Distinct from "
            "--source hrrr, the certified native hybrid-level route: this "
            "profile initialises from the pressure-level analysis, which "
            "is smoother near terrain than native levels, and it is not "
            "yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "gem-gdps",
        archives=(_MSC_DATAMART_DOOR,),
        name="GEM GDPS (Canadian global)",
        # `gem` is an ALIAS, not only the upstream model id.  The 2026-08-17
        # model battery measured `--source gem` refusing by name on every
        # door while every document called the model GEM, so the one word a
        # reader would type was the one word nothing accepted.  Table work,
        # not a code path.
        aliases=("gem", "gdps", "gem-global"),
        upstream_model_id="gem",
        file_family="GRIB2",
        decoder=(
            "packaged GDPS pressure-level profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="15km-latlon0.15",
        required_products=("15km-latlon0.15",),
        max_hour=240,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gem-gdps-grib2-v1",
        level_mapping="33-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gem-gdps-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "ECCC's GDPS (GEM global) through the GENERIC mapped route: "
            "MSC Datamart's 15 km regular lat-lon GRIB2 product, one "
            "single-message JPEG2000-packed file per variable-level, "
            "concatenable per valid time.  33 pressure levels "
            "(1015..1 hPa), surface/2 m/10 m state, in-band terrain and "
            "the single 0-10 cm ISBA soil layer are declared in the "
            "packaged mapping/composition documents -- there is no GDPS "
            "decode code on this route.  GDPS publishes orography, land "
            "mask and the ice analysis at analysis time only; the "
            "profile declares them once-per-cycle invariants and the "
            "generic engine broadcasts the proven analysis record.  "
            "Soil moisture is published as volumetric content, so the "
            "anticipated layer-mass derivation was not needed.  Model "
            "state is three-hourly; not yet accepted by unchanged stock "
            "WRF."
        ),
    ),
    _adapter(
        "gem-hrdps",
        archives=(_MSC_DATAMART_DOOR,),
        name="GEM HRDPS (Canadian high-resolution, 2.5 km)",
        aliases=("hrdps", "gem-hrdps-continental", "hrdps-continental"),
        upstream_model_id="gem",
        file_family="GRIB2",
        decoder=(
            "packaged HRDPS pressure-level profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="continental-2.5km-rlatlon0.0225",
        required_products=("continental-2.5km-rlatlon0.0225",),
        max_hour=48,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gem-hrdps-grib2-v1",
        level_mapping="28-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gem-hrdps-grib2-v1",
        forcing_interval_seconds=3600.0,
        notes=(
            "ECCC's HRDPS (GEM continental 2.5 km, Canada and the "
            "northern United States) through the GENERIC mapped route: MSC "
            "Datamart's rotated lat-lon GRIB2 product (GDT 3.1, 2540 x "
            "1290 at 0.0225 degree), one single-message JPEG2000 file per "
            "variable-level, hourly to 48 h, four cycles a day.  The "
            "rotated lattice, its grid-relative wind rotation, 28 pressure "
            "levels (1015..50 hPa), surface/2 m/10 m state, in-band "
            "terrain, snow water equivalent and depth, and the single "
            "0-10 cm soil layer are declared in the packaged mapping/"
            "composition documents -- there is no HRDPS decode code on "
            "this route.  Land mask and ice analysis exist at the analysis "
            "lead only and are broadcast as once-per-cycle invariants.  "
            "Covers only HRDPS's continental domain; a target outside it is "
            "refused by the coverage check.  Data: ECCC Data Server End-use "
            "Licence (commercial use allowed, attribution 'Data Source: "
            "Environment and Climate Change Canada').  Not yet accepted by "
            "unchanged stock WRF."
        ),
    ),
    _adapter(
        "gem-rdps",
        archives=(_MSC_DATAMART_DOOR,),
        name="GEM RDPS (Canadian regional, 10 km)",
        aliases=("rdps", "gem-regional"),
        upstream_model_id="gem",
        file_family="GRIB2",
        decoder=(
            "packaged RDPS pressure-level profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="10km-rlatlon0.09",
        required_products=("10km-rlatlon0.09",),
        max_hour=84,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gem-rdps-grib2-v1",
        level_mapping="33-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gem-rdps-grib2-v1",
        forcing_interval_seconds=3600.0,
        notes=(
            "ECCC's RDPS (GEM regional 10 km, North America) through the "
            "GENERIC mapped route: MSC Datamart's rotated lat-lon GRIB2 "
            "product (GDT 3.1, 1140 x 1045 at 0.090298 degree), GDPS's "
            "file naming and 33-level field set, hourly to 84 h, four "
            "cycles a day.  The rotated lattice and its grid-relative wind "
            "rotation are declared in the packaged mapping; orography, land "
            "mask and ice analysis exist at the analysis lead only and are "
            "broadcast as once-per-cycle invariants.  Data: ECCC Data "
            "Server End-use Licence (commercial use allowed, attribution "
            "'Data Source: Environment and Climate Change Canada').  Not "
            "yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "icon-global",
        name="ICON global 13 km (DWD)",
        aliases=("icon", "icon-13km", "dwd-icon", "dwd-icon-global"),
        upstream_model_id="icon",
        default_product="icosahedral-pressure-level",
        required_products=("pressure-level", "single-level", "soil-level",
                           "time-invariant", "grid-coordinates"),
        max_hour=180,
        decoder="gdt101_remap + vendored grib-core + mapped engine",
        upstream_ingest="native_gdt101_normalization_then_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-icon-global-grib2-v1",
        level_mapping="18-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="selected-uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="icon-global-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "DWD global R03B07, nominal 13 km, 2,949,120 native cells. "
            "A native Rust input normalizer verifies the external grid "
            "identity, remaps only the WPS domain plus a halo to 0.125 "
            "degrees, and seals raw-to-normalized provenance before prep. "
            "18 pressure levels (30..1000 hPa), TERRA soil and sea ice. "
            "Three-hour forcing; 00/12Z to f180, 06/18Z to f120. "
            "No observation assimilation or stock-WRF certification is implied."
        ),
    ),
    _adapter(
        "icon-eu",
        archives=(_ICON_EU_DOOR,),
        name="ICON-EU (DWD regional)",
        aliases=("dwd-icon-eu", "icon-eu-regular"),
        upstream_model_id="icon-eu",
        file_family="GRIB2",
        decoder=(
            "packaged ICON-EU regular-lat-lon profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="regular-lat-lon-pressure-level",
        required_products=(
            "pressure-level", "single-level", "soil-level", "time-invariant",
        ),
        max_hour=120,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-icon-eu-regular-grib2-v1",
        level_mapping="20-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="icon-eu-regular-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_ICON_EU_WINDOW,
        notes=(
            "DWD open data's ICON-EU regular-lat-lon product set through "
            "the GENERIC mapped route: one bz2-wrapped GRIB2 object per "
            "field per lead (the acquisition codec stages them "
            "transparently), the 20-level pressure ladder, once-per-cycle "
            "invariant FR_LAND and HSURF broadcast to every valid time, "
            "and the TERRA soil column -- temperature at nine depth "
            "nodes, soil water as column-integrated kg m-2 over the "
            "eight layers whose exact midpoints are those interior "
            "nodes, converted and surface-extended through the closed "
            "derivation catalog.  Everything model-shaped is rows in the "
            "packaged mapping/composition documents; there is no ICON "
            "decode code on this route.  The native icosahedral product "
            "set (GDT 101) is refused by the decoder with the grid "
            "family named.  Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "icon-d2",
        archives=(_ICON_D2_DOOR,),
        name="ICON-D2 2.2 km (DWD, Germany)",
        aliases=("icon-2km", "dwd-icon-d2"),
        upstream_model_id="icon-d2",
        default_product="icosahedral-model-level",
        required_products=("model-level", "single-level", "soil-level",
                           "time-invariant", "grid-coordinates"),
        max_hour=48,
        decoder="gdt101_remap + vendored grib-core + mapped engine",
        upstream_ingest="native_gdt101_normalization_then_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-icon-d2-grib2-v1",
        level_mapping="65-height-model-level-to-explicit-wrf-eta-v1",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="icon-d2-grib2-v1",
        forcing_interval_seconds=3600.0,
        # The measured top mass-level pressure plus a warm-season margin.
        certified_source_top_pa=6000.0,
        coverage=_ICON_D2_WINDOW,
        notes=(
            "DWD's convection-permitting ICON-D2, nominal 2.2 km, on its "
            "native R19B07 mesh of 542,040 cells over Germany and its "
            "neighbours. The same GDT-101 normalization as ICON global "
            "remaps the WPS domain plus a 0.25-degree halo to 0.02 degrees "
            "and seals raw-to-normalized provenance before prep. 65 "
            "height-based model levels with pressure and five condensate species, "
            "TERRA soil, once-per-cycle sea ice. Cycles every 3 h, each to "
            "f048, hourly forcing. Data: Deutscher Wetterdienst, CC BY 4.0. "
            "No observation assimilation or stock-WRF certification is implied."
        ),
    ),
    _adapter(
        "hrrr-ak",
        name="HRRR Alaska", aliases=("hrrrak", "hrrr-alaska"),
        default_product="sfc", required_products=("sfc", "nat"), max_hour=48,
        notes="Alaska grid/projection and field contract are not yet mapped.",
    ),
    _adapter(
        "gfs",
        hydro_forcing_profile="decoded-canonical-grid",
        ensemble_source="gefs",
        forecast_time_owner="gfs_forecast_hours",
        seed_fields=('air_pressure', 'air_temperature', 'eastward_wind',
                     'northward_wind', 'mean_sea_level_pressure'),
        archives=tuple(ArchiveWindow(
            row["transport"], row["record_start"], row["why"],
            tuple(row["documentation"]), user_note=row["license"])
            for row in fetch_endpoints.archive_rows("gfs")
            if row is next(r for r in fetch_endpoints.archive_rows("gfs")
                           if r["transport"] == row["transport"])),
        name="GFS (global, 0.25 degree)", aliases=("gfs-0p25", "gfs-0.25"),
        default_product="pgrb2.0p25", max_hour=384,
        upstream_ingest="full",
        status=AdapterStatus.CERTIFIED,
        field_mapping="gfs-pgrb2-raw-code-to-native-state-v1",
        level_mapping="21-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-gfs-forecast-series-v1",
        stock_wrf_gate=(
            "wrf-v4.6.1-pass-gfs-20260720-00z-250x200x49-f000-f003"
        ),
        runnable=True,
        runner="gfs_pgrb2_0p25_v1", canonical_preparation_source="gfs-pressure",
        forcing_interval_seconds=10800.0,
        # The certified 21-level pgrb2 ladder stops at 100 hPa.  The
        # product publishes every field up to 0.01 hPa, and the fetch
        # extends the ladder to a run's own model top, so a run asking
        # for 50 hPa fetches the 70 and 50 hPa levels by default.
        certified_source_top_pa=10000.0,
        extendable_source_top_pa=1.0,
        # No delay is declared because the completeness probe decides
        # publication object by object; a delay could only start the
        # walk after a cycle the probe would have accepted.
        cycles=CycleGrid(
            hours=(0, 6, 12, 18), search_hours=48,
            # Measured on the public mirror for every cycle of
            # 2026-09-24..26: pgrb2.0p25 f000 lands 3 h 32 min to
            # 3 h 50 min after the start and f384 5 h 12 min to
            # 5 h 30 min after it.
            usual_delay_hours=6.0,
            basis="NCEP runs the global system at 00/06/12/18 UTC; "
                  "publication is decided by the per-object "
                  "completeness probe; a whole run is usually "
                  "published within 6 hours of its start (measured "
                  "2026-09-26 on the public mirror)"),
        notes=(
            "Certified slice: one specified Lambert domain, GFS pgrb2.0p25 "
            "with complete 1000..100-hPa state and exact four Noah soil "
            "layers, WSM6, YSU, classic MM5 surface layer (option 91), Noah, "
            "and any explicit eta "
            "grid whose model top is covered by GFS. Numeric GRIB2 "
            "identifiers and both soil "
            "fixed surfaces are validated before decode; unchanged stock "
            "WRF v4.6.1 accepted the f000/f003 proof and completed a finite "
            "model step. The same named-source implementation now accepts "
            "namelist-bound static one-way Lambert hierarchies through at "
            "least d06; that broader envelope is runnable but remains "
            "separately stock-WRF gated."
        ),
    ),
    _adapter(
        "gfs-pressure", name="GFS canonical pressure initialization", aliases=(),
        default_product="pgrb2.0p25", max_hour=120,
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED, runnable=True,
        runner="mapped_composition_v1", packaged_profile="gfs-pressure-grib2-v1",
        forcing_interval_seconds=10800., certified_source_top_pa=100.,
        cycles=CycleGrid(hours=(0,6,12,18), search_hours=48, usual_delay_hours=6.,
            basis="Same operational GFS cycle and product; full canonical field inventory"),
        archives=(ArchiveWindow("aws", "2021-02-26T00", "NOAA public GFS archive",
            ("https://registry.opendata.aws/noaa-gfs-bdp-pds/",)),),
        notes="Canonical pressure representation for source-independent transforms; the default GFS runner is unchanged."
    ),
    _adapter(
        "hafs-a", name="HAFS-A pressure-level parent or storm nest", aliases=("hfsa",),
        default_product="atm.grb2", max_hour=126,
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED, runnable=True,
        runner="mapped_composition_v1", packaged_profile="hafs-pressure-grib2-v1",
        forcing_interval_seconds=10800.0,
        archives=(ArchiveWindow("aws", "2023-06-19T00", "Storm-keyed NOAA NODD archive; verify exact cycle and storm objects.",
            ("https://registry.opendata.aws/noaa-nws-hafs/",), checked_at="2026-10-24"),),
        cycles=CycleGrid(hours=(0, 6, 12, 18), search_hours=48,
            usual_delay_hours=8.0, basis="Conservative allowance; exact object publication is probed, timing unmeasured"),
        notes="Regional coverage from each GRIB grid. STORM and DOMAIN are acquisition table parameters. "
              "Five hydrometeor mass species; no published number moments. Same-cycle GDAS soil donor. "
              "Snow and sea ice use the declared GDAS donor; forecast acceptance pending.",
    ),
    _adapter(
        "hafs-b", name="HAFS-B pressure-level parent or storm nest", aliases=("hfsb",),
        default_product="atm.grb2", max_hour=126,
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED, runnable=True,
        runner="mapped_composition_v1", packaged_profile="hafs-pressure-grib2-v1",
        forcing_interval_seconds=10800.0,
        archives=(ArchiveWindow("aws", "2023-06-19T00", "Storm-keyed NOAA NODD archive; verify exact cycle and storm objects.",
            ("https://registry.opendata.aws/noaa-nws-hafs/",), checked_at="2026-10-24"),),
        cycles=CycleGrid(hours=(0, 6, 12, 18), search_hours=48,
            usual_delay_hours=8.0, basis="Conservative allowance; exact object publication is probed, timing unmeasured"),
        notes="Regional coverage from each GRIB grid. STORM and DOMAIN are acquisition table parameters. "
              "Five hydrometeor mass species; no published number moments. Same-cycle GDAS soil donor. "
              "Snow and sea ice use the declared GDAS donor; forecast acceptance pending.",
    ),
    _adapter(
        "gdas",
        record="analysis",
        forecast_time_owner="gdas_forecast_hours",
        archives=(_GLOBAL_PGRB2_ARCHIVE,),
        name="GDAS analysis (0.25 degree)", aliases=("gdas-0p25", "gdas-0.25"),
        file_family="GRIB2",
        decoder=(
            "packaged GDAS pgrb2 0.25-degree profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pgrb2.0p25", max_hour=9,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gdas-pgrb2-0p25-grib2-v1",
        level_mapping="33-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gdas-pgrb2-0p25-grib2-v1",
        forcing_interval_seconds=3600.0,
        cycles=CycleGrid(
            hours=(0, 6, 12, 18), search_hours=48,
            # Measured on the public mirror for every cycle of
            # 2026-09-24..26: pgrb2.0p25 f009 lands 7 h 1 min to
            # 7 h 20 min after the start.
            usual_delay_hours=8.0,
            basis="the analysis cycle runs on the global system's "
                  "00/06/12/18 UTC grid; publication is decided by "
                  "the per-object completeness probe; a whole cycle is "
                  "usually published within 8 hours of its start "
                  "(measured 2026-09-26 on the public mirror)"),
        notes=(
            "NCEP's analysis cycle through the GENERIC mapped route as a "
            "packaged profile: one pgrb2.0p25 file per hourly valid time "
            "(f000..f009 -- f010 is a measured 404), each carrying the "
            "33-level isobaric state with direct specific humidity, the "
            "surface/2m/10m fields, in-band terrain and the four Noah "
            "soil layers bound by their scaled type-106 depth pairs "
            "(the integer level key collides between the 0-0.1 m and "
            "0.1-0.4 m layers, so depth-pair rows are the identity).  "
            "Soil moisture is an NCEP local-table row selected by "
            "octets; no table is asked to name it.  The catalogue is "
            "record-for-record byte-identical to GFS pgrb2.0p25 "
            "(696/696 MEASURED), so this profile is the GFS shape at a "
            "different acquisition path with analysis-cycle semantics: "
            "the bytes stamp even f000 a FORECAST (generating process "
            "81 at hour 0, 96 after), and the one analysis-stamped "
            "product (pgrb2.1p00.anl) lacks all soil, the land mask and "
            "every 2 m/10 m field -- it is not an initialization route "
            "and this row does not claim it.  Publication lags the "
            "cycle by ~7 h (delayed-cutoff cycle), far later than GFS.  "
            "Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "gefs-v11-flat",
        ensemble_source="gefs-v11-flat",
        archives=(ArchiveWindow("aws", "2017-01-01T00", "Legacy flat layout; exact member and lead objects are required, not inferred from date.", ("https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&prefix=gefs.20180101/00/",), checked_at="2026-10-20"),),
        name="GEFS v11 flat archive member", aliases=(),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged GEFS ensemble-member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pgrb2a+pgrb2b/member",
        required_products=("pgrb2a", "pgrb2b"),
        max_hour=384,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gefs-v11-flat-grib2-v1",
        level_mapping="26-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gefs-v11-flat-grib2-v1",
        forcing_interval_seconds=21600.0,
        composition="Initialize from one verified member's pgrb2a+pgrb2b pair; ensemble means/spread are not trajectory states.",
        member_set="gefs-v11-flat-grib2-members-v1",
        notes="Verified 20180101/00 v11: one-degree, 21 members, six-hour leads, 26 pressure levels. Both products of one member are required. Not stock-WRF certified.",
    ),
    _adapter(
        "gefs-reforecast",
        ensemble_source="gefs-reforecast",
        archives=(ArchiveWindow("aws", "2000-01-01T00", "GEFSv12 historical reforecast; exact requested objects and indexed leads checked", ("https://registry.opendata.aws/noaa-gefs-reforecast/",), checked_at="2026-10-20"),),
        name="GEFSv12 retrospective daily member", aliases=(),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged GEFS ensemble-member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="reforecast/member",
        required_products=("pressure",),
        max_hour=384,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gefs-reforecast-grib2-v1",
        level_mapping="25-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="three-hour-through-f240-six-hour-after-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gefs-reforecast-grib2-v1",
        forcing_interval_seconds=10800.0,
        composition="Exact coarsest pressure grid; static land and same-cycle NOAA ice analysis donors.",
        member_set="gefs-reforecast-grib2-members-v1",
        notes="5 published members. f003 first instantaneous state, no f000; indexed native member verification. Not stock-WRF certified.",
    ),
    _adapter(
        "gefs-reforecast-weekly",
        ensemble_source="gefs-reforecast-weekly",
        archives=(ArchiveWindow("aws", "2000-01-01T00", "GEFSv12 historical reforecast; exact requested objects and indexed leads checked", ("https://registry.opendata.aws/noaa-gefs-reforecast/",), checked_at="2026-10-20"),),
        name="GEFSv12 retrospective weekly member", aliases=(),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged GEFS ensemble-member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="reforecast/member",
        required_products=("pressure",),
        max_hour=840,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gefs-reforecast-weekly-grib2-v1",
        level_mapping="25-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="three-hour-through-f240-six-hour-after-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gefs-reforecast-grib2-v1",
        forcing_interval_seconds=10800.0,
        composition="Exact coarsest pressure grid; static land and same-cycle NOAA ice analysis donors.",
        member_set="gefs-reforecast-weekly-grib2-members-v1",
        notes="11 published members. f003 first instantaneous state, no f000; indexed native member verification. Not stock-WRF certified.",
    ),
    _adapter("gefs-reforecast-land", name="NOAA surface donor: gefs-reforecast-land", kind=SourceKind.SURFACE_ANALYSIS, file_family="GRIB2", default_product="surface", max_hour=0, upstream_ingest="table_route_donor_only", notes="Partial surface state used only by a declared composition; cannot initialize an atmosphere alone."),
    _adapter("ncep-r1-seaice", name="NOAA surface donor: ncep-r1-seaice", kind=SourceKind.SURFACE_ANALYSIS, file_family="NetCDF", default_product="surface", max_hour=0, upstream_ingest="table_route_donor_only", notes="Partial surface state used only by a declared composition; cannot initialize an atmosphere alone."),
    _adapter(
        "gefs-v11",
        ensemble_source="gefs-v11",
        archives=(ArchiveWindow("aws", "2018-07-27T00", "Legacy product-directory layout; exact member and lead objects are required, not inferred from date.", ("https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&prefix=gefs.20190201/00/",), checked_at="2026-10-20"),),
        name="GEFS v11 legacy archive member", aliases=("gefs-legacy",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged GEFS ensemble-member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pgrb2a+pgrb2b/member",
        required_products=("pgrb2a", "pgrb2b"),
        max_hour=384,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gefs-v11-grib2-v1",
        level_mapping="26-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gefs-v11-grib2-v1",
        forcing_interval_seconds=21600.0,
        composition="Initialize from one verified member's pgrb2a+pgrb2b pair; ensemble means/spread are not trajectory states.",
        member_set="gefs-v11-grib2-members-v1",
        notes="Verified 20190201/00 v11: one-degree, 21 members, six-hour leads, 26 pressure levels. Both products of one member are required. Not stock-WRF certified.",
    ),
    _adapter(
        "gefs",
        ensemble_source="gefs",
        archives=(_GEFS_ARCHIVE,),
        name="GEFS (global ensemble member)", aliases=("gefs-ensemble",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged GEFS ensemble-member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pgrb2a+pgrb2b/member",
        required_products=("pgrb2a", "pgrb2b"),
        max_hour=384,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-gefs-ensemble-grib2-v1",
        level_mapping="31-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="gefs-ensemble-grib2-v1",
        forcing_interval_seconds=10800.0,
        composition="Initialize from one verified member's pgrb2a+pgrb2b pair; ensemble means/spread are not trajectory states.",
        member_set="gefs-ensemble-grib2-members-v1",
        notes=(
            "One MEMBER of NCEP's global ensemble through the GENERIC "
            "mapped route.  The member axis is table work: the packaged "
            "member set declares all 31 forecasts (control + 30 "
            "perturbed), the filename patterns, the byte verification "
            "triple, the encoded-size convention (30 EXCLUDES the "
            "control) and the geavg/gespr statistic namespace, and "
            "`woof-member-prep` stages a verified member with a "
            "hash-pinned receipt.  The field axis is table work too: "
            "the packaged profile's 31-level ladder is the measured "
            "exact union of the pgrb2a and pgrb2b isobaric sets (zero "
            "overlap on every variable), so a preparation needs BOTH "
            "files of the SAME member per valid time -- pass each pair "
            "with --input and the SAME pair again as --supplement: "
            "terrain is in band but MIGRATES products (measured: the "
            "analysis publishes orography in pgrb2a, every forecast "
            "step publishes it in pgrb2b), so only the pair carries it "
            "at every valid time.  Every selector "
            "pins PDT 1, so a statistic file (PDT 2) or an accumulation "
            "twin (PDT 11) can never satisfy a state selector, and the "
            "frame assembler's one-GRIB-member rule refuses a "
            "cross-member mix.  Noah soil rides the pair under the "
            "measured 66-percent ocean bitmap (layer 1 in pgrb2a, "
            "layers 2-4 in pgrb2b).  Not yet accepted by unchanged "
            "stock WRF."
        ),
    ),
    _adapter(
        "aigfs", parent_family="ai", lineage_donor_source="gdas",
        ensemble_source="aigefs",
        name="AIGFS (NOAA AI global forecast)", aliases=("ai-gfs",),
        decoder=(
            "packaged AIGFS operational profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pres", max_hour=384,
        required_products=("pres", "sfc"),
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-aigfs-gdas-hybrid-grib2-v1",
        level_mapping="13-pressure-level-atmosphere-only-v1",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="aigfs-gdas-hybrid-grib2-v1",
        forcing_interval_seconds=21600.0,
        composition=(
            "AIGFS publishes no soil temperature or moisture, no land-sea "
            "mask, no orography, no skin temperature, no surface pressure "
            "(PRMSL only) and no 2 m humidity; the hybrid profile borrows "
            "the land-surface state from the SAME CYCLE's GDAS 0.25-degree "
            "analysis (the caller's one supplement) through the "
            "cross-source composition and derives surface pressure at "
            "every lead from AIGFS's own PRMSL at the analysis terrain "
            "height, and a donor from any other cycle or grid refuses by "
            "name."
        ),
        notes=(
            "NCEP's GraphCast-based 0.25-degree AI forecast as pure table "
            "data, RUNNABLE as a HYBRID: the operational atmosphere (six "
            "3-D fields on 13 pressure levels topping at 50 hPa, plus 2 m "
            "temperature, 10 m wind and PRMSL) rides the generic mapped "
            "route while the six land-surface canonicals the product does "
            "not publish are composition_bound to the same-cycle GDAS "
            "analysis donor, decoded through the donor's own SHA-256-pinned "
            "mapping under the source_cycle_analysis_broadcast clock (one "
            "analysis record, carried to every lead, carried times named "
            "in the receipt).  Surface pressure, which AIGFS does not "
            "publish either, is derived at every lead from its own PRMSL "
            "and isobaric heights at the analysis terrain height, so the "
            "column mass follows the forecast.  The atmosphere-only profile "
            "aigfs-nomads-grib2-v1 remains shipped as the solo-refusal "
            "record.  ACQUISITION IDENTITY "
            "IS PART OF THE PRODUCT: operational bytes are NOMADS-only "
            "(SCN 25-89; 10-day rolling window, no filter CGI, no "
            "operational S3 mirror), stamped subCentre 0, while the "
            "noaa-nws-graphcastgfs-pds bucket serves a DIFFERENT "
            "experimental run under identical filenames stamped "
            "subCentre 2 -- every profile selector pins subcenter=0, so "
            "the imposter refuses by named identity octet instead of "
            "silently mixing model versions.  Deterministic (PDT 0/8 "
            "only); the PDT-8 precipitation windows are deliberately "
            "unselected (duplicate byte-identical tp at f006, shared "
            "paramId/level/step from f012).  Not flagged AI in-band: "
            "genProcId 137 and the front door are the identity."
        ),
    ),
    _adapter(
        "aigefs", parent_family="ai", lineage_donor_source="gdas",
        ensemble_source="aigefs",
        archives=(_AIGEFS_ARCHIVE,),
        name="AIGEFS (NOAA AI global ensemble member)", aliases=("ai-gefs",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged AI-ensemble member-hybrid profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="pres/member", max_hour=384,
        required_products=("sfc/member", "pres/member"),
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-aigefs-member-hybrid-grib2-v1",
        level_mapping="13-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="aigefs-member-hybrid-grib2-v1",
        forcing_interval_seconds=21600.0,
        composition="Select a member and combine pressure/surface products; averages are not member states.",
        member_set="aigefs-ensemble-grib2-members-v1",
        notes=(
            "NCEP's operational AI global ensemble (Project EAGLE), 31 "
            "members, through the GENERIC mapped route as a CROSS-SOURCE "
            "packaged profile.  The member product is a 13-level "
            "0.25-degree dry-dynamical + humidity atmosphere and NOTHING "
            "else -- no soil, no land mask, no orography, no skin "
            "temperature, no 2 m humidity -- so the packaged mapping "
            "declares those six gaps composition_bound and the packaged "
            "composition binds them to the shipped physical-analysis "
            "donor mapping under source_cycle_analysis_broadcast: the "
            "borrowed land state is the SAME cycle's analysis record, "
            "carried to every lead and named in the receipt.  Pass "
            "--input with ONE member's pres+sfc files (stage them "
            "through `woof-member-prep`, whose receipt carries the "
            "member identity the byte-identical leaf filenames cannot) "
            "and --supplement with the same cycle's 0.25-degree "
            "analysis file.  Every packaged selector pins PDT 1, the "
            "individual-member template, so deterministic bytes and "
            "ensemble mean/spread files refuse at the mapped decode; "
            "the packaged member set separately declares the "
            "verification contract (typeOfEnsembleForecast 3 on EVERY "
            "member including the control -- perturbationNumber 0 is "
            "the only control discriminator -- encoded ensemble size "
            "31, octet INCLUDES the control) and the NOMADS-only "
            "ensstat statistic namespace.  The mapped route itself "
            "cannot know WHICH member a caller intended: that identity "
            "lives in the member-prep receipt, not the leaf name.  "
            "Runnable, not stock-WRF certified, and the receipt says so."
        ),
    ),
    _adapter(
        "hgefs",
        name="Hybrid GEFS", aliases=("hybrid-gefs",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="sfc/avg", max_hour=240,
        required_products=("sfc/member", "pres/member"),
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select a member and combine pressure/surface products; averages are not member states.",
    ),
    _adapter(
        "ecmwf-open-data",
        ensemble_source="ecmwf-ens",
        archives=(_ECMWF_OPEN_DOOR,),
        name="ECMWF IFS (open data, 0.25 degree)", aliases=("ecmwf", "ifs"),
        file_family="GRIB2",
        decoder=(
            "packaged ECMWF open-data oper profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="oper", max_hour=360,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-ecmwf-open-data-oper-grib2-v1",
        level_mapping="14-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="ecmwf-open-data-oper-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "ECMWF's open-data IFS oper product through the GENERIC mapped "
            "route: one 0.25-degree global GDT-0 GRIB2 file per three-hourly "
            "step carries the 14-level pressure state (earlier publications "
            "carry 13 levels without 10 hPa, spell the soil layers on "
            "depth-below-land surfaces and carry no surface geopotential; "
            "the mapping's era ladder, record aliases and terrain "
            "derivation read them), the surface/2 m/10 m "
            "fields, in-band surface geopotential for terrain (converted to "
            "metres by the declared unit scale) and the four IFS soil "
            "layers, which the source addresses by ordinal on fixed-surface "
            "type 151 -- declared through the composition's indexed "
            "selector_depth_binding, not code.  2 m humidity derives from "
            "the published dewpoint.  Winds are earth-relative; CCSDS "
            "(DRT 42) packing decodes through the converged bridge.  "
            "LICENSING: this profile is authored against the 0.25-degree "
            "open-data distribution (CC-BY-4.0); ECMWF's native 9 km HRES "
            "is access-restricted -- a data-licensing fact, not a "
            "capability gap.  Snow and sea-ice fields remain "
            "policy-controlled (local-table/bitmap semantics not yet "
            "bound), and the route is not yet accepted by unchanged stock "
            "WRF."
        ),
    ),
    _adapter(
        "ecmwf-open-data-0p1",
        archives=(_ECMWF_OPEN_DOOR,),
        name="ECMWF IFS HRES (open data, 0.1 degree)",
        aliases=("ifs-0p1", "ifs-hres", "ecmwf-hres", "ecmwf-0p1"),
        file_family="GRIB2",
        decoder=(
            "packaged ECMWF open-data oper profile over the concatenated "
            "0.1-degree pl + sfc pair + vendored rusty-weather GRIB2 bridges"
        ),
        default_product="oper", max_hour=360,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-ecmwf-open-data-oper-grib2-v1",
        level_mapping="14-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="ecmwf-open-data-oper-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "ECMWF's IFS HRES open data at 0.1 degree, published from the "
            "2026-10-20 cycles under CC-BY-4.0, about 2 h after the "
            "0.25-degree set. Each lead is a pressure-level file and a "
            "surface file with disjoint records; the route concatenates "
            "the pair and the 0.25-degree oper profile decodes it (same "
            "14 levels, parameters and soil ordinals; the grid is read "
            "from the records). Surface geopotential is published at step "
            "0 only, so a start past step 0 takes terrain from the "
            "mapping's height-at-surface-pressure derivation, as the "
            "0.25-degree route does."
        ),
    ),
    _adapter(
        "ecmwf-ens",
        ensemble_source="ecmwf-ens",
        ensemble_control_source="ecmwf-open-data",
        archives=(_ECMWF_OPEN_DOOR, _ECMWF_ENS_ARCHIVE),
        name="ECMWF ENS (open data, member)",
        aliases=("ifs-ens", "ecmwf-ensemble"),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="enfo/member", required_products=("enfo/member",),
        max_hour=360,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-ecmwf-ens-open-grib2-v1",
        level_mapping="14-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True, runner="mapped_composition_v1",
        packaged_profile="ecmwf-ens-open-grib2-v1",
        forcing_interval_seconds=10800.0,
        member_set="ecmwf-ens-open-grib2-members-v1",
        composition=(
            "One indexed and byte-verified perturbed member supplies its "
            "atmosphere and four soil layers. The separate oper control "
            "is selected through ensemble_control_source."
        ),
        notes=(
            "The current open-data enfo-ef object contains 50 perturbed "
            "members. JSON index ranges select one member; the Rust GRIB "
            "inventory then verifies every selected message. The "
            "mapped member profile reads the same field definitions, "
            "including surface-pressure terrain derivation when surface "
            "geopotential is absent. The separately published oper-fc "
            "control makes 51 trajectories. Fetch-as-posted remains active "
            "for each member's forcing series. Historical publication "
            "layouts from 2024-03-18T12 use the AWS open-data archive, "
            "the published nine-level atmosphere and legacy soil aliases. "
            "TIGGE is CC-BY-NC-4.0 research-only, not a fallback. "
            "Data: ECMWF open data, CC BY 4.0. Not stock-WRF certified."
        ),
    ),
    _adapter(
        "aifs", parent_family="ai",
        archives=(_ECMWF_OPEN_DOOR, _AIFS_OPEN_ARCHIVE),
        name="ECMWF AIFS single (open data)",
        aliases=("aifs-v2", "aifs-single"),
        file_family="GRIB2",
        decoder=(
            "packaged AIFS single profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="oper",
        required_products=("oper",),
        max_hour=360,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-aifs-single-grib2-v1",
        level_mapping="13-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="aifs-single-grib2-v1",
        forcing_interval_seconds=21600.0,
        notes=(
            "ECMWF's AI forecast (AIFS single, open data 0.25-degree "
            "GRIB2) through the GENERIC mapped route: one file per "
            "six-hourly step carries the 13-level pressure state "
            "(50-1000 hPa; the extra 10 hPa surfaces outside the declared "
            "ladder are admitted-and-ignored, and specific humidity is "
            "not published there), the surface/2 m/10 m fields and the "
            "two-layer ordinal soil column; the land mask and surface "
            "geopotential ride the 0-hour file alone, so every lead takes "
            "them from its cycle's 0-hour file, which a later start "
            "fetches too.  THREE limits a reader must know.  (1) The "
            "soil column reaches 0.28 m: Noah's four layers are WRF's own "
            "shallow-column interpolation bracketed by the skin "
            "temperature at 0 m and the static deep-soil temperature at "
            "3 m, smoother than a full-depth soil analysis.  (2) No snow "
            "state and no sea-ice fraction are published, so runs start "
            "bare-ground and open-water everywhere; a winter or polar "
            "case should not initialise from this product.  (3) An AI "
            "emulator publishes no hydrometeors and its fields are not "
            "constrained by a dynamical balance, so spin-up behaviour is "
            "not the certified-source baseline.  Not yet accepted by "
            "unchanged stock WRF."
        ),
    ),
    _adapter(
        "earth2", parent_family="ai",
        lineage_donor_source="era5",
        lineage_receipts=("earth2-batch.json", "earth2-run.json"),
        lineage_root_keys=("earth2_batch_root",),
        lineage_runtime_keys=("earth2studio", "torch"),
        time_axis="supplied_times",
        acquisition_owner="woof.earth2_run",
        selection_owner="woof.earth2_members",
        selection_provider_hint="model",
        name="NVIDIA Earth-2 model output (Earth2Studio NetCDF)",
        aliases=("earth2studio",),
        model_table="earth2-models.v1.json", acquisition_area=True,
        acquisition_prep_handoff=True,
        acquisition_hints=(
            ("model", "string", None, "model key from this source's declared model table"),
            ("earth2_members", "string", None, "complete comma-separated noise-id roster"),
            ("earth2_batch_root", "string", None, "shared generated-member batch directory"),
            ("earth2_batch_size", "integer", None, "members per admitted GPU sub-batch"),
            ("earth2_weights_package", "string", None, "local pinned model weight package"),
        ),
        cycles=CycleGrid(tuple(range(24)), 0.0, basis="caller-selected analysis initialization"),
        upstream_model_id=None,
        file_family="NetCDF (Earth2Studio NetCDF4Backend)",
        decoder=(
            "packaged Earth-2 13-level profile + rw_netcdf/netcrust "
            "(mapped engine)"
        ),
        default_product="earth2studio deterministic forecast, NetCDF4Backend",
        required_products=(
            "one Earth2Studio forecast file: (time, lead_time, lat, lon) with "
            "z/t/u/v/q on 50..1000 hPa, t2m, u10m, v10m, msl",
            "the ERA5 analysis at that forecast's initialization (woof fetch "
            "--source era5 --era5-provider arco) over an area containing it",
        ),
        max_hour=8784,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-earth2-pl13-q-netcdf-v1",
        level_mapping="13-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-six-hour-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="earth2-pl13-q-netcdf-v1",
        forcing_interval_seconds=21600.0,
        composition=(
            "Earth-2 models publish no soil, land mask, orography, skin "
            "temperature, SST, sea ice, snow, surface pressure or 2 m "
            "humidity; the profile borrows the land-surface state from the "
            "ERA5 analysis AT the forecast's initialization (the caller's one "
            "supplement, keyless from ARCO) through the cross-source "
            "composition, and derives surface pressure at every lead from the "
            "model's own mean-sea-level pressure at the analysis terrain."
        ),
        notes=(
            "Any NVIDIA Earth-2 model run through Earth2Studio whose output "
            "carries the 13-level z/t/u/v/q vocabulary -- FourCastNet 3, SFNO, "
            "Pangu, Aurora, FengWu, GraphCast, AIFS -- as the initial and "
            "lateral boundary source of a WOOF domain, read straight from "
            "the file Earth2Studio's NetCDF4Backend writes (no reshaping).  "
            "Which models qualify, their weights and what the others need is "
            "woof/authorities/earth2-models.v1.json; `woof fetch --source "
            "earth2 --model fcn3 --cycle C --hours N` runs the model and "
            "writes the file and the analysis together.  Valid times are the "
            "file's initialization plus lead_time; the source cycle is the "
            "initialization, so a domain may start at any lead "
            "(--forecast-start-hour).  The land surface is the initialization "
            "analysis carried unchanged, the same frozen-surface class as the "
            "AI-GFS row; an AI emulator publishes no hydrometeors, so runs "
            "start cloud-free.  Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "rap-native",
        name="RAP hybrid levels (13 km CONUS)",
        aliases=("rap-awp130bgrb",), upstream_model_id="rap",
        default_product="awp130bgrb", max_hour=51,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-rap-native-grib2-v1",
        level_mapping="50-model-level-explicit-pressure-to-wrf-eta-v1",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True, runner="mapped_composition_v1",
        packaged_profile="rap-native-grib2-v1",
        archives=(_RAP_NATIVE_ARCHIVE,),
        forcing_interval_seconds=3600.0, coverage=_CONUS_13KM_LAMBERT,
        boundary_species=("qc", "qr", "qi", "qs", "qg"),
        notes=(
            "The generic mapped route reads the fifty native hybrid levels "
            "published on the 13 km CONUS Lambert grid, including explicit "
            "pressure, hydrometeors, aerosol numbers and nine RUC soil nodes. "
            "QNWFA/QNIFA use the operational HRRR Vtable's kg-1 semantics "
            "for NCEP local parameters 0/13/193 and 0/13/192. The product "
            "is horizontally interpolated by its producer; it is not RAP's "
            "full rotated native grid. Stock-WRF certification is pending."
        ),
    ),
    _adapter(
        "hrrr-native",
        static_source="hrrr-conus-v4",
        same_grid_pairing="identity",
        default_physics_profile=(
            "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
        ),
        # The route reads the AWS mirror's wrfnat, wrfsfc and same-cycle
        # wrfprs objects, all three first present at 2014-07-30 18Z
        # (measured 2026-10-10: 17Z holds none), the hrrr-prs row's window.
        archives=(ArchiveWindow("aws", _HRRR_NATIVE_ARCHIVE.start,
                                _HRRR_NATIVE_ARCHIVE.note,
                                _HRRR_NATIVE_ARCHIVE.documentation),),
        # The same wrfnat objects, so the same measured tops.
        source_top_eras=_HRRR_NATIVE_TOP_ERAS,
        name="HRRR native hybrid analysis (mapped)",
        aliases=("hrrr-mapped-native",), upstream_model_id="hrrr",
        default_product="wrfnat", max_hour=48,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-hrrr-native-grib2-v1",
        level_mapping="50-model-level-explicit-pressure-to-wrf-eta-v1",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True, runner="mapped_composition_v1",
        packaged_profile="hrrr-native-grib2-v1",
        forcing_interval_seconds=3600.0, coverage=_CONUS_3KM_LAMBERT,
        boundary_species=("qc", "qr", "qi", "qs", "qg"),
        notes=(
            "The generic mapped route reads the fifty native hybrid levels "
            "from wrfnat, with explicit pressure and aerosol numbers. Its "
            "same-cycle wrfprs analysis supplies the nine-node soil column "
            "and terrain through a hash-bound composition. This profile "
            "supports a native analysis donor independently of the selected "
            "lateral forcing source. Stock-WRF certification is pending."
        ),
    ),
    _adapter(
        "rap",
        archives=(_RAP_ARCHIVE,),
        name="RAP (32 km North America)",
        aliases=("rap-awip32",),
        file_family="GRIB2",
        decoder=(
            "packaged RAP awip32 profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="awip32",
        required_products=("awip32",),
        max_hour=51,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-rap-awip32-grib2-v1",
        level_mapping="39-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="rap-awip32-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_NORTH_AMERICA_32KM_LAMBERT,
        notes=(
            "RAP through the GENERIC mapped route as a packaged profile: "
            "one awip32 file (AWIPS grid 221, 32 km Lambert, North "
            "America) per hourly valid time carries the 39-level pressure "
            "state (the HRRR wrfprs ladder, byte for byte), the "
            "surface/2m/10m fields, in-band terrain and the nine-node RUC "
            "soil column.  Pressure-level humidity is RH and rides the "
            "declared RH-to-specific-humidity derivation the GFS profile "
            "proved.  There is no RAP decode code on this route.  The "
            "13 km hybrid product is a separate rap-native table row. "
            "The native rotated "
            "lat-lon wrfprs product (GDT 32769) is outside the declared "
            "grid families.  Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "nam",
        name="NAM (12 km North America)",
        default_product="awip12", max_hour=84),
    _adapter(
        "hiresw",
        name="HiResW (high-resolution window)", aliases=("hires",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        default_product="arw_2p5km/conus", max_hour=48,
        status=AdapterStatus.MEMBER_SELECTION_REQUIRED,
        composition="Select the concrete ARW/FV3 member before initialization.",
    ),
    _adapter(
        "href",
        name="HREF (high-resolution ensemble products)",
        aliases=("href-conus",),
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="ensprod/conus/sprd", max_hour=48,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent deterministic member; spread/probability products cannot initialize WRF.",
    ),
    _adapter(
        "sref",
        name="SREF (short-range ensemble products)",
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="ensprod/pgrb212/mean_3hrly", max_hour=87,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent member; the ensemble mean is not a dynamically balanced member state.",
    ),
    _adapter(
        "rtma",
        name="RTMA (real-time mesoscale analysis)",
        kind=SourceKind.SURFACE_ANALYSIS,
        default_product="2dvaranl_ndfd", max_hour=0,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D atmosphere source; RTMA may replace declared surface fields only.",
    ),
    _adapter(
        "urma",
        name="URMA (unrestricted mesoscale analysis)",
        kind=SourceKind.SURFACE_ANALYSIS,
        default_product="2dvaranl_ndfd", max_hour=0,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D atmosphere source; URMA may replace declared surface fields only.",
    ),
    _adapter(
        "nbm",
        name="NBM (National Blend of Models)", aliases=("blend",),
        kind=SourceKind.POSTPROCESSED_GUIDANCE,
        default_product="core/co", max_hour=264,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Provide a complete 3-D analysis/forecast state; NBM is postprocessed guidance.",
    ),
    _adapter(
        "rrfs",
        ensemble_source="rrfs-ens",
        requires_ensemble_calibration=True,
        archives=(_RRFS_OPS_ARCHIVE,),
        name="RRFS (operational 3 km CONUS)",
        aliases=("rrfs-ops",),
        file_family="GRIB2",
        decoder=(
            "packaged RRFS prslev+2dfld profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="prslev",
        required_products=("prslev", "2dfld"),
        max_hour=84,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-rrfs-prslev-2dfld-grib2-v1",
        level_mapping="45-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="rrfs-prslev-2dfld-grib2-v1",
        forcing_interval_seconds=3600.0,
        coverage=_CONUS_3KM_LAMBERT,
        notes=(
            "RRFS -- HRRR's operational successor, flowing today on "
            "noaa-rrfs-ops-pds and NOMADS rrfs/v1.0 through the GENERIC "
            "mapped route as a packaged "
            "profile.  The 3 km CONUS grid is bit-for-bit HRRR's Lambert "
            "(every geolocating octet identical, measured from real "
            "bytes), so the HRRR wrfprs machinery carries it; RRFS's own "
            "facts are table data: a 45-level pressure ladder (70 hPa "
            "where HRRR has 75, 2 hPa top, no 1013.2 hPa entry) and the "
            "state split across a prslev/2dfld file PAIR -- prslev alone "
            "carries zero surface fields and zero soil, so both products "
            "are required and both files are passed per valid time.  "
            "SPFH is direct; the nine-node RUC soil column is "
            "octet-identical to HRRR's.  There is no RRFS decode code on "
            "this route.  NOT reachable today and not claimed by this "
            "row: the natlev native-level product, the 3 km "
            "North-America rotated grid and the thinned subset files "
            "are not exposed by this route. The older prototype bucket "
            "noaa-rrfs-pds has retrospective data; its rrfs_a prefix "
            "was empty when checked on 2026-10-31. The relocatable firewx "
            "nest changes grid daily. The public rrfsens member subset "
            "has its own rrfs-ens row; its 250 hPa top and missing land "
            "state do not replace this full-column route. "
            "Not yet accepted by unchanged "
            "stock WRF."
        ),
    ),
    _adapter(
        "rrfs-ens",
        ensemble_source="rrfs-ens",
        name="RRFS ensemble (public member subset)", aliases=("rrfsens",),
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged RRFS member-with-companion profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="prslevnomads+2dfldnomads/member",
        required_products=("prslevnomads", "2dfldnomads", "prslev", "2dfld"),
        max_hour=60,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-rrfs-ens-member-companion-grib2-v1",
        level_mapping="28-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-hourly-forecast-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="rrfs-ens-member-companion-grib2-v1",
        member_set="rrfs-ops-subset-grib2-members-v1",
        forcing_interval_seconds=3600.0,
        coverage=_CONUS_3KM_LAMBERT,
        composition=(
            "Initialize from one verified member's prslevnomads+2dfldnomads "
            "pair together with its declared companion, the RRFS "
            "deterministic prslev+2dfld pair of the same cycle and valid "
            "time. The member column stops at 250 hPa and publishes no "
            "soil temperature, soil moisture, skin temperature or land "
            "mask, so without the companion there is no soil state and "
            "the model top sits above the data; such a member is refused. "
            "No automatic top lowering."
        ),
        notes=(
            "Five real member trajectories, m001 through m005, are posted "
            "in rrfsens.YYYYMMDD/CC on the operational-layout AWS bucket "
            "and NOMADS v1.0. Member 1 pressure and surface samples carry "
            "PDT 1, perturbation number 1, ensemble type 3 and size 5. "
            "The subset supplies 14 pressure levels from 250 to 1000 hPa. "
            "The packaged profile is table work on the generic mapped "
            "route: every member field pins product definition template 1, "
            "and the composition declares the deterministic run of the same "
            "cycle as the member's companion, which stacks its 14 levels "
            "above 250 hPa onto the member's columns and lends the skin "
            "temperature, land mask, sea ice, snow and the nine-node RUC "
            "soil column. The composition receipt names which levels and "
            "fields came from the member and which from the companion. "
            "Ensemble statistics remain the separate REFS source and cannot "
            "substitute for a member."
        ),
    ),
    _adapter(
        "rrfs-a",
        name="RRFS-A (frozen prototype feed)", aliases=("rrfsa",),
        default_product="prs-conus", max_hour=60,
        upstream_ingest="full",
        notes="The noaa-rrfs-pds bucket contains retrospective data; its "
              "rrfs_a prefix was empty when checked on 2026-10-31. "
              "The operational feed is the `rrfs` row.  "
              "rusty-weather ingest is mature; WRF state/soil mapping "
              "remains gated.",
    ),
    _adapter(
        "rrfs-public",
        name="RRFS public prototype",
        default_product="prs-conus", max_hour=60),
    _adapter(
        "refs",
        name="REFS (RRFS ensemble products)", aliases=("rrfs-ensemble",),
        kind=SourceKind.ENSEMBLE_STATISTIC,
        default_product="mean-conus", max_hour=60,
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Use a constituent RRFS member; mean/PMMN/spread products cannot initialize a member.",
    ),
    _adapter(
        "rrfs-firewx",
        name="RRFS fire-weather nest", aliases=("firewx",),
        default_product="2dfld-firewx", max_hour=36,
        required_products=("2dfld-firewx", "prs-firewx"),
        status=AdapterStatus.COMPOSITION_REQUIRED,
        composition="Combine the 2-D fire-weather product with a complete pressure/native atmosphere and soil state.",
    ),
    _adapter(
        "wrf",
        name="WRF archive (ARW history)", aliases=("wrf-gdex", "wrf-arw"),
        kind=SourceKind.WRF_ARCHIVE,
        file_family="NetCDF",
        default_product="surface",
        max_hour=0,
        status=AdapterStatus.ARCHIVE_MAPPING_REQUIRED,
        composition="Map a compatible WRF archive state, vertical coordinate, physics state, and boundary source.",
    ),
    _adapter(
        "era5",
        ensemble_preparation_owner="woof.era5_preparation",
        hydro_forcing_profile="decoded-canonical-grid",
        record="reanalysis",
        time_axis="analysis_times",
        selection_owner="woof.era5_member",
        selection_provider_hint="era5_provider",
        seed_fields=('air_pressure', 'air_temperature', 'eastward_wind',
                     'northward_wind', 'mean_sea_level_pressure'),
        archives=(_ERA5_ARCO_ARCHIVE, _ERA5_ARCHIVE),
        name="ERA5 reanalysis (ECMWF)",
        credentials=(_COPERNICUS_CDS_KEY,),
        upstream_model_id=None,
        file_family="GRIB1",
        default_product="pressure-level+single-level",
        required_products=("pressure-level", "single-level", "invariants"),
        max_hour=0,
        upstream_ingest="gpuwm_native_existing",
        status=AdapterStatus.CERTIFIED,
        field_mapping="era5-grib1-vtable-to-native-state-v1",
        level_mapping="pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-local-grib-time-series-v1",
        stock_wrf_gate=(
            "wrf-v4.6.1-pass-201ffde-250x200x49-12km-12h"
        ),
        runnable=True,
        runner="era5_combined_grib1_v1",
        # fetch.ERA5_PRESSURE_LEVELS_HPA: the complete 37-level ladder.
        certified_source_top_pa=100.0,
        case_data_file="era5-combined.grib",
        case_data_vtable="data/vtables/Vtable.ERA5_CDO",
        forcing_interval_seconds=21600.0,
        fetch_requires_retrieve=True,
        # THE ROW THAT MADE THE COLUMN.  `--cycle latest` used to be
        # refused here with the sentence "a reanalysis published with
        # a delay of several days" -- which describes a DELAY, and a
        # delay is a number, and a number resolves.  ERA5T (the
        # preliminary stream the CDS serves for recent dates) runs
        # about five days behind real time.  No probe: the CDS is a
        # keyed job API, not a file server, so there is no object to
        # HEAD and this delay IS the answer.
        cycles=CycleGrid(
            hours=(0, 6, 12, 18), delay_hours=120.0,
            search_hours=168,
            basis="ERA5T preliminary is published about five days "
                  "behind real time, on the 6-hourly analysis grid "
                  "this row's forcing_interval_seconds declares",
            # ERA5 HRES has an analysis every hour (CDS reanalysis and
            # ARCO full_37-1h both hourly; fetch._era5_times takes any
            # exact hour), so a window may start at 16Z and take its
            # 6 h boundaries at 22Z and 04Z.  ``latest`` stays 6-hourly.
            analysis_hours=tuple(range(24))),
        notes=(
            "Certified slice: one specified Lambert domain, WSM6, YSU, "
            "classic MM5 surface layer (option 91), Noah, an explicit "
            "validated eta coordinate, a "
            "model top covered by the supplied pressure levels, a "
            "uniformly spaced combined GRIB1 series beginning at f00, and "
            "explicit hash-bound WPS geometry/static/orography inputs. "
            "Unchanged stock WRF v4.6.1 accepted the 250x200 at 12 km, "
            "12-hour proof products and completed a finite model step. The "
            "named-source hierarchy route now accepts namelist-bound static "
            "one-way Lambert layouts through at least d06 using invariant "
            "SOILGEO or exact per-domain source orography; that broader "
            "envelope is not yet "
            "stock-WRF certified."
        ),
    ),
    _adapter(
        "era5-l137",
        record="reanalysis",
        time_axis="analysis_times",
        archives=(_ERA5_ARCHIVE,),
        name="ERA5 reanalysis (native 137 model levels)",
        aliases=("era5-model-level", "era5-ml"),
        credentials=(_COPERNICUS_CDS_KEY,),
        upstream_model_id=None,
        file_family="GRIB2",
        decoder=(
            "packaged ERA5 model-level profile + "
            "vendored rusty-weather GRIB2/GRIB1 bridges"
        ),
        default_product="model-level+pressure-level-surface",
        required_products=(
            "model-level atmosphere (all 137 levels)",
            "same-hour pressure-level/single-level analysis",
        ),
        # A reanalysis: every valid time is an analysis, never a lead.
        max_hour=0,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-era5-model-level-l137-grib2-v1",
        level_mapping="137-hybrid-sigma-pressure-model-level-to-explicit-wrf-eta-v1",
        cadence_mapping="uniform-hourly-analysis-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="era5-model-level-l137-grib2-v1",
        forcing_interval_seconds=3600.0,
        # The same reanalysis on the same release schedule as the
        # pressure-level row, on the hourly grid this variant reads.
        cycles=CycleGrid(
            hours=tuple(range(24)), delay_hours=120.0,
            search_hours=168,
            basis="the ERA5T release schedule, on the hourly grid "
                  "this row's forcing_interval_seconds declares"),
        composition=(
            "ERA5's model-level product publishes the prognostic "
            "atmosphere only; the land-surface and near-surface state -- "
            "surface pressure, orography, land fraction, skin "
            "temperature, the 2 m and 10 m diagnostics and the four-layer "
            "soil column -- is borrowed from the SAME HOUR's ERA5 "
            "pressure-level/single-level analysis, so a preparation takes "
            "two files and refuses by name if the donor hour disagrees."
        ),
        notes=(
            "The native vertical state, as pure table data: 137 hybrid "
            "sigma-pressure model levels whose A/B interface coefficients "
            "ride IN BAND in the GRIB2 Section-4 coordinate-values (pv) "
            "octets -- 276 values, read from the producer's own record "
            "rather than from a per-model coefficient table.  Pressure is "
            "materialized as p = A + B*ps against the borrowed surface "
            "pressure and geopotential height is integrated "
            "hydrostatically from the borrowed terrain, because the "
            "model-level product publishes z only at level 1 and a 3-D "
            "height borrow would be cross-ladder.  lnsp is NOT consumed: "
            "unit transforms are affine, there is no exp, and surface "
            "pressure is borrowed exactly instead of approximated.  "
            "Measured on real Copernicus CDS bytes: the Rust engine and "
            "the Python reference produce byte-identical air_pressure and "
            "geopotential_height (max ULP 0) and the preparation reaches "
            "rc 0 with wrfinput/wrfbdy written.  Not yet accepted by "
            "unchanged stock WRF, and this row does not certify ERA5 "
            "model-level requests on other grids or level subsets: the "
            "mapping's ladder is the full 1..137 set."
        ),
    ),
    _adapter(
        "20crv3",
        record="reanalysis",
        time_axis="analysis_times",
        name="20CRv3 reanalysis (member, GRIB2)",
        aliases=("20cr", "twentycrv3", "20crv3-member"),
        upstream_model_id=None,
        kind=SourceKind.ENSEMBLE_MEMBERS,
        file_family="GRIB2",
        decoder=(
            "packaged exact 20CRv3 member profile + "
            "vendored rusty-weather GRIB2 bridges"
        ),
        default_product="every-member-pressure+surface",
        required_products=("pressure-level", "surface", "one exact member"),
        # 20CRv3 files are analyses at successive valid times, not forecast
        # lead-hour products; the three-hour property belongs to cadence.
        max_hour=0,
        upstream_ingest="filename_member_manifest_v1",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-20crv3-member-grib2-v1",
        level_mapping="23-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-paired-three-hour-member-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="twentycrv3_member_grib2_v1",
        packaged_profile="20crv3-member-grib2-v1",
        forcing_interval_seconds=10800.0,
        notes=(
            "Runnable exact-member profile with filename-plus-hash-manifest "
            "member identity, packaged mapping/composition/provenance "
            "authorities, paired pressure/surface inputs, and native one-way "
            "Lambert hierarchy export through the mapping's max_dom=4. "
            "The route is not yet accepted by unchanged stock WRF and does "
            "not certify other 20CR products, arbitrary members mixed in one "
            "run, or a larger domain count."
        ),
    ),
    _adapter(
        "20crv3-cf",
        record="reanalysis",
        time_axis="analysis_times",
        name="20CRv3 reanalysis (ensemble mean, NetCDF)",
        aliases=("20crv3-netcdf", "20cr-netcdf", "20cr-cf"),
        upstream_model_id=None,
        # TRUTHFUL, and it is the one thing a reader must not miss: NOAA
        # PSL's NetCDF distribution of 20CRv3 is the ENSEMBLE MEAN
        # analysis.  Every variable in it carries `statistic = "Ensemble
        # Mean"`, which the packaged mapping binds as a selector attribute,
        # so a member file cannot be fed through this profile unnoticed and
        # a receipt can never say "member" about data that is not one.
        kind=SourceKind.ENSEMBLE_STATISTIC,
        file_family="NetCDF-CF",
        decoder=(
            "packaged 20CRv3 NetCDF profile + rw_netcdf/netcrust "
            "(woof.netcdf_bridge)"
        ),
        default_product="si-subdaily-pressure+surface+subsurface",
        required_products=(
            "prsSI", "sfcSI", "2mSI", "10mSI", "subsfcSI",
            "published invariant supplement",
        ),
        max_hour=0,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-20crv3-si-netcdf-v1",
        level_mapping="21-pressure-level-to-explicit-wrf-eta-v2",
        cadence_mapping="uniform-three-hour-analysis-series-v1",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="20crv3-netcdf-v1",
        forcing_interval_seconds=10800.0,
        # The PSL humidity series stops at 100 hPa and the mapping
        # declares the intersection, so 100 hPa is this route's ladder.
        certified_source_top_pa=10000.0,
        notes=(
            "The publicly downloadable form of 20CRv3: NOAA PSL's sub-daily "
            "NetCDF SI/MO series, fetched through woof fetch --source 20crv3-cf and decoded through the Rust rw_netcdf bridge with "
            "no per-source decode code -- the mapping, composition and "
            "provenance documents are packaged and pinned by SHA-256. "
            "Three-hourly analyses on 21 common pressure levels (1000-100 "
            "hPa; the humidity file stops at 100 hPa while the others reach 1 "
            "hPa, and the mapping declares the intersection), a complete "
            "surface state, and the exact four Noah soil layers. "
            "TWO limits a reader must know. (1) It is the ENSEMBLE MEAN "
            "analysis, not a member: it is smoother than any single 20CRv3 "
            "trajectory, and where a member state is wanted the route is "
            "--source 20crv3 over the every-member GRIB2 archive. The "
            "registry's refusal of GEFS/HREF/SREF means does not apply here "
            "for a stated reason -- those are FORECAST ensembles whose mean is "
            "not a dynamically balanced trajectory, while this is an analysis "
            "at its own valid time and is the only 20CRv3 form NOAA "
            "distributes as NetCDF -- but the difference is a judgement worth "
            "reading before a study depends on it. (2) Native fetch binds "
            "PSL's published time-invariant surface height and land fraction "
            "to every primary time through Rust exact coordinate subsetting. "
            "Soil temperature and moisture use the four published Noah layers; "
            "ocean initialization uses published skin temperature as an "
            "explicit SST proxy. No separate sub-daily SST is published. "
            "Not yet accepted by unchanged stock WRF."
        ),
    ),
    _adapter(
        "hex-global",
        time_axis="supplied_times",
        # THE BREAKAGE THIS PREVENTS (HEX-REGIONAL-2026-10-23/WET-BIAS.md):
        # a 3 km child of a hex run took the gfs/era5-class Morrison default,
        # whose wet-snow REFL_10CM drew a 55-65 dBZ shield over the trailing
        # stratiform where MRMS had 25-40 (Kansas nest +6 h: 42,822 km2 at
        # 50+ dBZ against MRMS 1,107; 4,302 on this suite).
        convection_permitting_physics_profile=(
            "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"),
        name="woof hex forecast export (global or regional window, NetCDF)",
        aliases=("hex", "woof hex", "hex-regional"),
        upstream_model_id=None,
        file_family="NetCDF (woof hex export frames)",
        decoder=(
            "packaged woof hex history NetCDF profile + rw_netcdf "
            "(woof.netcdf_bridge)"
        ),
        default_product="export --field-set full on a lat-lon window",
        required_products=(
            "woof hex export frames: --field-set full, --window "
            "region-latlon:CLAT,CLON,W_KM,H_KM or global-mesh",
            "the same frames as the in-band terrain supplement",
        ),
        max_hour=0,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="packaged-rw-wps-hex-history-netcdf-v1",
        level_mapping="55-mpas-model-level-to-explicit-wrf-eta-v1",
        cadence_mapping="mapping-declared-uniform-series",
        stock_wrf_gate="live-unchanged-stock-wrf-gate-pending",
        runnable=True,
        runner="mapped_composition_v1",
        packaged_profile="hex-history-netcdf-v1",
        local_posting=(
            "woof hex export RUN --follow writes each frame as the parent "
            "forecast closes it; python -m woof.local_posting --source "
            "hex-global --ready-list EXPORT/export-ready.jsonl publishes "
            "them for woof prep --as-posted"),
        # A parent produced beside the nest posts a lead per parent hour: a
        # daily mean over its first 24 h held the nest's head until the whole
        # parent forecast was done (box G 2026-10-23: the as-posted head
        # waited for f012 of a 12 h window).  Inland water the parent does
        # not resolve takes the start time's own 2 m air temperature.
        inland_air_mean_hours=0.0,
        notes=(
            "A woof hex run (any mesh: the 60 km global row, a regional "
            "mesh, an ensemble member) as the initial and lateral boundary "
            "source of a WOOF domain anywhere its export window covers. "
            "Export the run with `woof hex export RUN --field-set full "
            "--window region-latlon:CLAT,CLON,W_KM,H_KM` (or global-mesh): "
            "those frames carry CF 1-D lat/lon/Time/level coordinates and "
            "the mass-grid earth winds this profile reads.  The mapping "
            "decodes the 55 MPAS model levels directly (full pressure, "
            "heights from the init zgrid, T from theta with MPAS's kappa, "
            "q from the mixing ratio) with no pressure-level step.  SST is "
            "the water skin temperature, a named policy: hex carries no "
            "ocean model and holds the SST it was initialised with.  Sea "
            "ice and snow are the run's own land-surface state.  Boundary "
            "spacing is whatever the export carries (hourly by default for "
            "a parent store), so `woof domain --source hex-global` takes "
            "it from --cadence HOURS; the whole route (domain, fetch-geog, "
            "prep with --author-input-manifest, sim) is the fenced block "
            "in docs/native-mapped-source-authoring.md.  Not yet accepted "
            "by unchanged stock WRF."
        ),
    ),
    _adapter(
        "mapped",
        time_axis="supplied_times",
        name="Generic mapped source (rw-wps mapping v1)",
        aliases=("generic-mapped", "mapping-v1"),
        upstream_model_id=None,
        file_family="GRIB1/GRIB2/NetCDF",
        decoder=(
            "rw-wps.mapping.v1 + woof.mapped_source + "
            "vendored rusty-weather GRIB bridges"
        ),
        default_product="mapping-declared-inputs",
        max_hour=0,
        upstream_ingest="declarative_mapping_v1",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="rw-wps.mapping.v1-executable-consumer",
        level_mapping="mapping-declared-source-to-canonical-frame",
        cadence_mapping="mapping-declared-uniform-series",
        stock_wrf_gate="exact-mapping-composition-hash-evidence-required",
        runnable=True,
        runner="mapped_composition_v1",
        notes=(
            "Strict canonical decoding, exact-subset mixed-product composition, "
            "source-frame packing, native initialization, and atomic WRF export "
            "are implemented. Historical v1 ERA5 GRIB1, GFS GRIB2, and ERA5 "
            "NetCDF single-domain gates passed unchanged WRF v4.6.1. The "
            "current canonical-LF GFS GRIB2 contract also passed an unchanged "
            "WRF v4.6.1 d01 through d04 gate after replacing 2 m dewpoint with the "
            "downloaded RH2 contract and binding GRIB2 local-table authority. "
            "The superseded GFS hierarchy evidence remains explicitly "
            "invalidated. New mappings remain "
            "fail-closed to the explicit field, selector, soil, cadence, target, "
            "decoder, manifest, and composition contracts. General explicit "
            "descriptor/Vtable import and exact manifest authoring are available "
            "without adding a named adapter, but their result is validated, not "
            "automatically stock-WRF certified; this is not a claim that arbitrary "
            "undeclared GRIB/NetCDF is scientifically complete."
        ),
    ),
)




def _model_table_rows(base):
    """Expand declared model inventories without changing dynamic route registration."""
    import json
    result = []
    for template in base:
        if template.model_table is None or template.model_key is not None:
            continue
        table = Path(__file__).with_name("authorities") / template.model_table
        models = json.loads(table.read_text(encoding="utf-8"))["models"]
        for key, row in models.items():
            profile = row.get("profile")
            result.append(replace(template, source_id=template.source_id + "-" + key,
                model_key=key, model_row=row, display_name=row.get("name", key),
                aliases=tuple(template.source_id + "-" + alias for alias in row.get("aliases", ())),
                packaged_profile=profile, runnable=template.runnable and bool(profile),
                status=template.status if profile else AdapterStatus.COMPOSITION_REQUIRED,
                field_mapping="packaged-rw-wps-" + profile if profile else template.field_mapping,
                forcing_interval_seconds=float(row["step_hours"]) * 3600
                    if row.get("step_hours") else template.forcing_interval_seconds,
                selection_owner=row.get("selection_owner", template.selection_owner),
                notes=row.get("refusal") or row.get("needs") or template.notes))
    return tuple(result)



_ADAPTERS = _ADAPTERS + _model_table_rows(_ADAPTERS)


def _build_alias_map() -> dict[str, SourceAdapter]:
    aliases: dict[str, SourceAdapter] = {}
    for adapter in _ADAPTERS:
        for raw_name in (adapter.source_id, *adapter.aliases):
            name = raw_name.strip().lower().replace("_", "-")
            if name in aliases:
                raise RuntimeError(f"duplicate source-adapter alias: {name}")
            aliases[name] = adapter
    # Acquisition owners may expose table-derived model spellings. Existing
    # source ids and explicit aliases keep priority over a model shorthand.
    from importlib import import_module
    for adapter in _ADAPTERS:
        if adapter.acquisition_owner is not None and adapter.model_key is None:
            names = getattr(import_module(adapter.acquisition_owner), "source_aliases", lambda: ())()
            for name in names:
                normalized = name.strip().lower().replace("_", "-")
                aliases.setdefault(normalized, aliases.get(adapter.source_id + "-" + normalized, adapter))
    return aliases


_ALIASES = _build_alias_map()


def source_adapters() -> tuple[SourceAdapter, ...]:
    """Return the immutable canonical source inventory."""

    return _ADAPTERS


def packaged_profile_sources() -> Mapping[str, str]:
    """``source_id -> packaged profile`` for every source that names one.

    THE table both the preparation stage and the forecast stage read, so
    the answer to "is this a shipped profile, and which" is given once.
    It lives in the registry rather than in either stage because the
    registry is what a new model is added to: a row here is the entire
    difference between a model woof can decode and one it cannot.
    """

    return MappingProxyType({
        adapter.source_id: adapter.packaged_profile
        for adapter in _ADAPTERS
        if adapter.packaged_profile is not None
    })


def model_rows(adapter):
    """The declared model table for a route, with no model-specific code."""
    import json
    if adapter.model_table is None or adapter.model_key is not None:
        return {}
    rows = json.loads((Path(__file__).with_name("authorities") / adapter.model_table)
                      .read_text(encoding="utf-8"))["models"]
    return {key: row for key, row in rows.items() if row.get("kind") != "diagnostic"}


def _normalized_source(source):
    return source.strip().lower().replace("_", "-")


def get_source_adapter(source: str) -> SourceAdapter:
    """Resolve ids, aliases and <row>-<model> using the row's model table."""
    normalized = _normalized_source(source)
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    for adapter in _ADAPTERS:
        if adapter.model_table is None:
            continue
        for prefix in (adapter.source_id, *adapter.aliases):
            if not normalized.startswith(prefix + "-"):
                continue
            suffix = normalized[len(prefix) + 1:]
            for key, row in model_rows(adapter).items():
                if suffix in tuple(_normalized_source(v) for v in (key, *row.get("aliases", ()))):
                    profile = row.get("profile")
                    return replace(adapter, model_key=key, model_row=row,
                        runnable=adapter.runnable and bool(profile),
                        packaged_profile=profile,
                        status=adapter.status if profile else AdapterStatus.COMPOSITION_REQUIRED,
                        notes=row.get("refusal") or row.get("needs") or adapter.notes,
                        forcing_interval_seconds=float(row["step_hours"]) * 3600
                            if row.get("step_hours") else adapter.forcing_interval_seconds,
                        selection_owner=row.get("selection_owner", adapter.selection_owner))
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(adapter.source_id for adapter in _ADAPTERS)
        raise ValueError(f"unknown native source {source!r}; available sources: {available}") from exc


def canonical_source_token(source):
    """Route id plus model key when the caller selected a parameterized row."""
    adapter = get_source_adapter(source)
    suffix = "-" + adapter.model_key if adapter.model_key else ""
    return adapter.source_id + (suffix if suffix and not adapter.source_id.endswith(suffix) else "")


def ensemble_source_ids():
    """Eligible family ids, derived from runnable planning/selection rows."""
    planable = set(wizard_planable_source_ids())
    result = []
    for adapter in _ADAPTERS:
        if not adapter.runnable or not (adapter.source_id in planable
                                       or adapter.member_set or adapter.selection_owner):
            continue
        result.append(adapter.source_id)
        for key in model_rows(adapter):
            token = adapter.source_id + "-" + key
            if get_source_adapter(token).runnable:
                result.append(token)
    return tuple(dict.fromkeys(result))


def ensemble_source_argument(value):
    """Validate a family id/alias; the refusal lists the same rows as help."""
    try:
        token = canonical_source_token(value)
        if token in ensemble_source_ids():
            return token
    except ValueError:
        pass
    raise ValueError(f"ensemble source {value!r} is not eligible; eligible ids: "
                     + ", ".join(ensemble_source_ids()))


def wizard_planable_source_ids() -> tuple[str, ...]:
    """Every source id ``woof domain`` can emit a runnable config for.

    A row qualifies on exactly two declared facts: it is ``runnable`` (a
    strict implementation route exists) and it declares a
    ``forcing_interval_seconds`` (the wizard has a boundary cadence to
    write into ``namelist.wps``).  Nothing here enumerates model names, so
    a new row appears in `woof domain --help` the moment it is added --
    which is the whole claim this function is here to keep true.
    """

    return tuple(adapter.source_id for adapter in _ADAPTERS
                 if adapter.runnable
                 and adapter.forcing_interval_seconds is not None)


def wizard_cadence_named_source_ids() -> tuple[str, ...]:
    """Every runnable row whose boundary cadence the caller names.

    A row with no ``forcing_interval_seconds`` (a supplied-times source
    whose spacing is whatever its history files carry, such as a
    woof hex export) is still plannable by ``woof domain``: the caller
    passes ``--cadence HOURS`` naming the spacing of the files it will hand
    ``woof prep``, and that number is the ``interval_seconds`` written.
    Nothing here enumerates model names, so the next such row reaches the
    door, and its ``--help``, with no edit to the wizard.
    """

    return tuple(adapter.source_id for adapter in _ADAPTERS
                 if adapter.runnable
                 and adapter.forcing_interval_seconds is None)


def source_forcing_interval_seconds(source: str) -> float:
    """Seconds between SOURCE's native valid times, from its registry row.

    Raises a fail-closed :class:`ValueError` naming the breakage when the
    row declares none: a config emitted with a guessed boundary cadence
    would ask the preparation for valid times the source never published.
    """

    adapter = get_source_adapter(source)
    if adapter.forcing_interval_seconds is None:
        raise ValueError(
            f"source {adapter.source_id!r} declares no native forcing "
            "cadence, so there is no &share/interval_seconds to emit and a "
            "config written for it would name boundary times the source "
            "does not publish; its cadence comes from the mapping document "
            "a caller supplies, not from the registry")
    return float(adapter.forcing_interval_seconds)


def source_coverage_window(source: str) -> CoverageWindow | None:
    """SOURCE's declared native-grid window, or ``None`` when it is global."""

    return get_source_adapter(source).coverage_window


def source_top_pa_for_cycle(source, cycle) -> float | None:
    """The native model top (Pa) SOURCE's column reaches at CYCLE, or ``None``.

    Read from the row's ``source_top_eras`` (the last era starting at or
    before the cycle), never from a list of model names.  ``None`` when the
    row declares no eras, the source is unknown, or the cycle is not a
    concrete time (``latest``, absent): the preparation's decoded-data
    check then stays the only verdict, as before.
    """

    from datetime import timezone

    if not source or cycle is None:
        return None
    try:
        adapter = get_source_adapter(str(source))
    except (TypeError, ValueError):
        return None
    if not adapter.source_top_eras:
        return None
    if isinstance(cycle, str):
        if cycle == "latest":
            return None
        from woof.source_availability import parse_cycle
        cycle = parse_cycle(cycle)
    if not isinstance(cycle, datetime):
        return None
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    top = None
    for era in adapter.source_top_eras:
        if not era.start or cycle >= datetime.strptime(era.start, "%Y-%m-%dT%H"):
            top = float(era.top_pa)
    return top


def fetch_model_top_pa(source, p_top_pa) -> float | None:
    """The model top (Pa) a download for SOURCE must reach, or ``None``.

    ``p_top_pa`` is the model top of the run the download is for (its
    ``[shared].p_top``).  ``None`` means the request stays exactly as it
    was: the source's certified ladder already reaches that top, its row
    declares no certified top, or its fetch takes no model top (then the
    preparation's vertical-coverage refusal names the gap).  Otherwise
    the answer is ``p_top_pa`` itself, the value ``woof fetch
    --p-top-pa`` takes, so the fetch extends its ladder to the levels the
    run's own top needs.  A top above everything the product publishes is
    still returned: the fetch refuses it before any download, naming the
    deepest top it can serve.

    Read from the registry row, never from a list of model names, so
    ``woof go``, ``woof run-plan``, the desktop and the download price
    all ask the same table the same question.
    """

    if not source or p_top_pa is None or isinstance(p_top_pa, bool):
        return None
    try:
        top = float(p_top_pa)
        adapter = get_source_adapter(str(source))
    except (TypeError, ValueError):
        return None
    ceiling = adapter.certified_source_top_pa
    if (adapter.extendable_source_top_pa is None or ceiling is None
            or not math.isfinite(top) or top <= 0.0 or top > float(ceiling)):
        return None
    return top


def _current_config_sha256(name: str, canonical_sha256: str) -> str:
    """Bind the checked-out/package config bytes when they are available.

    Git may materialize JSON authorities with LF or CRLF line endings.  The
    evidence manifest must report the bytes the runtime will actually consume,
    rather than a hash copied from a different checkout platform.
    """
    path = Path(__file__).resolve().parents[1] / "configs" / name
    if not path.is_file():
        return canonical_sha256
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_capability_manifest() -> dict[str, object]:
    """Return the machine-readable, provenance-bound capability inventory."""

    return {
        "schema": "gpuwm-native-source-adapters-v1",
        "readiness_rule": (
            "runnable means the strict implementation route exists; stock-WRF "
            "certification is a separate exact evidence status"
        ),
        "certification_rule": (
            "certified_stock_wrf requires implemented field/level/cadence "
            "policies plus acceptance by unchanged stock wrf.exe within the "
            "declared source/domain envelope"
        ),
        "runtime_forbidden": ["WPS", "real.exe"],
        "rusty_weather_inventory": {
            "head": RUSTY_WEATHER_HEAD,
            "worktree_was_dirty_during_inventory": True,
            "bound_file_sha256": dict(RUSTY_WEATHER_INVENTORY_HASHES),
            "model_id_count": 23,
        },
        "source_count": len(_ADAPTERS),
        "runnable_source_count": sum(adapter.runnable for adapter in _ADAPTERS),
        # Every packaged profile, not one hand-named row: a profile that
        # ships is a profile whose digests a consumer can check.
        "packaged_source_authorities": {
            profile_id: dict(packaged_authority_sha256(profile_id))
            for profile_id in packaged_profile_ids()
        },
        # The ensemble member-addressing grammars ship under the same
        # rule: a packaged document is a document whose digest a
        # consumer can check.
        "packaged_member_grammars": {
            grammar_id: packaged_member_grammar_sha256(grammar_id)
            for grammar_id in packaged_member_grammar_ids()
        },
        "mapped_stock_wrf_evidence": [
            {
                "mapping_config": "rw-wps-era5-1974-probe.mapping.json",
                "composition_config": (
                    "rw-wps-era5-1974-terrain.composition.json"
                ),
                "mapping_sha256": (
                    "2ceb90e63d3265c0ebf871de4bfc7f622284681a9861c17e78e9fc11e3dc15b2"
                ),
                "composition_sha256": (
                    "9aa8c7aec96a1d52e83f61c470cdb2a636ca092af902581bafb05fc433ab57f3"
                ),
                "gate": "wrf-v4.6.1-pass-era5-grib1-single-domain",
                "source_format": "grib1",
            },
            {
                "mapping_config": "rw-wps-gfs-pressure-grib2.mapping.json",
                "composition_config": "rw-wps-gfs-terrain.composition.json",
                "mapping_sha256": (
                    "5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86"
                ),
                "composition_sha256": (
                    "e0c2adae105263b177d7e8f8bb87d0e99731bc4cda9cb6a4217971a0b49b18e1"
                ),
                "gate": (
                    "wrf-v4.6.1-clean-d66e442-pass-gfs-grib2-"
                    "canonical-lf-current-contract-d01-d04-z49"
                ),
                "source_format": "grib2",
                "domain_envelope": (
                    "specified Lambert d01 120x100x49 at 12 km plus three "
                    "one-way sibling nests d02-d04, each 60x60x49 at 4 km"
                ),
                "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
                "wrf_exe_sha256": (
                    "cfac96554c8f9796c7522aaf023131ea7681ddf12110a327e51a548958874089"
                ),
                "native_proof_sha256": (
                    "cb46c3539e6789710fa44717fe9af6871483ca035e1c41abfb3eb2fe4b804d28"
                ),
                "native_proof_content_sha256": (
                    "fddddaec90934ce2d37fd112a36b76fda623eba945df942e04b6f9053de77138"
                ),
                "native_export_manifest_sha256": (
                    "5330ad1aa50bdeb0a3de4c601c29660038a111d5b966b305c3f694e35d07d1ad"
                ),
                "native_input_manifest_sha256": (
                    "3f601dfa28419229ef5314a58345ca4b521642b796c1d49a7494e1d94f330abb"
                ),
                "wrfinput_d01_sha256": (
                    "1f5cf5d47e0a450e44bbf7033f0622b0636c852208fe76390e5719822f0014dc"
                ),
                "wrfinput_d02_sha256": (
                    "546255b0ad323b073f0f0d87545615ea07937f50393f12305d078d9128d262da"
                ),
                "wrfinput_d03_sha256": (
                    "7044d67ac77c9081e8bfbfd6692bfc18aa26fb1cef001423b729082ea5ff5003"
                ),
                "wrfinput_d04_sha256": (
                    "ce8ffdd60acfd41a316e3c7df74fe4b48063afa1b1c2138ab510d524fa8fc082"
                ),
                "wrfbdy_d01_sha256": (
                    "9f9d2430511fd807c4c0a75c1fb29e1d6f401893929d858a1002920a385e79fa"
                ),
                "stock_wrf_input_receipt_sha256": (
                    "25a643fe34ff1ddd39464129bdcfff074e938ced216608f8cab8d17196ea524c"
                ),
                "stock_wrf_stdout_sha256": (
                    "df9777532d6baec919937d684eb16bad1d22831239092effefa007d6a7eeeb8d"
                ),
                "stock_wrf_time_sha256": (
                    "46f06763d0f59d4b4b91fd84ac27dd98264f8d6f4a01f1a84a94b55544f01392"
                ),
                "stock_wrf_stderr_sha256": (
                    "bee69544f5cdbbfecfff392294a175311e7f19d505e3e8227aebee77727c43eb"
                ),
                "stock_wrf_namelist_sha256": (
                    "98693a363961c47afb37e52feb9bb9eb9125069802a89b5fd5aa30acedc22cfc"
                ),
                "wrfout_d01_sha256": (
                    "ad34744719f6bf120bffe1771fab28891b1968d3c324f149676ba215014f90f4"
                ),
                "wrfout_d02_sha256": (
                    "5f7518adc40c869249b70cf02a547c39a84bd86b4d8af9614659224c3df19241"
                ),
                "wrfout_d03_sha256": (
                    "9aa978f4088597abe90c8c466e295251b962af077342241134b24397971b6f23"
                ),
                "wrfout_d04_sha256": (
                    "91171bd4b0e603df531e10b06ca7613a824e26a88534010962f167f7bac13cbb"
                ),
                "native_total_seconds": 10.613378500012914,
                "stock_wrf_wall_seconds": 10.08,
                "stock_wrf_max_rss_kib": 1076808,
                "stock_wrf_result": (
                    "exit_0_success_complete_wrf_d02_d03_d04_six_steps_each"
                ),
                "stock_wrf_health": "no_nan_cfl_fatal_or_segfault_markers",
            },
        ],
        # An entry here means: this gate really passed, on the bytes named by
        # ``mapping_sha256``/``composition_sha256``, and those bytes are no
        # longer what the product ships.  ``mapping_config`` names the file
        # whose CURRENT bytes the ``replacement_*`` hashes report, so a reader
        # can see both what was proved and what is shipped.
        "invalidated_mapped_stock_wrf_evidence": [
            {
                "mapping_config": "rw-wps-gfs-pressure-grib2.mapping.json",
                "composition_config": "rw-wps-gfs-terrain.composition.json",
                "mapping_sha256": (
                    "726677d8c2365e6f533cc6dd5d7c795e198164326660c3630d885c83f406a11e"
                ),
                "composition_sha256": (
                    "266c98099b24f03a3bc986f275b44bbd6bf20dce1006ad05f6da39bc4a373bfb"
                ),
                "gate": "wrf-v4.6.1-pass-gfs-grib2-d01-d02",
                "source_format": "grib2",
                "reason": (
                    "superseded by trajectory-relevant RH2-to-q2 mapping and "
                    "explicit GRIB2 local-table authority; evidence does not "
                    "transfer across contract hashes"
                ),
                "replacement_mapping_sha256": (
                    _current_config_sha256(
                        "rw-wps-gfs-pressure-grib2.mapping.json",
                        "5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86",
                    )
                ),
                "replacement_composition_sha256": (
                    _current_config_sha256(
                        "rw-wps-gfs-terrain.composition.json",
                        "e0c2adae105263b177d7e8f8bb87d0e99731bc4cda9cb6a4217971a0b49b18e1",
                    )
                ),
                "replacement_status": "stock_wrf_certified_d01_d04_z49",
                "replacement_gate": (
                    "wrf-v4.6.1-clean-d66e442-pass-gfs-grib2-"
                    "canonical-lf-current-contract-d01-d04-z49"
                ),
            },
            {
                "mapping_config": "rw-wps-era5-netcdf.mapping.json",
                "composition_config": (
                    "rw-wps-era5-netcdf-terrain.composition.json"
                ),
                "mapping_sha256": (
                    "f278705331a81767d4d3532ff4dd4f739242a79b224747f51e722d462142daa8"
                ),
                "composition_sha256": (
                    "10b23b56b882ed534dbcb5219ac06f9dcc184226e7cc2aea3c6c03f3ad6be459"
                ),
                "gate": "wrf-v4.6.1-pass-era5-netcdf-single-domain",
                "source_format": "netcdf",
                "reason": (
                    "the shipped ERA5 NetCDF mapping gained an ordered list of "
                    "accepted time-coordinate spellings for ECMWF's "
                    "time/valid_time rename, which moved its bytes; evidence "
                    "does not transfer across contract hashes"
                ),
                "replacement_mapping_sha256": (
                    _current_config_sha256(
                        "rw-wps-era5-netcdf.mapping.json",
                        "d2c9ee08e45478a64e4d2bba689e9bad1d2e97bde713477ee2a4de26e31d7ad3",
                    )
                ),
                "replacement_composition_sha256": (
                    _current_config_sha256(
                        "rw-wps-era5-netcdf-terrain.composition.json",
                        "87387cd24d3fb4c8488eced6734f16c9db76400929139d8887eecf916eb07ce7",
                    )
                ),
                # NOT a replacement gate: stock wrf.exe has not been run
                # against these bytes.  Naming a passing gate here would be
                # the exact thing the reason above forbids -- old evidence
                # silently certifying a new contract -- so the field that
                # would carry it says what is actually true instead.
                "replacement_status": "stock_wrf_regate_required",
                "replacement_gate": None,
            },
        ],
        "sources": [adapter.to_dict() for adapter in _ADAPTERS],
    }


def source_for_cycle(source: str, cycle) -> str:
    """Select a dated publication's complete source row before planning.

    A publication can change its member population, vertical ladder and
    cadence as well as its URL. These are separate rows, selected by data.
    The chosen row still checks its own archive bounds and exact objects.
    """
    import json
    from datetime import datetime, timezone
    try:
        source = canonical_source_token(source)
    except ValueError:
        # Dynamically registered acquisition rows need not have a native
        # preparation adapter. Their route remains the validation owner.
        return source
    if isinstance(cycle, str):
        if cycle == "latest":
            return source
        from woof.source_availability import parse_cycle
        cycle = parse_cycle(cycle)
    if not isinstance(cycle, datetime):
        return source
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    rows = json.loads((Path(__file__).parent / "authorities" /
                       "source-publications.v1.json").read_text())
    for row in rows["sources"].get(source, []):
        if ((not row.get("valid_from") or cycle >= datetime.fromisoformat(row["valid_from"]))
                and (not row.get("valid_until") or cycle < datetime.fromisoformat(row["valid_until"]))
                and (not row.get("weekdays") or cycle.weekday() in row["weekdays"])):
            return canonical_source_token(row["source"])
    return source
