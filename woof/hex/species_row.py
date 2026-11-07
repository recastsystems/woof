"""The prognostic scalars a microphysics scheme declares, as DATA.

The v841 production chain was welded to WSM6's six species at every layer:
the config dict pinned ``mp_wsm6``, the preparation kernel hardcoded six
output pointers, the tendency/update/recovery contracts spelled six field
names, the driver checked ``scalars.shape[0] == 6`` and summed all six into
the moist coefficients, and the sealed adapter asserted the WSM6 order at
four sites.  Every one of those is the same fact restated, so a scheme with
a different species set could not reach the forecast door at all -- only the
column seam, which takes species by name.

This module is the ONE place that fact lives.  A scheme is a ROW; the row
names its species in model array order, tags each one with what it IS, and
carries the scheme's surface accumulators, effective radii and restart
extras.  Every consumer reads the row instead of restating the six.

THE BREAKAGE THE CLASS TAGS PREVENT, and why three tags were not enough.
``atm_compute_moist_coefficients`` (the ``moist_cell_coefficients_v841_f32``
and ``moist_edge_coefficients_v841_f32`` kernels) forms
``qtot = sum of the moist mass species`` and then ``cqw``/``cqu`` as
``1/(1 + ...)``, and ``moist_dpdz_v841_f32`` carries ``qtot`` into the
buoyancy term.  Those loops ran over ALL leading species unconditionally.
Two distinct ways that breaks on a non-WSM6 row:

* A NUMBER species (``ni``, ``nr``, ``nc``, ``nifa``, ``nwfa``) is a
  concentration, order 1e4 to 1e9 per kilogram or per cubic metre.  Folded
  into ``qtot`` it drives ``1/(1+qtot)`` to approximately zero and puts the
  buoyancy term off by orders of magnitude.  A VOLUME species (``qib``,
  rime ice volume) is not a mass either.
* A MASS species is not automatically a LOADING species.  P3's ``qir`` is
  the rime mass held INSIDE ``qi``, not beside it -- ``qi`` is already the
  total ice mass.  Summing ``qir`` into ``qtot`` double-counts the rimed
  fraction of the ice.  It is a mass, it must ride transport, and it must
  still stay out of the moisture loading.  A three-way mass/number/volume
  tag would have called it "mass" and silently double-counted it, which is
  a wrong answer that no shape check and no order assertion would catch.

Aerosol mass has its own tag with mass units and no moisture loading.

Hence the class tags, and the loading set is named by :data:`MASS_LOADING`
alone.

THE CONTIGUOUS-PREFIX INVARIANT.  The kernels sum a leading run of species,
so the row refuses any ordering whose mass-loading species are not a
contiguous prefix.  That turns the generalization into a single ``int32``
loop bound (:attr:`MicrophysicsSpeciesRow.n_mass_loading`) instead of a
per-species mask uploaded to the device, and it keeps the WSM6 arm's
accumulation ORDER -- sequential FP32, vapour first -- exactly as it is
today, which is what byte-identity on the frozen lane depends on.

DECLARED EXTRAS.  A later lane may need prognostic scalars that no
microphysics scheme declares -- transported, number-class, never in the
moisture loading -- plus its own surface accumulators.  It declares them
through :meth:`MicrophysicsSpeciesRow.with_extras` and every consumer in
the chain picks them up from the row, so adding a scalar is a declaration
and never another edit to a kernel, a contract, a shape check or an
assertion.  Extras are refused the :data:`MASS_LOADING` tag by name: the
contiguous-prefix invariant means an extra could only be appended AFTER the
scheme's own species, so a mass-loading extra would sit outside the summed
prefix and be silently omitted from the moisture loading it asked to join.
Refusing it is the honest answer; a scheme that genuinely loads the air
differently is a new scheme row, not an extra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .errors import ConfigurationRefusal


__all__ = [
    "AEROSOL_MASS",
    "ENERGY",
    "MASS_COMPONENT",
    "MASS_LOADING",
    "NUMBER",
    "SPECIES_CLASSES",
    "VOLUME",
    "MicrophysicsSpeciesRow",
    "SpeciesDeclaration",
    "SurfaceAccumulatorDeclaration",
    "KESSLER_SPECIES_ROW",
    "P3_SPECIES_ROW",
    "THOMPSON_AEROSOL_SPECIES_ROW",
    "THOMPSON_SPECIES_ROW",
    "WSM6_SPECIES_ROW",
    "declared_species_rows",
    "engine_species_row",
    "register_engine_row",
    "register_extended_row",
    "registered_extended_rows",
    "registered_scheme_aliases",
    "registered_species_rows",
    "species_row_for_mp_physics",
    "species_row_for_names",
    "species_row_for_scheme",
]


#: A moist mass mixing ratio that LOADS the air: it enters ``qtot`` and so
#: the moist coefficients and the buoyancy term.  WSM6's six are all of this
#: class; P3's are ``qv``, ``qc``, ``qr`` and ``qi``.
MASS_LOADING = "mass_loading"

#: A mass mixing ratio that is a COMPONENT of another species already in the
#: loading set, and therefore must not be summed again.  P3's ``qir`` (rime
#: mass inside the single ice category) is the reason this tag exists.
MASS_COMPONENT = "mass_component"

#: A number concentration.  Transported, never a mass.
NUMBER = "number"

#: A volume mixing ratio.  Transported, never a mass.  P3's ``qib``.
VOLUME = "volume"

# Specific turbulent kinetic energy, transported but never moisture loading.
ENERGY = "energy"

# Passive aerosol mass, excluded from moist loading.
AEROSOL_MASS = "aerosol_mass"

#: Every tag a species may carry.  Ordered so the loading class is first.
SPECIES_CLASSES: tuple[str, ...] = (
    MASS_LOADING,
    MASS_COMPONENT,
    NUMBER,
    VOLUME,
    ENERGY,
    AEROSOL_MASS,
)

#: The classes that enter ``qtot``.  A frozenset of one, written as a set so
#: the consumers read "is this species in the loading set" rather than
#: comparing against a bare string and inviting a second copy of the rule.
LOADING_CLASSES: frozenset[str] = frozenset((MASS_LOADING,))


def _clean_name(value: object, what: str) -> str:
    name = str(value).strip().lower()
    if not name:
        raise ValueError(f"{what} must be a non-empty name, got {value!r}")
    if name != str(value).strip():
        # Only a case fold is tolerated; surrounding space is already gone.
        pass
    return name


@dataclass(frozen=True, slots=True)
class SpeciesDeclaration:
    """One prognostic scalar, as the row declares it.

    ``history_name`` is the name this species writes under in a history
    file.  It is the WRF/MPAS registry spelling rather than the internal
    one, because the internal name is a port detail and the file is a user
    surface.
    """

    #: The internal name, lowercase.  The seam's canonical spelling.
    name: str
    #: One of :data:`SPECIES_CLASSES`.
    species_class: str
    #: Whether scalar transport carries this species.  Every species a
    #: scheme declares is transported; the flag exists so a declared extra
    #: can say otherwise without inventing a second table.
    transported: bool
    #: The name a history file writes this species under.
    history_name: str
    #: What the species is, in words, for the receipt and the reader.
    description: str
    #: True when this species was appended by :meth:`with_extras` rather
    #: than declared by the microphysics scheme itself.
    declared_extra: bool = False

    def __post_init__(self) -> None:
        name = _clean_name(self.name, "species name")
        if name != self.name:
            object.__setattr__(self, "name", name)
        if self.species_class not in SPECIES_CLASSES:
            raise ValueError(
                f"species {self.name!r} carries class "
                f"{self.species_class!r}, which is not one of "
                f"{SPECIES_CLASSES}.  The class decides whether the species "
                "enters the moisture loading, so an unrecognised tag is a "
                "wrong answer waiting to happen, not a label"
            )
        if not str(self.history_name).strip():
            raise ValueError(
                f"species {self.name!r} declares no history name; a species "
                "the door integrates but cannot write is state the user "
                "never sees"
            )
        if not str(self.description).strip():
            raise ValueError(f"species {self.name!r} declares no description")
        if not isinstance(self.transported, bool):
            raise TypeError(
                f"species {self.name!r} transported flag must be a bool"
            )

    @property
    def is_mass_loading(self) -> bool:
        """Whether this species enters ``qtot`` and the moist coefficients."""

        return self.species_class in LOADING_CLASSES

    @property
    def units(self) -> str:
        return {MASS_LOADING: "kg kg^{-1}", MASS_COMPONENT: "kg kg^{-1}",
                NUMBER: "kg^{-1}", VOLUME: "m^3 kg^{-1}",
                ENERGY: "m^2 s^{-2}", AEROSOL_MASS: "kg kg^{-1}"}[self.species_class]


@dataclass(frozen=True, slots=True)
class SurfaceAccumulatorDeclaration:
    """One 2-D per-cell accumulator a scheme (or a lane) carries.

    Separate from :class:`SpeciesDeclaration` because these are per-cell,
    not per-level, and because an accumulator may need FP64 where every
    prognostic scalar in this port is FP32: a bucket that integrates a rate
    over a whole forecast loses the small increments in FP32 long before
    the run ends.
    """

    #: Internal name, lowercase.
    name: str
    #: ``"float32"`` or ``"float64"``.
    dtype: str
    #: The name a history file writes it under.
    history_name: str
    #: What it accumulates, in words.
    description: str
    #: True when appended by :meth:`with_extras`.
    declared_extra: bool = False
    #: Native fallout may be signed, so neither sign nor monotonicity is an
    #: invariant. Finite values are still required. Extra buckets default off.
    signed_fallout: bool = False

    def __post_init__(self) -> None:
        name = _clean_name(self.name, "accumulator name")
        if name != self.name:
            object.__setattr__(self, "name", name)
        if self.dtype not in ("float32", "float64"):
            raise ValueError(
                f"accumulator {self.name!r} declares dtype {self.dtype!r}; "
                "only 'float32' and 'float64' are carried"
            )
        if not isinstance(self.signed_fallout, bool):
            raise TypeError("signed_fallout must be a bool")
        if not str(self.history_name).strip():
            raise ValueError(
                f"accumulator {self.name!r} declares no history name"
            )
        if not str(self.description).strip():
            raise ValueError(
                f"accumulator {self.name!r} declares no description"
            )


@dataclass(frozen=True, slots=True)
class MicrophysicsSpeciesRow:
    """One microphysics scheme's prognostic inventory, in array order."""

    #: The woof/hex/MPAS ``config_microp_scheme`` value this row serves.
    scheme_name: str
    #: The woof numeric ``mp_physics`` selector.
    mp_physics: int
    #: The ``microphysics_scheme`` string the pinned engine's column batch
    #: accepts, or None when the pinned engine carries no row for it.
    engine_scheme: str | None
    #: The species, in model scalar-array order.  The mass-loading species
    #: are a contiguous prefix; see the module docstring.
    species: tuple[SpeciesDeclaration, ...]
    #: The per-cell accumulators this scheme's phase two fills.
    surface_accumulators: tuple[SurfaceAccumulatorDeclaration, ...]
    #: The effective-radius fields the scheme publishes, in order.
    radius_names: tuple[str, ...]
    #: Persistent per-scheme state the restart payload must carry beyond
    #: the species themselves.
    restart_extra_names: tuple[str, ...]
    #: Why the pinned engine carries no row, when ``engine_scheme`` is None.
    engine_absent_reason: str = ""

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError(
                f"row {self.scheme_name!r} declares no species; a scheme "
                "that transports nothing is not a scheme"
            )
        names = tuple(item.name for item in self.species)
        if len(set(names)) != len(names):
            raise ValueError(
                f"row {self.scheme_name!r} repeats a species name: {names}"
            )
        if names[0] != "qv":
            raise ValueError(
                f"row {self.scheme_name!r} puts {names[0]!r} at index 0.  "
                "Water vapour is index 0 by contract: the preparation "
                "kernel reads the moist density and the dry-theta "
                "conversion from slot 0, the NoahMP sounding reads its raw "
                "qv from slot 0, and the interface kernel extrapolates "
                "surface pressure from slot 0.  Those nine sites were "
                "implicit before this row existed; the invariant is stated "
                "here so a new row cannot quietly break them"
            )
        if not self.species[0].is_mass_loading:
            raise ValueError(
                f"row {self.scheme_name!r} tags qv as "
                f"{self.species[0].species_class!r}; vapour loads the air"
            )
        loading = [item.is_mass_loading for item in self.species]
        first_non_loading = (
            loading.index(False) if False in loading else len(loading)
        )
        if any(loading[first_non_loading:]):
            trailing = [
                item.name
                for item in self.species[first_non_loading:]
                if item.is_mass_loading
            ]
            raise ValueError(
                f"row {self.scheme_name!r} orders its species so that "
                f"{trailing} carry the mass-loading class after a "
                "non-loading species.  The moist-coefficient kernels sum a "
                "contiguous LEADING run of species, so a loading species "
                "outside that prefix would be silently dropped from qtot "
                "and the run would integrate a moisture loading nobody "
                "declared.  Order the loading species first"
            )
        if self.engine_scheme is None and not self.engine_absent_reason.strip():
            raise ValueError(
                f"row {self.scheme_name!r} names no engine scheme and no "
                "reason; a refusal that does not say what would lift it is "
                "a dead end"
            )
        if self.engine_scheme is not None and self.engine_absent_reason.strip():
            raise ValueError(
                f"row {self.scheme_name!r} names engine scheme "
                f"{self.engine_scheme!r} AND an absence reason; one or the "
                "other"
            )
        radius = tuple(self.radius_names)
        if len(set(radius)) != len(radius):
            raise ValueError(f"row {self.scheme_name!r} repeats a radius name")
        accumulators = tuple(item.name for item in self.surface_accumulators)
        if len(set(accumulators)) != len(accumulators):
            raise ValueError(
                f"row {self.scheme_name!r} repeats an accumulator name"
            )

    # -- derived inventory ------------------------------------------------

    def names(self) -> tuple[str, ...]:
        """Every species name, in model scalar-array order."""

        return tuple(item.name for item in self.species)

    def n_species(self) -> int:
        """The length of the model's scalar block for this row."""

        return len(self.species)

    def mass_loading_names(self) -> tuple[str, ...]:
        """The species that enter ``qtot``, in order."""

        return tuple(item.name for item in self.species if item.is_mass_loading)

    @property
    def n_mass_loading(self) -> int:
        """The kernel loop bound: how many LEADING species enter ``qtot``.

        The contiguous-prefix invariant checked in ``__post_init__`` is what
        makes a single count sufficient.
        """

        return sum(1 for item in self.species if item.is_mass_loading)

    def names_of_class(self, species_class: str) -> tuple[str, ...]:
        if species_class not in SPECIES_CLASSES:
            raise ValueError(
                f"{species_class!r} is not one of {SPECIES_CLASSES}"
            )
        return tuple(
            item.name
            for item in self.species
            if item.species_class == species_class
        )

    def transported_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.species if item.transported)

    def declaration(self, name: str) -> SpeciesDeclaration:
        wanted = _clean_name(name, "species name")
        for item in self.species:
            if item.name == wanted:
                return item
        raise KeyError(
            f"row {self.scheme_name!r} declares no species {wanted!r}; it "
            f"declares {self.names()}"
        )

    def index_of(self, name: str) -> int:
        """The species' row in the ``(nSpecies, nLevels, nCells)`` block."""

        return self.names().index(self.declaration(name).name)

    def history_names(self) -> Mapping[str, str]:
        """Internal name to history-file name, for every species."""

        return MappingProxyType(
            {item.name: item.history_name for item in self.species}
        )

    def declared_extras(self) -> tuple[SpeciesDeclaration, ...]:
        """The species a lane appended, in order."""

        return tuple(item for item in self.species if item.declared_extra)

    def declared_extra_accumulators(
        self,
    ) -> tuple[SurfaceAccumulatorDeclaration, ...]:
        return tuple(
            item for item in self.surface_accumulators if item.declared_extra
        )

    def required_scalar_names(self) -> frozenset[str]:
        """The seam's requirement set, DERIVED rather than restated.

        ``physics_seam._required_scalar_names`` answers "which scalars must
        be present" and its silence is a state substitution.  It reads this,
        so the requirement row and the model's array can never disagree.
        """

        return frozenset(self.names())

    # -- extension --------------------------------------------------------

    def with_extras(
        self,
        species: Sequence[SpeciesDeclaration] = (),
        accumulators: Sequence[SurfaceAccumulatorDeclaration] = (),
    ) -> "MicrophysicsSpeciesRow":
        """A copy of this row carrying a lane's declared extra scalars.

        The extras are APPENDED, strictly after the scheme's own species,
        so the scheme's array indices never move and no consumer keyed to
        them has to change.  A mass-loading extra is refused: see the module
        docstring.
        """

        extras = tuple(species)
        for item in extras:
            if not isinstance(item, SpeciesDeclaration):
                raise TypeError(
                    "declared extras must be SpeciesDeclaration instances, "
                    f"got {type(item).__name__}"
                )
            if item.is_mass_loading:
                raise ConfigurationRefusal(
                    "declared_extra",
                    item.name,
                    (
                        f"extra scalar {item.name!r} is declared "
                        f"{MASS_LOADING!r}, but extras are appended after "
                        "the scheme's own species and the moist-coefficient "
                        "kernels sum only the contiguous LEADING run.  An "
                        "appended loading species would sit outside that "
                        "prefix, so it would be transported and then "
                        "silently omitted from qtot, cqw, cqu and the "
                        "buoyancy term -- the run would integrate a "
                        "moisture loading that the declaration says exists "
                        "and the dynamics never sees.  A scalar that "
                        "genuinely loads the air belongs to a scheme row, "
                        "not to an extras declaration"
                    ),
                    f"declare {item.name!r} as {NUMBER!r}, "
                    f"{MASS_COMPONENT!r} or {VOLUME!r}",
                )
        extras = tuple(replace(item, declared_extra=True) for item in extras)
        extra_accumulators = tuple(
            replace(item, declared_extra=True) for item in accumulators
        )
        for item in extra_accumulators:
            if not isinstance(item, SurfaceAccumulatorDeclaration):
                raise TypeError(
                    "declared extra accumulators must be "
                    "SurfaceAccumulatorDeclaration instances"
                )
        return replace(
            self,
            species=self.species + extras,
            surface_accumulators=self.surface_accumulators
            + extra_accumulators,
        )

    def require_engine_scheme(self) -> str:
        """The engine's ``microphysics_scheme``, or refuse by name."""

        if self.engine_scheme is not None:
            return self.engine_scheme
        raise ConfigurationRefusal(
            "config_microp_scheme",
            self.scheme_name,
            self.engine_absent_reason,
            "config_microp_scheme in "
            + " or ".join(
                repr(row.scheme_name)
                for row in _ROWS.values()
                if row.engine_scheme is not None
            ),
        )


# ---------------------------------------------------------------------------
# The rows.
#
# Species order is the ENGINE's own order where the pinned engine carries a
# row (woof/core/mpas_column_batch.py ``_SPECIES_BY_SCHEME``, read off the
# published 2.6.3 wheel), so the door's array and the seam's kwargs agree
# without a permutation anywhere.  All three engine rows already order
# their mass-loading species first, so the contiguous-prefix invariant
# costs no reordering.
# ---------------------------------------------------------------------------


#: A row's ``engine_scheme`` is the frozen column batch's scheme name where
#: that batch carries the scheme (wsm6, p3, thompson_aero, tempo), and the
#: engine's own selector (``mp_physics=N``) where the scheme runs through the
#: generic column seam, which takes the selector itself.  The "declared and
#: refused" rows this file used to carry (Kessler, Thompson mp=8) retired
#: with the limit they cited -- a frozen batch that built four schemes --
#: on 2026-10-11: every microphysics of WOOF's registry is reachable, and a
#: scheme with no row here gets one derived from the engine's plan
#: (:func:`engine_species_row`).


def _mass(name: str, history: str, description: str) -> SpeciesDeclaration:
    return SpeciesDeclaration(
        name=name,
        species_class=MASS_LOADING,
        transported=True,
        history_name=history,
        description=description,
    )


#: WSM6, the frozen lane.  Six mass species, all loading -- which is exactly
#: why the welded chain worked and why nothing caught the assumption.
WSM6_SPECIES_ROW = MicrophysicsSpeciesRow(
    scheme_name="mp_wsm6",
    mp_physics=6,
    engine_scheme="wsm6",
    species=(
        _mass("qv", "QVAPOR", "water vapour mixing ratio"),
        _mass("qc", "QCLOUD", "cloud water mixing ratio"),
        _mass("qr", "QRAIN", "rain water mixing ratio"),
        _mass("qi", "QICE", "cloud ice mixing ratio"),
        _mass("qs", "QSNOW", "snow mixing ratio"),
        _mass("qg", "QGRAUP", "graupel mixing ratio"),
    ),
    surface_accumulators=(
        SurfaceAccumulatorDeclaration(
            "rainnc", "float32", "RAINNC",
            "accumulated grid-scale precipitation",
        ),
        SurfaceAccumulatorDeclaration(
            "snownc", "float32", "SNOWNC", "accumulated grid-scale snow",
        ),
        SurfaceAccumulatorDeclaration(
            "graupelnc", "float32", "GRAUPELNC",
            "accumulated grid-scale graupel",
        ),
    ),
    radius_names=("effc", "effi", "effs"),
    restart_extra_names=("h_diabatic",),
)


#: P3 one-category two-moment ice (woof mp_physics=50).  The eight scalars
#: WRF's own mp=50 driver arm transports.  ``qir`` is the rime mass held
#: inside ``qi`` -- a mass, transported, and deliberately NOT loading; see
#: the module docstring.  ``qib`` is the rime volume.  No qs and no qg: P3's
#: single ice category spans what other schemes split into snow, graupel and
#: hail, which is also why the engine binds five precipitation slots and no
#: graupel accumulator.
P3_SPECIES_ROW = MicrophysicsSpeciesRow(
    scheme_name="mp_p3",
    mp_physics=50,
    engine_scheme="p3",
    species=(
        _mass("qv", "QVAPOR", "water vapour mixing ratio"),
        _mass("qc", "QCLOUD", "cloud water mixing ratio"),
        _mass("qr", "QRAIN", "rain water mixing ratio"),
        _mass("qi", "QICE", "total ice mass mixing ratio"),
        SpeciesDeclaration(
            name="ni",
            species_class=NUMBER,
            transported=True,
            history_name="QNICE",
            description="ice number concentration",
        ),
        SpeciesDeclaration(
            name="nr",
            species_class=NUMBER,
            transported=True,
            history_name="QNRAIN",
            description="rain number concentration",
        ),
        SpeciesDeclaration(
            name="qir",
            species_class=MASS_COMPONENT,
            transported=True,
            history_name="QIR",
            description="rime ice mass mixing ratio, a component of qi",
        ),
        SpeciesDeclaration(
            name="qib",
            species_class=VOLUME,
            transported=True,
            history_name="QIB",
            description="rime ice volume mixing ratio",
        ),
    ),
    surface_accumulators=(
        SurfaceAccumulatorDeclaration(
            "rainnc", "float32", "RAINNC",
            "accumulated grid-scale precipitation",
        ),
        SurfaceAccumulatorDeclaration(
            "snownc", "float32", "SNOWNC", "accumulated grid-scale snow",
        ),
    ),
    radius_names=("effc", "effi"),
    restart_extra_names=("h_diabatic", "th_old", "qv_old"),
)


#: Thompson aerosol-aware (woof mp_physics=28).  The ELEVEN scalars WRF's
#: own mp=28 driver arm transports and the engine's ``thompson_aero`` row
#: carries (gpuwm 2.6.3): WSM6's six masses, the three number moments and
#: the two aerosol number tracers.  ``nc`` is prognostic here, which is the
#: whole difference from classic Thompson.  The seam refuses ``rho_dry`` on
#: this row (mp_gt_driver builds its own density), refuses the rime pair,
#: and derives the surface aerosol emissions at first contact; radii are
#: WSM6's three.
#:
#: This row was DECLARED AND REFUSED from 2026-09-01 until the engine
#: published its mp=28 row, and the refusal retired exactly as its text
#: said it would: the engine gained a third row, the gate that held the
#: engine table to two rows failed, and the row took an ``engine_scheme``
#: with no other change in the door.  The number-block order below is the
#: ENGINE's (ni, nr, nc, nwfa, nifa), checked against
#: ``_SPECIES_BY_SCHEME`` by tests/test_species_row.py; the provisional
#: order the refused row carried (nr, ni, nc, nifa, nwfa) was this tree's
#: own precedent from a time the engine supplied none, and the engine's
#: published row is the precedent now.
THOMPSON_AEROSOL_SPECIES_ROW = MicrophysicsSpeciesRow(
    scheme_name="mp_thompson_aerosols",
    mp_physics=28,
    engine_scheme="thompson_aero",
    species=(
        _mass("qv", "QVAPOR", "water vapour mixing ratio"),
        _mass("qc", "QCLOUD", "cloud water mixing ratio"),
        _mass("qr", "QRAIN", "rain water mixing ratio"),
        _mass("qi", "QICE", "cloud ice mixing ratio"),
        _mass("qs", "QSNOW", "snow mixing ratio"),
        _mass("qg", "QGRAUP", "graupel mixing ratio"),
        SpeciesDeclaration(
            name="ni",
            species_class=NUMBER,
            transported=True,
            history_name="QNICE",
            description="ice number concentration",
        ),
        SpeciesDeclaration(
            name="nr",
            species_class=NUMBER,
            transported=True,
            history_name="QNRAIN",
            description="rain number concentration",
        ),
        SpeciesDeclaration(
            name="nc",
            species_class=NUMBER,
            transported=True,
            history_name="QNDROP",
            description="cloud droplet number concentration",
        ),
        SpeciesDeclaration(
            name="nwfa",
            species_class=NUMBER,
            transported=True,
            history_name="QNWFA",
            description="water-friendly aerosol number concentration",
        ),
        SpeciesDeclaration(
            name="nifa",
            species_class=NUMBER,
            transported=True,
            history_name="QNIFA",
            description="ice-friendly aerosol number concentration",
        ),
    ),
    surface_accumulators=(
        SurfaceAccumulatorDeclaration(
            "rainnc", "float32", "RAINNC",
            "accumulated grid-scale precipitation",
        ),
        SurfaceAccumulatorDeclaration(
            "snownc", "float32", "SNOWNC", "accumulated grid-scale snow",
        ),
        SurfaceAccumulatorDeclaration(
            "graupelnc", "float32", "GRAUPELNC",
            "accumulated grid-scale graupel",
        ),
    ),
    radius_names=("effc", "effi", "effs"),
    restart_extra_names=("h_diabatic",),
)


#: Kessler warm rain (woof mp_physics=1).  Three mass species, no ice.
#: Runs through the engine's generic column seam.
KESSLER_SPECIES_ROW = MicrophysicsSpeciesRow(
    scheme_name="mp_kessler",
    mp_physics=1,
    engine_scheme="mp_physics=1",
    species=(
        _mass("qv", "QVAPOR", "water vapour mixing ratio"),
        _mass("qc", "QCLOUD", "cloud water mixing ratio"),
        _mass("qr", "QRAIN", "rain water mixing ratio"),
    ),
    surface_accumulators=(
        SurfaceAccumulatorDeclaration(
            "rainnc", "float32", "RAINNC",
            "accumulated grid-scale precipitation",
        ),
    ),
    radius_names=(),
    restart_extra_names=("h_diabatic",),
)


#: Thompson two-moment (woof mp_physics=8).  The WSM6 six plus ice and
#: rain number.  Runs through the engine's generic column seam.
THOMPSON_SPECIES_ROW = MicrophysicsSpeciesRow(
    scheme_name="mp_thompson",
    mp_physics=8,
    engine_scheme="mp_physics=8",
    species=(
        _mass("qv", "QVAPOR", "water vapour mixing ratio"),
        _mass("qc", "QCLOUD", "cloud water mixing ratio"),
        _mass("qr", "QRAIN", "rain water mixing ratio"),
        _mass("qi", "QICE", "cloud ice mixing ratio"),
        _mass("qs", "QSNOW", "snow mixing ratio"),
        _mass("qg", "QGRAUP", "graupel mixing ratio"),
        SpeciesDeclaration(
            name="ni",
            species_class=NUMBER,
            transported=True,
            history_name="QNICE",
            description="ice number concentration",
        ),
        SpeciesDeclaration(
            name="nr",
            species_class=NUMBER,
            transported=True,
            history_name="QNRAIN",
            description="rain number concentration",
        ),
    ),
    surface_accumulators=(
        SurfaceAccumulatorDeclaration(
            "rainnc", "float32", "RAINNC",
            "accumulated grid-scale precipitation",
        ),
        SurfaceAccumulatorDeclaration(
            "snownc", "float32", "SNOWNC", "accumulated grid-scale snow",
        ),
        SurfaceAccumulatorDeclaration(
            "graupelnc", "float32", "GRAUPELNC",
            "accumulated grid-scale graupel",
        ),
    ),
    radius_names=("effc", "effi", "effs"),
    restart_extra_names=("h_diabatic",),
)


TEMPO_SPECIES_ROW = replace(
    THOMPSON_AEROSOL_SPECIES_ROW,
    scheme_name="mp_tempo", mp_physics=900, engine_scheme="tempo",
    species=THOMPSON_AEROSOL_SPECIES_ROW.species + (
        SpeciesDeclaration("ng", NUMBER, True, "QNGRAUPEL", "graupel number concentration"),
        SpeciesDeclaration("volg", VOLUME, True, "VOLG", "graupel volume in m3 per kg"),
    ),
)

_ROWS: Mapping[str, MicrophysicsSpeciesRow] = MappingProxyType(
    {
        row.scheme_name: row
        for row in (
            KESSLER_SPECIES_ROW,
            WSM6_SPECIES_ROW,
            THOMPSON_SPECIES_ROW,
            THOMPSON_AEROSOL_SPECIES_ROW,
            TEMPO_SPECIES_ROW,
            P3_SPECIES_ROW,
        )
    }
)

_ROWS_BY_SELECTOR: Mapping[int, MicrophysicsSpeciesRow] = MappingProxyType(
    {row.mp_physics: row for row in _ROWS.values()}
)


def registered_species_rows() -> Mapping[str, MicrophysicsSpeciesRow]:
    """Every declared row, keyed by ``config_microp_scheme`` value: the rows
    written here and the rows derived from the engine's own plan
    (:func:`register_engine_row`)."""

    if not _ENGINE_ROWS:
        return _ROWS
    return MappingProxyType({**_ROWS, **_ENGINE_ROWS})


def declared_species_rows() -> Mapping[str, MicrophysicsSpeciesRow]:
    """The rows written in this file alone (no engine-derived row)."""

    return _ROWS


#: Rows derived from the ENGINE's account of a microphysics selection
#: (``woof.core.mpas_column_suite.column_physics_plan``: the transported
#: species and which of them load the air), keyed by scheme name.  A scheme
#: WOOF runs needs no row written in this file: the resolver
#: (:mod:`woof.hex.physics_suites`) derives one from the plan and registers
#: it here.  A derived row never shadows a declared one and is never
#: replaced by a different row under the same name.
_ENGINE_ROWS: dict[str, MicrophysicsSpeciesRow] = {}

_ACCUMULATOR_TEXT = {
    "RAINNC": "accumulated grid-scale precipitation",
    "SNOWNC": "accumulated grid-scale snow",
    "GRAUPELNC": "accumulated grid-scale graupel",
    "HAILNC": "accumulated grid-scale hail",
}


def engine_species_row(
    *,
    scheme_name: str,
    mp_physics: int,
    species: Sequence[str],
    mass_species: Sequence[str],
    surface_accumulators: Sequence[str] | None = None,
    radius_names: Sequence[str] = (),
    history_names: Mapping[str, str] | None = None,
) -> MicrophysicsSpeciesRow:
    """A species row built from the engine's plan for one microphysics.

    ``species`` is the engine's transported set and ``mass_species`` the
    subset that loads the air.  The hex scalar block orders the loading
    species first (vapour at index 0), then the rest in the engine's order:
    the moist-coefficient kernels sum a contiguous leading run, and the seam
    takes every species by name, so the block's order is hex's to choose.
    ``surface_accumulators`` are the engine's bucket keys beside RAINC;
    when the plan does not publish them they follow from the species
    (RAINNC always, SNOWNC with ``qs``, GRAUPELNC with ``qg``).
    """

    given = tuple(_clean_name(name, "species name") for name in species)
    loading = tuple(_clean_name(name, "species name") for name in mass_species)
    unknown = sorted(set(loading) - set(given))
    if unknown:
        raise ValueError(
            f"the engine plan for {scheme_name!r} names mass species {unknown} "
            f"that are not among its transported species {given}")
    if "qv" not in loading:
        raise ValueError(
            f"the engine plan for {scheme_name!r} does not list qv among its "
            f"mass species {loading}; vapour loads the air")
    six = ("qv", "qc", "qr", "qi", "qs", "qg")
    ordered = (
        tuple(n for n in six if n in loading)
        + tuple(n for n in given if n in loading and n not in six)
        + tuple(n for n in given if n not in loading))
    spelled = dict(history_names or {})
    declarations = []
    for name in ordered:
        if name in loading:
            kind, what = MASS_LOADING, "mixing ratio"
        elif "vol" in name:
            kind, what = VOLUME, "volume"
        else:
            # The plan splits species two ways: loads the air, or does not.
            # Everything that does not is carried as a number-class scalar
            # (transported, positive-definite, outside qtot).
            kind, what = NUMBER, "number concentration"
        declarations.append(SpeciesDeclaration(
            name=name, species_class=kind, transported=True,
            history_name=str(spelled.get(name, name.upper())),
            description=f"{name} {what} (engine species of {scheme_name})",
        ))
    if surface_accumulators is None:
        keys = ["RAINNC"]
        if "qs" in given:
            keys.append("SNOWNC")
        if "qg" in given:
            keys.append("GRAUPELNC")
    else:
        keys = [str(key) for key in surface_accumulators if str(key) != "RAINC"]
    accumulators = tuple(
        SurfaceAccumulatorDeclaration(
            key.lower(), "float32", key,
            _ACCUMULATOR_TEXT.get(key, f"accumulated {key}"),
            # WRF Morrison sums signed category fallout (morrison.cu,
            # precipitation sums). The engine preserves these bucket words;
            # only its consumer-facing snow fraction is bounded to [0, 1].
            signed_fallout=(int(mp_physics) == 10
                            and key in ("RAINNC", "SNOWNC", "GRAUPELNC")))
        for key in keys
    )
    return MicrophysicsSpeciesRow(
        scheme_name=_clean_name(scheme_name, "scheme name"),
        mp_physics=int(mp_physics),
        engine_scheme=f"mp_physics={int(mp_physics)}",
        species=tuple(declarations),
        surface_accumulators=accumulators,
        radius_names=tuple(radius_names),
        restart_extra_names=("h_diabatic",),
    )


def register_engine_row(row: MicrophysicsSpeciesRow) -> MicrophysicsSpeciesRow:
    """Register a row derived from the engine's plan under its scheme name."""

    if not isinstance(row, MicrophysicsSpeciesRow):
        raise TypeError("an engine row must be a MicrophysicsSpeciesRow")
    if row.scheme_name in _ROWS:
        raise ConfigurationRefusal(
            "config_microp_scheme", row.scheme_name,
            "a declared species row already carries this name; a derived row "
            "that replaced it would change what a plain configuration runs",
            "a scheme name no declared row carries",
        )
    held = _ENGINE_ROWS.get(row.scheme_name)
    if held is not None:
        if held != row:
            raise ConfigurationRefusal(
                "config_microp_scheme", row.scheme_name,
                "an engine-derived species row is already registered under "
                "this name with other species; rows are not replaceable",
                "one engine tree per process",
            )
        return held
    _ENGINE_ROWS[row.scheme_name] = row
    return row


#: Rows a lane extended with :meth:`MicrophysicsSpeciesRow.with_extras`,
#: keyed by their exact scalar-name block.  A consumer that carries only
#: the NAMES of its block (the dynamics driver, the history writer, the
#: adapter's bucket check) resolves an extended block to the extended row
#: -- extras, history names and accumulators intact -- instead of to the
#: base row the prefix arm would return with the extras' declarations
#: dropped.  Registration is the provider's job on import; a block whose
#: extras were never registered still resolves by prefix, which is the
#: right answer for a caller that only needs the loading count.
_EXTENDED_ROWS: dict[tuple[str, ...], MicrophysicsSpeciesRow] = {}

#: Extended rows a provider also registered under a SCHEME ALIAS -- a
#: ``config_microp_scheme`` value of the provider's choosing that selects
#: the extended row wherever the run's row is resolved from the config
#: rather than from a scalar block: the dynamics driver's width check and
#: receipts, the configuration's engine-scheme admission, the door.  The
#: alias must not shadow a scheme row: the scheme rows are the engine's
#: inventory and an alias that replaced one would change what a plain
#: configuration runs by import order.
_EXTENDED_SCHEMES: dict[str, MicrophysicsSpeciesRow] = {}


def register_extended_row(
    row: MicrophysicsSpeciesRow, *, scheme_alias: str | None = None
) -> MicrophysicsSpeciesRow:
    """Register a row carrying declared extras under its exact name block.

    The base the row extends must be a registered scheme row and must be
    the row's leading run; a second registration under the same block
    must be the same row (a different one is refused, never replaced --
    which extras a block carries is a declaration, not an import order).
    ``scheme_alias`` additionally registers the row as a
    ``config_microp_scheme`` value (see :data:`_EXTENDED_SCHEMES`).
    """

    if not isinstance(row, MicrophysicsSpeciesRow):
        raise TypeError("an extended row must be a MicrophysicsSpeciesRow")
    extras = row.declared_extras()
    extra_accumulators = row.declared_extra_accumulators()
    if not extras and not extra_accumulators:
        raise ValueError(
            f"row {row.scheme_name!r} declares no extras; the scheme rows "
            "themselves are registered by scheme name, not here"
        )
    base = _ROWS.get(row.scheme_name)
    if base is None:
        raise ConfigurationRefusal(
            "config_microp_scheme",
            row.scheme_name,
            (
                f"extended row {row.scheme_name!r} extends no registered "
                f"scheme row.  Declared rows: {sorted(_ROWS)}"
            ),
            f"config_microp_scheme in {sorted(_ROWS)}",
        )
    names = row.names()
    if names[: base.n_species()] != base.names():
        raise ValueError(
            f"extended row {row.scheme_name!r} does not lead with its base "
            f"row's species {base.names()}; got {names}"
        )
    existing = _EXTENDED_ROWS.get(names)
    if existing is not None and existing != row:
        raise ConfigurationRefusal(
            "scalar_names",
            names,
            (
                "an extended row is already registered under this exact "
                "scalar block and rows are not replaceable; which extras a "
                "block carries is a declaration, not an import order"
            ),
            "a different scalar block, or the row already registered",
        )
    if scheme_alias is not None:
        alias = _clean_name(scheme_alias, "scheme alias")
        if alias in _ROWS:
            raise ConfigurationRefusal(
                "config_microp_scheme",
                alias,
                (
                    f"scheme alias {alias!r} is a scheme row's own name; an "
                    "alias that shadowed a scheme row would change what a "
                    "plain configuration runs by import order"
                ),
                "an alias that is not a scheme row name",
            )
        held = _EXTENDED_SCHEMES.get(alias)
        if held is not None and held != row:
            raise ConfigurationRefusal(
                "config_microp_scheme",
                alias,
                (
                    f"scheme alias {alias!r} already selects a different "
                    "extended row and aliases are not replaceable"
                ),
                "a different alias, or the row already registered",
            )
        _EXTENDED_SCHEMES[alias] = row
    if existing is None:
        _EXTENDED_ROWS[names] = row
        return row
    return existing


def registered_extended_rows() -> Mapping[tuple[str, ...], MicrophysicsSpeciesRow]:
    """A read-only view of every registered extended row, by name block."""

    return MappingProxyType(dict(_EXTENDED_ROWS))


def registered_scheme_aliases() -> Mapping[str, MicrophysicsSpeciesRow]:
    """A read-only view of every scheme alias an extended row registered."""

    return MappingProxyType(dict(_EXTENDED_SCHEMES))


def species_row_for_scheme(scheme_name: str) -> MicrophysicsSpeciesRow:
    """The row for an MPAS microphysics scheme name, or refuse by name."""

    wanted = _clean_name(scheme_name, "config_microp_scheme")
    row = _ROWS.get(wanted)
    if row is not None:
        return row
    row = _EXTENDED_SCHEMES.get(wanted)
    if row is not None:
        return row
    row = _ENGINE_ROWS.get(wanted)
    if row is not None:
        return row
    raise ConfigurationRefusal(
        "config_microp_scheme",
        scheme_name,
        (
            f"no species row declares {wanted!r}.  The forecast door sizes "
            "its scalar block, its preparation pointer table, its moist "
            "coefficients, its carrier contracts and its history names from "
            "the row, so a scheme with no row would run with a species set "
            "nobody declared.  Declared rows: "
            f"{sorted(_ROWS)}"
        ),
        f"config_microp_scheme in {sorted(_ROWS)}",
    )


def species_row_for_names(
    names: Sequence[str],
) -> MicrophysicsSpeciesRow:
    """The row whose species a model scalar block holds, or refuse by name.

    A call site that carries only the ordered NAMES of its scalar block --
    the dynamics driver is one -- needs the row to learn how many leading
    species load the air.  An exact match wins; failing that, a registered
    row that is a PREFIX of the given names matches, which is what a row
    carrying a lane's declared extras looks like.  The prefix match is safe
    for exactly the reason ``with_extras`` refuses a mass-loading extra: the
    appended species are never loading, so the base row's
    ``n_mass_loading`` is the extended row's too.
    """

    given = tuple(_clean_name(name, "scalar name") for name in names)
    for row in _ROWS.values():
        if row.names() == given:
            return row
    extended = _EXTENDED_ROWS.get(given)
    if extended is not None:
        return extended
    for row in _ENGINE_ROWS.values():
        if row.names() == given:
            return row
    # LONGEST prefix wins, and that is not a nicety: mp_kessler declares
    # (qv, qc, qr), which is a prefix of every other row, so a first-match
    # rule resolves an eleven-species aerosol block plus a lane's extras to
    # the three-species warm-rain row -- and the caller then sums three
    # species into qtot and prepares three of eleven.  Measured against the
    # registered provider row's block, which is exactly that shape.
    best: MicrophysicsSpeciesRow | None = None
    for row in _ROWS.values():
        base = row.names()
        if len(given) > len(base) and given[: len(base)] == base:
            if best is None or len(base) > len(best.names()):
                best = row
    if best is not None:
        return best
    raise ConfigurationRefusal(
        "scalar_names",
        given,
        (
            "no declared species row matches this scalar block, either "
            "exactly or as a leading run followed by declared extras.  The "
            "moist coefficients sum the leading mass-LOADING species and "
            "the count comes from the row, so an unmatched block would be "
            "summed to a length nobody declared -- silently including a "
            "number concentration or silently dropping a mass.  Declared "
            f"rows: {sorted(_ROWS)}"
        ),
        "a scalar block matching a declared species row",
    )


def species_row_for_mp_physics(mp_physics: int) -> MicrophysicsSpeciesRow:
    """The row for a woof numeric selector, or refuse by name."""

    selector = int(mp_physics)
    row = _ROWS_BY_SELECTOR.get(selector)
    if row is not None:
        return row
    for row in _ENGINE_ROWS.values():
        if row.mp_physics == selector:
            return row
    raise ConfigurationRefusal(
        "mp_physics",
        selector,
        (
            f"no species row declares woof mp_physics={selector}.  "
            "Declared selectors: "
            f"{sorted(_ROWS_BY_SELECTOR)}"
        ),
        f"mp_physics in {sorted(_ROWS_BY_SELECTOR)}",
    )
