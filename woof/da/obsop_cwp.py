"""H_CWP(x): the model's cloud water path, for differencing against GOES.

The satellite sibling of :func:`woof.da.obsop.simulated_reflectivity`, and
built the same way -- it delegates to the model's own definitions rather
than inventing a parallel one, and it says exactly which definitions it
took and why.

## The integral

``CWP(j, i) = 1000 * sum_k q_cond[k,j,i] * (c1h[k]*mu[j,i] + c2h[k])
* (-dnw[k]) / G``, in g m-2.

The measure ``(c1h*mu + c2h)*(-dnw)/G`` is the model's own eta-coordinate
column mass per unit area, taken verbatim from
``woof/verify/cases/moist_bubble.py:125-134`` (``_water_mass``), which is
this project's only pre-existing column water integral and the one its
conservation case is judged against.  It is used here in preference to
``rho_d * dz`` -- ``(1/state.alt) * (z_w[k+1]-z_w[k])`` -- for two
reasons, neither cosmetic:

* It is **exact in the discretisation**.  ``woof/core/kernels/diagnostics.cu``
  defines ``alt[k] = -(ph[k+1]-ph[k]) * rdnw[k] / (c1h[k]*mu + c2h[k])``,
  so ``dz/alt`` and ``(c1h*mu+c2h)*(-dnw)/G`` are the same number up to
  float rounding -- but the mass form telescopes exactly to the column dry
  mass, and the geometric form does not.
* ``alt`` **goes stale**.  ``woof/da/perturb.py:1639`` documents that
  ``p``/``al``/``alt`` are invalid after any state mutation until
  ``update_diagnostics`` reruns.  An observation operator evaluated on a
  perturbed or incremented member would silently use the pre-mutation
  density.  ``c1h``, ``c2h``, ``dnw`` are setup arrays and cannot go stale;
  ``mu`` is ``mub2d + mup`` and ``mup`` is prognostic and serialized.

``G`` is ``woof.core.constants.G`` = 9.81, the constant the measure it
came from uses.  Noted because it is not the only gravity in the tree:
``woof/core/rrtmgp.py:1096`` builds its condensate paths with 9.80665, so
a CWP compared against RRTMGP's ``clwp + ciwp`` differs by 0.035% on this
constant alone.

The ``1000 *`` converts kg m-2 to g m-2, matching
``woof/core/rrtmg_legacy_prep.py:536`` (which also emits g m-2) and the
unit ``gpuwm-obs.goes-cwp.v1`` publishes.

## Which condensate

The retrieval is optical: DCOMP inverts a reflected-radiance pair for
optical depth and effective radius, so the model condensate that
corresponds to it is the condensate the model's own radiation code treats
as optically active.  That definition already exists here, twice, and
agrees with itself:

* ``woof/core/rrtmg_legacy_prep.py:538-539`` -- ``gliqwp`` from ``qc``,
  ``gicewp`` from ``qi + qs``.
* ``woof/core/rrtmgp.py:1097-1098`` -- ``clwp = qc * mass_path``,
  ``ciwp = (qi + qs) * mass_path``.

So liquid is ``qc`` and ice is ``qi + qs``.  **Rain is excluded**, which
is not an omission: ``woof/core/rrtmgp.py:1243`` states that rain never
enters the model's own cloud-fraction condensate, and
``woof/core/refl.py:13`` puts ``qr``/``qs``/``qg`` on the *reflectivity*
side of the split.  Graupel and hail are excluded for the operator spec's
reason: large precipitating ice contributes little optical depth per unit
mass.

### The one place this diverges from `docs/obs-goes-cwp-operator-spec.md`

That spec's v1 rule is ``q_cond = qc + qi`` for an ice-phase observation,
with ``qs`` excluded.  This module's default includes ``qs``, because the
model's own optical definition includes it in both radiation paths, and
because a spec written before the operator was built is a weaker authority
than the code the operator has to be consistent with.  The divergence is
deliberate, it is recorded in every receipt as ``composition``, and the
spec's variant is one argument away -- ``CwpComposition(ice=("qc", "qi"))``.
Which is right is a scoreboard question, exactly as the spec says
("Revisit with the scorecard, not by argument"), and both arms are
reachable without a code change.

## Phase-aware composition

The retrieval's phase is *cloud-top* phase, so the operator cannot
integrate a phase-matched column naively.  Following the spec:

* observed liquid or supercooled liquid: ``qc``.
* observed ice or mixed: ``qc + qi + qs``.  An ice-topped deep column
  hides liquid beneath its top, and comparing model ice alone against a
  retrieval that integrated the whole optical column biases H(x) low.
* observed **clear-sky zero**: ``qc + qi + qs``.  The zero must be able to
  remove any cloud the model invented, whatever its phase; composing a
  clear observation against ``qc`` alone would leave model ice invisible
  to the one observation that most confidently says it should not exist.

The phase used is the **observation's**, never the model's.  That keeps H
a fixed function of the state given the data -- a state-dependent species
selection would make H discontinuous in ``x`` at the phase boundary, and
an ensemble straddling that boundary would produce a covariance the filter
has no right to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from woof.core import constants as c
from woof.obs.goes_cwp import CLASS_CLEAR, CLASS_ICE, CLASS_LIQUID


def _module_for(array):
    """The array module (CuPy or NumPy) an array belongs to."""
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy
    return np

#: kg m-2 -> g m-2, the unit the packs and the gridded product both use.
KG_TO_G = 1000.0


class CwpOperatorError(ValueError):
    """The state, the config and the composition cannot be reconciled."""


@dataclass(frozen=True)
class CwpComposition:
    """Which mixing ratios are integrated, per observed phase class.

    Defaults are the model's own optical-condensate definition.  See this
    module's docstring for the citation and for the one deliberate
    divergence from ``docs/obs-goes-cwp-operator-spec.md``.
    """

    liquid: tuple[str, ...] = ("qc",)
    ice: tuple[str, ...] = ("qc", "qi", "qs")
    clear: tuple[str, ...] = ("qc", "qi", "qs")

    def for_class(self, klass: int) -> tuple[str, ...]:
        if klass == CLASS_CLEAR:
            return tuple(self.clear)
        if klass == CLASS_LIQUID:
            return tuple(self.liquid)
        if klass == CLASS_ICE:
            return tuple(self.ice)
        raise CwpOperatorError(
            f"no condensate composition for phase class {klass!r}; the "
            f"classes are {CLASS_CLEAR} clear, {CLASS_LIQUID} liquid, "
            f"{CLASS_ICE} ice")

    def species(self) -> tuple[str, ...]:
        """Every species any branch names, deduplicated, in a stable order."""

        seen: list[str] = []
        for group in (self.liquid, self.ice, self.clear):
            for name in group:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def to_payload(self) -> dict:
        return {
            "liquid": list(self.liquid),
            "ice": list(self.ice),
            "clear": list(self.clear),
            "diverges_from_spec": (
                "docs/obs-goes-cwp-operator-spec.md v1 excludes qs from the "
                "ice branch; this default includes it, following the "
                "model's own radiation definition. Deliberate, recorded, "
                "and reversible with CwpComposition(ice=('qc', 'qi'))"
                if "qs" in self.ice else
                "matches docs/obs-goes-cwp-operator-spec.md v1 (qs excluded "
                "from the ice branch)"),
        }


def column_mass_per_area(c1h, c2h, dnw, mu, *, xp=np):
    """``(c1h*mu + c2h) * (-dnw) / G``, ``(nz, ny, nx)`` in kg m-2.

    The model's own layer mass per unit area, the quantity
    ``moist_bubble._water_mass`` multiplies a mixing ratio by.  Positive:
    ``dnw`` is negative bottom-up (``woof/core/grid.py:1-23``), which is
    the sign trap this function exists to contain in one place.
    """

    c1h = xp.asarray(c1h, dtype=xp.float64)
    c2h = xp.asarray(c2h, dtype=xp.float64)
    dnw = xp.asarray(dnw, dtype=xp.float64)
    mu = xp.asarray(mu, dtype=xp.float64)
    if c1h.ndim != 1 or c2h.ndim != 1 or dnw.ndim != 1:
        raise CwpOperatorError(
            f"c1h/c2h/dnw must be (nz,), got {c1h.shape}/{c2h.shape}/"
            f"{dnw.shape}")
    if not (c1h.shape == c2h.shape == dnw.shape):
        raise CwpOperatorError(
            f"c1h {c1h.shape}, c2h {c2h.shape} and dnw {dnw.shape} must "
            "share a length")
    if mu.ndim != 2:
        raise CwpOperatorError(f"mu must be (ny, nx), got {mu.shape}")
    if bool(xp.any(dnw >= 0.0)):
        raise CwpOperatorError(
            "dnw has a non-negative entry. Eta decreases upward in this "
            "model (woof/core/grid.py), so every dnw is strictly negative "
            "and the integral carries an explicit (-dnw). A positive dnw "
            "means the caller handed in a sign-flipped or reversed "
            "coordinate, and the integral would come out negative")
    chm = c1h[:, None, None] * mu[None] + c2h[:, None, None]
    return chm * (-dnw[:, None, None]) / float(c.G)


def cloud_water_path(species, *, c1h, c2h, dnw, mu):
    """Integrate one already-selected condensate set, ``(ny, nx)`` g m-2.

    ``species`` is an iterable of ``(nz, ny, nx)`` mixing ratios in
    kg kg-1 of dry air, which is the unit ``woof/core/state.py:381``
    declares for every ``q*``; that is why the dry-mass measure is the
    right one and no ``(1 + qv)`` moist-density correction appears (see
    ``woof/core/physics.py:1270``, which is a different density for a
    different consumer).
    """

    arrays = [array for array in species if array is not None]
    if not arrays:
        raise CwpOperatorError(
            "no condensate species were supplied, so this would integrate "
            "zero and report it as a model with no cloud. An intentionally "
            "empty composition is not expressible here")
    xp = _module_for(arrays[0])
    total = xp.asarray(arrays[0], dtype=xp.float64)
    for array in arrays[1:]:
        total = total + xp.asarray(array, dtype=xp.float64)
    mass = column_mass_per_area(c1h, c2h, dnw, mu, xp=xp)
    if total.shape != mass.shape:
        raise CwpOperatorError(
            f"condensate is {total.shape} but the column measure is "
            f"{mass.shape}")
    return xp.sum(total * mass, axis=0) * KG_TO_G


def simulated_cloud_water_path(state, cfg, *, c1h=None, c2h=None, dnw=None,
                               mu=None, obs_class=None,
                               composition: CwpComposition | None = None,
                               allow_missing_species: bool = False):
    """H_CWP(x): cloud water path ``(ny, nx)`` in g m-2.

    ``cfg`` is required and positional, exactly as it is for
    :func:`woof.da.obsop.simulated_reflectivity` and for the same reason:
    which condensate species the state is *supposed* to carry is a property
    of the active microphysics scheme, and ``DomainState`` does not carry
    its own ``RunConfig``.  Without it, a warm-rain run would integrate
    ``qc`` alone against an ice-phase satellite observation and report a
    model with no ice, quietly and wrongly.

    ``c1h``/``c2h``/``dnw``/``mu`` default to the state's own
    (``mu = mub2d + mup``, ``state.total_mu()``).  They are arguments
    because a checkpoint does not serialize the setup arrays -- see
    ``woof/state_serialization_contract.py:57-63`` -- so a
    checkpoint-driven provider must supply them, the same way
    :func:`woof.da.radar_assimilation.scheme_reflectivity_provider` must
    supply ``base_theta``.

    ``obs_class`` is an ``(ny, nx)`` array of
    :data:`woof.obs.goes_cwp.CLASS_CLEAR` / ``CLASS_LIQUID`` / ``CLASS_ICE``
    (anything else means "no observation here", and those columns are
    integrated under the clear composition so the returned field is still
    a complete diagnostic).  ``None`` integrates the clear composition
    everywhere, which is the plain column-condensate diagnostic.
    """

    composition = CwpComposition() if composition is None else composition
    if c1h is None:
        c1h = _require_attr(state, "c1h")
    if c2h is None:
        c2h = _require_attr(state, "c2h")
    if dnw is None:
        dnw = _require_attr(state, "dnw")
    if mu is None:
        mu = state.total_mu() if hasattr(state, "total_mu") else \
            _require_attr(state, "mu")

    available = {}
    missing = []
    for name in composition.species():
        value = getattr(state, name, None)
        if value is None:
            missing.append(name)
        else:
            available[name] = value
    if missing and not allow_missing_species:
        raise CwpOperatorError(
            f"the condensate composition names {missing}, which this state "
            f"does not carry (mp_physics={getattr(cfg, 'mp_physics', None)!r})"
            ". Integrating without them would report a model with no ice "
            "against a retrieval that saw some, biasing H(x) low and the "
            "analysis wetward. Either choose a composition this scheme can "
            "satisfy -- CwpComposition(ice=('qc',), clear=('qc',)) for a "
            "warm-rain run -- or pass allow_missing_species=True and record "
            "that you did")
    if not available:
        raise CwpOperatorError(
            f"this state carries none of {list(composition.species())}; "
            "there is no condensate to integrate")

    def integrate(names):
        chosen = [available[name] for name in names if name in available]
        if not chosen:
            raise CwpOperatorError(
                f"composition branch {list(names)} has no species on this "
                "state, so that branch would integrate to exactly zero and "
                "claim the model is cloud-free wherever it applies")
        return cloud_water_path(chosen, c1h=c1h, c2h=c2h, dnw=dnw, mu=mu)

    if obs_class is None:
        return integrate(composition.clear)

    xp = _module_for(next(iter(available.values())))
    classes = xp.asarray(obs_class)
    result = integrate(composition.clear)
    if classes.shape != result.shape:
        raise CwpOperatorError(
            f"obs_class is {classes.shape} but the model grid is "
            f"{result.shape}")
    for klass, names in ((CLASS_LIQUID, composition.liquid),
                         (CLASS_ICE, composition.ice)):
        if tuple(names) == tuple(composition.clear):
            continue
        where = classes == klass
        if not bool(xp.any(where)):
            continue
        result = xp.where(where, integrate(names), result)
    return result


def checkpoint_cwp_provider(run_cfg, *, c1h, c2h, dnw, mub2d,
                            composition: CwpComposition | None = None,
                            allow_missing_species: bool = False):
    """A CWP H(x) provider bound to the setup arrays a checkpoint lacks.

    ``c1h``, ``c2h``, ``dnw`` and ``mub2d`` are ``STATE_SETUP_ARRAYS``
    (``woof/state_serialization_contract.py:57-63``) and are not in a
    checkpoint; ``mup`` is, so the provider forms ``mu = mub2d + mup`` per
    member and the *perturbed column mass of that member* is what its own
    integral is taken over.

    Returns ``provider(member_index, state_arrays, obs_class)
    -> (ny, nx) g m-2``.  ``obs_class`` is a per-call argument rather than
    a bound one because it must be the class array that survived thinning,
    which is not known when the provider is built.
    """

    composition = CwpComposition() if composition is None else composition
    base_mu = np.asarray(mub2d, dtype=np.float64)
    if base_mu.ndim == 0:
        raise CwpOperatorError(
            "mub2d is a scalar. woof/core/state.py:700-704 sets the scalar "
            "state.mub to None with terrain precisely so unmigrated "
            "consumers fail loudly; pass the (ny, nx) mub2d")
    if base_mu.ndim != 2:
        raise CwpOperatorError(f"mub2d must be (ny, nx), got {base_mu.shape}")

    def provider(member_index: int, state, obs_class=None):
        if "mup" not in state or state.get("mup") is None:
            raise CwpOperatorError(
                f"member {member_index}: checkpoint carries no 'mup', so the "
                "column mass this member's integral is taken over is "
                "unknown. Using the base state's mub2d alone would evaluate "
                "every member's H(x) on the same column mass and erase the "
                "very covariance the filter reads")
        mu = base_mu + np.asarray(state["mup"], dtype=np.float64)
        namespace = _StateView(
            {name: (None if state.get(name) is None
                    else np.asarray(state[name], dtype=np.float64))
             for name in composition.species()})
        return np.asarray(
            simulated_cloud_water_path(
                namespace, run_cfg, c1h=c1h, c2h=c2h, dnw=dnw, mu=mu,
                obs_class=obs_class, composition=composition,
                allow_missing_species=allow_missing_species),
            dtype=np.float64)

    return provider


class _StateView:
    """The minimal duck-typed state the operator reads.

    Same trick as ``woof/da/radar_assimilation.py:510-516``: a checkpoint
    is a mapping, the operator reads attributes, and an explicit view is
    clearer than teaching the operator two access patterns.
    """

    def __init__(self, fields):
        self._fields = dict(fields)

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None


def _require_attr(state, name):
    value = getattr(state, name, None)
    if value is None:
        raise CwpOperatorError(
            f"the state carries no {name!r}, and it was not supplied. It is "
            "setup state (woof/state_serialization_contract.py:57-63) and "
            "a checkpoint does not have it; pass it explicitly")
    return value
