"""CPM-style inflow turbulence seeding for one-way LES nest children.

A one-way LES child receives smoothed parent fields through its lateral
boundary and has to grow its own turbulence while the inflow advects
across the domain.  The measured cost on the shipped 250 m child is a
spin-up fetch larger than the domain itself
(docs/internal/receipts/les/INFLOW-FETCH-D90-2026-08-03.md).  This
module seeds that transition: cell-blocked potential-temperature
perturbations applied to the child's rolling boundary VALUE tables on
the inflow-side relax-zone rows, refreshed on the FORCE cadence, so the
existing Davies relaxation nudges the relax zone toward a perturbed
target and advection carries the blocks into the interior where
buoyancy amplifies them.

Every constant is pinned in
docs/internal/receipts/les/INFLOW-GENERATOR-ACCEPTANCE-V2.md,
registered before any perturbed integration existed.  The mechanism in
one paragraph:

* Only the theta VALUE tables are touched, only on width rows
  ``d in [spec_zone, relax_zone)``: the rows whose ``fcx[d]``
  relaxation target the state_specified_relaxation kernel reads.  The
  spec-zone row, all tendency tables, and every other field are
  byte-untouched, so mass, momentum and the outer specified rim remain
  exactly the parent's.
* The increment is applied in the tables' own WRF-coupled units:
  ``delta_table = (c1h[k]*(mub2d + mu_table) + c2h[k]) * delta_theta``,
  the same hybrid mass coupling ``coupled_current`` (lbc_state.cu)
  uses, so ``delta_theta`` Kelvin shift the relaxation equilibrium by
  ``delta_theta`` Kelvin.
* Amplitude follows the pinned Eckert-number convention
  ``theta_max = scale * U^2 / (Ec * cp)`` with Ec = 0.2 and U the
  face-mean boundary-normal wind over the perturbed depth.
* The perturbed depth is the parent-diagnosed PBLH over the child's
  footprint (the parent physics driver's own ``pblh`` field), times a
  pinned fraction of 1.0, expressed in child base-state half-level
  heights.
* Draws are uniform in [-1, 1), one per (mass level, 8-cell along-face
  block), held for 100 s of model time, from a counter-based NumPy
  Philox generator keyed on (seed, grid_id, face, refresh index): the
  same draw on every card, every run, every restart.  The hold is
  quantised on the parent's CONFIGURED step, which is a run constant,
  so an adaptive parent holds each draw for the same 100 s and the
  index never repeats (:meth:`InflowPerturbation._refresh_for`).

OFF is absolute: when ``inflow_perturbation`` is false (the default)
:func:`build_inflow_perturbation` returns ``None`` and the coupler
executes not one instruction of this module, the OFF trajectory is
byte-identical to a build without this file (acceptance gate G1).
With ``inflow_perturbation_amplitude_scale = 0.0`` the hook runs its
read-only classification and skips every table write (gate G2).

This is an ArWen-over-WRF extension, entered in PROVENANCE.md (D10):
stock WRF v4.6.1 ships ``perturb_bdy`` (stochastic-pattern boundary
TENDENCY perturbation in the stoch package) and no cell-perturbation
path; the OFF default keeps every WRF-parity configuration untouched.
"""

from __future__ import annotations

import math

import numpy as np

from woof.core import constants as c

#: Along-face cell-block size, child cells (acceptance v2 pin).
BLOCK_CELLS = 8
#: Perturbation Eckert number (acceptance v2 pin).
ECKERT = 0.2
#: Fraction of the parent-footprint PBLH the perturbation reaches.
VERTICAL_FRACTION = 1.0
#: Model seconds one unit draw is held before being redrawn.
REFRESH_SECONDS = 100.0
#: Stable face -> RNG key codes.  Never renumber: the code is part of
#: every registered draw's key.
FACE_CODES = {"west": 0, "east": 1, "south": 2, "north": 3}

_FACE_MODES = ("inflow", "outflow")

#: The stable opening of the applicability refusal, quoted verbatim by
#: P6-LES-DECISIONS-RATIFIED-2026-08-05.md (G4) and by the shipped
#: mayfield config's comment.  Both enforcement seams and the shipped-
#: config sweep match on THIS constant rather than on a retyped
#: sentence, so the refusal stays greppable when its wording is edited.
PARENT_PBL_REFUSAL_MARKER = ("inflow_perturbation defines its vertical "
                             "extent")


def parent_pbl_refusal(parent_bl_pbl_physics: int | None) -> str | None:
    """The generator's applicability boundary, in ONE place.

    ``None`` when this parent can define the perturbation's depth, the
    refusal sentence otherwise.  ``None`` for the parent's selector
    means "no parent at all", which is the same answer for the same
    reason: there is no PBLH diagnostic to read.

    Two seams enforce this and they must not be two spellings of it:
    :func:`build_inflow_perturbation` refuses at coupler construction,
    and ``woof.experiment.build_experiment`` refuses at the config load
    every front door shares.  The construction refusal STAYS -- a
    spawned or relocated nest is built outside the config path -- but it
    is no longer the first place anyone learns, which is the defect it
    used to be: G4 records a four-domain LES tree admitted to the card
    and killed twelve seconds in, on a fact that was in the TOML.
    """
    if parent_bl_pbl_physics is not None and int(parent_bl_pbl_physics) != 0:
        return None
    reported = 0 if parent_bl_pbl_physics is None else int(
        parent_bl_pbl_physics)
    return (f"{PARENT_PBL_REFUSAL_MARKER} from the "
            "parent-diagnosed PBLH and therefore requires a parent "
            f"running a PBL scheme; parent bl_pbl_physics={reported}")


def refresh_ticks(parent_step_ticks: int, parent_dt_seconds: float) -> int:
    """One draw's holding BUCKET, in integer clock ticks.

    ``REFRESH_SECONDS`` snapped to a whole number of parent forces (at
    least one), times the ticks in one force.  This is the divisor
    :func:`refresh_index` cuts absolute model time with, and it must be
    a RUN CONSTANT: the pair handed in is the CONFIGURED step, never the
    live adaptive one.  See :meth:`InflowPerturbation._refresh_for`.
    """
    if parent_step_ticks <= 0:
        raise ValueError("parent_step_ticks must be positive")
    if not (parent_dt_seconds > 0.0 and math.isfinite(parent_dt_seconds)):
        raise ValueError("parent dt must be a positive finite number")
    forces_per_refresh = max(
        1, int(round(REFRESH_SECONDS / parent_dt_seconds)))
    return forces_per_refresh * int(parent_step_ticks)


def refresh_index(force_ticks: int, parent_step_ticks: int,
                  parent_dt_seconds: float) -> int:
    """The draw-holding counter for one force, in integer clock ticks.

    ``REFRESH_SECONDS`` is snapped to a whole number of parent forces
    (at least one), and the index is cut from the absolute tick count,
    so a restarted run recomputes the same index at the same model time.

    The pinned formula (INFLOW-GENERATOR-ACCEPTANCE-V2, item 7) is
    unchanged and unchangeable; it is expressed on :func:`refresh_ticks`
    so the bucket has one definition rather than two.  The function is
    correct for RUN-CONSTANT ``(step, dt)`` and only for those --
    ``force_ticks`` is absolute, so a divisor that moves makes
    ``floor(T / D(t))`` neither monotone nor onto.  Callers on a live
    clock go through :meth:`InflowPerturbation._refresh_for`, which
    supplies the configured pair.
    """
    return int(force_ticks) // refresh_ticks(parent_step_ticks,
                                             parent_dt_seconds)


def draw_unit_pattern(seed: int, grid_id: int, face: str,
                      refresh: int, nz: int, n_blocks: int) -> np.ndarray:
    """One held unit draw: (nz, n_blocks) float64 uniform in [-1, 1).

    Counter-based and fully keyed: Philox seeded by SeedSequence
    entropy (seed, grid_id, face code, refresh index), consumed in a
    fixed (level, block) order, so each (domain, face, cell-block,
    refresh) owns a fixed counter position.  Identical on every
    platform NumPy guarantees stream stability for.
    """
    if seed < 0 or refresh < 0:
        raise ValueError("seed and refresh index must be non-negative")
    sequence = np.random.SeedSequence(
        entropy=(int(seed), int(grid_id), FACE_CODES[face], int(refresh)))
    generator = np.random.Generator(np.random.Philox(sequence))
    return generator.uniform(-1.0, 1.0, size=(int(nz), int(n_blocks)))


def expand_blocks(pattern: np.ndarray, face_length: int) -> np.ndarray:
    """Spread per-block draws onto per-cell columns along the face."""
    if pattern.ndim != 2:
        raise ValueError("pattern must be (nz, n_blocks)")
    needed = -(-int(face_length) // BLOCK_CELLS)
    if pattern.shape[1] != needed:
        raise ValueError(
            f"pattern has {pattern.shape[1]} blocks, face of "
            f"{face_length} cells needs {needed}")
    return np.repeat(pattern, BLOCK_CELLS, axis=1)[:, :int(face_length)]


def perturbed_level_count(z_half_agl: np.ndarray, z_i: float) -> int:
    """Levels (contiguous from the surface) inside the pinned extent."""
    if not math.isfinite(z_i) or z_i <= 0.0:
        return 0
    limit = VERTICAL_FRACTION * float(z_i)
    below = np.asarray(z_half_agl, dtype=np.float64) < limit
    count = 0
    for flag in below:
        if not flag:
            break
        count += 1
    return count


def select_faces(inward_means: dict[str, float], mode: str) -> list[str]:
    """flow_dep_bdy face selection from signed inward-normal means.

    ``inward_means[face]`` is positive where the boundary-normal wind
    enters the domain.  ``"inflow"`` takes strictly-entering faces,
    ``"outflow"`` (the AC-P3.4 mutation control) the strictly-leaving
    ones; an exactly-zero mean face belongs to neither.
    """
    if mode not in _FACE_MODES:
        raise ValueError(f"faces mode must be one of {_FACE_MODES}")
    wanted = (lambda value: value > 0.0) if mode == "inflow" \
        else (lambda value: value < 0.0)
    return [face for face in ("west", "east", "south", "north")
            if wanted(inward_means[face])]


def face_amplitude(inward_mean: float, amplitude_scale: float) -> float:
    """The pinned Eckert-number amplitude for one face, in Kelvin."""
    speed = abs(float(inward_mean))
    return float(amplitude_scale) * speed * speed / (ECKERT * c.CP)


def add_coupled_theta(fields, face: str, delta_theta, state, *,
                      spec_zone: int, relax_zone: int) -> None:
    """Add one face's Kelvin increment to the coupled theta value table.

    ``delta_theta`` is (nz, face_length) float32 Kelvin (already
    amplitude-scaled and zeroed above the perturbed depth).  The write
    touches exactly the relax-target rows ``[spec_zone, relax_zone)``
    of the theta VALUE table; tendencies and all other tables are
    untouched.  All arithmetic is float32 elementwise on device.
    """
    import cupy as cp

    theta_value = fields["theta"][face][0]
    mu_value = fields["mu"][face][0][0]      # (ny, W) x-sides / (W, nx)
    c1h = state.c1h
    c2h = state.c2h
    rows = slice(int(spec_zone), int(relax_zone))
    delta = cp.asarray(np.ascontiguousarray(delta_theta, dtype=np.float32))
    if face in ("west", "east"):
        width = int(theta_value.shape[-1])
        columns = state.mub2d[:, :width] if face == "west" \
            else state.mub2d[:, ::-1][:, :width]
        mass = columns + mu_value                       # (ny, W)
        ch = c1h[:, None, None] * mass[None] + c2h[:, None, None]
        theta_value[:, :, rows] += ch[:, :, rows] * delta[:, :, None]
    else:
        width = int(theta_value.shape[1])
        columns = state.mub2d[:width, :] if face == "south" \
            else state.mub2d[::-1, :][:width, :]
        mass = columns + mu_value                       # (W, nx)
        ch = c1h[:, None, None] * mass[None] + c2h[:, None, None]
        theta_value[:, rows, :] += ch[:, rows, :] * delta[:, None, :]


class InflowPerturbation:
    """Per-coupler generator state: pinned geometry plus the config keys."""

    def __init__(self, child_node):
        run = child_node.cfg.run
        self.grid_id = int(child_node.cfg.grid_id)
        self.seed = int(run.inflow_perturbation_seed)
        self.amplitude_scale = float(run.inflow_perturbation_amplitude_scale)
        self.faces_mode = str(run.inflow_perturbation_faces)
        self.nz = int(run.nz)
        self._face_length = {"west": int(run.ny), "east": int(run.ny),
                             "south": int(run.nx), "north": int(run.nx)}
        self._n_blocks = {
            face: -(-length // BLOCK_CELLS)
            for face, length in self._face_length.items()}
        ratio = int(child_node.cfg.parent_grid_ratio)
        self._footprint = (
            int(child_node.cfg.j_parent_start) - 1,
            int(child_node.cfg.i_parent_start) - 1,
            max(1, int(run.ny) // ratio),
            max(1, int(run.nx) // ratio))
        self._z_half_agl: np.ndarray | None = None
        #: The draw-holding bucket in parent ticks, latched from the
        #: parent's CONFIGURED step on the first force (see
        #: :meth:`_refresh_for`), and the last index it produced.
        self._refresh_ticks: int | None = None
        self._last_refresh: int | None = None
        #: What was injected, per face: the last force's row and the
        #: run totals.  Read through :meth:`injection_receipt`.
        self.last_injection: dict[str, dict] = {}
        self.injection_totals: dict[str, dict] = {}

    # -- deterministic per-force ingredients --------------------------------

    def _base_half_heights(self, state) -> np.ndarray:
        """Domain-mean base-state half-level height AGL, cached (static)."""
        if self._z_half_agl is None:
            import cupy as cp

            phb = state.phb
            if getattr(phb, "ndim", 1) == 3:
                full = cp.asnumpy(phb.mean(axis=(1, 2), dtype=cp.float64))
            else:
                full = cp.asnumpy(cp.asarray(phb, dtype=cp.float64))
            full = full / c.G
            self._z_half_agl = (0.5 * (full[:-1] + full[1:]) - full[0])
        return self._z_half_agl

    def _parent_footprint_pblh(self, parent_node) -> float:
        import cupy as cp

        driver = parent_node.state.physics
        fields = getattr(driver, "fields", None) if driver is not None \
            else None
        if not fields or "pblh" not in fields:
            raise RuntimeError(
                "inflow_perturbation reads the parent's PBLH diagnostic "
                "and the parent physics driver does not carry one; the "
                "coupler-construction guard should have refused this "
                "configuration")
        j0, i0, nj, ni = self._footprint
        window = fields["pblh"][j0:j0 + nj, i0:i0 + ni]
        return float(window.mean(dtype=cp.float64))

    def _inward_means(self, state, n_levels: int) -> dict[str, float]:
        import cupy as cp

        levels = slice(0, int(n_levels))
        u, v = state.u, state.v
        return {
            "west": float(u[levels, :, 0].mean(dtype=cp.float64)),
            "east": -float(u[levels, :, -1].mean(dtype=cp.float64)),
            "south": float(v[levels, 0, :].mean(dtype=cp.float64)),
            "north": -float(v[levels, -1, :].mean(dtype=cp.float64)),
        }

    def _refresh_for(self, parent) -> int:
        """This force's draw-holding index, cut from the parent's clock.

        THE BUCKET COMES FROM THE CONFIGURED STEP, THE INDEX FROM THE
        LIVE TICK COUNT, and that split is the whole contract.

        ``refresh`` is not a per-step physical rate.  ``dtbc``, the nest
        boundary-interpolation weight and the Davies relaxation are, and
        399d95a86 correctly moved all three from ``clock.spec.*`` to the
        live ``clock.*`` -- leaving those on the configured value under
        an adaptive clock forces the boundaries on a different clock
        from the interior.  This one is a QUANTIZER of absolute model
        time into ``REFRESH_SECONDS`` buckets, and the same sweep took
        it along: ``clock.ticks`` stayed absolute while the divisor
        started breathing, and ``floor(T / D(t))`` with ``D`` varying is
        neither monotone nor onto.  Measured on a parent whose dt
        breathes 4-8 s over 3 h (tests/test_inflow_perturbation.py):
        356 backward index transitions in 1,910 forces, largest jump 8
        indices, holds running 4.0 s to 148.7 s against the pinned
        100.0 s.  Reading ``clock.spec`` makes the bucket a fixed number
        of TICKS, so the index is monotone by construction, exactly
        reproducible at the same model time across a restart -- which is
        what :func:`refresh_index`'s docstring has always claimed -- and
        bit-identical on every fixed clock, where live == spec for the
        whole run (the only writers of the live pair anywhere are
        core/adaptive_clock.py:780,785 and io/restart.py:4643-4644, both
        adaptive-only).

        The refusal below is the un-regressable half: a comment cannot
        catch a future re-conversion, and this does, however it arrives.
        """
        clock = parent.clock
        if self._refresh_ticks is None:
            # Latched once, deliberately.  A leg boundary mints fresh
            # clock OBJECTS and carries the tick count onto them
            # (runtime._retarget_tree_schedule) without rebuilding this
            # coupler, so re-reading the spec every force would make the
            # bucket depend on which clock object is current.  The tick
            # lattice is stable across that surgery -- runtime.py's own
            # step_count arithmetic already depends on it -- so latching
            # and re-reading agree; latching says so structurally.
            spec = clock.spec
            self._refresh_ticks = refresh_ticks(
                int(spec.step_ticks), float(spec.dt_fp32))
        refresh = int(clock.ticks) // self._refresh_ticks
        if self._last_refresh is not None and refresh < self._last_refresh:
            raise ValueError(
                "inflow_perturbation refresh index went backward on "
                f"grid_id={self.grid_id}, {self._last_refresh} -> "
                f"{refresh} at parent tick {int(clock.ticks)}.  The draw "
                "is keyed on (seed, grid_id, face, refresh index) alone, "
                f"so index {refresh} names a pattern this face has "
                "already imprinted on its relax rows: re-emitting it "
                "drives the Davies zone toward a target it has already "
                "relaxed toward, which is a repeating boundary theta "
                "pattern instead of the decorrelating one the mechanism "
                f"exists to make, and voids the {REFRESH_SECONDS:.0f} s "
                "held-draw contract.  The bucket is cut from the "
                "parent's CONFIGURED step, which is a run constant, so "
                "a varying dt cannot cause this; it means the parent's "
                "tick count regressed, or a reader was re-pointed at "
                "the LIVE step (399d95a86 did exactly that and made the "
                "index non-monotone under an adaptive clock).")
        self._last_refresh = refresh
        return refresh

    # -- what the run actually injected -------------------------------------

    def _record_injection(self, face: str, *, theta_max: float,
                          inward_mean: float, n_levels: int,
                          refresh: int) -> None:
        """Book one face's injection for the run receipt.

        Bookkeeping only: it writes no table and reads no device memory,
        so the zero-amplitude arm (acceptance G2) still performs exactly
        the read-only classification it is defined as, and OFF (G1)
        never reaches here at all.
        """
        self.last_injection[face] = {
            "theta_max_k": float(theta_max),
            "inward_mean_ms": float(inward_mean),
            "face_cells": int(self._face_length[face]),
            "levels": int(n_levels),
            "refresh": int(refresh),
        }
        total = self.injection_totals.setdefault(
            face, {"forces": 0, "theta_max_k_sum": 0.0,
                   "theta_max_k_max": 0.0, "levels_max": 0})
        total["forces"] += 1
        total["theta_max_k_sum"] += float(theta_max)
        total["theta_max_k_max"] = max(total["theta_max_k_max"],
                                       float(theta_max))
        total["levels_max"] = max(total["levels_max"], int(n_levels))

    def injection_receipt(self) -> dict:
        """What this generator has put on the boundary, for the receipt.

        THE OUTFLOW CONTROL CANNOT BE READ WITHOUT THIS.  The shipped
        ``faces = "outflow"`` mutation (acceptance G5) selects the
        complementary faces and takes each one's amplitude from ITS OWN
        wind -- ``theta_max = scale * U^2 / (Ec * cp)`` -- so the two
        arms are not amplitude-matched by construction and a slower
        leaving face injects quadratically less.  A null result from an
        unmatched control is not evidence that seeding the leaving edges
        fails to shorten the fetch; it is evidence that barely anything
        was seeded.  Emitting the injected amplitude, the perturbed face
        length and the perturbed level count per face is what lets a
        reader tell those two apart -- and lets the report say the
        control was unmatched when it was, which is a finding rather
        than a failure.

        ``forces`` counts faces SELECTED, including any at exactly zero
        amplitude, so a receipt distinguishes "never selected" from
        "selected and injected nothing".
        """
        return {
            "faces_mode": self.faces_mode,
            "grid_id": self.grid_id,
            "seed": self.seed,
            "amplitude_scale": self.amplitude_scale,
            "eckert": ECKERT,
            "last_force": {face: dict(row)
                           for face, row in self.last_injection.items()},
            "totals": {face: dict(row)
                       for face, row in self.injection_totals.items()},
        }

    # -- the force-time hook ------------------------------------------------

    def apply_at_force(self, node, fields) -> None:
        """Perturb the freshly written theta tables for one FORCE."""
        parent = node.parent
        run = node.cfg.run
        z_i = self._parent_footprint_pblh(parent)
        n_levels = perturbed_level_count(
            self._base_half_heights(node.state), z_i)
        if n_levels == 0:
            return
        inward = self._inward_means(node.state, n_levels)
        refresh = self._refresh_for(parent)
        for face in select_faces(inward, self.faces_mode):
            theta_max = face_amplitude(inward[face], self.amplitude_scale)
            self._record_injection(
                face, theta_max=theta_max, inward_mean=inward[face],
                n_levels=n_levels, refresh=refresh)
            if theta_max == 0.0:
                continue
            pattern = draw_unit_pattern(
                self.seed, self.grid_id, face, refresh,
                self.nz, self._n_blocks[face])
            delta = expand_blocks(pattern, self._face_length[face])
            delta = (theta_max * delta).astype(np.float32)
            delta[n_levels:, :] = np.float32(0.0)
            add_coupled_theta(
                fields, face, delta, node.state,
                spec_zone=int(run.spec_zone),
                relax_zone=int(run.relax_zone))


def build_inflow_perturbation(child_node):
    """The coupler-construction entry: ``None`` unless configured ON.

    The ``None`` return is the whole OFF contract: the coupler holds no
    object and executes no instruction of this module (acceptance G1).
    A parent without a PBL scheme is refused here, at construction, not
    silently at the first force: the vertical extent is defined by the
    parent's PBLH diagnostic and a PBL-off parent has none.  Since
    2026-09-03 ``woof.experiment.build_experiment`` refuses the same
    pairing at the config load, so a TOML-declared tree learns before it
    is prepared; this stays because a spawned or relocated nest is built
    outside that path.  Both read
    :func:`parent_pbl_refusal`, so there is one sentence, not two.
    """
    run = child_node.cfg.run
    if not run.inflow_perturbation:
        return None
    parent = child_node.parent
    refusal = parent_pbl_refusal(
        None if parent is None else parent.cfg.run.bl_pbl_physics)
    if refusal is not None:
        raise ValueError(refusal)
    return InflowPerturbation(child_node)


__all__ = [
    "BLOCK_CELLS", "ECKERT", "FACE_CODES", "PARENT_PBL_REFUSAL_MARKER",
    "REFRESH_SECONDS", "VERTICAL_FRACTION", "InflowPerturbation",
    "add_coupled_theta", "build_inflow_perturbation", "draw_unit_pattern",
    "expand_blocks", "face_amplitude", "parent_pbl_refusal",
    "perturbed_level_count", "refresh_index", "refresh_ticks",
]
