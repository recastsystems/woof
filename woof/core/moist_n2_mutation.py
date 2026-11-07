# woof/core/moist_n2_mutation.py
"""The moist-N2 mutation control: the saturated branch, forced off.

**This is instrument qualification, not a physics option.**  The LES
completion program spec §3.3 requires it in those words:

    a scratch build with the saturated BN2 switch forced to the dry branch
    (``woof/core/kernels/smag2d.cu``) must move the committed cloud-layer
    metrics; if it does not, **the instrument is rejected, not the engine
    passed**

and the oracle side's hand-off names the metrics it must move:
``cloud_frac``, ``n2_moist_frac`` and the cloud-layer metrics
(``docs/superpowers/receipts/les/wrf-moist-oracle-2026-08-04.md``).

An instrument that cannot fail cannot pass anything.  A moist verdict
built on an instrument that has never been shown to notice condensation is
a verdict about nothing, and this is how it gets shown.

------------------------------------------------------------------------
WHY A BUILD VARIANT AND NOT A CONFIG KNOB OR AN ENVIRONMENT VARIABLE
------------------------------------------------------------------------
Nothing in ``RunConfig`` may select broken physics: the config surface is
what a forecast is described by, and a mutation reachable from it is a
mutation that can be reached by accident.  Nor an environment variable --
**no environment variable in woof alters physics**, and the two that come
closest (``WOOF_NOAHMP_HOST_LEAVES``, ``WOOF_NOAHMP_STAGED_COLUMNS``)
select an execution route claimed bit-identical, not a different answer.

The established shape for an opt-in build-level mutation is the one
``woof/core/noahmp_vegeflux_gpu.py`` uses for its ``-DUSE_DEVICE_LIBM``
negative control: an explicit, defaulted-off Python switch; a ``#ifdef``
guard in the ``.cu``; a cache key that includes the variant; and a
kernel-manifest entry of its own.  This module is that shape, expressed
through the loader hook that already exists for it.

------------------------------------------------------------------------
THE CACHE TRAP THIS MODULE EXISTS TO AVOID
------------------------------------------------------------------------
``woof.core.kernels.load_module`` is ``lru_cache``d **on the module name
alone**, and ``get_kernel`` on ``(name, func)``.  Compiling a second
variant of ``smag2d.cu`` through them would silently hand back the FIRST,
UNMUTATED module -- a control arm that quietly runs the production kernel
and "proves" the mutation does nothing.  That failure is invisible in the
output, and it would falsify the instrument in the safe-looking direction.

So the mutant goes through ``get_kernel_int_defines``, whose cache key is
``(name, func, defines)``, and whose manifest key carries the define tier.
The two builds are therefore two entries, two compiled images, and two
cache slots, and neither can be served in place of the other.

------------------------------------------------------------------------
WHAT THE MUTATION DOES NOT TOUCH, DELIBERATELY
------------------------------------------------------------------------
The engagement diagnostic (``n2_moist_frac``, ``sat_frac``,
``cloud_frac``) is a HOST-side re-evaluation of WRF's predicate over the
state, not a readback of the branch the kernel took.  It is not mutated
and must not be: under the mutation it keeps reporting, accurately, what the
state the mutant produced looks like.  So the mutation moves those numbers
only through the trajectory -- a suppressed saturated branch gives a
different BN2, hence different K, hence different mixing, hence a
different cloud field -- which is exactly the causal chain the control is
supposed to exercise.  A mutation that moved the diagnostic directly would
prove only that the diagnostic reads its own switch.
"""
from __future__ import annotations

from contextlib import contextmanager

__all__ = ["MUTATION_DEFINE", "MUTATION_DEFINES", "MUTATION_TAG",
           "NO_MUTATION_TAG", "active", "receipt_tag", "forced_dry_branch",
           "calc_n2_kernel"]

#: The C preprocessor identifier guarded in ``smag2d.cu``.  Uppercase with
#: no leading digit because ``load_module_int_defines`` validates the
#: identifier and rejects anything else; value 1 because it also rejects
#: bools and any integer below 1, so "off" is expressed by not passing the
#: define at all rather than by passing zero.
MUTATION_DEFINE = "GPUWM_MUTATE_MOIST_N2_FORCE_DRY"
MUTATION_DEFINES: tuple[tuple[str, int], ...] = ((MUTATION_DEFINE, 1),)

#: What a receipt says about which build produced it.  A plain string on
#: BOTH arms, never ``None`` on one of them: a null would read as "not
#: recorded" and would let a mutant receipt merge into a scored draw set
#: unnoticed, whereas two different strings make the difference show up in
#: any configuration-identity check that includes the field.
MUTATION_TAG = "moist_n2_forced_dry"
NO_MUTATION_TAG = "none"

_ACTIVE = False


def active() -> bool:
    """True while the mutant kernel is selected in this process."""
    return _ACTIVE


def receipt_tag() -> str:
    """:data:`MUTATION_TAG` or :data:`NO_MUTATION_TAG`, for the receipt."""
    return MUTATION_TAG if _ACTIVE else NO_MUTATION_TAG


@contextmanager
def forced_dry_branch():
    """Select the mutant ``wrf_calc_n2`` for the duration of the block.

    Scoped rather than a global setter so a control arm cannot leak into a
    later scored run in the same process, and restored on the way out even
    if the integration raises.
    """
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = True
    try:
        yield
    finally:
        _ACTIVE = previous


def calc_n2_kernel(*, use_theta_m=0):
    """The ``wrf_calc_n2`` kernel for the build variant now selected.

    Both branches go through a cache whose key discriminates them, so the
    production kernel and the mutant are distinct compiled images for the
    whole process lifetime.
    """
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    defines = MUTATION_DEFINES if _ACTIVE else ()
    if use_theta_m:
        defines += (("GPUWM_USE_THETA_M", 1),)
    if defines:
        return get_kernel_int_defines("smag2d", "wrf_calc_n2", defines)
    return get_kernel("smag2d", "wrf_calc_n2")
