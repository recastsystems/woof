"""Noah-MP local-frame pricing: per-platform recordings, the card's own
row when it has one, the ceiling over the recorded platforms when it does
not, and the basis stated beside the number either way.

The five composed Noah-MP translation units (``noahmp_leaves.cu`` prepended
to a fragment that borrows its r_pow/r_exp/r_log) never compile alone, so
the standalone ``*.cu`` census in :mod:`woof.core.kernel_frame_recordings`
cannot reach them and ``sf_surface_physics = 4`` was refused a memory price
for as long as that census was the only source of frames.  The frames the
model launches are the frames of the units
:mod:`woof.core.noahmp_kernel_sources` composes, and those are what
:data:`woof.core.kernel_frame_recordings.NOAHMP_COMPOSED_FRAME_RECORDINGS`
records -- one row per compile platform, read on that platform.

Three rules:

* The frame is a compile attribute of (target architecture, NVRTC build).
  A card whose compile platform has a row is priced from that row, and
  the basis says the frames were MEASURED on its platform.
* A card whose compile platform has no row -- a card whose platform was
  not read (an absent card, a declared ``--vram-gib`` card, this machine
  under ``GPUWM_NO_LOCAL_GPU``), or a card on a pair nobody has read --
  is priced the way every other kernel is priced on an unrecorded
  platform: from the element-wise CEILING over the recorded rows, never
  below any reading, and the basis says so in words a user can see
  ("priced from the ceiling over the recorded platforms ...; not measured
  on this card").  This is the rule since 2026-09-11: a refusal by name on
  every unrecorded card kept Noah-MP out of reach of the shipped desktop
  runtime, whose card and compiler nobody had read, while the standalone
  kernels beside it were priced from their ceiling without complaint.
* A row must describe the units in the tree.  Each row binds the
  ``identity_sha256`` of every unit it read; a source, option or export
  edit stops the row matching, the row is withdrawn from both the exact
  match and the ceiling, and the basis says the platform's row went
  stale.  With NO usable reading at all -- no row, or every row stale --
  the units are priced at the tree's assumed frame bound
  (:func:`woof.core.kernel_frame_recordings.assumed_frame_bound`), the
  widest frame any module has been recorded at, and the basis says the
  number is an assumed bound and names the tool that takes a reading.
  Nothing here refuses a run for a missing reading.
  ``tests/test_noahmp_frame_provenance.py`` turns a stale row into a red
  CPU test the day the sources move.

Everything the estimator calls here is CPU-only.  :func:`measure_live` is
the GPU half -- what ``tools/measure_noahmp_frames.py`` runs in a fresh
process to take a row -- and the only function that imports CuPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from woof.core import kernel_frame_recordings as _kfr
from woof.core.noahmp_kernel_sources import (
    NOAHMP_PRICING_MODULES, NOAHMP_TRANSLATION_UNITS, compile_runtime_unit,
    runtime_unit,
)

#: The command that takes a row for the card in the machine.  Named in
#: every basis sentence for an unrecorded platform and in the one refusal
#: that remains, so the way to a measured price is never left to
#: archaeology.
MEASURE_COMMAND = ("python tools/measure_noahmp_frames.py measure "
                   "--output <receipt.json>")

#: Where the row it prints belongs.
RECORDINGS_MODULE = "woof/core/kernel_frame_recordings.py"

#: How the basis sentence closes when the units were priced at the
#: assumed bound: the reading that would replace the assumption.
ASSUMED_BOUND_WAY_OUT = (
    f"take a reading with `{MEASURE_COMMAND}` on any card and add its row to "
    f"{RECORDINGS_MODULE}")


def unit_identities() -> dict[str, str]:
    """Pricing key -> ``identity_sha256`` of the unit as the tree composes
    it right now.  What a recording row must match to be used."""
    return {runtime_unit(name).key: runtime_unit(name).identity()["identity_sha256"]
            for name in NOAHMP_TRANSLATION_UNITS}


def platform_fingerprint(profile) -> dict[str, str] | None:
    """The two fingerprint keys a profile carries, or ``None``."""
    platform = getattr(profile, "compile_platform", None)
    if platform is None:
        return None
    capability, build = platform
    return {"device_compute_capability": str(capability),
            "nvrtc_build": str(build)}


def platform_label(recording: _kfr.ComposedUnitFrameRecording) -> str:
    """``sm_86/12.9.86``: the short form of a row's platform, the way the
    basis sentence lists it."""
    return f"sm_{recording.compute_capability}/{recording.nvrtc_build}"


def stale_units(recording: _kfr.ComposedUnitFrameRecording,
                current: Mapping[str, str] | None = None) -> list[str]:
    """Pricing keys whose unit the row did not read as the tree holds it
    now -- missing from the row, or read from a different source, option
    tuple or export set.  Empty means the row describes this tree."""
    current = unit_identities() if current is None else current
    return sorted(
        key for key in NOAHMP_PRICING_MODULES
        if key not in recording.frames
        or recording.unit_identity.get(key) != current[key])


def usable_recordings() -> tuple[_kfr.ComposedUnitFrameRecording, ...]:
    """The rows that describe the Noah-MP units in this tree, in table
    order.  A stale row is withdrawn from pricing entirely: it is a
    reading of some other source, and neither an exact match nor the
    ceiling may lean on it."""
    current = unit_identities()
    return tuple(row for row in _kfr.NOAHMP_COMPOSED_FRAME_RECORDINGS
                 if not stale_units(row, current))


def composed_frame_ceiling(
        recordings: Iterable[_kfr.ComposedUnitFrameRecording]) -> dict[str, int]:
    """The element-wise maximum over ``recordings``: what a Noah-MP unit
    is charged on a compile platform nobody has read.  Never below any
    reading, by construction, which is the direction a rail gate can
    survive."""
    ceiling: dict[str, int] = {}
    for recording in recordings:
        for key, frame in recording.frames.items():
            if frame > ceiling.get(key, -1):
                ceiling[key] = int(frame)
    return ceiling


@dataclass(frozen=True)
class NoahMPFrameBasis:
    """Where a Noah-MP local-frame price came from, in numbers and in words.

    ``recording`` is the row the frames were read from when the card's
    compile platform has a usable row (``measured`` is then True); with
    no such row the frames are the ceiling over ``ceiling_over`` and
    ``reason`` says why this card had no row of its own -- its platform
    was not read, or the pair has no reading, or its row went stale.
    :meth:`sentence` is what plan review prints beside the number.
    """

    frames: Mapping[str, int]
    platform: tuple[str, str] | None
    recording: _kfr.ComposedUnitFrameRecording | None
    ceiling_over: tuple[_kfr.ComposedUnitFrameRecording, ...]
    reason: str
    #: True when no usable row existed and the frames are the tree's
    #: assumed bound rather than a ceiling over readings.
    assumed_bound: bool = False

    @property
    def measured(self) -> bool:
        return self.recording is not None

    @property
    def recorded_platforms(self) -> tuple[str, ...]:
        return tuple(platform_label(row) for row in self.ceiling_over)

    def sentence(self) -> str:
        if self.recording is not None:
            row = self.recording
            return (f"Noah-MP local frames measured on this card's compile "
                    f"platform {platform_label(row)} ({row.device}, "
                    f"read {row.measured})")
        if self.assumed_bound:
            frame = max(self.frames.values(), default=0)
            return (f"Noah-MP local frames priced at the assumed bound "
                    f"{frame} B per thread ({_kfr.ASSUMED_BOUND_PHRASE}): "
                    f"{self.reason}; to price from a reading instead, "
                    f"{ASSUMED_BOUND_WAY_OUT}")
        return (f"Noah-MP local frames priced from the ceiling over the "
                f"recorded platforms {', '.join(self.recorded_platforms)}; not "
                f"measured on this card ({self.reason}; a reading of this "
                f"card's own platform is `{MEASURE_COMMAND}` run on it)")


def frame_basis_for_profile(profile) -> NoahMPFrameBasis:
    """The Noah-MP frames for the card ``profile`` describes, and their
    basis.

    The card's own row when its compile platform was read and has a
    usable row; otherwise the ceiling over every usable row, with the
    reason this card had none.  With no usable row anywhere the frames
    are the tree's assumed bound, stated as one.  Never refuses.
    """
    current = unit_identities()
    rows = _kfr.NOAHMP_COMPOSED_FRAME_RECORDINGS
    usable = tuple(row for row in rows if not stale_units(row, current))
    if not usable:
        # No reading anywhere: priced at the assumed bound rather than
        # refused.  The bound is the widest frame any module in this tree
        # has been recorded at, so the reservation cannot be short, and
        # the sentence says the number is an assumption.
        if not rows:
            reason = "this tree holds no Noah-MP local-frame recording"
        else:
            stale = "; ".join(
                f"sm_{row.compute_capability} / NVRTC {row.nvrtc_build} "
                f"({row.box}): {', '.join(stale_units(row, current))}"
                for row in rows)
            reason = ("every Noah-MP recording here was read from different "
                      f"sources than the tree holds now ({stale})")
        bound = _kfr.assumed_frame_bound()
        return NoahMPFrameBasis(
            frames={key: bound for key in NOAHMP_PRICING_MODULES},
            platform=None, recording=None, ceiling_over=(), reason=reason,
            assumed_bound=True)
    fingerprint = platform_fingerprint(profile)
    platform: tuple[str, str] | None = None
    if fingerprint is None:
        # No card name here: the profile half of the basis sentence
        # already names the card, and on the absent-card route that is
        # the reference card, which the user never asked about.
        reason = ("its compile platform -- target architecture and NVRTC "
                  "build -- was not read, because the card is being priced "
                  "as one that is not in this machine or its toolchain did "
                  "not resolve when it was read")
    else:
        platform = (fingerprint["device_compute_capability"],
                    fingerprint["nvrtc_build"])
        recording = _kfr.noahmp_composed_recording_for(fingerprint)
        if recording is not None and recording in usable:
            return NoahMPFrameBasis(
                frames=dict(recording.frames), platform=platform,
                recording=recording, ceiling_over=usable, reason="")
        where = f"sm_{platform[0]} / NVRTC {platform[1]}"
        if recording is None:
            reason = f"no reading exists for its compile platform {where}"
        else:
            reason = (f"the row for its compile platform {where} (read "
                      f"{recording.measured}) no longer describes the "
                      f"Noah-MP units in this tree: "
                      f"{', '.join(stale_units(recording, current))} changed "
                      "source, options or exports since it was read, so the "
                      "row is withdrawn until the platform is re-read")
    return NoahMPFrameBasis(
        frames=composed_frame_ceiling(usable), platform=platform,
        recording=None, ceiling_over=usable, reason=reason)


def noahmp_frame_basis(modules: Iterable[str], profile) -> NoahMPFrameBasis | None:
    """The basis for the Noah-MP pricing keys in ``modules`` on
    ``profile``'s card, or ``None`` when the experiment selects none."""
    if not set(modules) & set(NOAHMP_PRICING_MODULES):
        return None
    return frame_basis_for_profile(profile)


def noahmp_frames(modules: Iterable[str], profile) -> dict[str, int]:
    """Frames for the Noah-MP pricing keys in ``modules`` on ``profile``'s
    card; empty when the experiment selects none of them."""
    selected = set(modules) & set(NOAHMP_PRICING_MODULES)
    if not selected:
        return {}
    basis = frame_basis_for_profile(profile)
    return {key: int(basis.frames[key]) for key in sorted(selected)}


# ---------------------------------------------------------------------------
# Taking a row (GPU).
# ---------------------------------------------------------------------------

#: The attributes recorded per exported kernel; ``local_size_bytes`` is the
#: one that is priced, the rest make a reading auditable.
ATTRIBUTES = ("local_size_bytes", "num_regs", "shared_size_bytes",
              "const_size_bytes", "max_threads_per_block", "ptx_version",
              "binary_version")


def measure_live(*, log=None) -> dict:
    """Compile every Noah-MP runtime unit through the production factory
    and read every export's function attributes.  Zero launches.

    Must run in a FRESH process: the first fatter-framed module a process
    loads raises ``cudaLimitStackSize`` for the life of the process, and
    the fresh default stack is part of the reading.  The kernel manifest
    being non-empty is the tell that something already compiled here.
    """
    import cupy as cp
    from woof.certify.compile_platform import (
        UNRESOLVED, compile_platform_fingerprint)
    from woof.certify.kernel_manifest import kernel_manifest

    if kernel_manifest():
        raise RuntimeError(
            "the Noah-MP frame measurement requires a fresh process: kernels "
            "are already compiled in this one, so its default stack limit "
            "may no longer be the fresh value a forecast starts from")
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError(
            "the Noah-MP frame measurement requires exactly one visible CUDA "
            "device; select it with CUDA_VISIBLE_DEVICES")
    fingerprint = compile_platform_fingerprint()
    for key in ("device_compute_capability", "nvrtc_build"):
        if fingerprint.get(key) in (None, "", UNRESOLVED):
            raise RuntimeError(
                f"the compile platform could not be resolved ({key} is "
                "unavailable); a reading with no platform cannot be a row")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    device = dict(
        name=name.decode() if isinstance(name, bytes) else str(name),
        multiprocessor_count=int(props["multiProcessorCount"]),
        max_threads_per_multiprocessor=int(props["maxThreadsPerMultiProcessor"]),
        default_stack_limit_bytes=(0 if cp.cuda.runtime.is_hip else int(cp.cuda.runtime.deviceGetLimit(0))))
    identities_before = unit_identities()
    frames: dict[str, int] = {}
    functions: dict[str, dict[str, dict[str, int]]] = {}
    for unit_name in NOAHMP_TRANSLATION_UNITS:
        unit = runtime_unit(unit_name)
        module = compile_runtime_unit(
            unit_name, module_key=f"noahmp_frame_measurement:{unit_name}",
            log_stream=log)
        rows = {}
        for symbol in unit.exports:
            # Attribute inspection only; this never launches the kernel.
            attributes = module.get_function(symbol).attributes
            rows[symbol] = {key: int(attributes[key]) for key in ATTRIBUTES}
        functions[unit.key] = rows
        frames[unit.key] = max(row["local_size_bytes"] for row in rows.values())
    if unit_identities() != identities_before:
        raise RuntimeError("the Noah-MP sources changed during the measurement")
    return {
        "method": ("compile_runtime_unit for every NOAHMP_TRANSLATION_UNITS "
                   "entry in a fresh process, then get_function(...)."
                   "attributes on every exported __global__; no launches"),
        "platform": fingerprint,
        "device": device,
        "frames": frames,
        "unit_identity": identities_before,
        "functions": functions,
    }


def render_row(measurement: Mapping, *, box: str, platform_family: str,
               measured: str) -> str:
    """The Python source of the row ``measurement`` becomes."""
    frames = measurement["frames"]
    identity = measurement["unit_identity"]
    lines = [
        "    ComposedUnitFrameRecording(",
        f"        box={box!r},",
        f"        device={measurement['device']['name']!r},",
        f"        compute_capability={measurement['platform']['device_compute_capability']!r},",
        f"        nvrtc_build={measurement['platform']['nvrtc_build']!r},",
        f"        platform_family={platform_family!r},",
        f"        measured={measured!r},",
        "        frames=MappingProxyType({",
    ]
    for key in sorted(frames):
        lines.append(f"            '{key}': {frames[key]},")
    lines.append("        }),")
    lines.append("        unit_identity=MappingProxyType({")
    for key in sorted(identity):
        lines.append(f"            '{key}':")
        lines.append(f"                '{identity[key]}',")
    lines.append("        }),")
    lines.append("    ),")
    return "\n".join(lines)


def compare_with_tree(measurement: Mapping) -> list[str]:
    """Differences between the live reading and the tree's row for its
    platform; empty means the row is exactly what this card compiles to."""
    recording = _kfr.noahmp_composed_recording_for(measurement["platform"])
    platform = measurement["platform"]
    if recording is None:
        return [f"no row for sm_{platform['device_compute_capability']} / "
                f"NVRTC {platform['nvrtc_build']} in {RECORDINGS_MODULE}"]
    problems = []
    for key in sorted(set(recording.frames) | set(measurement["frames"])):
        shipped = recording.frames.get(key)
        observed = measurement["frames"].get(key)
        if shipped != observed:
            problems.append(f"{key}: row {shipped} B, this card compiles {observed} B")
    for key in sorted(set(recording.unit_identity) | set(measurement["unit_identity"])):
        if recording.unit_identity.get(key) != measurement["unit_identity"].get(key):
            problems.append(f"{key}: the row was read from a different source "
                            "or option tuple than the tree holds")
    return problems


__all__ = [
    "ASSUMED_BOUND_WAY_OUT", "ATTRIBUTES", "MEASURE_COMMAND",
    "NoahMPFrameBasis",
    "RECORDINGS_MODULE", "compare_with_tree",
    "composed_frame_ceiling", "frame_basis_for_profile", "measure_live",
    "noahmp_frame_basis", "noahmp_frames", "platform_fingerprint",
    "platform_label", "render_row", "stale_units", "unit_identities",
    "usable_recordings",
]
