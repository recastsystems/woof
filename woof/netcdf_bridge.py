"""The NetCDF read path: Rust decode, Python interpretation.

Every NetCDF meteorological source woof reads is decoded by the
``rw_netcdf`` bridge (``tools/rustwx/crates/rw-netcdf``, built on the
vendored pure-Rust ``netcrust`` facade).  Python resolves selectors,
checks contracts and assembles frames; it does not decode the file.

This module used to have no reason to exist: :mod:`woof.mapped_source`
opened arbitrary third-party NetCDF with ``netCDF4.Dataset`` -- the C
library -- while ``netcrust`` sat vendored in the same tree, already
driven by ``rw_wrfbatch``, ``rw_sat`` and ``rw_fieldcmp`` for exactly
this job.  The GRIB half of the very same module had always shelled out
to ``grib2_inventory`` / ``grib2_dump``.  The two formats now go through
one idiom.

The surface below is deliberately shaped like the small part of the
``netCDF4`` API that the mapping decoder actually used -- ``.variables``,
``.dimensions``, attribute access, ``variable[:]`` -- so the selector and
contract logic above it is untouched.  What changed underneath is who
reads the bytes.

There is NO fallback to a Python NetCDF library.  A missing bridge is a
named refusal (:class:`NetcdfBridgeMissing`) carrying the exact command
to fix it, because a silent degradation into a second decoder is how two
readers drift apart without anyone noticing.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from concurrent.futures import Future
from typing import Mapping

import numpy as np

from woof.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir,
                           legacy_bridge_candidates, accept_resolved,
                           executable_name, packaged_bridge_dir)

#: Environment variable naming a prebuilt ``rw_netcdf``.
NETCDF_ENV = "WOOF_RW_NETCDF"

#: Logical name of the bridge, and so its filename stem.
NETCDF_NAME = "rw_netcdf"

#: Schemas this module speaks.  A bridge answering a different schema is
#: refused rather than parsed hopefully.
INVENTORY_SCHEMA = "gpuwm-rw-netcdf-inventory-v1"
DUMP_SCHEMA = "gpuwm-rw-netcdf-dump-v1"

_TIMEOUT_S = 900


class NetcdfBridgeMissing(RuntimeError):
    """The Rust NetCDF decoder is not installed, and nothing replaces it."""


class NetcdfDecodeError(RuntimeError):
    """The Rust NetCDF decoder refused a file or a variable."""


class NetcdfInputError(NetcdfDecodeError):
    """The decoder returned its controlled input-refusal status, not a crash.

    ``rw_netcdf`` exits 2 when it refuses what it was given; a crash, a
    timeout or a reader that answers nonsense exits otherwise.  Only the
    first says anything about the input: a caller that quarantines a file
    it cannot read must never quarantine one the decoder merely failed on.
    """


class NetcdfFileError(NetcdfInputError):
    """The decoder ran and refused the file at ``path``: its bytes, not the decoder.

    A subclass, so every ``except NetcdfDecodeError`` and every
    ``except NetcdfInputError`` still catches it.  It exists so a caller
    whose refusal tells the reader to restore or regenerate a file can
    tell that file's failure from the decoder's own: a missing, stale or
    incompatible ``rw_netcdf`` is :class:`NetcdfBridgeMissing`, a plain
    :class:`NetcdfDecodeError` or a ``FileNotFoundError`` naming the
    override, and its remedy is the decoder's.  Blaming a healthy parent
    for a stale decoder sent its reader to regenerate a forecast that was
    never damaged.
    """

    def __init__(self, message: str, *, path: Path | str,
                 reason: str | None = None):
        super().__init__(message)
        self.path = Path(path)
        #: The decoder's own words about the file, without the command
        #: that ran or the decoder's name in front: what a refusal that
        #: already names the file quotes.
        self.reason = reason if reason is not None else message


def _crate_dir() -> Path:
    return Path(__file__).resolve().parent.parent / RUSTWX_CRATE_RELATIVE


def netcdf_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for ``rw_netcdf``, best first.

    The same ladder every other bridge uses -- see :mod:`woof.bridges`.
    """

    filename = executable_name(NETCDF_NAME)
    candidates: list[Path] = []
    override = os.environ.get(NETCDF_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    candidates.extend((
        _crate_dir() / "target" / "release" / filename,
        _crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    return tuple(candidates)


def find_netcdf_bin() -> Path | None:
    """First candidate that can supply numeric and character records, or None.

    An environment override naming a missing file is a hard error:
    explicit configuration must fail loudly, not fall through.
    """

    override = os.environ.get(NETCDF_ENV)
    incompatible = []
    for candidate in netcdf_candidates():
        if candidate.is_file():
            resolved = accept_resolved(candidate.resolve())
            from woof.bridges import bridge_abi_matches
            compatible, _ = bridge_abi_matches(NETCDF_NAME, resolved)
            if not compatible:
                incompatible.append(resolved)
                if override and candidate == Path(override):
                    break
                continue
            return resolved
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{NETCDF_ENV} names a missing file: {candidate}.  Point it "
                f"at a built {NETCDF_NAME} binary, or unset {NETCDF_ENV} to "
                f"use the resolution ladder.")
    if incompatible:
        raise NetcdfDecodeError(
            f"The NetCDF reader at {incompatible[0]} cannot supply the numeric "
            "and character records, layer water conversion and source-layer soil recovery this input route requires, including "
            "WRF Times. Replace it with this release's reader.\n"
            + netcdf_remedy())
    return None


def netcdf_remedy() -> str:
    """The remedy for a missing decoder, true for THIS install."""

    return artifact_remedy(
        env_var=NETCDF_ENV, filename=executable_name(NETCDF_NAME),
        subject="the NetCDF decoder", crate_relative=RUSTWX_CRATE_RELATIVE,
        artifact=NETCDF_NAME)


def resolve_netcdf_bin() -> Path:
    """THE NetCDF decoder, or a refusal naming what to run.

    Never returns a Python decoder.  Reading a NetCDF meteorological
    source is decode work, and this project does decode in Rust; an
    install that cannot do it must say so at the front door rather than
    quietly producing numbers from a second implementation.
    """

    found = find_netcdf_bin()
    if found is not None:
        return found
    # `artifact_remedy` is already install-aware -- it prints the clone
    # for a wheel and the in-tree build for a checkout -- so nothing is
    # appended to it here.  A second, unconditional "or, from a
    # checkout" line was exactly the kind of remedy that names a
    # directory the reader does not have.
    raise NetcdfBridgeMissing(
        f"{NETCDF_NAME} is not installed, so this NetCDF source cannot be "
        f"decoded.  woof decodes NetCDF in Rust and has no Python "
        f"fallback.\n\n{netcdf_remedy()}")


def _run(arguments: list[str], *, what: str,
         file: Path | None = None) -> subprocess.CompletedProcess:
    """Run the decoder; a refusal is ``NetcdfFileError`` when ``file`` is the input it read."""

    completed = subprocess.run(
        arguments, capture_output=True, text=True, timeout=_TIMEOUT_S)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        if completed.returncode != 2:
            # A crash, a kill or any status other than the decoder's own
            # refusal says nothing about the file it was reading.
            raise NetcdfDecodeError(f"{what}: {detail}")
        if file is not None:
            reason = detail.removeprefix(f"{NETCDF_NAME}: ").strip()
            raise NetcdfFileError(f"{what}: {detail}", path=file,
                                  reason=reason or detail)
        raise NetcdfInputError(f"{what}: {detail}")
    return completed


class Dimension:
    """One NetCDF dimension.

    ``len(dimension)`` rather than only ``.size`` because that is how
    callers ask -- ``len(dataset.dimensions[name])`` is the netCDF4
    spelling this replaces, and a bare integer here would make that a
    ``TypeError`` at the one place a soil-layer count is checked.
    """

    __slots__ = ("name", "size", "_unlimited")

    def __init__(self, name: str, size: int, *, unlimited: bool = False):
        self.name = name
        self.size = int(size)
        self._unlimited = bool(unlimited)

    def __len__(self) -> int:
        return self.size

    def __int__(self) -> int:
        return self.size

    def isunlimited(self) -> bool:
        return self._unlimited

    def __repr__(self) -> str:
        return f"<Dimension {self.name!r}: size={self.size}>"


class Attributes(dict):
    """Variable attributes, with ``netCDF4``'s accessor spellings."""

    def ncattrs(self) -> list[str]:
        return list(self)

    def getncattr(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


#: netcrust's stored-type name -> the numpy dtype ``netCDF4`` would
#: report for the same variable.
#:
#: Callers ask ``np.dtype(variable.dtype).kind`` to tell a character
#: coordinate from a numeric one, so ``.dtype`` has to be a real numpy
#: dtype rather than the reader's debug spelling.  ``Promoted`` is what
#: a recovered HDF5 dimension scale reports: the raw index does not
#: expose a stored type, and the values arrive as f64.
_NUMPY_DTYPE = {
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "F32": np.float32, "F64": np.float64,
    "Char": np.dtype("S1"), "String": np.dtype(object),
    "Promoted": np.float64,
}


class Variable:
    """One NetCDF variable: metadata now, values on demand.

    Values are pulled through the bridge the first time they are asked
    for and then cached, so resolving a selector -- which only reads
    metadata -- never decodes a byte of payload.
    """

    def __init__(self, dataset: "Dataset", record: Mapping[str, object]):
        self._dataset = dataset
        self.name = str(record["name"])
        self.dimensions = tuple(str(d) for d in record["dimensions"])
        self.shape = tuple(int(s) for s in record["shape"])
        #: The reader's own spelling, kept for refusals that quote it.
        self.stored_dtype = str(record["dtype"])
        self._attributes = Attributes(record.get("attributes") or {})
        self._values: np.ndarray | None = None
        self._times: tuple[datetime, ...] | None = None
        #: Whether the next read skips CF decoding; see set_auto_mask.
        self._raw = False
        self._scale = True

    @property
    def dtype(self):
        """The numpy dtype ``netCDF4`` would report for this variable."""

        mapped = _NUMPY_DTYPE.get(self.stored_dtype)
        if mapped is None:
            raise NetcdfDecodeError(
                f"{self.name}: {NETCDF_NAME} reported stored type "
                f"{self.stored_dtype!r}, which woof.netcdf_bridge has no "
                f"numpy equivalent for; add it to _NUMPY_DTYPE rather than "
                f"guessing one")
        return np.dtype(mapped)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def is_character(self) -> bool:
        """True for a stored character/string variable."""

        return self.stored_dtype in ("Char", "String")

    # Attribute access: `getattr(variable, "standard_name", None)` and
    # `variable.units` are how the mapping decoder asks, matching netCDF4.
    def __getattr__(self, name: str):
        try:
            return self._attributes[name]
        except KeyError:
            raise AttributeError(
                f"{self.name!r} has no NetCDF attribute {name!r}") from None

    def ncattrs(self) -> list[str]:
        return self._attributes.ncattrs()

    def getncattr(self, name: str):
        return self._attributes.getncattr(name)

    #: CF attributes whose presence makes "masked" and "raw" different
    #: answers for the same variable.
    _CF_MASKING_ATTRS = ("_FillValue", "missing_value",
                         "scale_factor", "add_offset")

    def set_auto_mask(self, on: bool) -> None:
        """``netCDF4``'s masking switch, honoured by the Rust decoder.

        ``True`` (the default) is CF decoding: ``_FillValue`` and
        ``missing_value`` become NaN and ``scale_factor``/``add_offset``
        are applied, all in Rust.

        ``False`` asks for the unmasked stored values instead, and it is
        a real request rather than a formality: the gridded radar and
        satellite products write ``_FillValue = -9.99e30`` and read that
        sentinel back deliberately, so handing them NaN would change
        what their consistency checks see.  The bridge therefore
        re-decodes with ``--no-mask`` rather than refusing -- the switch
        is answered where the decoding happens, not worked around here.

        It turns off masking ONLY.  ``scale_factor``/``add_offset`` are
        still applied, which is what netCDF4-python does and what this
        class emulates; :meth:`set_auto_maskandscale` is the switch that
        turns both off.  Aliasing the two -- which this bridge did until
        the values were compared against netCDF4 on a packed ``i2``
        variable -- silently returned stored counts where the caller
        expected physical units, on a route (GOES ABI) whose products are
        packed in the wild.
        """

        self._set_policy(raw=not on, scale=self._scale if on else True)

    def set_auto_maskandscale(self, on: bool) -> None:
        """``netCDF4``'s combined switch: masking AND scaling together."""

        self._set_policy(raw=not on, scale=bool(on))

    def _set_policy(self, *, raw: bool, scale: bool) -> None:
        if raw != self._raw or scale != self._scale:
            self._raw = raw
            self._scale = scale
            # The cached values were decoded under the other policy.
            self._values = None
            self._times = None

    @property
    def attributes(self) -> Attributes:
        return self._attributes

    def _load(self) -> None:
        if self._values is not None:
            return
        held = (_beside_values(self._dataset.path, self.name)
                if (self._raw, self._scale) == _DEFAULT_POLICY else None)
        if held is not None:
            self._values, self._times = held
            return
        self._values, self._times = self._dataset._decode(
            self.name, raw=self._raw, scale=self._scale)

    def read_transformed(self, *, scale: float = 1.0, offset: float = 0.0, cache: bool = True):
        """Read with an explicit quantity-unit transform performed in Rust.

        CF masking and unpacking happen first. This does not alter the file,
        variable metadata, or the cached values in their original units.
        ``cache=False`` also leaves identity reads uncached, so a streaming
        caller can release each decoded float64 variable after conversion.
        """
        if not isinstance(cache, bool):
            raise TypeError("cache must be boolean")
        transform = (float(scale), float(offset))
        if not np.isfinite(transform).all() or transform[0] == 0.0:
            raise ValueError("unit transform needs a finite nonzero scale and finite offset")
        if transform == (1.0, 0.0) and cache:
            return self[...]
        values, _ = self._dataset._decode(
            self.name, raw=self._raw, scale=self._scale,
            unit_transform=None if transform == (1.0, 0.0) else transform)
        return values

    def read_window(self, *, i0: int, ni: int, j0: int, nj: int) -> np.ndarray:
        """``ni`` columns from ``i0`` and ``nj`` rows from ``j0`` of the
        last two dimensions, cut in Rust after CF decoding; never cached.

        The decoder refuses a window that leaves the variable, so a grid
        that does not fit is a named refusal rather than a smaller array.
        """
        window = tuple(int(value) for value in (i0, ni, j0, nj))
        if any(value < 0 for value in window) or window[1] == 0 or window[3] == 0:
            raise ValueError(f"{self.name}: invalid window {window}")
        values, _ = self._dataset._decode(
            self.name, raw=self._raw, scale=self._scale, window=window)
        return values

    def read_sample_window(self, *, i0: float, ni: int, j0: float, nj: int,
                           method: str, mask: str | None = None) -> np.ndarray:
        """Same-spacing nearest/bilinear sampling in Rust after CF decoding.

        ``bilinear-masked`` names a class field of the same file in
        ``mask`` (a land mask): each output point averages only the source
        points of its own class, as WPS interpolates land-only fields.
        """
        sample = {"i0": float(i0), "ni": int(ni), "j0": float(j0),
                  "nj": int(nj), "method": method}
        if (method == "bilinear-masked") != bool(mask):
            raise ValueError(f"{self.name}: bilinear-masked sampling and a mask go together")
        if mask:
            sample["mask"] = str(mask)
        if (method not in {"nearest", "bilinear", "bilinear-masked"}
                or not np.isfinite([sample["i0"], sample["j0"]]).all()
                or min(sample["i0"], sample["j0"]) < 0
                or min(sample["ni"], sample["nj"]) <= 0):
            raise ValueError(f"{self.name}: invalid sampled window {sample}")
        values, _ = self._dataset._decode(
            self.name, raw=self._raw, scale=self._scale, sample_window=sample)
        return values

    def __getitem__(self, item):
        self._load()
        assert self._values is not None
        values = self._values
        # A scalar variable is 0-dimensional, and `values[:]` on a 0-d
        # array is an IndexError -- but `variable[:]` is exactly how
        # netCDF4 spells "give me all of it", and callers use that
        # spelling uniformly.  ERA5 as delivered by the CDS carries a
        # scalar `number`, so this is a real shape, not a corner case.
        if values.ndim == 0 and isinstance(item, slice) and item == slice(None):
            return values[()]
        return values[item]

    def read_layer_water(self, thickness_variable: str):
        """Read declared layer water as volume fraction using native geometry.

        The file supplies both quantity units and a named thickness variable.
        This never changes the original variable or its cached values.
        """
        if not isinstance(thickness_variable, str) or not thickness_variable:
            raise ValueError("layer water requires a named thickness variable")
        receipt = {}
        values, _ = self._dataset._decode(
            self.name, water_layer_thickness=thickness_variable,
            conversion_receipt=receipt)
        return values, receipt

    def times(self) -> tuple[datetime, ...]:
        """Decoded CF instants, or a refusal.

        The calendar arithmetic happens in the bridge: turning
        ``hours since 1970-01-01`` and a number into an instant is
        decoding, not plumbing, and doing it here would be a second
        implementation of it.
        """

        self._load()
        if not self._times:
            raise NetcdfDecodeError(
                f"{self.name} has no CF reference-time units, so it cannot be "
                f"read as a time coordinate (units="
                f"{self._attributes.get('units')!r})")
        return self._times


class Dataset:
    """A NetCDF file, read through ``rw_netcdf``.

    Presents the read surface :mod:`woof.mapped_source` uses. Opening
    runs the metadata pass only.
    """

    def __init__(self, path: Path | str, *, executable: Path | None = None):
        self.path = Path(path)
        # A file that is not there is FileNotFoundError, exactly as
        # netCDF4 raises it.  Without this the bridge's own "cannot
        # open" refusal arrives as a decode error, and callers that
        # distinguish "no product yet" from "bad product" -- the DA
        # cycle does -- would stop being able to.
        if not self.path.is_file():
            raise FileNotFoundError(
                f"no such NetCDF file: {self.path}")
        self._executable = Path(executable) if executable else resolve_netcdf_bin()
        completed = _run(
            [os.fspath(self._executable), "inventory", os.fspath(self.path)],
            what=f"NetCDF inventory failed for {self.path}", file=self.path)
        try:
            document = json.loads(completed.stdout)
        except ValueError as error:
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} inventory for {self.path} is not JSON: {error}"
            ) from None
        if document.get("schema") != INVENTORY_SCHEMA:
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} answered inventory schema "
                f"{document.get('schema')!r}, expected {INVENTORY_SCHEMA!r}; "
                f"the installed bridge does not speak this release's "
                f"contract.\n\n{netcdf_remedy()}")
        self.format = str(document.get("format", "unknown"))
        self.metadata = dict(document.get("metadata") or {})
        #: Whether the reader proved every byte the header describes is
        #: in the file: True for a classic file it checked, False for a
        #: NetCDF-4 file (its own library checks at open), None from a
        #: reader older than the check, which says nothing either way.
        extent = document.get("extent_checked")
        self.extent_checked = None if extent is None else bool(extent)
        self.dimensions = {
            str(record["name"]): Dimension(
                str(record["name"]), int(record["len"]),
                unlimited=bool(record.get("unlimited")))
            for record in document.get("dimensions", [])
        }
        self.global_attributes = Attributes(
            document.get("global_attributes") or {})
        self.variables: dict[str, Variable] = {}
        for record in document.get("variables", []):
            variable = Variable(self, record)
            self.variables[variable.name] = variable

    # -- context manager, so `with Dataset(path) as dataset:` reads the
    # -- same as the netCDF4 call it replaces.
    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *_exception) -> bool:
        return False

    def close(self) -> None:
        return None

    def ncattrs(self) -> list[str]:
        return self.global_attributes.ncattrs()

    def getncattr(self, name: str):
        return self.global_attributes.getncattr(name)

    def set_auto_mask(self, on: bool) -> None:
        """Dataset-wide masking switch, applied to every variable.

        Same contract as :meth:`Variable.set_auto_mask`: masking on is
        what already happened, masking off is a no-op for variables that
        declare no CF masking attributes and a named refusal for any
        that do.
        """

        for variable in self.variables.values():
            variable.set_auto_mask(on)

    def set_auto_maskandscale(self, on: bool) -> None:
        self.set_auto_mask(on)

    def filepath(self) -> str:
        """The file this dataset reads, as ``netCDF4`` spells it.

        Callers put it in refusals -- "<path> is missing required field
        X" -- so a reader diagnosing a bad tape is told which tape.
        """

        return os.fspath(self.path)

    def __getattr__(self, name: str):
        """Global attributes as attributes: ``dataset.HYBRID_OPT``.

        The convention ``netCDF4`` established, and the one callers use
        through ``getattr(dataset, "ETAC", 0.2)``.  Only reached when
        normal lookup fails, so it can never shadow a real member.
        """

        try:
            return self.__dict__["global_attributes"][name]
        except KeyError:
            raise AttributeError(
                f"{self.path} has no global NetCDF attribute {name!r}"
            ) from None

    def _decode(self, name: str, *, raw: bool = False, scale: bool = True,
                unit_transform: tuple[float, float] | None = None,
                water_layer_thickness: str | None = None,
                conversion_receipt: dict | None = None,
                window: tuple[int, int, int, int] | None = None,
                sample_window: dict | None = None) -> tuple[np.ndarray, tuple[datetime, ...]]:
        """Decode one variable through the bridge.

        ``raw`` turns masking off; ``scale`` says whether
        ``scale_factor``/``add_offset`` still apply while it is off.  The
        two are separate for the same reason netCDF4-python keeps them
        separate -- see :meth:`Variable.set_auto_mask`.
        """

        with tempfile.TemporaryDirectory(prefix="gpuwm-netcdf-") as temporary:
            out = Path(temporary)
            command = [os.fspath(self._executable), "dump"]
            if raw:
                command.append("--raw" if not scale else "--no-mask")
            if unit_transform is not None:
                command.extend((f"--unit-scale={unit_transform[0]:.17g}",
                                f"--unit-offset={unit_transform[1]:.17g}"))
            if water_layer_thickness is not None:
                command.append(f"--water-layer-thickness={water_layer_thickness}")
            if window is not None:
                command.append("--window={}:{},{}:{}".format(*window))
            if sample_window is not None:
                if window is not None:
                    raise ValueError("exact and sampled windows cannot be combined")
                command += ["--sample-window={i0:.17g}:{ni},{j0:.17g}:{nj}".format(**sample_window),
                            f"--sample-method={sample_window['method']}"]
                if sample_window.get("mask"):
                    command.append(f"--sample-mask={sample_window['mask']}")
            names = [name] if isinstance(name, str) else list(name)
            command += [os.fspath(self.path), os.fspath(out), *names]
            _run(command,
                 what=f"NetCDF decode failed for {', '.join(names)} in {self.path}",
                 file=self.path)
            document = json.loads((out / "metadata.json").read_text("utf-8"))
            if document.get("schema") != DUMP_SCHEMA:
                raise NetcdfDecodeError(
                    f"{NETCDF_NAME} answered dump schema "
                    f"{document.get('schema')!r}, expected {DUMP_SCHEMA!r}")
            records = document.get("variables") or []
            if len(records) != len(names):
                raise NetcdfDecodeError(
                    f"{NETCDF_NAME} dumped {len(records)} variables for "
                    f"{names}; expected exactly {len(names)}")
            if isinstance(name, str):
                return self._decoded_record(
                    out, name, records[0], raw=raw,
                    unit_transform=unit_transform,
                    water_layer_thickness=water_layer_thickness,
                    conversion_receipt=conversion_receipt, window=window,
                    sample_window=sample_window)
            by_name = {str(record.get("name")): record for record in records}
            if set(by_name) != set(names):
                raise NetcdfDecodeError(
                    f"{NETCDF_NAME} dumped {sorted(by_name)} for {sorted(names)}")
            return {one: self._decoded_record(
                        out, one, by_name[one], raw=raw,
                        unit_transform=unit_transform,
                        water_layer_thickness=water_layer_thickness,
                        conversion_receipt=conversion_receipt, window=window,
                        sample_window=sample_window)
                    for one in names}

    def _decoded_record(self, out: Path, name: str, record, *, raw: bool,
                        unit_transform, water_layer_thickness,
                        conversion_receipt, window, sample_window):
        """``(values, times)`` of one dumped record, every check applied."""

        if water_layer_thickness is not None:
            conversion = record.get("water_layer_conversion")
            if (not isinstance(conversion, dict)
                    or conversion.get("thickness_variable") != water_layer_thickness
                    or conversion.get("target_units") != "m3 m-3"):
                raise NetcdfDecodeError(
                    f"{NETCDF_NAME} did not acknowledge layer water conversion; rebuild the reader")
            if conversion_receipt is not None:
                conversion_receipt.update(conversion)
        if unit_transform is not None and record.get("unit_transform") != list(unit_transform):
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} did not acknowledge the requested unit transform; rebuild the reader")
        if window is not None and record.get("window") != list(window):
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} did not acknowledge the requested window; rebuild the reader")
        if sample_window is not None and (
                record.get("sample_window") != {k: v for k, v in sample_window.items() if k != "mask"}
                or record.get("sample_mask") != sample_window.get("mask")):
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} did not acknowledge the requested sampled window; rebuild the reader")
        dtype = record.get("dtype", "<f8")
        expected_dtype = "|S1" if self.variables[name].is_character else "<f8"
        if dtype != expected_dtype:
            raise NetcdfDecodeError(
                f"{name}: decoded dtype {dtype!r}, expected {expected_dtype!r}")
        values = np.fromfile(out / str(record["filename"]), dtype=dtype)
        shape = tuple(int(s) for s in record["shape"])
        expected = int(np.prod(shape)) if shape else 1
        if values.size != expected:
            raise NetcdfDecodeError(
                f"{name}: decoded {values.size} values but shape {shape} "
                f"needs {expected}")
        values = values.reshape(shape)
        if not raw:
            values = _mask_default_fill(
                self.variables[name], values, record.get("cf") or {},
                unit_transform=unit_transform,
                layer_water=water_layer_thickness is not None)
        times = tuple(
            _parse_instant(text, name) for text in (record.get("times") or ())
        )
        return values, times

    def stored_words(self, names) -> dict[str, np.ndarray]:
        """Read stored numeric or character words in one Rust process.

        Unlike promoted CF reads, this preserves integer bits and NaN payloads.
        Python maps the bridge output using its declared dtype and shape only.
        """
        names = list(dict.fromkeys(names))
        if not names:
            return {}
        with tempfile.TemporaryDirectory(prefix="gpuwm-netcdf-words-") as temporary:
            out = Path(temporary)
            _run([os.fspath(self._executable), "stored-words", os.fspath(self.path),
                  os.fspath(out), *names], what=f"stored-word read failed for {self.path}", file=self.path)
            document = json.loads((out / "metadata.json").read_text("utf-8"))
            if document.get("schema") != "gpuwm-rw-netcdf-stored-words-v1":
                raise NetcdfDecodeError("stored-word bridge schema mismatch; rebuild rw_netcdf")
            records = document.get("variables", [])
            if len(records) != len(names) or [r.get("name") for r in records] != names:
                raise NetcdfDecodeError("stored-word bridge returned different variables")
            arrays = {}
            for record in records:
                name = record["name"]
                variable = self.variables[name]
                dtype = np.dtype(record["dtype"])
                shape = tuple(record["shape"])
                expected_dtype = np.dtype("S1") if variable.is_character else variable.dtype
                if dtype != expected_dtype or shape != variable.shape:
                    raise NetcdfDecodeError(f"{name}: stored-word dtype or shape changed")
                filename = str(record["filename"])
                if Path(filename).name != filename:
                    raise NetcdfDecodeError("stored-word filename leaves output directory")
                data = np.fromfile(out / filename, dtype=dtype)
                if data.size != variable.size:
                    raise NetcdfDecodeError(f"{name}: stored-word extent changed")
                arrays[name] = data.reshape(shape)
            return arrays

    def prefetch(self, names) -> int:
        """Decode several variables in one ``rw_netcdf`` run and cache them.

        Each variable is decoded under its own current mask/scale policy,
        exactly as its first ``variable[...]`` would decode it; variables
        sharing a policy share one run.  The bridge starts a process per
        run, so a reader that is about to read a whole file pays one start
        instead of one per variable.  Returns the number of runs made.
        """

        groups: dict[tuple[bool, bool], list[str]] = {}
        for name in names:
            variable = self.variables[name]
            if variable._values is not None:
                continue
            groups.setdefault((variable._raw, variable._scale), []).append(name)
        runs = 0
        for (raw, scale), group in groups.items():
            if (raw, scale) == _DEFAULT_POLICY and _BESIDE:
                left = []
                for name in group:
                    held = _beside_values(self.path, name)
                    if held is None:
                        left.append(name)
                    else:
                        self.variables[name]._values, self.variables[name]._times = held
                group = left
                if not group:
                    continue
            decoded = self._decode(group, raw=raw, scale=scale)
            runs += 1
            for name in group:
                variable = self.variables[name]
                variable._values, variable._times = decoded[name]
        return runs


#: A variable's policy until a reader switches it: CF masking and scaling on (``(raw, scale)``).
_DEFAULT_POLICY = (False, True)
#: Whole-file decodes started beside other work (:func:`decode_beside`), by the file's state:
#: a Future of ``{name: (values, times)}`` (None when the decode failed or the file moved).
_BESIDE_LOCK = threading.Lock()
_BESIDE: dict[tuple, Future] = {}


def _file_state(path) -> tuple:
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def decode_beside(path: Path | str) -> None:
    """Decode every variable of ``path`` under the default policy on a host thread, for a reader that will.

    Breakage it prevents (single-core law, 2026-10-29: DA member legs on
    woof 2.9.9 spent ~26 s at ~1 core before the engine started): a cycling-restart leg's worker decoded
    its run's wrfinput (1.25 GB) and wrfbdy on its main thread only after the analysis door had finished,
    although neither depends on anything the door writes.  Started at the leg's start
    (``woof.leg_server``), the decode runs beside the door; a later default-policy read of the file in
    the same state (:meth:`Dataset.prefetch`, :meth:`Variable._load`) takes each variable's decoded
    values instead of decoding it again: the same decoder, flags and post-processing, so the same values.
    Each variable is handed out once.  A file whose state moved (device, inode, size, modification or
    change time) is never served; a decode that fails is forgotten and the reader decodes as before.
    :func:`forget_beside` drops what nobody took."""

    try:
        state = _file_state(path)
    except OSError:
        return
    with _BESIDE_LOCK:
        if state in _BESIDE:
            return
        future: Future = Future()
        _BESIDE[state] = future

    def run():
        try:
            dataset = Dataset(path)
            names = list(dataset.variables)
            decoded = dict(dataset._decode(names, raw=False, scale=True)) if names else {}
            if _file_state(path) != state:
                decoded = None
        except BaseException:  # noqa: BLE001 - the reader decodes the file itself and meets the error
            decoded = None
        future.set_result(decoded)

    threading.Thread(target=run, name="netcdf-decode-beside", daemon=True).start()


def _beside_values(path, name: str):
    """``(values, times)`` of ``name`` from a :func:`decode_beside` of ``path`` in its current state (waiting
    for it), handed out once; None otherwise."""

    if not _BESIDE:
        return None
    try:
        state = _file_state(path)
    except OSError:
        return None
    with _BESIDE_LOCK:
        future = _BESIDE.get(state)
    if future is None:
        return None
    decoded = future.result()
    if not decoded:
        return None
    with _BESIDE_LOCK:
        return decoded.pop(name, None)


def keep_only_beside(paths) -> None:
    """Forget every :func:`decode_beside` result except those of ``paths`` in their current state: a leg's
    start drops what an earlier leg kept (a failed or refused leg keeps its decodes for its retry)."""

    keep = set()
    for path in paths:
        try:
            keep.add(_file_state(path))
        except OSError:
            pass
    with _BESIDE_LOCK:
        for key in [key for key in _BESIDE if key not in keep]:
            del _BESIDE[key]


def forget_beside() -> None:
    """Drop every :func:`decode_beside` result nobody took (a leg that ended)."""

    with _BESIDE_LOCK:
        _BESIDE.clear()


#: Decoded float64 bytes one ``rw_netcdf`` run hands back to a reader that
#: reads a whole file (the WRF restart and wrfinput importers).  One
#: decoder process per variable cost 345 process starts on NCAR's 12 km
#: restart; a run per batch lets the decoder work on a batch's variables
#: concurrently.  The cap bounds the batch's temporary files and the
#: decoder's working set (NCAR's 2.5 km restart is ~70 GB as float64).
DECODE_BATCH_BYTES = 4 << 30


class BatchedDecode:
    """Decode a file's variables in read order, one decoder run per batch.

    ``ready(name)`` is called before a variable is first read; when the
    variable has not been decoded yet, it and the variables after it are
    decoded together (up to :data:`DECODE_BATCH_BYTES`) and cached
    on the dataset exactly as their first ``variable[...]`` would cache
    them, under each variable's own mask/scale policy.
    """

    def __init__(self, dataset, names, *, batch_bytes=DECODE_BATCH_BYTES):
        self._dataset = dataset
        self._names = list(names)
        self._position = {name: index for index, name in enumerate(self._names)}
        self._next = 0
        self._batch_bytes = int(batch_bytes)
        self.runs = 0

    def ready(self, name: str) -> None:
        index = self._position.get(name)
        if index is None or index < self._next:
            return
        if getattr(self._dataset, "prefetch", None) is None:
            # A reader surface with no batch decode reads each variable on
            # its first ``variable[...]``, as it always did.
            self._next = len(self._names)
            return
        batch, nbytes = [], 0
        for later in self._names[index:]:
            size = 8 * int(getattr(self._dataset.variables[later], "size", 0))
            if batch and nbytes + size > self._batch_bytes:
                break
            batch.append(later)
            nbytes += size
        self.runs += self._dataset.prefetch(batch)
        self._next = index + len(batch)


#: The NetCDF library's default fill per stored type (netcdf.h NC_FILL_*),
#: widened to the f64 the decoder hands back.  Byte and character types
#: have none here, as in netCDF4-python.
_NC_DEFAULT_FILL = {
    "I16": -32767.0, "I32": -2147483647.0,
    "F32": float(np.float32(9.969209968386869e36)),
    "F64": 9.969209968386869e36,
    "U16": 65535.0, "U32": 4294967295.0,
    "I64": -9223372036854775806.0, "U64": 18446744073709551614.0,
}


def _mask_default_fill(variable: "Variable", values: np.ndarray,
                       cf: Mapping[str, object], *,
                       unit_transform: tuple[float, float] | None = None,
                       layer_water: bool = False) -> np.ndarray:
    """Values the writer never set, missing when the reader predates that rule.

    An element written masked with no ``_FillValue`` declared is stored as
    the library's default fill (9.97e36 for a float): a FINITE number, so
    every finiteness check downstream accepted it as data.  ``rw_netcdf``
    masks it itself and says so with ``default_fill_rule``; a reader
    installed before that release does not, and for it the same rule is
    applied here so the answer does not depend on which reader is
    installed.

    That reader unpacked what it read before handing it on, so the fill is
    looked for as it was handed on: times ``scale_factor``, plus
    ``add_offset``, then through any explicit unit transform, each step
    the same f64 operation the reader performs, so the comparison is
    exact.  A packed short's -32767 used to pass here unseen as
    -32767 * scale + offset.  A layer water conversion is not one formula
    per value, so after it the fill cannot be recognised on the answer;
    that is left to a reader with the rule.
    """

    if cf.get("default_fill_rule") is not None or values.dtype.kind != "f":
        return values
    if layer_water or "_FillValue" in variable.attributes:
        return values
    fill = _NC_DEFAULT_FILL.get(variable.stored_dtype)
    if fill is None:
        return values
    fill = np.float64(fill)
    scale, offset = cf.get("scale_factor"), cf.get("add_offset")
    if scale is not None:
        fill = fill * np.float64(scale)
    if offset is not None:
        fill = fill + np.float64(offset)
    if unit_transform is not None:
        fill = (fill * np.float64(unit_transform[0])
                + np.float64(unit_transform[1]))
    hit = values == fill
    if hit.any():
        values = values.copy()
        values[hit] = np.nan
    return values


def recover_wrf_soil(wrfinput, met_em, authority) -> tuple[dict, dict]:
    """Run the native source-layer recovery and load its declared output planes."""
    import hashlib
    from woof.filesystem_paths import io_path

    executable = resolve_netcdf_bin()
    with tempfile.TemporaryDirectory(prefix="gpuwm-soil-") as temporary:
        root = Path(temporary)
        spec = root / "authority.json"
        spec.write_text(json.dumps(authority, allow_nan=False), encoding="utf-8")
        out = root / "recovered"
        _run([os.fspath(executable), "recover-wrf-soil", os.fspath(io_path(wrfinput)),
              os.fspath(io_path(met_em)), os.fspath(spec), os.fspath(out)],
             what=f"Source-layer soil recovery failed for {wrfinput}")
        document = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        if (document.get("schema") != "gpuwm-wrf-soil-recovery-v1"
                or not isinstance(document.get("receipt"), dict)):
            raise NetcdfDecodeError("Native soil recovery returned an incompatible result")
        values = {}
        field_identities = {}
        for record in document.get("variables", []):
            name = record.get("name")
            filename = record.get("filename")
            if (name not in ("SMOIS", "SH2O") or name in values
                    or filename != f"{name}.f64" or record.get("dtype") != "<f8"
                    or record.get("units") != "m3 m-3"):
                raise NetcdfDecodeError("Native soil recovery returned an invalid variable record")
            shape = record.get("shape")
            if (not isinstance(shape, list) or not shape
                    or any(type(size) is not int or size <= 0 for size in shape)):
                raise NetcdfDecodeError(f"Native soil recovery returned invalid {name} geometry")
            array = np.fromfile(out / filename, dtype="<f8")
            if array.size != int(np.prod(shape)) or not np.all(np.isfinite(array)):
                raise NetcdfDecodeError(f"Native soil recovery returned incomplete or nonfinite {name}")
            array = array.reshape(shape)
            dimensions = record.get("dimensions", [])
            if dimensions and dimensions[0] == "Time":
                if shape[0] != 1:
                    raise NetcdfDecodeError("Native soil recovery must return one initialization time")
                array = array[0]
            values[name] = array
            with (out / filename).open("rb") as stream:
                field_identities[name] = {
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    "shape": shape, "dtype": "<f8", "units": "m3 m-3"}
        if "SMOIS" not in values:
            raise NetcdfDecodeError("Native soil recovery omitted total soil water")
        return values, dict(document["receipt"], recovered_fields=field_identities)


def _parse_instant(text: str, label: str) -> datetime:
    """One RFC-3339 instant from the bridge, as a naive UTC datetime.

    The bridge emits UTC and only UTC (a reference time spelled with a
    UTC offset is converted to UTC inside the decoder before any value
    is decoded), so the trailing ``Z`` is dropped rather than
    interpreted -- and the rest of the module keeps working in the
    naive-UTC datetimes it always used.
    """

    body = str(text)
    if not body.endswith("Z"):
        raise NetcdfDecodeError(
            f"{label}: decoded time {body!r} is not UTC-stamped")
    try:
        return datetime.fromisoformat(body[:-1])
    except ValueError as error:
        raise NetcdfDecodeError(
            f"{label}: cannot read decoded time {body!r}: {error}") from None


def open_dataset(path: Path | str, *, executable: Path | None = None) -> Dataset:
    """Open ``path`` for reading through the Rust decoder."""

    return Dataset(path, executable=executable)


__all__ = [
    "Attributes", "Dataset", "Dimension", "NetcdfBridgeMissing",
    "NetcdfDecodeError", "NetcdfFileError", "NetcdfInputError",
    "NETCDF_ENV", "NETCDF_NAME", "Variable", "find_netcdf_bin",
    "netcdf_candidates", "netcdf_remedy", "open_dataset", "resolve_netcdf_bin",
]
