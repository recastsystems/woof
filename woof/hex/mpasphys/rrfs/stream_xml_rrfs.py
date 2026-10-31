"""RRFS fork stream XML, interval aliases, output_timelevels and file IDs in Python.

The native-free replacement for the four places the RRFS route still reached into
``libmpasphys_native_rrfs.so`` or a compiled helper for stream configuration:

- ezxml's attribute semantics: ``ezxml_attr`` returns the FIRST attribute of a given name, and
  the published GSL streams files repeat ``filename_interval`` (MPAS-Model-rrfs at 1d6e370e,
  MPAS-Model-rrfs/src/external/ezxml/ezxml.c:83-99,490-550);
- ``extract_stream_interval`` (MPAS-Model-rrfs/src/framework/xml_stream_parser.c:925-1023) and the default
  filename-interval rules (MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1150-1209,1495-1560);
- the fork's ``output_timelevels`` parser and interval lookup, all in RKIND=real32
  (MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6059-6606, timestamps
  MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:1451-1534);
- the stream file ID, GNU libc ``rand()`` after ``srand(seed)`` (MPAS-Model-rrfs/src/framework/random_id.c:20-46),
  reusing the v8.4.2 route's isolated generator.

``stream_tree`` keeps the contract of the native hook it replaces
(``driver_rrfs.stream_tree(path, library)``: an ElementTree whose stream elements carry the
first-value attributes and the resolved ``filename_interval``), so the suite table's
``streams_parser`` row selects either one by name. Python only; no compiled artifact.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["first_value_xml", "parse_xml", "extract_stream_interval", "stream_tree",
           "parse_timelevels", "timelevel_query", "native_file_id"]

_WS = frozenset(" \t\n\r\x0c")  # Rust u8::is_ascii_whitespace, the rule the r11 artifact used


def first_value_xml(xml: str) -> str:
    """Drop every repeated attribute after its first occurrence; MPAS-Model-rrfs/src/external/ezxml/ezxml.c:83-99 (ezxml_attr).

    A byte-for-byte port of the r11 artifact's configuration filter (woof b3a0b0db,
    rw_mpas_rrfs_clock.rs:101-156): comments, CDATA, processing instructions and closing tags
    pass unchanged; field bytes never pass through it. The result is strict XML.
    """
    out: list[str] = []
    n = len(xml)
    pos = 0
    while pos < n:
        start = xml.find("<", pos)
        if start < 0:
            out.append(xml[pos:])
            break
        out.append(xml[pos:start])
        special = None
        if xml.startswith("<!--", start):
            special = (4, "-->")
        elif xml.startswith("<![CDATA[", start):
            special = (9, "]]>")
        elif xml.startswith("<?", start):
            special = (2, "?>")
        if special is not None:
            offset, end = special
            found = xml.find(end, start + offset)
            if found < 0:
                raise ValueError("unterminated XML declaration")
            pos = found + len(end)
            out.append(xml[start:pos])
            continue
        stop, quote, bracket = start + 1, "", 0
        while stop < n:
            b = xml[stop]
            if quote:
                if b == quote:
                    quote = ""
            elif b in "'\"":
                quote = b
            elif b == "[":
                bracket += 1
            elif b == "]":
                bracket -= 1
            elif b == ">" and bracket == 0:
                break
            stop += 1
        if stop == n:
            raise ValueError("unterminated XML element")
        pos = stop + 1
        if xml[start + 1] in "/!":
            out.append(xml[start:pos])
            continue
        cursor = start + 1
        while cursor < stop and xml[cursor] not in _WS and xml[cursor] != "/":
            cursor += 1
        out.append(xml[start:cursor])
        names: set[str] = set()
        while cursor < stop:
            whitespace = cursor
            while cursor < stop and xml[cursor] in _WS:
                cursor += 1
            if cursor == stop or xml[cursor] == "/":
                out.append(xml[whitespace:pos])
                break
            key = cursor
            while cursor < stop and xml[cursor] not in _WS and xml[cursor] != "=":
                cursor += 1
            name = xml[key:cursor]
            while cursor < stop and xml[cursor] in _WS:
                cursor += 1
            if cursor == stop or xml[cursor] != "=":
                raise ValueError("XML attribute requires =")
            cursor += 1
            while cursor < stop and xml[cursor] in _WS:
                cursor += 1
            if cursor == stop or xml[cursor] not in "'\"":
                raise ValueError("XML attribute requires quotes")
            delimiter = xml[cursor]
            cursor += 1
            while cursor < stop and xml[cursor] != delimiter:
                cursor += 1
            if cursor == stop:
                raise ValueError("unterminated XML attribute")
            cursor += 1
            if name not in names:
                names.add(name)
                out.append(xml[whitespace:cursor])
            if cursor == stop:
                out.append(">")
        # A tag without attributes has not yet emitted its closing delimiter.
        if cursor == stop and not (out and out[-1].endswith(">")):
            out.append(">")
    return "".join(out)


def parse_xml(path: str | Path) -> ET.Element:
    """Parse a streams file with ezxml's first-value attribute rule; MPAS-Model-rrfs/src/external/ezxml/ezxml.c:83-99,490-550."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as error:
        raise OSError(f"stream XML could not open {path}") from error
    return ET.fromstring(first_value_xml(text))


def extract_stream_interval(interval: str, interval_type: str, stream_id: str, streams: ET.Element) -> str:
    """Resolve ``stream:NAME:input_interval|output_interval``; MPAS-Model-rrfs/src/framework/xml_stream_parser.c:925-1023.

    A value that does not start with ``stream:`` comes back unchanged. Each refusal is the
    native function's own error return, named with its log message.
    """
    if not interval.startswith("stream:"):
        return interval
    # snprintf(match_stream_name, 256, "%s", interval+7); the LAST ':' splits stream from
    # attribute, while every character after any ':' accumulates into the attribute name.
    text = interval[7:][:255]
    copy_start, copy_from, name_chars = -1, -1, []
    for i, char in enumerate(text):
        if char == ":":
            copy_start, copy_from = i, i + 1
        if copy_from == i:
            name_chars.append(text[copy_from])
            copy_from += 1
    interval_name = "".join(name_chars)
    if copy_start < 0:
        # Native writes match_stream_name[-1] here (undefined), then fails the name check below.
        raise ValueError(f"Attribute '{interval_type}' of stream '{stream_id}' references an invalid "
                         f"attribute: '{interval_name}'.")
    match = text[:copy_start]
    if match == stream_id and interval_name == interval_type:
        raise ValueError(f"Attribute '{interval_type}' of stream '{stream_id}' references itself.")
    if interval_name not in ("input_interval", "output_interval"):
        raise ValueError(f"Attribute '{interval_type}' of stream '{stream_id}' references an invalid "
                         f"attribute: '{interval_name}'.")
    found = None
    for tag in ("immutable_stream", "stream"):
        for element in streams.findall(tag):
            if element.get("name") == match:
                found = element
                break
        if found is not None:
            break
    if found is None:
        raise ValueError(f"The '{interval_type}' attribute of stream '{stream_id}' refers to an undefined "
                         f"stream named '{match}'.")
    value = found.get(interval_name)
    if value is None:
        raise ValueError(f"The '{interval_type}' attribute of stream '{stream_id}' refers to an undefined "
                         f"attribute named '{interval_name}' of stream '{match}'.")
    if value in ("input_interval", "output_interval") or value.startswith("stream:"):
        raise ValueError(f"The '{interval_type}' attribute of stream '{stream_id}' contains an "
                         f"unexpandable value: '{value}'.")
    return value


def stream_tree(path, library=None):
    """The resolved stream tree, native-free; the ``streams_parser`` hook of the RRFS route.

    Same result as ``driver_rrfs.stream_tree`` (which parsed through the native library's ezxml
    and ``extract_stream_interval``): first-value attributes, and ``filename_interval`` resolved
    by the native default rules, MPAS-Model-rrfs/src/framework/xml_stream_parser.c:925-1023,1150-1209,1495-1560. ``library``
    is accepted for the hook's signature and never loaded.
    """
    tree = parse_xml(path)
    for element in tree:
        if element.tag not in ("stream", "immutable_stream"):
            continue
        intervals = {}
        for key in ("input_interval", "output_interval"):
            raw = element.get(key)
            if raw is None:
                continue
            try:
                intervals[key] = extract_stream_interval(raw, key, element.get("name"), tree)
            except ValueError as error:
                raise ValueError(f"native stream {element.get('name')} cannot resolve {key}: {error}") from None

        def recurring(key):
            """Native keyword predicate; MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1508-1529."""
            raw = element.get(key)
            return raw is not None and not any(token in raw for token in ("initial_only", "final_only", "none"))

        interval = element.get("filename_interval")
        direction = element.get("type", "")
        if interval is None:
            if "input" in direction and recurring("input_interval"):
                interval = intervals["input_interval"]
            elif "output" in direction and recurring("output_interval"):
                interval = intervals["output_interval"]
            elif (element.tag == "stream" and "output" in direction
                  and element.get("output_timelevels") is not None):
                interval = "output"
        else:
            for key in ("input_interval", "output_interval"):
                if key in interval:
                    interval = intervals[key] if recurring(key) else None
                    break
        element.set("filename_interval", interval or "none")
    return tree


# --- output_timelevels, RKIND=real32: MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6059-6606 ---

_F32 = np.float32
_F32_MAX = np.finfo(np.float32).max
_RUST_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\Z")
_RUST_SPECIAL = re.compile(r"[+-]?(?:inf|infinity|nan)\Z", re.IGNORECASE)


def _to_f32(text: str) -> np.float32:
    """Correctly rounded decimal -> binary32 (one rounding, ties to even), the RKIND=real32 value the
    native list-directed read gives; MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6152-6230."""
    if _RUST_SPECIAL.match(text):
        return _F32(float(text))
    if not _RUST_FLOAT.match(text):
        raise ValueError(f"invalid time number {text!r}")
    mantissa, _, exponent = text.lower().partition("e")
    exact = Fraction(mantissa if mantissa not in ("", "+", "-") else "0") * Fraction(10) ** int(exponent or 0)
    with np.errstate(over="ignore"):
        guess = _F32(float(exact)) if abs(exact) < Fraction(2) ** 200 else _F32(np.inf) * (1 if exact > 0 else -1)
    if not np.isfinite(guess):
        return guess
    best = guess
    for candidate in (np.nextafter(guess, _F32(-np.inf)), np.nextafter(guess, _F32(np.inf))):
        if not np.isfinite(candidate):
            continue
        gap_best, gap_candidate = abs(Fraction(float(best)) - exact), abs(Fraction(float(candidate)) - exact)
        if gap_candidate < gap_best or (gap_candidate == gap_best
                                        and int(np.array(candidate).view(np.uint32)) % 2 == 0):
            best = candidate
    return _F32(best)


def _number(raw: str) -> np.float32:
    """List-directed real read, D exponent allowed, list tail ignored; MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6152-6230."""
    first = re.split(r"[,/ \t]", raw.lstrip(), maxsplit=1)[0]
    return _to_f32(first.replace("D", "e").replace("d", "e"))


def _minutes(raw: str) -> np.float32:
    """A time string in minutes; MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6147-6230."""
    raw = raw.strip()
    if not raw:
        raise ValueError("empty time string")
    if raw[-1].isascii() and raw[-1].isdigit():
        return _F32(_number(raw) * _F32(60.0))
    start, total = 0, _F32(0.0)
    for index, char in enumerate(raw):
        if char in "hmsdD":
            if index == start:
                raise ValueError("unit without time number")
            value = _number(raw[start:index])
            if char == "h":
                total = _F32(total + _F32(value * _F32(60.0)))
            elif char == "m":
                total = _F32(total + value)
            elif char == "s":
                total = _F32(total + _F32(value / _F32(60.0)))
            else:
                total = _F32(total + _F32(_F32(value * _F32(24.0)) * _F32(60.0)))
            start = index + 1
    if start < len(raw):
        raise ValueError("trailing time number without unit")
    return total


def parse_timelevels(spec: str, *, reject_step_after_stop: bool = False) -> list[tuple[np.float32, np.float32, np.float32]]:
    """Segments (start hour, end hour, step minutes); MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6059-6140,6248-6351.

    mpas_split_string splits on literal spaces and counts empty segments (at most 32).
    """
    segments = spec.strip(" ").split(" ")
    if len(segments) > 32:
        raise ValueError("too many timelevel segments (max 32)")
    result = []
    with np.errstate(over="ignore", invalid="ignore"):
        for raw in segments:
            if not raw:
                continue
            parts = raw.split("-")
            if len(parts) > 3:
                raise ValueError("expected start, start-stop or start-stop-step")
            start = _F32(_minutes(parts[0]) / _F32(60.0))
            end = _F32(_minutes(parts[1]) / _F32(60.0)) if len(parts) >= 2 else start
            step = _minutes(parts[2]) if len(parts) == 3 else _F32(60.0)
            # 44c0849 mpas_output_timelevels.F:320 compares to absolute stop,
            # not the segment duration. Keep each RKIND rounding.
            if reject_step_after_stop and len(parts) == 3 and step > _F32(_F32(end * _F32(60.0)) + _F32(1.0e-4)):
                raise ValueError("expected start-stop-step, appears to be start-step-stop")
            result.append((start, end, step))
    if not result:
        raise ValueError("no valid timelevel segments")
    return result


def _interval(segments, hour):
    """MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6368-6465: (matched, step minutes, next start hour or -1)."""
    single = False
    for start, end, step in segments:
        if hour >= start and hour <= end:
            if abs(_F32(start - end)) < _F32(1.0e-6):
                single = True
            else:
                return True, step, _F32(-1.0)
    if single:
        upcoming = _F32_MAX
        for start, _end, _step in segments:
            if start > _F32(hour + _F32(1.0e-6)) and start < upcoming:
                upcoming = start
        if upcoming < _F32_MAX:
            return True, _F32(_F32(upcoming - hour) * _F32(60.0)), upcoming
        return True, _F32(0.0), _F32(-1.0)
    return False, _F32(0.0), _F32(-1.0)


def _future(segments, hour):
    """MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6480-6510: (next segment start hour or -1, its step minutes)."""
    upcoming, step = _F32_MAX, _F32(60.0)
    for start, _end, seg_step in segments:
        if start > hour and start < upcoming:
            upcoming, step = start, seg_step
    return (upcoming if upcoming < _F32_MAX else _F32(-1.0)), step


def _dt(value) -> str:
    """Native prints nine decimals then keeps eight fractional digits; MPAS-Model-rrfs/src/framework/mpas_timekeeping.F:1451-1472,1521-1534."""
    text = "%.9f" % float(_F32(value))
    return text[:-1]


def _bits(value) -> int:
    """The binary32 word of an RKIND value (the native oracle's transfer(x, 0));
    MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6554-6606."""
    return int(np.array(_F32(value), dtype=np.float32).view(np.uint32))


def timelevel_query(specification: str, seconds: int, *, reject_step_after_stop: bool = False) -> dict:
    """The output_timelevels alarm query at ``seconds`` after start; MPAS-Model-rrfs/src/framework/mpas_stream_manager.F:6554-6606,7147-7167.

    Same result as the r11 artifact's JSON (woof b3a0b0db rw_mpas_rrfs_clock.rs:157-179): seconds as the
    native eight-decimal text read back as numbers, plus the binary32 words of every segment and query value.
    """
    segments = parse_timelevels(specification, reject_step_after_stop=reject_step_after_stop)
    with np.errstate(over="ignore", invalid="ignore"):
        hour = _F32(_F32(int(seconds)) / _F32(3600.0))
        matched, step, upcoming = _interval(segments, hour)
        future, future_step = _future(segments, hour)
        interval_seconds = float(_dt(_F32(step * _F32(60.0))))
        next_seconds = float(_dt(_F32(upcoming * _F32(3600.0)))) if upcoming > 0 else None
        future_seconds = float(_dt(_F32(future * _F32(3600.0)))) if future > 0 else None
        future_interval = float(_dt(_F32(future_step * _F32(60.0))))
    return {"matched": bool(matched), "interval_seconds": interval_seconds, "next_seconds": next_seconds,
            "future_seconds": future_seconds, "future_interval_seconds": future_interval,
            "words": [[_bits(a), _bits(b), _bits(c)] for a, b, c in segments],
            "query_words": [_bits(hour), _bits(step), _bits(upcoming), _bits(future), _bits(future_step)]}


def native_file_id(seed: int, ordinal: int) -> str:
    """The ``ordinal``-th ten-character stream file ID after ``srand(seed)``; MPAS-Model-rrfs/src/framework/random_id.c:20-46.

    GNU libc's rand() sequence, from the v8.4.2 route's isolated generator (no process-global
    libc state is touched): each ID consumes ten words, so ID ``ordinal`` follows ``10*ordinal`` words.
    """
    if ordinal < 0:
        raise ValueError("file ID ordinal must be non-negative")
    from ..stream_metadata_v842 import NativeFileIds
    ids = NativeFileIds(int(seed) & 0xffffffff)
    for _ in range(ordinal):
        ids.next()
    return ids.next()
