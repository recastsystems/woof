"""Bind stable WOOF-ArWen grid IDs to compact, standard Fortran WPS slots.

WPS arrays and parent_id contain slots 1..max_dom.  An optional versioned
Fortran comment binds those slots to stable IDs; ordinary contiguous files
keep their historical bytes and interpretation.  Geometry is unchanged.
"""
from __future__ import annotations

import re
from typing import Sequence

_RESERVED = re.compile(r"^\s*!\s*GPUWM_DOMAIN_IDS[^\r\n]*", re.MULTILINE)
_DECLARATION = re.compile(r"\s*!\s*GPUWM_DOMAIN_IDS_V1\s*=\s*([1-9][0-9]*(?:\s*,\s*[1-9][0-9]*)*)\s*")


def validated_domain_ids(ids: Sequence[int]) -> tuple[int, ...]:
    """Require a unique positive grid inventory beginning with root d01."""
    result = tuple(ids)
    if (not result or any(type(value) is not int or value < 1 for value in result)
            or result[0] != 1 or len(set(result)) != len(result)):
        raise ValueError("WPS domain identity requires unique positive integer IDs with d01 first")
    return result


def validated_domain_order(domains) -> tuple[int, ...]:
    """Bind stable IDs to one root and parent-before-child experiment order."""
    domains = tuple(domains)
    ids = validated_domain_ids(tuple(domain.grid_id for domain in domains))
    seen = set()
    for domain in domains:
        parent = domain.parent_id
        if type(parent) is not int or (domain.grid_id == 1 and parent != 0):
            raise ValueError("d01 must be the sole root with parent_id=0")
        if domain.grid_id != 1 and parent not in seen:
            raise ValueError(f"d{domain.grid_id:02d} parent d{parent:02d} must precede it (parent-before-child order)")
        seen.add(domain.grid_id)
    return ids


def _declared_ids(text: str) -> tuple[int, ...] | None:
    matches = list(_RESERVED.finditer(text))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("WPS domain identity has duplicate reserved declarations")
    declaration = _DECLARATION.fullmatch(matches[0].group())
    if declaration is None:
        raise ValueError("Malformed or unsupported WPS domain identity; expected GPUWM_DOMAIN_IDS_V1")
    return validated_domain_ids(tuple(int(value.strip()) for value in declaration[1].split(",")))


def domain_ids_from_wps_text(text: str, max_dom: int) -> tuple[int, ...]:
    """Read strict slot identity, or the legacy contiguous order if absent."""
    if type(max_dom) is not int or max_dom < 1:
        raise ValueError("WPS max_dom must be a positive integer")
    ids = _declared_ids(text)
    if ids is None:
        return tuple(range(1, max_dom + 1))
    if len(ids) != max_dom:
        raise ValueError(f"WPS domain identity has {len(ids)} IDs but max_dom={max_dom}")
    return ids


def with_domain_ids(text: str, ids: Sequence[int]) -> str:
    """Replace a valid identity comment; contiguous output needs no comment."""
    ids = validated_domain_ids(ids)
    _declared_ids(text)  # Never repair malformed or ambiguous authority silently.
    result = _RESERVED.sub("", text)
    if result != text:
        result = result.lstrip("\r\n")
    if ids == tuple(range(1, len(ids) + 1)):
        return result
    return "! GPUWM_DOMAIN_IDS_V1 = " + ",".join(map(str, ids)) + "\n" + result
