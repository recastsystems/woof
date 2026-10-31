"""Licence gates on the MPAS suite routes: upstream code under restricted terms runs only by an opt-in that names it.

THE BREAKAGE THIS PREVENTS.  woof hex is Apache-2.0 and its public distribution carries only permissive
third-party code and US Government code.  Before this gate an MPAS suite run with ``config_lsm_scheme='sf_noahmp'``
ran Noah-MP v5.0 (MPAS's, or the RRFS fork's), whose UCAR terms forbid selling it or any work that contains it.
Nothing on the console or in the receipt said so, so a run (or a box built from one) could carry code that cannot
be sold without anyone having chosen it.  The owner ruling of 2026-10-14: the package stays Apache-2.0 and no
default routes into code that cannot be sold or is limited.

Not gated (owner ruling 2026-10-14): UGWP and the RRFS fork's NOAA GSL physics (MYNN-EDMF, MYNN-SFC, RUC, GFL,
smoke) are treated as US Government code; they ship and the fork-exact RRFS stage (hrrrv5) runs with no opt-in.
The table records them under ``not_gated``.  Their former opt-in names, ``noaa-gsl-unlicensed`` and ``ugwp-held``,
are listed under ``retired_terms``: a run that still names one gets one line saying so and runs exactly as it would
without it (scripts written while it gated are not broken by it).

What it does.  ``data/profiles/licence-gates.v1.json`` lists each (suite row, slot, scheme) that runs such code,
the component it runs and that component's terms.  :func:`admit` is called by the forecast door on every resolved
MPAS suite route (v8.4.2 and the RRFS fork):

* no gated selection active: the configuration is returned unchanged and nothing is printed (every default);
* a gated selection without an opt-in naming its terms: :class:`LicenceRefusal`, naming each selection, its terms,
  the permissive alternative and the exact opt-in;
* a hex-cuda gated selection whose port is not in this distribution (the public export withholds it,
  tools/licence_holds.json): :class:`LicenceRefusal` naming the licence, never a silent swap to another scheme;
* otherwise one ``LICENCE OPT-IN`` line per selection on stderr, and the physics receipt gains
  ``licence_opt_in`` (receipts of runs without a gated selection are unchanged, byte for byte).

The opt-in is ``--licence-opt-in ID`` (repeatable or comma-separated) or ``WOOF_HEX_LICENCE_OPT_IN=ID[,ID]``;
the receipt records which one named it.  The library resolvers (``resolve_namelist``, ``native_route``) are not
gated, so module replays are unchanged; the door is the product route.

``--licence-opt-in ucar-noahmp`` also builds Noah-MP v5.0 into the native framework library
(``woof hex physics-resources build``); the default build compiles a licence stub in its place, and
:func:`native_library_refusal` refuses an opted-in run on a stubbed library before any work.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from importlib import resources
import importlib.util
import json
import os
import sys
from typing import Any, Mapping, TextIO

from .physics_suites import load_table as load_suite_table

SCHEMA = "gpuwm-hex.licence-gates/v1"
TABLE_PATH = "data/profiles/licence-gates.v1.json"
ENVIRONMENT = "WOOF_HEX_LICENCE_OPT_IN"
DEFAULT_ROW = "mpasphys_v842"


class LicenceRefusal(ValueError):
    """A run selects restricted or unlicensed upstream code without opting in by name, or its port is withheld."""


@lru_cache(maxsize=1)
def load_table() -> Mapping[str, Any]:
    """The gate table, validated against itself and against the suite table."""
    table = json.loads(resources.files("woof.hex").joinpath(TABLE_PATH).read_text(encoding="utf-8"))
    if table.get("schema") != SCHEMA:
        raise ValueError(f"{TABLE_PATH}: schema {table.get('schema')!r}, expected {SCHEMA!r}")
    suites = load_suite_table()
    gated = {row["component"] for row in table["selections"]}
    for name in table.get("not_gated", {}):
        # A ruling that took an upstream out of the gate must not leave a selection that still refuses it.
        if name in table["components"] or name in gated:
            raise ValueError(f"{TABLE_PATH}: {name} is recorded as not gated and is still gated")
    for name in table.get("retired_terms", {}):
        if name in table["terms"]:
            raise ValueError(f"{TABLE_PATH}: {name} is both live and retired terms")
    for row in table["selections"]:
        component = table["components"].get(row["component"])
        if component is None:
            raise ValueError(f"{TABLE_PATH}: selection {row} names an unknown component")
        if component["terms"] not in table["terms"]:
            raise ValueError(f"{TABLE_PATH}: component {row['component']} names unknown terms {component['terms']!r}")
        slots = suites.get(row["row"], {}).get("slots", {})
        if row["slot"] not in slots or row["scheme"] not in slots[row["slot"]]["schemes"]:
            raise ValueError(f"{TABLE_PATH}: {row['row']} has no {row['slot']}={row['scheme']!r}; the gate would "
                             "never match it")
    return table


def known_terms() -> tuple[str, ...]:
    return tuple(load_table()["terms"])


def retired_terms() -> Mapping[str, Mapping[str, Any]]:
    return load_table().get("retired_terms", {})


def retired_named(arguments: Any, environ: Mapping[str, str] | None = None) -> list[str]:
    """The retired opt-in names this run still gives (they gate nothing; :func:`admit` says so once)."""
    environ = os.environ if environ is None else environ
    given = [value.strip() for value in environ.get(ENVIRONMENT, "").split(",")]
    for value in getattr(arguments, "licence_opt_in", None) or ():
        given.extend(item.strip() for item in str(value).split(","))
    return sorted({name for name in given if name in retired_terms()})


def requested(arguments: Any, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The terms this run opts in to, each mapped to what named it (``--licence-opt-in`` or the environment)."""
    environ = os.environ if environ is None else environ
    named: dict[str, str] = {}
    for value in environ.get(ENVIRONMENT, "").split(","):
        if value.strip():
            named[value.strip()] = ENVIRONMENT
    for value in getattr(arguments, "licence_opt_in", None) or ():
        for item in str(value).split(","):
            if item.strip():
                named[item.strip()] = "--licence-opt-in"
    for name in retired_terms():
        named.pop(name, None)
    unknown = sorted(set(named) - set(known_terms()))
    if unknown:
        raise LicenceRefusal(f"LICENCE: unknown opt-in {', '.join(unknown)} (from "
                             f"{', '.join(sorted({named[u] for u in unknown}))}); the known terms are "
                             f"{', '.join(known_terms())}")
    return named


def hits(config: Any) -> list[dict[str, Any]]:
    """Each active selection of ``config`` that runs gated upstream code."""
    table = load_table()
    row_key = getattr(config, "table_key", DEFAULT_ROW)
    found = []
    for row in table["selections"]:
        if row["row"] != row_key or config.selectors.get(row["slot"]) != row["scheme"]:
            continue
        if not config.active(row["slot"]):
            continue
        component = table["components"][row["component"]]
        found.append({"slot": row["slot"], "scheme": row["scheme"], "backend": str(config.backends[row["slot"]]),
                      "component": row["component"], "what": component["what"],
                      "upstream": component["upstream"], "terms": component["terms"],
                      "hex_port_hold": component.get("hex_port_hold")})
    return found


def cuda_module(row_key: str, slot: str, scheme: str) -> str | None:
    """The hex-cuda module the suite table names for this selection, relative to woof.hex.mpasphys."""
    descriptor = load_suite_table()[row_key]["slots"][slot].get("cuda_entries", {}).get(scheme)
    return None if descriptor is None else descriptor["module"]


def port_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(f"woof.hex.mpasphys.{module}") is not None
    except ModuleNotFoundError:
        return False


def _terms_line(terms_id: str) -> str:
    terms = load_table()["terms"][terms_id]
    clause = f': "{terms["clause"]}"' if terms.get("clause") else ""
    return f"{terms['name']} ({terms['summary']}){clause} [{terms['notice']}]"


def refusal_text(config: Any, found: list[dict[str, Any]], missing: list[str]) -> str:
    table = load_table()
    lines = ["LICENCE: refused. This MPAS suite run selects upstream code that is not under a permissive licence "
             "and does not opt in to it. No default route selects it; woof hex is Apache-2.0 and ships only "
             "permissive code."]
    for hit in found:
        if hit["terms"] not in missing:
            continue
        component = table["components"][hit["component"]]
        lines.append(f"LICENCE:   {hit['slot']}={hit['scheme']} ({hit['backend']}) runs {hit['what']} "
                     f"({hit['upstream']}), under {_terms_line(hit['terms'])}.")
        if component.get("note"):
            lines.append(f"LICENCE:     note: {component['note']}.")
        lines.append(f"LICENCE:     permissive alternative: {component['alternative']}")
    if any(hit["component"] == "noahmp-v5-mpas" for hit in found):
        lines.append("LICENCE:   Noah-MP: WRF-lineage v4.x (permissive) runs on the WOOF suite; not MPAS-bitwise; "
                     "the MPAS v5.0 port is withheld for licence from the public distribution.")
    lines.append(f"LICENCE:   to run it under those terms anyway, opt in by name: --licence-opt-in "
                 f"{','.join(missing)} (or {ENVIRONMENT}={','.join(missing)}). With native-cpu it runs from your own "
                 "MPAS build; a hex-cuda port of it ships only in private builds.")
    return "\n".join(lines)


def native_library_refusal(config: Any, library: Any) -> LicenceRefusal | None:
    """The refusal for a run whose gated selection needs code that ``library`` carries only as a licence stub.

    THE BREAKAGE THIS PREVENTS.  The default ``woof hex physics-resources build`` compiles the native
    framework library with a licence stub for Noah-MP v5.0 (a component's ``native_library_stub``
    in the gate table), so the library contains none of it.  MPAS's own init, driver and finalize call it
    for a run that selects it, whatever the slot's backend, so an opted-in run on a stubbed library would stop
    inside the stub after the whole setup.  This names the rebuild instead, before any work.  ``None`` when
    the run selects no such code or the library carries it (a lab build named by path records no stubs).
    """
    from .mpasphys.native_library_build import library_licence
    table = load_table()
    needed = [hit for hit in hits(config) if table["components"][hit["component"]].get("native_library_stub")]
    if not needed:
        return None
    stubbed = library_licence(library)["stubbed"]
    missing = [hit for hit in needed if table["components"][hit["component"]]["native_library_stub"] in stubbed]
    if not missing:
        return None
    terms = sorted({hit["terms"] for hit in missing})
    lines = [f"LICENCE: refused. The MPAS native framework library {library} was built with licence stubs "
             f"(the default `woof hex physics-resources build`), so it contains none of the code this run selects:"]
    for hit in missing:
        lines.append(f"LICENCE:   {hit['slot']}={hit['scheme']} ({hit['backend']}) needs {hit['what']} inside the "
                     f"native library (MPAS's own init and driver call it), under {_terms_line(hit['terms'])}.")
    lines.append(f"LICENCE:   build the library with that code under those terms: woof hex physics-resources build "
                 f"--licence-opt-in {','.join(terms)} (or name such a build with --native-physics-library).")
    return LicenceRefusal("\n".join(lines))


_ANNOTATED: dict[type, type] = {}


def _annotated_type(base: type) -> type:
    """``base`` with a receipt that carries the run's licence opt-ins (only such runs use it)."""
    kind = _ANNOTATED.get(base)
    if kind is None:
        def receipt(self) -> dict[str, Any]:
            result = base.receipt(self)
            result["licence_opt_in"] = [dict(row) for row in self.__dict__["_licence_opt_in"]]
            return result
        kind = type(base.__name__, (base,), {"receipt": receipt, "__module__": base.__module__,
                                              "__doc__": base.__doc__})
        _ANNOTATED[base] = kind
    return kind


def admit(config: Any, arguments: Any, *, stream: TextIO | None = None,
          environ: Mapping[str, str] | None = None) -> Any:
    """Refuse, or admit loudly, a resolved MPAS suite configuration (see the module docstring)."""
    found = hits(config)
    named = requested(arguments, environ)
    retired = retired_named(arguments, environ)
    if retired:
        # THE BREAKAGE THIS PREVENTS: a recipe written while noaa-gsl-unlicensed gated the RRFS stage would be
        # refused as an unknown opt-in once the ruling retired it; it runs, and says the name does nothing.
        print(f"LICENCE: opt-in {', '.join(retired)} is retired and gates nothing "
              f"({retired_terms()[retired[0]]['why']}); this run is the one it would be without it",
              file=sys.stderr if stream is None else stream, flush=True)
    if not found:
        return config
    missing = sorted({hit["terms"] for hit in found} - set(named))
    if missing:
        raise LicenceRefusal(refusal_text(config, found, missing))
    row_key = getattr(config, "table_key", DEFAULT_ROW)
    for hit in found:
        if hit["backend"] != "hex-cuda":
            continue
        module = cuda_module(row_key, hit["slot"], hit["scheme"])
        if module is not None and not port_present(module):
            hold = hit["hex_port_hold"]
            raise LicenceRefusal(
                f"LICENCE: refused. {hit['slot']}={hit['scheme']} on hex-cuda needs the hex port "
                f"woof.hex.mpasphys.{module}, which is not in this distribution"
                + (f": the public export withholds it for licence (tools/licence_holds.json hold {hold})"
                   if hold else "")
                + f". It runs {hit['what']} under {_terms_line(hit['terms'])}. Run it from your own MPAS build "
                f"with the native-cpu backend for that slot, or from a private build that carries the port.")
    stream = sys.stderr if stream is None else stream
    record = []
    for hit in found:
        source = named[hit["terms"]]
        where = ("the hex-cuda port in this build" if hit["backend"] == "hex-cuda"
                 else "your own MPAS build" if hit["backend"] == "native-cpu" else hit["backend"])
        print(f"LICENCE OPT-IN {hit['terms']} ({source}): {hit['slot']}={hit['scheme']} runs {hit['what']} from "
              f"{where}, under {_terms_line(hit['terms'])}. NOT Apache-2.0.", file=stream, flush=True)
        record.append({key: hit[key] for key in ("slot", "scheme", "backend", "component", "upstream", "terms")}
                      | {"named_by": source})
    values = {field.name: getattr(config, field.name) for field in dataclasses.fields(config) if field.init}
    annotated = _annotated_type(type(config))(**values)
    object.__setattr__(annotated, "_licence_opt_in", tuple(record))
    return annotated
