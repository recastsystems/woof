"""ERA5 selection metadata and transport to the native GRIB1 member checker."""
from __future__ import annotations
from collections.abc import Mapping
import json
import os
from pathlib import Path
import subprocess

SCHEMA = "arwen.era5-member-selection.v1"
#: The ``[fetch]`` hint that selects the ERA5 archive provider; ``woof go
#: --era5-provider`` is the same choice as a flag.
PROVIDER_KEY = "era5_provider"
#: The ten EDA members the CDS serves, spelled as the member grammar does.
EDA_MEMBERS = tuple(str(index) for index in range(10))
EDA_PROVIDER = "cds"
EDA_CADENCE_HOURS = 3


# The ERA5 adapter row names this module as its selection owner, so the
# generic ensemble line (woof/ensemble, held to the table law) asks these
# functions instead of comparing a source id.

def provider_of(table) -> str | None:
    """The provider a ``[fetch]`` table selects, if it names one."""
    return table.get(PROVIDER_KEY) if isinstance(table, Mapping) else None


def provider_flag(args) -> str | None:
    """``--era5-provider`` on a parsed namespace, if it was given."""
    return getattr(args, PROVIDER_KEY, None)


def wizard_arguments(args) -> list:
    """The ``woof domain`` arguments that carry the namespace's provider choice."""
    provider = provider_flag(args)
    return [] if not provider else ["--era5-provider", str(provider)]


def ensemble_members(*, provider=None, cycle=None) -> tuple:
    """The members an ensemble planner may draw from ERA5 at one analysis time.

    The EDA is served by the CDS only: with a CDS key in place and a
    provider other than ARCO every member is a trajectory of its own;
    otherwise ERA5 has one (the HRES reanalysis) and the planner says so.
    The EDA analyses are 3-hourly, so at any other hour (HRES is hourly,
    e.g. a 16Z start) there is no EDA member and HRES is the one
    trajectory.  Breakage this prevents: a 16Z ERA5 ensemble on a box
    with a CDS key built ten EDA trajectories that each refused with
    "ERA5 EDA initialization must be an exact 3-hourly UTC analysis time".
    """
    from woof.fetch import cds_credentials_present
    if provider == "arco" or not cds_credentials_present():
        return ()
    if cycle is not None and cycle.hour % EDA_CADENCE_HOURS:
        return ()
    return EDA_MEMBERS


def validate_member(member, *, cycle) -> str:
    """One EDA member of a source trajectory, validated as its fetch validates it."""
    return str(validate_selection(product_type="ensemble_members", provider=EDA_PROVIDER,
                                  member=member, cadence=EDA_CADENCE_HOURS, cycle=cycle))


def member_fetch_hints(fetch, member, *, window_seconds, root) -> dict:
    """The ``[fetch]`` table that acquires one EDA member's window under ``root``."""
    import math
    hours = math.ceil(float(window_seconds) / (EDA_CADENCE_HOURS * 3600)) * EDA_CADENCE_HOURS
    return dict(fetch, era5_provider=EDA_PROVIDER, era5_product="ensemble_members",
                member=member, cadence=EDA_CADENCE_HOURS, hours=hours,
                out=str(Path(root) / ("eda-" + str(member))))


def member_forcing() -> tuple:
    """What one member's acquisition publishes: the forcing file name and its interval (s)."""
    from woof.fetch import era5_combined_name
    return era5_combined_name(EDA_PROVIDER), EDA_CADENCE_HOURS * 3600


def validate_selection(*, product_type="reanalysis", member=None, cadence=6, provider="cds", cycle=None):
    if product_type not in ("reanalysis", "ensemble_members"):
        raise ValueError("ERA5 product must be reanalysis or ensemble_members")
    if provider not in ("cds", "arco"):
        raise ValueError("ERA5 provider must be cds or arco")
    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence <= 0:
        raise ValueError("ERA5 boundary cadence must be a positive whole number of hours")
    if product_type == "reanalysis":
        if member is not None:
            raise ValueError("ERA5 reanalysis has no EDA member. Select era5_product = 'ensemble_members' explicitly with era5_provider = 'cds', member = 0..9 and a cadence in multiples of 3 hours, or omit member for reanalysis")
        _warn_coarse_cadence(cadence)
        return None
    if provider != "cds" or cadence % 3:
        raise ValueError("ERA5 EDA requires CDS and a cadence that is a positive multiple "
                         "of its native 3-hour interval; this WOOF ARCO reader carries HRES reanalysis only")
    if isinstance(member, str) and len(member) == 1 and member.isascii() and member.isdigit():
        member = int(member)
    if isinstance(member, bool) or not isinstance(member, int) or not 0 <= member <= 9:
        raise ValueError("ERA5 EDA requires an explicit member number 0..9")
    if cycle is not None and (cycle.hour % 3 or cycle.minute or cycle.second or cycle.microsecond):
        raise ValueError("ERA5 EDA initialization must be an exact 3-hourly UTC analysis time")
    _warn_coarse_cadence(cadence)
    return member


def _warn_coarse_cadence(cadence: int) -> None:
    if cadence <= 6:
        return
    from woof.explain import warn_once
    warn_once(
        f"boundary-cadence:{cadence}",
        f"Boundary cadence is {cadence} hours; the requested cadence is retained, "
        "but forcing is coarser than the 6-hour reference. Use a shorter cadence "
        "when faster changes matter.",
        "The source publishes analyses at the selected hours. A longer interval "
        "is valid input, not a missing-data condition; temporal detail between "
        "the selected analyses is reduced.")


def _invoke(bridge, arguments, *, timeout=300):
    process = subprocess.run([str(bridge), *map(str, arguments)], capture_output=True,
        timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
    if process.returncode:
        detail = process.stderr[:8192].decode("utf-8", errors="replace").strip()
        raise ValueError("Native ERA5 member verification refused the payload: " + detail)
    if len(process.stdout) > 16384:
        raise ValueError("Native ERA5 member checker returned oversized metadata")
    report = json.loads(process.stdout)
    if report.get("schema") != SCHEMA or report.get("byte_preserving") is not True:
        raise ValueError("Native bridge does not implement the required ERA5 member selection contract")
    return report


def require_bridge():
    from woof.bridges import find_bridge
    bridge = find_bridge("grib1_bridge")
    if bridge is None:
        raise ValueError("ERA5 EDA needs the native grib1_bridge with --era5-member support; rebuild/install the matching WOOF bridge")
    report = _invoke(bridge, ["--era5-member-capabilities"], timeout=15)
    if report.get("members") != list(range(10)) or report.get("complete_input_census") is not True:
        raise ValueError("The selected native bridge lacks complete ten-member ERA5 EDA verification")
    if report.get("local_definitions") != [1, 17, 36]:
        raise ValueError("The selected native bridge lacks the ERA5 surface/soil/SST/sea-ice local definitions; rebuild/install the matching bridge")
    expected_lake = {"center": 98, "table": 228, "parameters": [8, 13, 14], "surface_only": True}
    if (report.get("table_qualified_census") is not True
            or report.get("lake_surface_parameters") != expected_lake):
        raise ValueError("The selected native bridge lacks table-qualified ERA5 lake-field member verification; rebuild/install the matching bridge")
    return Path(bridge)


def check_member(path, member, *, bridge=None):
    report = _invoke(bridge or require_bridge(), ["--check-era5-member", member, path])
    if report.get("member") != member or report.get("selected_only") is not True or report.get("messages", 0) <= 0:
        raise ValueError("Native ERA5 member receipt does not match the selected member")
    return report


def select_member(path, output, member, *, bridge=None):
    report = _invoke(bridge or require_bridge(), ["--era5-member", member, path, output])
    if report.get("member") != member or report.get("selected_only") is not True or report.get("input_messages") != 10 * report.get("messages", 0):
        raise ValueError("Native ERA5 selection did not verify all ten members for every field")
    return report
