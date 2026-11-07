"""Every composition the registry accepts can be STREAMED, or says why not.

THE DEFECT CLASS THIS EXISTS FOR
--------------------------------
``[tiles]`` streaming rebuilds a per-buffer twin of each physics adapter
(``woof.core.streaming._tile_scheme``).  An adapter it cannot rebuild stops
the run inside ``TiledRun.__init__`` -- after the prepared cache has been
restored, the domain built on the card and the tile buffers allocated.

That is exactly how the shipped DEFAULT suite shipped broken.
``RRTMGLegacyRadiation`` is a plain class whose constructor requires
``start_time``/``latitude_deg``/``longitude_deg``, the twin builder handled
only dataclasses and empty constructors, and so every ``[tiles]`` forecast
of the default physics selected ``StreamingRefused``.  Nothing caught it
because no test in the tree asked the question of the SHIPPED SCHEMES; the
twinning tests all used stand-ins that were dataclasses by construction.

So this suite asks the question of the registry itself, programmatically.
It enumerates ``woof/physics_registry_v2.json`` rather than naming suites,
so a scheme added to the registry tomorrow is audited the day it is added,
and a scheme added WITHOUT a tile constructor goes red here -- on CPU, in
the ordinary battery -- instead of an hour into somebody's forecast.

WHY THIS IS THE COMPLETE AUDIT, not merely the radiation one.  A physics
suite selects seven components, and only TWO of them install an adapter
OBJECT that a buffer must be given: ``PhysicsDriver.__init__`` assigns
``radiation_callable`` and ``cumulus_callable``
(``woof/core/physics.py:1827-1828``) and nothing else.  The other five --
microphysics, PBL, surface layer, land surface, turbulence -- are selected
by ``initialize_physics`` from the config, so a tile buffer built from the
domain's own ``cfg`` selects them identically, and their state travels as
ordinary carriers through the store.  The radiation and cumulus adapters are
handed over explicitly precisely BECAUSE they carry policy the config does
not replay (trace-gas overrides, ``column_chunk``, ``o3input``).
:func:`test_the_twinned_slots_are_still_exactly_two` holds that fact, so
this audit's scope cannot silently narrow.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from woof.core import streaming

REGISTRY = Path(streaming.__file__).resolve().parents[1] \
    / "physics_registry_v2.json"


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _templates() -> dict:
    return _registry()["templates"]


def _selection(template: dict) -> dict:
    return template.get("selections") or template.get("components") or {}


#: registry component id -> every adapter class that component can install.
#:
#: A registry value maps to more than one class where a RUN CONFIG key picks
#: between implementations of the same component: ``ra_rrtmg_variant``
#: selects legacy RRTMG or RTE+RRTMGP under the single ``rte-rrtmgp``
#: component, which is why the default suite's radiation is legacy RRTMG
#: while its registry component reads ``rte-rrtmgp``.  Auditing only the
#: component name would therefore have missed the one broken scheme.
#:
#: Not a hand-written list of what to audit -- the audit set comes from the
#: registry -- but the bridge from a registry id to a Python class, which
#: exists nowhere else in machine-readable form.
#: :func:`test_every_registry_component_is_mapped_to_its_adapters` fails if
#: the registry grows a value this does not cover.
COMPONENT_ADAPTERS: dict[str, tuple[tuple[str, str], ...]] = {
    "rte-rrtmgp": (
        ("woof.core.rrtmgp", "RRTMGPRadiation"),
        ("woof.core.rrtmg_legacy", "RRTMGLegacyRadiation"),
    ),
    "dudhia-shortwave": (
        ("woof.core.dudhia", "DudhiaShortwaveRadiation"),
        ("woof.core.rrtm_lw", "RRTMLongwaveRadiation"),
        ("woof.core.rrtm_lw", "RRTMDudhiaRadiation"),
        ("woof.core.analytic_radiation", "AnalyticClearSkyRadiation"),
    ),
    "wrf-cam3": (("woof.core.cam3", "Cam3Radiation"),),
    "kain-fritsch": (("woof.core.kf", "KainFritsch"),),
    # The two cumulus closures that entered the TEMPLATE walk when audit
    # R-067 gave each of them a named suite.  Both were selectable before
    # -- as per-domain component overrides on the tree route -- and this
    # audit walks templates, so neither was in its scope until a suite
    # named them.  Adding the rows is the point of the mapping check: a
    # registry value with no adapter class is an unaudited scheme, and
    # these two now go through the same twinning proof as the rest.
    "grell-freitas": (("woof.core.gf", "GrellFreitas"),),
    "grell-devenyi": (("woof.core.gf", "GrellDevenyi"),),
    "new-tiedtke": (("woof.core.ntiedtke", "NewTiedtke"),),
    # New SAS is a row of the column-cumulus table, installed through the
    # one table-driven adapter (woof.core.physics, cu_physics = 96).
    "new-sas": (("woof.core.column_cumulus", "ColumnCumulus"),),
    "old-gfs-sas": (("woof.core.column_cumulus_r8", "ColumnCumulus"),),
    "wrf-rrtm-dudhia": (("woof.core.rrtm_lw", "RRTMDudhiaRadiation"),),
    "off": (),
}

#: The components that install an adapter object.  See the module docstring.
ADAPTER_COMPONENTS = ("radiation", "cumulus")


def _adapter_classes():
    """(family, component_id, module, name, class-or-exc) over the registry."""
    wanted: set[tuple[str, str]] = set()
    for template in _templates().values():
        selection = _selection(template)
        for family in ADAPTER_COMPONENTS:
            value = selection.get(family)
            if value is not None:
                wanted.add((family, str(value)))
    for family, component in sorted(wanted):
        for module_name, class_name in COMPONENT_ADAPTERS.get(component, ()):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:          # pragma: no cover
                # cupy is absent on a CPU-only checkout.  Reported, never
                # silently skipped: the node with a card runs this too and
                # there it must import.
                yield family, component, module_name, class_name, exc
                continue
            yield (family, component, module_name, class_name,
                   getattr(module, class_name))


def test_the_registry_is_readable_and_not_empty():
    """The enumeration is programmatic; prove it enumerated something."""
    templates = _templates()
    assert len(templates) >= 10, templates
    assert all(_selection(t) for t in templates.values())


def test_every_registry_component_is_mapped_to_its_adapters():
    """A registry value with no adapter mapping is an UNAUDITED scheme.

    Red when someone adds a component to the registry without saying which
    class it installs, which is the moment the audit would otherwise start
    quietly covering less than it claims.
    """
    seen = set()
    for template in _templates().values():
        selection = _selection(template)
        for component in ADAPTER_COMPONENTS:
            value = selection.get(component)
            if value is not None:
                seen.add(str(value))
    unmapped = sorted(seen - set(COMPONENT_ADAPTERS))
    assert not unmapped, (
        f"registry components {unmapped} install an adapter this audit does "
        "not know about; add them to COMPONENT_ADAPTERS (or map to () if the "
        "component installs no adapter object)")


@pytest.mark.parametrize(
    "family,component,module_name,class_name,resolved",
    [pytest.param(*row, id=f"{row[1]}::{row[3]}")
     for row in _adapter_classes()])
def test_every_shipped_adapter_can_be_twinned_per_tile_buffer(
        family, component, module_name, class_name, resolved):
    """The audit itself: every adapter the registry can install is rebuildable.

    ``twin_support`` is what ``_tile_scheme`` dispatches on, so this cannot
    drift from the behaviour it audits.  ``None`` means a tile buffer cannot
    be built for that scheme, which means every [tiles] run of every suite
    selecting it dies in TiledRun.__init__.
    """
    if isinstance(resolved, ModuleNotFoundError):
        pytest.skip(f"{module_name} needs {resolved.name}; audited on the "
                    "node with a card, where it imports")
    support = streaming.twin_support(resolved)
    assert support is not None, (
        f"{class_name} (registry component {component!r}) has no per-buffer "
        "twin route: it is not a dataclass, its constructor requires "
        "arguments, and woof.core.streaming._TWIN_RECIPES has no recipe for "
        "it.  Every [tiles] forecast selecting this scheme will raise "
        "StreamingRefused inside TiledRun.__init__.  Give it a _TWIN_RECIPES "
        "entry that reproduces its policy and takes the tile's geography.")
    assert support in (streaming.TWIN_BY_REPLACE, streaming.TWIN_BY_RECIPE,
                       streaming.TWIN_BY_RECONSTRUCTION)


@pytest.mark.parametrize(
    "family,component,module_name,class_name,resolved",
    [pytest.param(*row, id=f"{row[1]}::{row[3]}")
     for row in _adapter_classes()])
def test_an_adapter_with_per_column_geography_is_twinned_at_the_tiles_extents(
        family, component, module_name, class_name, resolved):
    """Rebuildable is not enough: the twin must take the TILE's geography.

    A dataclass adapter without ``latitude_deg``/``longitude_deg`` init
    fields is rebuilt by ``dataclasses.replace(scheme)`` with the DOMAIN's
    geography, so every buffer would carry the domain's solar zenith angle
    over its own tile.  Every radiation adapter in the tree takes per-column
    lat/lon arrays, so every one of them must expose them.
    """
    if isinstance(resolved, ModuleNotFoundError):
        pytest.skip(f"{module_name} needs {resolved.name}")
    if family == "cumulus":
        pytest.skip("cumulus adapters carry no per-column geography; their "
                    "freshness is proven by the w0avg carrier tests instead")
    names = {p.name for p in inspect.signature(resolved).parameters.values()}
    assert {"latitude_deg", "longitude_deg"} <= names, (
        f"{class_name} takes no per-column geography, so its per-buffer twin "
        "would carry the domain's")


def test_the_twinned_slots_are_still_exactly_two():
    """The audit's SCOPE, held as source.

    If a future component starts installing a third adapter object on the
    driver, this audit would still pass while covering less than it claims.
    """
    # Read rather than imported: woof.core.physics needs a CUDA build, and
    # this is a question about its SOURCE.
    src = (Path(streaming.__file__).with_name("physics.py")
           ).read_text(encoding="utf-8")
    assert "self.radiation_callable = radiation" in src
    assert "self.cumulus_callable = cumulus" in src
    assigned = {line.split("=")[0].strip()
                for line in src.splitlines()
                if line.strip().startswith("self.")
                and "_callable = " in line}
    assert assigned == {"self.radiation_callable", "self.cumulus_callable"}, (
        f"PhysicsDriver installs adapter objects {sorted(assigned)}; "
        "streaming._tile_scheme twins only radiation and cumulus, so any "
        "other one is handed to every tile buffer SHARED -- which shares its "
        "carriers across buffers")


def test_the_default_suites_radiation_is_the_one_that_needed_a_recipe():
    """Records WHICH scheme the audit caught, so the finding is not folklore.

    Legacy RRTMG is the only shipped adapter that is neither a dataclass nor
    empty-constructible, and it is the radiation of the default suite -- so
    the twin builder's gap was not an edge case, it was the default path.
    """
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation

    assert streaming.twin_support(RRTMGLegacyRadiation) == \
        streaming.TWIN_BY_RECIPE
    by_recipe = {name for _f, _c, _m, name, resolved in _adapter_classes()
                 if not isinstance(resolved, ModuleNotFoundError)
                 and streaming.twin_support(resolved)
                 == streaming.TWIN_BY_RECIPE}
    # CAM3 also needs its table root and spectral policy carried to tiles.
    # The second entry is the column-cumulus table's adapter (New SAS,
    # cu_physics = 96), whose constructor requires its row's cu_physics; this
    # audit caught it the day its suite joined the template walk.
    assert by_recipe == {"RRTMGLegacyRadiation", "ColumnCumulus", "Cam3Radiation"}, (
        f"recipes now cover {sorted(by_recipe)}; every other shipped adapter "
        "is a dataclass or empty-constructible, and a new entry here is a "
        "new scheme that could not be rebuilt without one")
