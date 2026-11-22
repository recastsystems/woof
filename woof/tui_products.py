"""Read-only plot metadata for the terminal picker; no forecast defaults change.

This module owns two small tables and one question.

The tables are the curated plot PRESETS (``plot-presets.json``) and the
packaged record of what the wrfout history lane cannot draw and why
(``research-diagnostics.json``).  The question is the one a reader has
before a forecast rather than after it: *of the products I just chose,
which will this install actually draw?*

The record is deliberately NOT a second catalog.  The renderer owns the
product vocabulary and answers availability per store; the packaged JSON
carries only two things the renderer cannot state on its own -- the
REASON a product is unserved on this lane, written for a reader, and the
WINDOW a product needs before it means anything.  A row's absence from
it therefore says nothing at all, and nothing here refuses on absence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

#: The packaged lane record.  One file, read by both doors that ask this
#: question -- the research recipe validator and the preset picker -- so
#: the two cannot disagree about one configuration.
#:
#: NAMING DEBT, recorded rather than paid: the file is still called
#: ``research-diagnostics.json`` and is no longer research-specific.
#: Renaming it would move the sha256 authority block it carries, its
#: ``MANIFEST.in`` row and ``tools/build_rw_wps_release.py``, so the name
#: stays and this note says why.
DIAGNOSTICS_PATH = (Path(__file__).parent / "data" / "tui" /
                    "research-diagnostics.json")


def presets() -> dict:
    """Small curated requests, separate from the renderer-owned catalog."""
    return json.loads((Path(__file__).parent / "data" / "tui" /
                       "plot-presets.json").read_text(encoding="utf-8"))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def lane_capabilities(path: Path | None = None) -> dict:
    """The packaged lane record: unserved reasons and window requirements.

    THE reader for that file.  ``products`` carries the window a
    recorded product needs (``minimum_hours``) and the basis it was
    recorded from; ``unavailable`` carries, per product, the concrete
    breakage that stops this lane drawing it and what to do instead.
    Both are read straight out of the package, with the file's own
    digest, so a caller can say which bytes it decided on.

    A product in neither map is not a refusal and not an error.  It is a
    product this record says nothing about, and the caller runs it.
    """

    path = DIAGNOSTICS_PATH if path is None else Path(path)

    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"),
                              object_pairs_hook=_unique_object)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read packaged lane diagnostic capabilities at {path}: "
            f"{error}. Restore the matching WOOF-ArWen package data, then retry."
        ) from error
    try:
        if not isinstance(document, dict):
            raise ValueError("the document must be a JSON object")
        if document.get("schema") != "arwen.research.diagnostics.v1":
            raise ValueError("expected schema arwen.research.diagnostics.v1")
        if not isinstance(document["products"], dict) or not document["products"]:
            raise ValueError("products must be a nonempty object")
        if not isinstance(document["unavailable"], dict):
            raise ValueError("unavailable must be an object")
        for name, entry in document["products"].items():
            minimum = entry["minimum_hours"]
            if (not isinstance(name, str) or not isinstance(entry["kind"], str)
                    or isinstance(minimum, bool)
                    or not isinstance(minimum, int) or minimum < 0):
                raise ValueError(f"invalid diagnostic capability {name}")
        for name, reason in document["unavailable"].items():
            if not isinstance(name, str) or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"unavailable {name} records no reason")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid research diagnostic capabilities at {path}: {error}. "
            "Restore the matching WOOF-ArWen package data, then retry.") from error
    return {**document, "sha256": hashlib.sha256(payload).hexdigest()}


#: How the preset block describes itself, so a reader can price it.
#: It is the packaged lane record and nothing else: the authority on
#: what one RUN can draw is that run's own store catalog, read at render
#: time against the frames the invocation will draw.
PRESET_AVAILABILITY_BASIS = (
    "the packaged lane record only. What a given run can draw is decided "
    "by that run's own store catalog at render time; a product it cannot "
    "draw is dropped before the renderer is launched and named, with the "
    "engine's own reason, in that render's summary")


def preset_availability(capabilities=None) -> dict:
    """Per preset, the products this lane will not draw, and why.

    ONE source: the packaged record, which names the products the wrfout
    lane cannot serve AND the concrete reason, in the words a reader
    needs.

    The renderer's FILELESS requirement rows used to be folded in beside
    it and are not any more, for a measured reason.  That pair reads the
    build's requirement table and its wrfout import PLAN with no file
    open, and measured on the shipped wheel against a real child it
    called sixteen of the shipped ``snow`` preset's twenty-one products
    undrawable -- ``2m_temperature`` and ``500mb_height_winds`` among
    them -- on the very run that then drew 143 pictures of exactly those
    products.  A picker that tells a reader a product will not draw,
    about products every run draws, teaches that reader to skip the
    lines that are true.  The same reading retired the same pair at the
    ``woof downscale`` door; this was its last caller.

    The measurement that holds is the store's own catalog, asked about
    the frames one invocation is about to render
    (:func:`woof.render._available_window_request`), and that is where
    a product is dropped and named.

    This block STATES, never refuses.  A preset is a curated request,
    not a promise about one install, and narrowing the curated list to
    what this box happens to draw would hide the finding instead of
    reporting it.
    """

    capabilities = lane_capabilities() if capabilities is None else capabilities
    recorded = dict(capabilities.get("unavailable") or {})
    document = {}
    for preset in presets()["presets"]:
        document[preset["id"]] = {
            slug: recorded[slug] for slug in preset["products"]
            if recorded.get(slug)}
    return document


def catalog_document() -> dict:
    """Ask the installed renderer without building, staging, or probing a GPU."""
    from woof import bridges
    from woof.runplan import render_catalog

    with bridges.inspection_only():
        catalog = render_catalog()
    capabilities = lane_capabilities()
    local = catalog.get("local_run")
    if isinstance(local, dict) and local.get("products"):
        # The picker offers what a local run can draw.  The engine's whole
        # vocabulary -- every model's products, ensemble and blend
        # families included -- stays in the document as `vocabulary`, and
        # the products a wrfout can never carry are named, with the
        # engine's reason, in `local_run.unavailable`.
        catalog = {**catalog, "vocabulary": catalog.get("products"),
                   "products": [{"name": row["name"]}
                                for row in local["products"]]}
    return {**catalog, "presets": presets(),
            "preset_availability": preset_availability(capabilities),
            "preset_availability_basis": PRESET_AVAILABILITY_BASIS,
            "lane_record_sha256": capabilities["sha256"],
            "lanes": other_lanes()}


def other_lanes() -> dict:
    """Product lanes beside the wrfout catalog, listed rather than hidden.

    The observation-grid engine draws five products from an observation
    volume, and nothing in this tree listed them anywhere a reader
    looks.  They deliberately do NOT join the wrfout catalog: that one
    takes forecast frames and would refuse all five on every file, which
    is the menu-nothing-serves failure its own door exists to prevent.
    They are their own lane, named, with what they take.
    """

    from woof import rustwx_lanes

    return {
        "observation-grid": {
            "products": list(rustwx_lanes.OBSGRID_PRODUCTS),
            "input": "a gpuwm-obs.radar-grid.v1 observation volume "
                     "(--obs FILE.nc), not a wrfout frame",
            "engine": rustwx_lanes.OBSGRID_NAME,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true",
                        help="return the installed renderer catalog as JSON")
    args = parser.parse_args(argv)
    print(json.dumps(catalog_document() if args.catalog else presets()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
