"""Every non-Python file under ``gpuwm/`` must be declared as package data.

The whole suite reads its oracle fixtures and parameter tables out of the
source tree, so a file that is missing from ``[tool.setuptools.package-data]``
still passes every test while the built wheel silently omits it.  That is not a
hypothetical: before this gate existed the declaration was a hand-maintained
enumeration that had fallen 52 files behind, including three assets the model
loads at *runtime* rather than only under test --
``data/kf_lutab/kf_lutab.npz`` (``woof.core.kf``), ``data/noahmp/*.TBL``
(``woof.core.noahmp``) and ``data/ruc/oracle/tbq.csv``
(``woof.core.ruc._load_ruc_saturation_table``).  An install of that wheel
raises ``FileNotFoundError`` the first time those schemes are used.

The check runs setuptools' own :meth:`build_py.find_data_files` against the
real ``pyproject.toml`` rather than re-implementing pattern matching, so what
it reports is what the wheel would actually contain.  It is a filesystem walk,
not a ``git ls-files`` walk, so an undeclared file trips it before it is ever
committed.

(It also walks ``mpas_cycle_bridge/``, the other top-level package the
wheel carries.  That package's worker runs from the INSTALLED copy, in an
interpreter the cycle door names, and resolves what it reads beside its own
modules -- so a data file that grows there and is not declared is dropped
from the wheel and missed by every in-tree run, which finds it through the
checkout.  It carries no data file today; the walk is what keeps the first
one from being lost.)

Why the walk stops at ``gpuwm/``
-------------------------------
``tools`` is a declared package too, but it is not a data directory: it is
mostly scripts, plus two vendored Rust workspaces whose source is *build
input* and has no business in a wheel.  A blanket "everything under ``tools``
must ship" rule would be wrong in 260 MB of ways.

That exemption is exactly how the renderer's map assets disappeared.
``tools/rustwx/assets/basemap`` holds the Natural Earth and US Census
shapefiles ``rw_wrfbatch`` draws coastlines and borders from -- runtime data
living inside a directory this file had decided not to look at -- and the one
``package-data`` entry still written as an enumeration, ``tools =
["prepare_hrrr_*.sh"]``, did not name them.  Nothing failed.  A pip install
rendered a tropical cyclone over a blank rectangle.

So the rule is not "walk more", it is "every *runtime* asset must be
delivered by some declared mechanism, and the test names which".  For the
renderer's assets that mechanism is the bridge bundle rather than the wheel
(the wheel is 74.6 MiB against PyPI's 100 MB per-file cap and the assets
deflate to 20.2 MiB), which
:func:`test_the_renderer_asset_tree_is_delivered_by_a_declared_mechanism`
checks against ``woof.bridge_assets``.  The bundle alone was not enough: a
wheel install reaches it only through ``woof fetch-bridges``, which no
install text ran, so since 2.8.0 the ``recast-woof-data`` companion carries the
renderer's layer directories as well, and
:func:`test_the_companion_wheel_ships_the_renderers_map_layers` holds its
wheel to them.
"""

from __future__ import annotations

import glob as globlib
import os
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "woof"
#: The far side of the MPAS port seam: a top-level package the wheel must
#: carry (see ``_DECLARED_TOP_LEVEL``) whose worker runs from the INSTALLED
#: copy and resolves what it reads beside its own modules.  Walked like
#: ``gpuwm/`` so that the first data file it grows cannot be dropped from
#: the wheel unnoticed -- every in-tree test would keep finding that file
#: through the checkout while an installed copy refused the configuration
#: that needs it.
BRIDGE_ROOT = REPO_ROOT / "mpas_cycle_bridge"

#: The companion distribution: one repository, two pyproject.toml files.
#: ``recast-woof-data`` carries the RRTMGP and Thompson table directories since
#: 2.5.0, because the ``woof`` wheel measured 103.62 MiB against PyPI's
#: 100 MiB per-file cap.  This file gates BOTH, and it has to: the moment a
#: directory moved out of ``gpuwm/``, a walk that stops at ``gpuwm/`` would
#: have reported "all declared" over an empty result -- exactly the
#: vacuous-green shape the renderer's map assets went dark in.
COMPANION_ROOT = REPO_ROOT / "recast-woof-data"
COMPANION_PYPROJECT = COMPANION_ROOT / "pyproject.toml"
COMPANION_PACKAGE_ROOT = COMPANION_ROOT / "woof_data"

#: Extensions that ``[tool.setuptools.packages.find]`` already ships as modules,
#: plus their compiled droppings.  Everything else under ``gpuwm/`` is data.
_MODULE_SUFFIXES = (".py", ".pyc", ".pyo")

#: Directories that never belong in a distribution at all.
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})


def _require_source_tree() -> None:
    if not PYPROJECT.is_file() or not PACKAGE_ROOT.is_dir():
        pytest.skip("packaging gate needs the source tree, not an install")


def _require_companion_tree() -> None:
    if not COMPANION_PYPROJECT.is_file() or not COMPANION_PACKAGE_ROOT.is_dir():
        pytest.skip("packaging gate needs the source tree, not an install")


#: The ONLY files that may be deliberately absent from the wheel are the
#: size-externalized class below.  Anything else excluded from the wheel is
#: a packaging bug, and the list going stale is one too.
#:
#: There used to be a license-driven class as well: the four
#: CC-BY-NC-SA-4.0 RFMIP reference results, kept in the repository as test
#: fixtures and excluded from both artifacts.  Since 2.8.0 no RFMIP file is
#: in the repository at all (the tests fetch them from upstream,
#: woof/core/rfmip_upstream.py), so the class retired with the files and
#: :func:`test_no_rfmip_file_is_carried` holds the tree to it.
#:
#: Repo-relative, and it moved with its directory: the class below names
#: files in the COMPANION distribution, because that is where the
#: thompson/tables directory ships since 2.5.0.  The `woof` wheel
#: excludes nothing at all now, which
#: :func:`test_wheel_exclusions_are_exactly_the_pinned_lists` states
#: directly rather than leaving as an absence.

#: Size-driven externalized assets: published as GitHub release assets
#: because they exceed distribution-channel limits, staged by
#: ``woof fetch-tables`` under the thompson_contract size+SHA-256 pins.
#: Unlike the license class these MAY be absent from a checkout
#: (freezeH2O.dat is not in the public repository at all); when present
#: they still must not ship in the wheel or the sdist.  Pinned against
#: woof.table_assets and MANIFEST.in so the packaging exclusions and
#: the fetch contract cannot drift apart.
_EXTERNALIZED_FROM_WHEEL = frozenset({
    "recast-woof-data/woof_data/data/thompson/tables/freezeH2O.dat",
    "recast-woof-data/woof_data/data/thompson/tables/qr_acr_qg_V4.dat",
})

#: Third-party data this repository redistributes verbatim from a WRF
#: release, which therefore must SHIP rather than be excluded.
#:
#: ``CCN_ACTIVATE.BIN`` is the aerosol-aware Thompson (mp_physics=28)
#: activation table.  It is not generated by ``thompson_init`` and no
#: recompilation of WRF reproduces it -- ``table_ccnAct``
#: (phys/module_mp_thompson.F:5110-5166) only READS it, and the numbers are
#: offline parcel-model output (WRF's own comment at :5102-5108).  Until
#: 2026-08-01 the port did not redistribute it and ``pyproject.toml`` carried
#: an explicit ``exclude-package-data`` entry to keep a wheel built on a
#: machine that had staged it from publishing it.  That decision was
#: reversed: the committed copy is WRF v4.6.1's ``run/CCN_ACTIVATE.BIN`` bit
#: for bit, WRF's ``LICENSE.txt`` is a public-domain dedication, and the
#: notice ships in ``woof/data/wrf_radiation/LICENSE-WRF.txt``.  The pin
#: below is therefore inverted: these files must reach the wheel, and the
#: exclusion entry must be gone.  See ``woof/data/thompson/PROVENANCE.md``
#: and ``thompson_aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED``.
_REDISTRIBUTED_WRF_DATA = frozenset({
    "recast-woof-data/woof_data/data/thompson/tables/CCN_ACTIVATE.BIN",
})


def _build_system_requires() -> list[str]:
    """What this project declares it needs in order to be built at all."""

    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    return list(config.get("build-system", {}).get("requires", []))


def _package_data_declaration(pyproject: Path = PYPROJECT
                              ) -> dict[str, list[str]]:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"]["package-data"]


def _exclude_package_data_declaration(pyproject: Path = PYPROJECT
                                      ) -> dict[str, list[str]]:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"].get("exclude-package-data", {})


def _discover_packages(package_root: Path = PACKAGE_ROOT,
                       project_root: Path = REPO_ROOT) -> list[str]:
    """Import names of every package under ``package_root``.

    Parameterised over the project because there are two distributions in
    this repository now (``woof`` and the ``recast-woof-data`` companion), and
    the whole value of this file is that it asks setuptools rather than
    restating glob semantics -- which means it has to ask setuptools the
    same question once per project.
    """

    packages: list[str] = []
    for dirpath, dirnames, filenames in os.walk(package_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        if "__init__.py" not in filenames:
            continue
        relative = Path(dirpath).relative_to(project_root)
        packages.append(".".join(relative.parts))
    return packages


def _files_on_disk(package_root: Path = PACKAGE_ROOT,
                   anchor: Path = REPO_ROOT) -> set[str]:
    """Repo-relative POSIX paths of every data file under ``package_root``."""

    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(package_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            if name.endswith(_MODULE_SUFFIXES):
                continue
            path = Path(dirpath, name).relative_to(anchor)
            present.add(path.as_posix())
    return present


def _files_setuptools_would_ship(monkeypatch: pytest.MonkeyPatch,
                                 *, name: str = "woof",
                                 project_root: Path = REPO_ROOT,
                                 package_root: Path = PACKAGE_ROOT,
                                 pyproject: Path = PYPROJECT,
                                 anchor: Path | None = None) -> set[str]:
    """Ask setuptools itself which files the declaration selects.

    ``build_py.get_data_files_without_manifest`` is the exact code path that
    populates a wheel, so this measures the declaration rather than trusting a
    private re-implementation of glob semantics.

    An absent setuptools is a FAILURE here, not a skip.  It used to be
    ``pytest.importorskip("setuptools")``, and the 2026-08-13 test-estate
    audit measured the consequence: this file is on
    ``tools/battery/stage1_files.txt`` and was **56% dark** on the assembly
    venvs -- 4 passed, 5 skipped -- and the five that skipped are exactly
    the wheel-content assertions the file exists for (every data file
    declared, the exclusion lists, the renderer asset tree, the runtime
    assets, the Thompson table).  That is the same class as the defect
    ``fix(render): the map assets ship with the renderer that reads them``
    closed, and a stage-1 entry cannot be allowed to report green while its
    reason for existing is switched off.

    A skip would be defensible if setuptools were optional.  It is not:
    ``[build-system] requires`` in this project's own pyproject.toml names
    ``setuptools>=77``, so any environment that can build this project has
    it, and one that does not cannot answer the question this file asks.
    The remedy is one line -- ``pip install setuptools`` -- and the message
    below says so.
    """

    try:
        import setuptools
    except ModuleNotFoundError as exc:      # pragma: no cover - env defect
        requires = _build_system_requires()
        assert any(r.replace("_", "-").lower().startswith("setuptools")
                   for r in requires), (
            "setuptools is absent AND [build-system] requires no longer names "
            f"it ({requires}); this file's premise has changed, so decide "
            "deliberately whether the wheel-content assertions still apply")
        raise AssertionError(
            "setuptools is not installed, so the five wheel-content "
            "assertions in this file cannot run -- and this file is on "
            "tools/battery/stage1_files.txt, where a silent skip reports "
            f"green.  pyproject.toml's [build-system] requires {requires}, so "
            "every environment that can build this project has it.  Remedy: "
            "pip install setuptools (or install the project with "
            "'pip install -e .[dev,render]' in a venv built with it)."
        ) from exc

    from setuptools.command.build_py import build_py

    packages = _discover_packages(package_root, project_root)
    assert package_root.name in packages, (
        f"package discovery is broken for {name}: {packages}")

    distribution = setuptools.dist.Distribution({
        "name": name,
        "packages": packages,
        "package_data": _package_data_declaration(pyproject),
        "exclude_package_data": _exclude_package_data_declaration(pyproject),
    })
    command = build_py(distribution)
    command.finalize_options()

    # setuptools globs relative to the working directory.
    monkeypatch.chdir(project_root)

    # Results come back project-relative; re-anchor them on the repository
    # so both distributions' answers are comparable to `_files_on_disk`.
    prefix = (project_root.relative_to(anchor).as_posix() + "/"
              if anchor is not None and project_root != anchor else "")
    shipped: set[str] = set()
    for _package, src_dir, _build_dir, filenames in (
        command.get_data_files_without_manifest()
    ):
        for filename in filenames:
            shipped.add(prefix + Path(src_dir, filename).as_posix())
    return shipped


def test_the_wheel_content_assertions_cannot_go_dark_by_skipping() -> None:
    """The premise behind refusing to skip on an absent setuptools.

    The 2026-08-13 test-estate audit measured this file at 4 passed / 5
    skipped on the assembly venvs, and the five that skipped are the whole
    point of the file.  ``_files_setuptools_would_ship`` now raises instead,
    which is only defensible while setuptools really is a build requirement
    of this project rather than an optional extra.  This is that check: if
    the build backend ever moves, the assertion below fails and whoever
    moves it has to decide what these five assertions become, instead of
    silently inheriting a skip.
    """

    requires = _build_system_requires()
    assert requires, "pyproject.toml declares no [build-system] requires"
    assert any(r.replace("_", "-").lower().startswith("setuptools")
               for r in requires), (
        f"[build-system] requires is {requires} and no longer names "
        "setuptools, so 'every environment that can build this project has "
        "setuptools' has stopped being true.  _files_setuptools_would_ship "
        "raises on an absent setuptools on the strength of that sentence")


def test_every_data_file_under_gpuwm_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No data file may reach ``gpuwm/`` without ``package-data`` covering it."""

    _require_source_tree()
    present = _files_on_disk()
    assert present, "found no data files under gpuwm/ -- the walk is broken"

    shipped = _files_setuptools_would_ship(monkeypatch)
    # The verification oracles ship in gpuwm-kernels (3.0.0); their exclusion
    # is pinned by test_wheel_exclusions_are_exactly_the_pinned_lists.
    from woof import data_assets
    companion = {name for name in present
                 if any(name.startswith(f"woof/data/{tree}/")
                        for tree in data_assets.VERIFY_ORACLE_TREES)}
    undeclared = sorted(present - shipped - companion)
    assert not undeclared, (
        f"{len(undeclared)} data file(s) under gpuwm/ would be omitted from a "
        "wheel because [tool.setuptools.package-data] in pyproject.toml does "
        "not cover them. Widen an existing glob (do not append a filename):\n  "
        + "\n  ".join(undeclared)
    )


def test_every_data_file_under_the_bridge_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule for ``mpas_cycle_bridge/``, whose worker runs installed.

    The shape of the breakage this forecloses: the worker is invoked from
    site-packages, in the interpreter the cycle door names, and resolves
    what it reads beside its own file.  A wheel that drops such a file
    makes the installed copy refuse the configuration that needs it, while
    the whole in-tree suite stays green because it finds the file through
    the checkout.  The bridge carries no data file yet, so today this
    passes over an empty set -- which is the point: the assertion is in
    place before the first file, not after it is lost.
    """

    _require_source_tree()
    present = _files_on_disk(BRIDGE_ROOT, REPO_ROOT)

    shipped = _files_setuptools_would_ship(monkeypatch, package_root=BRIDGE_ROOT)
    undeclared = sorted(present - shipped)
    assert not undeclared, (
        f"{len(undeclared)} data file(s) under mpas_cycle_bridge/ would be "
        "omitted from a wheel because [tool.setuptools.package-data] in "
        "pyproject.toml does not cover them, and the installed worker reads "
        "them beside itself. Widen an existing glob (do not append a "
        "filename):\n  " + "\n  ".join(undeclared)
    )


def _companion_shipped(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    return _files_setuptools_would_ship(
        monkeypatch, name="recast-woof-data", project_root=COMPANION_ROOT,
        package_root=COMPANION_PACKAGE_ROOT, pyproject=COMPANION_PYPROJECT,
        anchor=REPO_ROOT)


def test_every_data_file_in_the_companion_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule, for the distribution the big tables moved into.

    Without this twin the split would have created exactly the hole this
    file was written to close: ``rrtmgp/`` and ``thompson/tables/`` left
    ``gpuwm/``, the walk above stopped finding them, and it would have
    reported "all declared" while a mis-declared companion shipped an
    empty wheel and every radiation run died at table load.
    """

    _require_companion_tree()
    present = _files_on_disk(COMPANION_PACKAGE_ROOT, REPO_ROOT)
    assert present, "found no data files in the companion -- walk is broken"

    shipped = _companion_shipped(monkeypatch)
    undeclared = sorted(present - shipped - _EXTERNALIZED_FROM_WHEEL)
    assert not undeclared, (
        f"{len(undeclared)} data file(s) in recast-woof-data/ would be omitted "
        "from its wheel because [tool.setuptools.package-data] in "
        "recast-woof-data/pyproject.toml does not cover them. Widen an existing "
        "glob (do not append a filename):\n  " + "\n  ".join(undeclared)
    )


def test_the_moved_directories_left_the_gpuwm_wheel_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither directory may ship in BOTH wheels, or in neither.

    The breakage on each side is concrete.  Shipping in both puts the
    64.21 MiB back into the ``woof`` wheel and PyPI rejects the upload at
    100 MiB -- which is the whole reason the companion exists.  Shipping
    in neither is worse and quieter: ``pip install recast-woof`` would succeed
    and then fail at the first radiation table read.

    Driven from ``woof.data_assets.COMPANION_TREES`` rather than a
    literal list, so moving the next directory out needs one entry there
    and no edit here.
    """

    _require_source_tree()
    _require_companion_tree()
    from woof import data_assets

    gpuwm_shipped = _files_setuptools_would_ship(monkeypatch)
    companion_shipped = _companion_shipped(monkeypatch)
    def under(name: str, root: str) -> bool:
        # An entry is a directory or, since 2.8.7, a single file whose
        # directory keeps other members in the woof wheel; the resolver's
        # own prefix rule (data_assets._is_companion) covers both.
        return name == root or name.startswith(root + "/")

    for tree in data_assets.COMPANION_TREES:
        stale = sorted(name for name in gpuwm_shipped
                       if under(name, f"woof/data/{tree}"))
        assert not stale, (
            f"the woof wheel still ships {len(stale)} file(s) under "
            f"woof/data/{tree}/, which woof.data_assets says the "
            f"recast-woof-data companion owns:\n  " + "\n  ".join(stale))
        carried = [name for name in companion_shipped
                   if under(name, f"recast-woof-data/woof_data/data/{tree}")]
        assert carried, (
            f"woof.data_assets.COMPANION_TREES names {tree!r} but the "
            "companion wheel would carry no file under it, so every "
            "consumer of that directory fails at load on a fresh install")


def test_wheel_exclusions_are_exactly_the_pinned_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the size-externalized assets.

    An exclusion pattern that silently widened would strip runtime data
    from the wheel.  The externalized assets may legitimately be absent
    from a checkout (the public repository ships them as release assets);
    when present they still must not reach the wheel.

    There is no third, redistribution-driven class any more.
    ``CCN_ACTIVATE.BIN`` was the only member and it ships now, so it must
    appear on the OTHER side of this comparison -- a re-added exclusion for
    it fails here as an unexpected exclusion.
    """

    _require_source_tree()
    _require_companion_tree()

    # The woof wheel excludes NOTHING now.  Both classes named files under
    # the two directories that moved to the companion in 2.5.0, so an
    # exclusion appearing here again means either a file came back without
    # its reasoning or somebody stripped runtime data from the wheel.
    # Since 3.0.0 the one exception is the verification-oracle class
    # (woof.data_assets.VERIFY_ORACLE_TREES), which the gpuwm-kernels
    # companion carries; anything else excluded is runtime data stripped.
    from woof import data_assets

    gpuwm_excluded = _files_on_disk() - _files_setuptools_would_ship(monkeypatch)
    oracle_files = {name for name in _files_on_disk()
                    if any(name.startswith(f"woof/data/{tree}/")
                           for tree in data_assets.VERIFY_ORACLE_TREES)}
    assert gpuwm_excluded == oracle_files, (
        "the woof wheel's exclusions are not exactly the verification "
        "oracles woof.data_assets.VERIFY_ORACLE_TREES moves to "
        "gpuwm-kernels:\n  unexpectedly excluded: "
        + "\n  ".join(sorted(gpuwm_excluded - oracle_files))
        + "\n  oracle files still shipped: "
        + "\n  ".join(sorted(oracle_files - gpuwm_excluded)))

    present = _files_on_disk(COMPANION_PACKAGE_ROOT, REPO_ROOT)
    shipped = _companion_shipped(monkeypatch)
    excluded = present - shipped
    expected = set(_EXTERNALIZED_FROM_WHEEL) & present
    assert excluded == expected, (
        "companion wheel exclusions drifted from the pinned lists:\n"
        f"  unexpectedly excluded: {sorted(excluded - expected)}\n"
        f"  expected but shipped/missing: {sorted(expected - excluded)}"
    )
    # The licence of the one CC-BY-SA-4.0 member ships beside it, so
    # PROVENANCE.md stays resolvable inside an installed wheel.
    assert ("recast-woof-data/woof_data/data/rrtmgp/LICENSE-CC-BY-SA-4.0"
            in shipped)
    assert ("recast-woof-data/woof_data/data/rrtmgp/"
            "rrtmgp-trace-gas-climatology.json" in shipped)
    # ...and the redistributed WRF table must be on the SHIPPED side.
    for relative in sorted(_REDISTRIBUTED_WRF_DATA):
        assert relative in shipped, relative


def test_externalized_assets_match_the_fetch_contract() -> None:
    """The packaging pin and woof.table_assets must name the same files,
    and every one must carry a size+SHA-256 pin in thompson_contract."""

    from woof.core.thompson_contract import CLASSIC_TABLE_ASSETS
    from woof.table_assets import EXTERNALIZED_TABLE_FILENAMES

    pinned_names = {Path(p).name for p in _EXTERNALIZED_FROM_WHEEL}
    assert pinned_names == set(EXTERNALIZED_TABLE_FILENAMES)
    contract_names = {asset.filename for asset in CLASSIC_TABLE_ASSETS}
    assert set(EXTERNALIZED_TABLE_FILENAMES) <= contract_names
    # MANIFEST.in must exclude exactly the same files from the sdist
    # (exclude-package-data governs the wheel only).  The companion's, now:
    # the directory holding both files moved there in 2.5.0, and so did the
    # exclusions.  The root MANIFEST.in must be empty of them for the same
    # reason -- an exclude naming a path that no longer exists is a rule
    # nobody is enforcing.
    companion_manifest = (COMPANION_ROOT / "MANIFEST.in").read_text(
        encoding="utf-8")
    manifest_excludes = {
        "recast-woof-data/" + line.split(None, 1)[1].strip()
        for line in companion_manifest.splitlines()
        if line.startswith("exclude ")
    }
    assert manifest_excludes == set(_EXTERNALIZED_FROM_WHEEL)
    root_manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    rules = [line for line in root_manifest.splitlines()
             if line.startswith(("exclude ", "prune "))]
    # What this test is actually about is the size- and license-driven
    # EXCLUDES: every one of them moved to recast-woof-data/MANIFEST.in with the
    # directory it named, and one left behind here would be a rule nobody
    # is enforcing.  The `exclude` lines that remain name development tests
    # RELEASE-EXCLUDE.txt already drops from the public tree: the n5s
    # toolchain test (its campaign harness, tools/n5s/**, is not a package
    # and never shipped) and the branch-dispositions gate (its ledger,
    # docs/branch-dispositions.md, is branch bookkeeping of the development
    # tree), and the three tests that import the 2.8.9 speed-lane bench
    # harness (tools/bench/**, RELEASE-EXCLUDE'd with them; without the
    # MANIFEST.in rows the sdist shipped them without the harness they
    # import).
    # The private config receipts are also snapshot exclusions. A direct
    # worktree build inherits them from configs' package-data unless the
    # sdist explicitly excludes them too.
    # tests/test_operational_package_excludes_probes.py measures that the
    # sdist really lacks the development probes.
    if (REPO_ROOT / "RELEASE-EXCLUDE.txt").is_file():
        root_excludes = [rule for rule in rules if rule.startswith("exclude ")]
        assert root_excludes == [
            "exclude tests/test_n5s_toolchain.py",
            "exclude tests/test_branch_dispositions.py",
            "exclude tests/test_regional_packing_bench.py",
            "exclude tests/test_timeline_qualification.py",
            "exclude configs/da_rerun_286/recent/active-241/case-plan.json",
            "exclude configs/da_rerun_286/recent/active-241/authority/authority-receipt.json",
            "exclude configs/da_rerun_286/recent/large-601-center32p2/case-plan.json",
            "exclude configs/da_rerun_286/recent/large-601-center32p2/authority/authority-receipt.json",
            "exclude configs/da_rerun_286/recent/requested-601-center31/case-plan.json",
            "exclude configs/da_rerun_286/recent/requested-601-center31/authority/authority-receipt.json",
            "exclude configs/da_rerun_286/recent/native-linux-pricing.json",
            "exclude configs/da_rerun_286/recent/native-linux-pricing-241.json",
            # The announcement draft: swept in by `recursive-include docs/public
            # *.md`, and it carries the project's own attribution text, which the
            # packet's attribution gate refuses in a public asset (2.9.2 cut).
            "exclude docs/public/ANNOUNCEMENT-DRAFT.md",
        ], (
            "the root MANIFEST.in's `exclude` lines drifted.  Size- and "
            "license-driven excludes belong in recast-woof-data/MANIFEST.in with the "
            "directory they name (moved there in 2.5.0); the only root excludes "
            "are RELEASE-EXCLUDE'd development tests and private config receipts; "
            "found {}".format(rules))
        release_exclude = {
            line.strip() for line in
            (REPO_ROOT / "RELEASE-EXCLUDE.txt").read_text(
                encoding="utf-8").splitlines()}
        # The one page the public tree keeps and the source archive does not
        # ship, with its reason: the repository is where a draft is worked on
        # (the claim-backing gate and the resolved-scale receipt read it
        # there); a PyPI asset is permanent, and the packet's attribution gate
        # refuses the project's own attribution text in one (2.9.2 cut).
        sdist_only = {"docs/public/ANNOUNCEMENT-DRAFT.md"}
        for rule in root_excludes:
            path = rule.split(None, 1)[1].strip()
            assert path in release_exclude or path in sdist_only, (
                f"root MANIFEST.in excludes {path} from the sdist but "
                "RELEASE-EXCLUDE.txt keeps it in the public tree, so the two "
                "release surfaces disagree about whether it ships")
    # The prunes keep other artifacts and development-only probes out of
    # the operational source distribution:
    #
    #   recast-woof-data          -- the companion distribution's source.  A
    #     fragment of it in woof's own sdist is worse than either whole
    #     answer: `woof_data/__init__.py` with none of its data is a
    #     package that says where the tables are and has none.
    #   woof/libexec/bridges -- the staged Rust binaries, which are wheel
    #     payload.  Measured on the 2.5.0 Linux shakeout: an sdist built
    #     after staging swept the staged native ELF executables in -- the
    #     18 declared at that commit -- for +19.67 MB.
    #   tilestream/skeptic -- one-shot fault-injection/remote mutation
    #     helpers. The public/Build68 package did not carry these scripts.
    #     (The six skeptic_*.py MODULES beside tilestream's code are the
    #     `recursive-exclude` line, measured by
    #     tests/test_operational_package_excludes_probes.py.)
    #
    # Pin the complete set so exclusions cannot drift silently.
    assert {rule for rule in rules if rule.startswith("prune ")} == {
        "prune recast-woof-data", "prune gpuwm-kernels", "prune gpuwm-kernels-startup",
        "prune woof/libexec/bridges", "prune tilestream/skeptic", "prune configs/da_rerun_286",
        # 3.0.0 SIZE-01: the native oracle fixtures stay in the repository.
        "prune tests/data/oracles"}, (
        "the root MANIFEST.in's prunes drifted: keep the companion source, "
        "staged platform binaries and development fault-injection probes "
        f"out of the operational source distribution; found {rules}")


def test_the_sdist_ships_the_advertised_installers_and_public_pages(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """D1: an extracted source archive must supply its install and docs doors."""
    import re
    from setuptools._distutils.filelist import FileList
    from tools.release_exclusions import matches, read_exclusions

    monkeypatch.chdir(REPO_ROOT)
    docs = {p.as_posix() for p in Path("docs").rglob("*") if p.is_file()}
    expected = {"install.sh", "install.ps1"}
    # Drafts are not documentation (the announcement draft carries the
    # project's own attribution text, refused by the packet's attribution
    # gate); the CHANGELOG's links are history, not advertised doors.
    expected.update(p for p in docs if p.startswith("docs/public/")
                    and p.endswith(".md") and not p.endswith("-DRAFT.md"))
    for source in ("README.md",):
        for ref in re.findall(r"docs/[A-Za-z0-9_./-]+",
                              Path(source).read_text(encoding="utf-8")):
            ref = ref.rstrip(".")
            path = Path(ref)
            assert path.exists(), f"{source} names a missing document: {ref}"
            if path.is_dir():
                expected.update(p.as_posix() for p in path.rglob("*.md"))
            else:
                expected.add(ref)

    rules = (read_exclusions(REPO_ROOT)
             if (REPO_ROOT / "RELEASE-EXCLUDE.txt").is_file() else [])
    # The top-level pages RELEASE-EXCLUDE.txt keeps private, offered to the
    # manifest so the privacy assertion below is a measurement.
    private_pages = {rule for rule in rules
                     if "/" not in rule and rule.endswith(".md")}
    files = FileList()
    files.allfiles = [os.fspath(Path(name)) for name in sorted(
        docs | {"install.sh", "install.ps1"} | private_pages)]
    for line in Path("MANIFEST.in").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            files.process_template_line(line)
    included = {name.replace(os.sep, "/") for name in files.files}
    assert not expected - included, (
        f"sdist omits advertised installers or pages: {sorted(expected - included)}")
    assert not [name for name in included if matches(name, rules)], (
        "the sdist documentation includes RELEASE-EXCLUDE'd private files")
    drafts = sorted(name for name in included if name.endswith("-DRAFT.md"))
    assert not drafts, f"the sdist ships a draft: {drafts}"


def test_the_renderer_asset_tree_is_delivered_by_a_declared_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No runtime asset under ``tools/`` may be delivered by nothing at all.

    The gap this closes: the wheel-coverage walk above deliberately stops at
    ``gpuwm/``, so a data directory under ``tools/`` could -- and did -- reach
    an installed user by no mechanism whatever, silently, and the only symptom
    was a plot with no geography on it.

    Every file under the renderer's asset root must therefore be accounted
    for by one of the two mechanisms that exist: the wheel's package-data, or
    the bridge bundle.  A new asset subdirectory that nobody added to
    ``bridge_assets.REQUIRED_ASSET_SUBDIRS`` is delivered by neither, and
    fails here by name rather than months later in an image.
    """

    _require_source_tree()
    from woof import bridge_assets

    asset_root = REPO_ROOT / "tools" / "rustwx" / "assets"
    if not asset_root.is_dir():
        pytest.skip("no vendored renderer assets in this tree")

    present: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(asset_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in filenames:
            present.add(
                Path(dirpath, name).relative_to(REPO_ROOT).as_posix())
    assert present, "found no renderer assets -- the walk is broken"

    carried_by_bundle = {
        path for path in present
        if Path(path).relative_to(
            asset_root.relative_to(REPO_ROOT)).parts[0]
        in bridge_assets.REQUIRED_ASSET_SUBDIRS
    }
    shipped_in_wheel = _files_setuptools_would_ship(monkeypatch)
    undelivered = sorted(present - carried_by_bundle - shipped_in_wheel)
    assert not undelivered, (
        f"{len(undelivered)} renderer asset file(s) would reach an installed "
        "user by no mechanism at all -- not the wheel's "
        "[tool.setuptools.package-data], and not the bridge bundle's "
        "woof.bridge_assets.REQUIRED_ASSET_SUBDIRS. A renderer without them "
        "draws plots with no coastlines or borders:\n  "
        + "\n  ".join(undelivered))

    # The declaration must also still name something real: a subdirectory
    # renamed on disk leaves REQUIRED_ASSET_SUBDIRS pointing at nothing, and
    # the bundle would be packed empty.
    for subdir in bridge_assets.REQUIRED_ASSET_SUBDIRS:
        candidate = asset_root / subdir
        assert candidate.is_dir(), (
            f"bridge_assets.REQUIRED_ASSET_SUBDIRS names {subdir!r}, which is "
            f"not a directory at {candidate}")
        assert any(p.is_file() for p in candidate.rglob("*")), (
            f"{candidate} holds no files; the bundle would carry no basemaps")


#: Patterns whose directory is BUILT, not authored, so an empty match in a
#: source checkout is the correct state rather than drift.
#:
#: Exactly one entry, and it is not a loophole: the prebuilt Rust a platform
#: wheel carries is staged into ``woof/libexec/bridges`` by
#: ``tools/stage_wheel_bridges.py`` between ``cargo build`` and
#: ``python -m build``, and the directory is gitignored.  A checkout that has
#: not been through that step legitimately has nothing there -- which is also
#: the state that produces the ``py3-none-any`` fallback wheel.
#:
#: The exemption is only for EMPTINESS.  When the directory does exist it must
#: carry files, so a half-staged or hand-emptied tree still fails here; that a
#: staged tree really produces a wheel carrying all eleven artifacts is proved
#: separately, by ``tests/test_wheel_bridge_staging.py``.
#:
#: The release cut does NOT stage it.  It publishes the sdist and the single
#: universal ``py3-none-any`` wheel this project has shipped 37 versions of,
#: and proves that wheel carries NO staged artifacts; the prebuilt Rust
#: reaches users as the per-platform release-asset bundles ``woof setup``
#: stages.  So this pattern matching nothing in CI is the normal state, not a
#: half-built release.
_STAGED_AT_RELEASE: set[tuple[str, str]] = {("woof", "libexec/bridges/*"),
                                            # the hex helpers, staged by the assembly's stage_hex_helpers.py
                                            # for a platform or ops wheel; a PyPI wheel fetches them on use
                                            ("woof.hex", "libexec/*")}


def test_no_package_data_pattern_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pattern that matches nothing is drift -- it names a moved directory."""

    _require_source_tree()
    monkeypatch.chdir(REPO_ROOT)

    declaration = _package_data_declaration()
    dead: list[str] = []
    declared_staged: set[tuple[str, str]] = set()
    for package, patterns in declaration.items():
        package_dir = REPO_ROOT.joinpath(*package.split("."))
        for pattern in patterns:
            staged = (package, pattern) in _STAGED_AT_RELEASE
            if staged:
                declared_staged.add((package, pattern))
                # Absent is fine; present-but-empty is not.
                staged_dir = package_dir / Path(pattern).parent
                if not staged_dir.is_dir():
                    continue
                if not any(child.is_file() for child in staged_dir.iterdir()):
                    dead.append(
                        f"{package}: {pattern} -- the staging directory exists "
                        f"but is empty, which is a half-built wheel rather "
                        f"than an unstaged checkout")
                continue
            if not package_dir.is_dir():
                dead.append(f"{package}: <no such package dir> ({pattern})")
                continue
            matches = globlib.glob(
                pattern, root_dir=package_dir, recursive=True
            )
            if not any((package_dir / m).is_file() for m in matches):
                dead.append(f"{package}: {pattern}")

    assert not dead, (
        "package-data pattern(s) match no file on disk; the data they named has "
        "moved or been deleted:\n  " + "\n  ".join(dead)
    )
    # The exemption list must not outlive the pattern it excuses.
    stale = _STAGED_AT_RELEASE - declared_staged
    assert not stale, (
        f"_STAGED_AT_RELEASE excuses {sorted(stale)}, which pyproject.toml no "
        f"longer declares; drop the entry rather than leaving a standing "
        f"exemption for a pattern that is gone")


def test_runtime_data_assets_survive_the_globs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the assets whose absence breaks an *installed* model, not just tests.

    The coverage test above is only as strong as the filesystem it walks. These
    three are loaded by physics code at run time, so an installed wheel without
    them is broken in the field rather than in CI -- worth failing loudly and by
    name if a future rewrite of the globs drops them.
    """

    _require_source_tree()
    _require_companion_tree()
    # The union of both wheels, which is what `pip install recast-woof` puts on
    # disk: the companion is a hard `==` dependency, so a runtime asset is
    # delivered if EITHER distribution carries it, and the reader of this
    # list should not have to know which.  The prefixes still say which,
    # so a file moving between them is a visible edit here.
    shipped = (_files_setuptools_would_ship(monkeypatch)
               | _companion_shipped(monkeypatch))
    for relative in (
        "woof/data/kf_lutab/kf_lutab.npz",
        "woof/data/morrison/constants.toml",
        "recast-woof-data/woof_data/data/thompson/tables/qr_acr_qsV2.dat",
        "recast-woof-data/woof_data/data/thompson/tables/thompson_aux_tables.dat",
        "recast-woof-data/woof_data/data/thompson/tables/MANIFEST.sha256",
        "woof/data/noahmp/MPTABLE.TBL",
        "woof/data/noahmp/SOILPARM.TBL",
        "woof/data/noahmp/GENPARM.TBL",
        "woof/data/noah_tables/VEGPARM.TBL",
        "woof/data/ruc/oracle/tbq.csv",
        "recast-woof-data/woof_data/data/rrtmgp/rrtmgp-gas-lw-g256.nc",
        "woof/data/wrf_radiation/RRTM_DATA",
    ):
        assert (REPO_ROOT / relative).is_file(), f"fixture vanished: {relative}"
        assert relative in shipped, (
            f"{relative} is loaded at run time but would not ship in a wheel"
        )
    # The externalized tables are loaded at run time too, but deliberately
    # NOT via the wheel: they are release assets staged by
    # `woof fetch-tables`, so the pin here is inverted -- they must never
    # ship even when present in the tree.
    for relative in sorted(_EXTERNALIZED_FROM_WHEEL):
        assert relative not in shipped, (
            f"{relative} is externalized and must not ship in the wheel"
        )


def test_no_shipped_data_file_is_swallowed_by_gitignore() -> None:
    """A shipped data file that .gitignore matches is a sprung trap, not a
    hypothetical.

    The first public-tree CI run (32280494350) paid for it: the release
    snapshot carries this repository's .gitignore, whose repo-wide
    ``*.nc`` rule (written for model OUTPUT) matched the companion's nine
    ``data/rrtmgp/*.nc``.  In the private repository they are tracked --
    force-added once, so nothing here ever noticed -- but any fresh
    ``git add`` of the snapshot silently drops them, and the public
    repository was born without its k-distributions: the companion wheel
    built from that checkout imports, version-checks, and dies at the
    first radiation table load.

    So the rule: every data file the two wheels ship must be invisible
    to .gitignore.  ``--no-index``, because check-ignore consults the
    index by default and a TRACKED-but-ignored file -- the exact trap --
    reports clean without it.  The remedy is a negation beside the rule
    that matches (the crate goldens' ``!*.nc`` is the precedent), never
    untracking the file.
    """

    _require_source_tree()
    _require_companion_tree()
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no .git here, so there is no ignore machinery to "
                    "spring; the gate runs in every real checkout")
    import subprocess

    candidates = sorted(_files_on_disk()
                        | _files_on_disk(COMPANION_PACKAGE_ROOT, REPO_ROOT))
    assert candidates, "found no data files at all -- the walk is broken"
    # NUL-terminated bytes, deliberately: a text-mode pipe on Windows
    # rewrites "\n" to "\r\n" on the way in, git then finds no path that
    # ends in a carriage return, and the gate reports clean over a probe
    # that measured nothing -- the exact vacuous green it exists to end.
    probe = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=str(REPO_ROOT),
        input="\0".join(candidates).encode("utf-8"),
        capture_output=True)
    # 0 = some matched, 1 = none matched; anything else is git failing.
    assert probe.returncode in (0, 1), probe.stderr.decode(errors="replace")
    swallowed = [entry for entry in
                 probe.stdout.decode("utf-8").split("\0") if entry]
    assert not swallowed, (
        f"{len(swallowed)} shipped data file(s) are matched by .gitignore, "
        "so a fresh `git add` of a release snapshot drops them and the "
        "published tree ships a companion that dies at table load.  Add a "
        "negation beside the rule that matches (see the crate goldens' "
        "!*.nc for the shape):\n  " + "\n  ".join(swallowed))


def test_the_thompson_aerosol_table_reaches_the_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """woof declares it ships this file, so the wheel must actually have it.

    ``CCN_ACTIVATE.BIN`` is the one Thompson coefficient artifact that is not
    a ``thompson_init`` product: it is third-party parcel-model output WRF
    redistributes, and until 2026-08-01 this repository did not.  The pin has
    been inverted rather than deleted, because it is the same failure in the
    other direction: an ``exclude-package-data`` entry left behind, or the
    ``.gitignore`` line reinstated, would ship an mp=28 that fails closed on a
    missing asset for every user while the constant still claimed otherwise.

    The pin is bound to ``AEROSOL_ASSET_REDISTRIBUTED`` so the decision moves
    in one place: flip that constant back and this test tells you the
    packaging entry has to move with it.
    """

    _require_companion_tree()
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_ASSET_REDISTRIBUTED,
        AEROSOL_TABLE_ASSETS,
    )

    assert AEROSOL_ASSET_REDISTRIBUTED is True, (
        "the port declares this asset redistributed; if that changed, the "
        "wheel exclusion and woof/data/thompson/PROVENANCE.md change with it")
    pinned = {asset.filename for asset in AEROSOL_TABLE_ASSETS}
    assert {Path(p).name for p in _REDISTRIBUTED_WRF_DATA} == pinned

    # The COMPANION wheel since 2.5.0 -- the claim is unchanged, the
    # distribution carrying the file is not.
    shipped = _companion_shipped(monkeypatch)
    withheld = sorted(_REDISTRIBUTED_WRF_DATA - shipped)
    assert withheld == [], (
        "these files are declared redistributed by woof yet the wheel would "
        f"not contain them: {withheld}\n"
        "Remove the matching [tool.setuptools.exclude-package-data] entry in "
        "recast-woof-data/pyproject.toml under the 'woof_data' key, or widen "
        "[tool.setuptools.package-data]."
    )

    # ...and the shipped bytes must be the pinned WRF v4.6.1 file, not some
    # other 35 KB table: shipping the wrong one is worse than shipping none.
    for relative in sorted(_REDISTRIBUTED_WRF_DATA):
        path = REPO_ROOT / relative
        assert path.is_file(), (
            f"{relative} is declared redistributed but is not in the tree")
        assert path.stat().st_size == next(
            asset.bytes for asset in AEROSOL_TABLE_ASSETS
            if asset.filename == path.name), (
            f"{relative} is present but is not the pinned WRF v4.6.1 file")


def test_the_redistributed_table_is_not_excluded_in_the_pyproject_declaration(
) -> None:
    """The same pin, asserted without setuptools, so it actually runs.

    Every other test in this module routes through
    :func:`_files_setuptools_would_ship`, which begins with
    ``pytest.importorskip("setuptools")``.  That is the right call for a
    measurement of real wheel contents -- a re-implementation of glob
    semantics would be a second authority -- but it has a consequence worth
    stating plainly: in an environment without setuptools those tests SKIP,
    and a skipped packaging gate is indistinguishable from a passing one in a
    summary line.  The project virtualenv used to run this suite is exactly
    such an environment (cupy, numpy, netCDF4, pytest; no setuptools), so the
    strongest control on this file's packaging was silently inert wherever it
    mattered most.

    This asserts the DECLARATION rather than the measurement, out of
    ``tomllib`` in the standard library, so it runs everywhere.  The two are
    complementary and neither replaces the other: this one cannot tell you
    that the file actually reaches the wheel, and the setuptools one cannot
    tell you anything at all when setuptools is absent.  Together they fail
    closed in both environments.
    """

    _require_companion_tree()

    # Both pyprojects, because a stale exclusion could be left in either:
    # the file's directory moved to the companion in 2.5.0, and an entry
    # kept in the root would be dead text while an entry added to the
    # companion would be live breakage.
    excluded: set[str] = set()
    with COMPANION_PYPROJECT.open("rb") as stream:
        companion = tomllib.load(stream)
    excluded |= {
        "recast-woof-data/woof_data/" + entry
        for entry in companion["tool"]["setuptools"].get(
            "exclude-package-data", {}).get("woof_data", [])}
    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    excluded |= {
        "woof/" + entry
        for entry in config["tool"]["setuptools"].get(
            "exclude-package-data", {}).get("woof", [])}

    still_excluded = sorted(set(_REDISTRIBUTED_WRF_DATA) & excluded)
    assert still_excluded == [], (
        "a pyproject.toml excludes a file this repository declares it DOES "
        f"redistribute: {still_excluded}\n"
        "Remove each from [tool.setuptools.exclude-package-data] "
        "(recast-woof-data/pyproject.toml under the 'woof_data' key since 2.5.0). "
        "Leaving the entry in place ships an mp_physics=28 that "
        "fails closed on a missing activation table for every user, while "
        "thompson_aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED claims "
        "otherwise."
    )


def test_license_metadata_is_an_spdx_expression_not_the_pasted_text() -> None:
    """``pip show woof`` must answer ``Apache-2.0``, not 202 pasted lines.

    ``license = { file = "LICENSE" }`` inlines the whole Apache-2.0 text
    into the METADATA ``License`` field -- the first field run of the
    published wheel got a screenful from ``pip show``.  PEP 639 spells
    the fix: ``license`` is the SPDX expression, ``license-files`` ships
    the actual texts in the wheel, and no deprecated ``License ::``
    classifier restates either.  setuptools grew that reading at 77, so
    the build floor must sit there too or an isolated build with an
    older backend ships the old flood again.
    """

    _require_source_tree()

    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)

    project = config["project"]
    assert project["license"] == "Apache-2.0", (
        "project.license must be the bare SPDX expression string; a table "
        "(file= or text=) puts the whole license text into `pip show`")

    # `licenses/*` joined the list at 2.7.0 and is not decoration: MIT
    # (Arm), the FDLIBM/SunPro notice, BSD-3-Clause (AER RRTMG, RTE+RRTMGP,
    # Py-ART/Argonne, MPAS, NumPy) and SIL OFL 1.1 each condition
    # redistribution on their text travelling with the copy, and a wheel
    # carrying only LICENSE + NOTICE performed none of them.  The glob is
    # asserted as a glob rather than expanded, because the set of texts
    # grows and the pin is on the DECLARATION, not on the inventory --
    # tests/test_licence_notices_ship.py holds the inventory.  A glob per
    # licence directory: licenses/* does not recurse, and the 2.9.3 candidate
    # wheel shipped none of licenses/soundings/ (the public sounding renderer's
    # three notices) while NOTICE named them.
    shipped = project.get("license-files", [])
    assert shipped == ["LICENSE", "NOTICE", "licenses/*", "licenses/soundings/*"], shipped
    for name in shipped:
        if name.endswith("/*"):
            directory = REPO_ROOT / name[:-2]
            assert directory.is_dir(), (
                f"pyproject globs a licence directory that does not exist: "
                f"{name}")
            assert any(directory.iterdir()), (
                f"pyproject globs an empty licence directory: {name}")
            continue
        assert (REPO_ROOT / name).is_file(), (
            f"pyproject names a license file that does not exist: {name}")

    # PEP 639 deprecates License classifiers, and setuptools >=77 refuses
    # to combine one with an SPDX expression -- a re-added classifier is a
    # broken release cut, not a cosmetic nit.
    stray = [entry for entry in project.get("classifiers", [])
             if entry.startswith("License ::")]
    assert stray == [], stray

    # The floor that makes all of the above real in an isolated build.
    requires = config["build-system"]["requires"]
    assert any(entry.replace(" ", "").startswith("setuptools>=")
               and float(entry.replace(" ", "").split(">=")[1]) >= 77
               for entry in requires), requires

    # And the legacy spelling must not linger to fight the PEP 639 one.
    assert "license-files" not in config.get("tool", {}).get(
        "setuptools", {}), (
        "[tool.setuptools] license-files duplicates [project] license-files")


# ---------------------------------------------------------------------------
# WHICH PACKAGES a wheel carries, as opposed to which data files
# ---------------------------------------------------------------------------
#
# Everything above measures ``package-data`` -- the files inside packages
# that are already known.  This section measures the step before it:
# ``[tool.setuptools.packages.find]``, which decides what a package IS.
#
# The breakage that earned it, measured on a real
# ``python -m build --wheel``: gpuwm-2.5.0-py3-none-any.whl carried
# ``recast-woof-data/woof_data/__init__.py`` and its top_level.txt named
# ``recast-woof-data``, because ``include = ["woof*"]`` is fnmatch over
# DIRECTORY names and the sibling distribution's project directory is
# called ``recast-woof-data``.  Installing that wheel drops a bare
# ``recast-woof-data/woof_data/`` into site-packages: a woof_data package
# with no ``data/`` beside it, which the resolver's checkout rung then
# finds -- one fragment of another distribution, shipped by this one, in
# a position to answer for it.

#: The top-level names this project's wheel may contain, verbatim from
#: the reasoning in pyproject.toml's ``find`` block.  Written out rather
#: than derived from the include patterns, because a pattern that has
#: started matching something new is exactly what this pins.
#: ``mpas_cycle_bridge`` is deliberate: it is the far side of the MPAS
#: port seam and must be importable from an installed copy because
#: ``woof.cycle.anchor`` imports from it at module scope -- see the
#: ``find`` block's own comment.
_DECLARED_TOP_LEVEL = {"woof", "tools", "tilestream", "configs", "docs",
                       "mpas_cycle_bridge"}


def _find_declaration(pyproject: Path = PYPROJECT) -> dict[str, list[str]]:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    return config["tool"]["setuptools"]["packages"]["find"]


def _packages_setuptools_would_ship(pyproject: Path = PYPROJECT,
                                    project_root: Path = REPO_ROOT
                                    ) -> list[str]:
    """The package list the wheel is built from, from setuptools itself.

    ``find_namespace_packages`` is the finder a pyproject ``find``
    directive runs (``namespaces`` defaults to true there), handed this
    project's own declared ``include``/``exclude``.  Asking it rather
    than re-implementing fnmatch is the same discipline the rest of this
    file follows for ``package-data``: what it reports is what the wheel
    would actually contain, and what ``top_level.txt`` would name.
    """

    from setuptools import find_namespace_packages

    declaration = _find_declaration(pyproject)
    return sorted(find_namespace_packages(
        where=str(project_root),
        include=list(declaration.get("include", ["*"])),
        exclude=list(declaration.get("exclude", []))))


def test_the_gpuwm_wheel_carries_no_member_of_the_companion() -> None:
    """The companion's project directory is not a package of this wheel.

    Concretely: a `pip install recast-woof` must not create
    ``site-packages/recast-woof-data/``.  What landed there was
    ``woof_data/__init__.py`` and not one byte of its data, so
    ``import woof_data`` succeeded, ``woof_data.data_root()`` named a
    directory that did not exist, and the companion's real wheel had a
    fragment of itself sitting in the same interpreter claiming the same
    import name.
    """

    _require_source_tree()
    _require_companion_tree()

    stowaways = [name for name in _packages_setuptools_would_ship()
                 if name.split(".")[0] == COMPANION_ROOT.name]
    assert not stowaways, (
        f"the woof wheel would ship {len(stowaways)} package(s) out of "
        f"the companion's own project directory, which belongs to the "
        f"recast-woof-data distribution:\n  " + "\n  ".join(stowaways))


def test_the_gpuwm_wheel_declares_only_the_top_level_names_it_means_to() -> None:
    """``top_level.txt`` is the wheel's claim on the import namespace.

    setuptools writes it from exactly this package list, so a stray root
    here is a name this distribution asserts ownership of in every
    environment it is installed into -- and uninstalling woof would then
    take that directory with it.
    """

    _require_source_tree()

    roots = {name.split(".")[0]
             for name in _packages_setuptools_would_ship()}
    assert roots == _DECLARED_TOP_LEVEL, (
        f"top_level.txt would name {sorted(roots)}; the declaration in "
        f"pyproject.toml means {sorted(_DECLARED_TOP_LEVEL)}. Unexpected: "
        f"{sorted(roots - _DECLARED_TOP_LEVEL)}; missing: "
        f"{sorted(_DECLARED_TOP_LEVEL - roots)}")


def test_the_companion_wheel_carries_exactly_its_one_package() -> None:
    """The arrow points one way: recast-woof-data ships woof_data and nothing.

    The same finder, on the other pyproject.  Its ``include`` names one
    package explicitly for the reason written beside it -- a glob would
    sweep ``data/`` and its subdirectories in as PEP 420 namespace
    packages -- and this is the measurement that the reason still holds.
    """

    _require_companion_tree()

    found = _packages_setuptools_would_ship(COMPANION_PYPROJECT,
                                            COMPANION_ROOT)
    assert found == [COMPANION_PACKAGE_ROOT.name], found


def test_no_wheel_can_carry_the_marker_that_identifies_a_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``woof.data_assets``'s checkout fallback is guarded on this file.

    That guard is only as strong as the claim "a wheel never contains a
    ``pyproject.toml`` beside the package", so the claim is measured
    rather than assumed, on both distributions and from both directions:

    * nothing is a package of the PROJECT ROOT itself, which is the only
      position from which the root ``pyproject.toml`` could be swept in
      as package data;
    * and no shipped data file is that file.

    If either ever stops holding, an installed wheel could carry the
    marker and the fallback would go live inside site-packages again --
    where a directory named ``recast-woof-data`` beside the package gets read
    as reference data with the version check bypassed.
    """

    _require_source_tree()
    _require_companion_tree()

    for pyproject, project_root in ((PYPROJECT, REPO_ROOT),
                                    (COMPANION_PYPROJECT, COMPANION_ROOT)):
        assert (project_root / "pyproject.toml").is_file()
        packages = _packages_setuptools_would_ship(pyproject, project_root)
        assert packages, f"{pyproject} discovers no packages at all"
        assert "" not in packages and "." not in packages, (
            f"{pyproject} makes the project root itself a package, so its "
            "pyproject.toml is a candidate for package-data and the "
            "checkout marker could ship inside a wheel")

    assert "pyproject.toml" not in _files_setuptools_would_ship(monkeypatch)
    assert "recast-woof-data/pyproject.toml" not in _companion_shipped(monkeypatch)


def _companion_sdist_default_files(monkeypatch: pytest.MonkeyPatch
                                   ) -> set[str]:
    """The companion sdist's file set BEFORE its MANIFEST.in is applied.

    ``sdist._add_defaults_python`` is the step that folds
    ``package-data`` into a source distribution, and the reason this
    measurement exists is that it folds ``exclude-package-data`` in with
    it.  That is not obvious from the names -- ``exclude-package-data``
    is documented as governing the wheel -- and a comment in
    ``recast-woof-data/MANIFEST.in`` asserted the opposite for a whole release
    line.  Same shape as ``tests/test_configs_are_packaged.py``'s sdist
    leg, pointed at the other distribution.
    """

    import setuptools
    from setuptools.command.sdist import sdist
    from distutils.filelist import FileList

    with COMPANION_PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)["tool"]["setuptools"]
    distribution = setuptools.dist.Distribution({
        "name": "recast-woof-data",
        "version": (COMPANION_PACKAGE_ROOT / "VERSION").read_text(
            encoding="utf-8").strip(),
        "packages": [COMPANION_PACKAGE_ROOT.name],
        "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {}),
    })
    distribution.script_name = "setup.py"
    command = sdist(distribution)
    command.finalize_options()
    command.filelist = FileList()
    monkeypatch.chdir(COMPANION_ROOT)
    command._add_defaults_python()
    return {COMPANION_ROOT.name + "/" + name.replace(os.sep, "/")
            for name in command.filelist.files}


def test_the_companion_sdist_carries_the_same_files_as_its_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One exclusion list, two artifacts -- measured, not assumed.

    ``recast-woof-data/MANIFEST.in`` used to say the four CC-BY-NC-SA-4.0 RFMIP
    reference results were "deliberately NOT excluded here", on the
    reasoning that the sdist is the repository's own form and a source
    build must be able to run the suite that reads them.  The built sdist
    disagreed: 14 of the 18 files under ``data/rrtmgp/`` reached it, the
    same 14 the wheel carries, because ``exclude-package-data`` governs
    the sdist's default file set too.

    The breakage a false line there causes is a workflow built on a
    fixture that is not in the artifact -- someone unpacking the sdist to
    reproduce the RFMIP acceptance gates finds four files missing and no
    statement anywhere that says so.

    Both counts move together or neither does.  Deliberately adding the
    fixtures back to the sdist is a licensing decision about a PUBLISHED
    artifact, so it is allowed to fail here: make the change, re-measure,
    and rewrite the paragraph in recast-woof-data/MANIFEST.in in the same
    commit.
    """

    _require_companion_tree()

    prefix = f"{COMPANION_ROOT.name}/{COMPANION_PACKAGE_ROOT.name}/data/"
    in_sdist = {name for name in _companion_sdist_default_files(monkeypatch)
                if name.startswith(prefix)}
    in_wheel = {name for name in _companion_shipped(monkeypatch)
                if name.startswith(prefix)}
    assert in_sdist == in_wheel, (
        "the companion's sdist and wheel carry different data files, so "
        "recast-woof-data/MANIFEST.in and recast-woof-data/pyproject.toml no longer "
        "describe one exclusion list. Only in the sdist: "
        f"{sorted(in_sdist - in_wheel)}; only in the wheel: "
        f"{sorted(in_wheel - in_sdist)}")


def test_no_rfmip_file_is_carried() -> None:
    """No RFMIP NetCDF is in the tree, so none can reach an artifact.

    The breakage: the four reference results are CC-BY-NC-SA-4.0, and a
    copy in the tree is a copy in the public repository and one packaging
    edit away from the companion wheel, attaching a non-commercial
    restriction to both; the input file's terms are ambiguous.  The tests
    that read them fetch them from upstream (woof/core/rfmip_upstream.py).
    """

    from woof.core.rfmip_upstream import RFMIP_FILES

    carried = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "recast-woof-data").rglob("*")
        if path.name in RFMIP_FILES
        or path.name in {"LICENSE-CC-BY-NC-SA-4.0", "LICENSE-CC-BY-4.0"})
    assert not carried, carried


def test_the_companion_wheel_ships_the_renderers_map_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The layers every picture's coastline, borders, state lines and
    counties are drawn from reach the wheel every install pulls.

    A wheel install's renderer has no map assets of its own, and
    ``woof.rustwx`` hands it the companion's.  A companion wheel without
    them draws every picture with no geography, which is the defect this
    copy exists to close; ``test_render_basemap_delivery`` holds the copy
    byte-identical to ``tools/rustwx/assets/basemap``.
    """

    _require_source_tree()
    _require_companion_tree()
    shipped = _companion_shipped(monkeypatch)
    prefix = "recast-woof-data/woof_data/data/basemap/"
    layers = ("natural_earth_10m/ne_10m_coastline",
              "natural_earth_10m/ne_10m_land",
              "natural_earth_10m/ne_10m_ocean",
              "natural_earth_10m/ne_10m_lakes",
              "natural_earth_10m/ne_10m_admin_0_boundary_lines_land",
              "natural_earth_10m/ne_10m_admin_1_states_provinces_lines",
              "natural_earth_110m/ne_110m_coastline",
              "natural_earth_110m/ne_110m_admin_0_countries",
              "us_counties_5m/cb_2023_us_county_5m")
    absent = [prefix + layer + suffix for layer in layers
              for suffix in (".shp", ".shx")
              if prefix + layer + suffix not in shipped]
    assert not absent, (
        "the recast-woof-data wheel would not carry these map layers, so a wheel "
        "install draws pictures with no geography:\n  " + "\n  ".join(absent))


def test_verify_oracle_exclusions_are_the_resolvers_trees() -> None:
    """pyproject's exclusion and the resolver name the same trees, in order.

    The breakage on either side: a tree excluded but not resolved to the
    companion is a verifier reading a path no install has; a tree resolved
    to the companion but still shipped puts its bytes back in a wheel that
    was over the 100,000,000-byte gate with them.
    """

    _require_source_tree()
    from woof import data_assets

    declared = _exclude_package_data_declaration().get("woof", [])
    assert declared == [f"data/{tree}/*" for tree in data_assets.VERIFY_ORACLE_TREES]
    for tree in data_assets.VERIFY_ORACLE_TREES:
        assert (PACKAGE_ROOT / "data" / tree).is_dir(), tree


def test_forecast_read_oracle_members_still_ship(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two oracle directories the forecast reads at run time stay here.

    ``ruc/oracle/tbq.csv`` is the RUC saturation table every sf_surface_physics=3
    run loads, and ``urban/oracle/ucm480/schema.json`` is the UCM argument
    schema: moving either to a companion would make a forecast depend on it.
    """

    _require_source_tree()
    shipped = _files_setuptools_would_ship(monkeypatch)
    for member in ("woof/data/ruc/oracle/tbq.csv",
                   "woof/data/urban/oracle/ucm480/schema.json"):
        assert member in shipped, member


def test_the_kernels_companion_build_carries_every_verify_oracle(tmp_path) -> None:
    """gpuwm-kernels/setup.py stages exactly the excluded trees, byte for byte."""

    _require_source_tree()
    import hashlib
    import importlib.util

    from woof import data_assets

    spec = importlib.util.spec_from_file_location(
        "gpuwm_kernels_setup_hook", REPO_ROOT / "gpuwm-kernels" / "setup.py")
    source = (REPO_ROOT / "gpuwm-kernels" / "setup.py").read_text(encoding="utf-8")
    # Load the hook without running setup(): everything above the call.
    namespace: dict = {"__file__": str(REPO_ROOT / "gpuwm-kernels" / "setup.py"),
                       "__name__": "gpuwm_kernels_setup_hook"}
    exec(compile(source.rsplit("\nsetup(", 1)[0], str(spec.origin), "exec"), namespace)
    manifest = namespace["stage"](tmp_path / "verify_oracles")
    namespace["check"](tmp_path / "verify_oracles")
    assert tuple(manifest["trees"]) == data_assets.VERIFY_ORACLE_TREES
    expected = {name.removeprefix("woof/data/") for name in _files_on_disk()
                if any(name.startswith(f"woof/data/{tree}/")
                       for tree in data_assets.VERIFY_ORACLE_TREES)}
    assert set(manifest["files"]) == expected
    for relative in sorted(expected)[:50]:
        data = (PACKAGE_ROOT / "data" / relative).read_bytes()
        assert manifest["files"][relative]["sha256"] == hashlib.sha256(data).hexdigest()


def test_verify_oracle_paths_resolve_to_the_checkout_here() -> None:
    """A checkout reads its own copy; data_path routes the trees the same way."""

    _require_source_tree()
    from woof import data_assets

    for tree in data_assets.VERIFY_ORACLE_TREES:
        assert data_assets.verify_oracle_path(tree) == PACKAGE_ROOT / "data" / tree
        assert data_assets.data_path(tree) == PACKAGE_ROOT / "data" / tree


def test_a_missing_verify_oracle_is_refused_plainly(tmp_path) -> None:
    from woof import data_assets

    with pytest.raises(data_assets.CompanionDataMissing) as caught:
        data_assets.require_verify_oracle(tmp_path / "absent")
    text = str(caught.value)
    assert "gpuwm-kernels" in text and "pip install --force-reinstall gpuwm-kernels" in text


def test_the_sdist_keeps_the_verify_oracles() -> None:
    """MANIFEST.in re-includes every verification-oracle tree in the source archive.

    exclude-package-data would otherwise strip them from the sdist as well
    (measured: 768 members gone, 75.5 MB -> 63.5 MB), leaving the in-tree
    tests and the rw-mpas oracle-column tests without their fixtures when
    run from the archive.
    """

    _require_source_tree()
    from woof import data_assets

    lines = [line.strip() for line in (REPO_ROOT / "MANIFEST.in").read_text().splitlines()
             if line.startswith("recursive-include woof/data/")]
    expected = {f"recursive-include woof/data/{tree} *" for tree in data_assets.VERIFY_ORACLE_TREES}
    assert expected <= set(lines), sorted(expected - set(lines))
