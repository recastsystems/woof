"""A new source or preparation runner is table work in the ensemble line.

The law (project ruling of 2026-08-16): adding a model must be metadata, not a
new code path, and no model name may reach generic identifiers or defaults.
The ensemble line once dispatched on names in code: ``runner == "<id>"`` in
the shared geography build, a hard-coded set of runner ids at the physical
store door, a ``(source, mode)`` pair in the decoder resolver, a list of two
source ids in the default selection policy, and one contract module per
source. Each of those facts is now a column of the source adapter table, a
field of the preparation runner table, or a packaged authority document.
A preparation chain's ID can carry a source name (``prepared:<source>``), so
what a chain checks before its fetch is a row of
``woof.regional_preparation.preparation_chain_reviews`` that a door reads,
not a branch on the chain ID.

The first test here is the law itself. It walks both tables for every source
id, alias and runner id, then reads the generic ensemble code with ``ast`` and
fails on any of them outside a docstring. The allow list is short and every
entry states its reason. The other tests are the acceptance side: a synthetic
row with the same capabilities is served by every generic door with no code
change, and the data move changed no contract byte.

Scope. The ensemble package and the ensemble tools are generic code and are
read whole. The older native preparations (their own modules outside this
package) name themselves by design and are not read here.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Words that are also a source id or alias, with the reason each is allowed.
#: An allowed word may be said. It may not be what a comparison or a match
#: case tests against: that is a branch on the id, whatever the word also means.
ALLOWED_WORDS = {
    "mapped": (
        "the generic table-driven route (a mapping and a composition document). "
        "It is the route a new model's rows use, so the word names no model. "
        "Its runner id is not allowed by this entry."),
    "wrf": (
        "WRF is the model whose input and output formats the engine reads and "
        "writes. Here the word names those formats, not the wrf archive row."),
    "blend": (
        "an English word (time blend, terrain blend) that is also an alias of "
        "the nbm row."),
    "hex": (
        "hexadecimal (a seed or digest spelled in hex: seed_hex, "
        "parent_dt_fp32_hex) that is also an alias of the hex-global row "
        "(2.9.2). Here the word names the number base, not the row."),
}

#: Identifiers defined outside the ensemble line whose historical name
#: carries a source id.
ALLOWED_NAMES = {
    "interpolate_era5_to_lambert": (
        "the horizontal mapper every route calls (woof/ingest/horiz.py). Its "
        "name predates the other sources and renaming it is outside this line."),
}

#: Whole files, each a single implementation's own code or a kept import path.
ALLOWED_FILES = {
    "woof/ensemble/gfs_physical_contract.py": (
        "import path kept for the native GFS preparation and its tests. It "
        "holds no table and no rule: it binds one packaged contract document "
        "to the generic reader."),
    "woof/ensemble/hrrr_physical_contract.py": (
        "import path kept for the native HRRR preparation tools and their "
        "tests. It holds no table and no rule: it binds one packaged contract "
        "document to the generic reader."),
    "woof/ensemble/gfs_posted_reuse.py": (
        "the native GFS preparation's own member initializer, the counterpart "
        "of tools/hrrr_posted_reuse.py. Only woof/gfs_direct.py imports it. "
        "The standalone preparation wheel has to carry it with that module, "
        "and tools/build_rw_wps_release.py decides which files of this package "
        "the wheel takes, so the file moves when that list moves it."),
}

#: Functions outside the ensemble package that the ensemble line reads its
#: capabilities through. Their string literals are held to the same law.
DOOR_FUNCTIONS = {
    "woof/bridges.py": ("resolve_source_decoder",),
    "woof/source_cli.py": (
        "as_posted_runners", "as_posted_refusal", "physical_option_refusal",
        "role_keyed_input_manifest", "shared_geography_target"),
}


def _table_ids():
    """Every source id, alias and runner id the two tables declare."""
    from woof.source_adapters import source_adapters
    from woof.source_cli import preparation_runners

    sources = set()
    for row in source_adapters():
        sources.update(name.lower() for name in (row.source_id, *row.aliases))
    runners = {name.lower() for name in preparation_runners()}
    assert len(sources) > 30 and len(runners) >= 5, "the tables were not read"
    return sources, runners


def _docstrings(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                found.add(id(body[0].value))
    return found


def _compared(tree):
    """The string constants a comparison or a match case tests against."""
    found = set()

    def operands(node):
        if isinstance(node, ast.Constant):
            found.add(id(node))
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for item in node.elts:
                operands(item)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in (node.left, *node.comparators):
                operands(operand)
        elif isinstance(node, ast.MatchValue):
            operands(node.value)
    return found


class _Law:
    def __init__(self):
        sources, runners = _table_ids()
        self.runners = runners
        ids = sorted(sources | runners, key=len, reverse=True)
        # A name counts where it stands as a word: not inside a longer word.
        self.words = re.compile(
            r"(?<![a-z0-9])(" + "|".join(re.escape(name) for name in ids) + r")(?![a-z0-9])")
        self.identifier_ids = {
            name: "_" + re.sub(r"[^a-z0-9]", "_", name) + "_" for name in ids}

    def in_text(self, text):
        return {match.group(1) for match in self.words.finditer(text.lower())}

    def in_identifier(self, identifier):
        padded = "_" + re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier).lower() + "_"
        return {name for name, needle in self.identifier_ids.items() if needle in padded}

    def strings(self, tree):
        skipped, compared = _docstrings(tree), _compared(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in skipped):
                for name in self.in_text(node.value):
                    # A comparison against the bare id is a branch on it.
                    branch = id(node) in compared and node.value.strip().lower() == name
                    yield (node.lineno, "compared string" if branch else "string",
                           node.value[:70], name)

    def identifiers(self, tree):
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.keyword) and node.arg:
                names = [node.arg]
            elif isinstance(node, ast.alias):
                names = [*node.name.split("."), *([node.asname] if node.asname else [])]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = node.module.split(".")
            else:
                continue
            for identifier in names:
                for name in self.in_identifier(identifier):
                    yield line, "identifier", identifier, name


def _generic_files():
    files = sorted((REPO / "woof" / "ensemble").glob("*.py"))
    files += sorted((REPO / "tools").glob("ensemble_*.py"))
    assert len(files) > 80, "the ensemble line was not found"
    return files


def test_no_ensemble_code_outside_the_tables_names_a_source_or_runner():
    """THE law. A hit is a model name in generic code, or a stale allowance."""
    law = _Law()
    used = set()
    violations = []

    def judge(path, line, kind, text, name):
        if path in ALLOWED_FILES:
            used.add(path)
        elif kind == "identifier" and text in ALLOWED_NAMES:
            used.add(text)
        elif name in ALLOWED_WORDS and kind != "compared string":
            used.add(name)
        else:
            violations.append(f"{path}:{line}: {kind} {text!r} names {name!r}")

    for file in _generic_files():
        path = file.relative_to(REPO).as_posix()
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for name in law.in_identifier(file.stem):
            judge(path, 0, "file name", file.stem, name)
        for line, kind, text, name in (*law.strings(tree), *law.identifiers(tree)):
            judge(path, line, kind, text, name)

    for path, functions in DOOR_FUNCTIONS.items():
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        found = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in functions}
        for missing in sorted(set(functions) - set(found)):
            violations.append(f"{path}: the door function {missing} is not defined")
        for function in found.values():
            for line, kind, text, name in law.strings(function):
                judge(path, line, kind, text, name)

    assert not violations, (
        "a source or runner id is named outside the tables. Move the fact into the "
        "source adapter table, the preparation runner table or an authority document:\n"
        + "\n".join(sorted(violations)))
    stale = (set(ALLOWED_WORDS) | set(ALLOWED_NAMES) | set(ALLOWED_FILES)) - used
    assert not stale, f"allow-list entries that no longer allow anything: {sorted(stale)}"
    reasons = {**ALLOWED_WORDS, **ALLOWED_NAMES, **ALLOWED_FILES}
    assert all(len(reason.split()) >= 8 for reason in reasons.values())


def test_the_allowed_files_are_what_their_reasons_say():
    """An allowance for a whole file is held to the reason it gives."""
    for relative in ("woof/ensemble/gfs_physical_contract.py",
                     "woof/ensemble/hrrr_physical_contract.py"):
        # No table and no rule: a docstring, imports, one document id, and
        # functions that only forward to the generic reader.
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        assigned = []
        for node in tree.body[1:]:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Assign):
                assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), relative
                assigned.extend(target.id for target in node.targets)
                continue
            assert isinstance(node, ast.FunctionDef), f"{relative}: {type(node).__name__} is not a forward"
            body = node.body[1:] if ast.get_docstring(node) else node.body
            assert len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(
                body[0].value, ast.Call), f"{relative}: {node.name} does more than forward"
        assert assigned == ["CONTRACT_ID"], relative
    # The member initializer is its own preparation's code: nothing else imports it.
    module = "gfs_posted_reuse"

    def imports_it(file):
        text = file.read_text(encoding="utf-8", errors="replace")
        if module not in text or file.stem == module:
            return False
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and (
                    (node.module or "").split(".")[-1] == module
                    or any(alias.name == module for alias in node.names)):
                return True
            if isinstance(node, ast.Import) and any(
                    alias.name.split(".")[-1] == module for alias in node.names):
                return True
        return False

    importers = sorted(
        file.relative_to(REPO).as_posix()
        for folder, pattern in ((REPO / "woof", "**/*.py"), (REPO / "tools", "*.py"))
        for file in folder.glob(pattern) if imports_it(file))
    assert importers == ["woof/gfs_direct.py"], importers


def test_the_law_reads_what_it_claims_to_read():
    """The scanner sees a literal, an identifier and a message, and skips prose."""
    law = _Law()
    tree = ast.parse(
        'def chosen(runner, source):\n'
        '    """A docstring may say HRRR or gfs_pgrb2_0p25_v1."""\n'
        '    is_hrrr = runner == "hrrr_f00_f12_v1"\n'
        '    if source in {"gfs", "rrfs"}:\n'
        '        raise ValueError("automatic HRRR geography is refused")\n'
        '    return is_hrrr\n')
    strings = {name for _, _, _, name in law.strings(tree)}
    assert strings == {"hrrr_f00_f12_v1", "gfs", "rrfs", "hrrr"}
    assert {kind for _, kind, _, name in law.strings(tree) if name != "hrrr"} == {"compared string"}
    assert {name for _, _, _, name in law.identifiers(tree)} == {"hrrr"}
    assert law.in_identifier("SharedPostedHrrr") == {"hrrr"}
    assert not law.in_text("a thermal threshold")
    # An allowed word is excused where it is said, never where it is tested.
    word = next(iter(ALLOWED_WORDS))
    said = ast.parse(f'route = build(route="{word}")\nlabel = "the {word} route"\n')
    assert {kind for _, kind, _, _ in law.strings(said)} == {"string"}
    for tested in (f'if route == "{word}":\n    pass\n',
                   f'if route not in ("{word}", other):\n    pass\n',
                   f'match route:\n    case "{word}":\n        pass\n'):
        assert [kind for _, kind, _, _ in law.strings(ast.parse(tested))] == ["compared string"]


# ---------------------------------------------------------------------------
# The tables carry the facts the old branches held.
# ---------------------------------------------------------------------------

def test_the_built_in_policy_reads_calibration_requirements_from_the_source_table(monkeypatch):
    from woof import source_adapters
    from woof.ensemble.automatic_sources import SourceSelectionPolicy

    flagged = sorted(row.source_id for row in source_adapters.source_adapters()
                     if row.requires_ensemble_calibration)
    document = SourceSelectionPolicy.unfitted().document
    # The exact list the code carried as a literal at 2.8.5 staging.
    assert document["requires_calibration"] == flagged == ["hrrr", "rrfs"]
    # The document's digest names the policy in every selection receipt.
    # This is the digest the literal list gave at e3ac018cb.
    assert SourceSelectionPolicy.unfitted().sha256 == (
        "c81b0306a9ff0f62c884f51e6bd17c5e1508069d1004565dc17f417df3d0f017")

    rows = source_adapters.source_adapters()
    extra = replace(source_adapters.get_source_adapter("gfs"), source_id="synthetic-cam",
                    aliases=(), requires_ensemble_calibration=True)
    monkeypatch.setattr(source_adapters, "source_adapters", lambda: (*rows, extra))
    assert SourceSelectionPolicy.unfitted().document["requires_calibration"] == [
        "hrrr", "rrfs", "synthetic-cam"]


def test_runner_rows_reproduce_the_facts_the_ensemble_line_branched_on():
    from woof import gfs_direct
    from woof.source_cli import as_posted_runners, preparation_runners

    rows = preparation_runners()
    named = {"native HRRR": None, "GFS": None, "mapped-source": None}
    for name, row in rows.items():
        if row.title in named:
            named[row.title] = (name, row)
    assert all(named.values())
    (hrrr_id, hrrr), (gfs_id, gfs), (mapped_id, mapped) = named.values()
    assert as_posted_runners() == {hrrr_id, gfs_id, mapped_id}
    assert {name for name, row in rows.items() if row.physical_stores} == {hrrr_id, gfs_id, mapped_id}
    assert {name for name, row in rows.items() if row.physical_base_prepared} == {hrrr_id}
    assert {name for name, row in rows.items() if row.provider_supplies_decode} == {gfs_id}
    assert {name for name, row in rows.items() if row.composition_inputs} == {mapped_id}
    # Prebuilt geography: the option each command reads, and what leaves it.
    assert (hrrr.shared_geography.cache_option, hrrr.shared_geography.superseded_options) == (
        "--static-cache", ("--geog-root",))
    assert hrrr.shared_geography.command is not None and hrrr.shared_geography.fields is None
    for row in (gfs, mapped):
        geography = row.shared_geography
        assert (geography.cache_option, geography.superseded_options) == ("--static-input", ())
        assert geography.fields is not None and geography.command is None
    assert {row.shared_geography.receipt_option for row in (hrrr, gfs, mapped)} == {"--static-receipt"}
    assert sorted(name for name, row in rows.items()
                  if row.shared_geography is not None and row.shared_geography.builds) == sorted(
                      (hrrr_id, gfs_id, mapped_id))
    # A repeat guarded by a test is not a second definition.
    assert gfs.input_manifest.schema == gfs_direct.INPUT_MANIFEST_SCHEMA
    assert gfs.input_manifest.lead_role_prefix == gfs_direct._AS_POSTED_LEAD_ROLE_PREFIX
    assert [row.input_manifest for row in (hrrr, mapped)] == [None, None]


def _synthetic_table(tmp_path, **capabilities):
    """The real rows plus one runner that exists only in this table."""
    from woof.source_cli import (PreparationRunner, RoleKeyedManifest, SharedGeography,
                                  preparation_runners)

    def command(option, cache, receipt):
        script = ("import sys, pathlib; pathlib.Path(sys.argv[1]).write_bytes(b'cache:' + "
                  "sys.argv[3].encode()); pathlib.Path(sys.argv[2]).write_text('{}')")
        return [sys.executable, "-c", script, str(cache), str(receipt), option("--geog-root")]

    values = dict(
        title="synthetic", as_posted=True, physical_stores=True,
        shared_geography=SharedGeography("--static-input", superseded_options=("--geog-root",),
                                         command=command),
        input_manifest=RoleKeyedManifest("synthetic-input-manifest-v1", "lead-"))
    values.update(capabilities)
    row = PreparationRunner(lambda args: [], lambda args: [], "prepared:staged", **values)
    return {**preparation_runners(), "synthetic_runner_v1": row}


def _physical_args(**selected):
    names = ("physical_input_store", "physical_input_provider", "physical_member_index",
             "physical_output_store", "physical_base_prepared")
    return SimpleNamespace(**{name: selected.get(name) for name in names})


def test_the_physical_store_door_admits_by_row_and_refuses_in_the_same_words(tmp_path):
    from woof.source_cli import physical_option_refusal, preparation_runners

    rows = preparation_runners()
    stores = _physical_args(physical_output_store="/member/physical")
    base = _physical_args(physical_base_prepared="/base")
    assert physical_option_refusal(_physical_args(), "no_such_runner") is None
    for name, row in rows.items():
        refusal = physical_option_refusal(stores, name)
        if row.physical_stores:
            assert refusal is None
        else:
            # The sentence the door printed when it held the names in code.
            assert refusal == "physical stores require native HRRR, GFS or mapped-source preparation"
        if row.physical_stores and not row.physical_base_prepared:
            assert physical_option_refusal(base, name) == (
                "physical-base-prepared is used only by the native HRRR route")
    assert physical_option_refusal(stores, None) == (
        "physical stores require native HRRR, GFS or mapped-source preparation")

    table = _synthetic_table(tmp_path, physical_base_prepared=True)
    assert physical_option_refusal(stores, "synthetic_runner_v1", runners=table) is None
    assert physical_option_refusal(base, "synthetic_runner_v1", runners=table) is None
    unlisted = next(name for name, row in rows.items() if not row.physical_stores)
    assert physical_option_refusal(stores, unlisted, runners=table) == (
        "physical stores require native HRRR, GFS, mapped-source or synthetic preparation")


def test_as_posted_is_a_row_field(monkeypatch):
    from woof import source_cli
    from woof.source_adapters import source_adapters

    rows = source_cli.preparation_runners()
    served = [row for row in source_adapters() if row.runner in rows]
    assert len(served) > 15
    whole = "prepares a whole fetched window"

    def reads_whole_window(source):
        return whole in (source_cli.as_posted_refusal(source) or "")

    for adapter in served:
        assert reads_whole_window(adapter.source_id) == (not rows[adapter.runner].as_posted)
    # Flip the field on every row and every source's answer flips with it.
    flipped = {name: replace(row, as_posted=not row.as_posted) for name, row in rows.items()}
    monkeypatch.setattr(source_cli, "preparation_runners", lambda: flipped)
    for adapter in served:
        assert reads_whole_window(adapter.source_id) == rows[adapter.runner].as_posted


def test_a_decoder_mode_is_a_declared_capability(monkeypatch, tmp_path):
    from woof import bridges

    declared = {name: modes for name, modes in bridges.DECODER_MODE_CONTRACTS.items()
                if "as_posted" in modes}
    assert len(declared) == 1, "one decoder declares append-only lead admission at 2.8.5"
    for source, name in bridges.SOURCE_DECODERS.items():
        if name not in declared:
            with pytest.raises(ValueError, match="no decoder mode contract is declared"):
                bridges.resolve_source_decoder(source, mode="as_posted")

    binary = tmp_path / "synthetic_bridge"
    binary.write_bytes(b"\x7fELF base contract only")
    monkeypatch.setitem(bridges.SOURCE_DECODERS, "synthetic", "synthetic_bridge")
    monkeypatch.setitem(bridges.DECODER_MODE_CONTRACTS, "synthetic_bridge", {"as_posted": {
        "marker": b"--admit-leads SERIES", "command": "--admit-leads",
        "required_by": "--as-posted", "label": "posted-mode"}})
    monkeypatch.setattr(bridges, "find_bridge", lambda name: binary)
    assert bridges.resolve_source_decoder("synthetic") == binary
    with pytest.raises(bridges.DecoderContractError,
                       match="does not implement --admit-leads, which --as-posted requires"):
        bridges.resolve_source_decoder("synthetic", mode="as_posted")
    binary.write_bytes(b"\x7fELF --admit-leads SERIES")
    assert bridges.resolve_source_decoder("synthetic", mode="as_posted") == binary
    with pytest.raises(ValueError, match="no decoder mode contract is declared"):
        bridges.resolve_source_decoder("synthetic", mode="no-such-mode")


def test_a_role_keyed_manifest_is_closed_by_its_row(monkeypatch, tmp_path):
    from woof import source_cli
    from woof.ensemble.acquisition_binding import mapped_input_closure

    table = _synthetic_table(tmp_path)
    monkeypatch.setattr(source_cli, "preparation_runners", lambda: table)
    cycle = "2026-10-02T18:00:00+00:00"
    acquisition = {"source": "gfs", "cycle": cycle, "files": [
        {"name": "lead0.grib2", "bytes": 10, "sha256": "a" * 64},
        {"name": "lead1.grib2", "bytes": 11, "sha256": "b" * 64}]}
    inputs = {"schema": "synthetic-input-manifest-v1", "source": {"model": "GFS", "cycle": cycle},
              "files": {"lead-000": {"name": "lead0.grib2", "sha256": "a" * 64},
                        "lead-001": {"name": "lead1.grib2", "sha256": "b" * 64},
                        "decoder": {"name": "bridge", "sha256": "c" * 64}}}
    assert mapped_input_closure(acquisition, inputs, "/source") == [
        {"path": "lead0.grib2", "bytes": 10, "sha256": "a" * 64},
        {"path": "lead1.grib2", "bytes": 11, "sha256": "b" * 64}]
    # A schema no row declares is read as a mapped manifest, as before.
    with pytest.raises(ValueError, match="no native primary payload records"):
        mapped_input_closure(acquisition, {**inputs, "schema": "undeclared-v1"}, "/source")


# ---------------------------------------------------------------------------
# Shared geography: the build and the options come from the row.
# ---------------------------------------------------------------------------

def _owner(tmp_path):
    from woof.ensemble.automatic_preparation import AutomaticPreparationOwner
    from woof.ensemble.automatic_sources import EnsembleSourceContext, resolve_ensemble_sources

    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    context = EnsembleSourceContext(start, start + timedelta(hours=12), object())
    owner = AutomaticPreparationOwner(resolve_ensemble_sources(1, context), context,
                                      output_root=tmp_path / "owner")
    launched = []

    def launch(command, *, identity, role):
        # The builder is a subprocess in production. Here its result is
        # written in place so the argument rewrite can be read back.
        output = Path(command[command.index("--output-root") + 1])
        request = json.loads(Path(command[command.index("--request") + 1]).read_bytes())
        output.mkdir(parents=True)
        cache, receipt = output / "native-static.npz", output / "native-static-receipt.json"
        cache.write_bytes(b"cache")
        receipt.write_text("{}")
        (output / "shared-geography.json").write_text(json.dumps({
            "status": "PASS", "runner": request["runner"], "cache": str(cache),
            "cache_sha256": hashlib.sha256(b"cache").hexdigest(), "receipt": str(receipt),
            "receipt_sha256": hashlib.sha256(b"{}").hexdigest()}))
        launched.append(request)
        owner.processes.append({"identity": identity, "role": role, "log": "log",
                                "process": SimpleNamespace(poll=lambda: 0)})

    owner._launch = launch
    return owner, launched


def _rewritten(owner, runner, tmp_path):
    geog = tmp_path / "geog"
    geog.mkdir(exist_ok=True)
    arguments = ("--geog-root", str(geog), "--preprocess-workers", "2")
    return owner._shared_static_arguments(SimpleNamespace(runner=runner), arguments, timeout=5)


def test_prebuilt_geography_options_come_from_the_row(tmp_path):
    from woof.source_cli import preparation_runners

    owner, launched = _owner(tmp_path)
    for name, row in preparation_runners().items():
        geography = row.shared_geography
        if geography is None:
            with pytest.raises(ValueError, match="has no shared native geography producer"):
                _rewritten(owner, name, tmp_path)
            continue
        if not geography.builds:
            # The row takes a caller's pair and names no build of its own.
            supplied = (geography.cache_option, "/case/static.npz", "--static-receipt", "/case/static.json")
            assert owner._shared_static_arguments(
                SimpleNamespace(runner=name), supplied, timeout=5) == supplied
            continue
        result = _rewritten(owner, name, tmp_path)
        assert launched[-1]["runner"] == name
        cache = result[result.index(geography.cache_option) + 1]
        receipt = result[result.index("--static-receipt") + 1]
        assert Path(cache).name == "native-static.npz" and Path(receipt).name == "native-static-receipt.json"
        assert ("--geog-root" in result) == ("--geog-root" not in geography.superseded_options)
        assert result[result.index("--preprocess-workers") + 1] == "2"
        # A caller's own pair is that authority: nothing is built or rewritten.
        supplied = (geography.cache_option, cache, "--static-receipt", receipt)
        count = len(launched)
        assert owner._shared_static_arguments(SimpleNamespace(runner=name), supplied, timeout=5) == supplied
        assert len(launched) == count


def test_a_new_runner_row_gets_shared_geography_with_no_code_change(monkeypatch, tmp_path):
    from woof import source_cli
    from woof.ensemble import automatic_static

    table = _synthetic_table(tmp_path)
    monkeypatch.setattr(source_cli, "preparation_runners", lambda: table)
    geog = tmp_path / "geog"
    geog.mkdir()
    result = automatic_static.build_shared_static(
        runner="synthetic_runner_v1", arguments=["--geog-root", str(geog)],
        output_root=tmp_path / "built")
    assert result["status"] == "PASS" and result["runner"] == "synthetic_runner_v1"
    assert Path(result["cache"]).read_bytes() == b"cache:" + str(geog).encode()
    assert result["cache_sha256"] == hashlib.sha256(Path(result["cache"]).read_bytes()).hexdigest()
    assert json.loads((tmp_path / "built" / "shared-geography.json").read_bytes()) == result

    owner, launched = _owner(tmp_path)
    rewritten = _rewritten(owner, "synthetic_runner_v1", tmp_path)
    assert launched[-1]["runner"] == "synthetic_runner_v1"
    assert "--geog-root" not in rewritten and "--static-input" in rewritten

    for runner, reason in (("no_such_runner", "has no shared native geography producer"),
                           (next(name for name, row in source_cli.preparation_runners().items()
                                 if row.shared_geography is not None and not row.shared_geography.builds),
                            "has no shared native geography producer")):
        with pytest.raises(ValueError, match=reason):
            automatic_static.build_shared_static(
                runner=runner, arguments=["--geog-root", str(geog)], output_root=tmp_path / runner)
    with pytest.raises(ValueError, match="requires the original geog-root authority"):
        automatic_static.build_shared_static(
            runner="synthetic_runner_v1", arguments=[], output_root=tmp_path / "no-geog")


# ---------------------------------------------------------------------------
# A door asks a preparation chain's row what the chain checks before its fetch.
# ---------------------------------------------------------------------------

def test_every_preparation_chain_declares_its_review():
    """Breakage it prevents: a chain added without a review row would reach a
    door's fetch with no review, and a door that tested the chain ID instead
    named a source in generic code (``chain == "prepared:<source>"``)."""
    from woof.regional_preparation import preparation_chain_reviews, preparation_chains

    reviews = preparation_chain_reviews()
    assert set(reviews) == set(preparation_chains())
    assert all(review is None or callable(review) for review in reviews.values())
    assert any(review is not None for review in reviews.values())


def test_a_chain_review_makes_the_call_the_chain_branches_made(monkeypatch, tmp_path):
    """Each row asks what the door's per-chain branches asked, with the same arguments."""
    from woof import go_cli, hrrr_route_inputs, era5_preparation
    from woof.regional_preparation import preparation_chain_reviews

    asked = []
    monkeypatch.setattr(go_cli, "plan_from_config",
                        lambda config, **options: asked.append(("plan", config, options)))
    monkeypatch.setattr(hrrr_route_inputs, "run_route_inputs",
                        lambda config, exp, *, raw: asked.append(("route inputs", config, exp, raw)))
    monkeypatch.setattr(era5_preparation, "review",
                        lambda config, exp, **kwargs: asked.append(("case inputs", config, exp, kwargs)))
    config, exp, raw = tmp_path / "member" / "experiment.toml", object(), {"fetch": {"source": "era5"}}
    scratch = tmp_path / "review"
    posting = {"transport": "aws", "as_posted": True, "late_after_minutes": 30.0}
    answered = {}
    for chain, review in preparation_chain_reviews().items():
        asked.clear()
        if review is not None:
            assert review(config, exp, raw=raw, scratch=scratch, posting=posting) is None
        answered[chain] = list(asked)
    assert answered == {
        "prepared:go": [("plan", config, dict(outdir=scratch / "plan", run_stamp=False,
                                              data_dir=scratch / "data", **posting))],
        "prepared:hrrr": [("route inputs", config, exp, raw)],
        "prepared:staged": [],
        "prepared:case-data": [("case inputs", config, exp,
                                  dict(raw=raw, scratch=scratch, posting=posting))]}


def _shipped_member(tmp_path, stem, edit=None):
    """A copy of a shipped config and the WPS namelist beside it, as a door writes a member."""
    import tomllib

    from woof.experiment import load_experiment

    folder = tmp_path / stem
    folder.mkdir()
    text = (REPO / "configs" / f"{stem}.toml").read_text(encoding="utf-8")
    config = folder / "experiment.toml"
    config.write_text(text if edit is None else edit(text), encoding="utf-8")
    (folder / "experiment.namelist.wps").write_bytes(
        (REPO / "configs" / f"{stem}.namelist.wps").read_bytes())
    return config, load_experiment(config), tomllib.loads(config.read_text(encoding="utf-8"))


def _edited(*pairs):
    """An edit of a shipped config that fails loudly when its text is not there."""
    def edit(text):
        for before, after in pairs:
            assert text.count(before) == 1, before
            text = text.replace(before, after)
        return text
    return edit


def _answer(call):
    try:
        call()
    except ValueError as error:
        return ("refused", type(error).__name__, str(error))
    return ("accepted",)


@pytest.mark.parametrize("stem, edit, outcome", [
    ("hrrr_native_quick_demo", None, "accepted"),
    # The hourly chain's namelists are Lambert-only.
    ("hrrr_native_quick_demo", _edited(('map_proj = "lambert"', 'map_proj = "mercator"'),
                                       ("map_proj = 1\n", "map_proj = 3\n")), "refused"),
    ("gfs_12km_quickstart", None, "accepted"),
    # The native chain's planner needs the fetch crop box.
    ("gfs_12km_quickstart", _edited(('area = "17.81,-116.67,53.13,-78.33"\n', "")), "refused"),
])
def test_a_chain_review_answers_what_the_chain_answers(tmp_path, stem, edit, outcome):
    """Unmocked: the row gives a shipped member config the chain planner's own
    answer, an acceptance or the same refusal sentence."""
    from woof import go_cli, runplan
    from woof.hrrr_route_inputs import run_route_inputs
    from woof.regional_preparation import preparation_chain_reviews

    config, exp, raw = _shipped_member(tmp_path, stem, edit)
    chain = runplan.prepared_chain_for_source(raw["fetch"]["source"], source_root=None)
    review = preparation_chain_reviews()[chain]
    scratch = tmp_path / "scratch"
    via_row = _answer(lambda: review(config, exp, raw=raw, scratch=scratch, posting={}))
    direct = {
        "prepared:go": lambda: go_cli.plan_from_config(
            config, outdir=scratch / "plan", run_stamp=False, data_dir=scratch / "data"),
        "prepared:hrrr": lambda: run_route_inputs(config, exp, raw=raw),
    }[chain]
    assert via_row[0] == outcome
    assert via_row == _answer(direct)


# ---------------------------------------------------------------------------
# Physical field contracts are documents read by one module.
# ---------------------------------------------------------------------------

GRID = {"mass_shape": [3, 4]}

#: Digests of the contracts for this grid and evidence. GFS retains the
#: Python table at e3ac018cb; HRRR includes the analyzed vegetation carrier.
#: Canonical JSON, then JSON in insertion order.
GOLDEN = {
    "gfs-pgrb2-0p25-physical-fields-v1": (
        "de7f6542d82cbec219d3ef0c5b5665fa86cfa33a24a84afb06117482389ce6d4",
        "2bcf3fa4c75907d5a43e7f321bbfca571bec01b58379b05c3180bef34d587bdf"),
    "hrrr-f00-f12-physical-fields-v1": (
        # development branch commit 6a69b356f, merged by b3462fb33:
        # retain analyzed surface VEGFRA (GRIB2 2/0/4) in percent on the
        # mass grid, rather than replacing it with monthly climatology.
        "153e9ae308606d21e33ae7d3c2fbc7ae33a52447bae11d9731cb2455d9a412fa",
        "6afa85fede7db449f12ac0dc02fec200b3c3e5c4a7158580c74498d33a799ddf"),
}


def _evidence(contract_id):
    from woof.ensemble.physical_fields import native_contract_document

    if native_contract_document(contract_id)["evidence"] is None:
        return {"decoder": "a" * 64}
    return {name: hashlib.sha256(name.encode()).hexdigest() for name in (
        "native_mapper", "raw_source_manifest", "water_temperature_assembly",
        "sealed_native_bridge_manifest")}


def test_every_packaged_contract_is_pinned_and_loads():
    from woof import source_authorities
    from woof.ensemble.physical_fields import native_contract_document, native_contract_sha256

    ids = source_authorities.packaged_physical_contract_ids()
    assert set(ids) == set(GOLDEN)
    for contract_id in ids:
        path = source_authorities.packaged_physical_contract(contract_id)
        data = path.read_bytes()
        assert b"\r" not in data
        assert hashlib.sha256(data).hexdigest() == native_contract_sha256(contract_id)
        assert native_contract_document(contract_id)["contract_id"] == contract_id


@pytest.mark.parametrize("contract_id", sorted(GOLDEN))
def test_the_documents_build_the_contracts_the_python_tables_built(contract_id):
    from woof.ensemble.physical_fields import (
        field_contract_sha256, native_field_contract, require_native_field_contract)

    contract = native_field_contract(contract_id, GRID, evidence=_evidence(contract_id))
    ordered = hashlib.sha256(json.dumps(contract, separators=(",", ":")).encode()).hexdigest()
    assert (field_contract_sha256(contract), ordered) == GOLDEN[contract_id]
    assert require_native_field_contract(contract_id, contract, GRID) is contract
    # Two builds never share a mutable array table.
    contract["arrays"]["levels_hpa"]["units"] = "changed"
    assert native_field_contract(contract_id, GRID, evidence=_evidence(contract_id))[
        "arrays"]["levels_hpa"]["units"] != "changed"


def test_the_import_paths_still_serve_the_same_contracts():
    from woof.ensemble import gfs_physical_contract, hrrr_physical_contract
    from woof.ensemble.physical_fields import field_contract_sha256

    gfs = gfs_physical_contract.native_gfs_field_contract(
        GRID, evidence=_evidence(gfs_physical_contract.CONTRACT_ID))
    hrrr = hrrr_physical_contract.hrrr_physical_field_contract(
        GRID, evidence=_evidence(hrrr_physical_contract.CONTRACT_ID))
    assert field_contract_sha256(gfs) == GOLDEN[gfs_physical_contract.CONTRACT_ID][0]
    assert field_contract_sha256(hrrr) == GOLDEN[hrrr_physical_contract.CONTRACT_ID][0]
    # The same statics commit adds VEGFRA to the bound parameter table.
    assert hrrr["evidence"]["hrrr_parameter_contract"] == (
        "9036da53e6bd39b5393da1cafb9a2164271386648920a2b70e3ee01e28011259")
    # Evidence names the document that defines the contract.
    assert gfs_physical_contract.contract_sha256() == hashlib.sha256(
        (REPO / "woof" / "authorities" / "rw-wps-gfs-pgrb2-0p25.physical-fields.json").read_bytes()
    ).hexdigest()


def test_the_analyzed_vegetation_carrier_keeps_its_percent_semantics():
    """The statics addition declares analyzed percent, not monthly climatology."""
    import numpy as np
    from woof.ensemble import hrrr_physical_contract
    from woof.ensemble.physical_fields import native_contract_document
    from woof.ingest.vegetation import initial_vegetation_fraction

    document = native_contract_document(hrrr_physical_contract.CONTRACT_ID)
    assert document["parameters"]["VEGFRA"] == [[2, 0, 4], "percent"]
    expected = {
        "units": "percent", "dimensions": ["y", "x"], "basis": "scalar",
        "source_fields": ["VEGFRA"],
        "operation": "Bilinear interpolation of analyzed vegetation fraction "
                     "in the source projection; percent units unchanged.",
    }
    assert document["arrays"]["field__VEGFRA"] == expected
    contract = hrrr_physical_contract.hrrr_physical_field_contract(
        GRID, evidence=_evidence(hrrr_physical_contract.CONTRACT_ID))
    assert contract["arrays"]["field__VEGFRA"] == expected

    # An analyzed mass-grid field overrides a different monthly climatology
    # and retains its values, including both endpoints, without rescaling.
    analyzed = np.tile(np.array([0.0, 25.0, 75.0, 100.0], np.float32), (3, 1))
    static = {"LANDMASK": np.ones((3, 4)), "GREENFRAC": np.zeros((12, 3, 4))}
    assert initial_vegetation_fraction(
        SimpleNamespace(fields={"VEGFRA": analyzed}), static,
        datetime(2026, 1, 1, tzinfo=timezone.utc)) is analyzed


def test_the_native_axes_a_document_repeats_are_the_ingest_constants():
    import numpy as np
    from woof.ensemble import hrrr_physical_contract
    from woof.ensemble.physical_fields import native_contract_document
    from woof.ingest.hrrr import HRRR_HYBRID_LEVELS, HRRR_SOIL_DEPTHS_M

    rules = native_contract_document(hrrr_physical_contract.CONTRACT_ID)["snapshot"]
    assert np.array_equal(np.asarray(rules["levels"]), HRRR_HYBRID_LEVELS)
    assert set(rules["leading_axis_length"].values()) == {len(HRRR_SOIL_DEPTHS_M)}


def test_a_new_contract_is_a_document_and_a_pin(monkeypatch, tmp_path):
    import numpy as np
    from woof import source_authorities
    from woof.ensemble import physical_fields as contracts

    row = {"units": "K", "dimensions": ["level", "y", "x"], "basis": "scalar",
           "source_fields": ["T"], "operation": "bilinear interpolation"}
    document = {
        "schema": contracts.NATIVE_CONTRACT_SCHEMA, "contract_id": "synthetic-physical-fields-v1",
        "description": "A contract that exists only in this test.", "unit_authorities": [],
        "vertical": {"kind": "pressure_levels", "units": "hPa", "values": "levels_hpa",
                     "pressure_field": None},
        "arrays": {"levels_hpa": {**row, "units": "hPa", "dimensions": ["level"]}, "field__TT": row},
        "codes": {"TT": [0, 0, 0]},
        "evidence": {"required": ["decoder"], "any_of": [["manifest", "input_plan"]],
                     "table_digests": {"code_table": "codes"}},
        "snapshot": {"levels": [1000.0, 850.0], "leading_axis_length": {"SOIL": 4}},
        "refusals": {
            "evidence_type": "synthetic fields need evidence",
            "evidence_missing": "synthetic fields need a decoder and a source authority",
            "vertical": "synthetic input vertical coordinate differs",
            "unknown_array": "synthetic input has no mapping for {name}",
            "incompatible_array": "synthetic input {name} has incompatible {attribute}",
            "snapshot_levels": "synthetic input levels differ",
            "snapshot_leading_axis": "synthetic input {name} has the wrong depth count"}}
    path = tmp_path / "synthetic.physical-fields.json"
    path.write_bytes((json.dumps(document, indent=2) + "\n").encode())
    monkeypatch.setattr(source_authorities, "_PACKAGED_PHYSICAL_CONTRACTS", {
        **source_authorities._PACKAGED_PHYSICAL_CONTRACTS,
        "synthetic-physical-fields-v1": {"file": str(path),
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})

    contract_id = "synthetic-physical-fields-v1"
    with pytest.raises(ValueError, match="need evidence"):
        contracts.native_field_contract(contract_id, GRID, evidence=None)
    with pytest.raises(ValueError, match="a decoder and a source authority"):
        contracts.native_field_contract(contract_id, GRID, evidence={"decoder": "a" * 64})
    contract = contracts.native_field_contract(
        contract_id, GRID, evidence={"decoder": "a" * 64, "input_plan": "b" * 64})
    assert contract["evidence"]["code_table"] == hashlib.sha256(
        json.dumps(document["codes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert contracts.require_native_field_contract(contract_id, contract, GRID) is contract
    changed = json.loads(json.dumps(contract))
    changed["arrays"]["field__TT"]["units"] = "degC"
    with pytest.raises(ValueError, match="field__TT has incompatible units"):
        contracts.require_native_field_contract(contract_id, changed, GRID)
    changed = json.loads(json.dumps(contract))
    changed["arrays"]["field__QQ"] = dict(row)
    with pytest.raises(ValueError, match="no mapping for field__QQ"):
        contracts.require_native_field_contract(contract_id, changed, GRID)
    snapshot = SimpleNamespace(levels_hpa=np.array([1000.0, 850.0]), fields={"SOIL": np.ones((4, 3, 4))})
    assert contracts.validate_native_snapshot(contract_id, snapshot) is snapshot
    snapshot.fields["SOIL"] = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="SOIL has the wrong depth count"):
        contracts.validate_native_snapshot(contract_id, snapshot)
    snapshot.levels_hpa = np.array([1000.0, 700.0])
    with pytest.raises(ValueError, match="levels differ"):
        contracts.validate_native_snapshot(contract_id, snapshot)

    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="hash differs"):
        contracts.native_field_contract(contract_id, GRID, evidence={"decoder": "a" * 64, "manifest": "b" * 64})
