"""The two front-door nesting legs, graded without a card.

``tools/tiles_door_legs.py`` needs a GPU, an ERA5 bundle and about half an
hour, so it runs on the GPU leg from ``tools/battery/tiles_gates.txt``.
Everything ABOUT it that can rot silently is checked here instead, on every
commit, for the reason ``tests/test_tiles_gate_manifest.py`` gives for
itself: a gate that stops existing must fail on a commit rather than at a
cut, on the node, when there is no time.

What rots:

* the two entries losing their ``LEG=`` environment, which would run leg A
  twice and report two passes;
* a template acquiring a committed ``vram_budget_bytes``, which turns a
  card fact into a claim about whichever box last touched it;
* a user path or a scratchpad path surviving into a committed case file;
* the SHAPES drifting -- if leg B stopped carrying a per-domain ``on`` it
  would still pass its own gate while testing leg A's shape twice.

The last one is the one worth stating plainly: these files are the only
place the two shapes are written down, so a gate that ran them faithfully
would still prove nothing if the files stopped describing what their names
say.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = REPOSITORY_ROOT / "configs" / "battery"
GATES = REPOSITORY_ROOT / "tools" / "battery" / "tiles_gates.txt"
RUNNER = REPOSITORY_ROOT / "tools" / "tiles_door_legs.py"

MODULE = "tools.tiles_door_legs"
PLACEHOLDER = "VRAM_BUDGET_BYTES_PLACEHOLDER"

LEG_A = CASES / "shape_tiles_streamed_parent_resident_child.toml.tmpl"
LEG_B = CASES / "shape_tiles_resident_parent_streamed_child.toml.tmpl"
CONTROL = CASES / "shape_tiles_control_resident_tree.toml"


def _gate_entries() -> list[tuple[str, str, tuple[str, ...]]]:
    """``(shard, module, env)`` parsed the way the battery parses it."""
    out = []
    for raw in GATES.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append((parts[0], parts[1], tuple(parts[2:])))
    return out


def _loaded(template: pathlib.Path) -> dict:
    """The template with a budget substituted, parsed as TOML.

    Substituting before parsing is deliberate coverage rather than a
    workaround: it proves the committed template MATERIALISES into valid
    TOML, which is the only thing the gate does to it before handing it to
    the front door.
    """
    text = template.read_text(encoding="utf-8")
    return tomllib.loads(text.replace(PLACEHOLDER, "1073741824"))


def test_the_runner_and_all_three_case_files_exist() -> None:
    for path in (RUNNER, LEG_A, LEG_B, CONTROL,
                 CASES / "shape_tiles_doorleg.namelist.wps"):
        assert path.is_file(), f"{path} is missing"


def test_both_legs_are_registered_on_the_gpu_shard_with_distinct_legs() -> None:
    """Two entries, same module, DIFFERENT ``LEG``.

    Same module twice is exactly what the manifest's own header allows
    ("two entries may share a module when they differ in environment"), and
    the environment column is what makes them two gates instead of one run
    twice.
    """
    ours = [(shard, env) for shard, module, env in _gate_entries()
            if module == MODULE]
    assert len(ours) == 2, ours
    assert all(shard == "gpu" for shard, _env in ours), ours
    assert sorted(env for _shard, env in ours) == [("LEG=A",), ("LEG=B",)]


def test_no_committed_case_file_names_a_machine() -> None:
    """No user path, no scratchpad path, no drive letter.

    The gate was developed with ``GPUWM_CASE_DATA_ROOT`` pointing at a
    home directory and the configs carried that path literally.  A
    committed copy that keeps it is unusable on every other box and leaks
    the author's filesystem into the repository.
    """
    forbidden = ("C:/", "C:\\", "/Users/", "\\Users\\", "AppData",
                 "scratchpad", "Temp/", "/home/")
    for path in sorted(CASES.glob("shape_tiles_*")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} names {token!r}"
        assert "${GPUWM_CASE_DATA_ROOT}" in text or "geog_root" not in text, \
            f"{path.name} sets geog_root without the case-data root variable"


@pytest.mark.parametrize("template", [LEG_A, LEG_B])
def test_a_leg_template_commits_no_card_budget(template: pathlib.Path) -> None:
    """``vram_budget_bytes`` stays a placeholder in the tree.

    It is a fact about a card.  A number here would be read as a
    requirement by the next person to run the gate on a different box, and
    would silently constrain the tile search to the wrong budget.
    """
    text = template.read_text(encoding="utf-8")
    assert PLACEHOLDER in text
    tiles = _loaded(template)["tiles"]
    assert tiles["vram_budget_bytes"] == 1073741824, \
        "the placeholder is not the value the gate substitutes"


def test_leg_a_is_a_streamed_parent_over_a_resident_child() -> None:
    config = _loaded(LEG_A)
    assert config["experiment"]["feedback"] == 1, "leg A must be two-way"
    assert config["tiles"]["mode"] == "on"
    child = config["domain"][1]
    assert child["grid_id"] == 2 and child["parent_id"] == 1
    assert child["tiles"] == {"mode": "off"}, \
        "this control intentionally keeps d02 resident while d01 streams"


def test_leg_b_is_a_resident_parent_over_a_streamed_child() -> None:
    """The roles-flipped twin, and the shape the rider was added for.

    The tree-wide table is OFF and carries only the budget: a per-domain
    table may not name the card, so that is the only place the budget can
    live.  That combination -- tree-wide off, per-domain on -- is exactly
    what used to produce an empty receipt while a grid streamed.
    """
    config = _loaded(LEG_B)
    assert config["experiment"]["feedback"] == 1, "leg B must be two-way"
    assert config["tiles"]["mode"] == "off"
    assert "vram_budget_bytes" in config["tiles"], \
        "the tree-wide table carries the budget even when its mode is off"
    child = config["domain"][1]
    assert child["grid_id"] == 2 and child["parent_id"] == 1
    assert child["tiles"] == {"mode": "on"}


def test_the_two_legs_differ_only_in_which_end_streams() -> None:
    """A confound check: same tree, same physics, opposite roads.

    If the two legs drifted apart in anything else, a difference between
    them would stop being attributable to the streaming road, which is the
    only thing they exist to compare.
    """
    a, b = _loaded(LEG_A), _loaded(LEG_B)
    for section in ("projection", "shared"):
        assert a[section] == b[section], section
    assert a["experiment"] == b["experiment"]
    assert [d.get("nx") for d in a["domain"]] == [d.get("nx") for d in b["domain"]]
    assert a["tiles"]["mode"] != b["tiles"]["mode"]
    assert a["domain"][1]["tiles"] != b["domain"][1]["tiles"]


def test_the_control_carries_no_tiles_table_anywhere() -> None:
    """The all-resident reference both legs are compared against.

    A control with a ``[tiles]`` table would be comparing streaming
    against streaming, which is the shape of every false pass this gate
    exists to prevent.
    """
    config = tomllib.loads(CONTROL.read_text(encoding="utf-8"))
    assert "tiles" not in config
    assert all("tiles" not in domain for domain in config["domain"])
    assert config["experiment"]["feedback"] == 1, \
        "the control must be the same two-way tree as the legs"


def test_the_runner_drives_the_documented_front_door() -> None:
    """``woof run <config> --outdir <dir>``, with no workaround flags.

    Asserted on the source because running it needs a card and a bundle.
    The whole claim of this gate is the ARGUMENT LIST -- a leg that reached
    the same shape through an internal entry point, or that needed one
    extra flag, would prove the opposite of what it says it proves.
    """
    source = RUNNER.read_text(encoding="utf-8")
    assert '"run", str(config),' in source
    assert '"--outdir", str(outdir)' in source
    # The receipt check, which is what separates "streamed" from "ran".
    assert "_streamed_any" in source
    assert "streamed_any" in source
