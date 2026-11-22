"""Every door that publishes a configuration answers the route question.

The domain, forcing, schedule and saved-setup editors, ``domain-fit``,
``domain-tiles`` and the following-nest authoring door are driven in
``test_companion_domains_route_companions.py``.  Four publishers are
left, and they are the ones a reader would not think to look for: the
bundle publisher two creation doors share, the local cycling publisher,
and the draft retained when a configuration is refused for memory.

Each is driven HERE at its own publisher function, over bytes the real
emission door wrote, with no catalog, no archive and no device: the
question the publisher has to answer -- what does this configuration's
route read beside it -- belongs to the configuration, not to the
selection, the recipe or the card that produced it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from woof.experiment import load_experiment
from woof.hrrr_route_inputs import route_input_paths, verify_round_trip

# Share the identical real emission; each caller still writes its own files.
from test_companion_domains_route_companions import _emit


def _fetch_source(config: Path):
    import tomllib

    return (tomllib.loads(config.read_text(encoding="utf-8")).get("fetch")
            or {}).get("source")


def _absent(config: Path):
    return sorted(role for role, path in route_input_paths(config).items()
                  if not path.is_file())


def _stage(tmp_path, config: Path, name: str):
    """What a creation door hands its publisher: the TOML and its WPS.

    The route's other files are deliberately left out of the stage, so
    what the publisher writes is what this asserts about.
    """

    stage = tmp_path / name
    stage.mkdir()
    destination = tmp_path / f"{name}.toml"
    (stage / destination.name).write_text(
        config.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    (stage / route_input_paths(destination)["wps_namelist"].name).write_text(
        route_input_paths(config)["wps_namelist"].read_text(encoding="utf-8"),
        encoding="utf-8", newline="\n")
    return stage, destination


def test_the_shared_bundle_publisher_writes_the_route_files(tmp_path):
    """The seam the case-catalog and research creators both publish through."""

    from woof.research_workspaces import _publish_bundle

    base = _emit(tmp_path)
    stage, destination = _stage(tmp_path, base, "bundle")
    published = _publish_bundle(
        stage, destination, exp=load_experiment(stage / destination.name),
        source=_fetch_source(base))

    assert _absent(destination) == []
    assert set(route_input_paths(destination).values()) <= set(published)
    paths = route_input_paths(destination)
    verify_round_trip(load_experiment(destination), paths["wps_namelist"],
                      paths["namelist_input"])


def test_the_shared_bundle_publisher_leaves_another_route_alone(tmp_path):
    from woof.research_workspaces import _publish_bundle

    base = _emit(tmp_path, source="gfs", name="global")
    stage, destination = _stage(tmp_path, base, "global-bundle")
    _publish_bundle(stage, destination,
                    exp=load_experiment(stage / destination.name),
                    source=_fetch_source(base))

    paths = route_input_paths(destination)
    assert paths["wps_namelist"].is_file()
    assert not paths["namelist_input"].exists()
    assert not paths["stock_namelist_input"].exists()
    assert not paths["target_domain"].exists()


def test_the_shared_bundle_publisher_cannot_be_asked_without_the_route(
        tmp_path):
    """A caller that does not answer the route question does not publish.

    Both parameters are keyword-only and neither has a default, so a
    publisher added later fails at its first call rather than shipping a
    configuration its route refuses.
    """

    from woof.research_workspaces import _publish_bundle

    signature = inspect.signature(_publish_bundle)
    for name in ("exp", "source"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    base = _emit(tmp_path)
    stage, destination = _stage(tmp_path, base, "unanswered")
    with pytest.raises(TypeError):
        _publish_bundle(stage, destination)
    assert not destination.exists()


def test_both_creation_doors_hand_the_publisher_their_configuration():
    """The two call sites answer it, rather than being able to omit it."""

    import ast

    from woof import case_catalog, research_workspaces

    for module in (case_catalog, research_workspaces):
        calls = [node for node in ast.walk(ast.parse(inspect.getsource(module)))
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "_publish_bundle"]
        assert calls, module.__name__
        for call in calls:
            assert {keyword.arg for keyword in call.keywords} >= {"exp", "source"}


def _retain(tmp_path, base: Path, name: str, monkeypatch):
    """The retained draft, through the real refusal that retains one."""

    from woof.configuration_recovery import (MemoryAdmissionError,
                                              RECOVERY_DIR_ENV,
                                              retain_final_candidate)

    stage, destination = _stage(tmp_path, base, name)
    directory = tmp_path / f"{name}-recovery"
    monkeypatch.setenv(RECOVERY_DIR_ENV, str(directory))
    error = MemoryAdmissionError("this box has no room for that tree",
                                 peak_envelope_bytes=1, budget_bytes=0)
    retain_final_candidate(
        error, text=(stage / destination.name).read_text(encoding="utf-8"),
        requested_path=destination, stage=stage)
    assert error.recovery_error is None, error.recovery_error
    assert error.recovery is not None
    return directory


def test_a_retained_draft_carries_the_route_files(tmp_path, monkeypatch):
    """Fit and Tile open this draft, so it has to be openable as a run."""

    base = _emit(tmp_path)
    directory = _retain(tmp_path, base, "refused", monkeypatch)

    draft = directory / "draft.toml"
    assert draft.is_file()
    assert _absent(draft) == []
    paths = route_input_paths(draft)
    verify_round_trip(load_experiment(draft), paths["wps_namelist"],
                      paths["namelist_input"])


def test_a_retained_draft_off_the_native_route_is_unchanged(
        tmp_path, monkeypatch):
    base = _emit(tmp_path, source="gfs", name="global")
    directory = _retain(tmp_path, base, "global-refused", monkeypatch)

    paths = route_input_paths(directory / "draft.toml")
    assert paths["wps_namelist"].is_file()
    assert not paths["namelist_input"].exists()
    assert not paths["target_domain"].exists()
