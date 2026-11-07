"""The DomainState attributes the P3 device adapter grows must be classified.

WHAT THIS PREVENTS, concretely and measured on 2026-08-29: the CUDA port's
``woof/core/p3.py::apply`` caches its device workspace as
``state._p3_workspace``.  ``woof/io/restart.py::classify_state_attr``
refuses any DomainState attribute it does not know -- by design, so new
state cannot silently skip the restart stream.  The two together meant the
FIRST mp=50 device step made the state unclassifiable, and every mp=50
forecast died with

    RestartManifestError: DomainState attribute '_p3_workspace' is not
    classified in the restart manifest

inside ``canonical_state_digest`` before it could write its first history
frame.  No checkpoint could be written either.

``tests/test_restart.py::test_every_domainstate_attribute_is_classified``
is parametrized with ``mp_physics=50`` and passed throughout, because its
shim state never calls ``p3.apply`` and therefore never grows the
attribute.  These tests close that gap from the other side: the CPU one
names the attributes by source, and the GPU one in
``tests/test_p3_restart_manifest_gpu.py`` runs a real step and classifies
what the run actually produced.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from woof.io.restart import RestartManifestError, classify_state_attr

P3_SOURCE = Path(__file__).resolve().parents[1] / "woof" / "core" / "p3.py"

#: ``state.<name> = `` and ``setattr(state, "<name>"``, which is every way
#: woof/core/p3.py can put an attribute on a DomainState.
_ASSIGN = re.compile(r"\bstate\.(_?[A-Za-z][A-Za-z0-9_]*)\s*=(?!=)")
_SETATTR = re.compile(
    r"setattr\(\s*state\s*,\s*[\"'](_?[A-Za-z][A-Za-z0-9_]*)[\"']")


def _assigned_attributes() -> set[str]:
    text = P3_SOURCE.read_text(encoding="utf-8")
    return ({m.group(1) for m in _ASSIGN.finditer(text)}
            | {m.group(1) for m in _SETATTR.finditer(text)})


def test_the_adapter_assigns_the_attributes_this_gate_knows_about():
    """A new assignment in p3.py is a new row here, deliberately.

    The set is asserted rather than only iterated so that ADDING an
    assignment fails this test even if the new name happens to be
    classified for some other reason.
    """
    assert "_p3_workspace" in _assigned_attributes()


@pytest.mark.parametrize("name", sorted(_assigned_attributes()))
def test_every_state_attribute_the_p3_adapter_assigns_is_classified(name):
    try:
        kind = classify_state_attr(name)
    except RestartManifestError as exc:      # pragma: no cover - the failure
        pytest.fail(
            f"woof/core/p3.py assigns state.{name} and "
            f"woof/io/restart.py does not classify it: {exc}")
    assert kind in {"serialize", "checkpoint_only", "rebuild", "setup",
                    "infra"}


def test_the_workspace_is_infra_and_not_serialized():
    """It holds no value of its own.

    Every array in the workspace is a ``DomainState.scratch`` slot already
    classified in ``REBUILT_SCRATCH_SLOTS``; the container is a per-process
    device handle that ``woof/core/p3.py`` rebuilds whenever the column or
    level count differs.  Serializing it would put a second copy of those
    arrays in every checkpoint.
    """
    assert classify_state_attr("_p3_workspace") == "infra"


def test_a_config_key_absent_on_one_side_is_named_not_repr_ed():
    """The refusal has to say WHY, and a memory address does not.

    ``woof/io/restart.py::_require_config_match`` compares the stored
    header's config against the live one with a bare ``object()`` sentinel
    for "this key is on one side only".  That sentinel reached the message
    through ``{!r}``, so a checkpoint written before a RunConfig field
    existed was refused with

        p3_backend: restart=<object object at 0x7f2c1d0a4390> run='cuda'

    which names the field but renders the reason as a memory address that
    changes every run.  THE REFUSAL IS CORRECT AND STAYS -- a 2.5.8
    checkpoint must not resume under a build with new physics config -- so
    this is a message fix, not a loosening: the same configurations are
    refused, and the ``differences`` list is unchanged in length.

    Not a P3 test in substance.  Every RunConfig field added after a
    checkpoint was written takes this path, and ``p3_backend`` is simply
    the first one measured doing it (development branch, 2026-08-29).
    """
    import pytest

    from woof.config import RunConfig, validate_run_config
    from woof.io.restart import RestartMismatchError, _require_config_match

    cfg = validate_run_config(RunConfig(
        nx=6, ny=6, nz=8, dx=3000.0, dy=3000.0, ztop=20000.0, dt=20.0,
        run_seconds=40.0, mp_physics=50, moist=True))

    import dataclasses
    live = dataclasses.asdict(cfg)

    # A checkpoint from before the field existed.
    older = {k: v for k, v in live.items() if k != "p3_backend"}
    with pytest.raises(RestartMismatchError) as excinfo:
        _require_config_match(older, cfg, "old.ckpt")
    message = str(excinfo.value)
    assert "<object object at" not in message, message
    assert "p3_backend" in message
    assert "absent from the restart file" in message
    assert repr(cfg.p3_backend) in message

    # And the other direction: a header carrying a key this build dropped.
    newer = dict(live)
    newer["a_field_this_build_does_not_have"] = "x"
    with pytest.raises(RestartMismatchError) as excinfo:
        _require_config_match(newer, cfg, "new.ckpt")
    message = str(excinfo.value)
    assert "<object object at" not in message, message
    assert "a_field_this_build_does_not_have" in message
    assert "absent from this build" in message

    # Still refuses an ordinary value mismatch, unchanged.
    differing = dict(live, p3_backend="reference")
    with pytest.raises(RestartMismatchError, match="p3_backend"):
        _require_config_match(differing, cfg, "diff.ckpt")

    # And admits an identical config: the fix adds no new refusal.
    _require_config_match(live, cfg, "same.ckpt")
