"""``woof cycle`` refuses invalid combinations before it writes anything."""

import pytest

from woof.cli import build_parser
from woof.cycle.cli import cycle_main
from woof.cycle.contracts import CycleRefusal


def _args(tmp_path, argv):
    parser = build_parser()
    base = ["cycle", "--root", str(tmp_path), "--epoch-anchor",
            "2026-08-14T18:00:00Z", "--parent-kind", "replay"]
    return parser.parse_args(base + argv)


def test_cycle_is_registered_on_the_front_door():
    parser = build_parser()
    args = parser.parse_args(["cycle", "--root", ".", "--epoch-anchor",
                              "2026-08-14T18:00:00Z", "--cycle-seconds",
                              "960", "--cycles", "3", "--parent-kind",
                              "replay"])
    assert args.command == "cycle"
    assert callable(args.func)
    assert args.resume is True
    assert args.max_forecast_only_cycles == 3
    assert args.allow_placement_clamp is False
    assert args.placement_provider == "none"


def test_dry_run_prints_the_lattice_and_writes_nothing(tmp_path, capsys):
    root = tmp_path / "run"
    args = _args(root, ["--cycle-seconds", "960", "--cycles", "3",
                        "--child-slots", "2", "--child-dt-seconds", "30",
                        "--dry-run"])
    assert cycle_main(args) == 0
    out = capsys.readouterr().out
    assert "boundary lattice" in out
    assert "2026-08-14T18:48:00+00:00" in out      # boundary 3 = 2880 s
    assert "4 child steps per parent step" in out
    assert not root.exists()


def test_dry_run_refuses_a_cycle_that_is_not_whole_parent_steps(tmp_path):
    # The parent dt is explicit: since 98e5f5ac4e9 (development branch
    # change) the DEFAULT resolves to a step that divides the cycle (60 s
    # for 900 s), and only a declared --parent-dt-seconds that does not
    # divide it is refused.
    args = _args(tmp_path, ["--cycle-seconds", "900", "--cycles", "3",
                            "--parent-dt-seconds", "120", "--dry-run"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    assert excinfo.value.observed["remainder_ticks"] == 60000


def test_dry_run_refuses_a_child_step_that_does_not_divide(tmp_path):
    args = _args(tmp_path, ["--cycle-seconds", "960", "--cycles", "1",
                            "--child-slots", "1", "--child-dt-seconds", "7",
                            "--dry-run"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    assert excinfo.value.observed["remainder"] == 120000 % 7000


def test_the_placement_seam_now_resolves(tmp_path):
    """The seam the CLI refuses on is joined; an unknown kind still refuses.

    This test asserted the opposite until the lanes were integrated: the
    spine landed before woof.cycle.placement existed, so "the module is
    absent" was the correct observed behaviour then.  Now that the
    placement lane and woof.cycle.engine have landed, the factory
    resolves, and what must still refuse by name is an unknown provider.
    """
    from woof.cycle.placement import build_placement_provider

    assert callable(build_placement_provider)
    with pytest.raises(CycleRefusal) as excinfo:
        build_placement_provider(kind="haruspicy", child_slots=1)
    assert excinfo.value.observed["placement_provider"] == "haruspicy"
    assert "tracker" in excinfo.value.observed["known"]


def test_replay_kind_runs_the_lattice_end_to_end(tmp_path, capsys):
    """The replay kind cycles a REAL parent, not a pair of integers.

    This test used to pass against a closure that returned
    ``{"parent_ticks": ..., "anchor_ticks": ...}`` and wrote nothing: the
    ledger advanced, receipts appeared, and the root held no anchor at
    all.  That is what "runs end to end" meant, and it is why the
    ingestion gate was never once reached through the front door.  The
    anchor assertion is what makes the name true.
    """
    from test_cycle_cli_frontdoor import _series

    root = tmp_path / "run"
    args = _args(root, ["--cycle-seconds", "960", "--cycles", "2",
                        "--parent-state", _series(tmp_path),
                        "--parent-mesh-id", "t48x48"])
    assert cycle_main(args) == 0
    from woof.cycle.ledger import CycleLedger
    assert CycleLedger(root).state()["last_completed_cycle"] == 2
    assert (root / "cycle_002" / "RECEIPT.json").is_file()
    assert sorted((root / "anchors").glob("*")), (
        "a cycle that leaves no anchor did not cycle a parent")
    assert "completed [1, 2]" in capsys.readouterr().out


def test_a_run_without_parent_state_writes_no_receipt(tmp_path):
    """The old end-to-end shape must now REFUSE, not receipt an empty root."""
    root = tmp_path / "run"
    args = _args(root, ["--cycle-seconds", "960", "--cycles", "2"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    assert "--parent-state" in str(excinfo.value)
    assert not (root / "cycle_002" / "RECEIPT.json").exists()


def test_a_model_parent_with_no_port_refuses_rather_than_pretending(tmp_path):
    """The adapter LANDED, so the refusal moved -- it must move accurately.

    This test used to assert ``"no engine adapter"``, which was the right
    thing to say while the closed-loop lane's capability was unmerged.
    It is now the WRONG thing to say: the tree can run the port's dycore.
    A refusal that tells a user a working feature is missing is its own
    defect, so what is asserted here is the refusal a model parent
    actually deserves -- name the port plumbing it was not given.
    """
    parser = build_parser()
    args = parser.parse_args(["cycle", "--root", str(tmp_path),
                              "--epoch-anchor", "2026-08-14T18:00:00Z",
                              "--parent-kind", "mpas-cuda",
                              "--cycle-seconds", "960", "--cycles", "1"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    assert excinfo.value.observed["parent_kind"] == "mpas-cuda"
    assert "no engine adapter" not in str(excinfo.value)
    assert excinfo.value.observed["missing"] == ["--port-root",
                                                "--port-config",
                                                "--port-steps"]
    # Refused at plan time: no ledger was opened, so the root is clean.
    assert not (tmp_path / "cycle_ledger.jsonl").exists()


def test_a_bad_epoch_anchor_names_what_it_got(tmp_path):
    args = _args(tmp_path, ["--cycle-seconds", "960", "--cycles", "1",
                            "--dry-run"])
    args.epoch_anchor = "yesterday"
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    assert excinfo.value.observed["epoch_anchor"] == "yesterday"
