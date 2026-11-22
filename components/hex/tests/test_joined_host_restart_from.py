"""A host the door started for --hosts resumes `--restart-from` from its own copy of the seal.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): from the 0.6.2 wheel on a clean RTX PRO 6000 box, the 120 km
globe on `--hosts local0:0,local1:0` sealed at 3 h (its words equal to one host's, 0 of 706,509,363), and the resume
with `--restart-from OUT/checkpoint.f003.pkl` stopped on host 1: "1 of its 2 rank shards are not on this host
(rank0001.pkl first) and it has no manifest mirror". The door starts every other host with --out
OUT/.multihost/host<N>, so that host sealed its shard and rank 0's manifest mirror there, but the --restart-from
reader took rank 0's path as given; only the moving carrier looked for the host's own copy. An ssh host on its own
disk has no file at rank 0's path at all, and the driver's argument check refused it before that.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from woof.hex import multihost  # noqa: E402
from woof.hex.mpasphys.checkpoint_v842 import resolve_route  # noqa: E402
from test_mpasphys_checkpoint_v842 import TEXT, write_checkpoint  # noqa: E402


def _seal_on_host_one(tmp_path: Path, *, keep_rank_zero_copy: bool) -> tuple[Path, Path]:
    sealed = tmp_path / "seal"
    sealed.mkdir()
    path, *_ = write_checkpoint(sealed)
    mine = sealed / ".multihost" / "host1" / path.name
    mine.parent.mkdir(parents=True)
    for suffix in ("", ".json"):
        source = Path(str(path) + suffix)
        (shutil.copy2 if keep_rank_zero_copy else shutil.move)(source, Path(str(mine) + suffix))
    return path, mine


@pytest.mark.parametrize("keep_rank_zero_copy", [True, False], ids=["virtual-host", "own-disk"])
def test_a_joined_host_resumes_from_its_own_copy(tmp_path, monkeypatch, keep_rank_zero_copy) -> None:
    given, mine = _seal_on_host_one(tmp_path, keep_rank_zero_copy=keep_rank_zero_copy)
    monkeypatch.setenv(multihost.ENV_JOIN, json.dumps(dict(host=1, address="x", port=1, token="t")))
    route, _ = resolve_route(SimpleNamespace(restart_from=given, history_every_minutes=6), TEXT)
    assert Path(route.carrier_path) == mine.resolve()


def test_rank_zero_and_one_host_keep_the_given_path(tmp_path, monkeypatch) -> None:
    given, mine = _seal_on_host_one(tmp_path, keep_rank_zero_copy=True)
    monkeypatch.delenv(multihost.ENV_JOIN, raising=False)
    monkeypatch.delenv(multihost.ENV_HOST_INDEX, raising=False)
    route, _ = resolve_route(SimpleNamespace(restart_from=given, history_every_minutes=6), TEXT)
    assert Path(route.carrier_path) == given.resolve()


def test_the_driver_accepts_a_joined_host_whose_disk_holds_only_its_copy(tmp_path, monkeypatch) -> None:
    from woof.hex.drivers import run_cuda_v841_forecast as driver

    given, mine = _seal_on_host_one(tmp_path, keep_rank_zero_copy=False)
    monkeypatch.setenv(multihost.ENV_JOIN, json.dumps(dict(host=1, address="x", port=1, token="t")))
    source = Path(driver.__file__).read_text()
    # the argument check names the joined host's copy before it refuses a missing --restart-from
    check = source[source.index('if args.restart_from is not None and not Path(args.restart_from).is_file():'):]
    check = check[:check.index('parser.error(f"--restart-from')]
    assert "joined_carrier" in check
    from woof.hex.mpasphys.checkpoint_shards_v842 import joined_carrier
    assert Path(joined_carrier(str(given))).is_file()
