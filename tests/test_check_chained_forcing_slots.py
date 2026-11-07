"""`woof check` prices the forcing slots the chained `woof go` forecast holds.

Breakage this prevents: on a 96 GB RTX PRO 6000 (94.42 GiB free measured on
test host, 2026-10-19) `woof check` refused the full-grid CONUS HRRR recipe by
0.50 GiB because it priced every hourly hrrr-prs boundary interval on the
card, while `woof go` runs that one-domain, one-card forecast chained and
holds one forcing slot (go_cli memory gate, since 9deedbc5066).  The two
gates disagreed about one card.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from woof.ingest.boundary_stream import chained_resident_forcing_slots

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "recipes" / "conus_hrrr_configuration.toml"


def _exp(domains=1, devices_enabled=False):
    return SimpleNamespace(domains=[object()] * domains,
                           devices=SimpleNamespace(enabled=devices_enabled))


def test_one_domain_one_card_chained_holds_one_slot():
    assert chained_resident_forcing_slots(_exp(), environ={}) == 1


def test_whole_schedule_when_not_chained_nested_or_split():
    assert chained_resident_forcing_slots(_exp(), environ={"WOOF_CHAINED_PREP": "0"}) is None
    assert chained_resident_forcing_slots(_exp(domains=2), environ={}) is None
    assert chained_resident_forcing_slots(_exp(devices_enabled=True), environ={}) is None


def _check(env_extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **env_extra}
    return subprocess.run(
        [sys.executable, "-m", "woof", "check", str(RECIPE), "--devices", "1",
         "--free-gib", "94.42", "--vram-gib", "94.97", "--explain"],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)


def test_check_admits_full_hrrr_on_one_96gb_card_as_go_does():
    chained = _check({})
    assert chained.returncode == 0, chained.stdout + chained.stderr
    assert "fits the 93.92 GiB budget" in chained.stdout
    whole = _check({"WOOF_CHAINED_PREP": "0"})
    assert whole.returncode != 0, whole.stdout
    assert "EXCEEDS the 93.92 GiB budget" in whole.stdout
