# SPDX-License-Identifier: Apache-2.0
"""WOOF_HEX_NO_LOCAL_GPU (or GPUWM_NO_LOCAL_GPU) keeps every differentiable-engine door off the card: the door
refuses by name ("no local GPU") before it reads any card, finds a runner or fetches an asset (host-only)."""
import argparse

import pytest

from woof.hex import ad_doors as d


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _no_card_reads(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("args"))
        raise AssertionError(f"a card was read: {calls[-1]}")

    def forbidden(name):
        def call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} ran under no local GPU")
        return call

    monkeypatch.setattr(d.subprocess, "run", run)
    monkeypatch.setattr(d.subprocess, "Popen", forbidden("Popen"))
    monkeypatch.setattr(d, "resolve_runner", forbidden("resolve_runner"))
    monkeypatch.setattr(d, "resolve_assets", forbidden("resolve_assets"))
    return calls


@pytest.mark.parametrize("variable", ["WOOF_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"])
@pytest.mark.parametrize("door", d.DOORS)
def test_every_door_refuses_no_local_gpu_before_any_card_runner_or_asset(door, variable, tmp_path, monkeypatch):
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "1")
    calls = _no_card_reads(monkeypatch)
    args = _parse([door, "--out", str(tmp_path / "out")])
    with pytest.raises(d.AdDoorError, match=f"^no local GPU: {variable} is set"):
        d.Context(args, door)
    assert calls == []                                   # no nvidia-smi, no runner, no asset fetch
    assert not (tmp_path / "out").exists()


def test_card_readers_and_the_window_launch_refuse_under_no_local_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_NO_LOCAL_GPU", "1")
    calls = _no_card_reads(monkeypatch)
    for read in (lambda: d.free_mib("0"), lambda: d.total_mib("0"), lambda: d.compute_capability("0"),
                 lambda: d.process_card_mib(1), lambda: d.admit_card("0", 1, 0.0),
                 lambda: d.run_window(["runner"], {}, tmp_path, tmp_path / "w", "0")):
        with pytest.raises(d.AdDoorError, match="^no local GPU"):
            read()
    assert calls == [] and not (tmp_path / "w").exists()


def test_table_refusals_still_answer_first_and_zero_means_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_NO_LOCAL_GPU", "1")
    _no_card_reads(monkeypatch)
    with pytest.raises(d.AdDoorError, match="has no row"):          # card-free refusals keep their own words
        d.Context(_parse(["tangent", "--row", "nosuch", "--out", str(tmp_path / "o")]), "tangent")
    monkeypatch.setenv("WOOF_HEX_NO_LOCAL_GPU", "0")
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    d.refuse_no_local_gpu()                                          # "0" and unset are not a declaration
