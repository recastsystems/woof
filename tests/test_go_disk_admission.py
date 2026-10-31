"""`woof go`'s own GFS chain asks run-plan's disk admission before it claims a folder or fetches.

The chain `woof go` runs by itself for GFS enters no run plan, so the
admission `woof run-plan` gives every run (download, preparation,
history, checkpoints and pictures against the free space of the disks
that hold them) was never asked on it: a drive with no room still had
its run folder claimed and its download started.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from woof import capabilities, disk_budget, go_cli
from woof.cli import build_parser
from tests.test_go_chain_events import gfs_config, staged_geog  # noqa: F401  (fixtures)


class _StageReached(Exception):
    """The chain got as far as starting a stage."""


def _go(tmp_path, monkeypatch, config, geog, *extra, free, download_free=None,
        seen=None):
    """Run the chain until its first stage; ``(stages started, refusal text)``.

    ``free`` is the free space of the run's disk and ``download_free``,
    when given, of a separate disk the download is sent to with
    ``--data-dir``.  ``seen`` collects the projections the admission
    compared, when a test wants to read them.
    """

    download = tmp_path / "downloads-elsewhere"
    argv = ["go", str(config), "--outdir", str(tmp_path / "case"),
            "--geog-root", str(geog), "--no-memory-gate", *extra]
    if download_free is not None:
        argv += ["--data-dir", str(download)]
    args = build_parser().parse_args(argv)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "missing", lambda *a, **k: [])
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: None)
    monkeypatch.setattr(go_cli, "resolve_bridge", lambda: tmp_path / "unused-bridge")

    def space(path):
        path = Path(path)
        if download_free is not None and (path == download or download in path.parents):
            return download_free
        return free

    monkeypatch.setattr(disk_budget, "free_bytes", space)
    if download_free is not None:
        monkeypatch.setattr(disk_budget, "same_disk", lambda first, second: False)
    if seen is not None:
        compare = disk_budget.disk_refusal

        def recorded(projection, run_free, **kwargs):
            seen.append(projection)
            return compare(projection, run_free, **kwargs)

        monkeypatch.setattr(disk_budget, "disk_refusal", recorded)
    stages: list[str] = []

    def stage(label, *_args, **_kwargs):
        stages.append(label)
        raise _StageReached(label)

    monkeypatch.setattr(go_cli, "_run_stage", stage)
    try:
        go_cli.go_main(args)
    except go_cli.GoRefusal as error:
        return stages, str(error)
    except _StageReached:
        pass
    return stages, None


def test_a_disk_with_no_room_is_refused_before_the_folder_or_the_fetch(
        tmp_path, monkeypatch, gfs_config, staged_geog):
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog, free=0)
    assert stages == [], f"a disk with no room still started {stages}"
    assert refused is not None
    assert "disk that holds its run directory has 0.0 GiB free" in refused
    assert "Refused before the download, so nothing was spent" in refused
    # No run folder was claimed and no download cache was made.
    assert not (tmp_path / "case").exists()


def test_a_disk_with_room_goes_on_to_the_first_stage(
        tmp_path, monkeypatch, gfs_config, staged_geog):
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog,
                          free=10 ** 15)
    assert refused is None
    assert stages == ["authority"]


@pytest.mark.parametrize("flags", [(), ("--prepare-only",)])
@pytest.mark.parametrize("same_disk", [True, False])
def test_hydro_is_priced_and_never_built_on_disk_refusal(
        tmp_path, monkeypatch, gfs_config, staged_geog, flags, same_disk):
    from woof.hydro import runtime
    built = []
    monkeypatch.setattr(disk_budget, "same_disk", lambda *a: same_disk)
    monkeypatch.setattr(runtime, "hydro_disk_estimate", lambda *a: (2**40, "hydro fake: 1024 GiB"))
    monkeypatch.setattr(runtime, "_launch_settings", lambda *a, **k: built.append(k))
    # Hydro is default-off. Ask for the subsystem whose disk admission
    # this test must exercise before any preparation is built.
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog,
                          "--hydro", "on", *flags, free=2**30)
    assert not stages and not built
    assert "Refused before the download and before hydro preparation, so nothing was spent." in refused
    assert "hydro fake: 1024 GiB" in refused
    assert not (tmp_path / "case").exists()
    (tmp_path / "exact-hydro-refusal.txt").write_text(refused)


def test_a_section_request_is_admitted_with_its_line(
        tmp_path, monkeypatch, gfs_config, staged_geog):
    """The run described to the admission carries the section line.

    The admission builds a run plan of this chain, and a plan naming an
    ``xsec:`` product with no line is refused when it is built; without
    the line a request `woof go` itself had admitted stopped there.
    """

    stages, refused = _go(
        tmp_path, monkeypatch, gfs_config, staged_geog,
        "--products", "composite_reflectivity,xsec:QCLOUD=0.01,0.1/wa",
        "--section=38.3,-99.0,38.3,-98.4", free=10 ** 15)
    assert refused is None
    assert stages == ["authority"]


def test_a_download_disk_with_no_room_is_named_and_refused(
        tmp_path, monkeypatch, gfs_config, staged_geog):
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog,
                          free=10 ** 15, download_free=0)
    assert stages == []
    assert refused is not None and "downloads about" in refused
    assert "0.0 GiB free" in refused
    assert not (tmp_path / "case").exists()
    assert not (tmp_path / "downloads-elsewhere").exists()


@pytest.mark.parametrize('flags,kept,pictures', [
    pytest.param(("--keep-checkpoints", "0", "--products", "none"), 0, False, id='flags1-0-False'),
])
def test_the_admission_prices_the_run_this_chain_makes(
        tmp_path, monkeypatch, gfs_config, staged_geog, flags, kept, pictures):
    seen: list[dict] = []
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog, *flags,
                          free=10 ** 15, seen=seen)
    assert refused is None and stages == ["authority"]
    assert len(seen) == 1, "the chain asked the admission once, before its first stage"
    projection = seen[0]
    # The download the fetch stage makes, into the managed cache under
    # the folder named on the command line, and its preparation.
    assert projection["download_bytes"] > 0
    assert projection["preparation_bytes"] > 0
    assert Path(projection["download_dir"]).parent == tmp_path / "case" / "downloads"
    # The checkpoint sets this chain keeps and the pictures it draws.
    assert projection["keep_checkpoints"] == kept
    assert (projection["picture_bytes"] > 0) is pictures


def _staging(monkeypatch, stream: int, certain: int) -> None:
    """Every preparation stages a frame stream of ``stream`` bytes, ``certain`` of them whatever the window.

    No source ``woof go``'s own chain drives composes one today, so the
    stream is handed to the pricing it is read from; everything after that
    (the folder, its disk, the refusal and the warning) is the admission's
    own.
    """

    from woof import download_budget

    def estimate(exp, *, chain, source, forcing_times, points=None):
        return {"bytes": stream, "min_bytes": certain, "max_bytes": stream,
                "valid_times": 3, "per_valid_time": stream // 3, "source": source,
                "composes": True, "basis": "a stream this test stages"}

    monkeypatch.setattr(download_budget, "compose_scratch_estimate", estimate)


def test_the_chain_refuses_a_frame_stream_its_disk_cannot_hold_before_the_folder(
        tmp_path, monkeypatch, gfs_config, staged_geog):
    monkeypatch.delenv("WOOF_COMPOSE_SCRATCH", raising=False)
    _staging(monkeypatch, 10 ** 15, 10 ** 15)
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog, free=10 ** 14)
    assert stages == []
    assert refused is not None and "stages its decoded frame stream" in refused
    # Measured beside the folder this chain's preparation writes.
    assert f"in a scratch folder in {(tmp_path / 'case').resolve()}" in refused
    assert "WOOF_COMPOSE_SCRATCH" in refused
    assert "Frame stream: a stream this test stages." in refused
    assert not (tmp_path / "case").exists()


def test_the_chain_refuses_a_scratch_variable_naming_no_folder_before_the_folder(
        tmp_path, monkeypatch, gfs_config, staged_geog):
    missing = tmp_path / "not-made"
    monkeypatch.setenv("WOOF_COMPOSE_SCRATCH", str(missing))
    _staging(monkeypatch, 10 ** 9, 10 ** 9)
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog, free=10 ** 15)
    assert stages == []
    assert refused is not None
    assert f"WOOF_COMPOSE_SCRATCH={missing} does not name an existing folder" in refused
    assert not (tmp_path / "case").exists()


def test_the_chain_says_a_stream_that_may_not_fit_and_goes_on(
        tmp_path, monkeypatch, capsys, gfs_config, staged_geog):
    monkeypatch.delenv("WOOF_COMPOSE_SCRATCH", raising=False)
    _staging(monkeypatch, 10 ** 15, 1)
    stages, refused = _go(tmp_path, monkeypatch, gfs_config, staged_geog, free=10 ** 14)
    assert refused is None and stages == ["authority"]
    out = capsys.readouterr().out
    assert "go: WARNING -- May not fit: this run's preparation stages" in out
    assert "Frame stream: a stream this test stages." in out
