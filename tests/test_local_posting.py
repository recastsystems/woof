"""A locally written source is published as an as-posted fetch the preparation reads.

The breakage: a WOOF-ArWen domain fed by a hex parent could only be prepared after
the whole parent run, because nothing wrote the posting folder that
``woof prep --as-posted`` waits on for a source that is produced rather than
downloaded.  These check the documents against the engine's own readers.
"""

from __future__ import annotations

from datetime import datetime
import json
import threading
import time

import pytest

from woof import local_posting
from woof.ingest.boundary_stream import PostedLeads, SourceBehind


def _posting(tmp_path, **kwargs):
    produced = tmp_path / "export"
    produced.mkdir()
    ready = produced / "export-ready.jsonl"
    posting = local_posting.LocalPosting(
        source="hex-global", cycle=datetime(2025, 5, 19, 12), leads=[0, 1, 2],
        out=tmp_path / "fetch", supplement_roles=["hex_in_band_surface"],
        ready_list=ready, poll_seconds=0.02, progress=lambda line: None, **kwargs)
    return posting, produced, ready


def _produce(produced, ready, posting, leads, delay=0.05):
    for lead in leads:
        time.sleep(delay)
        frame = produced / posting.name(lead)
        frame.write_bytes(f"frame {lead}".encode())
        with ready.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"frame": str(frame), "sha256": None}) + "\n")


def test_each_lead_is_published_in_order_and_read_by_the_preparation_s_wait(tmp_path):
    posting, produced, ready = _posting(tmp_path)
    producer = threading.Thread(target=_produce, args=(produced, ready, posting, [0, 1, 2]))
    producer.start()
    assert posting.run() == 0
    producer.join()
    folder = tmp_path / "fetch" / "posting"
    schedule = json.loads((folder / "schedule.json").read_text(encoding="utf-8"))
    assert [row["state"] for row in schedule["leads"]] == ["ready"] * 3
    assert schedule["cycle"] == "2025-05-19T12"
    waiter = PostedLeads(folder, source="hex-global", cycle="2025-05-19T12",
                         poll_seconds=0.01)
    assert len(waiter.route_table_sha256()) == 64
    for lead in (0, 1, 2):
        marker = waiter.wait(lead)
        (item,) = marker["objects"]
        assert item["name"] == f"frames/{posting.name(lead)}"
        assert (tmp_path / "fetch" / item["name"]).read_bytes() == f"frame {lead}".encode()
    manifest = json.loads((tmp_path / "fetch" / "fetch-manifest.json").read_text(encoding="utf-8"))
    assert manifest["prep"]["primary_files"] == [f"frames/{posting.name(lead)}" for lead in (0, 1, 2)]
    assert manifest["prep"]["supplement_files"] == manifest["prep"]["primary_files"]


def test_the_prep_flags_name_every_planned_file_and_supplement(tmp_path):
    posting, _produced, _ready = _posting(tmp_path)
    flags = posting.prep_arguments()
    assert flags.count("--input") == 3 and flags.count("--supplement") == 3
    assert flags[1].replace("\\", "/").endswith("frames/wrfout_d01_2025-05-19_12_00_00")


def test_a_quiet_producer_fails_the_wait_as_source_behind(tmp_path):
    posting, produced, ready = _posting(tmp_path, quiet_minutes=0.002)
    _produce(produced, ready, posting, [0], delay=0)
    assert posting.run() == 75
    waiter = PostedLeads(tmp_path / "fetch" / "posting", source="hex-global",
                         cycle="2025-05-19T12", poll_seconds=0.01)
    waiter.wait(0)
    with pytest.raises(SourceBehind):
        waiter.wait(1)


def test_hex_global_may_be_prepared_as_it_posts():
    from woof.source_cli import as_posted_refusal

    assert as_posted_refusal("hex-global") is None
    assert as_posted_refusal("hex-regional") is None


def test_the_command_line_parses_and_prints_its_prep_flags(tmp_path, capsys):
    # Box G 2026-10-23: the --name-template help carried the strftime %Y unescaped, and
    # argparse (3.14 checks help at add_argument) refused to build the parser at all.
    from woof import local_posting
    local_posting._parser().format_help()
    assert local_posting.main(["--source", "hex-global", "--cycle", "2025-05-19T12", "--leads", "0-1",
                               "--out", str(tmp_path), "--ready-list", str(tmp_path / "r.jsonl"),
                               "--print-prep-arguments"]) == 0
    assert "--input" in capsys.readouterr().out


def test_a_parent_posting_beside_its_nest_needs_no_later_lead_for_inland_water():
    # Box G 2026-10-23: the inland-water daily mean read every lead of the first 24 h, so the
    # as-posted head of a nest under a running hex parent waited for the whole parent forecast.
    from types import SimpleNamespace
    from woof.ingest.horiz import with_inland_air_temperature
    from woof.source_adapters import get_source_adapter
    assert get_source_adapter("hex-global").inland_air_mean_hours == 0.0
    assert get_source_adapter("gfs").inland_air_mean_hours is None
    statics = SimpleNamespace(inland_air_temperature=None)
    series = object()
    assert with_inland_air_temperature(statics, series, hours=0.0) is statics
