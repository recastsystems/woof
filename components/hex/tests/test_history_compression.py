"""Which MPAS-route files are written lossless-compressed (woof.hex.history_compression).

Ruling 2026-10-28: "Lossless compression is the default". History and diagnostics files that hold one output time
each are compressed; init/restart/LBC/da_state keep their PnetCDF container, and a file later records append to
is never compressed (a compressed file is written whole).
"""
import pytest

from woof.hex import history_compression as hc

HOURLY = dict(output_interval="01:00:00", filename_template="history.$Y-$M-$D_$h.$m.$s.nc")


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    for name in (hc.ENVIRONMENT, hc.LEVEL_ENVIRONMENT, hc.THREADS_ENVIRONMENT):
        monkeypatch.delenv(name, raising=False)


def test_history_and_diagnostics_frames_are_compressed_by_default():
    assert hc.for_stream("output", HOURLY, append=False, record_index=0) == {"level": 4, "threads": 0}
    assert hc.for_stream("diagnostics", dict(HOURLY, filename_interval="output_interval"),
                         append=False, record_index=None) == {"level": 4, "threads": 0}


@pytest.mark.parametrize("name", ["restart", "input", "lbc_in", "da_state", "iau", "diag_ugwp"])
def test_streams_native_or_jedi_read_keep_their_container(name):
    assert hc.for_stream(name, HOURLY, append=False, record_index=0) is None


@pytest.mark.parametrize("options,append,record", [
    (dict(HOURLY, filename_interval="none"), False, 0),                         # one file for the whole run
    (dict(HOURLY, filename_interval="06:00:00"), False, 0),                     # six frames a file
    (dict(output_interval="00:01:00", filename_template="history.nc"), False, 0),  # no time in the name
    (HOURLY, True, 1),                                                          # an append
    (HOURLY, False, 1),                                                         # a record past the first
])
def test_a_file_records_append_to_is_never_compressed(options, append, record):
    assert hc.for_stream("output", options, append=append, record_index=record) is None


def test_knobs_are_arwens(monkeypatch):
    monkeypatch.setenv(hc.ENVIRONMENT, "none")
    assert hc.for_stream("output", HOURLY, append=False, record_index=0) is None
    monkeypatch.setenv(hc.ENVIRONMENT, "deflate")
    monkeypatch.setenv(hc.LEVEL_ENVIRONMENT, "12")
    monkeypatch.setenv(hc.THREADS_ENVIRONMENT, "6")
    assert hc.for_stream("output", HOURLY, append=False, record_index=0) == {"level": 9, "threads": 6}
    monkeypatch.setenv(hc.ENVIRONMENT, "zstd")
    with pytest.raises(ValueError, match="unknown history compression"):
        hc.for_stream("output", HOURLY, append=False, record_index=0)
