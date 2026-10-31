"""The HRRR intermediate's own fetch asks for every record its decoder reads (mesh-gen-fast, 2026-10-29)."""
import pytest

from woof.hex import hrrr_intermediate as hi


def _hrrr_row():
    return hi.source_row("hrrr")


def test_fetch_is_asked_for_the_decoder_extras():
    # polygon from HRRR 2026-10-28T12 (an RTX 4090 host): the fetch took the plain 561-record subset and the decoder refused
    # "missing required field QNC hybrid level 1".
    row = _hrrr_row()
    assert set(row.decoder_extras) >= {"QNC", "QNI", "QNR", "RH2"}

    def fetch_hrrr(*, cycle, hours, analyzed_aerosol=False, extras=(), **kwargs):
        return None

    requests = hi._fetch_requests(fetch_hrrr, row)
    assert requests["extras"] == tuple(row.decoder_extras)
    assert requests.get("analyzed_aerosol", False) == bool(row.analyzed_aerosol)


def test_a_fetch_that_cannot_take_the_extras_is_refused_by_name():
    row = _hrrr_row()

    def old_fetch(*, cycle, hours, area, out):
        return None

    with pytest.raises(hi.IntermediateRefusal, match="takes no"):
        hi._fetch_requests(old_fetch, row)


def test_the_engine_fetch_takes_them():
    fetch = pytest.importorskip("woof.fetch").fetch_hrrr
    assert hi._fetch_requests(fetch, _hrrr_row())["extras"]
