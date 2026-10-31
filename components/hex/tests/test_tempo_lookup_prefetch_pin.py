"""The TEMPO lookup read-ahead reads the files the suite row's backend reads, and is never held unused."""
from __future__ import annotations

from types import SimpleNamespace

from woof.hex.mpasphys.route_table import framework_entry
from woof.hex.mpasphys.rrfs import framework_init_rrfs as fi


def _stub(monkeypatch, tmp_path):
    def path_in(named, filename, *, resource="mpas-source-rrfs", pin_set="rrfs"):
        return tmp_path / resource / pin_set / filename

    reads = []

    def words(cls, paths):
        reads.append(paths)
        return {}, []

    monkeypatch.setattr(fi, "tempo_lookup_path_in", path_in)
    monkeypatch.setattr(fi.FrameworkInitializationRRFS, "_tempo_lookup_words", classmethod(words))
    monkeypatch.setattr(fi, "_LOOKUP_PREFETCH", {})
    return reads


def _initialization(bound):
    init = object.__new__(fi.FrameworkInitializationRRFS)
    init.native = SimpleNamespace(tables=None, put_table=lambda *a, **k: None,
                                  **{name: bound[name] for name in ("resource", "pin_set") if name in bound})
    return init


def test_the_read_ahead_is_taken_on_the_default_rrfs_row(monkeypatch, tmp_path):
    # THE BREAKAGE (RRFS hrrrv5 on the 44c0849 default row, mv3km, V8X6000, 2026-10-29): the read-ahead resolved
    # the 1d6e370e fork's files, so it never matched; the 44c0849 files were read again on the main thread and the
    # unused read-ahead (~0.93 GB) stayed held for the whole run in every rank process.
    reads = _stub(monkeypatch, tmp_path)
    backend = framework_entry(SimpleNamespace(table_key="mpasphys_rrfs_44c0849"), "backend")
    bound = backend.keywords
    assert bound["pin_set"] == "rrfs-44c0849"
    configs = {"config_microp_scheme": "mp_tempo"}
    # The call _run_native_dynamics makes with the row's bound backend arguments.
    fi.FrameworkInitializationRRFS.prefetch_tables(
        configs, None, **{name: bound[name] for name in ("resource", "pin_set") if name in bound}).result(timeout=30)
    _initialization(bound).read_tempo_lookup_files()
    assert len(reads) == 1, "the read-ahead words were taken, nothing read twice"
    assert fi._LOOKUP_PREFETCH == {}


def test_a_read_ahead_for_other_files_is_dropped_not_held(monkeypatch, tmp_path):
    reads = _stub(monkeypatch, tmp_path)
    configs = {"config_microp_scheme": "mp_tempo"}
    fi.FrameworkInitializationRRFS.prefetch_tables(configs, None).result(timeout=30)
    _initialization({"resource": "mpas-source-rrfs-44c0849", "pin_set": "rrfs-44c0849"}).read_tempo_lookup_files()
    assert len(reads) == 2
    assert fi._LOOKUP_PREFETCH == {}
