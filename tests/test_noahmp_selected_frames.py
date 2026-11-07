"""Selected fused exports must be priced, and source drift withdraws a reading."""
from dataclasses import replace
from types import SimpleNamespace

from woof.core import noahmp_selected_frames as selected
from woof.core.kernel_frame_recordings import assumed_frame_bound
from woof.core import preflight as pf
from test_noahmp_frame_provenance import experiment


def test_selected_glacier_prices_its_fused_export_separately_from_default():
    frames, basis = selected.selected_frames(
        SimpleNamespace(compile_platform=selected.MEASURED_PLATFORM))
    assert frames['noahmp_glacier_composed'] == 792
    assert 'measured on' in basis
    profile = pf.DeviceLocalMemoryProfile(
        name='measured platform', multiprocessor_count=170,
        max_threads_per_multiprocessor=1536, default_stack_limit_bytes=1024,
        bare_context_bytes=0, compile_platform=selected.MEASURED_PLATFORM)
    default = pf.kernel_local_frame_bytes(experiment(), profile=profile)
    actual = pf.kernel_local_frame_bytes(experiment(opt_snf=3), profile=profile)
    assert default['noahmp_glacier_composed'] == 456
    assert actual['noahmp_glacier_composed'] == 792


def test_other_platform_is_labelled_as_unread():
    frames, basis = selected.selected_frames(
        SimpleNamespace(compile_platform=('89', 'unread')))
    assert frames['noahmp_glacier_composed'] == 792
    assert 'this compile platform was not read' in basis


def test_changed_selected_source_uses_assumed_bound(monkeypatch):
    original = selected.runtime_unit

    def changed(name, **kwargs):
        unit = original(name, **kwargs)
        return replace(unit, source=unit.source + '\n// changed selected source\n') if name == 'noahmp_glacier' else unit

    monkeypatch.setattr(selected, 'runtime_unit', changed)
    frames, basis = selected.selected_frames()
    assert frames['noahmp_glacier_composed'] == assumed_frame_bound()
    assert frames['noahmp_leaves'] == 496
    assert 'noahmp_glacier' in basis and 'changed' in basis


def test_only_selected_noahmp_reaches_extension_prices():
    assert not selected.selected_experiment(experiment())
    assert not selected.selected_experiment(experiment(lsm=2))
    for option in [dict(opt_alb=1), dict(opt_btr=3), dict(opt_rsf=4),
                   dict(opt_snf=5), dict(dveg=1), dict(opt_inf=2), dict(opt_run=4), dict(opt_run=6), dict(opt_run=7), dict(opt_rad=2), dict(opt_rad=1), dict(opt_frz=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3)]:
        assert selected.selected_experiment(experiment(**option))


def test_gecros_recording_is_opt_in_and_source_bound(monkeypatch):
    import json
    from pathlib import Path
    receipt = json.loads((Path(__file__).parents[1] / "woof/data/gecros/frame-sm120-nvrtc13088.json").read_text())
    assert receipt["identity"]["identity_sha256"] == selected.GECROS_IDENTITY
    assert max(v["local_size_bytes"] for v in receipt["attributes"].values()) == 0
    before = pf.kernel_local_frame_bytes(experiment())
    after = pf.kernel_local_frame_bytes(experiment(opt_crop=2))
    assert "noahmp_gecros" not in before
    assert after["noahmp_gecros"] == 0
    original = selected.runtime_unit
    monkeypatch.setattr(selected, "runtime_unit", lambda name: replace(original(name), source=original(name).source+"\n// drift"))
    assert selected.gecros_frame()[0] == assumed_frame_bound()
    assert not selected.gecros_frame()[2]
