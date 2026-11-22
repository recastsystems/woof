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
    # Since fix/noahmp-small-defects every Noah-MP run steps the selected
    # (resident) images (noahmp_surface_device.device_route), so the default
    # options are priced at the selected glacier frame too.
    assert default['noahmp_glacier_composed'] == 792
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
    # 2.9.14: the default options step the resident (selected) images too.
    assert selected.selected_experiment(experiment())
    assert not selected.selected_experiment(experiment(lsm=2))
    for option in [dict(opt_alb=1), dict(opt_btr=3), dict(opt_rsf=4),
                   dict(opt_snf=5), dict(dveg=1), dict(opt_inf=2), dict(opt_run=4), dict(opt_run=6), dict(opt_run=7), dict(opt_rad=2), dict(opt_rad=1), dict(opt_frz=2), dict(opt_stc=2), dict(opt_tbot=1), dict(opt_stc=3)]:
        assert selected.selected_experiment(experiment(**option))


def test_gecros_recording_is_opt_in_and_source_bound(monkeypatch):
    import json
    from pathlib import Path
    receipt = json.loads((Path(__file__).parents[1] / "woof/data/gecros/frame-sm120-nvrtc13459.json").read_text())
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


def test_selected_readings_are_current():
    """Breakage prevented: a selected Noah-MP source edited without a re-read
    is priced at the assumed frame bound (65,136 B/thread). That priced a
    90x90 opt_alb=3 domain at 18.2 GiB instead of 4.2 and refused it on
    8-24 GB cards (2026-11-09). Fix: tools/measure_selected_noahmp_frames.py --write."""
    frames, basis = selected.selected_frames(snow_papers=True)
    assert 'changed' not in basis, basis
    assert assumed_frame_bound() not in frames.values(), frames


def test_snicar_admission_prices_like_class():
    """opt_alb=3 on a 90x90 domain: measured 2.049 GiB this-process peak against
    2.047 for the default (skeptic c1/c2, RTX 5090). The price may add the SNICAR
    optics and transients, never a withdrawn frame reading."""
    class_ = pf.estimate_experiment(experiment(nx=90, ny=90, opt_alb=2))
    snicar = pf.estimate_experiment(experiment(nx=90, ny=90, opt_alb=3))
    extra = snicar.peak_envelope_bytes - class_.peak_envelope_bytes
    assert 0 < extra < 64 * 2**20, extra
