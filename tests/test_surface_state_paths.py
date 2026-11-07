"""Active surface state follows native Registry ownership and clocks."""
from types import SimpleNamespace

def test_feedback_excludes_domain_counters_and_restart_only_skin_state():
    from woof.core.surface_nests import feedback_fields
    cfg=SimpleNamespace(sst_update=1,sst_skin=1,tmn_update=1)
    names=feedback_fields(cfg)
    assert {'tlag','sstsk','ivgtyp','tmn'} <= set(names)
    assert not {'dtw','nday','nyear','sst','tsk_save'} & set(names)
    assert feedback_fields(SimpleNamespace()) == ()

def test_relocation_keeps_domain_history_but_does_not_interpolate_skin_memory():
    from woof.ingest.relocation_init import surface_continuation_fields, LAND_SURFACE_CONTINUATION_FIELDS
    cfg=SimpleNamespace(sst_update=1,sst_skin=1,tmn_update=1)
    assert {'dtw','tlag','tsk_save','sstsk'} <= set(surface_continuation_fields(cfg))
    assert not {'dtw','tsk_save'} & set(surface_continuation_fields(cfg,interpolate=True))
    assert surface_continuation_fields(SimpleNamespace()) == LAND_SURFACE_CONTINUATION_FIELDS

def test_ensemble_uses_private_surface_state():
    from woof.ensemble.suite_capabilities import plan_suite
    cfg=dict(mp_physics=8,ra_physics=0,sf_sfclay_physics=91,sf_surface_physics=2,
             bl_pbl_physics=1,cu_physics=0,sst_update=1,sst_skin=1,tmn_update=1)
    plan=plan_suite(cfg,members=2)
    assert not plan.uses_native_batch
    assert any('sst_update' in text for text in plan.native_fallback_reasons)
