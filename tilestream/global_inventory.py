"""Closed per-domain arrays that one owner updates between tile sweeps."""
from __future__ import annotations


def global_keys(keys):
    from tilestream.sfire_spotting import spotting_global_keys
    from tilestream.sfire_smoke import GLOBAL_KEYS
    keys = tuple(keys)
    found = set(spotting_global_keys(keys))
    # Native single-entry history time is source-inert and belongs to the
    # domain. It must survive restart/history without becoming a tile axis.
    found.update(key for key in keys if key == 'fire/lfn_time')
    # The SCM surface forcing controls (HFX_FORCE and its siblings,
    # woof.core.physics_inventory.SCM_SURFACE_FORCING_FIELDS) are 0-d
    # domain scalars with no admitted producer, held at their initialized
    # or restored words: no horizontal window to sweep
    # (tilestream.gather.classify refuses shape ()), so the domain owns
    # them as it owns LFN_TIME.
    from woof.core.physics_inventory import SCM_SURFACE_FORCING_FIELDS
    scm = {'fields/' + name for name in SCM_SURFACE_FORCING_FIELDS}
    found.update(key for key in keys if key in scm)
    # The 3D PWP ocean's solver status (woof.ocean.mixed_layer, shape (1,)
    # int32) is one device word every tile's column kernel ORs its failure
    # bits into, read at history/checkpoint boundaries (mixed_layer.check):
    # it belongs to the domain, not to a tile window.  Named breakage: with
    # sf_ocean_physics = 2 every store-direct and streamed forecast (full
    # CONUS HRRR on [devices] or --tiles) refused its start with "carrier
    # 'fields/_pwp_status' has shape (1,): no horizontal extent, so it is not
    # a field this store can hold" (test host run 2a, test host r1b/r1d, 2026-10-21).
    found.update(key for key in keys if key == 'fields/_pwp_status')
    # Native precipitation reset origin is a two-word split real64 domain
    # clock. Every column reads the same immutable origin. Treating shape
    # (2,) as a spatial tile carrier refuses the streamed forecast before
    # its first step (it has no horizontal extent).
    found.update(key for key in keys if key in ('scratch/prec_acc_epoch','scratch/bucket_epoch','scratch/tendency_epoch','scratch/noahmp_output_epoch'))
    ledger = {key for key in keys if key.startswith('state/chemdiag_ledger_')}
    if ledger - set(GLOBAL_KEYS):
        raise ValueError('unknown chemistry ledger carrier would bypass tile transport')
    found.update(ledger)
    return tuple(sorted(found))
