"""Native MPAS pool adapter to the shared, opt-in CUDA TC surface kernels.

No flux equations live here. Coefficients, citations and spray budgets are
owned by woof.tc. Native MYNN keeps its specific-humidity QSFC carrier.
"""
from dataclasses import replace
from types import SimpleNamespace


def configure(config, options):
    from woof.tc.config import DEFAULTS
    from woof.tc.curves import CURVES
    selected = {k: options.get(k, DEFAULTS[k]) for k in ('tc_flux_curve', 'tc_spray')}
    if selected['tc_flux_curve'] not in ('off', *CURVES):
        raise ValueError('unknown tc_flux_curve')
    if selected['tc_spray'] not in ('off', 'bao2000', 'barr2023'):
        raise ValueError('unknown tc_spray')
    if all(value == 'off' for value in selected.values()):
        return config
    if not hasattr(config, 'tc_surface'):
        raise ValueError('this native suite has no shared TC surface adapter')
    if (config.selectors['surface_layer'] not in (
            'sf_mynn', 'sf_monin_obukhov', 'sf_monin_obukhov_rev')
            or str(config.backends['surface_layer']) != 'hex-cuda'):
        raise ValueError('TC exchange requires a CUDA MYNN or MM5 surface layer')
    return replace(config, tc_surface=selected)


class NativeSurface:
    def __init__(self, config, provider):
        from woof.tc.config import DEFAULTS
        self.provider = provider
        self.cfg = SimpleNamespace(**(DEFAULTS | dict(config.tc_surface)),
            sf_sfclay_physics=5 if config.selectors['surface_layer'] == 'sf_mynn' else 91)
        self.tc_surface = None
        self.scratch = {}

    def prepare(self):
        # Hold first-layer device views before the stock callback retires its
        # input-only transaction entries. No atmospheric arithmetic or copy.
        self.atmosphere = {key: self.provider.prepared(native)[:1] for key, native in
            dict(u='u_p', v='v_p', temperature='t_p', qv='qv_p',
                 pressure='pres_hyd_p', dz='dz_p').items()}

    def fields(self):
        provider = self.provider
        d, s = provider['diag_physics'], provider['sfc_input']
        names = ('znt ust ustm mol hfx qfx qsfc zol psim psih lh u10 v10 '
                 'q2 chs chs2 cqs2 flhc flqc rmol cd cda ck cka cpm gz1oz0').split()
        f = {k: d[k] for k in names}
        f.update(t2=d['t2m'], th2=d['th2m'], psfc=provider.prepared('psfc_p'),
                 tsk=s['skintemp'], xland=s['xland'], xice=s['xice'])
        if not self.scratch:
            cp = provider.cp
            # Native MPAS folds MODIS lake class into water during init. Use
            # the preserved mask so hurricane fluxes do not reach lakes.
            mask = getattr(provider.native, 'tc_lakemask', None)
            if mask is None:
                raise ValueError('native TC exchange needs the original MODIS lake mask')
            self.scratch = {k: cp.zeros_like(f['ust']) for k in ('fm', 'fh', 'qstar')}
            self.scratch['lakemask'] = cp.asarray(mask, dtype=cp.float32)
        f.update(self.scratch)
        if self.cfg.sf_sfclay_physics == 5:
            f['ch'] = d['ch']
        else:
            f.update(fm=d['fm'], fh=d['fh'])
        return f

    def exchange(self):
        from woof.tc.surface import SurfaceBundle
        f = self.fields()
        if self.tc_surface is None:
            self.tc_surface = SurfaceBundle(self.cfg, f)
        self.tc_surface.exchange(self.atmosphere, f, SimpleNamespace(**f))

    def spray(self, call):
        if self.cfg.tc_spray == 'off':
            # Exchange opens the event-ring sample even with spray disabled.
            # Close it at the same post-LSM boundary without binding fields or
            # launching a spray kernel, so curve-only cost is measurable.
            self.tc_surface.spray(None, None)
            return
        # After land/ice and before PBL. Bind current resident pool words;
        # surface transaction scratch is not the authoritative flux storage.
        self.tc_surface.spray(self.atmosphere, self.fields())
        self.provider.commit()
