"""Resident native global coupling, geopotential and acoustic transitions."""
import numpy as np


class GlobalGeometry:
    def __init__(self, nx, ny, nz, *, profiles, metrics, rdx, rdy,
                 cfn, cfn1, cpovcv=np.float32(1004.5)/np.float32(717.5)):
        from woof.core.global_acoustic import GlobalAcousticTransport
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)):
            raise ValueError('global dimensions must be positive integers')
        shapes = ((ny, nx), (ny, nx), (ny, nx+1), (ny, nx+1),
                  (ny+1, nx), (ny+1, nx), (ny+1, nx))
        if len(metrics) != 7:
            raise ValueError('tx/ty/ux/uy/vx/vy/vxinv metrics required')
        GlobalAcousticTransport._validate((profiles, *metrics), ((8, nz+1), *shapes))
        if (not np.isfinite((rdx, rdy, cfn, cfn1, cpovcv)).all()
                or min(rdx, rdy, cpovcv) <= 0):
            raise ValueError('finite spacing, extrapolation and heat constants required')
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz))
        self.profiles, self.metrics = profiles, tuple(metrics)
        self.spacing = np.float32(rdx), np.float32(rdy)
        self.top = np.float32(cfn), np.float32(cfn1)
        self.cpovcv = np.float32(cpovcv)
        self.module = load_module_no_ftz('global_geometry', no_fmad=True)
        self.kernels = {name: self.module.get_function('global_'+name) for name in
                        ('mass_faces', 'couple_momentum', 'omega', 'small_prep_mass',
                         'small_prep', 'small_finish', 'slow_pressure', 'rhs_phi')}
        self.face_count = max((nx+1)*ny, nx*(ny+1))
        self.count = max(nz*self.face_count, (nz+1)*nx*ny)

    def _call(self, name, count, args):
        self.kernels[name](((count+127)//128,), (128,), args)

    def mass_faces(self, mu, mub, mut, muu, muv, *, mode=0):
        if mode not in (0, 1, 2):
            raise ValueError('native mass-face mode is 0, 1 or 2')
        from woof.core.global_acoustic import GlobalAcousticTransport
        nx, ny = self.nx, self.ny
        GlobalAcousticTransport._validate((mu, mub, mut, muu, muv),
            ((ny, nx), (ny, nx), (ny, nx), (ny, nx+1), (ny+1, nx)))
        self._call('mass_faces', self.face_count,
                   (*self.dims[:2], np.int32(mode), mu, mub, mut, muu, muv))

    def couple(self, *, muu, muv, mut, u, v, w, ru, rv, rw):
        self._call('couple_momentum', self.count,
                   (*self.dims, *(self.profiles[i] for i in range(4)),
                    self.metrics[3], self.metrics[6], self.metrics[1],
                    muu, muv, mut, u, v, w, ru, rv, rw))

    def omega(self, *, mu, mub, u, v, divv, ww):
        self._call('omega', self.nx*self.ny,
                   (*self.dims, *self.spacing, self.profiles[0], self.profiles[1],
                    self.profiles[4], self.metrics[0], self.metrics[3],
                    self.metrics[6], mu, mub, u, v, divv, ww))

    def prepare_mass(self, *, mub, mu0, mu, mu_save, muts):
        self._call('small_prep_mass', self.nx*self.ny,
                   (np.int32(self.nx*self.ny), mub, mu0, mu, mu_save, muts))

    def prepare(self, *, muu, muv, mut, muus, muvs, muts,
                u0, v0, w0, t0, ph0, pb, p, alt,
                u, v, w, t, ph, us, vs, ws, ts, phs, c2a):
        self._call('small_prep', self.count,
                   (*self.dims, self.cpovcv, *(self.profiles[i] for i in range(4)),
                    self.metrics[3], self.metrics[6], self.metrics[1],
                    muu, muv, mut, muus, muvs, muts, u0, v0, w0, t0, ph0,
                    pb, p, alt, u, v, w, t, ph, us, vs, ws, ts, phs, c2a))

    def finish(self, *, muu, muv, mut, muus, muvs, muts,
               us, vs, ws, ts, phs, mus, ww1, u, v, w, t, ph, mu, ww):
        self._call('small_finish', self.count,
                   (*self.dims, *(self.profiles[i] for i in range(4)),
                    self.metrics[3], self.metrics[4], self.metrics[1],
                    muu, muv, mut, muus, muvs, muts, us, vs, ws, ts, phs, mus, ww1,
                    u, v, w, t, ph, mu, ww))

    def pressure(self, *, muu, muv, ph, alt, p, pb, al, ru_tend, rv_tend):
        self._call('slow_pressure', self.nz*self.face_count,
                   (*self.dims, *self.spacing, self.profiles[0], self.profiles[1],
                    *self.metrics[2:6], muu, muv, ph, alt, p, pb, al, ru_tend, rv_tend))

    def rhs_phi(self, *, order, phi_adv_z, muu, muv, u, v, ww, ph, phb, tendency):
        if order not in (2, 3, 4, 5, 6) or phi_adv_z not in (1, 2):
            raise ValueError('native geopotential order 2..6 and vertical form 1/2 required')
        self._call('rhs_phi', (self.nz+1)*self.nx*self.ny,
                   (*self.dims, np.int32(order), np.int32(phi_adv_z),
                    *self.spacing, *self.top, self.profiles[2], self.profiles[3],
                    self.profiles[5], self.profiles[6], self.profiles[7],
                    self.metrics[2], self.metrics[5], self.metrics[1],
                    muu, muv, u, v, ww, ph, phb, tendency))
