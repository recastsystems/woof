"""Measure IAU additions, independently of the window's call counter.

The native increment holds dry rho, dry theta and edge u. The tendencies hold
rho_zz, rho_zz*theta_m and rho_edge*u. Compare their time integrals to the
analysis increment mapped through the instantaneous state by the chain rule.
This is a source-admission witness, not a claim about the evolved final state.
All field arithmetic and accumulation stay on the device; receipts reduce only
at output boundaries. No prognostic or source array is modified here.
"""
import numpy as np


class IauWitness:
    def __init__(self, cp):
        self.cp = cp
        self.actual = {}
        self.expected = {}
        self.calls = 0

    def before(self, sources, state, saved, diagnostics, increments, zz, cells_on_edge,
               moist, solve, weight, dt, correct):
        cp = self.cp
        nc, ne = solve
        cell = lambda a: a[..., :nc].astype(cp.float64)
        edge = lambda a: a[..., :ne].astype(cp.float64)
        old = {'rho': cell(sources.rho), 'theta_m': cell(sources.rho_theta), 'u': edge(sources.rho_u)}
        rho, thm = cell(state.rho), cell(saved.theta_m)
        q = cell(state.scalars[moist['index_qv']-1])
        dq = cell(increments['scalars'][moist['index_qv']-1])
        dr = cell(increments['rho']) / cell(zz)
        dth = cell(increments['theta'])
        rvord = float(np.float32(461.6) / np.float32(287.0))
        factor = 1 + rvord*q
        theta = thm/factor
        w = float(weight)
        rth = w*factor*(dth*rho + theta*dr)
        if correct:
            rth += rvord*theta*w*rho*dq
        else:
            # Native exact mode converts the total updated vapour tendency.
            qrate = cell(sources.scalars[moist['index_qv']-1]) + w*(dq*rho + q*dr)
            rth += rvord*theta*qrate
        ru = w*edge(diagnostics.h_edge)*edge(increments['u'])
        if correct:
            coe = cells_on_edge[:ne]
            a, b = coe[:,0], coe[:,1]
            # Exterior endpoints have no density increment on this route.
            width = increments['rho'].shape[-1]
            valid = (a>=0)&(a<width)&(b>=0)&(b<width)
            a = cp.clip(a, 0, width-1); b = cp.clip(b, 0, width-1)
            density = increments['rho'].astype(cp.float64) / cp.where(zz != 0, zz, 1).astype(cp.float64)
            edge_dr = cp.where(valid[None,:], .5*(density[:,a]+density[:,b]), 0)
            ru += w*edge_dr*edge(saved.normal_velocity)
        expected = {'rho': w*dr, 'theta_m': rth, 'u': ru}
        if not self.actual:
            self.actual = {k: cp.zeros_like(v) for k,v in old.items()}
            self.expected = {k: cp.zeros_like(v) for k,v in old.items()}
        return old, expected, float(dt), nc, ne

    def after(self, token, sources):
        old, expected, dt, nc, ne = token
        fields = {'rho': sources.rho[...,:nc], 'theta_m': sources.rho_theta[...,:nc], 'u': sources.rho_u[...,:ne]}
        for key, value in fields.items():
            self.actual[key] += (value.astype(self.cp.float64)-old[key])*dt
            self.expected[key] += expected[key]*dt
        self.calls += 1

    def receipt(self):
        cp = self.cp
        fields = {}
        for key, actual in self.actual.items():
            expected = self.expected[key]
            delta = actual-expected
            scale = float(cp.max(cp.abs(expected)))
            maximum = float(cp.max(cp.abs(delta)))
            fields[key] = dict(words=int(actual.size), actual_sum=float(cp.sum(actual)),
                expected_sum=float(cp.sum(expected)), actual_max_abs=float(cp.max(cp.abs(actual))),
                expected_max_abs=scale, max_abs_difference=maximum,
                relative_max_difference=maximum/scale if scale else maximum,
                expected_nonzero=int(cp.count_nonzero(expected)),
                finite=bool(cp.all(cp.isfinite(actual))) and bool(cp.all(cp.isfinite(expected))))
        passed = all(v['finite'] and v['relative_max_difference'] <= 2e-4 for v in fields.values())
        return dict(passed=passed, relative_tolerance=2e-4, calls=self.calls, quantity='sum of actual added coupled tendency times dt over real solve entities',
            reference='analysis increments mapped at each step through rho_zz, modified theta and edge-density chain rules',
            fields=fields)
