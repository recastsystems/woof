# SPDX-License-Identifier: Apache-2.0
"""Pieces the three closures share: constants, the 16-class soil table, the clock, and the flux and skin shifts.

The arithmetic is the closures' research code's, operation for operation, so a run with an option on is the run
their scores were measured with.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

LV_J_PER_KG = 2.501e6
CP_J_PER_KG_K = 1004.5
EPS_Q = 0.622
G_ACCEL = 9.80665

# 16-class soil porosity, reference (field-capacity) and wilting-point volumetric water, index 0 unused, as the closures'
# training code carries them (STATSGO 16-class soil categories; a table of published soil properties, not code).
MAXSMC = np.array([
    0.451,
    0.395, 0.410, 0.435, 0.485, 0.485, 0.451, 0.420, 0.477,
    0.476, 0.426, 0.492, 0.482, 0.485, 0.451, 0.200, 0.421,
], dtype=np.float32)
REFSMC = np.array([
    0.329,
    0.236, 0.383, 0.383, 0.360, 0.360, 0.329, 0.314, 0.387,
    0.382, 0.338, 0.404, 0.412, 0.360, 0.329, 0.170, 0.283,
], dtype=np.float32)
WLTSMC = np.array([
    0.155,
    0.068, 0.075, 0.114, 0.179, 0.179, 0.155, 0.175, 0.218,
    0.250, 0.219, 0.283, 0.286, 0.179, 0.155, 0.006, 0.028,
], dtype=np.float32)


def start_utc(provider: Any) -> datetime:
    """The run's start (``config_start_time``) as the networks' diurnal clock reads it."""
    try:
        st = str(provider.session.layout.configs.get("config_start_time", "")).strip()
        if st:
            return datetime.strptime(st[:19], "%Y-%m-%d_%H:%M:%S")
    except Exception:
        pass
    raise RuntimeError("closure: the run has no config_start_time, so the networks' diurnal clock is unknown")


def flux_deltas(d_bowen, d_esh, d_elh):
    """Sensible, latent and moisture flux shifts from a Bowen shift and two net residuals.

    The Bowen shift moves energy between the two fluxes and conserves their sum (dH + dLH == 0 when both residuals
    are zero); the moisture flux follows the latent heat flux at L_v."""
    d_hfx = d_bowen + d_esh
    d_lh = -d_bowen + d_elh
    return d_hfx, d_lh


def _is_torch(x: Any) -> bool:
    return type(x).__module__.split(".", 1)[0] == "torch"


class SkinBackup:
    """The transient skin-temperature and surface-humidity shift ('diag' coupling): saved after the land step, put back
    before the next one, so the surface diagnostics and MYNN see it and RUC's soil heat balance never does.

    Only the columns the hook shifted (``treated``: interior unfrozen land at that step) are put back.  THE BREAKAGE
    THIS PREVENTS: putting back the whole array replaced what the surface layer, the sea-ice step and the boundary
    update wrote to the other columns between two land steps (over water, MYNN-SFC's new surface humidity was replaced
    by the previous step's)."""

    def __init__(self):
        self._saved: dict[int, tuple[Any, Any, Any]] = {}

    def save(self, provider: Any, tsk_arr: Any, qsfc_arr: Any, treated: Any) -> None:
        """``treated`` is a bool array on the hook's own library: a torch tensor for the nets, cupy for the closure."""
        self._saved[id(provider)] = (tsk_arr.copy(), qsfc_arr.copy(), treated)

    def restore(self, provider: Any) -> None:
        bkp = self._saved.pop(id(provider), None)
        if bkp is None:
            return
        tsk_orig, qsfc_orig, treated = bkp
        n = tsk_orig.shape[-1]
        tsk = provider.field("sfc_input", "skintemp")[:n]
        qsfc = provider.field("diag_physics", "qsfc")[:n]
        if _is_torch(treated):
            import torch
            for live, saved in ((tsk, tsk_orig), (qsfc, qsfc_orig)):
                t = torch.from_dlpack(live)
                t.copy_(torch.where(treated, torch.from_dlpack(saved), t))
        else:
            provider.cp.copyto(tsk, tsk_orig, where=treated)
            provider.cp.copyto(qsfc, qsfc_orig, where=treated)
