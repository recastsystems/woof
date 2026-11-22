"""A door refuses a perturbation that is zero everywhere, by name, before any card work.

THE BREAKAGE: on box-al3 a plain 3 km tangent reported "valid True" on a bump of nothing (2026-11-11, doors6003
round 5, run r5-d3-b2): its direction.json held terms [] and words_per_field 0, every output field was 0, and the
whole-state dot of that run still passed on the runner own seeded direction. The check says the operator is right;
it says nothing about a direction with no words in it.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import cli

NLEV, NCOL = 6, 5
NAMES = ("theta_m", "qv", "rho_zz", "rho_p", "rtheta_p", "rho_theta", "exner", "pressure_p")


class _Map:
    """Each field in its own allocation, one copy, on the mesh columns."""

    def resolve(self, table, name, row=None, window_end=False):
        alloc = 100 + NAMES.index(name)
        return {"name": name, "alloc": alloc, "offset": 0, "shape": [NLEV, NCOL], "copies": [(alloc, 0)],
                "copy_shapes": [[NLEV, NCOL]]}


class _State:
    VALUES = {"theta_m": 300.0, "qv": 0.01, "rho_zz": 1.0, "rho_p": 0.0, "rtheta_p": 3.0, "rho_theta": 300.0,
              "exner": 0.9, "pressure_p": 10.0}

    def field(self, f):
        return np.full((NLEV, NCOL), self.VALUES[f["name"]])


def _box():
    """Five cells of a small box near 40.6N 91.3W (the 3 km storm box)."""
    return {"latCell": np.array([40.2, 40.4, 40.6, 40.8, 41.0]), "lonCell": np.array([-91.7, -91.5, -91.3, -91.1, -90.9])}


def _direction(at, amplitude=1.0):
    return d.theta_direction(d.load_table(), _Map(), _State(), _box(), at, amplitude, 1200.0, 3, 4.0)


def test_a_bump_on_the_mesh_is_a_direction_with_words():
    doc = _direction("40.6,-91.3")
    assert doc["words_per_field"] > 0 and any(t[2] != 0.0 for t in doc["terms"])


def test_a_bump_that_is_zero_on_every_cell_is_refused_by_name():
    # the box-al3 case: the globe default centre, thousands of km outside the box
    with pytest.raises(d.AdDoorError, match="the perturbation is zero on every cell of this mesh") as refusal:
        _direction(d.DEFAULT_AT)
    assert "20.4N -26.2E" in str(refusal.value) and "--at" in str(refusal.value)


def test_a_zero_amplitude_is_refused_by_name():
    with pytest.raises(d.AdDoorError, match="the perturbation is zero on every cell of this mesh"):
        _direction("40.6,-91.3", amplitude=0.0)


def test_the_recorded_empty_direction_file_is_refused():
    # direction.json of box-al3 r5-d3-b2/tangent, as written
    recorded = {"name": "theta bump: peak 1 K at 20.4N -26.2E, Gaussian R 1200 km, centred on model level 32 (width 4); "
                        "qv and density fixed; through the write-back", "peak_K": 1.0, "terms": [],
                "words_per_field": 0}
    with pytest.raises(d.AdDoorError, match="zero on every cell"):
        d.refuse_zero_direction(recorded)
    with pytest.raises(d.AdDoorError, match="zero on every cell"):
        d.refuse_zero_direction({**recorded, "terms": [[5, 0, 0.0], [5, 1, -0.0]], "words_per_field": 2})
    kept = {**recorded, "terms": [[5, 0, 0.0], [5, 1, 2.5e-3]], "words_per_field": 2}
    assert d.refuse_zero_direction(kept) is kept


def test_a_cooling_bump_keeps_the_cells_at_its_peak():
    """THE BREAKAGE: with a negative --amplitude the cells kept were those where the bump is BELOW 1e-4 of its
    peak (dth > 1e-4 * amplitude holds only there), so a cooling bump perturbed everything but the bump."""
    warm, cool = _direction("40.6,-91.3", 1.0), _direction("40.6,-91.3", -1.0)
    assert cool["words_per_field"] == warm["words_per_field"] > 0
    assert [t[:2] for t in cool["terms"]] == [t[:2] for t in warm["terms"]]
    assert all(c[2] == -w[2] for c, w in zip(cool["terms"], warm["terms"]))


def test_the_door_exits_2_on_the_refusal(monkeypatch, capsys, tmp_path):
    def refuse(arguments):
        return d.refuse_zero_direction({"name": "theta bump: peak 1 K at 20.4N -26.2E; x", "terms": []}) and 0

    monkeypatch.setattr(d, "run_tangent", refuse)
    monkeypatch.setattr(d, "run_adjoint_test", refuse)
    for door in ("tangent", "adjoint-test"):
        assert cli.main([door, "--row", "x1.40962", "--out", str(tmp_path / door)]) == 2
        assert "woof hex: the perturbation is zero on every cell of this mesh" in capsys.readouterr().err
