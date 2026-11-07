"""A fine donor must not be reduced to the coarse background before insertion."""
import json
from types import SimpleNamespace as N
import pytest
from woof.storm_merge import initialization_lattice


def frame(root, spacing):
    root.mkdir()
    (root/'frames.json').write_text(json.dumps({'frames':[dict(valid_time='t',
        latitude={'values':[0.,spacing]},longitude={'values':[0.,spacing]})]}))
    return root


def test_fine_donor_and_receiving_footprint_are_preserved(tmp_path):
    background,donor = frame(tmp_path/'b',.25),frame(tmp_path/'d',.02)
    grid=N(dx=1000.,dy=1000.,e_we=101,e_sn=101,
           ij_to_latlon=lambda x,y:(20.+y*.01,179.+x*.01))
    chosen=initialization_lattice(background,donor,valid_time='t',native_grids=[grid],
                                  centre=(20.5,179.5),radius_km=315.)
    assert chosen['spacing_deg']==pytest.approx(.02)
    assert chosen['south']<18. and chosen['south']+(chosen['ny']-1)*.02>23.
    assert chosen['nx']*.02<10., 'a dateline crossing must not become a nearly global grid'
    grid.dx=grid.dy=3000.
    coarse=initialization_lattice(background,donor,valid_time='t',native_grids=[grid],centre=(20.5,179.5))
    assert coarse['spacing_deg']==pytest.approx(3000./111194.92664455874)


def test_already_fine_background_keeps_its_original_lattice(tmp_path):
    background,donor = frame(tmp_path/'b',.01),frame(tmp_path/'d',.02)
    grid=N(dx=1000.,dy=1000.)
    assert initialization_lattice(background,donor,valid_time='t',native_grids=[grid],centre=(0.,0.)) is None
