"""Price the producer's real levels and reject an unposted outside cycle before mesh work."""
import json
from pathlib import Path

import pytest
from woof.hex import mesh_plan_door, card_fit
from woof.hex.cli import build_parser


def request(tmp_path, *extra):
    return build_parser().parse_args(['mesh-plan', '--global-mesh', '--point=37,-96.5',
        '--radius-km', '300', '--fine-dx-m', '3000', '--background-km', '96', '--card', '32gb',
        '--out', str(tmp_path / 'plan.json'), '--out-dir', str(tmp_path / 'run'),
        '--forecast-arg=--physics-suite=hrrrv5', *extra])


def test_producer_globe_prices_59_levels_before_build(tmp_path, monkeypatch):
    from woof.hex.swath import sizing
    monkeypatch.setattr(sizing, 'dry_run', lambda *a, **kw: {'predicted_cells': 119661})
    monkeypatch.setattr('woof.hex.engines.resolve', lambda *a: Path('/engine'))
    original = card_fit.predict
    seen = []
    def predict(*a, **kw):
        seen.append(kw['levels'])
        return original(*a, **kw)
    monkeypatch.setattr(card_fit, 'predict', predict)
    assert mesh_plan_door.run_global_mesh(request(tmp_path)) == 0
    assert seen and set(seen) == {59}
    assert json.loads((tmp_path / 'plan.json').read_text())['prediction_before_generation']['fits']


def test_unposted_outside_cycle_refused_before_mesh(tmp_path, monkeypatch):
    from woof.hex.swath import sizing
    monkeypatch.setattr(sizing, 'dry_run', lambda *a, **kw: {'predicted_cells': 119661})
    monkeypatch.setattr('woof.hex.engines.resolve', lambda *a: Path('/engine'))
    monkeypatch.setattr(card_fit, 'run_stage', lambda *a, **kw: pytest.fail('must not build'))
    with pytest.raises(Exception, match='gfs boundary cycle.*not published'):
        mesh_plan_door.run_global_mesh(request(tmp_path, '--run', '--generate', '--source', 'rrfs-gfs',
            '--cycle', '2026-10-14T03'))
    assert not (tmp_path / 'plan.json').exists()


def test_packaged_blend_checks_padded_decoder_box_not_only_cut_cells(monkeypatch):
    from woof.hex import blend_start as blend, hrrr_intermediate as intermediate
    # Measured 96 -> 3 km cut: its unpadded cells fit, but the decoder's box leaves the west edge.
    bad = intermediate.LatLonTarget(south=24.65, west=-123.991055, dlat=.025, dlon=.025, nx=2112, ny=1021)
    monkeypatch.setattr(intermediate, 'target_from_mesh', lambda *a: (bad, {}))
    assert not blend.decoder_box_fits(Path('cut.grid.nc'), 'rrfs')
    good = intermediate.LatLonTarget(south=36, west=-98, dlat=.025, dlon=.025, nx=81, ny=81)
    monkeypatch.setattr(intermediate, 'target_from_mesh', lambda *a: (good, {}))
    assert blend.decoder_box_fits(Path('cut.grid.nc'), 'rrfs')
