"""Stock-WRF MP13 export retains the Registry's unadvected RIMI state."""

def test_rimi_is_exported_as_input_state_without_boundary_arrays():
    from woof.wrf_direct import _load_contract, _physics_contract_bundle
    frozen = _load_contract()
    result = _physics_contract_bundle(frozen, 13)
    variables = {v['name']: v for v in result['wrfinput']['variables']}
    assert variables['RIMI']['attributes']['description'] == 'riming intensity'
    assert variables['RIMI']['attributes']['units'] == 'fraction'
    assert variables['RIMI']['dimensions'] == variables['QCLOUD']['dimensions']
    assert 'QGRAUP' not in variables
    boundaries = {v['name'] for v in result['wrfbdy']['variables']}
    assert not any(name.startswith('RIMI_B') for name in boundaries)
    assert all('RIMI' != v['name'] for v in frozen['wrfinput']['variables'])
