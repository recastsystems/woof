def test_morrison_aero_retains_the_snow_species_for_mynn():
    from woof.core.mynn_pbl_runtime import mynn_flag_qs
    assert mynn_flag_qs(40) is True
    assert mynn_flag_qs(10) is True
    assert mynn_flag_qs(28) is True
    assert mynn_flag_qs(1) is False
