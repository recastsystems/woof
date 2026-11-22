"""The WOOF-ArWen LES scalar must survive physics without loading moist air."""
from dataclasses import replace

from woof.hex.arwen_les import model_row, validate
from woof.hex.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig
from woof.hex.species_row import ENERGY


def test_les_adds_transported_energy_without_changing_moisture_loading():
    base = V841MpasColumnPhysicsSmagorinskyGwdoConfig()
    les = replace(base, config_les_model="prognostic_1.5_order",
                  config_les_surface="varying", config_mix_scalars=True,
                  config_pbl_scheme="off")
    validate(les)
    original, extended = model_row(base), model_row(les)
    assert extended.names() == original.names() + ("tke",)
    assert extended.n_mass_loading == original.n_mass_loading
    assert extended.species[-1].species_class == ENERGY
    assert extended.species[-1].transported
    assert extended.species[-1].units == "m^2 s^{-2}"


def test_default_keeps_its_original_scalar_row():
    base = V841MpasColumnPhysicsSmagorinskyGwdoConfig()
    validate(base)
    assert "tke" not in model_row(base).names()


def test_cadence_receipt_counts_surface_layer_when_les_replaces_ysu():
    from woof.hex import pbl_cadence
    from woof.hex.drivers.run_cuda_v841_forecast import _physics_cadence_proof

    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=1.0, requested="auto")
    receipts = [{"backend": {"cadence": {
        "surface_pbl_ran": True, "radiation_ran": False,
        "call_counts": {"sfclay": 1, "ysu": 0}}}}]
    proof = _physics_cadence_proof(decision, receipts, 1)
    assert proof["consistent"]
    assert proof["engine_call_counts_at_last_step"]["ysu"] == 0
