"""A diagnostic setting declares the selector that can publish it."""
from woof.physics_registry import physics_registry, registry_knob_is_read

def test_diagnostics_require_their_registered_reader():
    r = physics_registry()
    spec = r["parameters"]["kf_edrates"]
    assert spec["requires_reader"] is True
    for selector in (11, 99):
        assert registry_knob_is_read("kf_edrates", {"cu_physics": selector}, r)
    assert not registry_knob_is_read("kf_edrates", {"cu_physics": 0}, r)


def test_alternative_readers_keep_mandatory_conditions_and_missing_defaults():
    r = {"parameters": {"rate": {
        "read_when": {"active": True},
        "read_when_any": [{"scheme": 11}, {"scheme": 99}]},
        "active": {"default": False}, "scheme": {"default": 0}}}
    assert registry_knob_is_read("rate", {"active": True, "scheme": 11}, r)
    assert registry_knob_is_read("rate", {"active": True, "scheme": 99}, r)
    assert not registry_knob_is_read("rate", {"active": False, "scheme": 99}, r)
    assert not registry_knob_is_read("rate", {"active": True, "scheme": 0}, r)
