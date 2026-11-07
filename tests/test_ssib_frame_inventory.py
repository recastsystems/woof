"""Measured local-memory frames are covered by preflight's ceiling."""
import re
from pathlib import Path
import pytest

@pytest.mark.gpu
def test_column_and_sampling_frames_are_priced_from_their_readings():
    from woof.core.kernels import load_module
    from woof.core.preflight import KERNEL_MAX_LOCAL_SIZE_BYTES
    from woof.core.kernel_frame_recordings import KERNEL_LOCAL_FRAME_RECORDINGS
    from woof.certify.compile_platform import compile_platform_fingerprint
    fingerprint = compile_platform_fingerprint()
    key = (fingerprint["device_compute_capability"], fingerprint["nvrtc_build"])
    recorded = {name: value for row in KERNEL_LOCAL_FRAME_RECORDINGS
                if row.platform_key == key for name, value in row.frames.items()}
    root = Path(__file__).resolve().parents[1] / "woof/core/kernels"
    for name in ("ssib", "ssib_records", "soil_points", "record_select"):
        module = load_module(name)
        exports = re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)',
                             (root / (name + ".cu")).read_text())
        assert exports
        widest = max(module.get_function(n).attributes["local_size_bytes"] for n in exports)
        assert widest <= KERNEL_MAX_LOCAL_SIZE_BYTES[name]
        if name in recorded:
            assert widest == recorded[name]
    # Production binds state and forcing directly, so price that separately
    # from the packed-record replay entry point.
    from woof.core import ssib
    resident = ssib.kernel(resident=True, direct_forcing=True)
    assert resident.attributes["local_size_bytes"] <= KERNEL_MAX_LOCAL_SIZE_BYTES["ssib"]
