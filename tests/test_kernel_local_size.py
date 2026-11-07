from types import SimpleNamespace
from woof.core.kernel_local_size import kernel_local_size_bytes


def test_raw_kernel_local_frame_uses_its_driver_attributes():
    assert kernel_local_size_bytes(SimpleNamespace(
        attributes={"local_size_bytes": 1536})) == 1536
