"""Native HIP properties must size all existing column launchers correctly."""
from types import SimpleNamespace

from woof.hip_runtime import device_attribute, initialize_device_attributes


def test_native_properties_repair_shared_attributes_for_each_device():
    caches = [{"MultiProcessorCount": 1, "Unrelated": 17},
              {"MultiProcessorCount": 1, "Unrelated": 19}]
    properties = [{"multiProcessorCount": 304, "warpSize": 64,
                   "maxThreadsPerMultiProcessor": 2048},
                  {"multiProcessorCount": 80, "warpSize": 32,
                   "maxThreadsPerMultiProcessor": 1024}]
    runtime = SimpleNamespace(is_hip=True, getDeviceCount=lambda: 2,
                              getDeviceProperties=lambda i: properties[i])
    cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=runtime, Device=lambda i: SimpleNamespace(attributes=caches[i])))
    changes = initialize_device_attributes(cp)
    assert changes[0]["MultiProcessorCount"] == (1, 304)
    assert changes[1]["MultiProcessorCount"] == (1, 80)
    assert caches[0] == {"MultiProcessorCount": 304, "Unrelated": 17,
                         "WarpSize": 64, "MaxThreadsPerMultiProcessor": 2048}
    assert caches[1]["MaxThreadsPerMultiProcessor"] == 1024
    assert initialize_device_attributes(cp) == {0: {}, 1: {}}


def test_cuda_attributes_are_untouched():
    # No device lookup, enumeration or property query is needed on CUDA.
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=False)))
    assert initialize_device_attributes(cp) == {}


def test_direct_hip_query_uses_properties_instead_of_cuda_enums():
    runtime = SimpleNamespace(is_hip=True, getDevice=lambda: 3,
        getDeviceProperties=lambda i: {"multiProcessorCount": 304 + i,
                                      "maxThreadsPerMultiProcessor": 2048})
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    assert device_attribute(cp, "MultiProcessorCount") == 307
    assert device_attribute(cp, "MaxThreadsPerMultiProcessor", 0) == 2048


def test_direct_cuda_query_preserves_ordinal_and_enum():
    calls = []
    runtime = SimpleNamespace(is_hip=False, getDevice=lambda: 2,
        cudaDevAttrMultiProcessorCount=16,
        deviceGetAttribute=lambda enum, ordinal: calls.append((enum, ordinal)) or 170)
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    assert device_attribute(cp, "MultiProcessorCount") == 170
    assert calls == [(16, 2)]
