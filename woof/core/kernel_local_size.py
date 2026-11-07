"""Read per-thread local memory from either CUDA function API."""


def kernel_local_size_bytes(function) -> int:
    """Return the driver frame, including direct NVRTC function handles."""
    attributes = getattr(function, "attributes", None)
    if attributes is not None:
        return int(attributes["local_size_bytes"])
    from cupy.cuda import driver
    return int(driver.funcGetAttribute(
        driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function.ptr))
