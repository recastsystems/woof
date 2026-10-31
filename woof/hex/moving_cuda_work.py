"""Independent moving products must not drain one shared legacy CUDA stream."""
from functools import wraps
import os
import threading

_GEOMETRY = threading.RLock()


def independent_stream(function):
    @wraps(function)
    def call(*args, **kwargs):
        import cupy as cp
        if (os.environ.get('WOOF_HEX_MOVING_CUDA_STREAMS','1') == '0'
                or cp.cuda.get_current_stream().ptr != 0):
            return function(*args, **kwargs)
        # Nested moving operations inherit this stream, so profiles, integrals
        # and their finish kernel keep their original dependency order.
        with cp.cuda.Stream(non_blocking=True) as stream:
            try:
                return function(*args, **kwargs)
            finally:
                stream.synchronize()
    return call


def shared_geometry(attribute):
    """Publish immutable geometry only after its upload, across product streams."""
    def decorate(function):
        @wraps(function)
        def call(weights):
            import cupy as cp
            with _GEOMETRY:
                before=getattr(weights,attribute,None)
                result=function(weights)
                if result is not before:
                    cp.cuda.get_current_stream().synchronize()
                return result
        return call
    return decorate
