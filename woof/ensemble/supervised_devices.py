"""Physical card locks and stable logical ordinals for one ensemble worker."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
import os

from woof.ensemble.request import EnsembleRequest


@dataclass(frozen=True)
class EnsembleDeviceLease:
    visible: tuple
    locked: tuple

    @property
    def cuda_mask(self):
        return ",".join(gpu.uuid for gpu in self.visible)


def plan_ensemble_devices(request, identities, *, requested_uuid=None, visibility=None):
    """Resolve cards without CUDA or changing authored member ordinals.

    Explicit member IDs index the inherited visible order. Keeping that order
    in the worker's UUID mask avoids changing a spatial device graph or an
    explicitly selected member card while acquiring every used physical lock.
    """
    request = EnsembleRequest.from_mapping(request)
    identities = tuple(identities)
    if not identities or len({gpu.uuid for gpu in identities}) != len(identities):
        raise ValueError("ensemble supervision needs distinct physical GPU identities")
    if requested_uuid is not None:
        visible = tuple(gpu for gpu in identities if gpu.uuid == requested_uuid)
        if len(visible) != 1:
            raise ValueError("explicit ensemble GPU UUID is not present")
    elif visibility is None:
        visible = identities
    else:
        visible = []
        for value in visibility.split(","):
            value = value.strip()
            if value.isdigit():
                matches = tuple(gpu for gpu in identities if gpu.index == int(value))
            elif value.startswith("GPU-"):
                matches = tuple(gpu for gpu in identities if gpu.uuid.startswith(value))
            else:
                matches = ()
            if len(matches) != 1 or matches[0] in visible:
                raise ValueError("ensemble CUDA visibility must identify distinct available physical cards")
            visible.append(matches[0])
        visible = tuple(visible)
    ids = request.member_device_ids or tuple(range(len(visible)))
    if any(member_id >= len(visible) for member_id in ids):
        raise ValueError("ensemble member card is outside the inherited visible GPU order")
    return EnsembleDeviceLease(visible, tuple(visible[member_id] for member_id in ids))


@contextmanager
def input_device_lease(request, *, gpu_uuid, reservation_bytes, allow_shared_gpu, run_id):
    """The same physical leases at native input-directory forecast doors."""
    from woof import supervisor
    if request is None:
        gpu = supervisor.select_gpu(gpu_uuid)
        selected, mask = (gpu,), gpu.uuid
    else:
        lease = plan_ensemble_devices(request, supervisor.query_gpus(),
            requested_uuid=gpu_uuid, visibility=os.environ.get("CUDA_VISIBLE_DEVICES"))
        selected, mask = lease.locked, lease.cuda_mask
    with ExitStack() as locks:
        if not allow_shared_gpu:
            for gpu in sorted(selected, key=lambda item: item.uuid):
                locks.enter_context(supervisor.GPUFileLock(gpu.uuid, run_id=run_id))
        for gpu in selected:
            supervisor.preflight_exclusive_gpu(gpu.uuid, approved_pids={os.getpid()},
                allow_shared_gpu=allow_shared_gpu, reservation_bytes=reservation_bytes)
        yield mask


__all__ = ["EnsembleDeviceLease", "plan_ensemble_devices", "input_device_lease"]
