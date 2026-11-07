"""CU99 column-bank inventory; no member shares a trigger or held rate.

The column kernel has no horizontal exchanges. Joining its mass-grid
columns changes neither arithmetic nor the original per-member driver.
"""

from math import prod

KF_OLD_INPUTS = ("u", "v", "w", "temperature", "qv", "pressure", "rho",
                 "dz", "exner", "w0avg", "nca")
KF_OLD_CARRIERS = ("w0avg", "udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf")


def kf_old_driver_reservation(nz, ny, nx, *, members, edrates, multiprocessors=None):
    """Upper bound for a stock due call and its live hold/health temporaries.

    Persistent raw/coupled carriers have separate construction plans. The
    extra volumes bound the stock result validators, eligibility copies,
    history copies and clock intermediates, including allocation rounding.
    This does not replace the existing allocator margin.
    """
    amount = kf_old_call_reservation(nz, ny, nx, members=members,
        edrates=edrates, multiprocessors=multiprocessors)
    rounded = lambda size: ((int(size) + 511) // 512) * 512
    return amount + 12 * rounded(int(nz) * int(members) * int(ny) * int(nx) * 4) + 16 * rounded(int(members) * int(ny) * int(nx) * 4)


def kf_old_column_layout(nz, ny, nx, *, edrates):
    volume, surface = (int(nz), int(ny), int(nx)), (int(ny), int(nx))
    inputs = {name: (volume, "float32") for name in KF_OLD_INPUTS}
    inputs["w"] = ((int(nz) + 1, int(ny), int(nx)), "float32")
    inputs["nca"] = (surface, "float32")
    outputs = {name: (volume, "float32") for name in
        ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten")}
    outputs.update({name: (surface, "float32") for name in ("raincv", "pratec", "nca_out")})
    outputs.update({name: (surface, "int32") for name in ("cu_act_flag", "fatal")})
    if edrates:
        outputs.update({name: (volume, "float32") for name in
            ("udr_kf", "ddr_kf", "uer_kf", "der_kf")})
        outputs["timec_kf"] = (surface, "float32")
    return inputs, outputs


def kf_old_call_reservation(nz, ny, nx, *, members, edrates, multiprocessors=None):
    """Allocator-rounded joined outputs and the bounded live workspace.

    Gather/contiguous-return banks are priced separately. This includes the
    disabled diagnostic placeholder and leaves no unpriced native arrays.
    Without target SM metadata, reserve all columns as a conservative
    workspace ceiling. Configuration-only preflight never initializes CUDA.
    """
    from woof.core.kf import KF_TILE_BLOCKS_PER_SM, _TPB, kf_old_workspace_floats
    _, outputs = kf_old_column_layout(nz, ny, nx, edrates=edrates)
    rounded = lambda size: ((int(size) + 511) // 512) * 512
    amount = sum(rounded(prod(shape) * int(members) * 4) for shape, _ in outputs.values())
    columns = int(members) * int(ny) * int(nx)
    if multiprocessors is not None:
        columns = min(columns, int(multiprocessors) * KF_TILE_BLOCKS_PER_SM * _TPB)
    amount += rounded(kf_old_workspace_floats(int(nz), columns) * 4)
    if not edrates:
        amount += rounded(4)
    return amount
