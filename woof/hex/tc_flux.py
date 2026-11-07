"""Shared TC flux diagnostics at an existing forecast completion boundary."""


def final_receipt(backend, *, steps):
    """Harvest the shared engine's event meter, never from the step callback.

    A moving run emits one receipt per integration segment. Its caller sums
    those segments; no spatial TC scratch is interpolated between meshes.
    The inactive path neither imports CuPy nor changes the ordinary receipt.
    """
    driver = getattr(getattr(backend, '_seam', None), '_driver', None)
    if getattr(driver, 'tc_surface', None) is None:
        return None
    from woof.tc.receipt import receipt
    return receipt(driver, steps=steps)


def class_receipts(stacks, steps, *, devices=None):
    """Harvest each active class/block on its own card at forecast completion.

    Steps are calls made by this process, excluding restored restart counters.
    GPU intervals from different cards are not added as a wall-time delta.
    """
    from contextlib import nullcontext
    result = []
    for index, stack in enumerate(stacks):
        backend = stack['backend']
        driver = getattr(getattr(backend, '_seam', None), '_driver', None)
        if getattr(driver, 'tc_surface', None) is None:
            continue
        context = nullcontext()
        if devices is not None:
            import cupy as cp
            context = cp.cuda.Device(devices[index])
        with context:
            row = final_receipt(backend, steps=steps[index])
        result.append(dict(index=index, steps=steps[index],
                           device=None if devices is None else devices[index], receipt=row))
    return result


def attach_lake_mask(backend, mesh, *, local_timestep=False):
    """TC-only geography for the WOOF column route, at construction time.

    Native/static land use may already fold lakes into water. Keep the
    original exclusion independent of the stock surface-layer lake input,
    so enabling TC exchange does not change the baseline flux on lakes.
    """
    driver = getattr(getattr(backend, '_seam', None), '_driver', None)
    if getattr(driver, 'tc_surface', None) is None:
        return None
    if local_timestep:
        raise ValueError('TC lake-mask binding is not qualified for WOOF-route local time stepping')
    import numpy as np
    import cupy as cp
    mask = mesh.arrays.get('tc_lakemask')
    if mask is None:
        raise ValueError('TC exchange needs tc_lakemask; prepare with --preserve-lake-mask')
    mask = np.asarray(mask)
    if mask.shape != (mesh.n_cells,) or not np.isin(mask, [0, 1]).all():
        raise ValueError('tc_lakemask must be a binary nCells vector in target mesh order')
    shape = driver.fields['lakemask'].shape
    capacity = driver.fields['lakemask'].size
    if capacity not in (mask.size, mask.size + 1):
        raise ValueError('TC lake mask does not match the column pack')
    # Native's regional garbage column is always excluded.
    target = np.ones(capacity, dtype=np.float32)
    target[:mask.size] = mask
    garbage = getattr(mesh, 'garbage_cell', None)
    if garbage is not None:
        if not 0 <= int(garbage) < capacity:
            raise ValueError('target mesh garbage column is outside the TC pack')
        target[int(garbage)] = 1
    driver.tc_surface.lake_mask_override = cp.asarray(target.reshape(shape))
    return dict(source='tc_lakemask static field', lake_cells=int(mask.sum()))
