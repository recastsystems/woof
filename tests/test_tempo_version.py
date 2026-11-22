"""The TEMPO generation selector (tempo_version, tempo_hailhyperaware).

Breakages this prevents:
* an unknown generation name silently running one nobody selected;
* the v3.2.5 init state drifting from TEMPO v3.2.5's own tempo_init, whose
  only parameter change from v3.0.3 is roverrv = rdry / rv (params.F90:451;
  the native v3.2.5 dump differs from the v3.0.3 dump in that one record);
* the two generations' blob layouts drifting apart (one Rust init feeds both);
* the pinned v3.0.3 generation compiling with any option or header it did not
  compile with before (its byte identity to the previous release).
"""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config

pytest.importorskip("cupy")
from woof.core import tempo_runtime as tr  # noqa: E402

_TINY = dict(nx=8, ny=8, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0, dt=18.0,
             run_seconds=0.0, moist=True)


def test_default_is_latest_and_pin_is_selectable():
    cfg = RunConfig(**_TINY, mp_physics=900)
    assert cfg.tempo_version == "v3.2.5" == tr.TEMPO_DEFAULT_VERSION
    assert cfg.tempo_hailhyperaware is True
    assert set(tr.TEMPO_VERSIONS) == {"v3.2.5", "v3.0.3"}
    validate_run_config(dataclasses.replace(cfg, tempo_version="v3.0.3"))
    validate_run_config(dataclasses.replace(cfg, tempo_hailhyperaware=False))


@pytest.mark.parametrize("bad", ["v3.2.4", "3.2.5", "latest", ""])
def test_unknown_generation_is_refused(bad):
    with pytest.raises(ValueError, match="tempo_version"):
        validate_run_config(RunConfig(**_TINY, mp_physics=900, tempo_version=bad))
    with pytest.raises(ValueError, match="tempo_version"):
        tr.kernel_dir(bad)


def test_hail_switch_off_on_the_pin_is_refused():
    with pytest.raises(ValueError, match="tempo_hailhyperaware"):
        validate_run_config(RunConfig(**_TINY, mp_physics=900, tempo_version="v3.0.3",
                                      tempo_hailhyperaware=False))


def test_layouts_agree_and_v325_init_differs_only_in_roverrv():
    l303 = tr.blob_layout((tr.kernel_dir("v3.0.3") / "tempo_common.cuh").read_text())
    l325 = tr.blob_layout((tr.kernel_dir("v3.2.5") / "tempo_common.cuh").read_text())
    assert l303 == l325
    rng = np.random.default_rng(0)
    blob = rng.random(sum(n for _, n in l303))
    out = tr.with_v325_parameters(blob, l303)
    offsets, o = {}, 0
    for name, n in l303:
        offsets[name] = o
        o += n
    changed = np.nonzero(out.view(np.int64) != blob.view(np.int64))[0]
    assert list(changed) == [offsets["roverrv"]]
    assert out[offsets["roverrv"]] == blob[offsets["rdry"]] / blob[offsets["rv"]]


def test_ctx_layouts():
    assert tr.ctx_dtype("v3.0.3") is tr.CTX_DTYPE
    assert tr.ctx_dtype("v3.2.5").names[-2:] == ("hailhyperaware_flag", "diagnostic_aerosols_flag")
    assert tr.ctx_dtype("v3.2.5").itemsize == tr.CTX_DTYPE.itemsize + 8


def test_pinned_generation_keeps_its_headers():
    """kernels/tempo is the v3.0.3 port and nothing else: its headers name
    9adf9ef and none carries a v3.2.x construct."""
    for path in sorted(tr.kernel_dir("v3.0.3").glob("*.cuh")):
        text = path.read_text()
        assert "TEMPO_GEN" not in text and "l_qh" not in text and "c62699f" not in text, path.name
    assert "9adf9ef" in (tr.kernel_dir("v3.0.3") / "NOTICE-TEMPO.txt").read_text()
    assert "c62699f" in (tr.kernel_dir("v3.2.5") / "NOTICE-TEMPO.txt").read_text()
