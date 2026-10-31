"""Device words from compiled WRF and unchanged diff_opt=2 baseline."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
DATA=Path(__file__).parent/"data/wrf471_diff_opt1"
SOURCE_TRANSITION_SHA256="e91599cc84e37891666dd760c77c3ef2f4efe67e260c534c51ac136f2a710e58"


def _measured_current_sources(receipt):
    """Retain the historical GPU archive and gate the corrected native proof."""
    import hashlib
    from assembled_legacy_proofs import validate_coordinate, validate_coordinate_negative
    # The prior witness is retained byte for byte; it is not relabeled as a
    # zero-change transition after the 25435 incorrect old words were fixed.
    historical=(DATA/"measured-source-transition.json").read_bytes()
    assert hashlib.sha256(historical).hexdigest()==SOURCE_TRANSITION_SHA256
    proof=validate_coordinate(DATA.parents[2])
    validate_coordinate_negative(DATA.parents[2])
    assert proof["prior_gpu_archive_sha256"]==receipt["archive_sha256"]
    assert proof["native_archive_sha256"]==receipt["native_archive_sha256"]
    return {name: proof["capture_identity"]["module_source_sha256"][name]
            for name in ("smag2d","diff_opt1")}


@pytest.mark.skip(reason="pins the engine's historical bytes, which the WOOF rename moves (gpuwm -> woof in sources this record hashes); the same check runs on the engine's own tree in its battery")
def test_merged_coordinate_receipt_is_sealed_to_native_and_production_source():
    import hashlib
    from woof.core.kernels import module_source
    receipt=json.loads((DATA/"merged-gpu.json").read_text())
    proof=json.loads((DATA/"merged-attribution.json").read_text())
    assert receipt["archive_sha256"]==hashlib.sha256((DATA/"merged-gpu.npz").read_bytes()).hexdigest()
    assert receipt["native_archive_sha256"]==hashlib.sha256((DATA/"wrf471.npz").read_bytes()).hexdigest()
    assert proof["production_archive_sha256"]==receipt["archive_sha256"]
    assert proof["prior_archive_sha256"]==hashlib.sha256((DATA/"km2-gpu.npz").read_bytes()).hexdigest()
    assert len(receipt["cases"])==88
    assert sum(len(row["fields"]) for row in receipt["cases"])==272
    for name,digest in _measured_current_sources(receipt).items():
        assert hashlib.sha256(module_source(name).encode()).hexdigest()==digest

@pytest.mark.gpu
@requires_gpu
def test_coordinate_flux_every_compiled_wrf_output_word():
    import cupy as cp
    from woof.core.dycore import launch_coordinate_horizontal
    cases=json.loads((DATA/"wrf471.json").read_text())["cases"]
    with np.load(DATA/"wrf471.npz") as values:
        for case in cases:
            if case["family"]!="horizontal":continue
            get=lambda key: values[case["name"]+"_"+key]
            dev=lambda key: cp.asarray(get(key))
            result=dev("seed")
            launch_coordinate_horizontal(dev("field"),dev("km"),dev("mu"),dev("c1"),dev("c2"),
                dev("mt"),dev("mfu"),dev("mfv"),case["dx"],case["dy"],result,
                stagger=("","x","y","z")[case["stag"]],
                boundary_x=case["bx"],boundary_y=case["by"],
                theta_initial=dev("base") if case["perturb"] else None)
            np.testing.assert_array_equal(cp.asnumpy(result).view("u4"),get("expected").view("u4"),err_msg=case["name"])


@pytest.mark.gpu
@requires_gpu
def test_coordinate_km4_every_compiled_wrf_output_word():
    import cupy as cp
    from woof.core.kernels import get_kernel
    cases=json.loads((DATA/"wrf471.json").read_text())["cases"]
    with np.load(DATA/"wrf471.npz") as values:
        for case in cases:
            if case["family"]!="km4":continue
            get=lambda key: values[case["name"]+"_"+key]
            dev=lambda key: cp.asarray(get(key))
            nz,ny,nx=get("d11").shape
            km=cp.zeros((nz,ny,nx),"f4");kh=cp.zeros_like(km)
            get_kernel("diff_opt1","wrf_diff_opt1_km4")(
                ((nx+127)//128,ny,nz),(128,1,1),
                (dev("d11"),dev("d22"),dev("d12"),dev("mt"),np.float32(case["dx"]),
                 np.float32(case["dy"]),np.float32(.25),np.float32(1./3.),km,kh,
                 np.int32(nz),np.int32(ny),np.int32(nx),np.int32(case["bx"]),np.int32(case["by"])))
            for field,value in (("km",km),("kh",kh)):
                np.testing.assert_array_equal(cp.asnumpy(value).view("u4"),get("expected_"+field).view("u4"),err_msg=case["name"]+field)


@pytest.mark.gpu
@requires_gpu
def test_coordinate_km2_compiled_wrf_horizontal_coefficients_and_tke_words():
    import cupy as cp
    from tools.wrf_diffopt1_oracle.model_case import km2_case
    from tools.wrf_diffopt1_oracle.capture import stats
    from woof.core.dycore import launch_wrf_tke_km
    receipt=json.loads((DATA/"merged-gpu.json").read_text())
    measured={row["name"]:row["fields"] for row in receipt["cases"]}
    with np.load(DATA/"wrf471.npz") as values,np.load(DATA/"merged-gpu.npz") as pinned:
        for case in json.loads((DATA/"wrf471.json").read_text())["cases"]:
            if case["family"]!="km2":continue
            cfg,state=km2_case(case,values)
            km=state.scratch(state.p.shape,"smag_km");kh=state.scratch(state.p.shape,"smag_kh")
            rtke=state.scratch(state.p.shape,"smag_rtke");rtke[:]=np.float32(.125)
            launch_wrf_tke_km(state,cfg,km,kh,time_t=False)
            bn2=state._scratch["diff6_x"].reshape(-1)[:state.p.size].reshape(state.p.shape)
            for field,value in (("km",km),("kh",kh),("tke",state.tke),("bn2",bn2),
                                ("kmv",state._scratch["smag_kmv"]),
                                ("khv",state._scratch["smag_khv"])):
                actual=cp.asnumpy(value);key=case["name"]+"_"+field
                np.testing.assert_array_equal(actual.view("u4"),values[case["name"]+"_expected_"+field].view("u4"),err_msg=key)
            assert bool(cp.all(rtke==np.float32(.125)))


@pytest.mark.gpu
@requires_gpu
def test_coordinate_deformation_matches_compiled_word_receipt():
    import cupy as cp
    from tools.wrf_diffopt1_oracle.model_case import km2_case
    from tools.wrf_diffopt1_oracle.capture import stats
    from woof.core.dycore import launch_wrf_smag2d_km
    measured={row["name"]:row["fields"] for row in json.loads((DATA/"merged-gpu.json").read_text())["cases"]}
    with np.load(DATA/"wrf471.npz") as values,np.load(DATA/"merged-gpu.npz") as pinned:
        for case in json.loads((DATA/"wrf471.json").read_text())["cases"]:
            if case["family"]!="deform":continue
            cfg,state=km2_case(case,values)
            km=state.scratch(state.p.shape,"smag_km");kh=state.scratch(state.p.shape,"smag_kh")
            tensors=launch_wrf_smag2d_km(state,cfg,km,kh,time_t=False)
            for field,value in zip(("d11","d22","d12"),tensors):
                actual=cp.asnumpy(value)
                expected=values[case["name"]+"_expected_"+field]
                np.testing.assert_array_equal(actual.view("u4"),expected.view("u4"))
                if field == "d12":
                    # cal_deform_and_div copies the evaluated tensor at physical
                    # faces. Preserve the donor rule independently of the pin.
                    if case["bx"]:
                        np.testing.assert_array_equal(actual[:,:,0].view("u4"),actual[:,:,1].view("u4"))
                    if case["by"]:
                        np.testing.assert_array_equal(actual[:,0,:].view("u4"),actual[:,1,:].view("u4"))


@pytest.mark.gpu
@requires_gpu
def test_merged_coordinate_moved_words_are_only_the_tensor_donor_fix(tmp_path):
    import subprocess
    import sys
    capture=Path(__file__).parents[1]/"tools/wrf_diffopt1_oracle/capture.py"
    output=tmp_path/"donor-reverted"
    # Separate process keeps this arithmetic diagnostic out of the production
    # kernel loader and its cached function wrappers for the other tests.
    subprocess.run([sys.executable,str(capture),str(DATA),str(output),"--mode","km2",
                    "--revert-deformation-donor"],check=True)
    import hashlib
    from assembled_legacy_proofs import validate_coordinate_negative
    current = validate_coordinate_negative(DATA.parents[2])
    with np.load(output.with_suffix(".npz")) as control:
        assert set(control.files) == set(current["fields"])
        for key in control.files:
            assert hashlib.sha256(control[key].tobytes()).hexdigest() == current["fields"][key]["control_sha256"], key
    # The original archive attribution remains independently checked below.
    proof=json.loads((DATA/"merged-attribution.json").read_text())
    with np.load(DATA/"km2-gpu.npz") as prior,np.load(DATA/"merged-gpu.npz") as merged:
        observed={}
        for key in prior.files:
            mask=prior[key].view("u4")!=merged[key].view("u4")
            for index in np.argwhere(mask):
                location=tuple(int(value) for value in index)
                observed[(key,location)]=(int(prior[key].view("u4")[location]),int(merged[key].view("u4")[location]))
        recorded={(row["receipt"]+"_"+row["field"],tuple(row["index"])):
                  (int(row["prior_word"],16),int(row["merged_word"],16))
                  for row in proof["moved_word_attribution"]}
        assert observed==recorded
        assert len(observed)==proof["moved_words"]==492


@pytest.mark.gpu
@requires_gpu
def test_metric_diffusion_matches_merged_oracle_fix_baseline():
    """The 2.8.8 trajectory has an exact, separately named pin.

    The original a417 words remain intact in diff2-baseline.npz and the 2.8.7
    words in diff2-merged-baseline.npz, every 2.8.7 move attributed by the
    merge receipt.  diff2-288-baseline.npz is the 2.8.8 release line's
    measurement; diff2-288-repin.json records what moved and why
    (test_288_repin_receipt_seals_its_pins).  No tolerance: every word.
    """
    from tools.wrf_diffopt1_oracle.model_case import configuration,model_state,word_arrays
    from woof.core.dycore import step
    with np.load(DATA/"diff2-288-baseline.npz") as expected:
        for km in (2,4):
            for boundary in (False,True):
                for moist in (False,True):
                    # Preserve the sealed pre-CQ trajectory and legacy checkpoint physics.
                    cfg=replace(configuration(km=km,diff=2,mix=True,boundary=boundary,moist=moist),moist_cq=False)
                    state=model_state(cfg)
                    for _ in range(3):step(state,cfg)
                    name=f"diff2_k{km}_b{int(boundary)}_m{int(moist)}"
                    for field,actual in word_arrays(state).items():
                        np.testing.assert_array_equal(actual.view("u4"),expected[name+"_"+field].view("u4"),err_msg=name+field)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km",[2,4])
@pytest.mark.parametrize("mix",[False,True])
def test_coordinate_restart_retains_original_theta_and_exact_trajectory(tmp_path,km,mix):
    import cupy as cp
    from tools.wrf_diffopt1_oracle.model_case import configuration,model_state,word_arrays
    from woof.core.dycore import step,initialize_coordinate_reference
    from woof.io.restart import write_restart,restore_restart
    cfg=configuration(km=km,mix=mix)
    straight=model_state(cfg);split=model_state(cfg)
    initialize_coordinate_reference(straight,cfg)
    original=cp.asnumpy(straight._scratch["diff1_theta_initial"]).copy()
    for _ in range(4):step(straight,cfg)
    for _ in range(2):step(split,cfg)
    split.elapsed_seconds=2*cfg.dt
    path=write_restart(tmp_path/"split.npz",split,cfg)
    with np.load(path) as stored:
        np.testing.assert_array_equal(stored["scratch/diff1_theta_initial"].view("u4"),original.view("u4"))
    resumed=model_state(cfg)
    restore_restart(path,resumed,cfg)
    for _ in range(2):step(resumed,cfg)
    np.testing.assert_array_equal(cp.asnumpy(resumed._scratch["diff1_theta_initial"]).view("u4"),original.view("u4"))
    assert not np.array_equal(cp.asnumpy(resumed.thp),cp.asnumpy(model_state(cfg).thp))
    for field,actual in word_arrays(straight).items():
        np.testing.assert_array_equal(actual.view("u4"),word_arrays(resumed)[field].view("u4"),err_msg=field)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km",[2,4])
def test_legacy_metric_checkpoint_resumes_exact_trajectory(tmp_path,km):
    """The metric form resumes exactly from a checkpoint this build wrote.

    diff2-288-legacy-k*.npz are the current-build checkpoints
    (merged_metric_capture --record-checkpoints; sealed by
    test_288_legacy_checkpoints_are_sealed).  "Legacy" is the metric form's
    identity bytes: the default diff_opt=2/mix_full_fields selectors stay
    out of the config echo.  Every comparison runs on the card under test
    from the same stored bytes, so no word here depends on the card that
    recorded the fixture.  No tolerance: every word.
    """
    from dataclasses import replace
    from tools.wrf_diffopt1_oracle.model_case import configuration,model_state,word_arrays
    from woof.checkpoint_identity import DYCORE_MIXING_ALGORITHM_IDENTITY
    from woof.core.dycore import step
    from woof.io.restart import (restore_restart,read_restart_header,write_restart,
                                  RestartMismatchError)
    from tools.wrf_diffopt1_oracle.merged_metric_capture import (
        CHECKPOINT_STEPS,CURRENT_CHECKPOINT,checkpoint_configuration,checkpoint_payload_state)
    cfg=checkpoint_configuration(km)
    path=DATA/CURRENT_CHECKPOINT.format(km=km)
    header=read_restart_header(path)
    assert "diff_opt" not in header["config"]
    assert "mix_full_fields" not in header["config"]
    assert header["physics_setup"]["algorithms"]["dycore_mixing"]==DYCORE_MIXING_ALGORITHM_IDENTITY
    assert header["elapsed_seconds"]==CHECKPOINT_STEPS*cfg.dt
    # Independent direct payload seeding advances the stored state with the
    # production operator. It does not call the restore path.
    straight=checkpoint_payload_state(path,cfg)
    resumed=model_state(cfg);restore_restart(path,resumed,cfg)
    for _ in range(2):
        step(resumed,cfg)
        step(straight,cfg)
    straight_words=word_arrays(straight)
    resumed_words=word_arrays(resumed)
    assert set(resumed_words)==set(straight_words)
    for field,actual in resumed_words.items():
        np.testing.assert_array_equal(actual.view("u4"),straight_words[field].view("u4"),err_msg=field)
    # The same claim against an uninterrupted run: write at step 2 on this
    # card, resume, and match the 4-step trajectory word for word.
    uninterrupted=model_state(cfg);split=model_state(cfg)
    for _ in range(2*CHECKPOINT_STEPS):step(uninterrupted,cfg)
    for _ in range(CHECKPOINT_STEPS):step(split,cfg)
    written=write_restart(tmp_path/"split.npz",split,cfg)
    continued=model_state(cfg);restore_restart(written,continued,cfg)
    for _ in range(CHECKPOINT_STEPS):step(continued,cfg)
    continued_words=word_arrays(continued)
    for field,actual in word_arrays(uninterrupted).items():
        np.testing.assert_array_equal(continued_words[field].view("u4"),actual.view("u4"),err_msg=field)
    with pytest.raises(RestartMismatchError,match="mix_full_fields"):
        restore_restart(path,model_state(cfg),replace(cfg,mix_full_fields=False))
    # A checkpoint integrated before 2.8.8 records no dycore mixing identity
    # and is refused before restore, even under the terrain clock it ran (no
    # configuration change resumes it).  Since 2.9.11 it is also a format-6
    # file, and the format gate, which runs first, names that: it lacks the
    # stored MF_VX_INV map inverse format 7 carries.
    original=DATA/f"diff2-legacy-k{km}.npz"
    assert "dycore_mixing" not in read_restart_header(original)["physics_setup"]["algorithms"]
    old_cfg=replace(cfg,moist_cq=False,terrain_clock="measured")
    for live in (old_cfg,cfg):
        with pytest.raises(RestartMismatchError,match="checkpoint format 6"):
            restore_restart(original,model_state(live),live)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km",[2,4])
def test_legacy_metric_payload_continues_onto_288_pins(km):
    """The genuine a4177ebbf state, advanced by this build, keeps its 2.8.8 pins.

    Direct payload seeding only: the restore refuses that file (above).
    diff2-288-legacy.npz was measured on the RTX 5070 Ti (sm_120) like
    diff2-288-baseline.npz, so it certifies the Blackwell card.
    """
    from tools.wrf_diffopt1_oracle.model_case import configuration,word_arrays
    from woof.core.dycore import step
    from tools.wrf_diffopt1_oracle.merged_metric_capture import checkpoint_payload_state
    # Preserve the sealed pre-CQ trajectory and the measured terrain clock
    # (the default before 2.8.8) the checkpoint ran.
    cfg=replace(configuration(km=km,diff=2,mix=True),moist_cq=False,terrain_clock="measured")
    straight=checkpoint_payload_state(DATA/f"diff2-legacy-k{km}.npz",cfg)
    for _ in range(2):step(straight,cfg)
    with np.load(DATA/"diff2-288-legacy.npz") as expected:
        for field,actual in word_arrays(straight).items():
            np.testing.assert_array_equal(actual.view("u4"),expected[f"k{km}_"+field].view("u4"),err_msg=field)


def test_288_legacy_checkpoints_are_sealed():
    """The current-build checkpoints are the files their receipt names."""
    import hashlib
    from woof.checkpoint_identity import DYCORE_MIXING_ALGORITHM_IDENTITY
    from tools.wrf_diffopt1_oracle.merged_metric_capture import (
        CURRENT_CHECKPOINT,CURRENT_CHECKPOINT_RECEIPT)
    receipt=json.loads((DATA/CURRENT_CHECKPOINT_RECEIPT).read_text())
    assert receipt["dycore_mixing"]==DYCORE_MIXING_ALGORITHM_IDENTITY
    for km in (2,4):
        path=DATA/CURRENT_CHECKPOINT.format(km=km)
        assert receipt["checkpoints"][f"k{km}"]==hashlib.sha256(path.read_bytes()).hexdigest()


def test_metric_merge_receipt_preserves_original_words_and_seals_attribution():
    import hashlib
    from tools.wrf_diffopt1_oracle.capture import stats
    receipt=json.loads((DATA/"diff2-merged-attribution.json").read_text())
    digest=lambda name:hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    assert receipt["original_baseline_sha256"]==digest("diff2-baseline.npz")
    assert receipt["original_legacy_sha256"]==digest("diff2-legacy.npz")
    assert receipt["merged_baseline_sha256"]==digest("diff2-merged-baseline.npz")
    assert receipt["merged_legacy_sha256"]==digest("diff2-merged-legacy.npz")
    for km in (2,4):
        assert receipt["original_checkpoints"][f"k{km}"]==digest(f"diff2-legacy-k{km}.npz")
    moved=sum(row["merged_vs_original"]["different"] for row in receipt["fields"])
    assert moved==receipt["moved_words"]
    assert receipt["original_commit"]=="a4177ebbf342252405df3f6ed8309704daee94fc"
    assert {row["commit"] for row in receipt["control_groups"]}==set(receipt["native_provenance"])
    assert len(receipt["word_attribution_sha256"])==64
    assert len(receipt["controls"])==6
    assert all(row["runtime_sources"] for row in receipt["controls"])
    with np.load(DATA/"diff2-baseline.npz") as old,np.load(DATA/"diff2-merged-baseline.npz") as current:
        measured={key:stats(current[key],old[key]) for key in old.files}
    with np.load(DATA/"diff2-legacy.npz") as old,np.load(DATA/"diff2-merged-legacy.npz") as current:
        measured.update({"legacy_"+key:stats(current[key],old[key]) for key in old.files})
    assert {row["name"]:row["merged_vs_original"] for row in receipt["fields"]}==measured


def test_288_repin_receipt_seals_its_pins():
    """The 2.8.8 pins are the files the receipt measured, beside 2.8.7's."""
    import hashlib
    receipt=json.loads((DATA/"diff2-288-repin.json").read_text())
    digest=lambda name:hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    assert receipt["baseline_sha256"]==digest("diff2-288-baseline.npz")
    assert receipt["legacy_sha256"]==digest("diff2-288-legacy.npz")
    assert receipt["previous_baseline_sha256"]==digest("diff2-merged-baseline.npz")
    assert receipt["previous_legacy_sha256"]==digest("diff2-merged-legacy.npz")
    moved=receipt["moved_vs_2_8_7"]
    for kind,new,old in (("baseline","diff2-288-baseline.npz","diff2-merged-baseline.npz"),
                         ("legacy","diff2-288-legacy.npz","diff2-merged-legacy.npz")):
        with np.load(DATA/new) as current,np.load(DATA/old) as previous:
            assert set(current.files)==set(previous.files)==set(moved[kind+"_fields"])
            rows={key:{"words":int(current[key].size),
                       "moved":int((current[key].view("u4")!=previous[key].view("u4")).sum())}
                  for key in previous.files}
        assert rows==moved[kind+"_fields"]
        assert sum(row["moved"] for row in rows.values())==moved[kind]


@pytest.mark.parametrize("mutation", ("native_words", "compile_options", "source_bytes"))
def test_coordinate_current_native_proof_refuses_tampering(monkeypatch, mutation):
    import copy
    import assembled_legacy_proofs as proofs
    from tools.coordinate_donor_negative_control import compare_words
    # The current-boundary diagnostic counts signed zero as a changed word.
    native = np.array([0.0, 2.0], dtype=np.float32)
    control = np.array([-0.0, 2.0], dtype=np.float32)
    row = compare_words(control, native)
    assert row["words"] == 2 and row["different_words"] == 1
    assert row["control_sha256"] != row["native_sha256"]
    assert compare_words(native.copy(), native)["different_words"] == 0
    root = DATA.parents[2]
    proof = copy.deepcopy(proofs._load(root, "coordinate-native.json"))
    if mutation == "native_words":
        proof["native_different_words"] = 1
    elif mutation == "compile_options":
        proof["capture_identity"]["module_options"]["smag2d"].append("--ftz=true")
    else:
        proof["source_inputs"]["woof/core/kernels/smag2d.cu"]["sha256"] = "0" * 64
    monkeypatch.setattr(proofs, "_load", lambda root, name: proof)
    with pytest.raises(AssertionError):
        proofs.validate_coordinate(root)
