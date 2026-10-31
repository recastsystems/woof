"""CPU validation of measured current-source numerical proof receipts."""
import hashlib
import json
from pathlib import Path

SOURCE_EXTENSION_SHA = '6da5e6c2a5cee6601cf991bacb44d6f62dbbef848852fffff0e6418e5b99086e'
DIFFUSION_CACHE_EXTENSION_SHA = "79275bdf13467cc1c3087889f42ee609ec9652bd65f3bd76b54d1f6cda5fd4ba"

PROOF_SHAS = {
    "acoustic-published.json": "316811b8273c32e0361e9f58ed2159bbb097a8b5854b4b55e7890d18f88c58bb",
    "coordinate-native.json": "05487c580fc67496607ccd86902ca55b17ad090cdec4d33a63e917d97174b067",
    "coordinate-negative.json": "7facf57ad4471296745ca105e1808b57e1d41d97d39bf5ea396c55104b01b312",
    "driver-source.json": "e5fc416818eea3d0ebd64ce322a9f1bee32ebb9b89f369bf4ba8cfeaa9b4e7d3",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(root, name):
    path = Path(root) / "tests/data/assembled_legacy_proofs" / name
    assert _sha(path) == PROOF_SHAS[name], name
    return json.loads(path.read_text())


def _identity(proof, modules):
    from woof.core.kernels import module_options, module_source
    capture = proof["capture_identity"]
    assert capture["device"] == "b'NVIDIA RTX PRO 6000 Blackwell Server Edition'"
    assert capture["compute_capability"] == [12, 0]
    assert capture["nvrtc_version"] == [13, 2]
    assert capture["cupy_version"] == "14.2.0"
    assert capture["strict"] is False
    assert capture["default_diffusion"] is True
    for module in modules:
        current = hashlib.sha256(module_source(module).encode()).hexdigest()
        if capture["module_source_sha256"][module] != current:
            _module_source_extension(module, capture["module_source_sha256"][module], current)
        assert capture["module_options"][module] == list(module_options(module))


def _sealed_extension(root):
    extension = Path(root) / "tests/data/assembled_legacy_proofs/source-extension.json"
    assert _sha(extension) == SOURCE_EXTENSION_SHA, "source extension seal"
    return json.loads(extension.read_text())


def _module_source_extension(module, before, after):
    # A kernel source that moved only on a path the receipts do not take (e.g. a GPUWM_WRF_EXACT-only branch)
    # is admitted by the same fresh base-versus-tip reproduction of every receipt with zero differing words,
    # recorded by tools/battery/loader_source_receipt.py --moved-kernel.
    measured = _sealed_extension(Path(__file__).resolve().parents[1])
    transition = measured.get("module_sources", {}).get(module)
    assert transition is not None, f"module {module} source moved without a measured replay"
    assert (transition["before"], transition["after"]) == (before, after), module
    for family, arrays, words in (("coordinate", 272, 178128),
                                  ("drivers", 2618, 31686592),
                                  ("acoustic", 132, 305632)):
        row = measured["measurements"][family]
        assert (row["arrays"], row["words"], row["changed_arrays"], row["different_words"]) == (arrays, words, 0, 0)


def sealed_source_sha256(root, name, recorded):
    """The digest a proof's recorded source input must have now: the recorded one, or the measured after-pin."""
    transition = _sealed_extension(root)["sources"].get(name)
    if transition is not None and transition["before"] == recorded and transition["after"] != recorded:
        return transition["after"]["sha256"]
    return recorded["sha256"]


def _diffusion_cache_extension(root, before, after):
    from woof.core.kernels import module_options, module_source

    folder = Path(root) / "tests/data/assembled_legacy_proofs"
    path = folder / "diffusion-cache-source-extension.json"
    assert _sha(path) == DIFFUSION_CACHE_EXTENSION_SHA, "diffusion cache extension seal"
    measured = json.loads(path.read_text())
    assert measured["schema"] == "diffusion-cache-source-extension-v1"
    assert measured["source"] == "woof/nvrtc_ptx_cache.py"
    assert measured["before"] == before and measured["after"] == after
    assert measured["generator_sha256"] == _sha(Path(root) / "tools/battery/diffusion_cache_source_receipt.py")
    assert measured["numerics"] == "strict"
    for module in ("smag2d", "diff_opt1"):
        assert measured["module_options"][module] == list(module_options(module))
        assert measured["module_source_sha256"][module] == hashlib.sha256(module_source(module).encode()).hexdigest()
    driver = _load(root, "driver-source.json")
    expected = {}
    for family in ("horizontal-driver", "vertical-driver", "km-mutations"):
        artifact = family + ".json"
        assert _sha(folder / artifact) == driver["artifacts"][artifact]
        pin = json.loads((folder / artifact).read_text())
        expected[family] = {key: {"words": row["words"], "gpu_sha256": row["gpu_sha256"]}
                            for key, row in _word_rows(pin["cases"]).items()}
    coordinate = _load(root, "coordinate-native.json")
    expected["coordinate"] = {key: {"words": row["current_to_native"]["words"],
                                     "gpu_sha256": row["current_sha256"]}
                              for key, row in coordinate["fields"].items()}
    assert measured["fields"] == expected, "diffusion cache extension changed frozen words"


def _source_extension(root, name, before, after, schema=None):
    if name == "woof/nvrtc_ptx_cache.py" and schema in (
            "assembled-driver-native-word-proof-v1", "assembled-coordinate-native-word-proof-v1"):
        # Only these two families were reproduced with the changed cache.
        # The original WRF receipts and the unrelated acoustic seal stay intact.
        _diffusion_cache_extension(root, before, after)
        return
    # The 2.8.9 captures stay immutable.  A sealed orchestration source that a
    # microphysics port had to extend (new mass packages in calc_cq and
    # slow_buoyancy, new loader rows) is admitted only by a fresh base-versus-tip
    # reproduction of every affected numerical receipt with zero differing words.
    extension = Path(root) / "tests/data/assembled_legacy_proofs/source-extension.json"
    assert _sha(extension) == SOURCE_EXTENSION_SHA, "source extension seal"
    measured = json.loads(extension.read_text())
    assert measured["schema"] == "measured-legacy-source-extension-v1"
    assert measured["baseline"] == "cd336e62d6e6b6312c9636e0fdfff84801909b6a"
    for family, arrays, words in (("coordinate", 272, 178128),
                                  ("drivers", 2618, 31686592),
                                  ("acoustic", 132, 305632)):
        row = measured["measurements"][family]
        assert (row["arrays"], row["words"], row["changed_arrays"], row["different_words"]) == (arrays, words, 0, 0)
    transition = measured["sources"].get(name)
    assert transition is not None, name
    assert transition["before"] == before, name
    assert transition["after"] == after, name
    if name == "woof/core/dycore.py" and "moving_frame_source_replay" in measured:
        seal = measured["moving_frame_source_replay"]
        assert seal["receipt"] == "moving-frame-source-replay.json"
        path = extension.parent / seal["receipt"]
        assert _sha(path) == seal["sha256"], "moving frame replay seal"
        replay = json.loads(path.read_text())
        assert replay["schema"] == "measured-moving-frame-source-replay-v1"
        assert replay["base"] == "afe2905d568d41880ee04a1a26d3efeb80791034"
        assert replay["sources"][name]["after"] == after
        assert replay["measurements"] == measured["measurements"]
        assert replay["numerics"] == "strict"
        assert replay["generator"] == "tools/assembled_legacy_replay.py"
        assert replay["recorder"] == "tools/battery/moving_frame_source_receipt.py"
        assert replay["measurement_tool"] == "tools/battery/fusion_source_receipt.py"
        for role in ("generator", "recorder", "measurement_tool"):
            assert replay[role + "_sha256"] == _sha(Path(root) / replay[role])
        assert set(replay["dependencies"]) == {"woof/moving_frame.py", "woof/config.py"}
        for dependency, pin in replay["dependencies"].items():
            source = Path(root) / dependency
            assert pin == {"bytes": source.stat().st_size, "sha256": _sha(source)}
    if name == "woof/core/dycore.py" and "forecast_entry_replay" in measured:
        seal = measured["forecast_entry_replay"]
        assert seal["receipt"] == "forecast-entry-source-replay.json"
        path = extension.parent / seal["receipt"]
        assert _sha(path) == seal["sha256"], "forecast entry source replay seal"
        replay = json.loads(path.read_text())
        assert replay["purpose"] == "forecast entry admission"
        assert replay["base"] == "afe2905d568d41880ee04a1a26d3efeb80791034"
        assert replay["measurements"] == measured["measurements"]
        assert replay["numerics"] == "strict" and not replay["pin_changes"]
        assert replay["before"] == replay["after"] == measured["sources"]["woof/core/kernels/__init__.py"]["after"]
        assert replay["moved_sources"] == {name: {"sealed": seal["before"], "after": after}}
        assert replay["generator"] == "tools/assembled_legacy_replay.py"
        assert replay["recorder"] == "tools/battery/loader_source_receipt.py"
        assert replay["admission_recorder"] == "tools/battery/forecast_entry_source_receipt.py"
        for role in ("generator", "recorder", "admission_recorder"):
            assert replay[role + "_sha256"] == _sha(Path(root) / replay[role])
    if name == "woof/core/kernels/__init__.py" and "loader_slimming_replay" in measured:
        seal = measured["loader_slimming_replay"]
        assert seal["receipt"] == "slimming-loader-source-replay.json"
        path = extension.parent / seal["receipt"]
        assert _sha(path) == seal["sha256"], "slimming loader replay seal"
        replay = json.loads(path.read_text())
        assert replay["schema"] == "measured-slimming-loader-source-replay-v1"
        assert replay["source"] == name and replay["after"] == after
        # Re-measured for 2.9.7 (the S5 loader rows and the S2/S5 dycore
        # hooks): base 2.9.6 (e5b0c7ff52), zero differing words in all three
        # families, dycore.py re-earned as a moved source.  Previously
        # 89107bdeb2 (the 2.9.4 slimming).
        # Re-measured for 2.9.10 fix batch 9 (c49be92105, the WRF-file nest door, moved dycore.py: strict
        # per-domain p_perturbation): base 2.9.10 pass 5 (30c12f7ae6), zero differing words in all three
        # families on an RTX 5070 Ti. Previously e5b0c7ff52 (2.9.6).
        # Re-measured for 2.9.11 (the folds moved dycore.py, acoustic.py, wrf_exact.py and the loader, and
        # acoustic.cu on its GPUWM_WRF_EXACT-only branches): base 2.9.10 (b8165c59fe) against fold/2.9.11-eb3
        # f5e393fb8c, zero differing words in all three families on an RTX 5070 Ti; acoustic.cu admitted as a
        # moved kernel (--moved-kernel). Previously 30c12f7ae6 (2.9.10 pass 5).
        # Re-measured for 2.9.11 fix 6 (fix/hailcast: the loader names the hail module's third header,
        # hailcast_wrf39.cuh): base fold/2.9.11 fix 4 (a1d20edc51) against fold/2.9.11-fix6 5428796cb0, zero
        # differing words in all three families on an RTX 5090 (Blackwell, sm_120). Previously b8165c59fe.
        assert replay["base"] == "a1d20edc5115f9e42d898a9884bfcbbc2130e80d"
        assert replay["measurements"] == measured["measurements"]
        assert replay["numerics"] == "strict"
        assert replay["generator"] == "tools/assembled_legacy_replay.py"
        assert replay["recorder"] == "tools/battery/loader_source_receipt.py"
        for role in ("generator", "recorder"):
            assert replay[role + "_sha256"] == _sha(Path(root) / replay[role])


def _sources(root, proof):
    assert proof["source_inputs"]
    for name, row in proof["source_inputs"].items():
        path = Path(root) / name
        actual = {"bytes": path.stat().st_size, "sha256": _sha(path)}
        if actual != row:
            _source_extension(root, name, row, actual, proof.get("schema"))


def validate_published_acoustic(root):
    import numpy as np
    from woof.core.kernels import diffusion_kernel, function_options, module_options
    proof = _load(root, "acoustic-published.json")
    assert proof["schema"] == "assembled-published-acoustic-word-proof-v1"
    assert (proof["array_comparisons"], proof["words"], proof["changed_published_words"]) == (132, 305632, 0)
    assert proof["published_exact"] is True
    assert proof["default_acoustic_diffusion_route"] is False
    assert diffusion_kernel("acoustic", "advance_w_phi") is False
    for entry in ("calc_coefs", "advance_w_phi", "advance_w_phi_msf"):
        assert proof["acoustic_function_options"][entry] == list(function_options("acoustic", entry, module_options("acoustic"))) == ["-std=c++17"]
    _identity(proof, ("acoustic",))
    _sources(root, proof)
    assert {"woof/core/kernels/acoustic.cu", "woof/verify/smallstep_vertical_oracle.py",
            "tools/smallstep_wrf471_oracle/vertical_workspace.py"} <= set(proof["source_inputs"])
    path = Path(root) / "tests/data/wrf471_smallstep/vertical-reference.npz"
    assert _sha(path) == proof["recorded_fixture_sha256"]
    with np.load(path, allow_pickle=False) as frozen:
        expected = {key[:-7].replace("__", "/"): key for key in frozen.files if key.endswith("_native")}
        assert set(proof["fields"]) == set(expected)
        for name, key in expected.items():
            row, array = proof["fields"][name], frozen[key]
            assert row["different_words"] == 0 and row["all_uint32_words_equal_frozen"] is True
            assert row["words"] == array.size and array.dtype == np.float32
            assert row["actual_sha256"] == row["recorded_sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    return proof


def validate_coordinate(root):
    import numpy as np
    proof = _load(root, "coordinate-native.json")
    assert proof["schema"] == "assembled-coordinate-native-word-proof-v1"
    assert (proof["case_count"], proof["field_count"], proof["words"], proof["native_different_words"]) == (88, 272, 178128, 0)
    assert proof["native_exact"] is True
    assert (proof["prior_gpu_different_words"], proof["previously_native_equal_words"],
            proof["changed_previously_native_equal_words"]) == (25435, 152693, 0)
    _identity(proof, ("smag2d", "diff_opt1"))
    _sources(root, proof)
    data = Path(root) / "tests/data/wrf471_diff_opt1"
    assert _sha(data / "wrf471.npz") == proof["native_archive_sha256"]
    assert _sha(data / "wrf471.json") == proof["native_case_schema_sha256"]
    assert _sha(data / "merged-gpu.npz") == proof["prior_gpu_archive_sha256"]
    cases = json.loads((data / "wrf471.json").read_text())["cases"]
    families = {"horizontal": ("tendency",), "km4": ("km", "kh"), "deform": ("d11", "d22", "d12"),
                "km2": ("km", "kh", "kmv", "khv", "tke", "bn2")}
    expected = {case["name"] + "_" + field: case["name"] + ("_expected" if case["family"] == "horizontal" else "_expected_" + field)
                for case in cases for field in families[case["family"]]}
    assert set(proof["fields"]) == set(expected)
    with np.load(data / "wrf471.npz", allow_pickle=False) as native, np.load(data / "merged-gpu.npz", allow_pickle=False) as prior:
        for name, native_name in expected.items():
            row, reference, old = proof["fields"][name], native[native_name], prior[name]
            assert reference.dtype == old.dtype == np.float32 and reference.shape == old.shape
            assert row["current_to_native"]["words"] == reference.size
            assert row["current_to_native"]["different_words"] == 0
            assert row["current_sha256"] == row["native_sha256"] == hashlib.sha256(reference.tobytes()).hexdigest()
            assert row["prior_gpu_sha256"] == hashlib.sha256(old.tobytes()).hexdigest()
            equal = int(np.count_nonzero(old.view(np.uint32) == reference.view(np.uint32)))
            assert row["previously_native_equal_words"] == equal
            assert row["changed_previously_native_equal_words"] == 0
    return proof


def _word_rows(value, prefix=""):
    if isinstance(value, dict) and "gpu_sha256" in value:
        return {prefix: value}
    rows = {}
    for name, child in value.items():
        rows.update(_word_rows(child, prefix + "/" + name if prefix else name))
    return rows


def validate_drivers(root):
    root = Path(root)
    proof = _load(root, "driver-source.json")
    assert proof["schema"] == "assembled-driver-native-word-proof-v1"
    _identity(proof, ("smag2d",))
    folder = root / "tests/data/assembled_legacy_proofs"
    artifacts = {}
    for name, digest in proof["artifacts"].items():
        assert _sha(folder / name) == digest, name
        artifacts[name] = json.loads((folder / name).read_text())
    _sources(root, proof)
    native = artifacts["vertical-native-control.json"]
    assert native["schema"] == "vertical-declared-xkmv-native-control-v1"
    assert (native["case_count"], native["arrays"], native["words"],
            native["stock_w_control_different_words"], native["non_w_control_different_words"]) == (14, 1764, 21500976, 143424, 0)
    _sources(root, {"source_inputs": native["tool_inputs"]})
    for name, row in native["fixtures"].items():
        path = root / "tests/data/wrf471_diffusion/vertical-driver" / name
        assert path.stat().st_size == row["bytes"] and _sha(path) == row["sha256"]
    manifest = root / "tests/data/wrf471_diffusion/vertical-driver/manifest.json"
    assert _sha(manifest) == native["stock_manifest_sha256"]
    for receipt in (native["library_build_receipt"], native["preparation_build_receipt"]):
        assert receipt["wrf_commit"] == "f52c197ed39d12e087d02c50f412d90d418f6186"
        assert "-ffp-contract=off" in receipt["commands"][0] and "-fcheck=bounds" in receipt["commands"][0]
    totals = {"horizontal-driver": (364, 5575696), "km-mutations": (490, 4609920),
              "vertical-driver": (1764, 21500976)}
    answer = {}
    for family, (arrays, words) in totals.items():
        pin = artifacts[family + ".json"]
        assert pin["kernel_source_sha256"] == proof["capture_identity"]["module_source_sha256"]["smag2d"]
        assert pin["device"] == proof["capture_identity"]["device"]
        rows = _word_rows(pin["cases"])
        assert len(rows) == arrays and sum(row["words"] for row in rows.values()) == words
        if family != "vertical-driver":
            assert all(row["different_words"] == row["max_ulp"] == 0 for row in rows.values())
        else:
            assert set(rows) == set(native["fields"])
            stock_differences = 0
            for key, row in rows.items():
                reference = native["fields"][key]
                assert row["words"] == reference["words"]
                assert row["gpu_sha256"] == reference["controlled_native_sha256"], key
                assert reference["stock_reproduction"]["different_words"] == 0
                # Isotropic packets already have identical kmh/kmv words.
                # Require the exact byte-change inventory, including that
                # zero-change control, rather than claiming a change there.
                changed = ([] if reference["stock_w_coefficient_sha256"] ==
                           reference["declared_w_coefficient_sha256"] else ["kmh"])
                assert reference["changed_driver_inputs"] == changed
                assert (reference["stock_w_coefficient"], reference["declared_w_coefficient"]) == ("xkmh", "xkmv")
                assert row["different_words"] == reference["control_to_stock"]["different_words"]
                stock_differences += row["different_words"]
                if key.rsplit("/", 1)[1] != "w":
                    assert row["different_words"] == 0
                    assert reference["stock_native_sha256"] == reference["controlled_native_sha256"]
            assert stock_differences == 143424
        answer[family] = pin
    return answer


def validate_coordinate_negative(root):
    import ast
    from woof.core.kernels import module_source
    positive = validate_coordinate(root)
    proof = _load(root, "coordinate-negative.json")
    assert proof["schema"] == "current-coordinate-donor-negative-control-v1"
    assert proof["positive_proof_sha256"] == PROOF_SHAS["coordinate-native.json"]
    assert (proof["case_count"], proof["field_count"], proof["words"], proof["changed_native_words"]) == (40, 216, 139968, 492)
    assert proof["native_archive_sha256"] == positive["native_archive_sha256"]
    assert proof["capture_identity"] == positive["capture_identity"]
    assert proof["control_device"] == ast.literal_eval(positive["capture_identity"]["device"]).decode()
    assert proof["source_inputs"] == positive["source_inputs"]
    capture = Path(root) / "tools/wrf_diffopt1_oracle/capture.py"
    assert _sha(capture) == proof["capture_tool_sha256"]
    assert _sha(Path(root) / "tools/coordinate_donor_negative_control.py") == proof["measurement_tool_sha256"]
    tree = ast.parse(capture.read_text())
    donor = next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "donor" for target in node.targets))
    production = module_source("smag2d")
    assert production.count(donor) == 1
    assert proof["controlled_module_source_sha256"]["smag2d"] == hashlib.sha256(production.replace(donor, "").encode()).hexdigest()
    assert proof["controlled_module_source_sha256"]["diff_opt1"] == positive["capture_identity"]["module_source_sha256"]["diff_opt1"]
    expected = {name for name, row in positive["fields"].items() if row["family"] in ("km2", "deform")}
    assert set(proof["fields"]) == expected
    assert proof["changed_fields"] == {"deform_1_d12": 54, "deform_2_d12": 72, "deform_3_d12": 120,
                                      "deform_5_d12": 54, "deform_6_d12": 72, "deform_7_d12": 120}
    for name, row in proof["fields"].items():
        assert row["native_sha256"] == positive["fields"][name]["native_sha256"]
        assert row["words"] == positive["fields"][name]["current_to_native"]["words"]
        assert row["different_words"] == proof["changed_fields"].get(name, 0)
        if name not in proof["changed_fields"]:
            assert row["control_sha256"] == row["native_sha256"]
    return proof
