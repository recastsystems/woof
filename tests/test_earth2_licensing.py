"""Restricted weights cannot enter another source door through an alias or receipt."""
import json
import pytest

from woof.earth2_licensing import check_argv, licence_scope, model_row, require_licence

LICENCE = "CC BY-NC-SA 4.0"


@pytest.mark.parametrize("door", ["fetch", "prep", "domain", "go", "source-intermediate"])
def test_every_default_door_refuses_restricted_weights(door):
    with pytest.raises(ValueError, match=LICENCE):
        check_argv([door, "--source", "earth2-pangu"])
    check_argv([door, "--source", "earth2-pangu", "--accept-licence", LICENCE])


def test_exact_licence_and_normalized_source_are_required():
    with pytest.raises(ValueError, match=LICENCE):
        check_argv(["fetch", "--source=EARTH2_PANGU", "--accept-licence", "research"])
    with pytest.raises(ValueError, match=LICENCE):
        check_argv(["fetch", "--source", "earth2", "--model", "pangu"])


def test_source_intermediate_registers_the_named_licence_option(tmp_path):
    from woof.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "source-intermediate", "--source", "earth2-pangu",
        "--input", str(tmp_path / "forecast.nc"),
        "--valid-time", "2024-05-21T18:00:00",
        "--out", str(tmp_path / "output"),
        "--work-dir", str(tmp_path / "work"),
        "--accept-licence", LICENCE,
    ])
    assert args.accept_licence == [LICENCE]


def test_unresolved_terms_require_their_own_name():
    key, row = model_row("earth2-fengwu")
    with pytest.raises(ValueError, match="unresolved"):
        require_licence(key, row)
    require_licence(key, row, row["weights"]["licence"])


def test_acceptance_scope_does_not_leak_into_the_next_command():
    key, row = model_row("earth2-pangu")
    with licence_scope([LICENCE]):
        require_licence(key, row)
    with pytest.raises(ValueError, match=LICENCE):
        require_licence(key, row)


def test_current_authority_overrides_a_receipt_claim(tmp_path):
    receipt = tmp_path / "earth2-run.json"
    receipt.write_text(json.dumps({"model": "pangu", "model_row": {"licence_class": "permissive"}}))
    source = tmp_path / "renamed.nc"
    with pytest.raises(ValueError, match=LICENCE):
        check_argv(["prep", "--source", "earth2", "--input", str(source)])
    check_argv(["prep", "--source", "earth2", "--input", str(source), "--accept-licence", LICENCE])
    config = tmp_path / "run.toml"
    config.write_text(f'[fetch]\nsource = "earth2"\nsource_root = "{tmp_path}"\n')
    with pytest.raises(ValueError, match=LICENCE):
        check_argv(["go", str(config)])


def test_model_hidden_in_config_is_checked(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('[fetch]\nsource = "earth2"\nmodel = "pangu"\n')
    with pytest.raises(ValueError, match=LICENCE):
        check_argv(["go", str(config)])
    check_argv(["go", str(config), "--accept-licence", LICENCE])


def test_graphcast_weights_are_cc_by_and_run_bare(tmp_path):
    """GraphCast's weights are CC BY 4.0 since 2026-08-06 (google-deepmind/graphcast
    docs/weathernext1_graph/README.md), the terms the GenCast and WeatherNext 2 rows carry."""
    for name in ("graphcast-operational", "graphcast-small"):
        key, row = model_row(f"earth2-{name}")
        assert (row["licence_class"], row["weights"]["licence"]) == ("permissive", "CC BY 4.0")
        require_licence(key, row)
        check_argv(["fetch", "--source", f"earth2-{name}"])
    config = tmp_path / "run.toml"
    config.write_text('[fetch]\nsource = "earth2"\nmodel = "graphcast-small"\n')
    check_argv(["go", str(config)])


WINDGUST_LICENCE = "NVIDIA Community Model License"


def test_diagnostics_door_refuses_restricted_consumer_weights_by_name():
    """A consumers row under the NVIDIA Community licence never runs by default."""
    with pytest.raises(ValueError, match=WINDGUST_LICENCE):
        check_argv(["earth2-diagnose", "export.zarr", "--for", "windgust-afno", "--out", "diag"])
    with pytest.raises(ValueError, match=WINDGUST_LICENCE):
        check_argv(["earth2-diagnose", "export.zarr", "--for", "tc-tracker,windgust-afno", "--out", "diag"])
    check_argv(["earth2-diagnose", "export.zarr", "--for", "windgust-afno", "--out", "diag",
                "--accept-licence", WINDGUST_LICENCE])
    # The algorithm-only tracker (Apache-2.0, no weights) is permissive and runs bare.
    check_argv(["earth2-diagnose", "export.zarr", "--for", "tc-tracker", "--out", "diag"])


def test_diagnostics_door_refuses_inside_resolve_too():
    """Direct callers of the door (no argv pre-check) hit the same gate, and the option exists."""
    from woof.earth2_diagnose import resolve
    from woof.cli import build_parser

    with pytest.raises(ValueError, match=WINDGUST_LICENCE):
        resolve("windgust-afno")
    assert resolve("windgust-afno", accept_licence=(WINDGUST_LICENCE,)) == ["windgust-afno"]
    assert resolve("tc-tracker") == ["tc-tracker"]
    # Undeclared NGC terms are unresolved, not permissive.
    with pytest.raises(ValueError, match="unresolved"):
        resolve("precip-afno")
    args = build_parser().parse_args(["earth2-diagnose", "export.zarr", "--for", "windgust-afno",
                                      "--out", "diag", "--accept-licence", WINDGUST_LICENCE])
    assert args.accept_licence == [WINDGUST_LICENCE]
