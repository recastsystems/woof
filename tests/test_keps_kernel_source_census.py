"""The source census must compile the same pinned parameters as the loader."""
from pathlib import Path


def test_parameter_source_segments_reconstruct_loader_source():
    from woof.core.kernels import module_source
    from woof.table_assets import kernel_parameter_source_pieces
    from tools.literal_division_census import production_units
    root = Path(__file__).resolve().parents[1]
    pieces = kernel_parameter_source_pieces("keps")
    assert len(pieces) == 1
    assert pieces[0][0].startswith("woof/data/keps/tables/")
    unit = next(unit for unit in production_units(root) if unit.key == "kernels:keps")
    assert unit.source == module_source("keps")
    assert pieces[0][1] in unit.source
    assert pieces[0][0] in {label for _, label, _ in unit.segments}


def test_unregistered_parameter_source_has_no_segments():
    from woof.table_assets import kernel_parameter_source, kernel_parameter_source_pieces
    assert kernel_parameter_source_pieces("acoustic") == ()
    assert kernel_parameter_source("acoustic") == ""
