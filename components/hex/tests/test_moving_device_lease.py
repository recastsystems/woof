"""--device on the moving doors indexes the cards the caller can already see (a card queue's lease holds).

Breakage (V8X6000, 2026-10-29): under a lease of card 6 (CUDA_VISIBLE_DEVICES=<uuid>), ``moving run --device 0``
put the segment on physical card 0, held by another lane's exclusive lease.
"""
import pytest

from woof.hex.moving_run import visible_devices_for


def test_inherited_lease_is_indexed_not_replaced():
    env = {"CUDA_VISIBLE_DEVICES": "GPU-cb4d1604-8ce0-f994-5167-982427b12adf"}
    assert visible_devices_for(0, env) == "GPU-cb4d1604-8ce0-f994-5167-982427b12adf"
    assert visible_devices_for("1", {"CUDA_VISIBLE_DEVICES": "6,7"}) == "7"


def test_no_inherited_list_keeps_the_physical_index():
    assert visible_devices_for(0, {}) == "0"
    assert visible_devices_for("3", {"CUDA_VISIBLE_DEVICES": ""}) == "3"


def test_index_past_the_visible_cards_is_refused():
    with pytest.raises(SystemExit):
        visible_devices_for(1, {"CUDA_VISIBLE_DEVICES": "GPU-only-one"})


def test_a_hidden_card_list_is_nothing_inherited():
    """CUDA_VISIBLE_DEVICES=-1 (the CPU battery's conftest, "no card") leaves no card to index: the physical index."""
    assert visible_devices_for("1", {"CUDA_VISIBLE_DEVICES": "-1"}) == "1"
    assert visible_devices_for(0, {"CUDA_VISIBLE_DEVICES": "3,-1,4"}) == "3"
