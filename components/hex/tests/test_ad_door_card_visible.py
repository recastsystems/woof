import argparse

import pytest

from woof.hex import ad_doors


def _a(card=None, cards=None):
    return argparse.Namespace(card=card, cards=cards)


def test_card_counts_inside_scheduler_cards(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert ad_doors.card_of(_a(card=0)) == "7"
    assert ad_doors.cards_of(_a(card=0)) == ["7"]
    with pytest.raises(SystemExit):
        ad_doors.card_of(_a(card=1))


def test_cards_list_counts_inside_scheduler_cards(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert ad_doors.cards_of(_a(cards="1,0")) == ["5", "4"]


def test_card_plain_when_no_scheduler(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ad_doors.card_of(_a(card=1)) == "1"
    assert ad_doors.card_of(_a()) == "0"
