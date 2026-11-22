# SPDX-License-Identifier: Apache-2.0
"""A card hidden with CUDA_VISIBLE_DEVICES is not handed to the runner."""
import pytest

from woof.hex import ad_doors as d


@pytest.mark.parametrize("word", ["", " ", "-1"])
def test_a_hidden_card_is_refused_before_the_runner_starts(monkeypatch, word):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", word)
    with pytest.raises(d.AdDoorError, match="no card is visible"):
        d.refuse_hidden_card()


@pytest.mark.parametrize("word", [None, "0", "1,2"])
def test_a_named_card_or_none_set_goes_on(monkeypatch, word):
    if word is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", word)
    d.refuse_hidden_card()


@pytest.mark.parametrize("read", ["free_mib", "total_mib", "compute_capability"])
@pytest.mark.parametrize("word", ["", "-1"])
def test_a_hidden_card_is_not_read_for_its_memory_or_family(monkeypatch, read, word):
    # THE BREAKAGE THIS PREVENTS: with every card hidden the door still asked nvidia-smi (which ignores
    # CUDA_VISIBLE_DEVICES) about card 0 and said "card 0 has 7341 MiB free ... Free the card" (train, build 59da dry run)
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", word)
    asked = []
    monkeypatch.setattr(d.subprocess, "run", lambda *a, **k: asked.append(a))
    with pytest.raises(d.AdDoorError, match="no card is visible"):
        getattr(d, read)("0")
    assert asked == []
