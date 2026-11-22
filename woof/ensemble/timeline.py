"""Relay ordinary member step metadata into the shared C1 ledger."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def member_timeline(directory, *, member, card):
    from woof.stage_timing import ForecastLedger
    from woof.timeline.ledger import Ledger, _warn

    from woof.core.card_queue import granted_cards
    physical = granted_cards()
    mapped = int(physical[card]) if card < len(physical) and all(part.isdecimal() for part in physical) else card
    target = Ledger(Path(directory) / 'timeline.jsonl', product='ensemble', card=mapped)

    class MemberLedger:
        def emit(self, stage, phase='mark', *, attrs=None, **clocks):
            return target.emit(stage, phase, attrs={**(attrs or {}),
                'member_id': member, 'cuda_device_id': card,
                'span_id': f'ensemble-member-{member}'}, **clocks)

    watch = None
    try:
        watch = ForecastLedger(MemberLedger(), directory)
        watch.start()
    except BaseException as error:
        _warn(target.path, error)
    try:
        yield
    finally:
        if watch is not None:
            try:
                watch.finish()
            except BaseException as error:
                _warn(target.path, error)
