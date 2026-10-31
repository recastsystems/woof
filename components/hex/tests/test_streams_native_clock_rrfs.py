"""Native Gregorian stream clocks, including calendar and fractional intervals.

The oracle is MPAS's strict Fortran with its actual bundled ESMF; this is a
serial clock/contract comparison, never whole-forecast landing proof.
"""
from datetime import datetime, timedelta
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from woof.hex.streams_rrfs import (Stream, StreamAlarms, _advance, _clock,
                                 _interval, parse_streams)


@pytest.mark.parametrize('interval',[
    '1-0_00:00:00', '0-1-0_00:00:00', '1-0-0_00:00:00',
    '0-2-3_01:02:03.123456789', '0-13-0_00:00:00',
    '90:00', '0_0:0:1.25', '86400', '0', '0.00000001'])
@pytest.mark.parametrize('offset',[-70,0,70])
@pytest.mark.parametrize('origin_text',['2000-01-31_12:34:56.123456','2000-02-29_00:00:00'])
def test_clock_columns_match_strict_native(interval,offset,origin_text,tmp_path):
    oracle = os.environ.get('RRFS_NATIVE_STREAM_CLOCK')
    if not oracle:
        pytest.skip('strict standalone native clock oracle not supplied')
    origin = datetime.fromisoformat(origin_text.replace('_','T'))
    start = origin+timedelta(days=offset)
    probes = [start, start+timedelta(microseconds=1), start+timedelta(seconds=1.25),
              start+timedelta(seconds=600), start+timedelta(days=1),
              start+timedelta(days=28), start+timedelta(days=29), start+timedelta(days=60),
              start+timedelta(days=365), start+timedelta(days=730), start+timedelta(days=1461)]
    resets = [False,False,True,False,True,False,True,False,True,False,True]
    stamp = lambda when: when.strftime('%Y-%m-%d_%H:%M:%S.%f')
    inputs = [stamp(origin),stamp(start),interval,str(len(probes))]
    for when,reset in zip(probes,resets):
        inputs.extend((stamp(when),str(int(reset))))
    result = subprocess.run([oracle],input='\n'.join(inputs)+'\n',text=True,
                            capture_output=True,check=True,timeout=30)
    all_rows = result.stdout.strip().splitlines()
    rows = [row for row in all_rows if '|' in row]
    diagnostics = [row.strip() for row in all_rows if '|' not in row]
    # Bundled wrf_error_fatal prints and returns; timeintcmp then performs a
    # defined comparison. Preserve that witness without treating it as abort.
    assert all(row=='wrf_error_fatal: timeintcmp:  Can not compare two intervals with different months and years'
               for row in diagnostics),result.stdout
    assert len(rows)==len(probes),result.stdout
    # Parse the native deck, including its calendar interval spellings.
    deck = tmp_path/'streams.atmosphere'
    deck.write_text(f'''<streams><stream name="fixture" type="output"
      filename_template="$Y-$M-$D_$h:$m:$s" filename_interval="output_interval"
      output_interval="{interval}"><var name="theta"/></stream></streams>''')
    streams = parse_streams(deck)
    stream = streams['fixture']
    alarms = StreamAlarms(streams,start,origin,probes[-1])
    for when,reset,row in zip(probes,resets,rows):
        filename,ringing,previous,added,*fractions = row.split('|')
        assert stream.filename(when,origin)==filename
        assert alarms.due('fixture',when)==(ringing=='T'),row
        if reset:
            alarms.reset('fixture',when)
        expected_previous = _clock(datetime.fromisoformat(previous.replace('_','T')))
        expected_added = _clock(datetime.fromisoformat(added.replace('_','T')))
        if fractions:
            pn,pd,an,ad = map(int,fractions)
            expected_previous += Fraction(pn,pd)
            expected_added += Fraction(an,ad)
        else:
            pytest.fail('Native oracle must report rational fractional seconds')
        assert alarms._previous['fixture','output']==expected_previous,row
        assert _advance(_clock(when),_interval(interval))==expected_added,row
    receipt = os.environ.get('RRFS_NATIVE_STREAM_CLOCK_RECEIPT')
    if receipt:
        with Path(receipt).open('a') as output:
            output.write(json.dumps({'schema':'mpas-stream-native-clock-column/v1',
                'interval':interval,'origin':stamp(origin),'start':stamp(start),
                'probe_count':len(probes),'different_time_values':0,
                'native_diagnostics':diagnostics,
                'native_process_count':1,'transport_process_count':1,
                'whole_model_proof':False})+'\n')


def test_calendar_deck_and_one_off_alarm_remain_ringing(tmp_path):
    deck = tmp_path/'streams.atmosphere'
    deck.write_text('''<streams><stream name="fixture" type="output"
      filename_template="$Y-$M-$D" output_interval="0-1-0_00:00:00">
      <var name="theta"/></stream></streams>''')
    streams = parse_streams(deck)
    start = datetime(2000,1,31)
    assert streams['fixture'].filename(datetime(2000,2,29),start)=='2000-02-29'
    one_off = Stream('once',{'type':'input','input_interval':'initial_only'},())
    alarms = StreamAlarms({'once':one_off},start,start,start+timedelta(days=1))
    assert alarms.due('once',start+timedelta(hours=1),'input')
    alarms.reset('once',start+timedelta(hours=1),'input')
    assert not alarms.due('once',start+timedelta(hours=2),'input')


def test_startup_reset_preserves_future_final_only_alarm():
    stream = Stream('output',{'type':'output','output_interval':'final_only'},())
    start = datetime(2000,1,31)
    end = start+timedelta(days=1)
    alarms = StreamAlarms({'output':stream},start,start,end,do_restart=True)
    assert not alarms.due('output',start)
    assert alarms.due('output',end)
    alarms.reset('output',end)
    assert not alarms.due('output',end)
