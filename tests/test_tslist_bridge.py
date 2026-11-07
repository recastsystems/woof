"""Exercise the actual Rust ABI, including failed resume before mutation."""
import os
from pathlib import Path
import struct
import time
import unittest

import numpy as np

from woof.tslist_bridge import RustTslistWriter, read_stations, place_stations, validate_files


def bits(value):
    return f"{struct.unpack('I', struct.pack('f', value))[0]:08x}"


def placement():
    return "\t".join(["1", "35", "30", bits(34.809), bits(-81.719), bits(176.9),
                      "v1", bits(35), bits(30), "proof", "Native real input"]) + "\n"


class RustAbiTest(unittest.TestCase):
    def directory(self, name):
        path = Path(os.environ["TSLIST_TEST_SCRATCH"]) / f"{name}-{time.time_ns()}"
        path.mkdir(parents=True)
        return path

    def open(self, path, saved=None, placements=None):
        return RustTslistWriter(path, placement() if placements is None else placements,
            geographic=False, domain=1, levels=2, start="2000-01-24_12:00:00", saved=saved)

    def test_real_abi_checkpoint_resume(self):
        path = self.directory("ffi")
        writer = self.open(path)
        batch = np.ones((2, 1, 29), np.float32)
        batch[:, 0, 0] = [0.05, 0.10]
        writer.write_batch(batch[:1], 1)
        saved = writer.checkpoint()
        writer.write_batch(batch[1:], 1)
        writer.close()
        continued = {p.name: p.read_bytes() for p in path.iterdir()}
        writer = self.open(path, saved)
        writer.write_batch(batch[1:], 1)
        self.assertEqual(continued, {p.name: p.read_bytes() for p in path.iterdir()})
        with self.assertRaisesRegex(RuntimeError, "nonincreasing"):
            writer.write_batch(batch[1:], 1)
        writer.close()

    def test_bad_plan_and_shape_fail_without_rows(self):
        path = self.directory("ffi-errors")
        with self.assertRaisesRegex(RuntimeError, "eleven"):
            self.open(path, placements="bad plan\n")
        self.assertEqual(list(path.iterdir()), [])
        writer = self.open(path)
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "shape"):
            writer.write_batch(np.ones((1, 28), np.float32), 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
        writer.close()

    def test_native_file_read_and_real_nint(self):
        path = self.directory("placement") / "tslist"
        path.write_bytes((Path(__file__).parents[1] / "tools/tslist_wrf471_oracle/tslist").read_bytes())
        geographic, plan = read_stations(path, 1)
        self.assertFalse(geographic)
        row = place_stations(plan, np.array([35.4999999]), np.array([30.5]), 73, 60).split("\t")
        # REAL first: the first value rounds to binary32 35.5 before NINT.
        self.assertEqual(row[:3], ["1", "36", "31"])
        self.assertEqual(place_stations(plan, np.array([74.]), np.array([30.]), 73, 60), "")

    def test_read_only_prefix_validation(self):
        path = self.directory("validation")
        writer = self.open(path)
        batch = np.ones((2, 1, 29), np.float32)
        batch[:, 0, 0] = [0.05, 0.1]
        writer.write_batch(batch[:1], 1)
        saved = writer.checkpoint()
        writer.write_batch(batch[1:], 1)
        writer.close()
        receipt = dict(directory=str(path), plan=placement(), geographic=False,
            domain=1, levels=2, start="2000-01-24_12:00:00", files=saved)
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        validate_files(receipt)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
        target = path / "proof.d01.PR"
        damaged = bytearray(target.read_bytes())
        damaged[0] ^= 1
        target.write_bytes(damaged)
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "prefix"):
            validate_files(receipt)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})


class NativeGoAdmissionTest(unittest.TestCase):
    def test_unsupported_explicit_options_are_not_silently_dropped(self):
        import argparse
        from woof.go_cli import _add_go_arguments, _go_wrfinput, GoRefusal
        parser = argparse.ArgumentParser()
        _add_go_arguments(parser)
        for flags in (("--recipe", "surface-state"), ("--smoke",),
                      ("--hydro", "on"), ("--history-interval", "900")):
            with self.subTest(flags=flags):
                args = parser.parse_args(["--wrfinput", "/absent/native-input", *flags])
                # Admission must fail before trying to read that directory.
                with self.assertRaisesRegex(GoRefusal, "namelist.input"):
                    _go_wrfinput(args)


if __name__ == "__main__":
    unittest.main()
