"""Write the current real-native-input TEMF proof receipt."""
from pathlib import Path
import argparse
from tools.temf_wrf461_oracle.replay_native import main as replay_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--capture', type=Path,
        default=Path(__file__).resolve().parents[2]/'tests/fixtures/temf_real_native')
    args = parser.parse_args()
    return replay_main([str(args.capture), '--out', str(args.out)])


if __name__ == '__main__':
    raise SystemExit(main())
