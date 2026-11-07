"""Version-matched compiled kernel images for woof, and its verification oracles."""
from pathlib import Path

def data_path():
    return Path(__file__).resolve().parent / "data"


def verify_oracles_path():
    """The native-WRF verification oracles (woof.data_assets.VERIFY_ORACLE_TREES)."""
    return Path(__file__).resolve().parent / "verify_oracles"
