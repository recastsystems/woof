"""Version-matched startup compiler images for woof."""
from pathlib import Path

def data_path():
    return Path(__file__).resolve().parent / "data"
