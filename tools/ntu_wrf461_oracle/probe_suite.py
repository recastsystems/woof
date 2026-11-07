"""Run all column batches under a declared compiler-option control."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from woof.core import ntu
import run_suite

def main():
 ap=argparse.ArgumentParser();ap.add_argument('inputs',type=Path);ap.add_argument('outputs',type=Path);ap.add_argument('--option',action='append',default=[]);a=ap.parse_args()
 original=ntu.compile_options;ntu.compile_options=lambda:original()+tuple(a.option)
 sys.argv=['run_suite.py',str(a.inputs),str(a.outputs)];run_suite.main()
if __name__=='__main__':main()
