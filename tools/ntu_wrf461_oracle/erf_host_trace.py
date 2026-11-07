"""Generate an oracle-host ERF trace without changing the column arithmetic."""
import argparse
from pathlib import Path
HEADER=r'''
static FILE* ntu_erf_file=nullptr;
static float ntu_host_erff(float x){float y=erff(x);unsigned int a=__float_as_uint(x),b=__float_as_uint(y);unsigned long long r[3]={1,a,b};fwrite(r,8,3,ntu_erf_file);return y;}
static double ntu_host_erf(double x){double y=erf(x);unsigned long long r[3]={2,(unsigned long long)__double_as_longlong(x),(unsigned long long)__double_as_longlong(y)};fwrite(r,8,3,ntu_erf_file);return y;}
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('host_main',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();s=a.host_main.read_text();s=s.replace('#define NTU_ERFF(x)   erff(x)',HEADER+'\n#define NTU_ERFF(x)   ntu_host_erff(x)').replace('#define NTU_ERF(x)    erf(x)','#define NTU_ERF(x)    ntu_host_erf(x)').replace('int main(int argc, char** argv) {','int main(int argc, char** argv) {\n ntu_erf_file=fopen(argv[3],"wb");');a.output.write_text(s)
if __name__=='__main__':main()
