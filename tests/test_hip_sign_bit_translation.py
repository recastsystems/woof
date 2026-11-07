"""PTX XOR integer operands must retain signed zero and NaN payload bits."""
import subprocess
from woof.hip_source import _translate_assembly


def test_translated_integer_sign_xors_execute_as_bit_operations(tmp_path):
    original = r'''
#include <cstdint>
int main() {
  unsigned int x=0x3f800000u, y;
  unsigned long long a=0x3ff0000000000000ull,b;
  asm volatile("xor.b32 %0,%1,0x80000000;" : "=r"(y) : "r"(x));
  asm volatile("xor.b64 %0,%1,0x8000000000000000;" : "=l"(b) : "l"(a));
  if (y!=0xbf800000u || b!=0xbff0000000000000ull) return 1;
  for (unsigned int bits: {0u,0x80000000u,0x7fc12345u,0xffc12345u}) {
    asm volatile("xor.b32 %0,%1,0x80000000;" : "=r"(y) : "r"(bits));
    if (y!=(bits^0x80000000u)) return 2;
  }
  return 0;
}
'''.replace('#include <cstdint>', '#include <cstdint>\n#include <initializer_list>')
    source=tmp_path/'sign.cpp';source.write_text(_translate_assembly(original))
    binary=tmp_path/'sign';subprocess.run(['c++','-std=c++17',str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
