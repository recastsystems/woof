/* L5 referee witness only: x86 MXCSR flush-to-zero for the native YSU calls (shows the residue is subnormal flush). */
#include <xmmintrin.h>
unsigned int l5_csr_get(void){ return _mm_getcsr(); }
void l5_csr_set(unsigned int v){ _mm_setcsr(v); }
