/* Oracle links the pristine fork random_id.c; no replacement random algorithm. */
#include <stdio.h>
#include <stdlib.h>
void gen_random(int, char *);
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    srand((unsigned int)strtoul(argv[1], NULL, 10));
    for (unsigned long i=0; i<=strtoul(argv[2], NULL, 10); ++i) {
        char id[11] = {0}; gen_random(10, id);
        if (i==strtoul(argv[2], NULL, 10)) puts(id);
    }
    return 0;
}
