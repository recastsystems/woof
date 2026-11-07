"""Build an I/O-only observer for the kept, whole strict native executable.

Exported module addresses and sizes come from the executable's symbol
table, relocated by its actual ELF load address. The preload reads them when UDM opens its entry capture,
after udminit/radar_init. No executable or arithmetic instruction changes.
"""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

binary,source,out=map(Path,sys.argv[1:4]);out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("udm_translate",Path(__file__).with_name("translate.py"))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
decl=mod.radar_declarations(source)
header=subprocess.check_output(["readelf","-h",str(binary)],text=True)
assert "EXEC" in header or "DYN" in header
symbols={}
for line in subprocess.check_output(["nm","-S",str(binary)],text=True).splitlines():
    fields=line.split()
    if len(fields)!=4 or not fields[3].startswith("__module_mp_radar_MOD_"):
        continue
    name=fields[3].removeprefix("__module_mp_radar_MOD_")
    if name not in decl or decl[name].const or fields[2] not in ("B","D","R"):
        continue
    symbols[name]=dict(typ=decl[name].typ,bytes=int(fields[1],16),address=int(fields[0],16))
symbols=dict(sorted(symbols.items()))
(out/"radar-schema.json").write_text(json.dumps(symbols,indent=2)+"\n")
rows="\n".join(f'write(fd,(const void*)(base+{v["address"]}ULL),{v["bytes"]});' for v in symbols.values())
c=r'''#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <link.h>
static int seen;
static uintptr_t base;
static int locate(struct dl_phdr_info *info,size_t size,void *data) {
 if(!info->dlpi_name || !info->dlpi_name[0]){base=info->dlpi_addr;return 1;}return 0;
}
static void observe(const char *path) {
 if(seen || !strstr(path,"udm-driver-") || !strstr(path,".input")) return;
 seen=1;
 dl_iterate_phdr(locate,NULL);
 const char *dir=getenv("MISC_RADAR_STATE_DIR");if(!dir)return;
 char dest[4096];snprintf(dest,sizeof(dest),"%s/radar-state-%ld.bin",dir,(long)getpid());
 int fd=creat(dest,0600);if(fd<0)_exit(96);
// WRITE_STATE
 close(fd);
}
int open64(const char *path,int flags,...) {
 mode_t mode=0;if(flags&O_CREAT){va_list a;va_start(a,flags);mode=va_arg(a,int);va_end(a);}
 int (*next)(const char*,int,...)=dlsym(RTLD_NEXT,"open64");
 observe(path);return next(path,flags,mode);
}
int open(const char *path,int flags,...) {
 mode_t mode=0;if(flags&O_CREAT){va_list a;va_start(a,flags);mode=va_arg(a,int);va_end(a);}
 int (*next)(const char*,int,...)=dlsym(RTLD_NEXT,"open");
 observe(path);return next(path,flags,mode);
}
'''.replace("// WRITE_STATE",rows)
(out/"radar-observer.c").write_text(c)
subprocess.run(["gcc","-shared","-fPIC","-O2","-o",str(out/"radar-observer.so"),str(out/"radar-observer.c"),"-ldl"],check=True)
print(json.dumps(dict(symbols=len(symbols),bytes=sum(v["bytes"] for v in symbols.values()))))
