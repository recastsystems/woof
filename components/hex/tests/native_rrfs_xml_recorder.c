/* Capture callbacks from the original RRFS XML parser, without emulating it.
 * RRFS MPAS-Model-rrfs/src/framework/xml_stream_parser.c:1047-1949 at 1d6e370e.
 */
#include <stdio.h>
#include <string.h>
void xml_stream_parser(char *,void *,int *,int *);
void mpas_log_write_c(const char *m,const char *type) {
 if(strcmp(type,"MPAS_LOG_ERR")==0 || strcmp(type,"MPAS_LOG_CRIT")==0) fprintf(stderr,"%s\n",m);
}
void stream_mgr_create_stream_c(void *m,const char *n,int *d,const char *f,const char *i,const char *r,const char *t,int *im,int *p,int *cl,int *g,int *io,int *e) {
 printf("create\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\n",n,f,i,*im,*p,*cl,*g,*io);*e=0;
}
void stream_mgr_add_field_c(void *m,const char *n,const char *f,const char *p,int *e) {printf("field\t%s\t%s\t%s\n",n,f,p);*e=0;}
void stream_mgr_add_immutable_stream_fields_c(void *m,const char *n,const char *f,const char *p,int *e) {printf("include\t%s\t%s\t%s\n",n,f,p);*e=0;}
void stream_mgr_add_pool_c(void *m,const char *n,const char *f,const char *p,int *e) {printf("pool\t%s\t%s\t%s\n",n,f,p);*e=0;}
void stream_mgr_add_alarm_c(void *m,const char *n,const char *d,const char *r,const char *i,int *e) {printf("alarm\t%s\t%s\t%s\n",n,d,i);*e=0;}
void stream_mgr_add_pkg_c(void *m,const char *n,const char *p,int *e) {printf("package\t%s\t%s\n",n,p);*e=0;}
void stream_mgr_set_property_c(void *m,const char *n,const char *p,const char *v,int *e) {printf("property\t%s\t%s\t%s\n",n,p,v);*e=0;}
void stream_mgr_add_variable_output_alarm_c(void *m,const char *n,int *e) {printf("variable\t%s\n",n);*e=0;}
int main(int argc,char **argv) { int comm=0,status=0;void *manager=0;if(argc!=2)return 2;xml_stream_parser(argv[1],&manager,&comm,&status);return status; }
