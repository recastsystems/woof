GD_DEVICE void gd_column_one(GDTable *table,float *f,int *ints,GDScratch *work,int longwave=1,int shortwave=1) {
 GDClock clock={0.0f,0.0f,0.0f,0.0f};
 ints[5]=longwave;ints[6]=longwave?0:shortwave;
 etara_(f+0,f+1,f+83,f+165,f+247,f+329,f+411,f+412,f+494,f+576,f+658,f+740,f+822,f+904,f+986,f+1068,f+1150,f+1151,f+1152,f+1153,f+1154,f+1155,f+1156,f+1157,f+1158,f+1159,f+1160,f+1161,f+1162,f+1163,f+1164,f+1165,f+1166,f+1167,f+1168,f+1169,ints+0,ints+1,ints+2,f+1170,f+1171,ints+3,ints+4,ints+5,ints+6,f+1172,f+1173,f+1174,f+1175,ints+7,f+1176,ints+8,ints+9,ints+10,ints+11,ints+12,ints+13,ints+14,ints+15,ints+16,ints+17,ints+18,ints+19,ints+20,ints+21,ints+22,ints+23,ints+24,ints+25,ints+26,table,&clock,work);
 if(!(longwave&&shortwave))return;
 ints[6]=1;
 etara_(f+0,f+1,f+83,f+165,f+247,f+329,f+411,f+412,f+494,f+576,f+658,f+740,f+822,f+904,f+986,f+1068,f+1150,f+1151,f+1152,f+1153,f+1154,f+1155,f+1156,f+1157,f+1158,f+1159,f+1160,f+1161,f+1162,f+1163,f+1164,f+1165,f+1166,f+1167,f+1168,f+1169,ints+0,ints+1,ints+2,f+1170,f+1171,ints+3,ints+4,ints+5,ints+6,f+1172,f+1173,f+1174,f+1175,ints+7,f+1176,ints+8,ints+9,ints+10,ints+11,ints+12,ints+13,ints+14,ints+15,ints+16,ints+17,ints+18,ints+19,ints+20,ints+21,ints+22,ints+23,ints+24,ints+25,ints+26,table,&clock,work);
}
#ifndef GD_HOST_DIAGNOSTIC
extern "C" __global__ void gd_radiation_columns(GDTable *table,float *floats,int *integers,GDScratch *work,int ncol) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<ncol)gd_column_one(table,floats+c*1177,integers+c*27,work+c);
}

extern "C" __global__ void gd_radiation_columns_pooled(GDTable *table,float *floats,int *integers,GDScratch *work,int ncol,int slots,int longwave,int shortwave) {
 int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot>=slots)return;
 for(int c=slot;c<ncol;c+=slots)
  gd_column_one(table,floats+c*1177,integers+c*27,work+slot,longwave,shortwave);
}
extern "C" __global__ void gd_radiation_call(GDTable *table,float *floats,int *integers,GDScratch *scratch,GDClock *clocks,int ncol) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=ncol)return;
 float *f=floats+c*1177;int *ints=integers+c*27;
 GDScratch *work=scratch+c;GDClock &clock=clocks[c];
 etara_(f+0,f+1,f+83,f+165,f+247,f+329,f+411,f+412,f+494,f+576,f+658,f+740,f+822,f+904,f+986,f+1068,f+1150,f+1151,f+1152,f+1153,f+1154,f+1155,f+1156,f+1157,f+1158,f+1159,f+1160,f+1161,f+1162,f+1163,f+1164,f+1165,f+1166,f+1167,f+1168,f+1169,ints+0,ints+1,ints+2,f+1170,f+1171,ints+3,ints+4,ints+5,ints+6,f+1172,f+1173,f+1174,f+1175,ints+7,f+1176,ints+8,ints+9,ints+10,ints+11,ints+12,ints+13,ints+14,ints+15,ints+16,ints+17,ints+18,ints+19,ints+20,ints+21,ints+22,ints+23,ints+24,ints+25,ints+26,table,&clock,work);
}
#endif
