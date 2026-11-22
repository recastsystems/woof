// TEMPO 9adf9ef module_mp_tempo_ml.F90:28-81,104-203 (numerically unchanged at
// tempo_v3.2.5 c62699f: the v3.2.x edits only move allocations).
// Public TEMPO constants and network; covered by NOTICE-TEMPO.txt.
#pragma once
#include "tempo_column.cuh"
TEMPO_HD double tempo_nc_ml_trans_mean(int i) {
  const double values[7] = {0.000191556196486247, 3.58145042772654e-05,     3.12611085359273e-07, 5.74303078738579e-05, 84191.2092225319,     279.070551773565, 0.123679354084004};
  return values[i];
}
TEMPO_HD double tempo_nc_ml_trans_var(int i) {
  const double values[7] = {5.78143564171777e-08, 3.22834309750552e-08,     6.45745893307455e-11, 4.16625579383794e-08, 215694631.771185,     94.6576255386858, 0.384841247662964};
  return values[i];
}
TEMPO_HD double tempo_nc_ml_w00(int i) {
  const double values[168] = {-2.006957, -0.2812008, -0.339073, 1.596426, 2.395225, 1.76315,     -0.0626798, 1.267002, -0.02234177, -6.522605e-33, 0.4792154, -0.1253034,     -3.217191, -3.092887, -0.0863651, 1.071625, 0.09741028, 0.2255831,     -0.6929023, -0.02693799, -3.432344e-33, -0.8791879, -0.9359049, 1.083484,     -0.07909214, -0.0122418, -0.02815927, 0.1676407, 0.08252326, 0.6697816,     -0.4019359, 0.4687141, 0.001813132, -9.792186e-33, 0.0409322, 0.0113192,     -0.01354596, 0.00307771, -0.4635534, 0.03835761, -0.1015553, 0.7316446,     -0.05791711, -0.0002690362, -7.920147e-33, -0.1216918, -0.3190572, 0.09809405,     -0.16476, -0.03387314, 0.005422261, 0.04043967, 0.03901243, 0.07444729,     0.01954299, 0.06918761, 0.04823543, -8.637957e-33, 0.06371575, -0.09250915,     -1.109653, -1.373999, -0.2412623, -0.04482195, 0.1584691, 0.06353725,     0.0006248798, 0.04593191, -8.878673e-33, -0.4988684, 0.01110262, 0.04623203,     0.006581791, 0.03536217, -0.1890567, -0.08839592, 0.1327181, 0.03478973,     -0.1565902, -0.100401, -0.1179777, -8.879818e-33, -0.1383738, 0.02847495,     -0.005902881, 0.005615512, -0.6308192, -0.02431803, -0.141971, -0.3490018,     -0.9850957, -0.1449479, -8.059166e-33, -0.1186465, -1.165381, 0.069015,     0.003388841, -0.04041302, 0.1638467, 0.1147008, -0.04833491, -0.07755993,     -0.5137688, 0.04546477, 0.04101883, 6.752353e-33, 0.3541977, 0.04880851,     -0.00102834, -0.01280629, -0.1116254, -0.02204754, 0.07100908, 0.2354002,     0.07129629, 0.2489657, 8.080785e-33, -0.03449865, 0.06037927, -0.02023619,     0.7779589, 0.04680278, 0.7492616, 0.6545208, -1.09497, -1.176524,     -0.451585, 0.881124, -0.4551499, -8.708624e-33, 1.006558, -0.04979523,     -0.0006915367, -0.002993054, 0.01654614, -0.07141764, -0.2216591, 0.8637336,     0.8358089, -0.7576646, 8.186339e-33, 0.1161914, 0.7121871, -1.146734,     0.03925339, 0.9975697, -0.06953461, -0.07598846, -0.06418022, -0.01897495,     -0.03612464, -0.06703389, -0.103049, 9.364858e-33, -0.03949085, 1.005938,     0.0001625419, -0.01027657, 0.03823901, 0.3197246, -0.1160718, 0.04449431,     0.009831783, 0.6551342, -8.470687e-33, 0.1374123, -0.01782138, 0.01719949};
  return values[i];
}
TEMPO_HD double tempo_nc_ml_w01(int i) {
  const double values[24] = {-4.045869, 0.558111, -1.567351, -1.64972, 2.748608,     -1.909901, 0.3955558, 1.507247, 0.3599722, -0.0001173223, -0.9398569,     -0.6867028, -61.72853, -63.13766, 1.165811, -0.6848684, 0.1931683,     1.1208, 2.63087, 0.740169, -21.62499, 1.545568, 3.575141, -1.299604};
  return values[i];
}
TEMPO_HD double tempo_nc_ml_b00(int i) {
  const double values[24] = {-0.9842531, 0.3064759, -0.4500185, 1.28336, 1.384105, 0.528031,     0.8453538, 1.579872, 2.245679, -0.008679952, 0.4549862, -0.136581,     -2.576741, -2.483647, -0.2089484, 0.7607977, 1.847745, 0.7316047,     -0.287945, 2.227298, -2.314714, -0.2561245, -0.6993448, -0.1359731};
  return values[i];
}
TEMPO_HD double tempo_nc_ml_b01(int i) {
  const double values[1] = {1.572826};
  return values[i];
}
TEMPO_HD double tempo_cloud_number(double qc, double qr, double qi, double qs,
                                  double pressure, double temperature, double w) {
  const double input[7] = {qc, qr, qi, qs, pressure, temperature, w};
  double transformed[7], activated[24];
  for (int i = 0; i < 7; ++i)
    transformed[i] = (input[i] - tempo_nc_ml_trans_mean(i)) / sqrt(tempo_nc_ml_trans_var(i));
  for (int node = 0; node < 24; ++node) {
    double value = 0.0;
    for (int i = 0; i < 7; ++i) value += tempo_nc_ml_w00(node + 24*i) * transformed[i];
    activated[node] = tmax(value + tempo_nc_ml_b00(node), 0.0);
  }
  double value = 0.0;
  for (int node = 0; node < 24; ++node) value += tempo_nc_ml_w01(node) * activated[node];
  value = tmax(value + tempo_nc_ml_b01(0), 0.0);
  const double p = tmin(9.3010299957, tmax(-6.0, value));
  double bias = 1.0;
  if (p >= 0.0 && p < 3.0) {
    const double p2 = p*p, p3 = p2*p, p4 = p2*p2, p5 = p4*p;
    bias = -0.2704*p5 + 1.838*p4 - 5.127*p3 + 8.547*p2 - 8.439*p + 4.297;
  }
  return bias * tempo_pow(10.0, p);
}

// Native TEMPO main:801-823. Update internal cloud moments for radiation
// only; the prognostic cloud and droplet number have already been committed.
TEMPO_HDN inline void tempo_subgrid_cloud_radii(const TempoParams& P, const TempoCtx& C,
    int nz, TempoWork& W, const double* qc, const double* qr, const double* qi,
    const double* qs, const double* nc, const double* w,
    const float* qc_bl, const float* fraction_bl, int nt, int c0) {
  if (!W.did_micro || !C.re_cloud_flag || !C.re_ice_flag || !C.re_snow_flag) return;
  for (int k = 1; k <= nz; ++k) {
    const long long index = (long long)(k-1)*nt + c0;
    W.xrx[k] = qc[k];
    if (qc[k] <= PS(r1) && qc_bl[index] > 1.0e-9 && fraction_bl[index] > 0.0)
      W.xrx[k] += (double)qc_bl[index] / (double)fraction_bl[index];
    if (W.xrx[k] <= 1.0e-12) W.xrx[k] = 0.0;
    W.xncx[k] = tempo_cloud_number(W.xrx[k], qr[k], qi[k], qs[k], W.pres[k], W.temp[k], w[k]);
    W.qcten[k] = 0.0;
    W.ncten[k] = 0.0;
  }
  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, W.xncx, nullptr,
                        W.rc, W.nc, W.qcten, W.ncten, W.ilamc, W.mvd_c);
  effective_radius(P, C, nz, W.temp, W.l_qc, W.nc, W.ilamc, W.l_qi, W.ilami,
                   W.l_qs, W.rs, W.re_cloud, W.re_ice, W.re_snow);
}
