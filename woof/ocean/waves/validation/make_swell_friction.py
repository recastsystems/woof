"""Regenerate the equation table, without reading any wave-model implementation.

Grant and Madsen (1982), JGR 87, 469-481, equations 3-7,
doi:10.1029/JC087iC01p00469. Kelvin functions come from SciPy.
Run on CPU: python -m woof.ocean.waves.validation.make_swell_friction.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import ker, kei


def friction(log_ratio):
    ratio = 10.0**log_ratio
    def residual(fw):
        argument = 2*np.sqrt(1/(.4*ratio*np.sqrt(fw/2)))
        return fw-.08/(ker(argument)**2+kei(argument)**2)
    return brentq(residual, 1e-9, 1., xtol=1e-15)


def main():
    logs=np.linspace(np.log10(30.), 30., 1025)
    values=np.array([friction(x) for x in logs])
    midpoint=(logs[1:]+logs[:-1])/2
    exact=np.array([friction(x) for x in midpoint])
    error=float(np.max(np.abs((values[1:]+values[:-1])/2/exact-1)))
    assert error < .001
    text='''// Generated from Grant and Madsen (1982), equations 3-7.
// doi:10.1029/JC087iC01p00469. Regenerate with validation/make_swell_friction.py.
// fw=.08/(ker(2*sqrt(zeta))^2+kei(2*sqrt(zeta))^2),
// zeta=1/(.4*(a/z0)*sqrt(fw/2)); no fitted model coefficients.
__device__ __constant__ float wave_gm_friction[1025]={
'''
    text+=',\n'.join(','.join(f'{v:.9e}f' for v in values[i:i+8]) for i in range(0,1025,8))+'\n};\n'
    text+='''__device__ float wave_oscillatory_friction(float amplitude,float roughness){
 if(amplitude<=0||roughness<=0)return 0;
 float logratio=log10f(amplitude)-log10f(roughness);
 if(logratio<1.47712125472f)return .23f; // Published roughness > excursion limit.
 float position=(logratio-1.47712125472f)*(1024.f/(30.f-1.47712125472f));
 position=fminf(position,1024.f);
 int i=min((int)position,1023);float a=position-i;
 return wave_gm_friction[i]+a*(wave_gm_friction[i+1]-wave_gm_friction[i]);
}
'''
    Path(__file__).parents[1].joinpath('swell_friction.cu').write_text(text)
    print({'points':len(values),'maximum_midpoint_relative_error':error,
           'equation_friction_at_a_over_z0_200000':friction(np.log10(2e5))})


if __name__=='__main__':main()
