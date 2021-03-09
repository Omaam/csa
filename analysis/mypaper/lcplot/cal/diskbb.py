import numpy as np
import pandas as pd
from scipy import integrate


h = 1e-34 # J s
nu = 1e14 # Hz 
c = 3 * 10**10 # cm/s
k = 1e-23 # J K^{-1}
M_s = 1e26 # kg
R_s = 1e10 # cm

def diskbb(r, T, R_in):
    i = 10
    rad = np.deg2rad(i)
    coef = np.cos(rad) * 4 * np.pi * h * nu **3 / c **2
    T_r = T * (r/R_in)**(-3/4)
    integrand = r / (np.exp((h * nu) / (k * T_r)) - 1)
    return coef * integrand


if __name__ =='__main__':
    
    T = 1e4 # (K) optical
    R_in = 1e3 * R_s # (cm)
    distance = 8 * 1e18 # (cm)
    luminosity, _ = integrate.quad(diskbb, R_in, np.inf, args=(T, R_in))
    flux = luminosity / (4 * np.pi * distance**2)
    countrate = flux / (k * T)
    print('luminosity: {}'.format(luminosity))
    print('flux: {}'.format(flux))
    print('count rate: {}'.format(countrate))
