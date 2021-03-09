import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate


h = 1e-27 # erg s
c = 3 * 10**10 # cm/s
k = 1e-16 # erg K^{-1}
M_s = 1e26 # kg
R_s = 1e10 # cm
inc = 10 # degree

def T(r, T_in):
    t = T_in * (r / R_in) **(-3/4)
    return t

def B(r, T_in):
    coef = 2 * h * nu **3 / c**2
    e = np.exp((h * nu) / (k * T(r, T_in)))
    b = 1 / (e - 1)
    # print('r, B: {}, {}'.format(r, b))
    return coef * b

def L(R_in, R_out, T_in=1e6):
    # print('R_in, R_out: {:e}Rs, {:e}Rs'.format(R_in/R_s, R_out/R_s))
    l = integrate.quad(lambda r: B(r, T_in) * 2 * np.pi * r,
                       R_in,
                       R_out)
    return np.cos(np.deg2rad(inc)) * l[0]


if __name__ =='__main__':
    
    nu = 1e14 # Hz 
    T_opt = 1e4 # (K)
    T_in = 1e6 # (K) = 0.1 keV (UV)
    R_in = 1e3 * R_s
    R_out = 1e6 * R_s
    d = 8 * 1e3 * 1e18 # (cm)

    # calculation
    luminosity = L(R_in, R_out, T_in)
    flux = luminosity / (4 * np.pi * d**2)
    print('f_whole: {:.2e} erg/s/cm2/Hz'.format(flux))
    nuflux = flux * nu
    print('nuF_whole: {:.2e} erg/s/cm2'.format(nuflux))

    # calculation
    luminosity = L(R_in, 1e4*R_s, T_in)
    flux = luminosity / (4 * np.pi * d**2)
    print('f_ops: {:.2e} erg/s/cm2/Hz'.format(flux))
    nuflux = flux * nu
    print('nuF_ops: {:.2e} erg/s/cm2'.format(nuflux))

    # spectrum
    Rin_list = np.logspace(1, 5, 100)
    spec = []
    for R_in in Rin_list:
        spec.append(L(R_in, R_out, T_in) / (2 * np.pi * d**2))
    spec = np.array(spec)
    fig, ax = plt.subplots()
    ax.plot(Rin_list, nu * spec)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\nu F_\nu$')
    ax.set_xlabel(r'R/R_{\rm s}')
    plt.show()

    # spectrum
    nus = np.logspace(9, 17.5, 100)
    # print('nus: {}'.format(nus))
    spec = []
    for nu in nus:
        spec.append(L(R_in, R_out) / (2 * np.pi * d**2))
    fig, ax = plt.subplots()
    ax.plot(nus, nus*spec)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\nu F_\nu$')
    ax.set_xlabel(r'$\nu$ (Hz)')
    plt.show()
