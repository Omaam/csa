import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def lambda_fromcvdata(cvdata, mode='min'):
    spl = interpolate.interp1d(cvdata[:,0], cvdata[:,1], kind="cubic")
    lambdas = np.logspace(np.log10(cvdata[0,0]),
                          np.log10(cvdata[-1,0]), 100)
    idx_min = spl(lambdas).argmin()

    if mode == 'min':
        return lambdas[idx_min]

    elif mode == 'ose':
        idx_min_data = cvdata[:,1].argmin()
        mse_p_std_min = cvdata[idx_min_data,1:].sum()
        lambdas_aftermin = lambdas[idx_min:].copy()
        idx_ose = np.searchsorted(spl(lambdas[idx_min:]), mse_p_std_min)
        return lambdas[idx_min + idx_ose]


def show_cvdata(cvdata, spline=True, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(cvdata[:,0], cvdata[:,1], cvdata[:,2], fmt='o')
    if spline: # spline interpolation
        lambdas = np.logspace(np.log10(cvdata[0,0]),
                              np.log10(cvdata[-1,0]), 100)
        f = interpolate.interp1d(cvdata[:,0], cvdata[:,1], kind="cubic")
        plt.plot(lambdas, f(lambdas), linestyle=':', color='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('MSE')
    return ax
