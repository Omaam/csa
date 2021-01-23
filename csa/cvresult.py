import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def lambda_fromcvdata(cvdata, mode='min'):

    # get index at minmun 
    idx_min = cvdata[:,1].argmin()

    if mode == 'min':
        return cvdata[idx_min, 0]

    elif mode == 'ose':

        # MSE + error at minmun MSE
        mse_p_std_min = cvdata[idx_min_data, 1:].sum()

        # search the lambda which MSE is close to
        # the MSE + error at the minum.
        idx_ose = np.searchsorted(cvdata[idx_min:], mse_p_std_min)
        return cvdata[idx_min + idx_ose]


def show_cvdata(cvdata, ax=None):

    # make ax if None
    if ax is None:
        fig, ax = plt.subplots()

    # figure
    ax.errorbar(cvdata[:,0], cvdata[:,1], cvdata[:,2], fmt='o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('MSE')
    return ax
