import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from csa.tools import phist

logger = logging.getLogger(__name__)


def main():

    X = np.loadtxt('XY/X_000.dat')
    freqinfo = [0, 10, 2000]
    lagmap_corr_1, bin_edges_corr_1 = phist(X, freqinfo, lagrange=[-1, 1],
                                            bins=40)
    lagmap_corr_5, bin_edges_corr_5 = phist(X, freqinfo, lagrange=[-5, 5],
                                            bins=40)
    lagmap_anti_1, bin_edges_anti_1 = phist(X, freqinfo, lagrange=[-1, 1],
                                            bins=40, anti=True)
    lagmap_anti_5, bin_edges_anti_5 = phist(X, freqinfo, lagrange=[-5, 5],
                                            bins=40, anti=True)
    print('finish phist calcuration')

    # mask center region
    time = np.arange(lagmap_anti_1.shape[1]+1)
    lagcenter = int(len(bin_edges_corr_1)/2)
    lagmap_corr_5[lagcenter-3:lagcenter+3, :] = 0
    lagmap_anti_5[lagcenter-3:lagcenter+3, :] = 0
    print('masked lags are\n'
          f'{bin_edges_anti_5[lagcenter-3:lagcenter+4]}')

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    # corr [-1, 1]
    ax[0, 1].pcolormesh(time, bin_edges_corr_1, lagmap_corr_1, cmap='jet')
    ax[0, 1].set_ylabel(r'$\tau$')
    # xmin, xmax = ax[0, 1].set_xlim()
    ax[0, 1].axhline(0, linewidth=.8, color='grey')

    # anti [-1, 1]
    ax[1, 1].pcolormesh(time, bin_edges_anti_1, lagmap_anti_1, cmap='jet')
    ax[1, 1].set_xlabel('time')
    ax[1, 1].set_ylabel(r'$\tau^{\rm anti}$')
    # xmin, xmax = ax[1, 1].set_xlim()
    ax[1, 1].axhline(0, linewidth=.8, color='grey')
    
    # corr [-5, 5]
    ax[0, 0].pcolormesh(time, bin_edges_corr_5, lagmap_corr_5, cmap='jet')
    # xmin, xmax = ax[0, 1].set_xlim()
    ax[0, 0].axhline(0, linewidth=.8, color='grey')

    # anti [-5, 5]
    heatmap = ax[1, 0].pcolormesh(time, bin_edges_anti_5, lagmap_anti_5,
                                  cmap='jet')
    # xmin, xmax = ax[1, 0].set_xlim()
    ax[1, 0].axhline(0, linewidth=.8, color='grey')

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.09, 0.035, 0.86])
    cb = fig.colorbar(heatmap, cax=cbar_ax, extend='both',
                      label=r'$\log({\rm amplitude})$')
    plt.subplots_adjust(wspace=0.17)

    # plt.tight_layout()
    os.makedirs('log', exist_ok=True)
    plt.savefig('fig/lagmap.png')
    plt.show()


if __name__ == '__main__':
    main()
