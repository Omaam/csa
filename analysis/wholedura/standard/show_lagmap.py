import logging

import numpy as np
import matplotlib.pyplot as plt

from csa.tools import phist

logger = logging.getLogger(__name__)


def main():

    X = np.loadtxt('XY/X_000.dat')
    freqinfo = [0, 10, 2000]
    lagmap_corr, bin_edges_corr = phist(X, freqinfo, lagrange=[-1, 1],
                                        bins=40)
    lagmap_anti, bin_edges_anti = phist(X, freqinfo, lagrange=[-5, 5],
                                        bins=40, anti=True)
    print('finish phist calcuration')

    # axis setting
    time = np.arange(lagmap_anti.shape[1]+1)
    lagcenter = int(len(bin_edges_anti)/2)
    lagmap_anti[lagcenter-3:lagcenter+3, :] = 0
    print(f'masked lags are\n \
            {bin_edges_anti[lagcenter-3:lagcenter+3]}')

    # figure: linear
    fig, ax = plt.subplots(2, figsize=(4, 5), sharex=True)
    ax[0].pcolormesh(time, bin_edges_corr, lagmap_corr,
                     cmap='jet')
    ax[0].set_ylabel('lag')
    xmin, xmax = ax[0].set_xlim()
    ax[0].hlines(0, xmin, xmax, linewidth=.8, color='grey')

    ax[1].pcolormesh(time, bin_edges_anti, lagmap_anti,
                     cmap='jet')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('lag (anti)')
    xmin, xmax = ax[1].set_xlim()
    ax[1].hlines(0, xmin, xmax, linewidth=.8, color='grey')

    plt.tight_layout()
    plt.savefig('lagmap.png')
    plt.show()


if __name__ == '__main__':
    main()
