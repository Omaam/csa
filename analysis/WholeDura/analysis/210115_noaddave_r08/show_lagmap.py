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
    plt.rcParams["font.size"] = 10
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    bin_edges = [[bin_edges_corr_5, bin_edges_corr_1],
                 [bin_edges_anti_5, bin_edges_anti_1]]
    lagmaps = [[lagmap_corr_5, lagmap_corr_1],
               [lagmap_anti_5, lagmap_anti_1]]
    labels = [['(a)', '(b)'], ['(c)', '(d)']]
    for i in range(2):
        for j in range(2):

            # lagmap
            lagmap = ax[i, j].pcolormesh(time, bin_edges[i][j], lagmaps[i][j],
                                         cmap='jet', vmin=20, vmax=80)
            # color setting
            cmap = plt.get_cmap('jet')
            lagmap.cmap.set_over(color=cmap(256))
            lagmap.cmap.set_under(color='k')
            lagmap.cmap.set_bad(color='k')

            # drew horizontal line at lag of 0 s.
            ax[i, j].axhline(0, linewidth=.8, color='grey')

            # set text
            ax[i, j].text(0.97, 0.95, labels[i][j], ha='right', va='top',
                          fontsize=20, color='w', transform=ax[i, j].transAxes)

    # set_label
    ax[0, 0].set_ylabel(r'$\tau$ (s)')
    ax[1, 0].set_ylabel(r'$\tau^{\rm anti}$ (s)')

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.09, 0.035, 0.86])
    fig.colorbar(lagmap, cax=cbar_ax, extend='both',
                 label='Amplitude')
    plt.subplots_adjust(wspace=0.17)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    # plt.ylabel('Lag (s)')
    plt.xlabel(r'$t\ {\rm (s)}$')
    plt.subplots_adjust(left=0.06, right=0.85, bottom=0.09, top=0.95)

    # plt.tight_layout()
    os.makedirs('log', exist_ok=True)
    plt.savefig('fig/lagmap.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
