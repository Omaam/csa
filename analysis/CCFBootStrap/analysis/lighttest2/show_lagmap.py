import numpy as np
import matplotlib.pyplot as plt

from csa.tools import phist


def main():

    X = np.loadtxt('XY/X_000.dat')
    freqinfo = [0, 10, 2000]
    lagmap_corr, bin_edges_corr = phist(X, freqinfo, lagrange=[-1, 1], bins=30)
    lagmap_anti, bin_edges_anti = phist(X, freqinfo, lagrange=[-5, 5],
                                        bins=30, anti=True)
    time = np.arange(lagmap_anti.shape[1])
    # lagmap[nbins-1 : nbins+1, :] = 0
    fig, ax = plt.subplots(2)
    ax[0].pcolormesh(time, bin_edges_corr[:-1], np.log10(1+lagmap_corr),
                     cmap='jet', shading='gouraud')
    ax[1].pcolormesh(time, bin_edges_anti[:-1], np.log10(1+lagmap_anti),
                     cmap='jet', shading='gouraud')
    # plt.pcolormesh(time, bin_edges[:-1], lagmap, shading='gouraud')
    plt.show()


if __name__ == '__main__':
    main()
