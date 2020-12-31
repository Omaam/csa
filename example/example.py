import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from csa import stcs, istcs

def main():
    infile1 = 'xdata.dat'
    infile2 = 'odata.dat'
    freqinfo = [0, 0.5, 10000]
    lambdainfo = [1e-1, 1e2, 20]
    nsegment = 1000
    noverlap = 500

    # cvdata = cv(infile1, infile2, freqinfo, lambdainfo, 5)
    # plt.errorbar(cvdata[:,0], cvdata[:,1], cvdata[:,2], fmt='o')
    # plt.xscale('log')
    # plt.xlabel(r'$\lambda$')
    # plt.ylabel('MSE')
    # plt.show()

    xdata = np.loadtxt(infile1)
    odata = np.loadtxt(infile2)
    # f, t, X = stcs(infile1, infile2, freqinfo, 1e2, nsegment, noverlap)
    X = np.loadtxt('./x_stcs.dat')
    xrec, orec = istcs(X, freqinfo, xdata[:,0], odata[:,0],
                          nsegment, noverlap)
    # x_data = stcs('./gx339_x_fir_original.dat', './gx339_o_fir_original.dat', [0,10,2000], 20, 50, 1)

if __name__ == '__main__':
    main()
