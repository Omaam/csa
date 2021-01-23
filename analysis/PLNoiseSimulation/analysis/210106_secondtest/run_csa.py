import os
import time

import numpy as np
import matplotlib.pyplot as plt

from csa.csa import cs, cv, stcs
from csa.summary_handler import SummaryNew, read_summary, make_summary
from csa.cvresult import show_cvdata

CV = 1

def main():

    # variable setting
    infile1 = '../../simdata/xdata.dat'
    infile2 = '../../simdata/odata.dat'
    freqinfo = [0, 0.5, 200]
    lambdainfo = [1e-2, 1e2, 20]
    lam = 1

    # load data
    data1 = np.loadtxt(infile1)
    data2 = np.loadtxt(infile2)

    # cv
    if CV:
        cvdata = cv(data1, data2, freqinfo, lambdainfo)
        np.savetxt('cvdata.dat', cvdata)
        show_cvdata(cvdata)
        plt.savefig('cvdata.png')
        # plt.show()
    cvdata = np.loadtxt('cvdata.dat')

    freqs, x = cs(data1, data2, freqinfo, lam)
    np.savetxt('x.dat', x)
    sumdata = SummaryNew(x, freqinfo)
    sumdata.plot_omplot(lagrange=[-8, 8])
    plt.savefig('omplot.png')
    sumdata = SummaryNew(x, freqinfo)
    sumdata.plot_omplot(lagrange=[-8, 8], lag_mode='periodic')
    plt.savefig('omplot_peri.png')


if __name__ == "__main__":
    main()
