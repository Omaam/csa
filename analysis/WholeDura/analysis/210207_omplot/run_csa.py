import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from csa.run import cs, cv
from csa.summary_handler import SummaryNew
import csa.xhandler as xhan
from csa.cvresult import show_cvdata, lambda_fromcvdata
from fir_filter import fir


def extract_lc(lcdata, trange):
    time = lcdata[:, 0]
    cond = (trange[0] <= time) & (time <= trange[1])
    lcdata_ext = lcdata[cond, :]
    return lcdata_ext


def preprocess_for_gx339(path_to_data, ndata=25000):

    # load file
    df_ori = pd.read_csv(path_to_data, delim_whitespace=True)[:ndata]
    df_ori['t'] = df_ori.MJD * 24 * 3600
    df_ori['t'] = df_ori.t - df_ori.t[0]

    # FIR filter
    yx_fir = fir(df_ori.X)
    yo_fir = fir(df_ori.O)
    yx_firfir = fir(yx_fir[::-1])[::-1]
    yo_firfir = fir(yo_fir[::-1])[::-1]

    # average = 0 and deviation = 1
    yx_fir_zs = zscore(yx_firfir)
    yo_fir_zs = zscore(yo_firfir)

    # set data for stcs
    data1 = np.tile(df_ori.t, (2, 1)).T
    data2 = np.tile(df_ori.t, (2, 1)).T
    data1[:, 1] = yx_fir_zs
    data2[:, 1] = yo_fir_zs

    return data1, data2


def main():

    # for paper
    plt.rcParams["font.size"] = 20
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in

    # cv switch
    CV = 0

    # analytical info
    lambdainfo = [1e-2, 1e3, 20]
    freqinfo = [0, 2, 500]

    # preprocess
    data1, data2 = preprocess_for_gx339(
        '../../data/gx339_night1_2.txt', ndata=25000)

    # extract from 615 to 665
    data1 = extract_lc(data1, [615, 665])
    data2 = extract_lc(data2, [615, 665])

    # make dirs
    os.makedirs('cv', exist_ok=True)
    os.makedirs('fig', exist_ok=True)
    os.makedirs('res', exist_ok=True)

    # cv
    if CV:
        cvdata = cv(data1, data2, freqinfo, lambdainfo)
        np.savetxt('cv/cvdata.dat', cvdata)
        show_cvdata(cvdata)
        plt.savefig('cv/cvdata.png')
        # plt.show()
    cvdata = np.loadtxt('cv/cvdata.dat')
    lam = lambda_fromcvdata(cvdata)

    # CS
    freqs, x = cs(data1, data2, freqinfo, lam)
    np.savetxt('res/x.dat', x)

    # threshold r > 0.8
    x = xhan.query_forX(x, freqinfo, 'ratio', [0.8, 1])

    # show corr between -0.8 and 0.8
    sumdata = SummaryNew(x, freqinfo)
    sumdata.plot_omplot(lagrange=[-0.8, 0.8])
    plt.savefig('fig/omplot.png')

    # get anti sumdata
    sumdata = SummaryNew(x, freqinfo)
    sumdata_anti = sumdata.anti()

    # show anti between -8 and 8
    sumdata_anti.plot_omplot(lagrange=[-5, 5])
    plt.savefig('fig/omplot_anti.png')


if __name__ == "__main__":
    main()
