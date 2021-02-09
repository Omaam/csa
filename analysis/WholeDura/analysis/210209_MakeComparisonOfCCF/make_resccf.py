import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from csa.bootstrap import ccf
from fir_filter import fir


def preprocess_for_gx339(path_to_data, ndata=25000):
    # read file
    df_ori = pd.read_csv(path_to_data, delim_whitespace=True)[:25000]
    df_ori['t'] = df_ori.MJD * 24 * 3600
    df_ori['t'] = df_ori.t - df_ori.t[0]
    df_ori = df_ori.iloc[:ndata]

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

    # original data
    lcdata1, lcdata2 = preprocess_for_gx339('../../data/gx339_night1_2.txt')
    lags, corr = ccf(lcdata1[:, 1], lcdata2[:, 1], fs=20, maxlags=10)

    # r = 0.0
    lcdata1 = np.loadtxt('lcdata/data1_res_00.dat')
    lcdata2 = np.loadtxt('lcdata/data2_res_00.dat')
    lags, corr_res_00 = ccf(lcdata1[:, 1], lcdata2[:, 1], fs=20, maxlags=10)

    # r = 0.7
    lcdata1 = np.loadtxt('lcdata/data1_res_07.dat')
    lcdata2 = np.loadtxt('lcdata/data2_res_07.dat')
    lags, corr_res_07 = ccf(lcdata1[:, 1], lcdata2[:, 1], fs=20, maxlags=10)

    # r = 0.8
    lcdata1 = np.loadtxt('lcdata/data1_res_08.dat')
    lcdata2 = np.loadtxt('lcdata/data2_res_08.dat')
    lags, corr_res_08 = ccf(lcdata1[:, 1], lcdata2[:, 1], fs=20, maxlags=10)

    # r = 0.9
    lcdata1 = np.loadtxt('lcdata/data1_res_09.dat')
    lcdata2 = np.loadtxt('lcdata/data2_res_09.dat')
    lags, corr_res_09 = ccf(lcdata1[:, 1], lcdata2[:, 1], fs=20, maxlags=10)

    # figure
    plt.rcParams["font.size"] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    ax[0].plot(lags, corr, color='k')
    # ax[0].legend(['original (filterd)', 'residual'])
    # ax[0].set_ylabel('r')
    ax[0].axhline(0, color='grey', linewidth=0.5, zorder=-1)
    ax[0].axvline(0, color='grey', linewidth=0.5, zorder=-1)
    ax[0].text(0.97, 0.95, '(a)', ha='right', va='top',
               fontsize=20, color='k', transform=ax[0].transAxes)

    # r = 0.7, 0.8, 0.9
    ax[1].plot(lags, corr_res_00, color='grey')
    # ax[1].set_ylabel('r')
    ax[1].axhline(0, color='grey', linewidth=0.5, zorder=-1)
    ax[1].axvline(0, color='grey', linewidth=0.5, zorder=-1)
    ax[1].text(0.97, 0.95, '(b)', ha='right', va='top',
               fontsize=20, color='k', transform=ax[1].transAxes)

    # r = 0.7, 0.8, 0.9
    ax[2].plot(lags, corr_res_07, alpha=.7)
    ax[2].plot(lags, corr_res_08, alpha=.7)
    ax[2].plot(lags, corr_res_09, alpha=.7)
    # ax[2].legend([r'r=0.7', r'r=0.8', r'r=0.9'])
    # ax[2].set_ylabel('r')
    # ax[2].set_xlabel('lag')
    ax[2].axhline(0, color='grey', linewidth=0.5, zorder=-1)
    ax[2].axvline(0, color='grey', linewidth=0.5, zorder=-1)
    ax[2].text(0.97, 0.95, '(c)', ha='right', va='top',
               fontsize=20, color='k', transform=ax[2].transAxes)

    # make width space zero
    plt.subplots_adjust(wspace=0)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    fig.text(0.03, 0.55, 'Correlation coefficient',
             ha='center', va='center', rotation='vertical')
    # plt.ylabel('Correlation coefficient')
    plt.xlabel('Lag (s)')
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.98, hspace=0)

    # arrange and save
    # plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/comparison_resccf.png', dpi=300)
    plt.savefig('fig/comparison_resccf.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
