import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.ticker as ptick
# import os
# from astropy.time import Time


if __name__ == '__main__':

    plt.rcParams["font.size"] = 19
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in

    # observed
    df_lc = pd.read_csv('./gx339_night1_2.csv')
    df_lc['sec'] = df_lc.MJD * 24 * 60 * 60
    df_lc['sec'] = df_lc.sec - df_lc.sec[0]
    df_X_obs = df_lc[['sec', 'X']]
    df_O_obs = df_lc[['sec', 'O']]
    np.savetxt('gx339_x_original.dat', df_X_obs.values)
    np.savetxt('gx339_o_original.dat', df_O_obs.values)

    # FIR low-path filter
    df_X_fir = pd.read_csv('./gx339_x_fir_original.dat', sep=' ',
                           names=['sec', 'flx'])
    df_O_fir = pd.read_csv('./gx339_o_fir_original.dat', sep=' ',
                           names=['sec', 'flx'])

    fig, ax = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(16, 5))
    ax[1, 0].scatter(df_lc.sec, df_lc.O, c='k', s=5)
    ax[0, 0].scatter(df_lc.sec, df_lc.X, c='k', s=5)
    ax[1, 1].scatter(df_O_fir.sec, df_O_fir.flx, c='k', s=5)
    ax[0, 1].scatter(df_X_fir.sec, df_X_fir.flx, c='k', s=5)
    # ax[0].set_xticks([])
    plt.subplots_adjust(hspace=0, wspace=.02)
    # ax[0].yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    # ax[0].yaxis.offsetText.set_fontsize(10)
    # ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # ax[1].yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
    # ax[1].yaxis.offsetText.set_fontsize(10)
    # ax[1].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # ax[1].set_xlabel(r'$t-t_0\ {\rm (s)}$')
    # ax[0].set_ylabel('Count rate of optical')
    # ax[1].set_ylabel('Count rate of X-ray')
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')

    # setting
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    # plt.ylabel(r'Count rate (${\rm cts\,s^{-1}}$)')
    fig.text(0.03, 0.5, r'Count rate (${\rm s^{-1}}$)',
             ha='center', va='center', rotation='vertical')
    plt.xlabel(r'$t\;{\rm (s)}$')
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.12, top=0.95)
    plt.show()
    fig.savefig('lc_original_whole.png', transparent=True, dpi=300)
    fig.savefig('lc_original_whole.pdf', transparent=True)
