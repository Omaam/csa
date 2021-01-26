import os
import platform
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from fir_filter import fir
from csa.bootstrap import lcbootstrap


def load_data(fname):
    df_ori = pd.read_csv(fname, delim_whitespace=True)[:25000]
    df_ori['t'] = df_ori.MJD * 24 * 3600
    df_ori['t'] = df_ori.t - df_ori.t[0]
    return df_ori


def _get_savename(basename: str, path_to_dir, digit=3, ext='dat'):
    files = os.listdir(path_to_dir)
    for i in range(10**(digit)):
        suffix = str(i).rjust(digit, '0')
        search_file = basename + '_' + suffix + '.' + ext
        if (search_file in files) is False:
            path_to_out = os.path.join(path_to_dir, search_file)
            return path_to_out
            break


def _save_data(data, savename, path_to_dir='.'):
    path_to_data = os.path.join(path_to_dir, savename)
    np.savetxt(path_to_data, data)


def _collect_flux(files):
    F = []
    for f in files:
        data = np.loadtxt(f)
        F.append(data[:, 1])
    return np.array(F)


def _init_analysis(cv=False, stcs=False, istcs=False):

    # make cv and XY dir
    if os.path.exists('cv') is False:
        os.mkdir('cv')
    if os.path.exists('XY') is False:
        os.mkdir('XY')

    # initialize
    if cv:
        files_cv = sorted(glob.glob('cv/cvdata*'))
        for_sure = input('Are you want to remove "cvdata*.dat"? yes/no ')
        if for_sure == 'yes':
            list(map(lambda f: os.remove(f), files_cv))
            msg_cv = 'DELETED'
    else:
        msg_cv = 'NOT DELETED'

    if stcs:
        files_X = sorted(glob.glob('XY/X*'))
        for_sure = input('Are you want to remove "X*.dat"? yes/no ')
        if for_sure == 'yes':
            list(map(lambda f: os.remove(f), files_X))
            msg_stcs = 'DELETED'
    else:
        msg_stcs = 'NOT DELETED'

    if istcs:
        files_data = sorted(glob.glob('XY/data*'))
        for_sure = input('Are you want to remove "data*.dat"? yes/no ')
        if for_sure == 'yes':
            list(map(lambda f: os.remove(f), files_data))
            msg_istcs = 'DELETED'
    else:
        msg_istcs = 'NOT DELETED'

    # output results
    print(f'cvdatas: {msg_cv}')
    print(f'Xs: {msg_stcs}')
    print(f'lightcurves: {msg_istcs}')


def preprocess_for_gx339(path_to_data, ndata=None):
    # read file
    df_ori = load_data(path_to_data)
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

    # analytical info
    DROPRATE = 0
    # SHOW_RANGE = None

    # preprocess
    data1, data2 = preprocess_for_gx339('../../../data/gx339_night1_2.txt')

    # ALL: get lightcurves, collect flux, and get quantile
    files_data1 = sorted(glob.glob('XY/data1_???.dat'))
    files_data2 = sorted(glob.glob('XY/data2_???.dat'))
    Y1_all = _collect_flux(files_data1)
    Y2_all = _collect_flux(files_data2)
    y1_low_all, y1_med_all, y1_hig_all = lcbootstrap(Y1_all, DROPRATE)
    y2_low_all, y2_med_all, y2_hig_all = lcbootstrap(Y2_all, DROPRATE)

    # XPS: get lightcurves, collect flux, and get quantile
    files_data1 = sorted(glob.glob('XY/data1_xps_???.dat'))
    files_data2 = sorted(glob.glob('XY/data2_xps_???.dat'))
    Y1_xps = _collect_flux(files_data1)
    Y2_xps = _collect_flux(files_data2)
    y1_low_xps, y1_med_xps, y1_hig_xps = lcbootstrap(Y1_xps, DROPRATE)
    y2_low_xps, y2_med_xps, y2_hig_xps = lcbootstrap(Y2_xps, DROPRATE)

    # OPS: get lightcurves, collect flux, and get quantile
    files_data1 = sorted(glob.glob('XY/data1_ops_???.dat'))
    files_data2 = sorted(glob.glob('XY/data2_ops_???.dat'))
    Y1_ops = _collect_flux(files_data1)
    Y2_ops = _collect_flux(files_data2)
    y1_low_ops, y1_med_ops, y1_hig_ops = lcbootstrap(Y1_ops, DROPRATE)
    y2_low_ops, y2_med_ops, y2_hig_ops = lcbootstrap(Y2_ops, DROPRATE)

    # XPS + OPS: get lightcurves, collect flux, and get quantile
    files_data1 = sorted(glob.glob('XY/data1_xops_???.dat'))
    files_data2 = sorted(glob.glob('XY/data2_xops_???.dat'))
    Y1_xops = _collect_flux(files_data1)
    Y2_xops = _collect_flux(files_data2)
    y1_low_xops, y1_med_xops, y1_hig_xops = lcbootstrap(Y1_xops, DROPRATE)
    y2_low_xops, y2_med_xops, y2_hig_xops = lcbootstrap(Y2_xops, DROPRATE)

    # query
    files_data1 = sorted(glob.glob('XY/data1_???.dat'))
    files_data2 = sorted(glob.glob('XY/data2_???.dat'))
    data1_rec = np.loadtxt(files_data1[0])
    data2_rec = np.loadtxt(files_data2[0])
    trange = [data1_rec[:, 0].mean()-15, data1_rec[:, 0].mean()+15]
    print('trange: {}'.format(trange))
    i_q = np.where((trange[0] <= data1[:, 0]) & (data1[:, 0] < trange[1]))[0]
    i_q_rec = np.where((trange[0] <= data1_rec[:, 0]) & (data1_rec[:, 0] < trange[1]))[0]

    # figure
    fig, ax = plt.subplots(2, figsize=(7, 5), sharex=True)

    # observed (FIRFIR) and XPS + OPS
    ax[0].plot(data2[i_q, 0], data2[i_q, 1], color='grey', alpha=.5)
    ax[0].plot(data2_rec[i_q_rec, 0], y2_med_xps[i_q_rec], color='tab:orange')
    ax[0].plot(data2_rec[i_q_rec, 0], y2_med_ops[i_q_rec], color='tab:blue')
    ax[0].fill_between(data2_rec[i_q_rec, 0], y2_low_xps[i_q_rec],
                       y2_hig_xps[i_q_rec], color='tab:orange', alpha=.5)
    ax[0].fill_between(data2_rec[i_q_rec, 0], y2_low_ops[i_q_rec],
                       y2_hig_ops[i_q_rec], color='tab:blue', alpha=.5)
    ax[0].set_ylabel('flux')
    ax[0].text(0.97, 0.95, 'Optical', ha='right', va='top', fontsize=17,
               transform=ax[0].transAxes)

    ax[1].plot(data1[i_q, 0], data1[i_q, 1], color='grey', alpha=.5)
    ax[1].plot(data1_rec[i_q_rec, 0], y1_med_xps[i_q_rec], color='tab:orange')
    ax[1].plot(data1_rec[i_q_rec, 0], y1_med_ops[i_q_rec], color='tab:blue')
    ax[1].fill_between(data1_rec[i_q_rec, 0], y1_low_xps[i_q_rec],
                       y1_hig_xps[i_q_rec], color='tab:orange', alpha=.5)
    ax[1].fill_between(data1_rec[i_q_rec, 0], y1_low_ops[i_q_rec],
                       y1_hig_ops[i_q_rec], color='tab:blue', alpha=.5)
    ax[1].legend(['original (filterd)', 'XPS', 'OPS'], loc='best')
    ax[1].set_ylabel('flux')
    ax[1].set_xlabel('time')
    ax[1].text(0.97, 0.95, 'X-ray', ha='right', va='top', fontsize=17,
               transform=ax[1].transAxes)

    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/reclc.png')
    plt.show()


if __name__ == '__main__':
    main()
