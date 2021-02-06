import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import zscore

from fir_filter import fir


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


def preprocess_for_gx339(path_to_data, ndata=2000):
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

    # preprocess
    data1, data2 = preprocess_for_gx339(
        '../../data/gx339_night1_2.txt', ndata=2000)

    # XPS + OPS:get lightcurves, collect flux, and get quantile
    files_data1 = sorted(glob.glob('XY/data1_xops_*.dat'))
    files_data2 = sorted(glob.glob('XY/data2_xops_*.dat'))

    for i, (fname_data1_xops, fname_data2_xops) in \
            enumerate(zip(files_data1, files_data2)):

        # prepare 
        data1_res = data1.copy()
        data2_res = data2.copy()

        # get xops lightcurves
        data1_xops = np.loadtxt(fname_data1_xops)
        data2_xops = np.loadtxt(fname_data2_xops)

        # subtract
        data1_res[:, 1] -= data1_xops[:, 1]
        data2_res[:, 1] -= data2_xops[:, 1]

        fname_data1_res = 'data1_res_{}.dat'.format(str(i).rjust(3, '0'))
        fname_data2_res = 'data2_res_{}.dat'.format(str(i).rjust(3, '0'))
        _save_data(data1_res, fname_data1_res, 'XY')
        _save_data(data2_res, fname_data2_res, 'XY')


if __name__ == '__main__':
    main()
