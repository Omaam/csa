import os
import re
import platform
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from tqdm import tqdm

from csa.run import stcs, istcs, cv, _get_frecvec
from csa.tools import segment_time, query_lightcurve
from csa.cvresult import lambda_fromcvdata
from csa.xhandler import query_forX, subtractX, signiftest, addX, _make_summary 
from fir_filter import fir
from csa.bootstrap import ccfbootstrap, ccf
from send_email import send_email


# loggging
if os.path.exists('log') is False:
    os.mkdir('log')
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    '%(asctime)s %(name)s %(lineno)d[%(levelname)s][%(funcName)s]%(message)s')

# logger for StreamHandler
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.addHandler(sh)

# logger for FileHandler
fh = logging.FileHandler(f'log/{__file__}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.setLevel(logging.DEBUG)


# core check
if platform.system() == 'Darwin':
    CORE = 8
elif platform.system() == 'Linux':
    CORE = 30
print(f'OS is {platform.system()} -> the number of used core is {CORE}')


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


def preprocess_for_gx339(path_to_data, ndata=2000):
    # read file
    df_ori = load_data(path_to_data)
    df_ori = df_ori.iloc[:ndata]

    # FIR filter
    yx_fir = fir(df_ori.X)
    yo_fir = fir(df_ori.O)
    yx_firfir = fir(yx_fir[::-1])[::-1]
    yo_firfir = fir(yo_fir[::-1])[::-1]
    logger.debug('yx_firfir: %s', yx_firfir.shape)
    logger.debug('yo_firfir: %s', yo_firfir.shape)

    # average = 0 and deviation = 1
    yx_fir_zs = zscore(yx_firfir)
    yo_fir_zs = zscore(yo_firfir)

    # set data for stcs
    data1 = np.tile(df_ori.t, (2, 1)).T
    data2 = np.tile(df_ori.t, (2, 1)).T
    data1[:, 1] = yx_fir_zs
    data2[:, 1] = yo_fir_zs

    return data1, data2


@send_email(True)
def main():

    # initialize analysis
    _init_analysis(cv=0, stcs=0, istcs=0)

    # analysis switch
    CV = 0
    STCS = 0
    ISTCS = 0
    CCF = 1

    # analytical info
    LAMBDAINFO = [1e0, 1e3, 20]
    FREQINFO = [0, 10, 2000]
    TPERSEG = 50  # seconds
    TOVERLAP = 49  # seconds
    BASEWIDTH_TRIANG = 2*(TPERSEG - TOVERLAP)
    Fs = 20  # samples / second
    R_THRESHOLD = 0.8
    DROPRATE = None
    NBOOT = 1
    SETVERBOSE = True

    # preprocess
    data1, data2 = preprocess_for_gx339(
        '../../data/gx339_night1_2.txt', ndata=25000)

    # main analysis
    # search X files in XY and convert to index
    files_X = sorted(glob.glob('XY/X_???.dat'))
    files_X_num = [int(re.split('[_.]', s)[1]) for s in files_X]

    # make lightcurve which corresponds to indcies of X files
    for i in tqdm(files_X_num):

        # get X file
        path_to_fileX = os.path.join(files_X[i])
        X = np.loadtxt(path_to_fileX)

        # ratio threshold
        X = query_forX(X, FREQINFO, 'ratio', [R_THRESHOLD, 1])

        # separation of XPS and OPS from X
        # X_xps = signiftest(X, FREQINFO, [-0.8, 0.8], 0.05, periodic=True)
        # np.savetxt('XY/X_xps_000.dat', X_xps)
        X_xps = np.loadtxt('./XY/X_xps_000.dat')
        data1_xps = np.loadtxt('./XY/data1_xps_000.dat')

        # get freq for XPS
        M = []
        for x in X_xps.T:
            masks = np.nan_to_num(x[:2000] / x[:2000])
            M.append(masks)
        M = np.array(M).T
        print(M.shape)

        # freq
        freqs = _get_frecvec(FREQINFO)

        # time
        time = np.arange(0, 1192, 1)

        # plot mask
        plt.pcolormesh(time, freqs, M, cmap='binary')
        plt.xlabel(r'$t$ (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig('fig/spectrogram_xps.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    main()
