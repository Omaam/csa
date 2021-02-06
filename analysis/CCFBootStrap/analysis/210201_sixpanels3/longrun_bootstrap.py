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

from csa.run import stcs, istcs, cv
from csa.tools import segment_time, query_lightcurve
from csa.cvresult import lambda_fromcvdata
from csa.xhandler import query_forX, subtractX, signiftest, addX
from fir_filter import fir
from csa.bootstrap import ccfbootstrap, ccf
from send_email import send_email


# loggging
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    '%(asctime)s %(name)s %(lineno)d[%(levelname)s][%(funcName)s]%(message)s')

# logger for StreamHandler
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.addHandler(sh)

# logger for FileHandler
os.makedirs('log', exist_ok=True)
fh = logging.FileHandler(f'log/{__file__}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.setLevel(logging.DEBUG)


def get_ncore():
    print(platform.platform())
    if 'Ubuntu' in platform.platform():
        core = 4
    elif 'centos' in platform.platform():
        core = 40
    elif 'macOS' in platform.platform():
        core = 8
    else:
        core = None
    print(f'OS is {platform.system()}',
          f' -> the number of used core is {core}')
    return core

# set core
CORE = get_ncore()


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


def run_bootstrap(data1, data2):

    # initialize analysis
    _init_analysis(cv=0, stcs=0, istcs=0)

    # analysis switch
    CV = 1
    STCS = 1
    ISTCS = 1
    CCF = 0

    # analytical info
    LAMBDAINFO = [1e-2, 1e4, 20]
    FREQINFO = [0, 10, 2000]
    TPERSEG = 50  # seconds
    TOVERLAP = 49  # seconds
    BASEWIDTH_TRIANG = 2*(TPERSEG - TOVERLAP)
    Fs = 20  # samples / second
    DROPRATE = 0.5
    NBOOT = 50
    SETBERBOSE = False

    # main analysis
    if CV:
        # divide data into segment
        segranges = segment_time(data1[0, 0], data1[-1, 0],
                                 TPERSEG, TOVERLAP)
        ncv = 3
        inds_cv = np.random.randint(0, len(segranges), ncv)
        lams = []
        for i in tqdm(inds_cv):
            segrange = segranges[i]
            data1_seg = query_lightcurve(data1, segrange)
            data2_seg = query_lightcurve(data2, segrange)
            cvdata = cv(data1_seg, data2_seg, FREQINFO, LAMBDAINFO,
                        droprate=DROPRATE, max_workers=CORE,
                        set_verbose=SETBERBOSE)
            np.savetxt('cv/cvdata_{}.txt'.format(str(i).rjust(4, '0')),
                       cvdata)
            lam = lambda_fromcvdata(cvdata)
            lams.append(lam)
        lam_min = np.mean(lams)
    else:
        # get lam value from previous cv data
        cvfiles = glob.glob('cv/cvdata*')
        lams = []
        for cvfile in cvfiles:
            cvdata = np.loadtxt(cvfile)
            lam = lambda_fromcvdata(cvdata)
            lams.append(lam)
        lam_min = min(lams)
    print(f'lambda_min: {lam_min}')

    if STCS:
        # start stcs
        for _ in tqdm(range(NBOOT)):
            freqs, t, X = stcs(data1, data2, FREQINFO, lam_min,
                               TPERSEG, TOVERLAP, droprate=DROPRATE,
                               max_workers=CORE, set_verbose=SETBERBOSE)
            filename = _get_savename('X', 'XY')
            np.savetxt(filename, X)

    if ISTCS:
        # search X files in XY and convert to index
        files_X = sorted(glob.glob('XY/X_*'))
        files_X_num = [int(re.split('[_.]', s)[1]) for s in files_X]

        # make lightcurve which corresponds to indcies of X files
        for i in tqdm(files_X_num):
            path_to_fileX = os.path.join(files_X[i])
            X = np.loadtxt(path_to_fileX)

            # separation of XPS and OPS from X
            X_xps = signiftest(X, FREQINFO, [-0.8, 0.8], 0.05, periodic=True)
            X_noxps = subtractX(X, X_xps)
            X_noxps_nolag08 = query_forX(X_noxps, FREQINFO, 'lag', [-0.8, 0.8],
                                         mode='cut')
            X_ops = query_forX(X_noxps_nolag08, FREQINFO, 'lag', [0.8, 2.5],
                               mode='ext', anti=True, periodic=False)
            X_xops = addX(X_xps, X_ops)

            # ALL: reconstruct and save
            data1_rec, data2_rec = istcs(X, data1, data2, FREQINFO,
                                         TPERSEG, TOVERLAP,
                                         max_workers=CORE,
                                         set_verbose=SETBERBOSE,
                                         basewidth=BASEWIDTH_TRIANG)
            fname_data1 = 'data1_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2 = 'data2_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_rec, fname_data1, 'XY')
            _save_data(data2_rec, fname_data2, 'XY')

            # XPS: reconstruct and save
            data1_xps, data2_xps = istcs(X_xps, data1, data2, FREQINFO,
                                         TPERSEG, TOVERLAP,
                                         max_workers=CORE,
                                         set_verbose=SETBERBOSE,
                                         basewidth=BASEWIDTH_TRIANG)
            fname_data1_xps = 'data1_xps_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_xps = 'data2_xps_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_xps, fname_data1_xps, 'XY')
            _save_data(data2_xps, fname_data2_xps, 'XY')

            # OPS: reconstract and save
            data1_ops, data2_ops = istcs(X_ops, data1, data2, FREQINFO,
                                         TPERSEG, TOVERLAP,
                                         max_workers=CORE,
                                         set_verbose=SETBERBOSE,
                                         basewidth=BASEWIDTH_TRIANG)
            fname_data1_ops = 'data1_ops_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_ops = 'data2_ops_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_ops, fname_data1_ops, 'XY')
            _save_data(data2_ops, fname_data2_ops, 'XY')

            # XPS + OPS: reconstract and save
            data1_xops, data2_xops = istcs(X_xops, data1, data2, FREQINFO,
                                           TPERSEG, TOVERLAP,
                                           max_workers=CORE,
                                           set_verbose=SETBERBOSE,
                                           basewidth=BASEWIDTH_TRIANG)
            fname_data1_xops = 'data1_xops_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_xops = 'data2_xops_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_xops, fname_data1_xops, 'XY')
            _save_data(data2_xops, fname_data2_xops, 'XY')

    if CCF:
        # ALL:get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_???.dat'))
        files_data2 = sorted(glob.glob('XY/data2_???.dat'))
        Y1_all = _collect_flux(files_data1)
        Y2_all = _collect_flux(files_data2)
        logger.debug(f'Y1_all: {Y1_all.shape}')
        logger.debug(f'Y2_all: {Y2_all.shape}')
        lags, q_low_all, q_med_all, q_hig_all = ccfbootstrap(
            Y1_all, Y2_all, maxlags=10, fs=Fs, droprate=DROPRATE)

        # XPS:get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_xps_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_xps_*.dat'))
        Y1_xps = _collect_flux(files_data1)
        Y2_xps = _collect_flux(files_data2)
        logger.debug(f'Y1_xps: {Y1_xps.shape}')
        logger.debug(f'Y2_xps: {Y2_xps.shape}')
        lags, q_low_xps, q_med_xps, q_hig_xps = ccfbootstrap(
            Y1_xps, Y2_xps, maxlags=10, fs=Fs, droprate=DROPRATE)

        # OPS:get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_ops_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_ops_*.dat'))
        Y1_ops = _collect_flux(files_data1)
        Y2_ops = _collect_flux(files_data2)
        logger.debug(f'Y1_ops: {Y1_ops.shape}')
        logger.debug(f'Y2_ops: {Y2_ops.shape}')
        lags, q_low_ops, q_med_ops, q_hig_ops = ccfbootstrap(
            Y1_ops, Y2_ops, maxlags=10, fs=Fs, droprate=DROPRATE)

        # XPS + OPS:get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_xops_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_xops_*.dat'))
        Y1_xops = _collect_flux(files_data1)
        Y2_xops = _collect_flux(files_data2)
        logger.debug(f'Y1_xops: {Y1_xops.shape}')
        logger.debug(f'Y2_xops: {Y2_xops.shape}')
        lags, q_low_xops, q_med_xops, q_hig_xops = ccfbootstrap(
            Y1_xops, Y2_xops, maxlags=10, fs=Fs, droprate=DROPRATE)

        # figure
        fig, ax = plt.subplots(2, figsize=(5, 7), sharex=True)

        # observed (FIRFIR) and XPS + OPS
        lags, corr = ccf(zscore(data1[:, 1]), zscore(data2[:, 1]),
                         fs=Fs, maxlags=10)
        ax[0].plot(lags, corr, color='grey')
        ax[0].plot(lags, q_med_xops, color='r')
        ax[0].fill_between(lags, q_hig_xops, q_low_xops, alpha=.5, color='r')
        ax[0].legend(['original (filterd)', 'xops'])
        ax[0].set_ylabel('r')

        # XPS and OPS
        ax[1].fill_between(lags, q_hig_xps, q_low_xps, alpha=.5)
        ax[1].fill_between(lags, q_hig_ops, q_low_ops, alpha=.5)
        ax[1].plot(lags, q_med_xps, lags, q_med_ops)
        ax[1].legend(['xps', 'ops'])
        ax[1].set_xlabel('lag')
        ax[1].set_ylabel('r')

        # figure arrange and save
        plt.tight_layout()
        os.makedirs('fig', exist_ok=True)
        plt.savefig('fig/ccf.png')
        # plt.show()


@send_email(True)
def main():

    # preprocess
    data1, data2 = preprocess_for_gx339(
        '../../data/gx339_night1_2.txt', ndata=24000)

    # divide
    istart = np.arange(0, 23000, 1000)
    iend = istart + 2000
    data1_seg = np.array([data1[istart[i]:iend[i], :] \
                       for i in range(len(istart))])
    data2_seg = np.array([data2[istart[i]:iend[i], :] \
                       for i in range(len(istart))])

    # choice segment to analyze from 1 to n_segment - 1
    np.random.seed(20210123)
    id_cho = np.random.choice(23, 12, replace=False)[6:]
    logger.debug('id_cho: {}'.format(id_cho))

    # extract data
    data1_cho = data1_seg[id_cho]
    data2_cho = data2_seg[id_cho]
    logger.debug('data1_cho: {}'.format(data1_cho.shape))
    logger.debug('data2_cho: {}'.format(data2_cho.shape))

    # start analysis for chosen data
    for d1, d2 in zip(data1_cho, data2_cho):

        # logger
        logger.debug('d1: {}'.format(d1.shape))
        logger.debug('d2: {}'.format(d2.shape))

        # make directory and move
        assert all(d1[:, 0] == d2[:, 0]), 'TimeError'
        name_dir = 'from{}to{}'.format(int(round(d1[0, 0])),
                                       int(round(d1[-1, 0])))
        if os.path.exists(name_dir) is False:
            os.mkdir(name_dir)
        os.chdir(name_dir)

        # run analysis
        run_bootstrap(d1, d2)

        # back directory
        os.chdir('..')


if __name__ == '__main__':
    main()
