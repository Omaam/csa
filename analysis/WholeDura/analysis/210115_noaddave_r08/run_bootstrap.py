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
                        set_verbose=SETVERBOSE)
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
                               max_workers=CORE, set_verbose=SETVERBOSE)
            filename = _get_savename('X', 'XY')
            np.savetxt(filename, X)

    if ISTCS:
        # search X files in XY and convert to index
        files_X = sorted(glob.glob('XY/X_*'))
        files_X_num = [int(re.split('[_.]', s)[1]) for s in files_X]

        # make lightcurve which corresponds to indcies of X files
        for i in tqdm(files_X_num):

            # get X file
            path_to_fileX = os.path.join(files_X[i])
            X = np.loadtxt(path_to_fileX)

            # ratio threshold
            X = query_forX(X, FREQINFO, 'ratio', [R_THRESHOLD, 1])

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
                                         TPERSEG, TOVERLAP, max_workers=CORE,
                                         set_verbose=SETVERBOSE, add_ave=False,
                                         basewidth=BASEWIDTH_TRIANG)
            fname_data1 = 'data1_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2 = 'data2_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_rec, fname_data1, 'XY')
            _save_data(data2_rec, fname_data2, 'XY')

            # XPS: reconstruct and save
            data1_xps, data2_xps = istcs(X_xps, data1, data2, FREQINFO,
                                         TPERSEG, TOVERLAP, max_workers=CORE,
                                         set_verbose=SETVERBOSE, add_ave=False,
                                         basewidth=BASEWIDTH_TRIANG)
            fname_data1_xps = 'data1_xps_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_xps = 'data2_xps_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_xps, fname_data1_xps, 'XY')
            _save_data(data2_xps, fname_data2_xps, 'XY')

            # OPS: reconstract and save
            data1_ops, data2_ops = istcs(X_ops, data1, data2, FREQINFO,
                                         TPERSEG, TOVERLAP, max_workers=CORE,
                                         set_verbose=SETVERBOSE, add_ave=False,
                                         basewidth=BASEWIDTH_TRIANG)
            fname_data1_ops = 'data1_ops_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_ops = 'data2_ops_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_ops, fname_data1_ops, 'XY')
            _save_data(data2_ops, fname_data2_ops, 'XY')

            # XPS + OPS: reconstract and save
            data1_xops, data2_xops = istcs(X_xops, data1, data2, FREQINFO,
                                           TPERSEG, TOVERLAP, max_workers=CORE,
                                           set_verbose=SETVERBOSE,
                                           add_ave=False,
                                           basewidth=BASEWIDTH_TRIANG)
            fname_data1_xops = 'data1_xops_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_xops = 'data2_xops_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_xops, fname_data1_xops, 'XY')
            _save_data(data2_xops, fname_data2_xops, 'XY')

            # Residual
            data1_res = data1.copy()
            data2_res = data2.copy()
            data1_res[:, 1] = data1[:, 1] - data1_xops[:, 1]
            data2_res[:, 1] = data2[:, 1] - data2_xops[:, 1]
            fname_data1_res = 'data1_res_{}.dat'.format(str(i).rjust(3, '0'))
            fname_data2_res = 'data2_res_{}.dat'.format(str(i).rjust(3, '0'))
            _save_data(data1_res, fname_data1_res, 'XY')
            _save_data(data2_res, fname_data2_res, 'XY')

    if CCF:
        # ALL: get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_???.dat'))
        files_data2 = sorted(glob.glob('XY/data2_???.dat'))
        Y1_all = _collect_flux(files_data1)
        Y2_all = _collect_flux(files_data2)
        logger.debug(f'Y1_all: {Y1_all.shape}')
        logger.debug(f'Y2_all: {Y2_all.shape}')
        lags, q_low_all, q_med_all, q_hig_all = ccfbootstrap(
            Y1_all, Y2_all, maxlags=10, fs=Fs)

        # XPS: get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_xps_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_xps_*.dat'))
        Y1_xps = _collect_flux(files_data1)
        Y2_xps = _collect_flux(files_data2)
        logger.debug(f'Y1_xps: {Y1_xps.shape}')
        logger.debug(f'Y2_xps: {Y2_xps.shape}')
        lags, q_low_xps, q_med_xps, q_hig_xps = ccfbootstrap(
            Y1_xps, Y2_xps, maxlags=10, fs=Fs)

        # OPS: get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_ops_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_ops_*.dat'))
        Y1_ops = _collect_flux(files_data1)
        Y2_ops = _collect_flux(files_data2)
        logger.debug(f'Y1_ops: {Y1_ops.shape}')
        logger.debug(f'Y2_ops: {Y2_ops.shape}')
        lags, q_low_ops, q_med_ops, q_hig_ops = ccfbootstrap(
            Y1_ops, Y2_ops, maxlags=10, fs=Fs)

        # XPS + OPS: get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_xops_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_xops_*.dat'))
        Y1_xops = _collect_flux(files_data1)
        Y2_xops = _collect_flux(files_data2)
        logger.debug(f'Y1_xops: {Y1_xops.shape}')
        logger.debug(f'Y2_xops: {Y2_xops.shape}')
        lags, q_low_xops, q_med_xops, q_hig_xops = ccfbootstrap(
            Y1_xops, Y2_xops, maxlags=10, fs=Fs)

        # Residual: get lightcurves, collect flux, and get quantile
        files_data1 = sorted(glob.glob('XY/data1_res_*.dat'))
        files_data2 = sorted(glob.glob('XY/data2_res_*.dat'))
        Y1_res = _collect_flux(files_data1)
        Y2_res = _collect_flux(files_data2)
        logger.debug(f'Y1_res: {Y1_res.shape}')
        logger.debug(f'Y2_res: {Y2_res.shape}')
        lags, q_low_res, q_med_res, q_hig_res = ccfbootstrap(
            Y1_res, Y2_res, maxlags=10, fs=Fs)

        # figure
        plt.rcParams["font.size"] = 13
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams['xtick.direction'] = 'in'  # x axis in
        plt.rcParams['ytick.direction'] = 'in'  # y axis in
        fig, ax = plt.subplots(4, figsize=(5, 9), sharex=True)

        # observed (FIRFIR) and XPS + OPS
        lags, corr = ccf(zscore(data1[:, 1]), zscore(data2[:, 1]),
                         fs=Fs, maxlags=10)
        ax[0].plot(lags, corr, color='k')
        ax[0].text(0.97, 0.95, '(a)', ha='right', va='top', fontsize=17,
                   transform=ax[0].transAxes)
        ax[0].axhline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[0].axvline(0, color='grey', linewidth=0.5, zorder=-1)

        # XPS and OPS
        ax[1].fill_between(lags, q_hig_xps, q_low_xps, alpha=.5)
        ax[1].fill_between(lags, q_hig_ops, q_low_ops, alpha=.5)
        ax[1].plot(lags, q_med_ops, lags, q_med_xps)
        ax[1].axhline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[1].axvline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[1].text(0.97, 0.95, '(b)', ha='right', va='top', fontsize=17,
                   transform=ax[1].transAxes)

        # XOPS
        ax[2].plot(lags, q_med_xops, color='r')
        ax[2].fill_between(lags, q_hig_xops, q_low_xops,
                           alpha=.5, color='tab:red')
        ax[2].axhline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[2].axvline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[2].text(0.97, 0.95, '(c)', ha='right', va='top', fontsize=17,
                   transform=ax[2].transAxes)

        # residual
        ax[3].plot(lags, q_med_res, color='tab:cyan')
        ax[3].axhline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[3].axvline(0, color='grey', linewidth=0.5, zorder=-1)
        ax[3].text(0.97, 0.95, '(d)', ha='right', va='top', fontsize=17,
                   transform=ax[3].transAxes)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False,
                        left=False, right=False)
        fig.text(0.03, 0.5, 'Correlation coefficient',
                 ha='center', va='center', rotation='vertical')
        # plt.ylabel('Correlation coefficient')
        plt.xlabel('Lag (s)')
        plt.subplots_adjust(left=0.13, right=0.95, bottom=0.05,
                            top=0.98, hspace=0)

        # arrange and show
        plt.savefig('fig/ccf_compare.png', dpi=300)
        # plt.show()


if __name__ == '__main__':
    main()