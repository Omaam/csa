import time
import os
import sys
sys.path.append('summary_handler')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from make_matrix import mkmat_cs, mkmat_cs_w
from fista.fista import fista
from window_function import WindowGenerator
from summary_handler.summary_handler import SummaryNew


__all__ = ['cs', 'cv', 'stcs', 'istcs']


# decorators
def stopwatch(func):
    def wrapper(*arg, **kargs):
        start = time.time()
        res = func(*arg, **kargs)
        dura = (time.time() - start)
        print(time.strftime(f'{func.__name__} %H:%M\'%S\"',
                            time.gmtime(dura)))
        return res
    return wrapper


def change_directory(path_to_dir):
    def _change_directory(func):
        def wrapper(*args, **kargs):
            current_dir = os.getcwd()
            if os.path.exists(path_to_dir) is False:
                os.makedirs(path_to_dir)
            os.chdir(path_to_dir)
            results = func(*args, **kargs)
            os.chdir(current_dir)
            return(results)
        return wrapper
    return _change_directory


# functions for csa
def _sub_ave(flux):
    f = np.array(flux)
    f_out = f - f.mean()
    return f_out

def _get_frecvec(freqinfo):
    freq_edge = np.linspace(freqinfo[0],
                            freqinfo[1],
                            int(freqinfo[2] + 1))
    freq_vec = (freq_edge[:-1] + freq_edge[1:]) / 2
    return freq_vec


def _complement_index(ind):
    comp = np.ones(len(ind), dtype=bool)
    comp[ind] = False
    return comp


def _query_lightcurve(data, timerage):
    data_out = data[np.where((timerage[0] <= data[:,0]) & (data[:,0] < timerage[1]))]
    return data_out


def _drop_sample(data, rate, seed=0):
    n_data = data.shape[0]
    n_drop = int(n_data * rate)
    index_drop = np.random.choice(n_data - 1, n_drop, replace=False)
    data_del = np.delete(data, index_drop, 0)
    return data_del


def _segment_time(t_sta, t_end, tperseg, toverlap):
    # set constant
    tstep = tperseg - toverlap
    nstep = (t_end - t_sta - tperseg) // tstep + 2

    # calcurate edge of lefts and rights
    edge_left = t_sta + np.arange(nstep)*tstep
    edge_right = t_sta + tperseg + np.arange(nstep)*tstep

    # concat edges
    segranges = np.array(list(zip(edge_left, edge_right)))
    return segranges


def _search_index(original, condition):
    l = list(original)
    indices = np.array(list(map(lambda a: l.index(a),
                                condition)))
    return indices


# main functions; cs, cv stcs istcs
def cs(infile1, infile2, freqinfo, lam):

    # load infile
    data1 = np.loadtxt(infile1)
    data2 = np.loadtxt(infile2)
    print('data1: {}'.format(data1.shape))
    print('data2: {}'.format(data2.shape))

    return fista(data1, data2, freqinfo, lam)

@stopwatch
def cv(data1, data2, freqinfo, lambdainfo, nfold=5):

    # load infile
    nrow1 = data1.shape[0]
    nrow2 = data2.shape[0]
    print('data1: {}'.format(data1.shape))
    print('data2: {}'.format(data2.shape))

    # use window and subtract average
    data1_win = data1.copy()
    data2_win = data2.copy()
    t_min = np.hstack([data1[:,0], data2[:,0]]).min()
    t_max = np.hstack([data1[:,0], data2[:,0]]).max()
    window = WindowGenerator([t_min, t_max])
    window.hann()
    # data1_win[:,1] = data1[:,1] * window.gene(data1[:,0])
    # data2_win[:,1] = data2[:,1] * window.gene(data2[:,0])
    data1_win[:,1] = _sub_ave(data1[:,1]) * window.gene(data1[:,0])
    data2_win[:,1] = _sub_ave(data2[:,1]) * window.gene(data2[:,0])

    # plt.plot(data1[:,0], data1[:,1],
    #          data1_win[:,0], data1_win[:,1])
    # plt.show()

    # set freq info
    freq_lo = freqinfo[0]
    freq_up = freqinfo[1]
    nfreq   = freqinfo[2]
    delta_freq = (freq_up - freq_lo) / nfreq
    freqdata = _get_frecvec(freqinfo)

    # set lambda info
    lambda_lo = lambdainfo[0]
    lambda_up = lambdainfo[1]
    nlambda   = lambdainfo[2]
    lambda_vec = np.logspace(np.log10(lambda_lo),
                             np.log10(lambda_up),
                             nlambda)
    print('lambda_vec:\n{}'.format(lambda_vec))

    rms_mean_vec = []
    rms_sd_vec = []
    for lam in tqdm(lambda_vec):

        # get nfold index
        rms_vec = []
        kf1 = KFold(nfold, shuffle=True, random_state=1)
        kf2 = KFold(nfold, shuffle=True, random_state=2)
        sp1 = kf1.split(data1_win)
        sp2 = kf2.split(data2_win)
        for (index_tr1, index_te1), (index_tr2, index_te2) in zip(sp1, sp2):
            data1_tr = data1_win[index_tr1]
            data1_te = data1_win[index_te1]
            data2_tr = data2_win[index_tr2]
            data2_te = data2_win[index_te2]

            # calcurate rms
            freq, x_tr = fista(data1_tr, data2_tr, freqinfo, lam)
            A_mat_te = mkmat_cs(data1_te[:,0], data2_te[:,0], freqdata)
            y_te = np.dot(A_mat_te, x_tr)
            data_te = np.hstack([data1_te[:,1], data2_te[:,1]])
            rms = np.sqrt(np.dot((data_te - y_te).T, data_te - y_te))
            rms_vec.append(rms)

        rms_vec = np.array(rms_vec)
        rms_mean_vec.append(rms_vec.mean())
        rms_sd_vec.append(rms_vec.std())
    rms_mean_vec = np.array(rms_mean_vec)
    rms_sd_vec = np.array(rms_sd_vec)
    cvdata = np.vstack([lambda_vec, rms_mean_vec, rms_sd_vec]).T

    return cvdata


@stopwatch
def stcs(data1, data2, freqinfo, lam, tperseg, toverlap,
         window='hann', x_name='X.dat'):

    # calucurate segranges
    t_min = np.hstack([data1[:,0], data2[:,0]]).min()
    t_max = np.hstack([data1[:,0], data2[:,0]]).max()
    segranges = _segment_time(t_min, t_max, tperseg, toverlap)

    # load infile
    df_data1 = pd.DataFrame(data1)
    df_data2 = pd.DataFrame(data2)

    # output condition
    df_stcsinfo = pd.DataFrame({'cols': ['freq_lo', 'freq_up',
                                         'nfreq', 'lambda',
                                         'tperseg', 'toverlap'],
                                'vals': [freqinfo[0], freqinfo[1],
                                         freqinfo[2], lam,
                                         tperseg, toverlap]})
    df_stcsinfo.to_csv('stcsinfo.txt', sep=' ', header=False, index=False)
    print(df_stcsinfo)

    # short time CS
    print('start stcs')
    x_series = []
    for i in trange(segranges.shape[0]):

        # set segrange
        segrange = segranges[i]

        # query time which is in segrange
        data1_seg = _query_lightcurve(data1, segrange)
        data2_seg = _query_lightcurve(data2, segrange)
        data1_seg_win = data1_seg.copy()
        data2_seg_win = data2_seg.copy()

        # use window fuction
        window = WindowGenerator(segrange)
        window.hann()
        data1_seg_win[:,1] = _sub_ave(data1_seg[:,1]) * window.gene(data1_seg[:,0])
        data2_seg_win[:,1] = _sub_ave(data2_seg[:,1]) * window.gene(data2_seg[:,0])
        acf = window.acf
        ecf = window.ecf

        # drop rows
        # rate_drop = 0.1
        # data1_seg_del = _drop_sample(data1_seg_win, rate_drop, seed=11)
        # data2_seg_del = _drop_sample(data2_seg_win, rate_drop, seed=12)

        # estimate
        freq, x_seg = fista(data1_seg_win, data2_seg_win, freqinfo, lam)
        x_series.append(x_seg*ecf)
    print('finish stcs')

    # visualize each range
    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].plot(data1[:,0], data1[:,1])
    # ax[1].fill_betweenx(np.arange(segranges.shape[0]),
    #                     segranges[:,0], segranges[:,1])
    # ax[1].set_ylabel('Index')
    # ax[1].set_xlabel('Time')
    # plt.show()

    # output
    Zxx = np.array(x_series)
    t = np.mean(segranges, axis=1)
    np.savetxt(x_name, Zxx)
    return freq, t, Zxx


@stopwatch
def istcs(X, data1, data2, freqinfo, tperseg, toverlap, **winargs):
    '''
    T: ndarray
        The series of start time of each segment
    '''
    # calucurate segranges
    t_min = np.hstack([data1[:,0], data2[:,0]]).min()
    t_max = np.hstack([data1[:,0], data2[:,0]]).max()
    segranges = _segment_time(t_min, t_max, tperseg, toverlap)

    # prepare ndarray for reconstraction
    data1_rec = data1.copy()
    data2_rec = data2.copy()
    data1_rec[:,1] = np.zeros(data1.shape[0])
    data2_rec[:,1] = np.zeros(data2.shape[0])

    for i in trange(segranges.shape[0]):

        # set segrange
        segrange = segranges[i]

        # query time which is in segrange
        data1_seg = _query_lightcurve(data1, segrange)
        data2_seg = _query_lightcurve(data2, segrange)
        y1_ave = np.mean(data1_seg[:,1])
        y2_ave = np.mean(data2_seg[:,1])

        # search index
        indices_t1 = _search_index(data1[:,0], data1_seg[:,0])
        indices_t2 = _search_index(data2[:,0], data2_seg[:,0])

        # make summary instance
        # (later summary instance -> x instance)
        x = X[i]
        summ = SummaryNew(x, freqinfo)

        # reconstruct
        y1, y2 = summ.pred(data1_seg[:,0], data2_seg[:,0])
        window = WindowGenerator(segrange)
        window.triang(winargs['basewidth'])
        w1 = window.gene(data1_seg[:,0])
        w2 = window.gene(data2_seg[:,0])
        wy1 = w1 * (y1 + np.mean(data1[:,1]))
        wy2 = w2 * (y2 + np.mean(data2[:,1]))

        # substitution; to conserve energy, it is divided by
        # Energy Correction Factor (ECF)
        win_sect = window.sect
        need_sect = (tperseg - toverlap) * 1
        data1_rec[indices_t1, 1] = data1_rec[indices_t1, 1] \
                                 + wy1 * (need_sect/win_sect)
        data2_rec[indices_t2, 1] = data2_rec[indices_t2, 1] \
                                 + wy2 * (need_sect/win_sect)

    return data1_rec, data2_rec


@change_directory('example')
def main():

    # analysis option
    CV = 0
    STCS = 1
    ISTCS = 1

    # constant
    tperseg = 1000
    toverlap = 980
    basewidth_triang = 2*(tperseg - toverlap)

    # load data
    data1 = np.loadtxt('xdata.dat')
    data2 = np.loadtxt('odata.dat')
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    freqinfo = [0, 0.5, 2000]

    # cross-validation
    if CV:
        np.random.seed(2021)
        cv_sta = np.random.randint(toverlap, np.min([n1, n2]) - tperseg)
        cv_end = cv_sta + tperseg
        print(f'time range of cv: [{cv_sta}, {cv_end}]')
        cvdata = cv(data1[cv_sta:cv_end], data2[cv_sta:cv_end], freqinfo,
                    [1e-2, 1e2, 20])
        np.savetxt('cvdata.dat', cvdata)

        # plot cvdata
        plt.plot(lambdas, f(lambdas), linestyle=':')
        plt.errorbar(cvdata[:,0], cvdata[:,1], cvdata[:,2], fmt='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('MSE')
        plt.show()
    cvdata = np.loadtxt('./cvdata.dat')
    f = interpolate.interp1d(cvdata[:,0], cvdata[:,1], kind="cubic")
    lambdas = np.logspace(np.log10(cvdata[0,0]),
                          np.log10(cvdata[-1,0]), 100)
    idx_min = f(lambdas).argmin()
    lam_min = lambdas[idx_min]
    print(f'lam_min = {lam_min}')

    # short-time common signal analysis
    if STCS:
        freqs, t, X = stcs(data1, data2, freqinfo, lam_min,
                           tperseg, toverlap)

    # inverse short-time common signal analysis
    if ISTCS:
        X = np.loadtxt('X.dat')
        data1_rec, data2_rec = istcs(X, data1, data2, freqinfo,
                             tperseg, toverlap,
                             basewidth=basewidth_triang)

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(data2[:,0], data2[:,1],
                   data2_rec[:,0], data2_rec[:,1],)
        ax[0].set_ylabel('Optical flux')
        ax[0].legend(['original', 'istcs'])
        ax[1].plot(data1[:,0], data1[:,1],
                   data1_rec[:,0], data1_rec[:,1],)
        ax[1].set_ylabel('X-ray flux')
        ax[1].set_xlabel('Time')
        # fig.savefig('lc_ecf.png')
        # plt.show()


if __name__ == '__main__':
    main()
