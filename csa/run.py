from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
ncpu = os.cpu_count()
MAX_WORKERS = ncpu
print(f'number of cpu: {ncpu}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from csa.make_matrix import mkmat_cs, mkmat_cs_w
from csa.fista import fista
from csa.window_function import WindowGenerator
from csa.summary_handler import SummaryNew
from csa.deco import stopwatch, change_directory
import csa.xhandler as xhan
from csa.cvresult import show_cvdata, lambda_fromcvdata

__all__ = ['cs', 'cv', 'stcs', 'istcs']


# analysis option
CV = 0
STCS = 1
ISTCS = 1

FIGSHOW = 1


# functions for csa
def _sub_ave(flux):
    f = np.array(flux)
    f_out = f - f.mean()
    return f_out


def _get_minmax(*args):
    a = np.hstack(args)
    return a.min(), a.max()

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


def _drop_sample(data, rate):
    n_data = data.shape[0]
    n_drop = int(n_data * rate)
    index_drop = np.random.RandomState().choice(
        n_data - 1, n_drop, replace=False)
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
def cs(data1, data2, freqinfo, lam):
    freqs, x = fista(data1, data2, freqinfo, lam)
    return freqs, x


def _cv(data1, data2, freqinfo, lam, nfold=5, droprate=None):

    # drop sample
    if droprate:
        data1 = _drop_sample(data1, droprate)
        data2 = _drop_sample(data2, droprate)

    # freq vector
    freqs = _get_frecvec(freqinfo)

    # get nfold index
    kf1 = KFold(nfold, shuffle=True)#, random_state=1)
    kf2 = KFold(nfold, shuffle=True)#, random_state=2)
    sp1 = kf1.split(data1)
    sp2 = kf2.split(data2)

    # cross-validation
    rms = np.zeros(nfold)
    for i, ((i_tr1, i_te1), (i_tr2, i_te2)) in enumerate(zip(sp1, sp2)):
        # split data into one for train and test
        data1_tr, data1_te = data1[i_tr1], data1[i_te1]
        data2_tr, data2_te = data2[i_tr2], data2[i_te2]

        # calcurate rms
        freq, x_tr = fista(data1_tr, data2_tr, freqinfo, lam)
        A_mat_te = mkmat_cs(data1_te[:,0], data2_te[:,0], freqs)
        y_te = np.dot(A_mat_te, x_tr)
        data_te = np.hstack([data1_te[:,1], data2_te[:,1]])
        rms[i] = np.sqrt(np.dot((data_te - y_te).T, data_te - y_te))
    return rms.mean(), rms.std()


@stopwatch
def cv(data1, data2, freqinfo, lambdainfo, nfold=5, droprate=None):

    # use window and subtract average
    data1_win = data1.copy()
    data2_win = data2.copy()
    t_minmax = _get_minmax(data1[:,0], data2[:,0])
    window = WindowGenerator(t_minmax)
    window.hann()
    data1_win[:,1] = _sub_ave(data1[:,1]) * window.gene(data1[:,0])
    data2_win[:,1] = _sub_ave(data2[:,1]) * window.gene(data2[:,0])


    # set lambda info
    lambdas = np.logspace(np.log10(lambdainfo[0]),
                          np.log10(lambdainfo[1]),
                          lambdainfo[2])
    print('lambdas:\n{}'.format(lambdas))

    # cross-validation with multi process
    cvdata = np.zeros((3, lambdas.shape[0])).T
    cvdata[:,0] = lambdas
    with ProcessPoolExecutor(MAX_WORKERS) as executor:
        futures = tqdm([executor.submit(_cv, lam=lam,
                                        data1=data1_win,
                                        data2=data2_win,
                                        freqinfo=freqinfo,
                                        droprate=droprate)
                       for lam in lambdas])
        for k, future in enumerate(futures):
            cvdata[k,1:] = future.result()

    return cvdata


def _stcs(data1, data2, segrange, freqinfo, lam, droprate=None):
    # print(f'start {segrange}')

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
    # data1_seg_win[:,1] = data1_seg[:,1] * window.gene(data1_seg[:,0])
    # data2_seg_win[:,1] = data2_seg[:,1] * window.gene(data2_seg[:,0])
    # acf = window.acf
    # ecf = window.ecf

    # drop rows
    if droprate:
        data1_seg_win = _drop_sample(data1_seg_win, droprate)
        data2_seg_win = _drop_sample(data2_seg_win, droprate)

    # estimate
    freq, x_seg = fista(data1_seg_win, data2_seg_win, freqinfo, lam)
    return x_seg

@stopwatch
def stcs(data1, data2, freqinfo, lam, tperseg, toverlap,
         window='hann', x_name='X.dat', droprate=None):

    # calucurate segranges
    t_min, t_max = _get_minmax(data1[:,0], data2[:,0])
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

    # short time CS with multithread
    X = np.zeros((freqinfo[2]*4, segranges.shape[0]))
    with ProcessPoolExecutor(MAX_WORKERS) as executor:
        futures = tqdm([executor.submit(_stcs, segrange=segrange,
                                        data1=data1, data2=data2,
                                        freqinfo=freqinfo,
                                        lam=lam, droprate=droprate)
                       for segrange in segranges])
        for i, future in enumerate(futures):
            X[:,i] = future.result()

    # output
    t = np.mean(segranges, axis=1)
    freq = _get_frecvec(freqinfo)
    np.savetxt(x_name, X)
    return freq, t, X

def _istcs(x, segrange, data1, data2, freqinfo, need_sect, **winargs):
    # print(f'start {segrange}')

    # query time which is in segrange
    data1_seg = _query_lightcurve(data1, segrange)
    data2_seg = _query_lightcurve(data2, segrange)
    data1_seg_out = data1_seg.copy()
    data2_seg_out = data2_seg.copy()

    # make summary instance
    # (later summary instance -> x instance)
    summ = SummaryNew(x, freqinfo)
    y1, y2 = summ.pred(data1_seg[:,0], data2_seg[:,0])

    # reconstruct
    window = WindowGenerator(segrange)
    window.triang(winargs['winargs']['basewidth'])
    print(segrange, data1_seg[:,1].mean())
    wy1 = window.gene(data1_seg[:,0])*(y1+ data1_seg[:,1].mean())
    wy2 = window.gene(data2_seg[:,0])*(y2 + data2_seg[:,1].mean())

    # substitution; to conserve energy, it is divided by
    # Energy Correction Factor (ECF)
    win_sect = window.sect
    data1_seg_out[:,1] = wy1 * (need_sect/win_sect)
    data2_seg_out[:,1] = wy2 * (need_sect/win_sect)

    # print(f'finish {segrange}')
    return data1_seg_out, data2_seg_out


@stopwatch
def istcs(X, data1, data2, freqinfo, tperseg, toverlap, **winargs):
    '''
    T: ndarray
        The series of start time of each segment
    '''
    # calucurate segranges
    t_min, t_max = _get_minmax(data1[:,0], data2[:,0])
    segranges = _segment_time(t_min, t_max, tperseg, toverlap)

    # prepare ndarray for reconstraction
    y1_rec = data1.copy()
    y2_rec = data2.copy()
    y1_rec[:,1] = np.zeros(data1.shape[0])
    y2_rec[:,1] = np.zeros(data2.shape[0])

    with ProcessPoolExecutor(MAX_WORKERS) as executor:
        need_sect = (tperseg - toverlap) * 1
        futures = tqdm([executor.submit(_istcs, x=x, segrange=segrange,
                                 data1=data1, data2=data2,
                                 freqinfo=freqinfo, need_sect=need_sect,
                                 winargs=winargs)
                       for segrange, x in zip(segranges, X.T)])
        for i, future in enumerate(futures):
            # get results
            data1_seg_out, data2_seg_out = future.result()

            # search index
            indices_t1 = _search_index(data1[:,0], data1_seg_out[:,0])
            indices_t2 = _search_index(data2[:,0], data2_seg_out[:,0])

            # add results
            y1_rec[indices_t1, 1] = y1_rec[indices_t1, 1] \
                                     + data1_seg_out[:,1]
            y2_rec[indices_t2, 1] = y2_rec[indices_t2, 1] \
                                     + data2_seg_out[:,1]

    return y1_rec, y2_rec


@change_directory('../example')
def main():

    # constant
    tperseg = 1000
    toverlap = 500
    basewidth_triang = 2*(tperseg - toverlap)

    # load data
    data1 = np.loadtxt('xdata.dat')
    data2 = np.loadtxt('odata.dat')
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    freqinfo = [0, 0.5, 2000]

    # cross-validation
    if CV:
        cv_sta = np.random.randint(toverlap, np.min([n1, n2]) - tperseg)
        cv_end = cv_sta + tperseg
        print(f'time range of cv: [{cv_sta}, {cv_end}]')
        cvdata = cv(data1[cv_sta:cv_end], data2[cv_sta:cv_end], freqinfo,
                    [1e-2, 1e3, 20])
        np.savetxt('cvdata.dat', cvdata)

        # plot cvdata
        show_cvdata(cvdata)
        plt.savefig('cvcurve.png')
        if FIGSHOW:
            plt.show()
    cvdata = np.loadtxt('./cvdata.dat')
    lam_min = lambda_fromcvdata(cvdata, mode='min')
    print(f'lam_min = {lam_min:.3f}')

    # short-time common signal analysis
    if STCS:
        freqs, t, X = stcs(data1, data2, freqinfo, lam_min,
                           tperseg, toverlap, droprate=None)
        np.savetxt('time.dat', t)
        np.savetxt('freq.dat', freqs)


    # inverse short-time common signal analysis
    if ISTCS:
        X = np.loadtxt('X.dat')
        print(X.shape)

        # query lag comp. around true lag
        # X_lag = xhan.signiftest(X, freqinfo, testrange=[-10, 10])
        # X_lag = xhan.query_forX(X, freqinfo, 'lag', [3, 5])
        # X_rem = xhan.subtractX(X, X_lag)

        # istcs
        y1_rec, y2_rec = istcs(X, data1, data2, freqinfo,
                               tperseg, toverlap,
                               basewidth=basewidth_triang)
        # data1_lag, data2_lag = istcs(X_lag, data1, data2, freqinfo,
        #                              tperseg, toverlap,
        #                              basewidth=basewidth_triang)
        # data1_rem, data2_rem = istcs(X_rem, data1, data2, freqinfo,
        #                              tperseg, toverlap,
        #                              basewidth=basewidth_triang)

        # figure
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(data2[:,0], data2[:,1],
                   y2_rec[:,0], y2_rec[:,1],)
                   # data2_lag[:,0], data2_lag[:,1],
                   # data2_rem[:,0], data2_rem[:,1],)
        ax[0].set_ylabel('Optical flux')
        ax[0].legend(['original', 'istcs', 'lag', 'rem'])
        ax[1].plot(data1[:,0], data1[:,1],
                   y1_rec[:,0], y1_rec[:,1],)
                   # data1_lag[:,0], data1_lag[:,1],
                   # data1_rem[:,0], data1_rem[:,1],)
        ax[1].set_ylabel('X-ray flux')
        ax[1].set_xlabel('Time')
        fig.savefig('lc_ncf_noavesub.png')
        if FIGSHOW:
            plt.show()


if __name__ == '__main__':
    main()
