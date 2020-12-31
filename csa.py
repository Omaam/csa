import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import random
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from make_matrix import mkmat_cs, mkmat_cs_w
from fista.fista import fista
from window_function import Window
from summary_handler.summary_handler import SummaryNew

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

def _query_lightcurve(data, limit):
    data_out = data[np.where((limit[0] <= data[:,0]) & (data[:,0] < limit[1]))]
    return data_out

def _drop_sample(data, rate, seed=0):
    n_data = data.shape[0]
    n_drop = int(n_data * rate)
    index_drop = np.random.choice(n_data - 1, n_drop, replace=False)
    data_del = np.delete(data, index_drop, 0)
    return data_del

def cs(infile1, infile2, freqinfo, lam):

    # load infile
    data1 = np.loadtxt(infile1)
    data2 = np.loadtxt(infile2)
    print('data1: {}'.format(data1.shape))
    print('data2: {}'.format(data2.shape))

    return fista(data1, data2, freqinfo, lam)

def cv(infile1, infile2, freqinfo, lambdainfo, nfold):

    # load infile
    data1 = np.loadtxt(infile1)
    nrow1 = data1.shape[0]
    data2 = np.loadtxt(infile2)
    nrow2 = data2.shape[0]
    print('data1: {}'.format(data1.shape))
    print('data2: {}'.format(data2.shape))

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
        sp1 = kf1.split(data1)
        sp2 = kf2.split(data2)
        for (index_tr1, index_te1), (index_tr2, index_te2) in zip(sp1, sp2):
            # print('index_tr1: {}'.format(index_tr1))
            # print('index_te1: {}'.format(index_te1))
            # print('index_tr2: {}'.format(index_tr2))
            # print('index_te2: {}'.format(index_te2))
            data1_tr = data1[index_tr1]
            data1_te = data1[index_te1]
            data2_tr = data2[index_tr2]
            data2_te = data2[index_te2]

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


def stcs(infile1, infile2, freqinfo, lam, t_perseg, t_overlap,
         window='hann', x_name='x_stcs.dat'):

    # load infile
    data1 = np.loadtxt(infile1)
    data2 = np.loadtxt(infile2)
    df_data1 = pd.DataFrame(data1)
    df_data2 = pd.DataFrame(data2)

    # condition
    cols = ['freq_lo', 'freq_up', 'nfreq', 'lambda', 't_perseg', 't_overlap']
    cols_val = [freqinfo[0], freqinfo[1], freqinfo[2], lam, t_perseg, t_overlap]
    df_stcsinfo = pd.DataFrame({'cols': cols,
                                    'vals': cols_val})
    df_stcsinfo.to_csv('stcsinfo.txt', sep=' ', header=False, index=False)
    print(df_stcsinfo)

    # short time CS
    print('start stcs')
    t_st = data1[0,0] if data1[0,0] >= data2[0,0] else data2[0,0]
    t_en = t_st + t_perseg
    t_en_limit = data1[-1, 0] if data1[-1, 0] <= data2[-1, 0] else data2[-1, 0]
    iteration = int((t_en_limit - t_en) / t_overlap)

    x_series = []
    t_series = []
    for i in trange(iteration):
        # set time range
        t_st_q = t_st + i * t_overlap
        t_en_q = t_en + i * t_overlap

        # divide
        data1_seg = _query_lightcurve(data1, [t_st_q, t_en_q])
        data2_seg = _query_lightcurve(data1, [t_st_q, t_en_q])
        data1_seg[:,1] = data1_seg[:,1] \
                       * signal.get_window(window, data1_seg.shape[0])
        data2_seg[:,1] = data2_seg[:,1] \
                       * signal.get_window(window, data2_seg.shape[0])

        # drop rows
        rate_drop = 0.1
        data1_seg_del = _drop_sample(data1_seg, rate_drop, seed=11)
        data2_seg_del = _drop_sample(data2_seg, rate_drop, seed=12)

        # estimate
        freq, x_seg = fista(data1_seg, data2_seg, freqinfo, lam)
        x_series.append(x_seg)
        t_series.append((t_st_q + t_en_q) * 0.5)

    print('finish stcs')
    Zxx = np.array(x_series)
    t_data = np.array(t_series)
    np.savetxt(x_name, Zxx)
    return freq, t_data, Zxx


def istcs(X, freqinfo, t1, t2, tsegment, toverlap):
    '''
    T: ndarray
        The series of start time of each segment
    '''
    x = X[0]
    summ = SummaryNew(x, freqinfo)
    y1, y2 = summ.pred(t1)
    print(y1.shape)
    return y1, y2

if __name__ == '__main__':
    infile1 = 'lc1.dat'
    infile2 = 'lc2.dat'
    freqinfo = [0, 0.5, 200]
    lambdainfo = [1e-1, 1e2, 20]
    cvdata = cv(infile1, infile2, freqinfo, lambdainfo, 5)

    plt.errorbar(cvdata[:,0], cvdata[:,1], cvdata[:,2], fmt='o')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.show()

    x_data = stcs('./lc1.dat', './lc2.dat', [0, 0.5, 20], 1e-1, 10, 1)
    np.savetxt('x_sample.dat', x_data)
    # x_data = stcs('./gx339_x_fir_original.dat', './gx339_o_fir_original.dat',
    #               [0,10,2000], 20, 50, 1)
    print(x_data.shape)
