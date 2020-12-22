import numpy as np
import pandas as pd

from fista_lib import fg_func, find_ik, prox_map
from make_matrix import mkmat_cs, mkmat_cs_w

def _get_frecvec(freqinfo):
    freq_edge = np.linspace(freqinfo[0],
                            freqinfo[1],
                            int(freqinfo[2] + 1))
    freq_vec = (freq_edge[:-1] + freq_edge[1:]) / 2
    return freq_vec

def fista(data1, data2, freqinfo, lam):

    # frequency setup
    freq_lo = freqinfo[0]
    freq_up = freqinfo[1]
    nfreq   = freqinfo[2]
    delta_freq = (freq_up - freq_lo) / nfreq
    freqdata = _get_frecvec(freqinfo)

    # make matrix file
    A_mat = mkmat_cs(data1[:,0], data2[:,0], freqdata)
    A_w_mat = mkmat_cs_w(data1[:,0], data2[:,0], freqdata)

    # prepare for estimation
    nrow1 = data1.shape[0]
    nrow2 = data2.shape[0]
    ncol  = A_mat.shape[1]

    data = np.hstack([data1, data2])

    h_vec = np.hstack([data1[:,1] * np.sqrt(nrow2), data2[:,1] * np.sqrt(nrow1)])

    # estimation
    tolerance = 5.e-8
    eta = 1.2
    x = np.repeat(0.0, ncol)
    x_pre = x
    y = x
    L = 1e-3
    L_pre = L
    k_max = 500
    t = 1

    cost = 0.0
    cost_pre = cost
    for k in range(k_max):
        ik = find_ik(y, L_pre, eta, A_w_mat, h_vec, lam)
        L = eta**ik * L_pre
        x = prox_map(y, L, A_w_mat, h_vec, lam)
        t_new = (1. + np.sqrt(1. + 4 * t**2))/2.
        y_new = x + (t - 1.)/ t_new * (x - x_pre)
        x_pre = x
        y = y_new
        t = t_new
        L_pre = L

        cost = fg_func(y_new, A_w_mat, h_vec, lam)
        if k > 1 and (cost_pre - cost) / cost < tolerance:
            # print('k: {}'.format(k))
            # print('cost: {}'.format(cost))
            break
        cost_pre = cost

    return x

if __name__ == '__main__':
    infile1 = 'lc1.dat'
    infile2 = 'lc2.dat'
    freqinfo = [0, 0.5, 200]
    x = fista(infile1, infile2, freqinfo, 1)
    df_x = pd.DataFrame({'a': x[:200],
                         'b': x[200:400],
                         'c': x[400:600],
                         'd': x[600:800]})
    from summary import make_summary
    import matplotlib.pyplot as plt
    df_sum = make_summary(x, freqinfo)
    plt.scatter(df_sum.lag, df_sum.norm12)
    plt.show()
    print(df_sum)
