import numpy as np
import pandas as pd

from .make_matrix import mkmat_cs, mkmat_cs_w


def soft_thres(a, b, c, d, L, lam):
    ''' Calcurate soft threshold
    '''
    l = np.sqrt(a**2 + b**2 + c**2 + d**2) * L
    ans = 0.0
    if lam < l:
        ans = 1.0 - lam /l
    elif -lam <= l and l <= lam:
        ans = 0.0
    elif l < -lam:
        ans = 1.0 + lam /l
    return ans

def prox_map(y, L, A_mat, h_vec, lam):
    b = y - 2.0/L * np.dot(A_mat.T, np.dot(A_mat, y) - h_vec)
    N2 = b.shape[0]
    N = N2 / 2
    b_new = np.repeat(0.0, 2 * N)
    for i in range(int(N/2)):
        factor = soft_thres(b[i],
                            b[int(i + N/2)],
                            b[int(i + N)],
                            b[int(i + N/2 * 3)], L, lam)
        b_new[int(i)]       = b[int(i)] * factor
        b_new[int(i + N/2)] = b[int(i + N/2)] * factor
        b_new[int(i + N)]   = b[int(i + N)] * factor
        b_new[int(i + N/2 * 3)] = b[int(i + N/2 * 3)] * factor
    return b_new

def g_func(J_vec, lam):
    N2 = J_vec.shape[0]
    N = N2 / 2
    sigsum = 0.0
    for i in range(int(N/2)):
        sigsum = sigsum + np.sqrt(J_vec[int(i + N/2 * 0)]**2 +
                                  J_vec[int(i + N/2 * 1)]**2 +
                                  J_vec[int(i + N/2 * 2)]**2 +
                                  J_vec[int(i + N/2 * 3)]**2)
        ans = lam * sigsum
    return ans

def f_func(J_vec, A_mat, h_vec):
    vec_tmp = h_vec - np.dot(A_mat, J_vec)
    ans = np.dot(vec_tmp, vec_tmp.T)
    return ans

def fg_func(J_vec, A_mat, h_vec, lam):
    return f_func(J_vec, A_mat, h_vec) + g_func(J_vec, lam)

def diff_func(y, A_mat, h_vec):
    return 2 * np.dot(A_mat.T, np.dot(A_mat, y) - h_vec)

def q_func(x, y, L, A_mat, h_vec, lam):
    term1 = f_func(y, A_mat, h_vec)
    term2 = np.sum((x - y) * diff_func(y, A_mat, h_vec))
    term3 = L / 2.0 * np.dot((x - y).T, x - y)
    term4 = g_func(x, lam)
    ans = term1 + term2 + term3 + term4
    return ans

def find_ik(y, L_pre, eta, A_mat, h_vec, lam):
    ik_max = 1000
    ik = 0
    while ik <= ik_max:
        L = eta**ik * L_pre
        pLy = prox_map(y, L, A_mat, h_vec, lam)
        fgfunc_val = fg_func(pLy, A_mat, h_vec, lam)
        qfunc_val  = q_func(pLy, y, L, A_mat, h_vec, lam)
        if fgfunc_val <= qfunc_val:
            break
        ik = ik + 1
    return ik

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

    h_vec = np.hstack([data1[:,1] * np.sqrt(nrow2),
                       data2[:,1] * np.sqrt(nrow1)])

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

    return freqdata, x

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
