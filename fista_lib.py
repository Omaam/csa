import numpy as np
import pandas as pd

__all__ = ['prox_map', 'fg_func', 'find_ik']

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
