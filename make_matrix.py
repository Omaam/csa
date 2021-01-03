import numpy as np

def make_matrix(t, f):
    t = np.array([t]).astype(float)
    f = np.array([f]).astype(float)
    phase = 2 * np.pi * np.dot(t.T, f)
    mat = np.hstack([np.cos(phase), np.sin(phase)])
    return mat

def mkmat_cs(t1, t2,  freq_vec):
    delta_f = np.mean(np.diff(freq_vec))
    mat_a1 = 2 * delta_f * make_matrix(t1, freq_vec)
    mat_a2 = 2 * delta_f * make_matrix(t2, freq_vec)
    mat_b1 = np.zeros_like(mat_a2)
    mat_b2 = np.zeros_like(mat_a1)
    mat_1 = np.hstack([mat_a1, mat_b1])
    mat_2 = np.hstack([mat_b2, mat_a2])
    mat = np.vstack([mat_1, mat_2])
    return mat

def mkmat_cs_w(t1, t2,  freq_vec):
    N1 = t1.shape[0]
    N2 = t2.shape[0]
    delta_f = np.mean(np.diff(freq_vec))
    mat_a1 = 2 * delta_f * make_matrix(t1, freq_vec) * np.sqrt(N2)
    mat_a2 = 2 * delta_f * make_matrix(t2, freq_vec) * np.sqrt(N1)
    mat_b1 = np.zeros_like(mat_a2)
    mat_b2 = np.zeros_like(mat_a1)
    mat_1 = np.hstack([mat_a1, mat_b1])
    mat_2 = np.hstack([mat_b2, mat_a2])
    mat = np.vstack([mat_1, mat_2])
    return mat
