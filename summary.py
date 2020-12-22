import numpy as np
import pandas as pd

def limit_phase(phase_diff):
    if phase_diff < -np.pi: # phase_lag > pi
        phase_diff_out = 2*np.pi + phase_diff
    elif phase_diff >= np.pi:
        phase_diff_out = -2*np.pi + phase_diff
    else:
        phase_diff_out = phase_diff
    return phase_diff_out

def make_summary(x, freq_info):
    # x
    x_data = x.reshape(4, int(freq_info[2])).T
    df_x = pd.DataFrame(x_data, columns=['a', 'b', 'c', 'd'])
    # freq
    freq_lo = freq_info[0]
    freq_up = freq_info[1]
    n_freq = int(freq_info[2])
    delta_freq = (freq_up - freq_lo) / n_freq
    df_x['freq'] = freq_lo + delta_freq * (df_x.index.values + 0.5)
    df_x['period'] = 1 / df_x.freq.values
    # norm
    df_x['norm1'] = np.sqrt(df_x.a.values**2 + df_x.b.values**2)
    df_x['norm2'] = np.sqrt(df_x.c.values**2 + df_x.d.values**2)
    df_x['norm12'] = np.sqrt(df_x.norm1.values**2 + df_x.norm2.values**2)
    # lag
    df_x['alpha1'] = np.arctan2(df_x.b[df_x.a != 0], df_x.a[df_x.a != 0])
    df_x['alpha2'] = np.arctan2(df_x.d[df_x.c != 0], df_x.c[df_x.c != 0])
    df_x.fillna(0, inplace=True)
    delta_alpha = list(map(limit_phase, df_x.alpha2 - df_x.alpha1))
    df_x['lag'] = delta_alpha / (2 * np.pi * df_x.freq.values)

    df_sum = df_x[['lag', 'norm12', 'norm1', 'norm2', 'period', 'freq']][df_x.norm12 > 0]

    return df_sum
