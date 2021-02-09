import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sig
from scipy.stats import zscore
from scipy import fftpack

from csa.bootstrap import ccf


sns.set(style='whitegrid')


def add_bigaxis(fig, xlabel=None, ylabel=None,
                adjust_para=(0.1, 0.1, 0.9, 0.9)):
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # adjust_para = (left, bottom, right, top)
    plt.subplots_adjust(*adjust_para)
    fig.tight_layout()
    return fig


def psd(h, fs=1):
    N = len(h)
    T = N / fs
    H = fftpack.fft(h)
    freqs = fftpack.fftfreq(N)
    energy = np.abs(H)**2
    power = energy / T
    return freqs[1:int(N/2)], power[1:int(N/2)]


def main():
    # read file
    df_ori = pd.read_csv('./gx339_night1_2.txt', delim_whitespace=True)[:25000]
    df_ori['t'] = df_ori.MJD * 24 * 3600
    df_ori['t'] = df_ori.t - df_ori.t[0]

    x = zscore(df_ori.X)
    o = zscore(df_ori.O)
    t = df_ori.t
    N = df_ori.shape[0]
    fs = 20
    w = sig.hann(N)
    print(f'df_ori: {df_ori.shape}')

    # x-ray
    H_x = np.fft.fft(x * w)
    freq = np.fft.fftfreq(N, 0.05)
    h_x = np.real(np.fft.ifft(H_x))

    filt = ~((0.1 <= freq) & (freq <= 0.3))
    H_x_filt = H_x
    H_x_filt[filt] = 0
    h_x_filt = np.real(np.fft.ifft(H_x_filt))

    # optical
    H_o = np.fft.fft(o * w)
    freq = np.fft.fftfreq(N, 0.05)
    h_o = np.real(np.fft.ifft(H_o))

    filt = ~((0.1 <= freq) & (freq <= 0.3))
    H_o_filt = H_o
    H_o_filt[filt] = 0
    h_o_filt = np.real(np.fft.ifft(H_o_filt))

    # plot
    duration = 50 * fs
    start = np.random.randint(0, N-duration)
    end = start + duration

    fig = plt.figure(figsize=(10, 7))

    plt.subplot2grid((2, 4), (0, 0), rowspan=1)
    plt.title('whole period')
    plt.plot(t, h_o, t, h_o_filt)

    plt.subplot2grid((2, 4), (0, 1), rowspan=1)
    plt.plot(t[start:end], h_o[start:end], t[start:end], h_o_filt[start:end])
    plt.title('specific period')

    plt.subplot2grid((2, 4), (1, 0), rowspan=1)
    plt.plot(t, h_x, t, h_x_filt)
    plt.xlabel('time (s)')

    plt.subplot2grid((2, 4), (1, 1), rowspan=1)
    plt.plot(t[start:end], h_x[start:end], t[start:end], h_x_filt[start:end])
    plt.xlabel('time (s)')

    plt.subplot2grid((2, 4), (0, 2), rowspan=1)
    freqs, pows = psd(H_x_filt, 20)
    plt.plot(freqs, pows)
    plt.title('PSD')
    plt.yscale('log')
    plt.xscale('log')

    plt.subplot2grid((2, 4), (1, 2), rowspan=1)
    freqs, pows = psd(H_o_filt, 20)
    plt.plot(freqs, pows)
    plt.xlabel('frequency (Hz)')
    plt.yscale('log')
    plt.xscale('log')

    # CCF
    plt.subplot2grid((2, 4), (1, 3), rowspan=2)
    lag, corr = ccf(h_x_filt, h_o_filt, 0.05)
    plt.plot(lag, corr)
    plt.title('CCF')
    plt.xlabel('lag (s)')
    plt.ylabel('Correlation coefficient')

    # setting
    fig = add_bigaxis(fig, ylabel='flux')
    plt.tight_layout()
    plt.savefig('fig/lc_with_ccf.png')
    # plt.show()
    plt.close()

    # save ccf
    fig, ax = plt.subplots()
    lag, corr = ccf(h_x_filt, h_o_filt, maxlags=10, fs=20)
    plt.plot(lag, corr)
    # plt.title('CCF')
    plt.xlabel('Lag (s)')
    plt.ylabel('Correlation coefficient')
    plt.tight_layout()
    plt.savefig('fig/ccf.png')
    plt.show()

    # fig, ax = plt.subplots(2, 3, figsize=(10,7))
    # ax[0, 0].plot(t, h_o, t, h_o_filt)
    # ax[0, 1].plot(t[start:end], h_o[start:end],
    #               t[start:end], h_o_filt[start:end])
    # ax[1, 0].plot(t, h_x, t, h_x_filt)
    # ax[1, 1].plot(t[start:end], h_x[start:end],
    #               t[start:end], h_x_filt[start:end])


if __name__ == '__main__':
    main()
