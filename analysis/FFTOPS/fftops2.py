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
    # h_x = np.real(np.fft.ifft(H_x))

    filt = ~((0.1 <= freq) & (freq <= 0.3))
    H_x_filt = H_x
    H_x_filt[filt] = 0
    h_x_filt = np.real(np.fft.ifft(H_x_filt))

    # optical
    H_o = np.fft.fft(o * w)
    freq = np.fft.fftfreq(N, 0.05)
    # h_o = np.real(np.fft.ifft(H_o))

    filt = ~((0.1 <= freq) & (freq <= 0.3))
    H_o_filt = H_o
    H_o_filt[filt] = 0
    h_o_filt = np.real(np.fft.ifft(H_o_filt))

    # plot
    np.random.seed(20210203)
    duration = 100 * fs
    start = np.random.randint(0, N-duration)
    end = start + duration

    # figure
    fig, ax = plt.subplots(3, figsize=(8, 7))

    # light curves
    ax[0].plot(t[start:end], h_x_filt[start:end])
    ax[1].plot(t[start:end], h_o_filt[start:end])
    ax[0].set_ylabel('Standardized flux')
    ax[1].set_ylabel('Standardized flux')
    ax[1].set_xlabel('Time (s)')
    ax[0].text(0.97, 0.95, 'X-ray', ha='right', va='top', fontsize=15,
               transform=ax[0].transAxes)
    ax[1].text(0.97, 0.95, 'Optical', ha='right', va='top', fontsize=15,
               transform=ax[1].transAxes)

    # ccf
    lag, corr = ccf(h_x_filt, h_o_filt, fs=20, maxlags=10)
    ax[2].plot(lag, corr)
    ax[2].set_xlabel('Lag (s)')
    ax[2].set_ylabel('Correlation')

    # setting
    plt.tight_layout()
    plt.savefig('lc_ccf.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
