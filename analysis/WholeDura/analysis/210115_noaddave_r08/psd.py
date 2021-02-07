import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


def psd(h, fs=1):
    N = len(h)
    T = N / fs
    H = fftpack.fft(h)
    freqs = fftpack.fftfreq(N, 1/fs)
    energy = np.abs(H)**2
    power = energy / T
    return freqs[1:int(N/2)], power[1:int(N/2)]


def main():

    # load xps
    lcdata1_xps = np.loadtxt('./XY/data1_xps_000.dat')

    # load ops
    lcdata1_ops = np.loadtxt('./XY/data1_ops_000.dat')

    # window function
    N = len(lcdata1_xps)
    w = np.hanning(N)

    # XPS
    freq, H_xps = psd(lcdata1_xps[:, 1] * w, 20)

    # OPS
    freq, H_ops = psd(lcdata1_ops[:, 1] * w, 20)

    # figure
    plt.rcParams["font.size"] = 15
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['xtick.direction'] = 'in'  # x axis in
    plt.rcParams['ytick.direction'] = 'in'  # y axis in
    fig, ax = plt.subplots(1)
    ax.plot(freq, H_xps, color='tab:orange', alpha=.7)
    ax.plot(freq, H_ops, color='tab:blue', alpha=.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Standardized power')

    # arrange
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.95)
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/psd.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
