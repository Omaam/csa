import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
import seaborn as sns
sns.set_style('whitegrid')


def mkmat(t, f):
    t = np.array([t])
    f = np.array([f])
    phase = 2 * np.pi * np.dot(t.T, f)
    mat = np.hstack([np.cos(phase), np.sin(phase)])
    return mat


def get_frecvec(low, hig, num):
    freq_edge = np.linspace(low, hig, int(num + 1))
    freq_vec = (freq_edge[:-1] + freq_edge[1:]) / 2
    return freq_vec


def powerlaw(freq_vec, beta=2):
    pow_vec = freq_vec ** -beta
    return pow_vec


def psd(h, fs=1):
    N = len(h)
    T = N / fs
    H = fftpack.fft(h)
    freqs = fftpack.fftfreq(N)
    energy = np.abs(H)**2
    power = energy / T
    return freqs[1:int(N/2)], power[1:int(N/2)]


def get_plsimulation(n_time, lag, f_vec, seed=0):
    '''return y1_data, y2_data: ndarray
    '''
    assert isinstance(lag, int), 'lag must be int'

    # rondom phase
    np.random.seed(seed)
    power_vec = powerlaw(f_vec)
    realpart_vec = np.random.normal(0, np.sqrt(power_vec))
    imagpart_vec = np.random.normal(0, np.sqrt(power_vec))

    # calculation
    x_vec = np.hstack([realpart_vec, imagpart_vec])
    delta_f = np.diff(f_vec).mean()
    t_vec = np.linspace(0, 2 * n_time - 1, 2* n_time)
    A_mat = 2 * delta_f * mkmat(t_vec, f_vec)
    y_vec = np.dot(A_mat, x_vec)
    data1 = np.array([t_vec[:n_time], y_vec[lag:lag+n_time]]).T
    data2 = np.array([t_vec[:n_time], y_vec[:n_time]]).T

    return data1, data2



def show_psd(f_vec, ax=None, seed=0):
    # random phase
    np.random.seed(seed)
    power_vec = powerlaw(f_vec)
    realpart_vec = np.random.normal(0, np.sqrt(power_vec))
    imagpart_vec = np.random.normal(0, np.sqrt(power_vec))
    ranpower = realpart_vec ** 2 + imagpart_vec ** 2

    # plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(f_vec, ranpower, label='randomized power')
    ax.plot(f_vec, power_vec, label=r'$S(f)\sim f^{-2}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('freq')
    ax.set_ylabel('power')
    ax.legend(loc='best')
    return ax


def main():

    # frequency
    freq_lo = 0
    freq_hi = 0.5
    nfreq = 200
    f_vec = get_frecvec(freq_lo, freq_hi, nfreq)

    # calculation
    n_time = 100
    lag = 5
    xdata, odata = get_plsimulation(n_time, lag, f_vec, seed=19)
    np.savetxt('xdata.dat', xdata)
    np.savetxt('odata.dat', odata)

    # calcurate psd
    freqs, power_x = psd(xdata[:,1])
    freqs, power_o = psd(odata[:,1])


    # plot lc
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1]})
    # optical
    ax[0, 0].plot(odata[:, 0], odata[:, 1])
    ax[0, 0].set_ylabel('flux')
    ax[0, 1].plot(freqs, power_o)
    ax[0, 1].plot(freqs, powerlaw(freqs), linestyle=':')
    ax[0, 1].set_ylabel('power')
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xscale('log')
    # x-ray
    ax[1, 0].plot(xdata[:, 0], xdata[:, 1])
    ax[1, 0].set_xlabel('time')
    ax[1, 0].set_ylabel('flux')
    ax[1, 1].plot(freqs, power_x)
    ax[1, 1].plot(freqs, powerlaw(freqs), linestyle=':')
    ax[1, 1].set_ylabel('power')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('freq')
    # show and save
    plt.tight_layout()
    plt.savefig('lc_psd.png')
    plt.show()


if __name__ == '__main__':
    main()
