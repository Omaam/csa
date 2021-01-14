import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import signal

from csa.deco import change_directory


def ccf(x, y, fs=1, maxlags=None):
    # standardized
    x, y = zscore(x), zscore(y)

    # calcurate correlation and lags
    n_x, n_y = len(x), len(y)
    T = max(n_x, n_y)
    r = signal.correlate(y, x, mode='full') / T
    lags = np.arange(-n_x + 1, n_y) / fs

    # query
    maxlags = 2 * T - 1 if maxlags is None else maxlags
    lag_out = lags[((-maxlags <= lags) & (lags <= maxlags))]
    r_out = r[((-maxlags <= lags) & (lags <= maxlags))]
    return lag_out, r_out


def ccfbootstrap(Y1, Y2, droprate=1, fs=1,
                 q=(0.025, 0.5, 0.975), maxlags=None):

    # check whether the shapes of Y1 and Y2 are the same.
    assert Y1.shape == Y2.shape, \
           f'the sizes of Y1 and Y2 must be the same: \
             {Y1.shape} != {Y2.shape}'
    n_sample = Y1.shape[1] * droprate

    # ccf
    C = []
    for i in range(len(Y1)):
        lags, c = ccf(Y1[i], Y2[i], fs=fs, maxlags=maxlags)
        C.append(c)
    C = np.array(C)

    # get quantile
    c_low, c_med, c_hig = np.quantile(C, q, axis=0)
    c_lowdev = (c_med - c_low)/np.sqrt(n_sample)
    c_higdev = (c_med - c_low)/np.sqrt(n_sample)
    print(c_low[10], c_med[10], c_hig[10])
    print((c_med-c_lowdev)[10], c_med[10], (c_med+c_higdev)[10])
    return lags, c_med-c_lowdev, c_med, c_med+c_higdev


@change_directory('../example')
def main():
    data1 = np.loadtxt('xdata.dat')
    data2 = np.loadtxt('odata.dat')

    nlc = 10000
    Y1 = np.tile(data1[:, 1].T, (nlc, 1))
    Y1 += np.random.normal(0, 2*data1[:, 1].std(), Y1.shape)
    Y2 = np.tile(data1[:, 1].T, (nlc, 1))
    Y2 += np.random.normal(0, 2*data2[:, 1].std(), Y2.shape)

    lags, c_low, c_med, c_hig = ccfbootstrap(Y1, Y2)
    plt.fill_between(lags, c_low, c_hig, alpha=.5)
    plt.plot(lags, c_med)
    plt.show()


if __name__ == '__main__':
    main()
