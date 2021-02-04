import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import signal
from tqdm import tqdm, trange

# decorators

def stopwatch(func):
    def wrapper(*arg, **kargs):
        start = time.time()
        print(f'start {func.__name__}')
        res = func(*arg, **kargs)
        dura = (time.time() - start)
        print(time.strftime(f'finish {func.__name__}: %H:%M\'%S\"',
                            time.gmtime(dura)))
        return res
    return wrapper


def change_directory(path_to_dir):
    def _change_directory(func):
        def wrapper(*args, **kargs):
            current_dir = os.getcwd()
            if os.path.exists(path_to_dir) is False:
                os.makedirs(path_to_dir)
            os.chdir(path_to_dir)
            results = func(*args, **kargs)
            os.chdir(current_dir)
            return(results)
        return wrapper
    return _change_directory


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


def ccfbootstrap(Y1, Y2, fs=1, q=(0.025, 0.5, 0.975), maxlags=None):

    # check whether the shapes of Y1 and Y2 are the same.
    assert Y1.shape == Y2.shape, \
           f'the sizes of Y1 and Y2 must be the same: \
             {Y1.shape} != {Y2.shape}'

    # ccf
    C = []
    for i in range(len(Y1)):
        lags, c = ccf(Y1[i], Y2[i], fs=fs, maxlags=maxlags)
        C.append(c)
    C = np.array(C)

    # get quantile
    c_low, c_med, c_hig = np.quantile(C, q, axis=0)
    return lags, c_low, c_med, c_hig


@change_directory('example')
def main():
    data1 = np.loadtxt('xdata.dat')
    data2 = np.loadtxt('odata.dat')

    nlc = 10000
    Y1 = np.tile(data1[:,1].T, (nlc, 1))
    Y1 += np.random.normal(0, 2*data1[:,1].std(), Y1.shape)
    Y2 = np.tile(data1[:,1].T, (nlc, 1))
    Y2 += np.random.normal(0, 2*data2[:,1].std(), Y2.shape)

    lags, c_low, c_med, c_hig = ccfbootstrap(Y1, Y2)
    plt.fill_between(lags, c_low, c_hig, alpha=.5)
    plt.plot(lags, c_med)
    plt.show()


if __name__ == '__main__':
    main()
