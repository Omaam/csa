import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def _triang(t, basewidth, T):
    bw = basewidth
    if (((T-bw)/2 <= t) & (t < T/2)):
        w = 2*t/bw - (T-bw)/bw
    elif ((T/2 <= t) & (t < (T+bw)/2)):
        w = -2*t/bw + (T+bw)/bw
    else:
        w = 0
    return w

def _hann(t, T):
    w = 0.5 - 0.5*np.cos(2*np.pi*t/T)
    return w

class WindowGenerator():

    def __init__(self, segrange):

        # set first time
        self.t0 = segrange[0]
        self.tm1 = segrange[1]
        self.T = segrange[1] - segrange[0]

        # make function
        w = lambda t: 0*t + 1
        self.func = w

    def __call__(self, t):
        assert np.all((t >= self.t0) & (t <= self.tm1)), ValueError(
            f't must be between {self.t0:.3f} <= t <= {self.tm1:.3f}')
        return self.func(t - self.t0)

    def gene(self, t):
        assert np.all((t >= self.t0) & (t <= self.tm1)), ValueError(
            f't must be between {self.t0:.3f} <= t <= {self.tm1:.3f}')
        return self.func(t - self.t0)

    def acf(self):
        n = 1000
        t = np.linspace(self.t0, self.tm1, n)
        w = self.func(t - self.t0)
        acf = 1 / (np.sum(w)/n)
        return acf


    def ecf(self):
        n = 1000
        t = np.linspace(self.t0, self.tm1, n)
        w = self.func(t - self.t0)
        ecf = np.sqrt(1 / (np.sum(w**2)/n))
        return ecf

    def hann(self):
        self.func = np.frompyfunc(lambda t: _hann(t, self.T),
                                  1, 1)
        self.sect = 0.5 * self.T
        self.ecf = 1.63
        self.acf = 2.0

    def triang(self, basewidth):
        # set triangular function as universal function by using
        # numpy.frompyfunc
        self.func = np.frompyfunc(lambda t: _triang(t, basewidth, self.T),
                                  1, 1)
        self.sect = 0.5 * basewidth
        self.acf = (basewidth/2)/self.T
        self.ecf = 2/3 * basewidth #?


def main():
    np.random.seed(2021)
    t = np.arange(1, 101) + np.random.normal(0, 0.1, 100)
    w = Window(t)
    tt = np.linspace(2, 99, 10000)
    w.triang(basewidth=2)
    plt.plot(tt, w(tt))
    w.hann()
    plt.plot(tt, w(tt))
    plt.legend(['triang', 'hann'])
    plt.show()

if __name__ == '__main__':
    main()
