import numpy as np
import pandas as pd

from csa.signiftest import LagSignifTest
from csa.tools import phist


__all__ = ['signiftest', 'query_forX']


def _get_freq(freqinfo):
    freq_edge = np.linspace(freqinfo[0],
                            freqinfo[1],
                            int(freqinfo[2] + 1))
    freq_vec = (freq_edge[:-1] + freq_edge[1:]) / 2
    return freq_vec


def _limit_phase(phase_diff):
    if phase_diff < -np.pi:  # phase_lag > pi
        phase_diff_out = 2*np.pi + phase_diff
    elif phase_diff >= np.pi:
        phase_diff_out = -2*np.pi + phase_diff
    else:
        phase_diff_out = phase_diff
    return phase_diff_out


def _make_summary(x, freqinfo, anti=False):

    # x
    x_data = x.reshape(4, int(freqinfo[2])).T
    df_x = pd.DataFrame(x_data, columns=['a', 'b', 'c', 'd'])

    # freq
    df_x['freq'] = _get_freq(freqinfo)
    df_x['period'] = 1 / df_x.freq.values

    # norm
    df_x['norm1'] = np.sqrt(df_x.a.values**2 + df_x.b.values**2)
    df_x['norm2'] = np.sqrt(df_x.c.values**2 + df_x.d.values**2)
    df_x['norm12'] = np.sqrt(df_x.norm1.values**2 + df_x.norm2.values**2)

    # lag
    df_x['alpha1'] = np.arctan2(df_x.b[df_x.a != 0], df_x.a[df_x.a != 0])
    df_x['alpha2'] = np.arctan2(df_x.d[df_x.c != 0], df_x.c[df_x.c != 0])
    df_x.fillna(0, inplace=True)
    delta_alpha = list(map(_limit_phase, df_x.alpha2 - df_x.alpha1))
    df_x['lag'] = delta_alpha / (2 * np.pi * df_x.freq.values)

    # summary
    df_sum = df_x[['lag', 'norm12', 'norm1', 'norm2',
                   'period', 'freq']][df_x.norm12 > 0]
    if anti:
        df_sum = _make_antisum(df_sum)

    return df_sum


def _make_periodicsum(x, freqinfo, lagrange):
    # load df_sum
    df_sum = _make_summary(x, freqinfo)
    n_ori = df_sum.shape[0]

    # duplicate periodical counterparts
    sumdata = df_sum.values
    for row in df_sum.itertuples(index=False):
        # get periodical lag
        lagdup_plus = np.arange(row.lag, lagrange[1], row.period)
        lagdup_minu = -np.arange(-row.lag, np.abs(lagrange[0]), row.period)
        lagdup = np.unique(np.hstack([lagdup_plus, lagdup_minu, row.lag]))

        # append to sumdata
        sumdata_dup = np.tile(row, (len(lagdup), 1))
        sumdata_dup[:, 0] = lagdup
        sumdata = np.append(sumdata, sumdata_dup, axis=0)

    # recreate df_sum for output
    df_sum_out = pd.DataFrame(sumdata[n_ori:], columns=df_sum.columns)
    return df_sum_out


def _make_periodicsumold(x, freqinfo, peridicrange):
    df_sum = _make_summary(x, freqinfo)
    columns = df_sum.columns.values
    ndarray_sum = df_sum.values
    for para in list(ndarray_sum):
        (lag, norm12, norm1, norm2, period, freq) = para
        # plus
        i = 0
        lag_peri = lag
        while peridicrange[0] < lag + (i*period) <= peridicrange[1]:
            lag_peri = lag + (i*period)
            new_col = np.array([lag_peri, norm12, norm1,
                                norm2, period, freq])
            ndarray_sum = np.vstack([ndarray_sum, new_col])
            i += 1
        # munus
        i = -1
        lag_peri = lag - period
        # while lag_peri > peridicrange[0]:
        while peridicrange[0] < lag + (i*period) <= peridicrange[1]:
            lag_peri = lag + (i*period)
            new_col = np.array([lag_peri, norm12, norm1,
                                norm2, period, freq])
            ndarray_sum = np.vstack([ndarray_sum, new_col])
            i += -1
    df_sum_out = pd.DataFrame(ndarray_sum, columns=columns)
    return df_sum_out


def _search_index(a, v):
    idx_out = []
    for vv in v:
        idx_match = int(np.where(a == vv)[0])
        idx_out.append(idx_match)
    return np.array(idx_out)


def _make_antisum(df_sum):
    df_sum_out = df_sum.copy()
    df_sum_out.loc[:, 'lag'] = list(map(
        lambda lag, peri: lag - peri/2 if lag >= 0 else lag + peri/2,
        df_sum_out['lag'], df_sum_out['period']))
    return df_sum_out


# main functions

def query_forX(X, freqinfo, para, pararanges,
               mode='ext', anti=False, periodic=False):

    '''
    '''
    X_out = _adjuster_forX(_query_forX, X, freqinfo, para, pararanges,
                           mode=mode, anti=anti, periodic=periodic)
    return X_out


def _query_forX(x, freqinfo, para, pararanges,
                mode='ext', anti=False, periodic=False):

    # get basic quantity
    df_sum = _make_summary(x, freqinfo, anti=anti)
    if periodic:
        periodicrange = np.quantile(df_sum.lag, [0.01, 0.99])
        df_sum = _make_periodicsum(x, freqinfo, periodicrange)
    freqs = _get_freq(freqinfo)
    pararanges = np.array(pararanges)

    # make the shape of pararanges (:,2) even if
    # the original shape is (:, 1)
    if isinstance(pararanges[0], np.ndarray) is False:
        pararanges = [pararanges]

    # if para = 'ratio', then make ratio
    if para == 'ratio':
        df_sum['ratio'] = list(map(lambda n1, n2: n1/n2 if n1 < n2 else n2/n1,
                                   df_sum.norm1, df_sum.norm2))

    # search lag components and put 0
    flags = np.zeros(freqs.shape[0])
    for pararange in pararanges:
        freq_quer = df_sum.query(
            f'{pararange[0]} <= {para} < {pararange[1]}').freq
        idx_quer = _search_index(freqs, freq_quer)
        if idx_quer.size != 0:
            flags[idx_quer] = 1

    # reverse if 'cut' mode
    if 'cut' in mode:
        flags = 1 - flags

    # extract x where flag == 1
    x_out = x * np.tile(flags, 4)
    return x_out


def signiftest(X, freqinfo, testrange, lagbinwidth=1,
               iteration=1000, ci=0.9, periodic=False,
               anti=False):
    '''
    '''
    X_out = _adjuster_forX(_signiftest, X, freqinfo, testrange,
                           lagbinwidth=lagbinwidth,
                           iteration=iteration, ci=ci,
                           periodic=periodic, anti=anti)
    return X_out


def _signiftest(x, freqinfo, testrange, lagbinwidth=1,
                iteration=1000, ci=0.9, periodic=False,
                anti=False):

    df_sum = _make_summary(x, freqinfo, anti=anti)
    tester = LagSignifTest(df_sum, lagrange=testrange,
                           lag_binwidth=lagbinwidth)
    tester.make_model(iteration=iteration)
    signifrange = tester.get_signifrange(ci=ci)
    x_signif = query_forX(x, freqinfo, para='lag',
                          pararanges=signifrange,
                          periodic=periodic)
    return x_signif


def subtractX(X_minuend, X_subtrahend):
    X_mask = np.where(X_subtrahend == 0, 1, 0)
    X_diff = X_minuend * X_mask
    return X_diff


def addX(*X_addends):
    X_sum = X_addends[0].copy()
    for X_addend in X_addends[1:]:
        cond_nonzero = (X_addend != 0)
        X_sum[cond_nonzero] += X_addend[cond_nonzero]
    return X_sum


def _adjuster_forX(func, X, *args, **kargs):
    '''Adjuster for a X.
       By using this, it is possible to use
       functions without taking care of the
       shape of the X.
    '''
    # for cs
    if len(X.shape) == 1:
        X_out = func(X, *args, **kargs)
    # for stcs
    elif len(X.shape) == 2:
        X_out = X.copy()
        for i, x in enumerate(X.T):
            X_out[:, i] = func(x, *args, **kargs)
    return X_out


def main():
    freqinfo = [0, 0.5, 2000]
    X = np.loadtxt('../example/X.dat')
    lags, powsums = phist(X, freqinfo, lagrange=[-10, 10])

    x_cut = query_forX(X, freqinfo, 'lag', [-2, 2], 'cut', periodic=True)
    testrange = [-10, 10]
    x_signif = signiftest(x_cut, freqinfo, testrange, iteration=100,
                          periodic=True)
    print(x_signif)


if __name__ == '__main__':
    main()
