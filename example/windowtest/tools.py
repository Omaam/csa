import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm, trange


def get_freq(freqinfo):
    freq_edge = np.linspace(freqinfo[0],
                            freqinfo[1],
                            int(freqinfo[2] + 1))
    freq_vec = (freq_edge[:-1] + freq_edge[1:]) / 2
    return freq_vec


def _limit_phase(phase_diff):
    if phase_diff < -np.pi: # phase_lag > pi
        phase_diff_out = 2*np.pi + phase_diff
    elif phase_diff >= np.pi:
        phase_diff_out = -2*np.pi + phase_diff
    else:
        phase_diff_out = phase_diff
    return phase_diff_out


def make_summary(x, freqinfo, anti=False):

    # x
    x_data = x.reshape(4, int(freqinfo[2])).T
    df_x = pd.DataFrame(x_data, columns=['a', 'b', 'c', 'd'])

    # freq
    df_x['freq'] = get_freq(freqinfo)
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
    df_sum = df_x[['lag', 'norm12', 'norm1', 'norm2',\
                   'period', 'freq']][df_x.norm12 > 0]
    if anti:
        df_sum = _make_antisum(df_sum)

    return df_sum


def phist(X, freqinfo, bins=10, lagrange=None, density=False):

    # add axis if X has 1-dimension
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    phists = []
    for i, x in enumerate(X.T):
        # make sumarry
        df_sum = make_summary(x, freqinfo)

        # get values
        lags = df_sum.lag
        norms = df_sum.norm12

        # get edges
        bin_edges = np.histogram_bin_edges(lags, bins=bins, range=lagrange)

        # get index which each value belongs to
        inds = np.digitize(lags, bin_edges) - 1

        # add each power
        powsums = np.zeros(len(bin_edges)-1)
        for ind in range(len(powsums)):
            powsums[ind] = norms[inds == ind].sum()
        if density:
            powsums /= powsums.sum()
        phists.append(powsums)

    phists = np.squeeze(np.array(phists).T)

    return phists, bin_edges


def plot_cbplot(x, freqinfo, threshold=None, lagrange=None,
                bin_edges=None, ax=None, anti=False):
    '''
    '''
    # make summary
    df_sum = make_summary(x, freqinfo, anti=anti)

    # figure
    if ax == None:
        fig, ax = plt.subplots()

    # color babble plot
    size = list(map(lambda n1, n2: n1/n2 if n1 <= n2 else n2/n1,
                    df_sum.norm1, df_sum.norm2))
    log_size = np.log10(1 + np.array(size)) * 750
    im = ax.scatter(df_sum.lag, df_sum.norm12, c=df_sum.freq,
                    cmap=cm.jet, alpha=0.8, s=log_size,
                    norm=Normalize(freqinfo[0], freqinfo[1]))
    ax.set_ylim(np.min(df_sum.norm12))
    ax.set_yscale('log')
    ax.set_ylabel('Power')
    ax.set_xlabel('Lag')
    xlim = lagrange if lagrange else (bin_edges[0], bin_edges[-1])
    ax.set_xlim(xlim)

    # colorbat setting
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    plt.colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_text("Frequency")
    cax.xaxis.set_label_position("top")

    return ax


def plot_omplot(x, freqinfo, lagrange=None, bins=10, anti=False,
                lag_mode='standard', threshold=None):
    ''' plot omplot which is combined with color babble plot and
        histgram and poer histgram
    '''
    # make summary
    df_sum = make_summary(x, freqinfo, anti=anti)
    # periodic sum
    if lag_mode == 'periodic':
        summ_peri = make_periodicsum(lagrange=lagrange)
        self.df_sum = summ_peri.df_sum

    fig = plt.figure(figsize=(8, 2*2+2))
    height = 0.8/2
    ax0 = fig.add_axes([0.10, 0.1+0*height, 0.85, height-0.2])

    # get hist value
    h, bin_edges = np.histogram(df_sum.lag, bins=bins, range=lagrange,
                                density=True)
    ph, bin_edges = phist(x, freqinfo, bins=bins, lagrange=lagrange,
                          density=True)

    # figure
    binwidth = np.diff(bin_edges).mean()
    ax0.bar(bin_edges[:-1], h,  color="r", width=binwidth,
            align="edge", alpha=0.5, edgecolor="k", label="Number")
    ax0.bar(bin_edges[:-1], ph, color="b", width=binwidth,
            align="edge", alpha=0.5, edgecolor="k", label="Power")
    ax0.set_ylabel('Density')
    ax0.set_xlabel('Lag')
    ax0.legend(loc='best')

    # color babble plot
    ax1 = fig.add_axes([0.10, 0.1+1*height-0.2, 0.85, height+0.2],
                       sharex=ax0)
    ax1 = plot_cbplot(x, freqinfo, ax=ax1, lagrange=lagrange,
                      bin_edges=bin_edges, threshold=threshold)
    ax1.set_xlabel(None)
    plt.setp(ax1.get_xticklabels(), visible=False)

    return fig


def main():

    freqinfo = [0, 0.5, 2000]
    X = np.loadtxt('../X.dat')
    testrange = [-10, 10]

    lagmap, bin_edges = phist(X, freqinfo, lagrange=[-10,10], bins=20)
    time = np.arange(lagmap.shape[1])
    nbins = len(bin_edges) - 1
    lagmap[nbins-1 : nbins+1, :] = 0
    plt.pcolormesh(time, bin_edges[:-1], lagmap, shading='gouraud')
    plt.show()

    j = np.random.randint(0, lagmap.shape[-1])
    print(f'index: {j}')
    plot_omplot(X[:,j], freqinfo, lagrange=[-10, 10], bins='auto')
    plt.savefig('omplot_noavesub.png')
    plt.show()


if __name__ =='__main__':
    main()

