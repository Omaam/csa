import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

def set_bins(low, hig, binsize):

    def get_deci_nonzero(a):
        a = float(round(a,10))
        deci_str = str(a).split(".")[1]
        for i, num in enumerate(list(deci_str)):
            if num != '0':
                break
        return i+1

    low = float(round(low, 10))
    hig = float(round(hig, 10))
    # Get digit
    min_abs = abs(low) if abs(low) <= abs(hig) else abs(hig)
    if min_abs > 1:
        nd_max = 0
    else:
        dig_low = get_deci_nonzero(low)
        dig_hig = get_deci_nonzero(hig)
        nd_max = dig_low if dig_low >= dig_hig else dig_hig
    # Float into integer
    low_tmp = np.floor(low * 10 ** nd_max)
    hig_tmp =  np.ceil(hig * 10 ** nd_max)
    binsize_tmp = binsize * 10 ** nd_max
    # Make bins
    bins = np.arange(low_tmp, hig_tmp + binsize_tmp, binsize_tmp)
    # Change digit into original
    bins = bins / 10 **nd_max
    return bins

def hist(df_sum, lagbins, density=True):
    df_sum_pow = df_sum.copy()
    hist_value = pd.cut(df_sum_pow.lag.values, bins=lagbins).value_counts().sort_index().values
    if density:
        hist_value = hist_value / len(df_sum)
    bins_center = (lagbins[:-1] + lagbins[1:]) / 2
    return bins_center, hist_value

def powerhist(df_sum, lagbins, density=True):
    df_sum_pow = df_sum.copy()
    df_sum_pow['id_bins'] = pd.cut(df_sum_pow.lag.values, bins=lagbins)
    powerhist_value = df_sum_pow.groupby('id_bins')['norm12'].sum().values
    if density:
        powerhist_value = powerhist_value / df_sum_pow.norm12.sum()
    bins_center = (lagbins[:-1] + lagbins[1:]) / 2
    return bins_center, powerhist_value



def plot_cbplot(df_sum, freq_info, threshold=None, lagrange=None, ax=None):

    import matplotlib.cm as cm
    if ax == None:
        fig, ax = plt.subplots()
    # color babble plot
    size = list(map(lambda norm1, norm2: norm1/norm2 if norm1 <= norm2 else norm2/norm1,
                    df_sum.norm1.values,
                    df_sum.norm2.values))
    size_log = np.log10(1 + np.array(size)) * 750
    im = ax.scatter(df_sum['lag'].values, df_sum['norm12'].values,
                    c=df_sum['freq'].values, cmap=cm.jet,
                    alpha=0.8, s=size_log,
                    norm=Normalize(freq_info[0], freq_info[1]))
    ax.set_yscale('log')
    ax.set_ylim(np.min(df_sum.norm12))
    ax.set_ylabel('Power')
    ax.set_xlabel('Lag')
    if lagrange:
        ax.set_xlim(lagrange)

    # colorbat setting
    from mpl_toolkits.axes_grid1.colorbar import colorbar
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    colorbar(im, cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_text("Frequency")
    cax.xaxis.set_label_position("top")

    return ax

def plot_omplot(df_sum, freq_info, lagrange=None, lagbinwidth=None, lag_mode='standard', ci_list=None, ci_label=None):
    '''plot omplot which is combined with color babble plot and histgram and poer histgram

    ci_list: (lag, ci_value)
    ci_label: ci name e.g. 'ci 95%'
    '''
    fig = plt.figure(figsize=(8, 2*2+2))
    height = 0.8/2

    # get hist value
    ax0 = fig.add_axes([0.10, 0.1+0*height, 0.85, height-0.2])
    if lagrange == None:
        lag_min = df_sum.lag.min()
        lag_max = df_sum.lag.max()
    else:
        lag_min = lagrange[0]
        lag_max = lagrange[1]
    if lagbinwidth == None:
        lagbinwidth = (lag_max - lag_min) / 20
    lagbins = set_bins(lag_min, lag_max, lagbinwidth)

    # periodic sum
    if lag_mode == 'periodic':
        df_sum = make_periodicsum(lagrange=lagrange)

    # hist and power hist
    bins_center, hist_value = hist(df_sum, lagbins)
    bins_center, phist_value = powerhist(df_sum, lagbins)
    ax0.bar(bins_center, hist_value,  color="r", width=lagbinwidth,
            align="center", alpha=0.5, edgecolor="k", label="Number")
    ax0.bar(bins_center, phist_value, color="b", width=lagbinwidth,
            align="center", alpha=0.5, edgecolor="k", label="Power")
    if ci_list:
        ax0.plot(ci_list[0], ci_list[1], label=ci_label, color='r', alpha=0.7)
    ax0.set_ylabel('Density')
    ax0.set_xlabel(r'$\tau$')
    ax0.legend(loc='best')

    # color babble plot
    ax1 = fig.add_axes([0.10, 0.1+1*height-0.2, 0.85, height+0.2], sharex=ax0)
    ax1 = plot_cbplot(df_sum, freq_info, ax=ax1, lagrange=lagrange)
    ax1.set_xlabel(None)
    plt.setp(ax1.get_xticklabels(), visible=False)

    return fig

if __name__ == '__main__':
    df_sum = pd.read_csv('./period0620/sum.dat', sep=' ', names=['lag', 'norm12', 'norm1', 'norm2', 'period', 'freq'])
    print(df_sum)
    plot_omplot(df_sum, [0, 10, 2000], lagrange=[-5, 5])
    plt.show()
